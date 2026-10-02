#!/usr/bin/env python3

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Needed when unpickling mmdet3d box objects.
import projects.mmdet3d_plugin  # noqa: F401


CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


# ============================================================
# CAMERA GEOMETRY
# ============================================================
#
# This is intentionally the SAME geometry assumption we used
# for the stationary-video StreamPETR run.
#
# Original presumed calibration:
#
# K_2048x1536 =
# [1376.712891,    0,       1015.622070]
# [   0,        1379.82338, 756.072259]
# [   0,           0,          1]
#
# Input image was 4096 x 3072 => exactly x2.
# ============================================================

K = np.array(
    [
        [2753.425782, 0.0,         2031.244140],
        [0.0,         2759.646760, 1512.144518],
        [0.0,         0.0,            1.0],
    ],
    dtype=np.float64,
)


# Camera coordinates:
#
#   +X = image right
#   +Y = image down
#   +Z = optical forward
#
# StreamPETR / nuScenes ego:
#
#   +X = forward
#   +Y = left
#   +Z = up
#
# This was our assumed camera -> ego rotation:
#
# camera x(right)   -> ego -Y
# camera y(down)    -> ego -Z
# camera z(forward) -> ego +X

# CONTROL EXPERIMENT:
# camera frame == fixed reference/global frame.
#
# Must match the geometry supplied during inference.
CAM_TO_EGO = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )

# Inverse rotation because it is orthonormal.
EGO_TO_CAM = CAM_TO_EGO.T

# We assumed zero camera translation during the inference.
CAMERA_TRANSLATION_EGO = np.zeros(3, dtype=np.float64)


BOX_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),

    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),

    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def ego_to_camera(points_ego):
    """
    points_ego: (..., 3)

    Convert StreamPETR ego/reference-frame points to camera coords.
    """

    points_ego = np.asarray(points_ego, dtype=np.float64)

    centered = points_ego - CAMERA_TRANSLATION_EGO

    points_cam = centered @ EGO_TO_CAM.T

    return points_cam


def project_points(points_cam):
    """
    Camera coordinates -> image pixels.

    Returns:
        uv: (N,2)
        valid_depth: z > 0
    """

    points_cam = np.asarray(points_cam, dtype=np.float64)

    z = points_cam[:, 2]

    valid = z > 1e-4

    uv = np.full(
        (len(points_cam), 2),
        np.nan,
        dtype=np.float64,
    )

    if np.any(valid):
        p = points_cam[valid]

        projected = (K @ p.T).T

        uv[valid] = projected[:, :2] / projected[:, 2:3]

    return uv, valid


def in_image(pt, width, height):
    x, y = pt

    return (
        np.isfinite(x)
        and np.isfinite(y)
        and 0 <= x < width
        and 0 <= y < height
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        default="/tmp/streampetr_static_video_outputs.pkl",
    )

    parser.add_argument(
        "--video",
        default=(
            "/home/levif/src/StreamPETR/"
            "20260817_133718_300f_4096x3072.mp4"
        ),
    )

    parser.add_argument(
        "--output",
        default="/tmp/streampetr_static_projected_3d.mp4",
    )

    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all saved prediction frames",
    )

    args = parser.parse_args()

    pred_path = Path(args.predictions)
    video_path = Path(args.video)
    output_path = Path(args.output)

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    print("Loading predictions:")
    print(pred_path)

    with pred_path.open("rb") as f:
        outputs = pickle.load(f)

    if args.limit > 0:
        outputs = outputs[:args.limit]

    print("Prediction frames:", len(outputs))

    # --------------------------------------------------------
    # Open source video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video {}".format(video_path)
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps <= 0:
        source_fps = 30.0

    print("Video resolution: {}x{}".format(width, height))
    print("Video FPS       :", source_fps)

    # --------------------------------------------------------
    # Use ffmpeg pipe rather than OpenCV writer.
    # More reliable H.264 output.
    # --------------------------------------------------------

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "{}x{}".format(width, height),
            "-r",
            str(source_fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    for frame_idx, output in enumerate(outputs):

        ok, frame = cap.read()

        if not ok:
            print(
                "Video ended at frame {}".format(frame_idx)
            )
            break

        pred = output.get("pts_bbox", output)

        boxes_obj = pred["boxes_3d"]
        scores = to_numpy(pred["scores_3d"])
        labels = to_numpy(pred["labels_3d"]).astype(int)

        # mmdet3d directly provides all eight physical corners.
        corners = to_numpy(boxes_obj.corners)

        num_score_pass = 0
        num_in_front = 0
        num_visible = 0

        for box_idx in range(len(scores)):

            score = float(scores[box_idx])

            if score < args.score_thr:
                continue

            num_score_pass += 1

            label = int(labels[box_idx])

            corners_ego = corners[box_idx]

            # ----------------------------------------------
            # Ego/reference coordinates -> camera coordinates
            # ----------------------------------------------

            corners_cam = ego_to_camera(corners_ego)

            uv, depth_valid = project_points(corners_cam)

            if np.count_nonzero(depth_valid) == 0:
                # Entire predicted cuboid is behind camera.
                continue

            num_in_front += 1

            # Determine whether at least one projected point
            # is actually on the image.
            visible_points = [
                in_image(uv[j], width, height)
                for j in range(8)
                if depth_valid[j]
            ]

            if not any(visible_points):
                continue

            num_visible += 1

            # ----------------------------------------------
            # Draw cuboid edges
            # ----------------------------------------------

            for a, b in BOX_EDGES:

                # Don't draw an edge if either endpoint is
                # behind the camera.
                if not depth_valid[a] or not depth_valid[b]:
                    continue

                pa = uv[a]
                pb = uv[b]

                if not (
                    np.all(np.isfinite(pa))
                    and np.all(np.isfinite(pb))
                ):
                    continue

                p1 = (
                    int(round(pa[0])),
                    int(round(pa[1])),
                )

                p2 = (
                    int(round(pb[0])),
                    int(round(pb[1])),
                )

                cv2.line(
                    frame,
                    p1,
                    p2,
                    (0, 255, 0),
                    4,
                    cv2.LINE_AA,
                )

            # ----------------------------------------------
            # Label around projected 3D center
            # ----------------------------------------------

            center_ego = corners_ego.mean(axis=0)

            center_cam = ego_to_camera(
                center_ego.reshape(1, 3)
            )

            center_uv, center_valid = project_points(
                center_cam
            )

            if center_valid[0]:

                u = int(round(center_uv[0, 0]))
                v = int(round(center_uv[0, 1]))

                if 0 <= label < len(CLASS_NAMES):
                    name = CLASS_NAMES[label]
                else:
                    name = str(label)

                text = "{} {:.2f}".format(
                    name,
                    score,
                )

                cv2.putText(
                    frame,
                    text,
                    (u, v),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # --------------------------------------------------
        # Diagnostic overlay
        # --------------------------------------------------

        diagnostic = (
            "frame={} score_pass={} "
            "in_front={} visible={}"
        ).format(
            frame_idx,
            num_score_pass,
            num_in_front,
            num_visible,
        )

        cv2.rectangle(
            frame,
            (20, 20),
            (1150, 100),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            diagnostic,
            (40, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )

        ffmpeg.stdin.write(frame.tobytes())

        if (
            (frame_idx + 1) % 10 == 0
            or frame_idx == 0
        ):
            print(
                "[{:03d}] score-pass={} "
                "in-front={} visible={}".format(
                    frame_idx,
                    num_score_pass,
                    num_in_front,
                    num_visible,
                )
            )

    cap.release()

    ffmpeg.stdin.close()
    rc = ffmpeg.wait()

    if rc != 0:
        raise RuntimeError(
            "ffmpeg exited with status {}".format(rc)
        )

    print()
    print("========================================")
    print("PROJECTION VIDEO COMPLETE")
    print("========================================")
    print("Output:", output_path)


if __name__ == "__main__":
    main()
