#!/usr/bin/env python3

import argparse
import copy
import os
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_detector


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import projects.mmdet3d_plugin  # noqa: F401,E402


def force_identity_image_transform(pipeline, width, height):
    """
    Make ResizeCropFlipRotImage leave our already-prepared
    4096x3072 video frames geometrically unchanged.
    """
    found = False

    for step in pipeline:
        if step.get("type") != "ResizeCropFlipRotImage":
            continue

        conf = step["data_aug_conf"]

        conf["H"] = int(height)
        conf["W"] = int(width)
        conf["final_dim"] = (int(height), int(width))

        conf["resize_lim"] = (1.0, 1.0)
        conf["bot_pct_lim"] = (0.0, 0.0)
        conf["rot_lim"] = (0.0, 0.0)
        conf["rand_flip"] = False

        found = True
        break

    if not found:
        raise RuntimeError(
            "ResizeCropFlipRotImage not found in test pipeline"
        )

    return pipeline


def extract_video_frames(video_path, output_dir, limit):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video: {}".format(video_path)
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if width != 4096 or height != 3072:
        raise RuntimeError(
            "Expected 4096x3072 input, got {}x{}".format(
                width, height
            )
        )

    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError("Invalid video FPS: {}".format(fps))

    paths = []

    for i in range(limit):
        ok, frame = cap.read()

        if not ok:
            break

        path = output_dir / "frame_{:06d}.jpg".format(i)

        if not cv2.imwrite(
            str(path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        ):
            raise RuntimeError(
                "Failed writing {}".format(path)
            )

        paths.append(path)

    cap.release()

    if not paths:
        raise RuntimeError("No video frames extracted")

    return paths, fps


def make_static_video_infos(dataset, frame_paths, fps, intrinsic):
    """
    Reuse the nuScenes dataset record structure, but replace its
    camera/image/pose information with our stationary-camera sequence.
    """

    if not dataset.data_infos:
        raise RuntimeError("Dataset contains no template infos")

    base = copy.deepcopy(dataset.data_infos[0])

    if "cams" not in base or "CAM_FRONT" not in base["cams"]:
        raise RuntimeError(
            "Template dataset does not contain CAM_FRONT"
        )

    base_cam = copy.deepcopy(base["cams"]["CAM_FRONT"])

    # Camera optical coordinates:
    #   +x = image right
    #   +y = image down
    #   +z = forward
    #
    # StreamPETR / nuScenes ego coordinates:
    #   +x = forward
    #   +y = left
    #   +z = up
    #
    # camera -> ego/reference
    # CONTROL EXPERIMENT:
    #
    # Define the StreamPETR reference frame to be the fixed global frame.
    # Assume camera axes == reference/global axes.
    #
    # No camera rotation is supplied.
    #
    # Therefore:
    #   camera -> reference rotation = I
    #   camera -> reference translation = 0
    #   lidar/reference -> ego        = I
    #   ego -> global                 = I
    #   ego_pose                      = I
    #
    # Any REAL camera motion is intentionally NOT compensated.
    cam_to_ego = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )

    infos = []

    for i, frame_path in enumerate(frame_paths):
        info = copy.deepcopy(base)
        cam = copy.deepcopy(base_cam)

        timestamp_us = int(
            round((float(i) / fps) * 1_000_000.0)
        )

        cam["data_path"] = str(frame_path)

        # sensor(camera) -> reference/ego
        cam["sensor2lidar_rotation"] = cam_to_ego.copy()
        cam["sensor2lidar_translation"] = np.zeros(
            3,
            dtype=np.float64,
        )

        cam["cam_intrinsic"] = intrinsic.copy()
        cam["timestamp"] = timestamp_us

        # Only one physical camera.
        info["cams"] = {
            "CAM_FRONT": cam,
        }

        # Mock LiDAR/reference sensor sits exactly at ego origin.
        info["lidar2ego_rotation"] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        info["lidar2ego_translation"] = [
            0.0,
            0.0,
            0.0,
        ]

        # Camera/ego is completely stationary.
        #
        # This causes StreamPETR's generated ego_pose to be:
        #
        #     I_4x4
        #
        # for every frame.
        info["ego2global_rotation"] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        info["ego2global_translation"] = [
            0.0,
            0.0,
            0.0,
        ]

        info["timestamp"] = timestamp_us

        # Make the entire video one temporal scene.
        info["scene_token"] = "static-video-scene"
        info["frame_idx"] = i

        info["token"] = "static-video-{:06d}".format(i)

        info["prev"] = (
            ""
            if i == 0
            else "static-video-{:06d}".format(i - 1)
        )

        info["next"] = (
            ""
            if i == len(frame_paths) - 1
            else "static-video-{:06d}".format(i + 1)
        )

        info["sweeps"] = []

        infos.append(info)

    return infos


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        default=(
            "/home/levif/src/StreamPETR/"
            "20260817_133718_300f_4096x3072.mp4"
        ),
    )

    parser.add_argument(
        "--config",
        default=(
            "projects/configs/StreamPETR/"
            "stream_petr_r50_704_bs2_seq_428q_nui_60e.py"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "ckpts/"
            "stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth"
        ),
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--frame-dir",
        default="/tmp/streampetr_static_video_frames",
    )

    parser.add_argument(
        "--output",
        default="/tmp/streampetr_static_video_outputs.pkl",
    )

    parser.add_argument("--gpu", default="0")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = Path(args.video).resolve()
    frame_dir = Path(args.frame_dir)
    output_path = Path(args.output)

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()

    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    print("Extracting video frames...")

    frame_paths, fps = extract_video_frames(
        video_path,
        frame_dir,
        args.num_frames,
    )

    # Original calibration:
    #
    # 2048 x 1536
    #
    # [1376.712891,    0,       1015.622070]
    # [   0,        1379.823380, 756.072259]
    # [   0,           0,         1]
    #
    # Current image is exactly 2x in both dimensions.
    scale_x = 4096.0 / 2048.0
    scale_y = 3072.0 / 1536.0

    K = np.array(
        [
            [
                1376.712891 * scale_x,
                0.0,
                1015.622070 * scale_x,
            ],
            [
                0.0,
                1379.823380 * scale_y,
                756.072259 * scale_y,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    print()
    print("========================================")
    print("STATIC VIDEO STREAM-PETR")
    print("========================================")
    print("Video :", video_path)
    print("Frames:", len(frame_paths))
    print("FPS   :", fps)
    print("Input : 4096 x 3072")
    print()
    print("Scaled camera intrinsic:")
    print(K)
    print()
    print("Mock ego_pose:")
    print(np.eye(4))
    print()

    cfg = Config.fromfile(str(config_path))

    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    cfg.data.test.test_mode = True
    cfg.data.samples_per_gpu = 1

    # Important: preserve deterministic sequence order.
    cfg.data.workers_per_gpu = 0

    cfg.data.test.pipeline = (
        force_identity_image_transform(
            cfg.data.test.pipeline,
            width=4096,
            height=3072,
        )
    )

    dataset = build_dataset(cfg.data.test)

    dataset.data_infos = make_static_video_infos(
        dataset,
        frame_paths,
        fps,
        K,
    )

    # Some mmdet datasets keep a group flag with one entry
    # per data_info.
    if hasattr(dataset, "flag"):
        dataset.flag = np.zeros(
            len(dataset.data_infos),
            dtype=np.uint8,
        )

    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    model = build_detector(
        cfg.model,
        test_cfg=cfg.get("test_cfg"),
    )

    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)

    load_checkpoint(
        model,
        str(checkpoint_path),
        map_location="cpu",
    )

    model = MMDataParallel(
        model.cuda(),
        device_ids=[0],
    )

    model.eval()

    # Reset only once: temporal memory should persist throughout
    # the complete 300-frame sequence.
    try:
        model.module.pts_bbox_head.reset_memory()
        print("Temporal memory reset once.")
    except Exception as exc:
        print(
            "Could not explicitly reset memory:",
            repr(exc),
        )

    outputs = []

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print()
    print("Running inference...")

    with torch.no_grad():
        for i, data in enumerate(loader):
            result = model(
                return_loss=False,
                rescale=True,
                **data
            )

            outputs.extend(result)

            if (
                (i + 1) % 10 == 0
                or (i + 1) == len(frame_paths)
            ):
                print(
                    "Inference [{}/{}]".format(
                        i + 1,
                        len(frame_paths),
                    )
                )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("wb") as f:
        pickle.dump(outputs, f)

    print()
    print("========================================")
    print("DONE")
    print("========================================")
    print("Frames processed:", len(outputs))
    print("Predictions     :", output_path)


if __name__ == "__main__":
    main()
