#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def slugify(name):
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    return name.strip("_") or "run"


def run_and_tee(cmd, log_path, env=None):
    print("\nRunning:")
    print(" ".join(str(x) for x in cmd))
    print()

    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(
            "Command failed with exit code {}. See {}".format(rc, log_path)
        )


def find_new_results_json(start_time):
    candidates = []

    for p in ROOT.glob("test/**/results_nusc.json"):
        try:
            if p.stat().st_mtime >= start_time - 2:
                candidates.append(p)
        except FileNotFoundError:
            pass

    if not candidates:
        raise RuntimeError(
            "Inference finished, but no new results_nusc.json was found."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def evaluate_nuscenes(result_path, output_dir, data_root, version, eval_set):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print("Evaluating against ground truth")
    print("==============================")

    nusc = NuScenes(
        version=version,
        dataroot=str(data_root),
        verbose=False,
    )

    detection_cfg = config_factory("detection_cvpr_2019")

    evaluator = NuScenesEval(
        nusc=nusc,
        config=detection_cfg,
        result_path=str(result_path),
        eval_set=eval_set,
        output_dir=str(output_dir),
        verbose=False,
    )

    evaluator.main(render_curves=False)

    metrics_path = output_dir / "metrics_summary.json"

    with open(metrics_path) as f:
        metrics = json.load(f)

    return metrics


def benchmark_model(config_path, checkpoint_path, max_samples, warmup):
    import torch

    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model

    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_detector

    # Register StreamPETR modules.
    import projects.mmdet3d_plugin  # noqa: F401

    print("\n==============================")
    print("Benchmarking model")
    print("==============================")

    cfg = Config.fromfile(str(config_path))

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_detector(
        cfg.model,
        test_cfg=cfg.get("test_cfg"),
    )

    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
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

    detector = model.module

    # ---------------------------------------------------------
    # Instrument the actual tensors being processed.
    # ---------------------------------------------------------
    spatial = {
        # A sparse-patch frame can call the backbone many times. These fields
        # are reset before every model forward and accumulated by the hook.
        "backbone_input_shape": None,
        "backbone_input_shapes": [],
        "backbone_call_count": 0,
        "input_views": 0,
        "input_height": None,
        "input_width": None,
        "input_pixels_per_sample": 0,

        # Exact image memory passed to PETR cross-attention.
        "cross_attention_image_tokens": None,

        # Exact current query count passed into the transformer.
        "cross_attention_query_tokens": None,

        # Sparse historical memory used by temporal attention.
        "temporal_memory_tokens": None,

        "cross_attention_score_pairs_per_layer": None,
        "temporal_attention_score_pairs_per_layer": None,
        "decoder_layers": None,
    }

    def reset_frame_spatial():
        spatial["backbone_input_shape"] = None
        spatial["backbone_input_shapes"] = []
        spatial["backbone_call_count"] = 0
        spatial["input_views"] = 0
        spatial["input_height"] = None
        spatial["input_width"] = None
        spatial["input_pixels_per_sample"] = 0
        spatial["cross_attention_image_tokens"] = None
        spatial["cross_attention_query_tokens"] = None
        spatial["temporal_memory_tokens"] = None
        spatial["cross_attention_score_pairs_per_layer"] = None
        spatial["temporal_attention_score_pairs_per_layer"] = None
        spatial["decoder_layers"] = None

    def backbone_pre_hook(module, inputs):
        if not inputs:
            return

        x = inputs[0]

        if not torch.is_tensor(x):
            return

        shape = tuple(int(v) for v in x.shape)
        spatial["backbone_input_shape"] = list(shape)
        spatial["backbone_input_shapes"].append(list(shape))
        spatial["backbone_call_count"] += 1

        # Full-frame mode has one call with N cameras. Sparse-patch mode has
        # one or more calls, normally with N=1. Sum every call for true
        # backbone pixel work per frame.
        if x.dim() == 4:
            n, c, h, w = shape
            spatial["input_views"] += n
            spatial["input_height"] = h
            spatial["input_width"] = w
            spatial["input_pixels_per_sample"] += n * h * w

    def transformer_pre_hook(module, inputs):
        # PETRTemporalTransformer.forward:
        #
        #   memory, tgt, query_pos, pos_embed,
        #   attn_masks, temp_memory, temp_pos
        #
        # memory is EXACTLY the image token tensor used as K/V
        # in cross-attention.

        if len(inputs) < 2:
            return

        memory = inputs[0]
        tgt = inputs[1]

        if torch.is_tensor(memory) and memory.dim() == 3:
            spatial["cross_attention_image_tokens"] = int(memory.shape[1])

        if torch.is_tensor(tgt) and tgt.dim() == 3:
            spatial["cross_attention_query_tokens"] = int(tgt.shape[1])

        if len(inputs) > 5:
            temp_memory = inputs[5]

            if torch.is_tensor(temp_memory) and temp_memory.dim() == 3:
                spatial["temporal_memory_tokens"] = int(
                    temp_memory.shape[1]
                )

        try:
            spatial["decoder_layers"] = len(module.decoder.layers)
        except Exception:
            try:
                spatial["decoder_layers"] = int(
                    cfg.model.pts_bbox_head.transformer.decoder.num_layers
                )
            except Exception:
                spatial["decoder_layers"] = None

    backbone_handle = detector.img_backbone.register_forward_pre_hook(
        backbone_pre_hook
    )

    transformer_handle = (
        detector.pts_bbox_head.transformer.register_forward_pre_hook(
            transformer_pre_hook
        )
    )

    # Reset temporal state before benchmarking.
    try:
        detector.pts_bbox_head.reset_memory()
    except Exception:
        pass

    total_available = len(dataset)

    if max_samples <= 0:
        sample_limit = total_available
    else:
        sample_limit = min(max_samples, total_available)

    if sample_limit <= warmup:
        warmup = max(0, sample_limit // 5)

    print("Dataset samples: {}".format(total_available))
    print("Benchmark samples: {}".format(sample_limit))
    print("Warmup samples: {}".format(warmup))

    iterator = iter(data_loader)

    model_times = []
    end_to_end_times = []

    torch.cuda.empty_cache()
    spatial_samples = []

    for i in range(sample_limit):

        # Includes waiting for the next dataloader item.
        full_start = time.perf_counter()

        try:
            data = next(iterator)
        except StopIteration:
            break

        torch.cuda.synchronize()
        reset_frame_spatial()

        if i == warmup:
            # Reset BEFORE the first measured forward, retaining its peak.
            torch.cuda.reset_peak_memory_stats()
        model_start = time.perf_counter()

        with torch.no_grad():
            model(
                return_loss=False,
                rescale=True,
                **data
            )

        torch.cuda.synchronize()

        model_elapsed = time.perf_counter() - model_start
        full_elapsed = time.perf_counter() - full_start

        record = dict(spatial)
        record.update(sample_index=i, measured=i >= warmup,
                      model_latency_ms=1000*model_elapsed)
        meta = data.get('img_metas')
        while isinstance(meta, (list, tuple)) or hasattr(meta, 'data'):
            if isinstance(meta, (list, tuple)):
                meta = meta[0]
            else:
                meta = meta.data
        if isinstance(meta, dict):
            record['sample_token'] = meta.get('sample_idx')
            record['oracle_tile_stats'] = meta.get('oracle_tile_stats')
            record['sparse_camera_stats'] = meta.get('sparse_camera_stats')
            record['sparse_runtime_stats'] = meta.get('sparse_runtime_stats')
        q = record['cross_attention_query_tokens']
        k = record['cross_attention_image_tokens']
        t = record['temporal_memory_tokens']
        layers = record['decoder_layers']
        record['cross_attention_score_pairs_per_layer'] = q*k
        record['cross_attention_score_pairs_all_decoder_layers'] = q*k*layers
        if t is not None:
            record['temporal_attention_score_pairs_per_layer'] = q*(q+t)
            record['temporal_attention_score_pairs_all_decoder_layers'] = q*(q+t)*layers
        spatial_samples.append(record)

        if i >= warmup:
            model_times.append(model_elapsed)
            end_to_end_times.append(full_elapsed)

        if (i + 1) % 20 == 0 or (i + 1) == sample_limit:
            print(
                "Benchmark [{}/{}]".format(
                    i + 1,
                    sample_limit
                )
            )

    backbone_handle.remove()
    transformer_handle.remove()

    if not model_times:
        raise RuntimeError("No benchmark samples were measured.")

    measured = len(model_times)

    total_model_time = sum(model_times)
    total_e2e_time = sum(end_to_end_times)

    model_fps = measured / total_model_time
    end_to_end_fps = measured / total_e2e_time

    mean_latency_ms = 1000.0 * np.mean(model_times)
    median_latency_ms = 1000.0 * np.median(model_times)
    p95_latency_ms = 1000.0 * np.percentile(model_times, 95)

    peak_allocated_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )

    peak_reserved_gb = (
        torch.cuda.max_memory_reserved() / (1024 ** 3)
    )

    image_tokens = spatial["cross_attention_image_tokens"]
    query_tokens = spatial["cross_attention_query_tokens"]
    temporal_tokens = spatial["temporal_memory_tokens"]
    decoder_layers = spatial["decoder_layers"]

    if image_tokens is not None and query_tokens is not None:
        spatial["cross_attention_score_pairs_per_layer"] = (
            image_tokens * query_tokens
        )

    if (
        query_tokens is not None
        and temporal_tokens is not None
    ):
        # StreamPETR self-attention uses current queries as Q and
        # [current queries + temporal memory] as K/V.
        spatial["temporal_attention_score_pairs_per_layer"] = (
            query_tokens
            * (query_tokens + temporal_tokens)
        )

    if (
        spatial["cross_attention_score_pairs_per_layer"] is not None
        and decoder_layers is not None
    ):
        spatial["cross_attention_score_pairs_all_decoder_layers"] = (
            spatial["cross_attention_score_pairs_per_layer"]
            * decoder_layers
        )
    else:
        spatial["cross_attention_score_pairs_all_decoder_layers"] = None

    if (
        spatial["temporal_attention_score_pairs_per_layer"] is not None
        and decoder_layers is not None
    ):
        spatial["temporal_attention_score_pairs_all_decoder_layers"] = (
            spatial["temporal_attention_score_pairs_per_layer"]
            * decoder_layers
        )
    else:
        spatial["temporal_attention_score_pairs_all_decoder_layers"] = None

    # Variable tile counts require aggregates, not the last frame's shape.
    measured_spatial = [r for r in spatial_samples if r['measured']]
    spatial['last_backbone_input_shape'] = spatial.pop('backbone_input_shape')
    aggregate_keys = (
        'backbone_call_count', 'input_views', 'input_height', 'input_width',
        'input_pixels_per_sample',
        'cross_attention_image_tokens', 'cross_attention_query_tokens',
        'temporal_memory_tokens', 'cross_attention_score_pairs_per_layer',
        'cross_attention_score_pairs_all_decoder_layers',
        'temporal_attention_score_pairs_per_layer',
        'temporal_attention_score_pairs_all_decoder_layers')
    spatial['per_sample'] = spatial_samples
    spatial['aggregate_scope'] = 'measured samples excluding warmup'
    spatial['statistics'] = {}
    for key in aggregate_keys:
        values = [r[key] for r in measured_spatial if r.get(key) is not None]
        if values:
            spatial[key] = float(np.mean(values))
            spatial['statistics'][key] = dict(mean=float(np.mean(values)),
                min=float(np.min(values)), max=float(np.max(values)),
                p95=float(np.percentile(values, 95)))

    performance = {
        "samples_measured": measured,
        "warmup_samples": warmup,

        "model_fps": float(model_fps),
        "end_to_end_fps": float(end_to_end_fps),

        "mean_model_latency_ms": float(mean_latency_ms),
        "median_model_latency_ms": float(median_latency_ms),
        "p95_model_latency_ms": float(p95_latency_ms),

        "peak_gpu_allocated_gb": float(peak_allocated_gb),
        "peak_gpu_reserved_gb": float(peak_reserved_gb),

        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }

    return performance, spatial


def git_metadata():
    def cmd(args):
        try:
            return subprocess.check_output(
                args,
                cwd=str(ROOT),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    return {
        "commit": cmd(["git", "rev-parse", "HEAD"]),
        "branch": cmd(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ),
        "status": cmd(["git", "status", "--short"]),
    }


def append_summary_csv(csv_path, row):
    fields = [
        "timestamp",
        "run_name",

        "mAP",
        "NDS",
        "mATE",
        "mASE",
        "mAOE",
        "mAVE",
        "mAAE",

        "model_fps",
        "end_to_end_fps",
        "mean_latency_ms",
        "p95_latency_ms",
        "peak_gpu_allocated_gb",

        "input_pixels_per_sample",
        "cross_attention_image_tokens",
        "cross_attention_query_tokens",
        "cross_attention_pairs_per_layer",
        "cross_attention_pairs_all_layers",

        "temporal_memory_tokens",
        "temporal_attention_pairs_per_layer",

        "run_directory",
    ]

    exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

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
        "--data-root",
        default="./data/nuscenes",
    )

    parser.add_argument(
        "--dataset-version",
        default="v1.0-mini",
    )

    parser.add_argument(
        "--eval-set",
        default="mini_val",
    )

    parser.add_argument(
        "--benchmark-samples",
        type=int,
        default=0,
        help="0 = benchmark entire validation set",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--gpu",
        default="0",
    )

    parser.add_argument(
        "--name",
        default=None,
        help="Optional. If omitted, you will be prompted.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Prompt for experiment name.
    # ---------------------------------------------------------
    run_name = args.name

    while not run_name:
        run_name = input(
            "\nName this StreamPETR run: "
        ).strip()

    slug = slugify(run_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    runs_root = ROOT / "experiment_runs"
    runs_root.mkdir(exist_ok=True)

    run_dir = runs_root / "{}_{}".format(
        timestamp,
        slug,
    )

    run_dir.mkdir(parents=True)

    config_path = (ROOT / args.config).resolve()
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    data_root = (ROOT / args.data_root).resolve()

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    # Keep exact config used for this experiment.
    shutil.copy2(
        config_path,
        run_dir / "{}_config.py".format(slug),
    )

    from mmcv import Config
    Config.fromfile(str(config_path)).dump(str(run_dir / 'resolved_config.py'))

    # Save git state/diff as well.
    git_info = git_metadata()

    with open(run_dir / "{}_git.json".format(slug), "w") as f:
        json.dump(git_info, f, indent=2)

    with open(run_dir / "{}_git_diff.patch".format(slug), "w") as f:
        subprocess.run(
            ["git", "diff"],
            cwd=str(ROOT),
            stdout=f,
            text=True,
        )

    # ---------------------------------------------------------
    # 1. Run StreamPETR inference.
    # ---------------------------------------------------------
    inference_log = run_dir / "{}_inference.log".format(slug)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = "{}:{}".format(
        ROOT,
        env.get("PYTHONPATH", ""),
    )

    inference_cmd = [
        "bash",
        "tools/dist_test.sh",
        str(config_path),
        str(checkpoint_path),
        "1",
        "--format-only",
    ]

    inference_start = time.time()

    run_and_tee(
        inference_cmd,
        inference_log,
        env=env,
    )

    generated_results = find_new_results_json(
        inference_start
    )

    saved_results = (
        run_dir
        / "{}_results_nusc.json".format(slug)
    )

    shutil.copy2(
        generated_results,
        saved_results,
    )

    print("\nSaved predictions:")
    print(saved_results)

    # ---------------------------------------------------------
    # 2. Evaluate predictions against nuScenes ground truth.
    # ---------------------------------------------------------
    eval_dir = run_dir / "evaluation"

    metrics = evaluate_nuscenes(
        saved_results,
        eval_dir,
        data_root,
        args.dataset_version,
        args.eval_set,
    )

    # ---------------------------------------------------------
    # 3. Benchmark runtime + spatial/temporal compute proxies.
    # ---------------------------------------------------------
    performance, spatial = benchmark_model(
        config_path,
        checkpoint_path,
        args.benchmark_samples,
        args.warmup,
    )

    # ---------------------------------------------------------
    # 4. Assemble complete experiment record.
    # ---------------------------------------------------------
    record = {
        "run_name": run_name,
        "run_slug": slug,
        "timestamp": timestamp,

        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "dataset_version": args.dataset_version,
        "eval_set": args.eval_set,

        "git": git_info,

        "detection_metrics": {
            "mAP": metrics["mean_ap"],
            "NDS": metrics["nd_score"],

            "mATE": metrics["tp_errors"].get(
                "trans_err"
            ),
            "mASE": metrics["tp_errors"].get(
                "scale_err"
            ),
            "mAOE": metrics["tp_errors"].get(
                "orient_err"
            ),
            "mAVE": metrics["tp_errors"].get(
                "vel_err"
            ),
            "mAAE": metrics["tp_errors"].get(
                "attr_err"
            ),

            "per_class_AP": metrics.get(
                "label_aps", {}
            ),
        },

        "performance": performance,

        "spatial_temporal_compute": spatial,

        "files": {
            "predictions": str(saved_results),
            "evaluation_directory": str(eval_dir),
            "inference_log": str(inference_log),
        },
    }

    metrics_file = (
        run_dir
        / "{}_metrics.json".format(slug)
    )

    with open(metrics_file, "w") as f:
        json.dump(
            record,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # 5. Append a compact row to global experiment CSV.
    # ---------------------------------------------------------
    row = {
        "timestamp": timestamp,
        "run_name": run_name,

        "mAP": metrics["mean_ap"],
        "NDS": metrics["nd_score"],

        "mATE": metrics["tp_errors"].get("trans_err"),
        "mASE": metrics["tp_errors"].get("scale_err"),
        "mAOE": metrics["tp_errors"].get("orient_err"),
        "mAVE": metrics["tp_errors"].get("vel_err"),
        "mAAE": metrics["tp_errors"].get("attr_err"),

        "model_fps": performance["model_fps"],
        "end_to_end_fps": performance["end_to_end_fps"],
        "mean_latency_ms": performance[
            "mean_model_latency_ms"
        ],
        "p95_latency_ms": performance[
            "p95_model_latency_ms"
        ],
        "peak_gpu_allocated_gb": performance[
            "peak_gpu_allocated_gb"
        ],

        "input_pixels_per_sample": spatial[
            "input_pixels_per_sample"
        ],

        "cross_attention_image_tokens": spatial[
            "cross_attention_image_tokens"
        ],

        "cross_attention_query_tokens": spatial[
            "cross_attention_query_tokens"
        ],

        "cross_attention_pairs_per_layer": spatial[
            "cross_attention_score_pairs_per_layer"
        ],

        "cross_attention_pairs_all_layers": spatial[
            "cross_attention_score_pairs_all_decoder_layers"
        ],

        "temporal_memory_tokens": spatial[
            "temporal_memory_tokens"
        ],

        "temporal_attention_pairs_per_layer": spatial[
            "temporal_attention_score_pairs_per_layer"
        ],

        "run_directory": str(run_dir),
    }

    summary_csv = runs_root / "summary.csv"

    append_summary_csv(
        summary_csv,
        row,
    )

    # ---------------------------------------------------------
    # Final console summary.
    # ---------------------------------------------------------
    d = record["detection_metrics"]

    print()
    print("========================================")
    print("EXPERIMENT COMPLETE: {}".format(run_name))
    print("========================================")

    print()
    print("Detection / 3D accuracy")
    print("-----------------------")
    print("mAP : {:.4f}".format(d["mAP"]))
    print("NDS : {:.4f}".format(d["NDS"]))
    print("mATE: {:.4f} m".format(d["mATE"]))
    print("mASE: {:.4f}".format(d["mASE"]))
    print("mAOE: {:.4f} rad".format(d["mAOE"]))
    print("mAVE: {:.4f} m/s".format(d["mAVE"]))

    print()
    print("Performance")
    print("-----------")
    print(
        "Model FPS: {:.2f}".format(
            performance["model_fps"]
        )
    )
    print(
        "End-to-end FPS: {:.2f}".format(
            performance["end_to_end_fps"]
        )
    )
    print(
        "Mean latency: {:.2f} ms".format(
            performance["mean_model_latency_ms"]
        )
    )
    print(
        "P95 latency: {:.2f} ms".format(
            performance["p95_model_latency_ms"]
        )
    )
    print(
        "Peak GPU allocation: {:.3f} GB".format(
            performance["peak_gpu_allocated_gb"]
        )
    )

    print()
    print("Spatial image-side compute")
    print("--------------------------")
    print(
        "Input pixels/sample:",
        spatial["input_pixels_per_sample"],
    )
    print(
        "Image tokens entering cross-attention:",
        spatial["cross_attention_image_tokens"],
    )
    print(
        "Queries:",
        spatial["cross_attention_query_tokens"],
    )
    print(
        "Q x K pairs / decoder layer:",
        spatial["cross_attention_score_pairs_per_layer"],
    )
    print(
        "Q x K pairs / all decoder layers:",
        spatial[
            "cross_attention_score_pairs_all_decoder_layers"
        ],
    )

    print()
    print("Temporal side")
    print("-------------")
    print(
        "Historical memory tokens:",
        spatial["temporal_memory_tokens"],
    )
    print(
        "Temporal attention pairs / layer:",
        spatial[
            "temporal_attention_score_pairs_per_layer"
        ],
    )

    print()
    print("Saved run:")
    print(run_dir)

    print()
    print("Global comparison table:")
    print(summary_csv)

    print()
    print("Full metrics:")
    print(metrics_file)


if __name__ == "__main__":
    main()
