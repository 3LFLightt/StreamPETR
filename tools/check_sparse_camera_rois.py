#!/usr/bin/env python3
"""CPU preflight for the native sparse-camera experiment.

Compares every sparse-pipeline validation sample against the normal baseline
pipeline, ensuring the six images and all calibration/pose tensors are exactly
unchanged.  It also validates ROI-token coverage and patch bookkeeping.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def unwrap(value):
    from mmcv.parallel import DataContainer
    while isinstance(value, (list, tuple, DataContainer)):
        value = value.data if isinstance(value, DataContainer) else value[0]
    return value


def as_numpy(value):
    value = unwrap(value)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main():
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='projects/configs/StreamPETR/stream_petr_sparse_native_1p3_h64.py')
    parser.add_argument(
        '--baseline-config',
        default='projects/configs/StreamPETR/stream_petr_r50_704_bs2_seq_428q_nui_60e.py')
    args = parser.parse_args()

    sparse_cfg = Config.fromfile(args.config)
    base_cfg = Config.fromfile(args.baseline_config)
    sparse_cfg.data.test.test_mode = True
    base_cfg.data.test.test_mode = True
    sparse_ds = build_dataset(sparse_cfg.data.test)
    base_ds = build_dataset(base_cfg.data.test)
    assert len(sparse_ds) == len(base_ds)

    rows = []
    compare_keys = ('intrinsics', 'extrinsics', 'lidar2img', 'timestamp',
                    'img_timestamp', 'ego_pose', 'ego_pose_inv')

    for i in range(len(sparse_ds)):
        sparse = sparse_ds[i]
        base = base_ds[i]
        s_img = as_numpy(sparse['img'])
        b_img = as_numpy(base['img'])
        np.testing.assert_array_equal(s_img, b_img)
        assert s_img.shape[0] == 6 and s_img.shape[1] == 3

        for key in compare_keys:
            np.testing.assert_allclose(
                as_numpy(sparse[key]), as_numpy(base[key]), rtol=0, atol=0)

        meta = unwrap(sparse['img_metas'])
        spec = meta['sparse_camera_spec']
        stats = meta['sparse_camera_stats']
        assert spec['num_cameras'] == 6
        assert spec['image_height'] == s_img.shape[-2]
        assert spec['image_width'] == s_img.shape[-1]
        fh, fw = spec['feature_height'], spec['feature_width']

        selected = np.zeros((6, fh, fw), dtype=np.bool_)
        roi_ids = set()
        for roi in spec['rois']:
            rid = roi['roi_id']
            assert rid not in roi_ids
            roi_ids.add(rid)
            cam = roi['camera_id']
            x0, y0, x1, y1 = roi['token_rect']
            assert 0 <= cam < 6
            assert 0 <= x0 < x1 <= fw and 0 <= y0 < y1 <= fh
            selected[cam, y0:y1, x0:x1] = True
        assert int(selected.sum()) == stats['selected_token_count']
        assert selected.any()

        covered_roi_ids = []
        processed_pixels = 0
        for patch in spec['patches']:
            cam = patch['camera_id']
            x0, y0, x1, y1 = patch['crop']
            a = spec['patch_alignment']
            assert 0 <= cam < 6
            assert 0 <= x0 < x1 <= spec['image_width']
            assert 0 <= y0 < y1 <= spec['image_height']
            assert x0 % a == y0 % a == x1 % a == y1 % a == 0
            processed_pixels += (x1-x0)*(y1-y0)
            for rid in patch['roi_ids']:
                roi = spec['rois'][rid]
                assert roi['camera_id'] == cam
                tx0, ty0, tx1, ty1 = roi['token_rect']
                px = (tx0*spec['feature_stride'], ty0*spec['feature_stride'],
                      tx1*spec['feature_stride'], ty1*spec['feature_stride'])
                assert x0 <= px[0] and y0 <= px[1]
                assert x1 >= px[2] and y1 >= px[3]
                covered_roi_ids.append(rid)
        assert sorted(covered_roi_ids) == list(range(len(spec['rois'])))
        assert processed_pixels == stats['processed_patch_pixels']
        assert processed_pixels <= stats['raw_patch_pixels']

        rows.append(dict(
            sample_index=i,
            sample_token=meta.get('sample_idx'),
            roi_count=stats['roi_count'],
            selected_tokens=stats['selected_token_count'],
            patch_count=stats['patch_count'],
            processed_patch_pixels=stats['processed_patch_pixels'],
            source_pixels=stats['source_pixels'],
            processed_fraction=stats['processed_fraction']))
        if (i + 1) % 20 == 0 or i + 1 == len(sparse_ds):
            print('Checked {}/{} frames'.format(i + 1, len(sparse_ds)))

    report = ROOT / 'experiment_runs' / (Path(args.config).stem + '_preflight.json')
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(rows, indent=2))
    for key in ('roi_count', 'selected_tokens', 'patch_count',
                'processed_patch_pixels', 'processed_fraction'):
        values = [r[key] for r in rows]
        print('{}: mean={:.3f}, min={}, max={}'.format(
            key, float(np.mean(values)), min(values), max(values)))
    print('All six images and calibration/pose tensors match baseline exactly.')
    print('Sparse ROI/patch preflight passed; report:', report)


if __name__ == '__main__':
    main()
