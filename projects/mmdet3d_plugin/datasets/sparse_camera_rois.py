"""Oracle ROI metadata for six-camera sparse native-scale inference."""
import time
import numpy as np
from mmdet.datasets import DATASETS
from mmdet.datasets.builder import PIPELINES

from .nuscenes_dataset import CustomNuScenesDataset
from .pipelines.oracle_tile_geometry import project_box, square_crop
from .pipelines.sparse_camera_geometry import (
    roi_to_token_rect, processing_patch, merge_processing_patches,
    rect_area, selected_token_mask)


@DATASETS.register_module()
class OracleSparseNuScenesDataset(CustomNuScenesDataset):
    """Evaluation-only dataset exposing GT box corners to the ROI selector."""
    def get_data_info(self, index):
        if not self.test_mode:
            raise ValueError('OracleSparseNuScenesDataset is evaluation-only')
        results = super().get_data_info(index)
        ann = self.get_ann_info(index)
        boxes = ann['gt_bboxes_3d']
        labels = np.asarray(ann['gt_labels_3d'])
        results['oracle_corners'] = boxes[labels >= 0].corners.numpy()
        return results


@PIPELINES.register_module()
class OracleSparseCameraROIs:
    """Attach sparse ROI/processing-patch metadata without altering images.

    This transform must run AFTER the normal test-time ResizeCropFlipRotImage,
    so its coordinates and camera matrices exactly match the images seen by the
    baseline checkpoint.  It does not resize/crop/normalize an image itself.
    """
    def __init__(self, margin=1.3, near=0.1, feature_stride=16,
                 halo=64, patch_alignment=32, merge_patches=True):
        if margin < 1 or near <= 0:
            raise ValueError('Require margin >= 1 and near > 0')
        if feature_stride <= 0 or patch_alignment <= 0 or halo < 0:
            raise ValueError('Invalid stride/alignment/halo')
        if patch_alignment % feature_stride:
            raise ValueError('patch_alignment must be a multiple of feature_stride')
        self.margin = float(margin)
        self.near = float(near)
        self.feature_stride = int(feature_stride)
        self.halo = int(halo)
        self.patch_alignment = int(patch_alignment)
        self.merge_patches = bool(merge_patches)

    def __call__(self, results):
        start = time.perf_counter()
        corners = results.pop('oracle_corners')
        images = results['img']
        ncam = len(images)
        if ncam != 6:
            raise RuntimeError('Expected six physical nuScenes cameras, got %d' % ncam)

        heights = [int(im.shape[0]) for im in images]
        widths = [int(im.shape[1]) for im in images]
        if len(set(heights)) != 1 or len(set(widths)) != 1:
            raise RuntimeError('Sparse camera path requires equal preprocessed camera sizes')
        height, width = heights[0], widths[0]
        if height % self.patch_alignment or width % self.patch_alignment:
            raise RuntimeError('Preprocessed image size must be patch-alignment divisible')
        feature_h = height // self.feature_stride
        feature_w = width // self.feature_stride

        roi_records = []
        per_camera_raw_patches = [[] for _ in range(ncam)]
        roi_id = 0
        for camera_id in range(ncam):
            intrinsic = np.asarray(results['intrinsics'][camera_id])
            extrinsic = np.asarray(results['extrinsics'][camera_id])
            for box_id, box in enumerate(corners):
                bounds = project_box(box, intrinsic, extrinsic, self.near)
                if bounds is None:
                    continue
                crop = square_crop(bounds, width, height, self.margin)
                if crop is None:
                    continue
                token_rect = roi_to_token_rect(
                    crop, width, height, self.feature_stride)
                if token_rect is None:
                    continue
                patch = processing_patch(
                    token_rect, width, height,
                    stride=self.feature_stride,
                    halo=self.halo,
                    alignment=self.patch_alignment)
                rec = dict(
                    roi_id=roi_id,
                    camera_id=camera_id,
                    box_id=box_id,
                    projected_bounds=[float(v) for v in bounds],
                    roi_crop=[int(v) for v in crop],
                    token_rect=[int(v) for v in token_rect])
                roi_records.append(rec)
                per_camera_raw_patches[camera_id].append(
                    dict(crop=patch, roi_ids=[roi_id]))
                roi_id += 1

        if not roi_records:
            raise RuntimeError(
                'Oracle sparse selector produced zero ROIs for sample ' +
                str(results.get('sample_idx')))

        patch_records = []
        raw_patch_count = 0
        raw_patch_pixels = 0
        for camera_id, raw in enumerate(per_camera_raw_patches):
            raw_patch_count += len(raw)
            raw_patch_pixels += sum(rect_area(p['crop']) for p in raw)
            merged = merge_processing_patches(raw, enabled=self.merge_patches)
            for patch in merged:
                patch_records.append(dict(
                    camera_id=camera_id,
                    crop=[int(v) for v in patch['crop']],
                    roi_ids=[int(v) for v in patch['roi_ids']]))

        mask = selected_token_mask(ncam, feature_h, feature_w, roi_records)
        selected_count = int(mask.sum())
        processed_pixels = sum(rect_area(p['crop']) for p in patch_records)
        source_pixels = ncam * height * width
        if processed_pixels > raw_patch_pixels:
            raise AssertionError('Patch merging increased processed pixels')

        spec = dict(
            version=1,
            num_cameras=ncam,
            image_height=height,
            image_width=width,
            feature_stride=self.feature_stride,
            feature_height=feature_h,
            feature_width=feature_w,
            margin=self.margin,
            near=self.near,
            halo=self.halo,
            patch_alignment=self.patch_alignment,
            merge_patches=self.merge_patches,
            rois=roi_records,
            patches=patch_records)
        stats = dict(
            roi_count=len(roi_records),
            selected_token_count=selected_count,
            raw_patch_count=raw_patch_count,
            patch_count=len(patch_records),
            raw_patch_pixels=int(raw_patch_pixels),
            processed_patch_pixels=int(processed_pixels),
            source_pixels=int(source_pixels),
            processed_fraction=float(processed_pixels / float(source_pixels)),
            roi_build_ms=1000.0 * (time.perf_counter() - start))

        # Images and calibration are intentionally untouched.
        results['sparse_camera_spec'] = spec
        results['sparse_camera_stats'] = stats
        return results
