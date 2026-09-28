"""Sparse native-scale six-camera StreamPETR inference detector."""
import torch
from mmdet.models import DETECTORS

from .petr3d import Petr3D


@DETECTORS.register_module()
class SparsePetr3D(Petr3D):
    """Inference-only StreamPETR detector with sparse image-side processing.

    Modes:
      patch:       run backbone/neck only on native-scale processing patches,
                   scatter retained ROI features into the original 6-camera grid.
      full_roi:    full-image backbone/neck, but only ROI tokens enter attention.
      full_tokens: full-image backbone/neck and all tokens enter attention.

    The decoder, object queries, temporal memory, camera matrices and checkpoint
    are unchanged.  Patches are never represented as additional cameras.
    """
    def __init__(self, sparse_mode='patch', sparse_feature_stride=16, **kwargs):
        super().__init__(**kwargs)
        if sparse_mode not in ('patch', 'full_roi', 'full_tokens'):
            raise ValueError('Unknown sparse_mode: %s' % sparse_mode)
        if sparse_feature_stride <= 0:
            raise ValueError('sparse_feature_stride must be positive')
        self.sparse_mode = sparse_mode
        self.sparse_feature_stride = int(sparse_feature_stride)

    @staticmethod
    def _spec(img_metas):
        if not img_metas or 'sparse_camera_spec' not in img_metas[0]:
            raise RuntimeError('SparsePetr3D requires sparse_camera_spec metadata')
        return img_metas[0]['sparse_camera_spec']

    def _selected_mask(self, spec, device):
        n = int(spec['num_cameras'])
        h = int(spec['feature_height'])
        w = int(spec['feature_width'])
        mask = torch.zeros((n, h, w), dtype=torch.bool, device=device)
        for rec in spec['rois']:
            cam = int(rec['camera_id'])
            x0, y0, x1, y1 = [int(v) for v in rec['token_rect']]
            mask[cam, y0:y1, x0:x1] = True
        if not mask.any():
            raise RuntimeError('Sparse ROI token mask is empty')
        return mask

    @staticmethod
    def _topk_from_mask(mask):
        # Flattening N,H,W matches StreamPETR memory's view-major flattening.
        return torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(1)[None]

    def _backbone_neck(self, x):
        if self.use_grid_mask:
            x = self.grid_mask(x)
        feats = self.img_backbone(x)
        if isinstance(feats, dict):
            feats = list(feats.values())
        if self.with_img_neck:
            feats = self.img_neck(feats)
        return feats[self.position_level]

    def extract_sparse_patch_feat(self, img, img_metas):
        if img.dim() != 5 or img.size(0) != 1:
            raise RuntimeError('Sparse patch inference currently requires batch size 1')
        spec = self._spec(img_metas)
        ncam = int(spec['num_cameras'])
        _, n, _, image_h, image_w = img.shape
        if n != ncam:
            raise RuntimeError('Camera count differs between image tensor and ROI spec')
        if image_h != int(spec['image_height']) or image_w != int(spec['image_width']):
            raise RuntimeError('Image dimensions differ from ROI spec')
        stride = self.sparse_feature_stride
        feature_h = int(spec['feature_height'])
        feature_w = int(spec['feature_width'])
        if feature_h != image_h // stride or feature_w != image_w // stride:
            raise RuntimeError('Feature-grid dimensions disagree with configured stride')

        selected = self._selected_mask(spec, img.device)
        filled = torch.zeros_like(selected)
        full = None
        patch_shapes = []

        for patch in spec['patches']:
            cam = int(patch['camera_id'])
            x0, y0, x1, y1 = [int(v) for v in patch['crop']]
            if x0 % stride or y0 % stride or x1 % stride or y1 % stride:
                raise RuntimeError('Processing patch is not feature-stride aligned')
            crop = img[:, cam, :, y0:y1, x0:x1]
            feat = self._backbone_neck(crop)
            patch_shapes.append([int(v) for v in crop.shape])
            expected_h = (y1 - y0) // stride
            expected_w = (x1 - x0) // stride
            if feat.shape[-2:] != (expected_h, expected_w):
                raise RuntimeError(
                    'Patch feature shape %s != expected (%d,%d) for crop %s' %
                    (tuple(feat.shape[-2:]), expected_h, expected_w,
                     (x0, y0, x1, y1)))
            if full is None:
                full = feat.new_zeros(
                    (1, ncam, int(feat.shape[1]), feature_h, feature_w))

            gx0, gy0 = x0 // stride, y0 // stride
            gx1, gy1 = x1 // stride, y1 // stride
            keep = selected[cam, gy0:gy1, gx0:gx1] & ~filled[cam, gy0:gy1, gx0:gx1]
            ys, xs = torch.nonzero(keep, as_tuple=True)
            if ys.numel():
                full[0, cam, :, gy0 + ys, gx0 + xs] = feat[0, :, ys, xs]
                filled[cam, gy0 + ys, gx0 + xs] = True

        if full is None:
            raise RuntimeError('Sparse processing generated zero backbone patches')
        missing = selected & ~filled
        if missing.any():
            missing_count = int(missing.sum().item())
            raise RuntimeError('%d selected ROI tokens were not produced by a patch' % missing_count)

        topk = self._topk_from_mask(selected)
        runtime = dict(
            mode='patch',
            selected_tokens=int(selected.sum().item()),
            filled_tokens=int(filled.sum().item()),
            patch_count=len(spec['patches']),
            backbone_patch_shapes=patch_shapes)
        return full, topk, runtime

    def extract_sparse_full_feat(self, img, img_metas, prune_tokens):
        feats = super().extract_img_feat(img, 1)
        spec = self._spec(img_metas)
        selected = self._selected_mask(spec, feats.device)
        expected = tuple(int(v) for v in feats.shape[-2:])
        configured = (int(spec['feature_height']), int(spec['feature_width']))
        if expected != configured:
            raise RuntimeError('Full feature grid %s != ROI grid %s' % (expected, configured))
        topk = self._topk_from_mask(selected) if prune_tokens else None
        runtime = dict(
            mode='full_roi' if prune_tokens else 'full_tokens',
            selected_tokens=int(selected.sum().item()),
            full_tokens=int(selected.numel()),
            patch_count=0)
        return feats, topk, runtime

    def forward_roi_head(self, location, **data):
        if not self.training and data.get('sparse_topk_indexes') is not None:
            return {'topk_indexes': data['sparse_topk_indexes']}
        return super().forward_roi_head(location, **data)

    def simple_test(self, img_metas, **data):
        if self.training:
            raise RuntimeError('SparsePetr3D is intended for evaluation/inference only')
        if self.sparse_mode == 'patch':
            feats, topk, runtime = self.extract_sparse_patch_feat(data['img'], img_metas)
        elif self.sparse_mode == 'full_roi':
            feats, topk, runtime = self.extract_sparse_full_feat(
                data['img'], img_metas, prune_tokens=True)
        else:
            feats, topk, runtime = self.extract_sparse_full_feat(
                data['img'], img_metas, prune_tokens=False)

        data['img_feats'] = feats
        data['sparse_topk_indexes'] = topk
        img_metas[0]['sparse_runtime_stats'] = runtime

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
