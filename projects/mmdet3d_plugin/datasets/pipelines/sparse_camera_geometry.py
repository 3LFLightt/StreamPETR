"""Geometry helpers for sparse native-scale camera processing.

All rectangles are integer half-open (x0, y0, x1, y1).  ROI selection lives
in the original, normally-preprocessed camera coordinate system.  Processing
patches may be larger because of halo/context, but the selected token set does
not depend on halo or patch merging.
"""
import math
import numpy as np


def roi_to_token_rect(crop, width, height, stride=16):
    """Return the stride-grid cells intersecting a pixel ROI."""
    if stride <= 0:
        raise ValueError('stride must be positive')
    x0, y0, x1, y1 = [int(v) for v in crop]
    x0, x1 = max(0, x0), min(int(width), x1)
    y0, y1 = max(0, y0), min(int(height), y1)
    if x1 <= x0 or y1 <= y0:
        return None
    fw = int(math.ceil(width / float(stride)))
    fh = int(math.ceil(height / float(stride)))
    tx0 = max(0, min(fw, x0 // stride))
    ty0 = max(0, min(fh, y0 // stride))
    tx1 = max(tx0 + 1, min(fw, int(math.ceil(x1 / float(stride)))))
    ty1 = max(ty0 + 1, min(fh, int(math.ceil(y1 / float(stride)))))
    return (tx0, ty0, tx1, ty1)


def token_rect_to_pixels(token_rect, stride=16):
    tx0, ty0, tx1, ty1 = [int(v) for v in token_rect]
    return (tx0 * stride, ty0 * stride, tx1 * stride, ty1 * stride)


def processing_patch(token_rect, width, height, stride=16, halo=64,
                     alignment=32):
    """Expand retained tokens by halo and align the backbone crop outwards."""
    if halo < 0 or alignment <= 0:
        raise ValueError('halo must be >= 0 and alignment must be positive')
    x0, y0, x1, y1 = token_rect_to_pixels(token_rect, stride)
    x0, y0 = max(0, x0 - halo), max(0, y0 - halo)
    x1, y1 = min(int(width), x1 + halo), min(int(height), y1 + halo)

    x0 = (x0 // alignment) * alignment
    y0 = (y0 // alignment) * alignment
    x1 = int(math.ceil(x1 / float(alignment))) * alignment
    y1 = int(math.ceil(y1 / float(alignment))) * alignment
    x1, y1 = min(int(width), x1), min(int(height), y1)

    # The baseline StreamPETR input dimensions are alignment-divisible.  Keep
    # edge patches valid even for other dimensions by shifting inward.
    if (x1 - x0) % alignment:
        need = alignment - ((x1 - x0) % alignment)
        shift = min(x0, need)
        x0 -= shift
        need -= shift
        x1 = min(int(width), x1 + need)
    if (y1 - y0) % alignment:
        need = alignment - ((y1 - y0) % alignment)
        shift = min(y0, need)
        y0 -= shift
        need -= shift
        y1 = min(int(height), y1 + need)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('empty processing patch')
    return (int(x0), int(y0), int(x1), int(y1))


def rect_area(rect):
    x0, y0, x1, y1 = rect
    return max(0, x1 - x0) * max(0, y1 - y0)


def rect_union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


def merge_processing_patches(patches, enabled=True):
    """Merge same-camera processing patches only when pixels do not increase.

    `patches` are dictionaries containing `crop` and arbitrary membership
    metadata.  The retained ROI token rectangles are never changed.
    """
    groups = [dict(crop=tuple(p['crop']), roi_ids=list(p.get('roi_ids', [])))
              for p in patches]
    if not enabled:
        return groups
    while len(groups) > 1:
        best = None
        for i, a in enumerate(groups):
            for j in range(i + 1, len(groups)):
                b = groups[j]
                union = rect_union(a['crop'], b['crop'])
                before = rect_area(a['crop']) + rect_area(b['crop'])
                after = rect_area(union)
                if after > before:
                    continue
                saving = before - after
                # Prefer the largest saving, then stable input order.
                key = (-saving, i, j)
                if best is None or key < best[0]:
                    best = (key, union, a['roi_ids'] + b['roi_ids'])
        if best is None:
            break
        (_, i, j), union, roi_ids = best
        groups[i] = dict(crop=union, roi_ids=roi_ids)
        del groups[j]
    return groups


def selected_token_mask(num_cameras, feature_h, feature_w, records):
    """Boolean N,H,W mask from per-camera token rectangles."""
    mask = np.zeros((num_cameras, feature_h, feature_w), dtype=np.bool_)
    for rec in records:
        cam = int(rec['camera_id'])
        x0, y0, x1, y1 = [int(v) for v in rec['token_rect']]
        if not (0 <= cam < num_cameras):
            raise ValueError('camera index out of bounds')
        if not (0 <= x0 < x1 <= feature_w and 0 <= y0 < y1 <= feature_h):
            raise ValueError('token rectangle out of bounds')
        mask[cam, y0:y1, x0:x1] = True
    return mask


def feature_cell_center(camera_id, tx, ty, stride=16):
    """Original-camera pixel center of a retained feature cell."""
    return dict(camera_id=int(camera_id),
                x=float(tx * stride + stride / 2.0),
                y=float(ty * stride + stride / 2.0))
