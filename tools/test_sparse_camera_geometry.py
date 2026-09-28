#!/usr/bin/env python3
"""Pure-NumPy tests for native sparse-camera ROI/patch bookkeeping."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load the pure geometry module directly so these tests do not require MMCV.
import importlib.util
_mod_path = ROOT / 'projects/mmdet3d_plugin/datasets/pipelines/sparse_camera_geometry.py'
_spec = importlib.util.spec_from_file_location('sparse_camera_geometry', str(_mod_path))
_geom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_geom)
roi_to_token_rect = _geom.roi_to_token_rect
processing_patch = _geom.processing_patch
merge_processing_patches = _geom.merge_processing_patches
selected_token_mask = _geom.selected_token_mask
rect_area = _geom.rect_area
feature_cell_center = _geom.feature_cell_center


def test_token_rect():
    r = roi_to_token_rect((17, 33, 63, 81), 704, 256, 16)
    assert r == (1, 2, 4, 6), r


def test_halo_does_not_change_selection():
    roi = (101, 65, 201, 145)
    token = roi_to_token_rect(roi, 704, 256, 16)
    a = processing_patch(token, 704, 256, 16, halo=0, alignment=32)
    b = processing_patch(token, 704, 256, 16, halo=64, alignment=32)
    assert a != b
    assert token == roi_to_token_rect(roi, 704, 256, 16)
    px = (token[0] * 16, token[1] * 16, token[2] * 16, token[3] * 16)
    for patch in (a, b):
        assert patch[0] <= px[0] and patch[1] <= px[1]
        assert patch[2] >= px[2] and patch[3] >= px[3]
        assert all(v % 32 == 0 for v in patch)


def test_merge_never_increases_pixels():
    patches = [
        dict(crop=(32, 32, 192, 192), roi_ids=[0]),
        dict(crop=(128, 32, 288, 192), roi_ids=[1]),
        dict(crop=(448, 32, 608, 192), roi_ids=[2]),
    ]
    before = sum(rect_area(p['crop']) for p in patches)
    merged = merge_processing_patches(patches, enabled=True)
    after = sum(rect_area(p['crop']) for p in merged)
    assert after <= before
    ids = sorted(x for p in merged for x in p['roi_ids'])
    assert ids == [0, 1, 2]


def test_merge_does_not_change_token_set():
    records = [
        dict(camera_id=0, token_rect=(3, 3, 8, 8)),
        dict(camera_id=0, token_rect=(6, 4, 10, 9)),
        dict(camera_id=1, token_rect=(1, 1, 2, 2)),
    ]
    before = selected_token_mask(6, 16, 44, records)
    raw = []
    for i, rec in enumerate(records[:2]):
        raw.append(dict(crop=processing_patch(rec['token_rect'], 704, 256,
                                              16, 64, 32), roi_ids=[i]))
    merge_processing_patches(raw, enabled=True)
    after = selected_token_mask(6, 16, 44, records)
    assert np.array_equal(before, after)



def test_view_major_flat_indexing():
    records = [dict(camera_id=2, token_rect=(4, 3, 5, 4))]
    mask = selected_token_mask(6, 16, 44, records)
    flat = np.flatnonzero(mask.reshape(-1))
    assert flat.tolist() == [2 * 16 * 44 + 3 * 44 + 4]

def test_overlap_counted_once():
    records = [
        dict(camera_id=2, token_rect=(5, 5, 10, 10)),
        dict(camera_id=2, token_rect=(8, 8, 12, 12)),
    ]
    mask = selected_token_mask(6, 16, 44, records)
    expected = 25 + 16 - 4
    assert int(mask.sum()) == expected


def test_feature_center_maps_to_original_camera():
    # A patch beginning at x=128,y=64 has global feature origin (8,4).
    patch = (128, 64, 320, 224)
    gx0, gy0 = patch[0] // 16, patch[1] // 16
    local_x, local_y = 3, 2
    global_x, global_y = gx0 + local_x, gy0 + local_y
    center = feature_cell_center(4, global_x, global_y, 16)
    assert center['camera_id'] == 4
    assert center['x'] == patch[0] + local_x * 16 + 8
    assert center['y'] == patch[1] + local_y * 16 + 8


def test_edge_patch_stays_aligned():
    token = roi_to_token_rect((0, 0, 33, 35), 704, 256, 16)
    patch = processing_patch(token, 704, 256, 16, halo=64, alignment=32)
    assert patch[0] == 0 and patch[1] == 0
    assert all(v % 32 == 0 for v in patch)


def test_randomized_coverage():
    rng = np.random.RandomState(7)
    for _ in range(1000):
        x0 = int(rng.randint(0, 700))
        y0 = int(rng.randint(0, 252))
        x1 = int(rng.randint(x0 + 1, 705))
        y1 = int(rng.randint(y0 + 1, 257))
        x1, y1 = min(x1, 704), min(y1, 256)
        token = roi_to_token_rect((x0, y0, x1, y1), 704, 256, 16)
        patch = processing_patch(token, 704, 256, 16, halo=64, alignment=32)
        px = (token[0]*16, token[1]*16, token[2]*16, token[3]*16)
        assert patch[0] <= px[0] <= px[2] <= patch[2]
        assert patch[1] <= px[1] <= px[3] <= patch[3]
        assert rect_area(patch) > 0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('PASS', test.__name__)
    print('{} sparse-camera geometry tests passed'.format(len(tests)))


if __name__ == '__main__':
    main()
