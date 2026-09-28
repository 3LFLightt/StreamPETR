# Oracle GT ROI ablation: native-scale patches stay in their six physical cameras.
# The normal baseline resize/crop/normalize/pad pipeline is preserved.
_base_ = ['./stream_petr_r50_704_bs2_seq_428q_nui_60e.py']

collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                'img_timestamp', 'ego_pose', 'ego_pose_inv']
class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900,
    'W': 1600,
    'rand_flip': True,
}

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # First do EXACTLY the baseline deterministic test-time image transform.
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    # Then select ROI metadata in that unchanged six-camera coordinate system.
    dict(type='OracleSparseCameraROIs', margin=1.3, near=0.1,
         feature_stride=16, halo=64, patch_alignment=32,
         merge_patches=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PETRFormatBundle3D', collect_keys=collect_keys,
                 class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img'] + collect_keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'scene_token', 'sample_idx',
                            'sparse_camera_spec', 'sparse_camera_stats')),
        ])
]

model = dict(
    type='SparsePetr3D',
    sparse_mode='patch',
    sparse_feature_stride=16,
)

data = dict(
    samples_per_gpu=1,
    val=dict(type='OracleSparseNuScenesDataset', pipeline=test_pipeline,
             test_mode=True),
    test=dict(type='OracleSparseNuScenesDataset', pipeline=test_pipeline,
              test_mode=True),
)
evaluation = dict(pipeline=test_pipeline)
