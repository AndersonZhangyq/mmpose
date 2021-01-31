log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='PCKh', key_indicator='PCKh')

optimizer = dict(
    type='Adam',
    lr=5e-3,
)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

channel_cfg = dict(
    num_output_channels=16,
    dataset_joints=16,
    dataset_channel=list(range(16)),
    inference_channel=list(range(16)))

# model settings
model = dict(
    type='TopDown',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='MSPN',
        unit_channels=256,
        num_stages=1,
        num_units=4,
        num_blocks=[3, 4, 6, 3],
        norm_cfg=dict(type='BN')),
    keypoint_head=dict(
        type='TopDownMSMUHead',
        out_shape=(64, 64),
        unit_channels=256,
        out_channels=channel_cfg['num_output_channels'],
        num_stages=1,
        num_units=4,
        use_prm=False,
        norm_cfg=dict(type='BN'),
        loss_keypoint=[
            dict(
                type='JointsMSELoss', use_target_weight=True, loss_weight=0.25)
        ] * 3 + [
            dict(
                type='JointsOHKMMSELoss',
                use_target_weight=True,
                loss_weight=1.)
        ]),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='megvii',
        shift_heatmap=False,
        modulate_kernel=5))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    use_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file=None,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        kernel=[(11, 11), (9, 9), (7, 7), (5, 5)],
        encoding='Megvii'),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=[
            'img',
        ],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/mpii'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_val.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_val.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
)
