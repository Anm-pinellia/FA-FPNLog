# dataset settings
dataset_type = 'DOTADatasetV10Seg'
data_root = r'G:/Datasets/DOTA/DOTA_V1.0_splited_mini/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_seg=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=1.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False, with_seg=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + '/train/labelTxt-v1.5/DOTA-v1.5_train/',
        ann_file=data_root + '/train/annfiles/',
        img_prefix=data_root + 'train/images/',
        seg_prefix=data_root + 'train/segvagues/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + '/val/labelTxt-v1.5/DOTA-v1.5_val/',
        ann_file=data_root + '/val/annfiles/',
        img_prefix=data_root + '/val/images/',
        seg_prefix=data_root + 'val/segvagues/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/val/images/',
        img_prefix=data_root + '/val/images/',
        seg_prefix=data_root + 'val/segvagues/',
        pipeline=test_pipeline))
