# dataset settings
dataset_type = 'ADE20KDataset'
data_root = '/app/AI/data/ade/ADEChallengeData2016'
crop_size = (448, 448) # 여기 448이랑 scale에서 448이랑 일치해야 함. (?)
#crop_size = (512, 512)

# ====================== Train Setting ======================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2016, 448),  # [224 * 9 = 2016, 224 * 2 = 448]
        ratio_range=(0.5, 2.0),
        keep_ratio=True
        ),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='Pad', size_divisor=224, pad_val=0),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='PackSegInputs')
]

# ====================== Test Setting ======================
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=(2016, 448), keep_ratio=True),
#     # add loading annotation after ``Resize`` because ground truth
#     # does not need to do resize data transform
#     dict(type='LoadAnnotations', reduce_zero_label=True),
#     dict(type='PackSegInputs')
# ]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2016, 448), keep_ratio=True),
    dict(type='ResizeToMultiple', size_divisor=224, ),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]


img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
#img_ratios = [1.0]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            #[dict(type='Pad', size_divisor=224)],

            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], 
            [dict(type='LoadAnnotations')], 
            [dict(type='PackSegInputs')]
        ])
]


train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
