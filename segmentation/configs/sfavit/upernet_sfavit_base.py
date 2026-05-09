_base_ = [
    '../_base_/models/upernet_sfavit.py', 
    '../_base_/datasets/ade20k.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

pretrained = '/sfavit_t_224.pth.tar'

# ========================== Model ==================================
#divisor=224
crop_size = (448, 448)

data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
            model_name='sfavit_b_224',
            pretrained_path=pretrained,
            drop_rate=0.,
            drop_path_rate=0.2,
            out_norm="LN"),
    decode_head=dict(in_channels=[64, 128, 256, 512], num_classes=150),
    auxiliary_head=dict(in_channels=256, num_classes=150))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    accumulative_counts=1,

    optimizer=dict(
        type='AdamW', lr=1e-04, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
