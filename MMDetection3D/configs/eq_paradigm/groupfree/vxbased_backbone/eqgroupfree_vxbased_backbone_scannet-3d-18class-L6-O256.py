_base_ = [
    '../../vxbased_datasets/scannet-3d-18class.py',
    './groupfree3d.py',
    '../../../_base_/schedules/schedule_3x.py',
    '../../../_base_/default_runtime.py'
]

# model settings
model = dict(
    bbox_head=dict(
        num_classes=18,
        size_cls_agnostic=False,
        bbox_coder=dict(
            type='GroupFree3DBBoxCoder',
            num_sizes=18,
            num_dir_bins=1,
            with_rot=False,
            size_cls_agnostic=False,
            mean_sizes=[[0.76966727, 0.8116021, 0.92573744],
                        [1.876858, 1.8425595, 1.1931566],
                        [0.61328, 0.6148609, 0.7182701],
                        [1.3955007, 1.5121545, 0.83443564],
                        [0.97949594, 1.0675149, 0.6329687],
                        [0.531663, 0.5955577, 1.7500148],
                        [0.9624706, 0.72462326, 1.1481868],
                        [0.83221924, 1.0490936, 1.6875663],
                        [0.21132214, 0.4206159, 0.5372846],
                        [1.4440073, 1.8970833, 0.26985747],
                        [1.0294262, 1.4040797, 0.87554324],
                        [1.3766412, 0.65521795, 1.6813129],
                        [0.6650819, 0.71111923, 1.298853],
                        [0.41999173, 0.37906948, 1.7513971],
                        [0.59359556, 0.5912492, 0.73919016],
                        [0.50867593, 0.50656086, 0.30136237],
                        [1.1511526, 1.0546296, 0.49706793],
                        [0.47535285, 0.49249494, 0.5802117]]),
        sampling_objectness_loss=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=8.0),
        objectness_loss=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        center_loss=dict(
            type='SmoothL1Loss', beta=0.04, reduction='sum', loss_weight=10.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            reduction='sum',
            loss_weight=10.0 / 9.0),
        semantic_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)),
    test_cfg=dict(
        sample_mod='kps',
        nms_thr=0.25,
        score_thr=0.0,
        per_class_proposal=True,
        prediction_stages='last_three'))

# dataset settings
data = dict(samples_per_gpu=4, workers_per_gpu=4)

# Set find_unused.
find_unused_parameters = True

# optimizer
lr = 0.003
optimizer = dict(
    lr=lr,
    weight_decay=0.0005,
    paramwise_cfg=dict(
        custom_keys={
            'bbox_head.decoder_layers': dict(lr_mult=0.1, decay_mult=1.0),
            'bbox_head.decoder_self_posembeds': dict(
                lr_mult=0.1, decay_mult=1.0),
            'bbox_head.decoder_cross_posembeds': dict(
                lr_mult=0.1, decay_mult=1.0),
            'bbox_head.decoder_query_proj': dict(lr_mult=0.1, decay_mult=1.0),
            'bbox_head.decoder_key_proj': dict(lr_mult=0.1, decay_mult=1.0),
            'neck': dict(lr_mult=0.1, decay_mult=1.0),
        }))


optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[56, 68])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=80)
checkpoint_config = dict(interval=1, max_keep_ckpts=10)