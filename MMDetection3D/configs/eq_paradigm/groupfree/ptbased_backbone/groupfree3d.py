model = dict(
    type='GroupFree3DNet',
    backbone=dict(
        type='MMDet3DPointNet2Backbone',
        model_cfg=None,
        input_channels=3,
        SA_CONFIG=dict(
            NPOINTS=[2048, 1024, 512, 256],
            MLPS=[
                [[64, 64, 128]], [[128, 128, 256]],
                [[128, 128, 256]], [[128, 128, 256]],
            ],
            RADIUS=[[0.2], [0.4], [0.8], [1.2]],
            NSAMPLE=[[64], [32], [16], [16]],
            USE_XYZ=True,
            NORMALIZE_XYZ=True,
            ),
        ),
    neck=dict(
        type='MMDet3DQNetNeck',
        model_cfg=None,
        point_cloud_range=None,
        voxel_size=None,
        SUPPORT_CFG=dict(
            PREPROCESSING_FUNC='_preprocess_point_support_features',
            SUPPORT_KEYS=['l0', 'l1', 'l2', 'l3'],
        ),
        QUERY_POSITION_CFG=dict(
            SELECTION_FUNCTION='_get_keypoints_query_position',
            KEYPOINTS_SRC='raw_points',
            KEYPOINTS_SAMPLE_METHOD='FPS',
            KEYPOINTS_NUM=1024,
        ),
        # path to your qnet configure file.
        QNET_CFG_FILE='configs/eq_paradigm/groupfree/ptbased_backbone/qnet.yaml',
        QUERY_FEATURE_CFG=dict(
            POSTPROCESSING_FUNC='_process_keypoints_query_features',
        )
    ),
    bbox_head=dict(
        type='GroupFree3DHead',
        in_channels=288,
        num_decoder_layers=6,
        num_proposal=256,
        transformerlayers=dict(
            type='BaseTransformerLayer',
            attn_cfgs=dict(
                type='GroupFree3DMHA',
                embed_dims=288,
                num_heads=8,
                attn_drop=0.1,
                dropout_layer=dict(type='Dropout', drop_prob=0.1)),
            ffn_cfgs=dict(
                embed_dims=288,
                feedforward_channels=2048,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn',
                             'norm')),
        aux_module_cfg=dict(
            in_channels=192,  # layer-wise supervision.
            gt_per_seed=3,
            conv_channels=(256, 256),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            offset_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=20.0),
            class_loss=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=20.0), ),
        pred_layer_cfg=dict(
            in_channels=288, shared_conv_channels=(288, 288), bias=True),
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
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss', beta=1.0, reduction='sum', loss_weight=10.0),
        semantic_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(sample_mod='kps'),
    test_cfg=dict(
        sample_mod='kps',
        nms_thr=0.25,
        score_thr=0.0,
        per_class_proposal=True,
        prediction_stages='last'))
