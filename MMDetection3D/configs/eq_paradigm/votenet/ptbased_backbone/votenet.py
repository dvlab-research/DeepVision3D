model = dict(
    type='VoteNet',
    backbone=dict(
        type='MMDet3DPointNet2Backbone',
        model_cfg=None,
        input_channels=4,
        SA_CONFIG=dict(
            NPOINTS=[2048, 1024, 512, 256],
            MLPS=[
                [[64, 64, 128]], [[128, 128, 256]],
                [[128, 128, 256]], [[128, 128, 256]],
            ],
            RADIUS=[[0.2], [0.4], [0.8], [1.2]],
            NSAMPLE=[[64], [32], [16], [16]],
            USE_XYZ=True,
            NORMALIZE_XYZ=True),
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
        QNET_CFG_FILE='configs/eq_paradigm/votenet/ptbased_backbone/qnet.yaml',
        QUERY_FEATURE_CFG=dict(
            POSTPROCESSING_FUNC='_process_keypoints_query_features',
        )
    ),
    bbox_head=dict(
        type='VoteHead',
        vote_module_cfg=dict(
            in_channels=256,
            vote_per_seed=1,
            gt_per_seed=3,
            conv_channels=(256, 256),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            norm_feats=True,
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=10.0)),
        aux_module_cfg=dict(
            in_channels=192,
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
        vote_aggregation_cfg=dict(
            type='PointSAModule',
            num_point=256,
            radius=0.3,
            num_sample=16,
            mlp_channels=[256, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True),
        pred_layer_cfg=dict(
            in_channels=128, shared_conv_channels=(128, 128), bias=True),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        objectness_loss=dict(
            type='CrossEntropyLoss',
            class_weight=[0.2, 0.8],
            reduction='sum',
            loss_weight=5.0),
        center_loss=dict(
            type='ChamferDistance',
            mode='l2',
            reduction='sum',
            loss_src_weight=10.0,
            loss_dst_weight=10.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0 / 3.0),
        semantic_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mod='vote'),
    test_cfg=dict(
        sample_mod='seed',
        nms_thr=0.25,
        score_thr=0.05,
        per_class_proposal=True))
