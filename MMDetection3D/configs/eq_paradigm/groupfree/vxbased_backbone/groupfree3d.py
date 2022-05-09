voxel_size = [0.05, 0.05, 0.05]
point_cloud_range = [0, 0, 0, 8, 8, 8]
grid_size = [160, 160, 160]  # point_cloud_range // voxel_size

model = dict(
    type='GroupFree3DNet',
    backbone=dict(
        type='MMDet3DVoxelBackbone',
        model_cfg=None,
        VOXEL_LAYER_CFG=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(40000, 40000)
        ),
        VOXEL_ENCODER_CFG=dict(type='HardSimpleVFE'),
        input_channels=4,
        grid_size=grid_size,  # point_cloud_range // voxel_size
        INIT_CONV_CFG=dict(
          conv_type='subm',
          out_channels=16,
          kernel_size=3,
          indice_key='init_conv',
          stride=1,
          padding=0,
        ),
        BACKBONE_CFG=dict(
          block_types=[
              [ 'default_block' ],
              [ 'default_block', 'default_block', 'default_block' ],
              [ 'default_block', 'default_block', 'default_block' ],
              [ 'default_block', 'default_block', 'default_block' ],
            ],
          out_channels=[16, 32, 64, 128],
          conv_type=['subm', 'spconv', 'spconv', 'spconv'],
          kernel_size=[3, 3, 3, 3],
          stride=[1, 2, 2, 2],
          padding=[1, 1, 1, 1],
        ),
    ),
    neck=dict(
        type='MMDet3DQNetNeck',
        model_cfg=None,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        SUPPORT_CFG=dict(
            PREPROCESSING_FUNC='_preprocess_voxel_support_features',
            SUPPORT_KEYS=['x_conv3', 'x_conv4'],
        ),
        QUERY_POSITION_CFG=dict(
            SELECTION_FUNCTION='_get_keypoints_query_position',
            KEYPOINTS_SRC='raw_points',
            KEYPOINTS_SAMPLE_METHOD='FPS',
            KEYPOINTS_NUM=1024,
        ),
        # path to your qnet configure file.
        QNET_CFG_FILE='configs/eq_paradigm/groupfree/vxbased_backbone/qnet.yaml',
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
            in_channels=96,  # layer-wise supervision.
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
