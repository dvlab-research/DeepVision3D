"""
PVRCNN-T in EQ-Paradigm paper: https://arxiv.org/pdf/2203.01252.pdf which
 utilizes proposal grids as query positions for refinement.
"""

import torch.nn as nn
from .roi_head_template import RoIHeadTemplate

from pcdet.models.neck import __all__ as __neck_all__
from pcdet.models.dense_heads import PointHeadAuxLoss


class ProposalGridHead(RoIHeadTemplate):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        # Define proposal grid feature generation module.
        qnet_cfg = self.model_cfg.get('QNet_CONFIG')
        self.qnet = __neck_all__[qnet_cfg.NAME](qnet_cfg,
                                                point_cloud_range=point_cloud_range,
                                                voxel_size=voxel_size)
        # must use proposal grid as query position.
        assert qnet_cfg.QUERY_POSITION_CFG.SELECTION_FUNCTION == '_get_proposal_grids_query_position'
        grid_size = qnet_cfg.QUERY_POSITION_CFG.PROPOSAL_GRID_SIZE
        pre_channel = grid_size * grid_size * grid_size * self.qnet.num_point_features

        # Define aux loss computation module.
        aux_loss_cfg = self.model_cfg.get('AUX_LOSS_CONFIG', None)
        if aux_loss_cfg is not None:
            self.aux_loss_module = PointHeadAuxLoss(
                num_class=aux_loss_cfg.NUM_CLASS,
                input_channels=aux_loss_cfg.INPUT_CHANNELS,
                model_cfg=aux_loss_cfg)
        else:
            self.aux_loss_module = None

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)

        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)

        self.init_weights()

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    
        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)

    def get_loss(self, tb_dict=None):
        rcnn_loss, tb_dict = super().get_loss(tb_dict)

        # add auxiliary loss (optional):
        if self.aux_loss_module is not None:
            loss_point, tb_dict = self.aux_loss_module.get_loss(tb_dict)
            rcnn_loss = rcnn_loss + loss_point
            tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        batch_dict = self.qnet(batch_dict)
        # Producing auxiliary loss (optional):
        batch_dict = self.aux_loss_module(batch_dict) if self.aux_loss_module is not None else batch_dict
        pooled_features = batch_dict['proposal_grid_features']  # (BxN, 6x6x6, C)

        # Box Refinement
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        shared_features = self.shared_fc_layer(pooled_features)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
