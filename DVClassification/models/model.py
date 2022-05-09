import torch
import torch.nn as nn
import torch.nn.functional as F

from . import __backbone__, __neck__


class get_model(nn.Module):
    def __init__(self, model_cfg, normal_channel, num_class):
        super().__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.num_class = num_class

        # build backbone.
        backbone_cfg = model_cfg.BACKBONE
        self.backbone = __backbone__[backbone_cfg.NAME](
            model_cfg=backbone_cfg, input_channels=in_channel)

        output_channels = self.backbone.num_point_features

        # build neck (optional).
        neck_cfg = model_cfg.get('NECK', None)
        self.neck, self.aux_module = None, None
        if neck_cfg is not None:
            self.neck = __neck__[neck_cfg.NAME](
                model_cfg=neck_cfg, point_cloud_range=None, voxel_size=None,)
            output_channels = self.neck.num_point_features

            # define auxiliary supervision. (optional).
            aux_cfg = model_cfg.get('AUX_SUPP_CFG', None)
            if aux_cfg is not None:
                aux_mlp = aux_cfg.MLP
                aux_module = []
                for i in range(1, len(aux_mlp)):
                    aux_module.extend([
                        nn.Linear(aux_mlp[i-1], aux_mlp[i]),
                        nn.BatchNorm1d(aux_mlp[i]),
                        nn.ReLU(),
                    ])
                aux_module.append(nn.Linear(aux_mlp[-1], num_class))
                self.aux_module = nn.Sequential(*aux_module)

        # finally build classification head.
        self.class_head = nn.Sequential(
            nn.Linear(output_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_class))

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        ret_dict = dict(
            xyz=xyz, feat=norm
        )

        ret_dict = self.backbone(ret_dict)
        if self.neck is not None:
            ret_dict = self.neck(ret_dict)
        if self.training and self.aux_module is not None:
            aux_features = ret_dict['aux_query_features']  # h, b, n, c
            # h, b, n, c ---> b, h, n, c
            aux_features = aux_features.permute(1, 0, 2, 3).contiguous()
            b, h, n, c = aux_features.size()
            aux_features = aux_features.view(b * h * n, c)
            aux_cls_preds = self.aux_module(aux_features)
            aux_cls_preds = F.log_softmax(aux_cls_preds, -1)
            aux_cls_preds = aux_cls_preds.view(b, h * n, self.num_class)
        else:
            aux_cls_preds = None

        # obtain classification results.
        point_features = ret_dict['point_features']  # b, n, c
        b, n, c = point_features.size()
        point_features = point_features.view(b * n, c)

        # obtain candidate classification result.
        candidate_cls_preds = self.class_head(point_features)  # b * n, num_class
        candidate_cls_preds = candidate_cls_preds.view(b, n, self.num_class)

        # obtain final classification result.
        voted_cls_preds = candidate_cls_preds.mean(1)

        # generate classification score.
        voted_cls_preds = F.log_softmax(voted_cls_preds, -1)  # b, num_class
        candidate_cls_preds = F.log_softmax(candidate_cls_preds, -1)  # b, n, num_class

        return voted_cls_preds, [candidate_cls_preds, aux_cls_preds]


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def _aux_cls_loss(self, aux_cls_preds, target):
        b, n, c = aux_cls_preds.size()
        target = target.view(b, 1).repeat(1, n).contiguous()
        loss = F.nll_loss(aux_cls_preds.view(b * n, c), target.view(-1))
        return loss

    def forward(self, voted_cls_preds, target, aux_cls_preds=None):
        total_loss = F.nll_loss(voted_cls_preds, target)

        if aux_cls_preds is not None:
            candidate_cls_preds, aux_cls_preds = aux_cls_preds
            candidate_cls_loss = self._aux_cls_loss(candidate_cls_preds, target)
            aux_cls_loss = self._aux_cls_loss(aux_cls_preds, target)
            total_loss = total_loss + candidate_cls_loss + aux_cls_loss

        return total_loss
