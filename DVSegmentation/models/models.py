import torch.nn as nn
import torch
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from . import __backbone__, __neck__
from utils import utils
from ops.ops import voxelization


class Semantic(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        in_channel = 6 if model_cfg.input_channel else 3
        self.use_coords = model_cfg.use_coords
        self.num_class = model_cfg.classes

        # build backbone.
        backbone_cfg = model_cfg.BACKBONE
        self.backbone = __backbone__[backbone_cfg.NAME](
            model_cfg=backbone_cfg, input_channels=in_channel, grid_size=[512,512,512]
        )

        output_channels = self.backbone.num_point_features

        # build neck (optional).
        neck_cfg = model_cfg.get('NECK', None)
        self.neck, self.aux_module = None, None
        if neck_cfg is not None:
            self.neck = __neck__[neck_cfg.NAME](
                model_cfg=neck_cfg,
                voxel_size=1/model_cfg.scale, point_cloud_range=[0,0,0,20,20,10], coord_inverse=False
            )
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
                aux_module.append(nn.Linear(aux_mlp[-1], self.num_class))
                self.aux_module = nn.Sequential(*aux_module)

        self.classifier = nn.Linear(output_channels, self.num_class)


    def forward(self, batch):
        for key in batch:
            value = batch[key]
            if isinstance(value, torch.Tensor):
                batch[key] = value.cuda()

        batch['batch_size'] = len(batch['offsets']) - 1
        batch['sparse_shape'] = batch['spatial_shape']
        batch['voxel_coords'] = batch['voxel_locs']
        feats = batch['feats']
        coords_float = batch['locs_float']
        point_feats = feats if not self.use_coords else torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(point_feats, batch['v2p_map'], 4)
        batch['voxel_features'] = voxel_feats
        batch['points'] = torch.cat((batch['locs'][:, :1].to(coords_float), coords_float), dim=1)

        ret_dict = self.backbone(batch)

        if self.neck is not None:
            ret_dict = self.neck(ret_dict)

        query_features = ret_dict['query_features']
        scores = self.classifier(query_features)   # (B, N, C)
        scores = scores.view(-1, scores.shape[-1])   # (B * N, C)

        if self.training and self.aux_module is not None:
            aux_features = ret_dict['aux_query_features']  # (nAux, B, N, C')
            nAux, B, N, C = aux_features.size()
            aux_features = aux_features.view(-1, C)
            aux_scores = self.aux_module(aux_features)
            aux_scores = aux_scores.view(nAux, B * N, -1)     # (nAux, B * N, C'')
        else:
            aux_scores = None

        sampled_indices = ret_dict['query_sampled_indices']   # (B, N)
        sampled_indices = (sampled_indices + batch['offsets'][:-1].view(-1, 1)).view(-1)   # (B * N)

        return scores, aux_scores, sampled_indices


def model_fn_decorator(cfg, test=False):
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()

    def model_fn(batch, model, epoch):
        semantic_scores, aux_semantic_scores, sampled_indices = model(batch)

        if not test:
            semantic_labels = batch['labels'][sampled_indices].to(semantic_scores.device)
            loss, loss_out = loss_fn(semantic_scores, semantic_labels, aux_semantic_scores)

        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores

            if test:
                return preds

            p = semantic_scores.max(1)[1].cpu().numpy()
            gt = semantic_labels.cpu().numpy()
            i, u, target = utils.intersectionAndUnion(p, gt, cfg.classes, cfg.ignore_label)

            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), semantic_labels.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

            meter_dict['intersection'] = (i, 1)
            meter_dict['union'] = (u, 1)
            meter_dict['target'] = (target, 1)

            return loss, preds, visual_dict, meter_dict


    def loss_fn(semantic_scores, semantic_labels, aux_semantic_scores=None):

        loss_out = {}

        semantic_loss = criterion(semantic_scores, semantic_labels)
        loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])

        aux_semantic_loss = 0
        if aux_semantic_scores is not None:
            for i in range(len(aux_semantic_scores)):
                aux_semantic_loss += criterion(aux_semantic_scores[i], semantic_labels)
            aux_semantic_loss /= len(aux_semantic_scores)
            loss_out['semantic_loss_aux'] = (aux_semantic_loss, semantic_labels.shape[0])

        loss = semantic_loss + aux_semantic_loss
        return loss, loss_out


    return model_fn
