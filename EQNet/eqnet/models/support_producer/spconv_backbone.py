from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


from ..base import BasicEQModules
from eqnet.utils.spconv_utils import spconv, replace_feature
from ...utils.spconv_utils import replace_feature, spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=None,
                 indice_key=None, stride=1, padding=0,
                 conv_type=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            out_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(out_channels)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackboneBase(BasicEQModules):
    def __init__(self, model_cfg, adapt_model_cfg):
        super().__init__(model_cfg=model_cfg, adapt_model_cfg=adapt_model_cfg)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        sparse_shape = self.model_cfg.grid_size
        if isinstance(sparse_shape, list):
            sparse_shape = np.array(sparse_shape).astype(np.int64)
        self.sparse_shape = sparse_shape[::-1] + [1, 0, 0]
        input_channels = self.model_cfg.input_channels

        default_block = post_act_block
        block_dict = {
            'default_block': default_block,
            'sparse_res_block': SparseBasicBlock,
        }

        conv_input_cfg = self.model_cfg.get(
            'INIT_CONV_CFG',
            dict(
                conv_type='subm', out_channels=16, kernel_size=3,
                indice_key='init_conv', stride=1, padding=0,
            ))
        self.conv_input = default_block(
            input_channels,
            out_channels=conv_input_cfg.get('out_channels'),
            kernel_size=conv_input_cfg.get('kernel_size'),
            indice_key=conv_input_cfg.get('indice_key'),
            stride=conv_input_cfg.get('stride'),
            padding=conv_input_cfg.get('padding'),
            conv_type=conv_input_cfg.get('conv_type'),
            norm_fn=norm_fn)
        input_channels = conv_input_cfg.get('out_channels')

        # by default: VoxelBackBone8x in OpenPCDet.
        backbone_cfg = self.model_cfg.get(
            'BACKBONE_CFG',
            # block types in each level.
            dict(
                block_types=[
                    ['default_block'],
                    ['default_block', 'default_block', 'default_block'],
                    ['default_block', 'default_block', 'default_block'],
                    ['default_block', 'default_block', 'default_block'],
                    ['default_block']
                ],
                # output channels of each level.
                out_channels=[
                    16, 32, 64, 64, 128
                ],
                # conv type of 1st layer in each level.
                conv_type=[
                    'subm', 'spconv', 'spconv', 'spconv', 'spconv'
                ],
                # ksize of 1st layer in each level.
                kernel_size=[
                    3, 3, 3, 3, [3, 1, 1]
                ],
                # stride of 1st layer in each level.
                stride=[
                    1, 2, 2, 2, [2, 1, 1]
                ],
                # padding of 1st layer in each level.
                padding=[
                    1, 1, 1, [0, 1, 1], 0
                ]
            )
        )
        block_types = backbone_cfg['block_types']
        out_channels = backbone_cfg['out_channels']
        conv_type = backbone_cfg['conv_type']
        kernel_size = backbone_cfg['kernel_size']
        stride = backbone_cfg['stride']
        padding = backbone_cfg['padding']
        self.block_num = len(block_types)
        self.level_strides = []

        for i in range(self.block_num):
            blocks = []
            cur_block_types = block_types[i]
            cur_block_num = len(cur_block_types)
            block0 = block_dict[cur_block_types[0]](
                input_channels, out_channels[i], kernel_size[i],
                indice_key='block%d_%d' % (i, 0),
                stride=stride[i],
                padding=padding[i],
                conv_type=conv_type[i],
                norm_fn=norm_fn)
            blocks.append(block0)
            input_channels = out_channels[i]

            # update level stride.
            cur_level_stride = stride[i] if not isinstance(stride[i], list) else stride[i][-1]
            if len(self.level_strides) == 0:
                self.level_strides.append(cur_level_stride)
            else:
                self.level_strides.append(self.level_strides[-1] * cur_level_stride)

            for j in range(1, cur_block_num):
                blocks.append(
                    block_dict[cur_block_types[j]](
                        input_channels, input_channels, 3,
                        indice_key='block%d' % i,
                        stride=1, padding=1, conv_type='subm', norm_fn=norm_fn
                    )
                )

            setattr(self, 'conv%d'%(i+1), spconv.SparseSequential(*blocks))

        self.num_point_features = input_channels
        # target output key. By default, the output from the last spconv block is the target output.
        self.encoded_spconv_tensor_key = self.model_cfg.get(
            'ENCODED_SPCONV_TENSOR_KEY', 'conv%d'%self.block_num)

    def _forward_input_dict(self, input_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                voxel_features: [-1, c]
                voxel_coords: [-1, 4], (batch_idx, z_idx, y_idx, x_idx)
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = input_dict['voxel_features'], input_dict['voxel_coords']
        batch_size = input_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=input_dict.get('sparse_shape', self.sparse_shape),
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_out = []
        encoded_spconv_tensor, encoded_spconv_stride = None, None
        for i in range(self.block_num):
            cur_block_key = 'conv%d'%(i+1)
            x = getattr(self, cur_block_key)(x)
            if cur_block_key == self.encoded_spconv_tensor_key:
                encoded_spconv_tensor = x
                encoded_spconv_stride = self.level_strides[i]
            x_out.append(x)
        ret = dict(
            x_out=x_out,
            encoded_spconv_tensor=encoded_spconv_tensor,
            encoded_spconv_stride=encoded_spconv_stride,
        )
        ret.update(input_dict)
        return ret


class DVSegVoxelBackbone(VoxelBackboneBase):
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, adapt_model_cfg=kwargs)

    def _parse_input_dict(self, data_dict):

        input_dict = dict(
            batch_size=data_dict['batch_size'],
            voxel_features=data_dict['voxel_features'],
            voxel_coords=data_dict['voxel_coords'],
            sparse_shape=data_dict['sparse_shape']
        )
        return input_dict

    def _parse_output_dict(self, output_dict):
        x_out = output_dict['x_out']
        new_data_dict = dict(
            encoded_spconv_tensor=output_dict['encoded_spconv_tensor'],
            encoded_spconv_tensor_stride=output_dict['encoded_spconv_stride'],
            multi_scale_3d_features=dict(),
            multi_scale_3d_strides=dict(),
        )
        for i, x in enumerate(x_out):
            new_data_dict['multi_scale_3d_features']['x_conv%d' % (i + 1)] = x
            new_data_dict['multi_scale_3d_strides']['x_conv%d' % (i + 1)] = self.level_strides[i]
        return new_data_dict


# Define adapt function to different codebase.
class PCDetVoxelBackbone(VoxelBackboneBase):
    """ Adapter for OpenPCDet codebase. """
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg,
                         adapt_model_cfg=kwargs)

    def _parse_input_dict(self, data_dict):
        input_dict = dict(
            batch_size=data_dict['batch_size'],
            voxel_features=data_dict['voxel_features'],
            voxel_coords=data_dict['voxel_coords']
        )
        return input_dict

    def _parse_output_dict(self, output_dict):
        x_out = output_dict['x_out']
        new_data_dict = dict(
            encoded_spconv_tensor=output_dict['encoded_spconv_tensor'],
            encoded_spconv_tensor_stride=output_dict['encoded_spconv_stride'],
            multi_scale_3d_features=dict(),
            multi_scale_3d_strides=dict(),
        )
        for i, x in enumerate(x_out):
            new_data_dict['multi_scale_3d_features']['x_conv%d' % (i+1)] = x
            new_data_dict['multi_scale_3d_strides']['x_conv%d' % (i+1)] = self.level_strides[i]
        return new_data_dict


from mmdet.models import BACKBONES
from mmdet3d.ops import Voxelization
from mmcv.runner import force_fp32
@BACKBONES.register_module()
class MMDet3DVoxelBackbone(VoxelBackboneBase):
    """ Adapter for mmdetection3d codebase. """
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg,
                         adapt_model_cfg=kwargs)

        # need to define voxelization layer.
        voxel_layer_cfg = self.model_cfg.VOXEL_LAYER_CFG
        self.voxel_layer = Voxelization(**voxel_layer_cfg)

        voxel_encoder_cfg = self.model_cfg.VOXEL_ENCODER_CFG
        self.num_voxel_encoder_features = voxel_encoder_cfg.get('NUM_FEATURES', 4)

    @torch.no_grad()
    @force_fp32()
    def _voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []

        for idx, res in enumerate(points):
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)

        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    @force_fp32(out_fp16=True)
    def _voxel_encoder(self, features, num_points, coors):
        """Encoder voxel features according to interior points."""
        points_mean = features[:, :, :self.num_voxel_encoder_features].sum(
            dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
        return points_mean.contiguous()

    def _split_point_feats(self, points):
        """Split coordinates and features of input points.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
        """
        xyz = points[..., 0:3].contiguous()
        if points.size(-1) > 3:
            features = points[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None

        return xyz, features

    def _parse_input_dict(self, points):
        """ Do voxelization layer and obtain voxel_features and voxel_coords.

        :param :
            points: (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).
        :return:
            batch_size: scalar.
            voxel_features: [-1, c]
            voxel_coords: [-1, 4], (batch_idx, z_idx, y_idx, x_idx)
        """
        # b, n, 3 / b, c, n
        xyz, feat_transposed = self._split_point_feats(points)

        xyz_min, _ = xyz.min(1)  # b, 3
        min_normed_xyz = xyz - xyz_min[:, None, :]

        if feat_transposed is not None:
            feat = feat_transposed.transpose(1, 2).contiguous()  # b, n, c
            min_normed_xyz_feat = torch.cat([min_normed_xyz, feat], dim=-1)
        else:
            min_normed_xyz_feat = min_normed_xyz

        #### All points' min position are [0, 0, 0].
        voxels, num_points, coors = self._voxelize(min_normed_xyz_feat)
        voxel_features = self._voxel_encoder(voxels, num_points, coors)

        # obtain target points.
        bs, npoint = min_normed_xyz.shape[:2]
        batch_idx = torch.arange(bs).view(bs, 1, 1).repeat(1, npoint, 1).contiguous()
        batch_idx = batch_idx.to(min_normed_xyz.device)
        min_normed_points = torch.cat([batch_idx, min_normed_xyz], dim=-1).view(-1, 4)

        input_dict = dict(
            batch_size=bs,
            voxel_features=voxel_features,
            voxel_coords=coors,
            points=min_normed_points,
            xyz_min=xyz_min,
        )
        return input_dict

    def _parse_output_dict(self, output_dict):
        x_out = output_dict['x_out']
        new_data_dict = dict(
            encoded_spconv_tensor=output_dict['encoded_spconv_tensor'],
            encoded_spconv_tensor_stride=output_dict['encoded_spconv_stride'],
            multi_scale_3d_features=dict(),
            multi_scale_3d_strides=dict(),

            points=output_dict['points'],
            xyz_min=output_dict['xyz_min'],
            batch_size=output_dict['batch_size'],
        )
        for i, x in enumerate(x_out):
            new_data_dict['multi_scale_3d_features']['x_conv%d' % (i+1)] = x
            new_data_dict['multi_scale_3d_strides']['x_conv%d' % (i+1)] = self.level_strides[i]
        return new_data_dict
