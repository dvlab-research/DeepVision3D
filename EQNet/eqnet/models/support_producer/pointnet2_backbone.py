import torch
import torch.nn as nn

from ..base import BasicEQModules

from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_modules


class EQPointNet2BackboneBase(BasicEQModules):
    def __init__(self, model_cfg, adapt_model_cfg):
        super().__init__(model_cfg=model_cfg,
                         adapt_model_cfg=adapt_model_cfg)
        input_channels = self.model_cfg.input_channels
        channel_in = input_channels - 3

        self.SA_modules = nn.ModuleList()
        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                    normalize_xyz=self.model_cfg.SA_CONFIG.get('NORMALIZE_XYZ', False),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.num_point_features = channel_in

        fp_mlps = self.model_cfg.get('FP_MLPS', None)
        if fp_mlps is not None:
            self.FP_modules = nn.ModuleList()
            for k in range(fp_mlps.__len__()):
                pre_channel = fp_mlps[k + 1][-1] if k + 1 < len(fp_mlps) else channel_out
                self.FP_modules.append(
                    pointnet2_modules.PointnetFPModule(
                        mlp=[pre_channel + skip_channel_list[k]] + fp_mlps[k]
                    )
                )
            self.num_point_features = fp_mlps[0][-1]
        else:
            self.FP_modules = None

    def _get_multi_scale_dict(self, l_xyz, l_features):
        multi_scale_features = dict()
        multi_scale_xyz = dict()

        for i, (xyz, features) in enumerate(zip(l_xyz, l_features)):
            multi_scale_features['l%d' % i] = features
            multi_scale_xyz['l%d' % i] = xyz
        return multi_scale_features, multi_scale_xyz

    def _forward_input_dict(self, input_dict):
        """
        Args:
            input_dict:
                batch_size: int
                points: (B, N, 3)
                features: (B, C, N)
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        xyz = input_dict['init_xyz']
        features = input_dict['init_features']

        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(batch, 1).long()
        l_xyz, l_features, l_indices = [xyz], [features], [indices]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_indices = self.SA_modules[i](l_xyz[i], l_features[i], ret_indices=True)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_indices.append(torch.gather(l_indices[-1], 1, li_indices.long()))

        ret = dict(
            sa_xyz=l_xyz,
            sa_features=l_features,
            sa_indices=l_indices,
        )

        if self.FP_modules is not None:
            fp_xyz = [l_xyz[-1]]
            fp_features = [l_features[-1]]
            fp_indices = [l_indices[-1]]

            for i in range(-1, -(len(self.FP_modules) + 1), -1):
                fp_features.append(
                    self.FP_modules[i](
                        l_xyz[i - 1], l_xyz[i], l_features[i - 1], fp_features[-1]
                    )
                )  # (B, C, N)
                fp_xyz.append(l_xyz[i - 1])
                fp_indices.append(l_indices[i - 1])

            ret.update(dict(
                fp_xyz=fp_xyz,
                fp_features=fp_features,
                fp_indices=fp_indices,
            ))
        input_dict.update(ret)
        return input_dict


# Define adapt function to different codebase.
class PCDetPointNet2Backbone(EQPointNet2BackboneBase):
    """ Adapter for OpenPCDet codebase. """
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg,
                         adapt_model_cfg=kwargs)

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def _parse_input_dict(self, data_dict):
        """

        :param data_dict:
            batch_size: int
            points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        :return:
            input_dict:
                points: (B, N, 3)
                features: (B, C, N)
        """
        batch_size = data_dict['batch_size']
        points = data_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        # 1. Process input: from [n1 + n2 + ..., 4 + C] -> xyz: (b, n, 3) + features: (b, c, n)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(
            0, 2, 1).contiguous() if features is not None else None

        input_dict = dict(
            init_xyz=xyz,
            init_features=features,
            xyz_batch_cnt=xyz_batch_cnt,
            batch_idx=batch_idx,
        )
        return input_dict

    def _parse_output_dict(self, output_dict):
        """ Parse output dict to OpenPCDet required format.

        :param output_dict:
            init_xyz:
            init_features:
            xyz_batch_cnt:

            sa_xyz: A list of float tensor with shape [b, n, 3],
            sa_features: A list of float tensor with shape [b, c, n]
            sa_indices: A list of long tensor with shape [b, n]

            (optional):
            fp_xyz: A list of float tensor with shape [b, n, 3],
            fp_features: A list of float tensor with shape [b, c, n],
            fp_indices: A list of long tensor with shape [b, n],
        :return:
        """
        new_data_dict = dict()

        # parse multi-level features from backbones.
        multi_scale_features, multi_scale_xyz = self._get_multi_scale_dict(
            output_dict['sa_xyz'][1:], output_dict['sa_features'][1:])
        new_data_dict['multi_scale_3d_features'] = multi_scale_features
        new_data_dict['multi_scale_3d_xyz'] = multi_scale_xyz

        if 'fp_xyz' in output_dict:
            point_features = output_dict['fp_features'][-1].permute(0, 2, 1).contiguous()  # (B, N, C)
            point_coords = torch.cat((
                output_dict['batch_idx'][:, None].float(), output_dict['fp_xyz'][-1].view(-1, 3)), dim=1)
            print(point_features.size(), point_coords.size(), point_coords)
            print('---- debugging in PCDetPointNet2Backbone ----')

            new_data_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
            new_data_dict['point_coords'] = point_coords
        return new_data_dict


from mmdet.models import BACKBONES
@BACKBONES.register_module()
class MMDet3DPointNet2Backbone(EQPointNet2BackboneBase):
    """ Adapter for mmdetection3d codebase. """
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg,
                         adapt_model_cfg=kwargs)

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
        """

        :param :
            points: (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).
        Returns:
            init_xyz: (B, N, 3)
            init_features: (B, C, N)
            points: (-1, 4) for keypoints sampling.
            batch_size: B
        """
        xyz, features = self._split_point_feats(points)

        # cast xyz from [bs, npoint, 3] -> [-1, 4] {batch_idx, x, y, z}
        bs, npoint = xyz.shape[:2]
        batch_idx = torch.arange(bs).view(bs, 1, 1).repeat(1, npoint, 1).contiguous()
        batch_idx = batch_idx.to(xyz.device)
        points = torch.cat([batch_idx, xyz], dim=-1).view(-1, 4)

        input_dict = dict(
            init_xyz=xyz,
            init_features=features,
            points=points,
            batch_size=bs,
        )
        return input_dict

    def _parse_output_dict(self, output_dict):
        """ Parse output dict to mmdet3d required format.

        :param output_dict:
            init_xyz:
            init_features:
            xyz_batch_cnt:

            sa_xyz: A list of float tensor with shape [b, n, 3],
            sa_features: A list of float tensor with shape [b, c, n]
            sa_indices: A list of long tensor with shape [b, n]

            (optional):
            fp_xyz: A list of float tensor with shape [b, n, 3],
            fp_features: A list of float tensor with shape [b, c, n],
            fp_indices: A list of long tensor with shape [b, n],
        :return:
        """
        new_data_dict = dict()

        multi_scale_features, multi_scale_xyz = self._get_multi_scale_dict(
            output_dict['sa_xyz'][1:], output_dict['sa_features'][1:])
        new_data_dict['multi_scale_3d_features'] = multi_scale_features
        new_data_dict['multi_scale_3d_xyz'] = multi_scale_xyz

        key_mapping_pairs = {
            'sa_xyz': 'sa_xyz',
            'sa_features': 'sa_features',
            'sa_indices': 'sa_indices',
            'fp_xyz': 'fp_xyz',
            'fp_features': 'fp_features',
            'fp_indices': 'fp_indices',
            'points': 'points',
            'batch_size': 'batch_size',
        }
        for k, v in key_mapping_pairs.items():
            if k in output_dict:
                new_data_dict[v] = output_dict[k]
        return new_data_dict


class DVClsPointNet2Backbone(EQPointNet2BackboneBase):
    """ Adapter for DV_cls codebase. """
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg,
                         adapt_model_cfg=kwargs)

    def _parse_input_dict(self, data_dict):
        """

        :param data_dict:
            xyz: (B, 3, N).
            feature: (batch_size, C, N) or none.
        :return:
            input_dict:
                init_xyz: (B, N, 3)
                init_features: (B, C, N)
                points: (-1, 4) for keypoints sampling.
                batch_size: B
        """
        xyz = data_dict['xyz']
        features = data_dict['feat']

        xyz = xyz.permute(0, 2, 1).contiguous()  # b, n, 3
        # cast xyz from [b, n, 3] -> [-1, 4] {batch_idx, x, y, z}
        bs, npoint = xyz.shape[:2]
        batch_idx = torch.arange(bs).view(bs, 1, 1).repeat(1, npoint, 1).contiguous()
        batch_idx = batch_idx.to(xyz.device)
        points = torch.cat([batch_idx, xyz], dim=-1).view(-1, 4)

        input_dict = dict(
            init_xyz=xyz,
            init_features=features,
            points=points,
            batch_size=bs,
        )
        return input_dict

    def _parse_output_dict(self, output_dict):
        """ Parse output dict to OpenPCDet required format.

        :param output_dict:
            init_xyz:
            init_features:
            xyz_batch_cnt:

            point_features: [b, c, n]
            points: [-1, 4], for keypoint sampling.
            batch_size: b, a scalar used in qnet.
        :return:
        """
        new_data_dict = dict()

        # parse multi-level features from backbones.
        multi_scale_features, multi_scale_xyz = self._get_multi_scale_dict(
            output_dict['sa_xyz'][1:], output_dict['sa_features'][1:])
        new_data_dict['multi_scale_3d_features'] = multi_scale_features
        new_data_dict['multi_scale_3d_xyz'] = multi_scale_xyz

        new_data_dict['point_features'] = output_dict['sa_features'][-1]

        key_mapping_pairs = {
            'points': 'points',
            'batch_size': 'batch_size',
        }
        for k, v in key_mapping_pairs.items():
            if k in output_dict:
                new_data_dict[v] = output_dict[k]
        return new_data_dict
