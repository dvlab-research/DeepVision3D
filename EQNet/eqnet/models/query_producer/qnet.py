"""
An adaptor for using Q-Net including:
* 1. assign query position.
* 2. extract query features.
* 3. post-process query features.
"""
import copy
import torch
from torch import nn
import numpy as np
import spconv

from ..base import BasicEQModules
from .modules.qnet import QNet
from eqnet.utils import query_helper
from eqnet.utils import utils as eq_utils

from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from pcdet.utils import common_utils


class QNetNeckBase(BasicEQModules):
    """
    The base class of qnet.
    """
    def __init__(self, model_cfg, adapt_model_cfg):
        """

        :param model_cfg: Default model config for building Q-Net.
        :param adapt_model_cfg: Adaptive model configuration.
            One can specify the Q-Net structure in adapt_model_cfg
        """
        super().__init__(model_cfg=model_cfg, adapt_model_cfg=adapt_model_cfg)
        self.point_cloud_range = self.model_cfg.point_cloud_range
        self.voxel_size = self.model_cfg.voxel_size
        self.coord_inverse = self.model_cfg.get('coord_inverse', True)

        # support features & position pre-processing.
        self.support_cfg = self.model_cfg.get('SUPPORT_CFG')
        preprocessing_func = self.support_cfg.get('PREPROCESSING_FUNC')
        assert preprocessing_func in ['_preprocess_voxel_support_features',
                                      '_preprocess_point_support_features']
        self.support_preprocessing_func = getattr(self, preprocessing_func)
        self.support_feature_keys = self.support_cfg.get('SUPPORT_KEYS')

        # query position selection.
        self.q_pos_cfg = self.model_cfg.get('QUERY_POSITION_CFG')
        selection_functions = self.q_pos_cfg.get('SELECTION_FUNCTION')
        # assert assigned selection_func to be a list.
        if not isinstance(selection_functions, list):
            selection_functions = [selection_functions]
        self.selection_functions = []
        for selection_function in selection_functions:
            assert selection_function in ['_get_bev_query_position',
                                          '_get_point_query_position',
                                          '_get_keypoints_query_position',
                                          '_get_proposal_grids_query_position']
            self.selection_functions.append(
                getattr(self, selection_function))

        # build qnet.
        qnet_cfg_file = self.model_cfg.get('QNET_CFG_FILE')
        # some hand-crafted new features for overwriting default configurations.
        qnet_handcrafted_cfg = self.model_cfg.get('QNET_CFG', None)
        self.qnet = QNet(qnet_cfg_file, qnet_handcrafted_cfg)
        self.num_point_features = self.qnet.q_target_chn

        # query feature post-processing
        self.q_feature_cfg = self.model_cfg.get('QUERY_FEATURE_CFG')
        postprocessing_functions = self.q_feature_cfg.get('POSTPROCESSING_FUNC')
        # assert assigned postprocessing_func to be a list.
        if not isinstance(postprocessing_functions, list):
            postprocessing_functions = [postprocessing_functions]
        self.postprocessing_functions = []
        for postprocessing_function in postprocessing_functions:
            assert postprocessing_function in ['_process_bev_query_features',
                                               '_process_point_query_features',
                                               '_process_keypoints_query_features',
                                               '_process_proposal_grids_query_features']
            self.postprocessing_functions.append(
                getattr(self, postprocessing_function))

    ############################### Support Preprocessing Function ##################################
    """ 1. According to support key, get output features from backbone.
        2. Generate support features and support points.
        3. map support features & points to target representations.
            support features: M1 + M2 + ..., C
            support points: M1 + M2 + ..., 4: {batch_index, x, y, z}
    """
    def _preprocess_voxel_support_features(self, data_dict):
        support_dict = dict()

        for key in self.support_feature_keys:
            cur_support_cfg = self.support_cfg.get(key, None)
            if cur_support_cfg is not None:
                cur_vs = cur_support_cfg.get('VOXEL_SIZE', None)
            else:
                cur_vs = None
            cur_stride = data_dict['multi_scale_3d_strides'][key] if cur_vs is None else 1
            cur_vs = self.voxel_size if cur_vs is None else cur_vs

            cur_sparse_features = data_dict['multi_scale_3d_features'][key]
            cur_indices = cur_sparse_features.indices  # -1, 4
            cur_features = cur_sparse_features.features.contiguous()  # -1, c

            cur_xyz = common_utils.get_voxel_centers(
                cur_indices[:, 1:4].contiguous(), cur_stride, cur_vs, self.point_cloud_range, self.coord_inverse)
            batch_indices = cur_indices[:, 0:1].contiguous()
            cur_xyz = torch.cat([batch_indices, cur_xyz], dim=-1)

            support_dict[key] = {
                'support_features': cur_features,
                'support_points': cur_xyz,
            }
        data_dict['multi_scale_support_sets'] = support_dict
        return data_dict

    def _preprocess_point_support_features(self, data_dict):
        support_dict = dict()

        for key in self.support_feature_keys:
            cur_point_features = data_dict['multi_scale_3d_features'][key]  # b, c, n
            # b, c, n ---> b, n, c
            cur_point_features = cur_point_features.permute(0, 2, 1).contiguous()

            cur_point_xyz = data_dict['multi_scale_3d_xyz'][key]  # b, n, 3
            # b, n, 3 ---> -1, 4 {batch_idx, x, y, z}
            b, n = cur_point_xyz.shape[:2]
            batch_idx = torch.arange(b).view(b, 1, 1).repeat(1, n, 1).contiguous().to(cur_point_xyz.device)
            cur_point_xyz = torch.cat([batch_idx, cur_point_xyz], dim=-1).view(-1, 4)

            support_dict[key] = {
                'support_features': cur_point_features.view(-1, cur_point_features.shape[-1]),
                'support_points': cur_point_xyz.view(-1, cur_point_xyz.shape[-1]),
            }
        data_dict['multi_scale_support_sets'] = support_dict
        return data_dict

    ############################### Query Position Selection Function ###############################
    """ Generate query positions.
        Return:
            * query_xyz: A float tensor with shape [batch_size, num_query, 3]
                used in the q-net for obtaining query features.
            * query_indices: A float tensor with shape [-1, 4]: {batch_index, x, y, z}
                for post-processing query features.
            * sample_indices (optional): A long tensor with shape [batch_size, num_query].
                For keypoints query position generation:
                    indicating the indices of sampled_key_points from raw points.  
    """
    def _get_query_position(self, data_dict):
        query_position, query_indices, sampled_indices = [], [], []
        for selection_func in self.selection_functions:
            cur_query_ret = selection_func(data_dict)
            cur_query_position, cur_query_indices, cur_sampled_indices = cur_query_ret

            query_position.append(cur_query_position)
            query_indices.append(cur_query_indices)
            sampled_indices.append(cur_sampled_indices)

        return query_position, query_indices, sampled_indices

    def _get_bev_query_position(self, data_dict):
        """ Obtain bev-based query positions for SSD head. """
        t = data_dict['points']
        batch_size = data_dict['batch_size']

        # get configuration.
        point_cloud_range = np.array(self.q_pos_cfg.get('POINT_CLOUD_RANGE'))
        voxel_size = np.array(self.q_pos_cfg.get('VOXEL_SIZE'))

        # H * W, 3
        bev_xyz = query_helper.obtain_bev_query_position(t, point_cloud_range, voxel_size)
        bev_xyz = bev_xyz.unsqueeze(0).repeat(batch_size, 1, 1).contiguous()

        bev_xyz, bev_indices = query_helper.obtain_bev_query_indices(
            bev_xyz, voxel_size, point_cloud_range)
        return bev_xyz, bev_indices, None

    def _get_point_query_position(self, data_dict):
        """ Obtain point query positions for point-based heads. """
        batch_size = data_dict.get('batch_size')

        point_indices = data_dict['points'][:, :4].contiguous()  # -1, 4: {batch_index, x, y, z}
        point_xyz = point_indices.view(batch_size, -1, 4)[..., 1:4].contiguous()

        num_query = point_xyz.shape[1]
        sampled_indices = torch.arange(num_query).view(1, num_query).repeat(batch_size, 1).contiguous().long()
        sampled_indices = sampled_indices.to(point_xyz.device)
        return point_xyz, point_indices, sampled_indices

    def _get_keypoints_query_position(self, data_dict):
        batch_size = data_dict['batch_size']

        # get configuration.
        keypoints_src = self.q_pos_cfg.get('KEYPOINTS_SRC')
        keypoints_sample_method = self.q_pos_cfg.get('KEYPOINTS_SAMPLE_METHOD')
        keypoints_num = self.q_pos_cfg.get('KEYPOINTS_NUM')

        if keypoints_src == 'raw_points':
            src_points = data_dict['points'][:, 1:4]
            batch_indices = data_dict['points'][:, 0].long()
        elif keypoints_src == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                data_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = data_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError

        keypoints_list, sampled_indices_list = [], []
        keypoints_batchidx = src_points.new_zeros((
            batch_size, keypoints_num, 1))

        for bs_idx in range(batch_size):
            keypoints_batchidx[bs_idx] = bs_idx
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            if keypoints_sample_method == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), keypoints_num
                ).long()

                if sampled_points.shape[1] < keypoints_num:
                    times = int(keypoints_num / sampled_points.shape[1]) + 1
                    non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                    cur_pt_idxs[0] = non_empty.repeat(times)[:keypoints_num]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
            elif keypoints_sample_method == 'RS':
                cur_pt_idxs = np.random.choice(sampled_points.shape[1],
                                               min(sampled_points.shape[1], keypoints_num),
                                               replace=False)
                if sampled_points.shape[1] < keypoints_num:
                    times = int(keypoints_num / sampled_points.shape[1]) + 1
                    cur_pt_idxs = cur_pt_idxs.repeat(times)[:keypoints_num]
                cur_pt_idxs = torch.from_numpy(cur_pt_idxs).cuda().long().unsqueeze(0)
                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
            else:
                raise NotImplementedError

            sampled_indices_list.append(cur_pt_idxs)
            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        keypoints_indices = torch.cat([keypoints_batchidx, keypoints], dim=-1).view(-1, 4)
        sampled_indices = torch.cat(sampled_indices_list, dim=0)
        return keypoints, keypoints_indices, sampled_indices

    def _get_proposal_grids_query_position(self, data_dict):
        """ Obtain point query positions for VoxelRCNN head.
        Use proposal grid as query positions. """
        batch_size = data_dict.get('batch_size')
        rois = data_dict['rois']  # batch_size, proposal_num, 7

        # get configuration.
        grid_size = self.q_pos_cfg.get('PROPOSAL_GRID_SIZE')

        proposal_grid_points = query_helper.obtain_proposal_grid_query_position(
            rois, grid_size)  # -1, grid_size ** 3, 3
        proposal_grid_points = proposal_grid_points.view(batch_size, -1, 3)

        # generate batch_indices.
        proposal_grid_batch_idx = torch.arange(batch_size).view(batch_size, 1, 1).repeat(
            1, proposal_grid_points.shape[1], 1).contiguous().float().to(proposal_grid_points.device)
        proposal_grid_indices = torch.cat([
            proposal_grid_batch_idx, proposal_grid_points], dim=-1).view(-1, 4)
        return proposal_grid_points, proposal_grid_indices, None

    ############################### Query Feature PostProcessing Function ###########################
    """ Post-process query features.
        Map query features to the required key in the following modules.
        Input:
            query_features: batch_size, Q1 + Q2 + ..., channel_num
            query_indices: 
                A list of float tensors. Each float tensor has a shape of
                    batch_size * Qi (-1), 4 (batch_index, x, y, z)
            data_dict:
                query_features_before_fusion (optional): A float tensor with a shape as
                    batch_size, Q1 + Q2 + ..., C1 + C2 + ... 
        E.g.: 
            "spatial_features" for SSD head.
            "point_features" & "point_coords" for PointRCNN head / PVRCNN head.
    """
    def _process_features(self, query_features, query_indices, data_dict):
        b = query_features.shape[0]
        start_idx = 0

        query_features_before_fusion = data_dict.get('query_features_before_fusion', None)
        for i, postprocessing_func in enumerate(self.postprocessing_functions):
            cur_query_indices = query_indices[i]
            # -1, 4 ---> b, npoint, 4
            cur_query_indices = cur_query_indices.view(b, -1, 4)
            cur_query_num = cur_query_indices.shape[1]

            # b, npoint, c
            end_idx = start_idx + cur_query_num
            cur_query_features = query_features[:, start_idx:end_idx, :].contiguous()
            if query_features_before_fusion is not None:
                cur_query_features_before_fusion = query_features_before_fusion[:, start_idx:end_idx, :].contiguous()
            else:
                cur_query_features_before_fusion = None
            start_idx = end_idx

            data_dict = postprocessing_func(
                cur_query_features, cur_query_features_before_fusion, query_indices[i], data_dict)
        return data_dict

    def _process_bev_query_features(
            self, bev_features, bev_features_before_fusion, bev_indices, data_dict):
        """

        :param bev_features: b, n, c
        :param bev_indices: -1, 4
        :param data_dict:
        :return:
        """
        batch_size = data_dict['batch_size']

        # get configurations.
        point_cloud_range = np.array(self.q_pos_cfg.get('POINT_CLOUD_RANGE'))
        voxel_size = np.array(self.q_pos_cfg.get('VOXEL_SIZE'))
        spatial_shape = eq_utils.obtain_spatial_shape(voxel_size, point_cloud_range).astype(np.int).tolist()

        bev_features = bev_features.view(-1, bev_features.shape[-1])
        sp_tensor = spconv.SparseConvTensor(
            features=bev_features,
            indices=bev_indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )
        sp_features = sp_tensor.dense()
        N, C, D, H, W = sp_features.shape
        sp_features = sp_features.view(N, C * D, H, W)

        data_dict['spatial_features'] = sp_features
        return data_dict

    def _process_point_query_features(
            self, point_features, point_features_before_fusion, point_indices, data_dict):
        point_features = point_features.flatten(0, 1)  # -1, c
        data_dict['point_features'] = point_features
        data_dict['point_coords'] = point_indices
        return data_dict

    def _process_keypoints_query_features(
            self, point_features, point_features_before_fusion, point_indices, data_dict):
        point_features = point_features.flatten(0, 1)  # -1, c
        data_dict['point_features'] = point_features
        data_dict['point_coords'] = point_indices
        data_dict['point_features_before_fusion'] = point_features_before_fusion.flatten(0, 1)
        return data_dict

    def _process_proposal_grids_query_features(
            self, proposal_grid_features, proposal_grid_features_before_fusion, proposal_grid_indices, data_dict):
        batch_size = data_dict.get('batch_size')

        # get configuration.
        grid_size = self.q_pos_cfg.get('PROPOSAL_GRID_SIZE')
        proposal_grid_features = proposal_grid_features.view(
            batch_size, -1, grid_size ** 3, proposal_grid_features.shape[-1])
        data_dict['proposal_grid_features'] = proposal_grid_features.flatten(0, 1)

        data_dict['proposal_grid_coords'] = proposal_grid_indices
        return data_dict

    ############################### Main Forward Function ###########################
    def _forward_input_dict(self, input_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        # 1. preprocess support sets.
        input_dict = self.support_preprocessing_func(input_dict)

        # 2. generate query positions: [b, n, 3] / [b*n, 4]
        query_positions, query_indices, sampled_indices = self._get_query_position(input_dict)
        query_positions = torch.cat(query_positions, dim=1)  # concatenate at num_point dimension.
        input_dict['query_positions'] = query_positions
        # 2.1. ignore none in sampled_indices and concatenate as return.
        query_sampled_indices = []
        for indices in sampled_indices:
            if indices is not None:
                query_sampled_indices.append(indices)
        if len(query_sampled_indices) > 0:
            query_sampled_indices = torch.cat(query_sampled_indices, dim=1)
        input_dict['query_sampled_indices'] = query_sampled_indices

        # 3. do qnet.
        input_dict = self.qnet(input_dict)

        # 4. post-process query features.
        input_dict = self._process_features(
            input_dict['query_features'], query_indices, input_dict)
        return input_dict


# Define adapt function to different codebases.
class PCDetQNetNeck(QNetNeckBase):
    """ Adapter for OpenPCDet codebase. """
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg,
                         adapt_model_cfg=kwargs)

    def _parse_input_dict(self, data_dict):
        """ Parse input dictionary for OpenPCDet codebase.

        :param data_dict:
        :return:
            input_dict
        """
        key_mapping_pairs = {
            'multi_scale_3d_strides': 'multi_scale_3d_strides',
            'multi_scale_3d_features': 'multi_scale_3d_features',
            'multi_scale_3d_xyz': 'multi_scale_3d_xyz',
            'points': 'points',
            'batch_size': 'batch_size',
            'voxel_coords': 'voxel_coords',
            'rois': 'rois',
        }
        input_dict = dict()
        for k, v in key_mapping_pairs.items():
            if k in data_dict.keys():
                input_dict[v] = data_dict[k]
        return input_dict

    def _parse_output_dict(self, output_dict):
        """ Parse output dictionary for OpenPCDet

        :param output_dict:
        :return:
            new_data_dict
        """
        key_mapping_pairs = {
            'spatial_features': 'spatial_features',
            'point_features': 'point_features',
            'point_coords': 'point_coords',
            'point_features_before_fusion': 'point_features_before_fusion',
            'proposal_grid_features': 'proposal_grid_features',
            'proposal_grid_coords': 'proposal_grid_coords',

            'query_positions': 'query_positions',
            'query_features': 'query_features',
            'query_features_before_fusion': 'query_features_before_fusion',
            'aux_query_features': 'aux_query_features',
        }
        new_data_dict = dict()
        for k, v in key_mapping_pairs.items():
            if k in output_dict.keys():
                new_data_dict[v] = output_dict[k]
        return new_data_dict


from mmdet.models import NECKS
@NECKS.register_module()
class MMDet3DQNetNeck(QNetNeckBase):
    """ Adapter for mmdetection3d codebase. """
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg,
                         adapt_model_cfg=kwargs)

    def _parse_input_dict(self, data_dict):
        """ Parse input dictionary for OpenPCDet codebase.

        :param data_dict:
        :return:
            input_dict
        """
        key_mapping_pairs = {
            'multi_scale_3d_features': 'multi_scale_3d_features',
            'multi_scale_3d_xyz': 'multi_scale_3d_xyz',
            'batch_size': 'batch_size',
            'points': 'points',
            'multi_scale_3d_strides': 'multi_scale_3d_strides',
            'voxel_coords': 'voxel_coords',
            'xyz_min': 'xyz_min',
        }

        input_dict = dict()
        for k, v in key_mapping_pairs.items():
            if k in data_dict.keys():
                input_dict[v] = data_dict[k]
        return input_dict

    def _parse_output_dict(self, output_dict):
        """ Parse output dictionary for OpenPCDet

        :param output_dict:
        :return:
            new_data_dict
        """
        q_pos, q_features, q_indices = (
            output_dict['query_positions'], output_dict['query_features'], output_dict['query_sampled_indices'])
        q_features = q_features.permute(0, 2, 1).contiguous()  # b, n, c -> b, c, n

        if 'xyz_min' in output_dict:
            # for voxel-based backbone, which will first use min-norm.
            q_pos += output_dict['xyz_min'][:, None, :]

        new_data_dict = {
            'fp_xyz': [q_pos],  # bs, npoint, 3
            'fp_features': [q_features],  # bs, c, npoint
            'fp_indices': [q_indices],  # bs, npoint
        }
        if 'aux_query_features' in output_dict:
            q_aux_features = output_dict['aux_query_features']
            new_data_dict.update({
                'fp_interval_features': [q_aux_features]  # h, bs, npoint, c
            })
        return new_data_dict


class DVClsQNetNeck(QNetNeckBase):
    """ Adapter for DV_cls codebase. """
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg,
                         adapt_model_cfg=kwargs)

    def _parse_input_dict(self, data_dict):
        """ Parse input dictionary for OpenPCDet codebase.

        :param data_dict:
        :return:
            input_dict
        """
        key_mapping_pairs = {
            'multi_scale_3d_features': 'multi_scale_3d_features',
            'multi_scale_3d_xyz': 'multi_scale_3d_xyz',
            'points': 'points',
            'batch_size': 'batch_size',
        }
        input_dict = dict()
        for k, v in key_mapping_pairs.items():
            if k in data_dict.keys():
                input_dict[v] = data_dict[k]
        return input_dict

    def _parse_output_dict(self, output_dict):
        """ Parse output dictionary for OpenPCDet

        :param output_dict:
        :return:
            new_data_dict
        """
        query_features = output_dict['query_features']  # b, n, c

        # vote merging.
        new_data_dict = dict(
            point_features=query_features,
        )

        if 'aux_query_features' in output_dict:
            new_data_dict.update({
                'aux_query_features': output_dict['aux_query_features']  # h, bs, npoint, c
            })
        return new_data_dict



class DVSegQNetNeck(QNetNeckBase):
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg, adapt_model_cfg=kwargs)

    def _parse_input_dict(self, data_dict):
        key_mapping_pairs = {
            'multi_scale_3d_strides': 'multi_scale_3d_strides',
            'multi_scale_3d_features': 'multi_scale_3d_features',
            'points': 'points',
            'batch_size': 'batch_size',
            'voxel_coords': 'voxel_coords',
        }
        input_dict = dict()
        for k, v in key_mapping_pairs.items():
            if k in data_dict.keys():
                input_dict[v] = data_dict[k]
        return input_dict

    def _parse_output_dict(self, output_dict):
        query_features = output_dict['query_features']  # b, n, c

        new_data_dict = dict(
            query_features=query_features,
        )

        if 'aux_query_features' in output_dict:
            new_data_dict.update({
                'aux_query_features': output_dict['aux_query_features']  # h, bs, npoint, c
            })

        if 'query_sampled_indices' in output_dict:
            new_data_dict.update({
                'query_sampled_indices': output_dict['query_sampled_indices']
            })
        return new_data_dict