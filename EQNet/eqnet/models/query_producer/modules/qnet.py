from functools import partial
from easydict import EasyDict

import copy
import torch
from torch import nn

from eqnet.models import support_processor
from eqnet.utils.config import cfg, cfg_from_yaml_file, merge_new_config
from eqnet.utils import attention_helper
from eqnet.positional_encoding import crpe, rpe
from eqnet import transformer


class QNet(nn.Module):
    def __init__(self, cfg_file, qnet_handcrafted_cfg=None):
        super().__init__()
        cfg_from_yaml_file(cfg_file, cfg)
        qnet_handcrafted_cfg = EasyDict() if qnet_handcrafted_cfg is None else qnet_handcrafted_cfg
        model_cfg = merge_new_config(cfg, qnet_handcrafted_cfg)
        self.model_cfg = model_cfg

        # 1. Support feature preprocessing module.
        self.support_feature_processor_cfg = self.model_cfg.get('SUPPORT_FEATURE_PROCESSOR')
        self.support_feature_processor = support_processor.__all__[self.support_feature_processor_cfg.NAME](
            self.support_feature_processor_cfg)

        # 2. Define Q-Net.
        # hierarchical feature num.
        self.num_levels = len(self.support_feature_processor.target_chn)

        # q-encoder & q-decoder num in each level.
        self.num_q_layers = self.model_cfg.get('NUM_Q_LAYERS')
        self.q_decoder_layer_type = self.model_cfg.get('Q_DECODER_LAYER_TYPE', 'TransformerDecoderLayer')
        self.q_encoder_layer_type = self.model_cfg.get('Q_ENCODER_LAYER_TYPE', 'TransformerEncoderLayer')
        # head_num, dropout_rate, dim_feedforward and neighbor num of local-wise attention in each level.
        self.q_head_per_level = self._get_per_level_param('Q_HEAD_PER_LEVEL')
        self.q_dim_feedforward_per_level = self._get_per_level_param('Q_DIM_FEEDFORWARD_PER_LEVEL')
        self.q_dropout_per_level = self._get_per_level_param('Q_DROPOUT_PER_LEVEL')
        self.q_local_size_per_level = self._get_per_level_param('Q_LOCAL_SIZE_PER_LEVEL')
        # attention & rpe version.
        self.q_version = self.model_cfg.get('Q_VERSION', 'v2')
        # rpe setting.
        self.rpe_type = self.model_cfg.get('RPE_TYPE', 'CRPE')
        self.rpe_cfg = self.model_cfg.get(f'{self.rpe_type}_CONFIG')
        self.rpe_point_cloud_range = self.rpe_cfg.get('POINT_CLOUD_RANGE')
        self.rpe_quan_size = self.rpe_cfg.get('QUANTIZE_SIZE')

        # auxiliary loss setting if needed.
        self.aux_loss_channel = self.model_cfg.get('AUX_LOSS_CHANNEL', None)
        self.aux_loss_supp_type = self.model_cfg.get('AUX_LOSS_SUPP_TYPE')
        aux_loss_converter = {
            # d: A list of tensor with shape [b, n, c]
            'layer-wise-supp': lambda d: torch.stack(d, dim=0),  # h, b, n, aux_channels
            # num_layers, b, n, aux_channels * num_levels
            'level-wise-supp': lambda d: torch.stack(d, dim=0).view(
                self.num_levels, self.num_q_layers, *d[0].size()).permute(1, 2, 3, 0, 4).contiguous().flatten(-2, -1)
        }
        self.aux_loss_supp_converter = aux_loss_converter[self.aux_loss_supp_type]
        self.aux_loss_mappers = nn.ModuleList()
        # target output channel for query features.
        self.q_target_chn = self.model_cfg.get('Q_TARGET_CHANNEL')

        # define rpe_layer
        if self.rpe_type == 'RPE':
            self.rpe_layer = rpe.RPE(
                self.q_head_per_level[0],
                point_cloud_range=self.rpe_point_cloud_range,
                quan_size=self.rpe_quan_size
            )

        # define decoder and encoder function,
        self.q_target_chn_per_level = self.support_feature_processor.target_chn
        # q-encoder and related modules.
        self.q_encoder = nn.ModuleList()
        # q-decoder and related modules.
        self.q_decoder = nn.ModuleList()
        for i in range(self.num_levels):
            crpe_q_func, crpe_k_func, crpe_v_func = None, None, None
            if self.rpe_type == 'CRPE':
                crpe_q_func = partial(
                    crpe.ContextualRPEIdxQ,
                    version=self.q_version,
                    point_cloud_range=self.rpe_point_cloud_range,
                    nhead=self.q_head_per_level[i],
                    channels=self.q_target_chn_per_level[i],
                    quan_size=self.rpe_quan_size
                )
                crpe_k_func = partial(
                    crpe.ContextualRPEIdxK,
                    version=self.q_version,
                    point_cloud_range=self.rpe_point_cloud_range,
                    nhead=self.q_head_per_level[i],
                    channels=self.q_target_chn_per_level[i],
                    quan_size=self.rpe_quan_size
                )
                crpe_v_func = partial(
                    crpe.ContextualRPEIdxV,
                    version=self.q_version,
                    point_cloud_range=self.rpe_point_cloud_range,
                    nhead=self.q_head_per_level[i],
                    channels=self.q_target_chn_per_level[i],
                    quan_size=self.rpe_quan_size
                )

            decoder_layers = nn.ModuleList([
                transformer.__all__[self.q_decoder_layer_type](
                    self.q_version, self.q_target_chn_per_level[i], self.q_head_per_level[i],
                    self.q_dim_feedforward_per_level[i], self.q_dropout_per_level[i],
                    ctx_rpe_query=crpe_q_func,
                    ctx_rpe_key=crpe_k_func,
                    ctx_rpe_value=crpe_v_func)
                for layer_idx in range(self.num_q_layers)
            ])
            self.q_decoder.append(decoder_layers)

            encoder_layers = nn.ModuleList([
                transformer.__all__[self.q_encoder_layer_type](
                    self.q_version, self.q_target_chn_per_level[i], self.q_head_per_level[i],
                    self.q_dim_feedforward_per_level[i], self.q_dropout_per_level[i],
                    ctx_rpe_query=crpe_q_func,
                    ctx_rpe_key=crpe_k_func,
                    ctx_rpe_value=crpe_v_func)
                for layer_idx in range(self.num_q_layers - 1)
            ])
            self.q_encoder.append(encoder_layers)

            # build aux loss.
            if self.aux_loss_channel is None or self.aux_loss_channel == 'None':
                aux_loss_mapper = nn.Identity()
            else:
                aux_loss_mapper = nn.Sequential(
                    nn.Linear(self.q_target_chn_per_level[i], self.aux_loss_channel),
                    nn.LayerNorm(self.aux_loss_channel))
            aux_loss_mappers = nn.ModuleList([copy.deepcopy(aux_loss_mapper)
                                              for layer_idx in range(self.num_q_layers)])
            self.aux_loss_mappers.append(aux_loss_mappers)

        num_features_before_fusion = sum(self.q_target_chn_per_level)
        merging_mlp = self.model_cfg.get('MERGING_MLP', [])
        merging_mlp = [num_features_before_fusion] + merging_mlp + [self.q_target_chn]
        merging_mlp_layers = []
        for k in range(len(merging_mlp) - 1):
            merging_mlp_layers.extend([
                nn.Linear(merging_mlp[k], merging_mlp[k + 1], bias=False),
                nn.BatchNorm1d(merging_mlp[k + 1]),
                nn.ReLU()
            ])
        self.merging_mlp = nn.Sequential(*merging_mlp_layers)

    def _get_per_level_param(self, key):
        param = self.model_cfg.get(key)
        if not isinstance(param, list):
            param = [param] * self.num_levels
        assert len(param) == self.num_levels
        return param

    def _compute_relative_pos(self, pos1, pos2):
        # pos1: A float tensor with shape [n, 3]
        # pos2: A float tensor with shape [n, m, 3]
        relative_pos = pos2[:, :, :] - pos1[:, None, :]  # n, m, 3
        return relative_pos

    def qnet(self, data_dict, query_features, query_pos,
             support_features, support_pos, support_mask, level_idx):
        """
        :param data_dict:
        :param query_features: A float tensor with shape [bs, query_num, c]
        :param query_pos: A float tensor with shape [bs, query_num, 3]

        :param support_features: A float tensor with shape [bs, key_num, c]
        :param support_pos: A float tensor with shape [bs, key_num, 3]
        :param support_mask: A bool tensor with shape [bs, key_num], 0 valid / 1 padding.
        :param level_idx
        :return:
        """
        batch_size = data_dict['batch_size']

        # generate index pair for cross-attention layer.
        # cross-attention: query & support.
        (query_pos, query_features, query_batch_cnt, support_pos, support_features, support_batch_cnt,
         ca_index_pair, ca_index_pair_batch, support_key_pos) = attention_helper.ca_attention_mapper(
            query_pos, query_features, support_pos, support_features, support_mask,
            self.q_local_size_per_level[level_idx])
        ca_relpos = self._compute_relative_pos(query_pos, support_key_pos)

        # self-attention: support & support.
        sa_index_pair, sa_index_pair_batch, sa_key_pos = attention_helper.sa_attention_mapper(
            support_pos, support_batch_cnt, self.q_local_size_per_level[level_idx])
        sa_relpos = self._compute_relative_pos(support_pos, sa_key_pos)

        # rpe.
        ca_rpe_weights, sa_rpe_weights = None, None
        if self.rpe_type == 'RPE':
            ca_rpe_weights = self.rpe_layer(ca_relpos)
            sa_rpe_weights = self.rpe_layer(sa_relpos)

        # 3. Do transformer.
        aux_ret = []
        num_q_layers = self.num_q_layers
        for i in range(num_q_layers):
            # do q-decoder layer
            query_features = self.q_decoder[level_idx][i](
                query_features, support_features,
                index_pair=ca_index_pair,
                query_batch_cnt=query_batch_cnt,
                key_batch_cnt=support_batch_cnt,
                index_pair_batch=ca_index_pair_batch,
                relative_atten_weights=ca_rpe_weights,
                rpe_distance=ca_relpos,)

            # do q-encoder layer: self-attention on support features.
            # Only do these self-attention on the not last layer.
            if i != (num_q_layers - 1):
                support_features = self.q_encoder[level_idx][i](
                    support_features,
                    index_pair=sa_index_pair,
                    query_batch_cnt=support_batch_cnt,
                    key_batch_cnt=support_batch_cnt,
                    index_pair_batch=sa_index_pair_batch,
                    relative_atten_weights=sa_rpe_weights,
                    rpe_distance=sa_relpos,)

            if self.training:
                aux_ret.append(self.aux_loss_mappers[level_idx][i](
                    query_features.view(batch_size, -1, self.q_target_chn_per_level[level_idx])))

        # 4. Revert window representation back.
        return query_features.view(batch_size, -1, self.q_target_chn_per_level[level_idx]), aux_ret

    def forward(self, data_dict):
        batch_size = data_dict['batch_size']

        # 1. preprocess support features & get support features / points / mask.
        support_dict = self.support_feature_processor(data_dict)
        support_features = support_dict['support_features']
        support_points = support_dict['support_points']
        support_mask = support_dict['support_mask']

        query_positions = data_dict['query_positions']  # batch_size, query_num, 3

        # 2. Hierarchical inference.
        query_feature_list = []
        aux_feature_list = []
        for i in range(self.num_levels):
            # 3. generate target path.
            cur_support_features = support_features[i]
            cur_support_points = support_points[i]
            cur_support_mask = support_mask[i]

            query_features = cur_support_features.new_zeros(
                (batch_size, query_positions.shape[1], self.q_target_chn_per_level[i]))

            # query_features: b, num_query, c
            # aux_query_features: a list of [b, num_query, aux_c]
            query_features, aux_query_features = self.qnet(
                data_dict, query_features, query_positions,
                cur_support_features, cur_support_points, cur_support_mask, i)

            query_feature_list.append(query_features)
            aux_feature_list.extend(aux_query_features)

        # --- c. merge different features.
        # Aggregate result from different path.
        query_features = torch.cat(query_feature_list, dim=-1)  # b, n, c1 + c2 + ...
        data_dict['query_features_before_fusion'] = query_features
        if self.training:
            aux_query_features = self.aux_loss_supp_converter(aux_feature_list)
            data_dict['aux_query_features'] = aux_query_features

        _, q_num, c_num = query_features.size()
        query_features = self.merging_mlp(query_features.view(-1, c_num))

        data_dict['query_features'] = query_features.view(batch_size, q_num, self.q_target_chn)
        return data_dict