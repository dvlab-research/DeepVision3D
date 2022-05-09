"""
Contextual Relative positional encoding.
"""
import numpy as np
import torch
from torch import nn
from timm.models.layers import trunc_normal_

from eqnet.ops import crpe


class ContextualRPEIdxQ(nn.Module):
    def __init__(self, version, point_cloud_range, nhead, channels, quan_size):
        """
        :param point_cloud_range: [xmin, ymin, zmin, xmax, ymax, zmax]
        :param nhead: head number in attention.
        :param channels:
        :param quan_size: for quantize relative positional difference in each axis.
        """
        super().__init__()

        self.point_cloud_range = point_cloud_range
        self.nhead = nhead
        self.channels = channels
        self.quan_size = quan_size

        assert version in ['v1', 'v2']
        self.crpe_utils = crpe.__all__[version]

        self.dir_num = 3  # encode relative coordinates between x, y and z, respectively.
        self.hdim = channels // self.nhead
        self.hdim_per_dir = self.hdim // self.dir_num
        assert self.hdim_per_dir * self.dir_num == self.hdim, 'hdim must be divisible by dir_num'

        # Quantize the relative difference by the quantize size.
        pc_range = np.array(self.point_cloud_range).reshape(2, 3)
        pc_range_min, pc_range_max = pc_range[0], pc_range[1]
        max_distance = pc_range_max - pc_range_min  # max distance in each axis.
        self.register_buffer("max_distance",
                             torch.from_numpy(max_distance).float())

        quan_max_distance = np.ceil(max_distance * 2 / self.quan_size).astype(np.int) + 1
        self.relative_bias_table_x = nn.Parameter(
            torch.zeros(quan_max_distance[0], self.nhead, self.hdim_per_dir))
        trunc_normal_(self.relative_bias_table_x, std=.02)

        self.relative_bias_table_y = nn.Parameter(
            torch.zeros(quan_max_distance[1], self.nhead, self.hdim_per_dir))
        trunc_normal_(self.relative_bias_table_y, std=.02)

        self.relative_bias_table_z = nn.Parameter(
            torch.zeros(quan_max_distance[2], self.nhead, self.hdim_per_dir))
        trunc_normal_(self.relative_bias_table_z, std=.02)

    def _norm_relpos(self, relpos):
        relpos = relpos + self.max_distance.view(1, 1, 3)
        relpos = relpos / self.quan_size
        return relpos

    def _forward_axis(self, relpos, query_batch_cnt, query_features, relative_bias_table):
        """
        :param relpos: A float tensor with shape [total_query_num, local_size]
            indicating the relative positional difference in a specific axis.
        :param query_batch_cnt: a integer tensor with shape [batch_size]
            indicating the query_num in each batch.
        :param query_features: A float tensor with shape [total_query_num, nhead, hdim_per_dir].
        :param relative_bias_table: Lookup table for specific direction.
            A float tensor with shape [max_quan_distance, nhead, hdim_per_dir]
        :return:
            relative_pos_embedding: [total_query_num, local_size, nhead]: indicating attention weight.
        """
        relative_embedding = self.crpe_utils.rpe_q_index(
            relpos, query_batch_cnt, query_features, relative_bias_table)
        return relative_embedding

    def forward(self, relpos, query_features, scaling, query_batch_cnt):
        """
        :param relpos: A float tensor with shape [total_query_num, local_size, 3]
            indicating the relative positional difference in xyz-axes.
        :param query_features: A float tensor with shape [total_query_num, nhead, hdim].
        :param query_batch_cnt: a integer tensor with shape [batch_size]
            indicating the query_num in each batch.
        :param scaling: a constant scalar
        :return:
            relative_pos_embedding: [total_query_num, local_size, nhead]: indicating attention weight.
        """
        relpos = self._norm_relpos(relpos)

        total_query_num, nhead, hdim = query_features.size()
        query_features = query_features.view(total_query_num, nhead, self.dir_num, self.hdim_per_dir)
        query_features = torch.split(query_features, 1, dim=2)

        relative_embedding = 0
        relative_embedding = (relative_embedding +
                              self._forward_axis(relpos[..., 0].contiguous(),
                                                 query_batch_cnt,
                                                 query_features[0].squeeze(2).contiguous(),
                                                 self.relative_bias_table_x))

        relative_embedding = (relative_embedding +
                              self._forward_axis(relpos[..., 1].contiguous(),
                                                 query_batch_cnt,
                                                 query_features[1].squeeze(2).contiguous(),
                                                 self.relative_bias_table_y))

        relative_embedding = (relative_embedding +
                              self._forward_axis(relpos[..., 2].contiguous(),
                                                 query_batch_cnt,
                                                 query_features[2].squeeze(2).contiguous(),
                                                 self.relative_bias_table_z))
        relative_embedding = relative_embedding * scaling
        return relative_embedding


class ContextualRPEIdxK(ContextualRPEIdxQ):

    def _forward_axis(self, relpos, key_features,
                query_batch_cnt, key_batch_cnt,
                index_pair_batch, index_pair,
                relative_bias_table):
        """
        :param relpos: A float tensor with shape [total_query_num, local_size]
            indicating the relative positional difference in a specific axis.
        :param key_features: A float tensor with shape [total_key_num, nhead, hdim_per_dir].

        :param query_batch_cnt: a integer tensor with shape [batch_size]
            indicating the query_num in each batch.
        :param key_batch_cnt: a integer tensor with shape [batch_size]
            indicating the key_num in each batch.

        :param index_pair_batch: a integer tensor with shape [total_query_num],
            indicating the batch_index of each index_pair.
        :param: index_pair: a integer tensor with shape [total_query_num, local_size],
            the index pair for computing attention.

        :param relative_bias_table: Lookup table for specific direction.
            A float tensor with shape [max_quan_distance, nhead, hdim_per_dir]
        :return:
            relative_pos_embedding: [total_query_num, local_size, nhead]: indicating attention weight.
        """
        relative_embedding = self.crpe_utils.rpe_k_index(
            relpos, key_features, query_batch_cnt, key_batch_cnt, index_pair_batch, index_pair, relative_bias_table)
        return relative_embedding

    def forward(self, relpos, key_features, scaling,
                query_batch_cnt, key_batch_cnt,
                index_pair_batch, index_pair):
        """
        :param relpos: A float tensor with shape [total_query_num, local_size, 3]
            indicating the relative positional difference in xyz-axes.
        :param key_features: A float tensor with shape [total_key_num, nhead, hdim].
        :param scaling: A constant scalar.

        :param query_batch_cnt: a integer tensor with shape [batch_size]
            indicating the query_num in each batch.
        :param key_batch_cnt: a integer tensor with shape [batch_size]
            indicating the key_num in each batch.

        :param index_pair_batch: a integer tensor with shape [total_query_num],
            indicating the batch_index of each index_pair.
        :param: index_pair: a integer tensor with shape [total_query_num, local_size],
            the index pair for computing attention.
        :return:
            relative_pos_embedding: [total_query_num, local_size, nhead]: indicating attention weight.
        """
        relpos = self._norm_relpos(relpos)

        total_key_num, nhead, hdim = key_features.size()
        key_features = key_features.view(total_key_num, nhead, self.dir_num, self.hdim_per_dir)
        key_features = torch.split(key_features, 1, dim=2)

        relative_embedding = 0
        relative_embedding = (relative_embedding +
                              self._forward_axis(relpos[..., 0].contiguous(),
                                                 key_features[0].squeeze(2).contiguous(),
                                                 query_batch_cnt, key_batch_cnt,
                                                 index_pair_batch, index_pair,
                                                 self.relative_bias_table_x))

        relative_embedding = (relative_embedding +
                              self._forward_axis(relpos[..., 1].contiguous(),
                                                 key_features[1].squeeze(2).contiguous(),
                                                 query_batch_cnt, key_batch_cnt,
                                                 index_pair_batch, index_pair,
                                                 self.relative_bias_table_y))

        relative_embedding = (relative_embedding +
                              self._forward_axis(relpos[..., 2].contiguous(),
                                                 key_features[2].squeeze(2).contiguous(),
                                                 query_batch_cnt, key_batch_cnt,
                                                 index_pair_batch, index_pair,
                                                 self.relative_bias_table_z))
        relative_embedding = relative_embedding * scaling
        return relative_embedding


class ContextualRPEIdxV(ContextualRPEIdxQ):

    def _forward_axis(self, relpos, attn_weight, value_features,
                query_batch_cnt, key_batch_cnt,
                index_pair_batch, index_pair, relative_bias_table):
        """
        new_features = Attn_weight (value_vector + rpe_bias)

        :param relpos: A float tensor with shape [total_query_num, local_size]
            indicating the relative positional difference in a specific axis.
        :param attn_weight: A float tensor with shape [total_query_num, local_size, nhead]
            indicating the attention weight from attention computation.
        :param value_features: A float tensor with shape [total_key_num, nhead, hdim_per_dir].

        :param query_batch_cnt: a integer tensor with shape [batch_size]
            indicating the query_num in each batch.
        :param key_batch_cnt: a integer tensor with shape [batch_size]
            indicating the key_num in each batch.

        :param index_pair_batch: a integer tensor with shape [total_query_num],
            indicating the batch_index of each index_pair.
        :param: index_pair: a integer tensor with shape [total_query_num, local_size],
            the index pair for computing attention.

        :param relative_bias_table: Lookup table for specific direction.
            A float tensor with shape [max_quan_distance, nhead, hdim_per_dir]
        :return:
            new_features: [total_query_num, nhead, hdim_per_dir]: output_features after attention.
        """
        new_features = self.crpe_utils.rpe_v_index(
            relpos, attn_weight, value_features,
            query_batch_cnt, key_batch_cnt,
            index_pair_batch, index_pair, relative_bias_table)
        return new_features

    def forward(self, relpos, attn_weight, value_features,
                query_batch_cnt, key_batch_cnt,
                index_pair_batch, index_pair):
        """
        :param relpos: A float tensor with shape [total_query_num, local_size, 3]
            indicating the relative positional difference in xyz-axes.
        :param attn_weight: A float tensor with shape [total_query_num, local_size, nhead]
            indicating the attention weight from attention computation.
        :param value_features: A float tensor with shape [total_key_num, nhead, hdim_per_dir].

        :param query_batch_cnt: a integer tensor with shape [batch_size]
            indicating the query_num in each batch.
        :param key_batch_cnt: a integer tensor with shape [batch_size]
            indicating the key_num in each batch.

        :param index_pair_batch: a integer tensor with shape [total_query_num],
            indicating the batch_index of each index_pair.
        :param: index_pair: a integer tensor with shape [total_query_num, local_size],
            the index pair for computing attention.
        :return:
            new_features: [total_query_num, nhead, hdim_per_dir]: output_features after attention.
        """
        relpos = self._norm_relpos(relpos)

        total_key_num, nhead, hdim = value_features.size()
        total_query_num = attn_weight.shape[0]
        value_features = value_features.view(total_key_num, nhead, self.dir_num, self.hdim_per_dir)
        value_features = torch.split(value_features, 1, dim=2)

        new_features = []
        new_features.append(
            self._forward_axis(relpos[..., 0].contiguous(), attn_weight,
                               value_features[0].squeeze(2).contiguous(),
                               query_batch_cnt, key_batch_cnt,
                               index_pair_batch, index_pair, self.relative_bias_table_x)
        )
        new_features.append(
            self._forward_axis(relpos[..., 1].contiguous(), attn_weight,
                               value_features[1].squeeze(2).contiguous(),
                               query_batch_cnt, key_batch_cnt,
                               index_pair_batch, index_pair, self.relative_bias_table_y)
        )
        new_features.append(
            self._forward_axis(relpos[..., 2].contiguous(), attn_weight,
                               value_features[2].squeeze(2).contiguous(),
                               query_batch_cnt, key_batch_cnt,
                               index_pair_batch, index_pair, self.relative_bias_table_z)
        )
        new_features = torch.stack(new_features, dim=2)
        new_features = new_features.view(total_query_num, nhead, hdim)
        return new_features