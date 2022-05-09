import torch

from eqnet.ops.knn.knn_utils import knn_query
from eqnet.ops.grouping.grouping_utils import grouping_operation


def convert_batch_to_stack(t, mask=None):
    """
    Cast a tensor (t) for batch representation [b, n, c] to stack representation [N1 + N2 + ..., c]

    :param t: A float tensor with shape [b, n, c]
    :param mask: (optional) A bool tensor with shape [b, n]: 0 useful; 1 ignore.
    :return:
        new_t: A float tensor with shape [N1 + N2 + ..., c]
        new_batch_cnt: A float tensor with shape [b]: [N1, N2, ...]
    """
    b, n, c = t.size()
    batch_cnt = t.new_zeros(b).int()
    if mask is not None:
        t_list = []
        for i in range(b):
            cur_t = t[i]
            cur_mask = mask[i]
            cur_t = cur_t[torch.logical_not(cur_mask)]
            t_list.append(cur_t)
            batch_cnt[i] = cur_t.shape[0]
        new_t = torch.cat(t_list, dim=0)
    else:  # mask is None
        for i in range(b):
            batch_cnt[i] = n
        new_t = t.view(b * n, c)
    return new_t, batch_cnt


def ca_attention_mapper(query_pos, query_features,
                        key_pos, key_features, key_mask,
                        local_size):
    """ Attention mapper function.

    * Find the query-key pair for computing attention
        (For each token in query, figure out its nearest k token in keys)
    * Cast feature and position to the desired representation for attention computation.

    :param query_pos: A float tensor with shape [b, num_query, 3]
    :param query_features: A float tensor with shape [b, num_query, c]

    :param key_pos: A float tensor with shape [b, num_key, 3]
    :param key_features: A float tensor with shape [b, num_key, c]
    :param key_mask: A bool tensor with shape [b, num_key]: 0 useful key; 1 ignore key.

    :param local_size: nsample for knn.

    :return:
        query_pos: A float tensor with shape [N1 + N2 + ..., 3]
        query_features: A float tensor with shape [N1 + N2 + ..., c]

        key_pos: A float tensor with shape [M1 + M2 + ..., 3]
        key_features: A float tensor with shape [M1 + M2 + ..., c]

        key_attn_pos: A float tensor with shape [N1 + N2 + ..., local_size, 3],
            indicating the position of keys for computing attention.

        index_pair: A int tensor with shape [N1 + N2 + ..., local_size], indicating the indices of keys
            involving in the attention computation of each query.
        query_cnt: A int tensor with shape [b], indicating query_num in each batch.
        key_cnt: A int tensor with shape [b], indicating key_num in each batch.
        index_pair_batch: A int tensor with shape [N1 + N2 + ...], indicating the batch_index of each query-key pair.
    """
    batch_size, query_amount_per_batch = query_pos.shape[:2]

    # 1. cast query_pos, query_features, key_pos, key_features to target representation.
    query_pos, query_cnt = convert_batch_to_stack(query_pos)
    query_features, query_cnt = convert_batch_to_stack(query_features)
    key_pos, key_cnt = convert_batch_to_stack(key_pos, key_mask)
    key_features, key_cnt = convert_batch_to_stack(key_features, key_mask)

    # 2. Get index pair for computing attention.
    index_pair = knn_query(
        local_size,
        key_pos, key_cnt,
        query_pos, query_cnt).int()  # N1 + N2 + ..., local_size
    index_pair = index_pair.view(-1, local_size)

    # 3. Get key_attn_pos.
    # N1 + N2 + ..., local_size, 3
    key_attn_pos = grouping_operation(
        key_pos, key_cnt, index_pair, query_cnt).permute(0, 2, 1).contiguous()

    # 4. get index_pair_batch.
    index_pair_batch = []
    for i in range(batch_size):
        index_pair_batch.append(
            index_pair.new_ones(query_amount_per_batch).int() * i)
    index_pair_batch = torch.cat(index_pair_batch)

    return (query_pos, query_features, query_cnt,
            key_pos, key_features, key_cnt,
            index_pair, index_pair_batch,
            key_attn_pos)


def sa_attention_mapper(query_pos, query_batch_cnt,
                        local_size):
    """ Attention mapper function.

    * Find the query-key pair for computing attention
        (For each token in query, figure out its nearest k token in keys)
    * Cast feature and position to the desired representation for attention computation.

    :param query_pos: A float tensor with shape [N1 + N2 + ..., 3]
    :param query_cnt: A int tensor with shape [b], indicating query_num in each batch.
    :param local_size: nsample for knn.

    :return:
        index_pair: A int tensor with shape [N1 + N2 + ..., local_size], indicating the indices of keys
            involving in the attention computation of each query.
        index_pair_batch: A int tensor with shape [N1 + N2 + ...], indicating the batch_index of each query-key pair.
        query_attn_pos: A float tensor with shape [N1 + N2 + ..., local_size, 3],
            indicating the position of keys for computing attention.
    """
    # Get index pair for computing self-attention.
    index_pair = knn_query(
        local_size,
        query_pos, query_batch_cnt,
        query_pos, query_batch_cnt).int()  # N1 + N2 + ..., local_size
    index_pair = index_pair.view(-1, local_size)

    query_attn_pos = grouping_operation(
        query_pos, query_batch_cnt, index_pair, query_batch_cnt).permute(0, 2, 1).contiguous()

    index_pair_batch = []
    for i in range(query_batch_cnt.shape[0]):
        index_pair_batch.append(
            index_pair.new_ones(query_batch_cnt[i]).int() * i)
    index_pair_batch = torch.cat(index_pair_batch)
    return index_pair, index_pair_batch, query_attn_pos