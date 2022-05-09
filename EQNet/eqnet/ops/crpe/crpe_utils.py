import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from eqnet.ops.crpe import crpe_cuda


class RelativePositionalEmbeddingQIndex(Function):
    """
    Based on:
        * the generated attention pair index (total_query_num, local_size);
        * query features (total_query_num, nhead, hdim)
        * key features (total_key_num, nhead, hdim)

    Generate the attention weight matrix.
        * (total_query_num, local_size, nhead)
    """

    @staticmethod
    def forward(ctx,
                relpos: torch.Tensor,
                query_batch_cnt: torch.Tensor,
                query_features: torch.Tensor,
                lookup_table: torch.Tensor,):
        """

        :param ctx:
        :param relpos: A float tensor with shape [total_query_num, local_size]
        :param query_batch_cnt: A integer tensor with shape [bs], indicating the query amount for each batch.
        :param query_features: A float tensor with shape [total_query_num, nhead, hdim]
        :param lookup_table: A float tensor with shape [l, nhead, hdim]
        :return:
            output: A float tensor with shape [total_query_num, local_size, nhead]
        """
        assert relpos.is_contiguous()
        assert query_batch_cnt.is_contiguous()
        assert query_features.is_contiguous()
        assert lookup_table.is_contiguous()

        b = query_batch_cnt.shape[0]
        total_query_num, local_size = relpos.size()
        l, nhead, hdim = lookup_table.size()

        assert query_features.shape[0] == total_query_num

        output = torch.cuda.FloatTensor(total_query_num, local_size, nhead).zero_()

        crpe_cuda.rpe_q_wrapper(
            b, total_query_num, local_size, nhead, hdim, l,
            query_batch_cnt,
            relpos, lookup_table, query_features,
            output)
        ctx.for_backwards = (
            b, total_query_num, local_size, nhead, hdim, l,
            query_batch_cnt,
            relpos, lookup_table, query_features
        )
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: [total_query_num, local_size, nhead]
        Returns:
            grad_query_features:  [total_query_num, nhead, hdim]
            grad_lookup_table: [l, nhead, hdim]
        """
        (b, total_query_num, local_size, nhead, hdim, l,
         query_batch_cnt,
         relpos, lookup_table, query_features) = ctx.for_backwards

        grad_lookup_table = Variable(torch.cuda.FloatTensor(
            l, nhead, hdim).zero_())
        grad_query_features = Variable(torch.cuda.FloatTensor(
            total_query_num, nhead, hdim).zero_())

        grad_out_data = grad_out.data.contiguous()
        crpe_cuda.rpe_q_grad_wrapper(
            b, total_query_num, local_size, nhead, hdim, l,
            query_batch_cnt,
            relpos, lookup_table, query_features,
            grad_out_data, grad_lookup_table.data, grad_query_features.data)
        return None, None, grad_query_features, grad_lookup_table


rpe_q_index = RelativePositionalEmbeddingQIndex.apply

class RelativePositionalEmbeddingKIndex(Function):
    """
    Based on:
        * the generated attention pair index (total_query_num, local_size);
        * query features (total_query_num, nhead, hdim)
        * key features (total_key_num, nhead, hdim)

    Generate the attention weight matrix.
        * (total_query_num, local_size, nhead)
    """

    @staticmethod
    def forward(ctx,
                relpos: torch.Tensor,
                key_features: torch.Tensor,
                query_batch_cnt: torch.Tensor,
                key_batch_cnt: torch.Tensor,
                index_pair_batch: torch.Tensor,
                index_pair: torch.Tensor,
                lookup_table: torch.Tensor,
                ):
        """
        :param ctx:
        :param relpos: A float tensor with shape [total_query_num, local_size]
        :param key_features: A float tensor with shape [total_key_num, nhead, hdim]

        :param query_batch_cnt: A integer tensor with shape [bs], indicating the query amount for each batch.
        :param key_batch_cnt: A integer tensor with shape [bs], indicating the key amount of each batch.

        :param index_pair_batch: A integer tensor with shape [total_query_num], indicating the batch
            index of each query.
        :param index_pair: A integer tensor with shape [total_query_num, local_size]

        :param lookup_table: A float tensor with shape [l, nhead, hdim]
        :return:
            output: A float tensor with shape [total_query_num, local_size, nhead]
        """
        assert relpos.is_contiguous()
        assert key_features.is_contiguous()

        assert query_batch_cnt.is_contiguous()
        assert key_batch_cnt.is_contiguous()

        assert index_pair_batch.is_contiguous()
        assert index_pair.is_contiguous()
        assert lookup_table.is_contiguous()

        b = query_batch_cnt.shape[0]
        total_query_num, local_size = index_pair.size()
        l = lookup_table.shape[0]
        total_key_num, nhead, hdim = key_features.size()

        output = torch.cuda.FloatTensor(total_query_num, local_size, nhead).zero_()

        crpe_cuda.rpe_k_wrapper(
            b, total_query_num, local_size, total_key_num, nhead, hdim, l,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, relpos, lookup_table, key_features,
            output)
        ctx.for_backwards = (
            b, total_query_num, local_size, total_key_num, nhead, hdim, l,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, relpos, lookup_table, key_features
        )
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: [total_query_num, local_size, nhead]
        Returns:
            grad_key_features:  [total_key_num, nhead, hdim]
            grad_lookup_table: [l, nhead, hdim]
        """
        (b, total_query_num, local_size, total_key_num, nhead, hdim, l,
         query_batch_cnt, key_batch_cnt, index_pair_batch,
         index_pair, relpos, lookup_table, key_features) = ctx.for_backwards

        grad_lookup_table = Variable(torch.cuda.FloatTensor(
            l, nhead, hdim).zero_())
        grad_key_features = Variable(torch.cuda.FloatTensor(
            total_key_num, nhead, hdim).zero_())

        grad_out_data = grad_out.data.contiguous()
        crpe_cuda.rpe_k_grad_wrapper(
            b, total_query_num, local_size, total_key_num, nhead, hdim, l,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, relpos, lookup_table, key_features,
            grad_out_data, grad_lookup_table.data, grad_key_features.data)
        return None, grad_key_features, None, None, None, None, grad_lookup_table


rpe_k_index = RelativePositionalEmbeddingKIndex.apply


class RelativePositionalEmbeddingValueIndex(Function):
    """
    Based on:
        * the generated attention pair index (total_query_num, local_size);
        * query features (total_query_num, nhead, hdim)
        * key features (total_key_num, nhead, hdim)

    Generate the attention weight matrix.
        * (total_query_num, local_size, nhead)
    """

    @staticmethod
    def forward(ctx,
                relpos: torch.Tensor,
                attn_weight: torch.Tensor,
                value_features: torch.Tensor,
                query_batch_cnt: torch.Tensor,
                key_batch_cnt: torch.Tensor,
                index_pair_batch: torch.Tensor,
                index_pair: torch.Tensor,
                lookup_table: torch.Tensor):
        """

        :param ctx:
        :param relpos: A float tensor with shape [total_query_num, local_size]
        :param attn_weight: A float tensor with shape [total_query_num, local_size, nhead]
        :param value_features: A float tensor with shape [total_key_num, nhead, hdim]

        :param query_batch_cnt: A integer tensor with shape [bs], indicating the query amount for each batch.
        :param key_batch_cnt: A integer tensor with shape [bs], indicating the key amount of each batch.

        :param index_pair_batch: A integer tensor with shape [total_query_num], indicating the batch
            index of each query.
        :param index_pair: A integer tensor with shape [total_query_num, local_size]
            We ignore those index whose value is -1.

        :param lookup_table: A float tensor with shape [l, nhead, hdim]
        :return:
            output: A float tensor with shape [total_query_num, nhead, hdim]
        """
        assert relpos.is_contiguous()
        assert attn_weight.is_contiguous()
        assert value_features.is_contiguous()

        assert query_batch_cnt.is_contiguous()
        assert key_batch_cnt.is_contiguous()

        assert index_pair_batch.is_contiguous()
        assert index_pair.is_contiguous()
        assert lookup_table.is_contiguous()

        b = query_batch_cnt.shape[0]
        total_query_num, local_size = index_pair.size()
        total_key_num, nhead, hdim = value_features.size()
        l = lookup_table.shape[0]  # lookup_table: l, nhead, hdim.

        # Need to ensure that every tensor in query features have an output.
        assert total_query_num == attn_weight.shape[0]

        output = torch.cuda.FloatTensor(total_query_num, nhead, hdim).zero_()

        crpe_cuda.rpe_v_wrapper(
            b, total_query_num, local_size, total_key_num, nhead, hdim, l,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, attn_weight, value_features,
            relpos, lookup_table,
            output)
        ctx.for_backwards = (
            b, total_query_num, local_size, total_key_num, nhead, hdim, l,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, attn_weight, value_features,
            relpos, lookup_table
        )
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: [total_query_num, nhead, hdim]
        Returns:
            grad_attn_weight:  [total_query_num, local_size, nhead]
            grad_value_features: [total_key_num, nhead, hdim]
            grad_lookup_table: [l, nhead, hdim]
        """
        (b, total_query_num, local_size, total_key_num, nhead, hdim, l,
         query_batch_cnt, key_batch_cnt, index_pair_batch,
         index_pair, attn_weight, value_features,
         relpos, lookup_table) = ctx.for_backwards

        grad_attn_weight = Variable(torch.cuda.FloatTensor(
            total_query_num, local_size, nhead).zero_())
        grad_value_features = Variable(torch.cuda.FloatTensor(
            total_key_num, nhead, hdim).zero_())
        grad_lookup_table = Variable(torch.cuda.FloatTensor(
            l, nhead, hdim).zero_())

        grad_out_data = grad_out.data.contiguous()
        crpe_cuda.rpe_v_grad_wrapper(
            b, total_query_num, local_size, total_key_num, nhead, hdim, l,
            query_batch_cnt, key_batch_cnt, index_pair_batch,
            index_pair, attn_weight, value_features,
            relpos, lookup_table,
            grad_out_data, grad_attn_weight.data, grad_value_features.data,
            grad_lookup_table.data)
        return None, grad_attn_weight, grad_value_features, None, None, None, None, grad_lookup_table


rpe_v_index = RelativePositionalEmbeddingValueIndex.apply