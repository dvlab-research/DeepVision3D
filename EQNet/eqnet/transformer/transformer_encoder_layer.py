from typing import Optional, List


from torch import nn, Tensor
from .multi_head_attention import MultiheadAttention
from .utils import _get_activation_fn


class TransformerEncoderLayer(nn.Module):

    def __init__(self, version, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu",
                 ctx_rpe_query=None, ctx_rpe_key=None, ctx_rpe_value=None):
        super().__init__()
        self.self_attn = MultiheadAttention(
            version, d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # define positional encoding in transformer encoder layer.
        # partial function for initialization.
        self.ctx_rpe_query = ctx_rpe_query() if ctx_rpe_query is not None else None
        self.ctx_rpe_key = ctx_rpe_key() if ctx_rpe_key is not None else None
        self.ctx_rpe_value = ctx_rpe_value() if ctx_rpe_value is not None else None

    def forward(self, src,
                index_pair,
                query_batch_cnt,
                key_batch_cnt,
                index_pair_batch,
                attn_mask: Optional[Tensor] = None,

                relative_atten_weights=None,
                rpe_distance=None,):
        src2 = self.self_attn(src, src, src,
                              index_pair=index_pair,
                              query_batch_cnt=query_batch_cnt,
                              key_batch_cnt=key_batch_cnt,
                              index_pair_batch=index_pair_batch,
                              attn_mask=attn_mask,

                              relative_atten_weights=relative_atten_weights,

                              ctx_rpe_query=self.ctx_rpe_query,
                              ctx_rpe_key=self.ctx_rpe_key,
                              ctx_rpe_value=self.ctx_rpe_value,
                              rpe_distance=rpe_distance,
                              )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src