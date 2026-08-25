"""Quant ATK plugin: PAGED_BBND dim0 non-contiguous KV (FIA fused cache).

atk ... -p aclnn_genericblocksparseattention_quant_uncon.py
"""

import math

import torch
from aclnn_genericblocksparseattention_quant import (  # noqa: F401
    GenericBlockSparseAttentionQuantApi,
)
from aclnn_genericblocksparseattention_quant import (
    GenericBlockSparseAttentionQuantInputProcess as _ContigInputProcess,
)
from atk.tasks.api_execute import register


def make_fused_noncontig_kv(key: torch.Tensor, value: torch.Tensor):
    """FIA-style fused KV: k0v0k1v1, dim0 stride = 2 * page."""
    p, bs, n, d = key.shape
    kv_shape = (2, p, bs, n, d)
    kv_cache = torch.zeros(math.prod(kv_shape), dtype=key.dtype, device=key.device).view(kv_shape)
    hidden_size = kv_cache.shape[2:].numel()
    kv_cache.as_strided_(
        size=kv_shape,
        stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
    )
    k, v = kv_cache.unbind(0)
    k.copy_(key)
    v.copy_(value)
    return k, v


@register("aclnn_genericblocksparseattentioninputprocess")
class GenericBlockSparseAttentionQuantInputProcessUncon(_ContigInputProcess):
    def init_by_input_data(self, input_data):
        key = input_data.kwargs["key"]
        value = input_data.kwargs["value"]
        if not isinstance(key, torch.Tensor):
            key = torch.as_tensor(key)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        k, v = make_fused_noncontig_kv(key, value)
        input_data.kwargs["key"] = k
        input_data.kwargs["value"] = v
        return super().init_by_input_data(input_data)
