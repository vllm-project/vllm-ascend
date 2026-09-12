# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vllm-ascend project

import torch

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

# One 32-byte DMA block contains 16 16-bit elements.
DMA_ALIGNMENT_ELEMENTS = 16
SUPPORTS_REARRANGE_QKV = get_ascend_device_type() in (AscendDeviceType.A2, AscendDeviceType.A3)


def rearrange_mixed_qkv(layer, mixed_qkv: torch.Tensor | None):
    """Use the custom QKV rearrange when the device and layout support it."""
    if (
        mixed_qkv is None
        or not SUPPORTS_REARRANGE_QKV
        or mixed_qkv.dtype not in (torch.bfloat16, torch.float16)
        or not mixed_qkv.is_contiguous()
    ):
        return layer.rearrange_mixed_qkv(mixed_qkv)

    q_dim = layer.key_dim // layer.tp_size
    k_dim = q_dim
    v_dim = layer.value_dim // layer.tp_size
    if q_dim % DMA_ALIGNMENT_ELEMENTS != 0 or v_dim % DMA_ALIGNMENT_ELEMENTS != 0:
        return layer.rearrange_mixed_qkv(mixed_qkv)

    num_tokens = mixed_qkv.shape[0]
    packed_qkv = torch.ops._C_ascend.npu_rearrange_qkv(mixed_qkv, q_dim, k_dim, v_dim)
    query, key, value = packed_qkv.split([num_tokens * q_dim, num_tokens * k_dim, num_tokens * v_dim])
    return (
        query.view(1, num_tokens, q_dim // layer.head_k_dim, layer.head_k_dim),
        key.view(1, num_tokens, k_dim // layer.head_k_dim, layer.head_k_dim),
        value.view(1, num_tokens, v_dim // layer.head_v_dim, layer.head_v_dim),
    )
