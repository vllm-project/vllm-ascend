# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch
from vllm.utils.torch_utils import get_dtype_size


def reshape_combined_attention_kv_cache(
    raw_cache: torch.Tensor,
    kv_cache_shape: tuple[int, ...],
    dtype: torch.dtype,
    page_stride_bytes: int,
    num_blocks_per_kv_block: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """为带填充的物理页创建按 block 跨步的 K/V 视图。"""
    if len(kv_cache_shape) != 5 or kv_cache_shape[0] != 2:
        raise ValueError("Combined Attention cache must have shape [K/V, blocks, block_size, heads, dim].")
    dtype_size = get_dtype_size(dtype)
    if page_stride_bytes % dtype_size:
        raise ValueError("Physical Attention page is not aligned to its dtype.")

    hidden_size = math.prod(kv_cache_shape[2:])
    if num_blocks_per_kv_block < 1:
        raise ValueError("The number of kernel blocks per KV block must be positive.")
    if page_stride_bytes % (num_blocks_per_kv_block * dtype_size):
        raise ValueError("Padded combined Attention pages must split into dtype-aligned kernel blocks.")
    kernel_stride_bytes = page_stride_bytes // num_blocks_per_kv_block
    if kernel_stride_bytes < 2 * hidden_size * dtype_size:
        raise ValueError("Physical Attention page is too small for its kernel blocks.")
    dense_strides = [math.prod(kv_cache_shape[dim + 1 :]) for dim in range(len(kv_cache_shape))]
    combined_cache = torch.as_strided(
        raw_cache.view(dtype),
        size=kv_cache_shape,
        stride=(
            hidden_size,
            kernel_stride_bytes // dtype_size,
            *dense_strides[2:],
        ),
    )
    return combined_cache[0], combined_cache[1]
