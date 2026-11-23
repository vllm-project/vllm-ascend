# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import math

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import AttentionSpec, compute_layer_kv_cache_shape_bytes

Step5CacheViewLayout = tuple[tuple[tuple[int, ...], ...], tuple[torch.dtype, ...], int]


class Step5AttentionCacheView:
    def get_kv_cache_view_layout(
        self,
        spec: AttentionSpec,
        num_blocks: int,
        block_stride: int,
        kernel_block_size: int,
        cache_dtype: str,
    ) -> Step5CacheViewLayout:
        """Describe K and V in the backend's axis order."""
        kernel_blocks_per_page, remainder = divmod(spec.block_size, kernel_block_size)
        if remainder:
            raise ValueError("Kernel block size must divide the Step5 cache block size.")
        shape = self.get_attn_backend().get_kv_cache_shape(
            num_blocks * kernel_blocks_per_page,
            kernel_block_size,
            spec.num_kv_heads,
            spec.head_size,
            (getattr(spec, "cache_dtype_str", None) or "auto") if spec.is_draft else cache_dtype,
        )
        if len(shape) != 5 or shape[0] != 2:
            raise ValueError("Step5 attention cache must describe K and V.")
        component_shape = shape[1:]
        kernel_stride, remainder = divmod(block_stride, kernel_blocks_per_page)
        itemsize = get_dtype_size(spec.dtype)
        if remainder or kernel_stride % itemsize:
            raise ValueError("Step5 attention pages must split into dtype-aligned kernel blocks.")
        if 2 * math.prod(component_shape[1:]) * itemsize > kernel_stride:
            raise ValueError("Step5 K and V must fit inside one physical kernel block.")
        return (component_shape, component_shape), (spec.dtype, spec.dtype), kernel_stride


class Step5AuxiliaryCacheView:
    @staticmethod
    def get_kv_cache_view_layout(
        spec: AttentionSpec,
        num_blocks: int,
        block_stride: int,
        kernel_block_size: int,
        cache_dtype: str,
    ) -> Step5CacheViewLayout:
        """Describe one typed [B, H, N, C] view for the layer's binder."""
        blocks, heads, tokens, content_bytes = compute_layer_kv_cache_shape_bytes(spec, num_blocks)
        if math.prod((heads, tokens, content_bytes)) > block_stride:
            raise ValueError("Step5 auxiliary cache view must fit inside one physical page.")
        shape = (blocks, heads, tokens, content_bytes // get_dtype_size(spec.dtype))
        return (shape,), (spec.dtype,), block_stride
