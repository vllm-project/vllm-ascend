# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import math

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import AttentionSpec, compute_layer_kv_cache_shape_bytes

from vllm_ascend.worker.utils import make_page_strided_cache_view, reshape_combined_attention_kv_cache


class Step5AttentionCacheView:
    def create_kv_cache_view(
        self,
        raw_cache: torch.Tensor,
        spec: AttentionSpec,
        num_blocks: int,
        block_stride: int,
        kernel_block_size: int,
        cache_dtype: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the page's K and V with the backend's axis order."""
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
        return reshape_combined_attention_kv_cache(raw_cache, shape, spec.dtype, block_stride, kernel_blocks_per_page)


class Step5AuxiliaryCacheView:
    @staticmethod
    def create_kv_cache_view(
        raw_cache: torch.Tensor,
        spec: AttentionSpec,
        num_blocks: int,
        block_stride: int,
        kernel_block_size: int,
        cache_dtype: str,
    ) -> torch.Tensor:
        """Expose one typed [B, H, N, C] view for the layer's binder."""
        if not isinstance(raw_cache, torch.Tensor):
            raise ValueError("Step5 auxiliary cache must use one raw tensor.")
        blocks, heads, tokens, content_bytes = compute_layer_kv_cache_shape_bytes(spec, num_blocks)
        if math.prod((heads, tokens, content_bytes)) > block_stride:
            raise ValueError("Step5 auxiliary cache view must fit inside one physical page.")
        shape = (blocks, heads, tokens, content_bytes // get_dtype_size(spec.dtype))
        return make_page_strided_cache_view(raw_cache, shape, spec.dtype, block_stride)
