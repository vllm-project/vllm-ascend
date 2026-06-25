# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

import torch
from typing_extensions import Self
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec, SlidingWindowMLASpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.core.single_type_kv_cache_manager import CompressAttentionManager
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def _get_c8_k_cache_dtype() -> torch.dtype:
    return torch.float8_e4m3fn if get_ascend_device_type() == AscendDeviceType.A5 else torch.int8


def _get_c8_k_scale_cache_dtype() -> torch.dtype:
    return torch.float32 if get_ascend_device_type() == AscendDeviceType.A5 else torch.float16


@dataclass(frozen=True, kw_only=True)
class AscendMLAAttentionSpec(MLAAttentionSpec):
    """MLAAttentionSpec extended to support DSA models, with optional Sparse C8 support.

    When Sparse C8 is enabled, the KV cache tuple changes from
    (kv_cache[0]: bfloat16, kv_cache[1]: bfloat16, kv_cache[2]: bfloat16)
    to
    (kv_cache[0]: bfloat16, kv_cache[1]: bfloat16, kv_cache[2]: int8, kv_cache[3]: float16).

    The semantic meaning of each KV cache entry is as follows:
    1. kv_cache[0] stores kv_lora.
    2. kv_cache[1] stores k_rope.
    3. kv_cache[2] stores the key tensor from the indexer module.
    4. kv_cache[3] stores the key scale tensor from the indexer module,
       and exists only when Sparse C8 is enabled.

    The main changes are as follows:
    1. The key tensor from the indexer module stored in kv_cache[2] is
       converted from bf16 to int8 to reduce memory usage. It is then
       processed with int8 precision in Lightning_indexer computation
       to improve computational efficiency.
    2. The quantization scale of the key tensor in the indexer module
       must also be stored for the Lightning_indexer_quant operator,
       and is therefore saved in kv_cache[3].
    """

    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    sparse_head_dim: tuple[int, ...] | None = None
    cache_sparse_c8: bool = False
    c8_k_cache_dtype: torch.dtype = field(default_factory=_get_c8_k_cache_dtype)
    c8_k_scale_cache_dtype: torch.dtype = field(default_factory=_get_c8_k_scale_cache_dtype)

    @property
    def page_size_bytes(self) -> int:
        if self.cache_sparse_c8:
            assert self.sparse_head_dim is not None
            assert len(self.sparse_head_dim) == 3
            num_heads_per_page = self.block_size * self.num_kv_heads

            kv_lora_rank, qk_rope_head_dim, index_head_dim = self.sparse_head_dim

            # A5: kv_lora and k_rope are merged into a single CKV tensor (fp8).
            # A3: separate kv_lora + k_rope (bf16).
            if qk_rope_head_dim == 0:
                kv_dtype = self.c8_k_cache_dtype  # A5 CKV: float8_e4m3fn
                kv_dim = kv_lora_rank
            else:
                kv_dtype = self.dtype  # A3 kv_lora + k_rope: bfloat16
                kv_dim = kv_lora_rank + qk_rope_head_dim

            kv_bytes = num_heads_per_page * kv_dim * get_dtype_size(kv_dtype)
            qli_bytes = num_heads_per_page * index_head_dim * get_dtype_size(self.c8_k_cache_dtype)
            qli_scale_bytes = num_heads_per_page * 1 * get_dtype_size(self.c8_k_scale_cache_dtype)
            return kv_bytes + qli_bytes + qli_scale_bytes

        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    @property
    def sparse_kv_cache_ratio(self) -> tuple[float, float, float, float | None]:
        """
        Compute the relative byte share of each KV cache entry.

        Returns:
            A tuple containing the ratios for:
            - kv_cache[0]
            - kv_cache[1]
            - kv_cache[2]
            - kv_cache[3] (None if Sparse C8 is disabled or Sparse C8 on A5 device)
        """

        assert self.sparse_head_dim is not None

        def get_sparse_head_dim_virtual() -> tuple[int, int, int, int]:
            assert self.sparse_head_dim is not None
            assert self.cache_sparse_c8 is True

            kv_lora_rank, qk_rope_head_dim, index_k_head_dim = self.sparse_head_dim

            if qk_rope_head_dim == 0:
                # A5: ckv (float8_e4m3fn) and qli share c8_k_cache_dtype;
                ckv_virtual = kv_lora_rank * get_dtype_size(self.c8_k_cache_dtype)
                qk_rope_virtual = 0
                qli_virtual = index_k_head_dim * get_dtype_size(self.c8_k_cache_dtype)
                scale_virtual = get_dtype_size(self.c8_k_scale_cache_dtype)
                return (ckv_virtual, qk_rope_virtual, qli_virtual, scale_virtual)

            # A3: keep the original element-count / byte mix
            factor = get_dtype_size(self.dtype) // get_dtype_size(self.c8_k_cache_dtype)
            index_k_head_dim_virtual = index_k_head_dim // factor

            assert get_dtype_size(self.dtype) == get_dtype_size(self.c8_k_scale_cache_dtype)
            index_k_scale_head_dim_virtual = 1

            return (
                kv_lora_rank,
                qk_rope_head_dim,
                index_k_head_dim_virtual,
                index_k_scale_head_dim_virtual,
            )

        if self.cache_sparse_c8:
            virtual_dims = get_sparse_head_dim_virtual()
            total_virtual_head_dim = sum(virtual_dims)

            if virtual_dims[1] == 0:
                # A5: ckv merged (kv_lora + k_rope + scale) → 3-tensor
                return (
                    total_virtual_head_dim / virtual_dims[0],  # kv_cache[0]: ckv
                    total_virtual_head_dim / virtual_dims[2],  # kv_cache[1]: qli
                    total_virtual_head_dim / virtual_dims[3],  # kv_cache[2]: qli_scale
                    None,  # kv_cache[3] does not exist for A5
                )
            else:
                # A3: 4-tensor
                return (
                    total_virtual_head_dim / virtual_dims[0],  # kv_cache[0]
                    total_virtual_head_dim / virtual_dims[1],  # kv_cache[1]
                    total_virtual_head_dim / virtual_dims[2],  # kv_cache[2]
                    total_virtual_head_dim / virtual_dims[3],  # kv_cache[3]
                )

        return (
            self.head_size / self.sparse_head_dim[0],  # kv_cache[0]
            self.head_size / self.sparse_head_dim[1],  # kv_cache[1]
            self.head_size / self.sparse_head_dim[2],  # kv_cache[2]
            None,  # kv_cache[3] does not exist
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        assert len(cache_dtype_str_set) == 1, (
            "All attention layers in the same KV cache group must use the same quantization method."
        )
        cache_sparse_c8_set = set(spec.cache_sparse_c8 for spec in specs)
        assert len(cache_sparse_c8_set) == 1, (
            "All attention layers in the same KV cache group must use the same sparse C8 setting."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            scale_dim=specs[0].scale_dim,
            scale_dtype=specs[0].scale_dtype,
            sparse_head_dim=specs[0].sparse_head_dim,
            dtype=specs[0].dtype,
            cache_dtype_str=cache_dtype_str_set.pop(),
            cache_sparse_c8=specs[0].cache_sparse_c8,
        )

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        # Note(hc): each dcp rank only need save
        # (max_model_len//dcp_world_size) tokens locally.
        if dcp_world_size * pcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
        return cdiv(max_model_len, self.block_size * self.compress_ratio) * self.page_size_bytes


@dataclass(frozen=True, kw_only=True)
class AscendSlidingWindowMLASpec(SlidingWindowMLASpec):
    """Sliding window attention with MLA cache format."""

    cache_dtype_str: str | None = None
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1
    model_version: str | None = None

    def __post_init__(self):
        pass

    @property
    def storage_block_size(self) -> int:
        return self.block_size

    @property
    def real_page_size_bytes(self) -> int:
        return self.storage_block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, AscendSlidingWindowMLASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be AscendSlidingWindowMLASpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        sliding_window_set = set(spec.sliding_window for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(sliding_window_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version and sliding "
            "window size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=sliding_window_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
        )


def register_ascend_kv_cache_specs() -> None:
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendMLAAttentionSpec,
        manager_class=CompressAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendSlidingWindowMLASpec,
        manager_class=SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowMLASpec,
    )
