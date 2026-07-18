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
    """MLAAttentionSpec extended to support DSA models, with independent SFA and LI C8 support.

    When LI C8 is enabled, the KV cache tuple changes from
    (kv_cache[0]: bfloat16, kv_cache[1]: bfloat16, kv_cache[2]: bfloat16)
    to
    (kv_cache[0]: bfloat16, kv_cache[1]: bfloat16, kv_cache[2]: int8, kv_cache[3]: float16).

    The semantic meaning of each native KV cache entry is as follows:
    1. kv_cache[0] stores kv_lora.
    2. kv_cache[1] stores k_rope.
    3. kv_cache[2] stores the key tensor from the indexer module.
    4. kv_cache[3] stores the key scale tensor from the indexer module,
       and exists only when LI C8 is enabled.

    With SFA C8, kv_lora, k_rope, and per-tile quantization scales are
    packed into kv_cache[0]. The resulting cache is (packed_kv, indexer_k)
    or (packed_kv, indexer_k, indexer_scale) when LI C8 is also enabled.

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
    cache_sparse_sfa_c8: bool = False
    cache_sparse_li_c8: bool = False
    c8_k_cache_dtype: torch.dtype = field(default_factory=_get_c8_k_cache_dtype)
    c8_k_scale_cache_dtype: torch.dtype = field(default_factory=_get_c8_k_scale_cache_dtype)
    sfa_dcp_replicated_indexer_size: int = 1

    @property
    def page_size_bytes(self) -> int:
        if self.cache_sparse_sfa_c8:
            assert self.sparse_head_dim is not None
            assert len(self.sparse_head_dim) == 3
            num_heads_per_page = self.block_size * self.num_kv_heads

            ckv_head_dim, qk_rope_head_dim, index_head_dim = self.sparse_head_dim
            assert qk_rope_head_dim == 0

            ckv_bytes = num_heads_per_page * ckv_head_dim * get_dtype_size(self.c8_k_cache_dtype)
            qli_dtype = self.c8_k_cache_dtype if self.cache_sparse_li_c8 else self.dtype
            qli_bytes = (
                num_heads_per_page * index_head_dim * self.sfa_dcp_replicated_indexer_size * get_dtype_size(qli_dtype)
            )
            qli_scale_bytes = (
                num_heads_per_page * self.sfa_dcp_replicated_indexer_size * get_dtype_size(self.c8_k_scale_cache_dtype)
                if self.cache_sparse_li_c8 and index_head_dim > 0
                else 0
            )
            return ckv_bytes + qli_bytes + qli_scale_bytes

        if self.cache_sparse_li_c8:
            assert self.sparse_head_dim is not None
            assert len(self.sparse_head_dim) == 3

            k_head_dim, v_head_dim, index_head_dim = self.sparse_head_dim
            assert index_head_dim > 0
            num_heads_per_page = self.block_size * self.num_kv_heads
            return num_heads_per_page * (
                (k_head_dim + v_head_dim) * get_dtype_size(self.dtype)
                + index_head_dim * self.sfa_dcp_replicated_indexer_size * get_dtype_size(self.c8_k_cache_dtype)
                + self.sfa_dcp_replicated_indexer_size * get_dtype_size(self.c8_k_scale_cache_dtype)
            )

        if (
            self.sparse_head_dim is not None
            and len(self.sparse_head_dim) == 3
            and self.sfa_dcp_replicated_indexer_size > 1
        ):
            k_head_dim, v_head_dim, index_head_dim = self.sparse_head_dim
            replicated_head_size = k_head_dim + v_head_dim + index_head_dim * self.sfa_dcp_replicated_indexer_size
            return (
                self.block_size
                * self.num_kv_heads
                * (
                    replicated_head_size * get_dtype_size(self.dtype)
                    + self.scale_dim * get_dtype_size(self.scale_dtype)
                )
            )

        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    @property
    def sparse_kv_cache_ratio(self) -> tuple[float, float | None, float | None, float | None]:
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

        if self.cache_sparse_sfa_c8:
            ckv_head_dim, qk_rope_head_dim, index_k_head_dim = self.sparse_head_dim
            assert qk_rope_head_dim == 0

            ckv_virtual = ckv_head_dim * get_dtype_size(self.c8_k_cache_dtype)
            if index_k_head_dim == 0:
                return (
                    1.0,
                    None,
                    None,
                    None,
                )

            qli_dtype = self.c8_k_cache_dtype if self.cache_sparse_li_c8 else self.dtype
            qli_virtual = index_k_head_dim * self.sfa_dcp_replicated_indexer_size * get_dtype_size(qli_dtype)
            scale_virtual = (
                self.sfa_dcp_replicated_indexer_size * get_dtype_size(self.c8_k_scale_cache_dtype)
                if self.cache_sparse_li_c8
                else 0
            )
            total_virtual_head_dim = ckv_virtual + qli_virtual + scale_virtual

            return (
                total_virtual_head_dim / ckv_virtual,
                total_virtual_head_dim / qli_virtual,
                total_virtual_head_dim / scale_virtual if scale_virtual > 0 else None,
                None,
            )

        k_head_dim, v_head_dim, index_head_dim = self.sparse_head_dim
        replicated_index_head_dim = index_head_dim * self.sfa_dcp_replicated_indexer_size
        if self.cache_sparse_li_c8:
            k_virtual = k_head_dim * get_dtype_size(self.dtype)
            v_virtual = v_head_dim * get_dtype_size(self.dtype)
            qli_virtual = replicated_index_head_dim * get_dtype_size(self.c8_k_cache_dtype)
            scale_virtual = self.sfa_dcp_replicated_indexer_size * get_dtype_size(self.c8_k_scale_cache_dtype)
            total_virtual_head_dim = k_virtual + v_virtual + qli_virtual + scale_virtual
            return (
                total_virtual_head_dim / k_virtual,
                total_virtual_head_dim / v_virtual,
                total_virtual_head_dim / qli_virtual,
                total_virtual_head_dim / scale_virtual,
            )

        total_virtual_head_dim = k_head_dim + v_head_dim + replicated_index_head_dim
        return (
            total_virtual_head_dim / k_head_dim,
            total_virtual_head_dim / v_head_dim,
            total_virtual_head_dim / replicated_index_head_dim if replicated_index_head_dim > 0 else None,
            None,
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        layout_set = {
            (
                spec.block_size,
                spec.num_kv_heads,
                spec.head_size,
                spec.scale_dim,
                spec.scale_dtype,
                spec.sparse_head_dim,
                spec.dtype,
            )
            for spec in specs
        }
        assert len(layout_set) == 1, (
            "All attention layers in the same KV cache group must use the same KV cache layout."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        assert len(cache_dtype_str_set) == 1, (
            "All attention layers in the same KV cache group must use the same quantization method."
        )
        cache_sparse_sfa_c8_set = set(spec.cache_sparse_sfa_c8 for spec in specs)
        assert len(cache_sparse_sfa_c8_set) == 1, (
            "All attention layers in the same KV cache group must use the same sparse SFA C8 setting."
        )
        cache_sparse_li_c8_set = set(spec.cache_sparse_li_c8 for spec in specs)
        assert len(cache_sparse_li_c8_set) == 1, (
            "All attention layers in the same KV cache group must use the same sparse LI C8 setting."
        )
        sfa_dcp_replicated_indexer_size_set = set(spec.sfa_dcp_replicated_indexer_size for spec in specs)
        assert len(sfa_dcp_replicated_indexer_size_set) == 1, (
            "All attention layers in the same KV cache group must use the same SFA DCP replicated indexer size."
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
            cache_sparse_sfa_c8=specs[0].cache_sparse_sfa_c8,
            cache_sparse_li_c8=specs[0].cache_sparse_li_c8,
            sfa_dcp_replicated_indexer_size=sfa_dcp_replicated_indexer_size_set.pop(),
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
