from dataclasses import dataclass, fields
from typing import Optional

import torch
import vllm
from typing_extensions import Self
from vllm.config import VllmConfig
from vllm.utils import cdiv, get_dtype_size
from vllm.v1.core.single_type_kv_cache_manager import (FullAttentionManager,
                                                       spec_manager_map)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec


@dataclass(frozen=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    use_mla: bool
    use_sparse: bool

    @property
    def page_size_bytes(self) -> int:
        # For MLA we only store a single latent vector
        coef = 1 if self.use_mla else 2
        sfa_bytes = 128 * self.block_size * get_dtype_size(
            self.dtype) if self.use_sparse else 0

        return coef * self.block_size * self.num_kv_heads * self.head_size \
                * get_dtype_size(self.dtype) + sfa_bytes


vllm.v1.kv_cache_interface.AttentionSpec = AttentionSpec


@dataclass(frozen=True)
class AscendFullAttentionSpec(FullAttentionSpec, AttentionSpec):
    sliding_window: Optional[int] = None
    attention_chunk_size: Optional[int] = None
    """
    When hybrid allocator is disabled and the model contains both full 
    attention layers and sliding window attention layers, sliding 
    window attention are regarded as full attention in KV cache manager 
    (blocks are allocated for all tokens), while computed as sliding window 
    attention in model runner.
    In this case, we use FullAttentionSpec and record the sliding window size.
    Default to None for not using sliding window attention.
    """

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = \
            vllm_config.parallel_config.decode_context_parallel_size
        # Note(hc): each dcp rank only need save
        # (max_model_len//dcp_world_size) tokens locally.
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    @classmethod
    def merge_window_sizes(cls, window_sizes: set[int]) -> Optional[int]:
        if len(window_sizes) == 0:
            return None
        elif len(window_sizes) == 1:
            return window_sizes.pop()
        else:
            raise ValueError(
                "All attention layers in the same KV cache group must have the "
                "same window size.")

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single 
        FullAttentionSpec object.
        """
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be "
            "FullAttentionSpec.")

        sliding_window = set(spec.sliding_window for spec in specs
                             if spec.sliding_window is not None)
        attention_chunk_size = set(spec.attention_chunk_size for spec in specs
                                   if spec.attention_chunk_size is not None)
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            use_mla=specs[0].use_mla,
            use_sparse=specs[0].use_sparse,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec.")
        assert (
            (merged_spec.sliding_window is not None) +
            (merged_spec.attention_chunk_size is not None) <= 1
        ), ("Model with both sliding window layers and chunked local attention "
            "layers is not supported.")
        return merged_spec


spec_manager_map.update({AscendFullAttentionSpec: FullAttentionManager})

vllm.v1.kv_cache_interface.FullAttentionSpec = AscendFullAttentionSpec
