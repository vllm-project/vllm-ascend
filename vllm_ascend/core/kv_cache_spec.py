from dataclasses import dataclass

import torch
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import AttentionSpec, SlidingWindowSpec


@dataclass(frozen=True)
class CompressAttentionSpec(AttentionSpec):
    nope_dim: int
    rope_dim: int
    scale_dim: int
    nope_dtype: torch.dtype = torch.bfloat16
    rope_dtype: torch.dtype = torch.bfloat16
    scale_dtype: torch.dtype = torch.bfloat16


    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        base_page_size = self.block_size * self.head_size * 1 * get_dtype_size(
            self.dtype)
        indexer_page_size = self.block_size * self.indexer_head_size * 1 * get_dtype_size(
            self.dtype)
        page_size = (base_page_size + indexer_page_size)

        return page_size

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        max_model_len = vllm_config.model_config.max_model_len
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes


def pad_to_128(x: int):
    return ((x + 127) // 128) * 128


@dataclass(frozen=True)
class Compress4AttentionSpec(CompressAttentionSpec):
    # c4
    #   nope+rope+scale head_dim =  448    +    64  +  7
    #                           A5: fp8        bf16   fp8
    #                           A3: bf16       bf16    /
    #   pad to 128 to make sure the performance is ok
    compress_ratio: int = 4

    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        page_size = self.block_size * (
            self.nope_dim * get_dtype_size(self.nope_dtype) + \
            self.rope_dim * get_dtype_size(self.rope_dtype) + \
            self.scale_dim * get_dtype_size(self.scale_dtype)
        )
        page_size = pad_to_128(page_size)
        if self.page_size_padded is not None:
            assert (
            self.page_size_padded + page_size >= page_size
        ), f"pad_size >= 0 is required, however got {self.page_size_padded}"
            return self.page_size_padded + page_size
        return page_size

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        max_model_len = vllm_config.model_config.max_model_len
        return cdiv(max_model_len, self.block_size * self.compress_ratio) * self.page_size_bytes

@dataclass(frozen=True)
class Compress128AttentionSpec(CompressAttentionSpec):
    # c128
    #   nope+rope+scale head_dim = 448    +    64  +  7
    #                           A5: fp8        bf16   fp8
    #                           A3: bf16       bf16    /
    #   pad to 128 to make sure the performance is ok
    compress_ratio: int = 128

    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        base_page_size = self.block_size * (
            self.nope_dim * get_dtype_size(self.nope_dtype) + \
            self.rope_dim * get_dtype_size(self.rope_dtype) + \
            self.scale_dim * get_dtype_size(self.scale_dtype)
        )
        base_page_size = pad_to_128(base_page_size)
        page_size = base_page_size


        page_size = self.block_size * self.head_size * get_dtype_size(
            self.dtype)
        if self.page_size_padded is not None:
            assert (
            self.page_size_padded + page_size >= page_size
        ), f"pad_size >= 0 is required, however got {self.page_size_padded}"
            return self.page_size_padded + page_size
        return page_size


    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        max_model_len = vllm_config.model_config.max_model_len
        return cdiv(max_model_len, self.block_size * self.compress_ratio) * self.page_size_bytes

@dataclass(frozen=True)
class SWAAttentionSpec(SlidingWindowSpec):
    sliding_window: int = 128

    @property
    def page_size_bytes(self) -> int:
        page_size = (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )
        if self.page_size_padded is not None:
            assert (
            self.page_size_padded + page_size >= page_size
        ), f"pad_size >= 0 is required, however got {self.page_size_padded}"
            return self.page_size_padded + page_size
        return page_size

@dataclass(frozen=True)
class C4AttnKVStateSpec(SWAAttentionSpec):
    sliding_window: int = 8


@dataclass(frozen=True)
class C4AttnScoreStateSpec(SWAAttentionSpec):
    sliding_window: int = 8


@dataclass(frozen=True)
class C128AttnKVStateSpec(SWAAttentionSpec):
    sliding_window: int = 128


@dataclass(frozen=True)
class C128AttnScoreStateSpec(SWAAttentionSpec):
    sliding_window: int = 128

@dataclass(frozen=True)
class C4IndexerKVStateSpec(SWAAttentionSpec):
    sliding_window: int = 8


@dataclass(frozen=True)
class C4IndexerScoreStateSpec(SWAAttentionSpec):
    sliding_window: int = 8


@dataclass(frozen=True)
class C4IndexerSpec(AttentionSpec):
    # indexer attn
    #   value head_dim = 128    A3: int8 A5: fp8

    # indexer attn
    #   scale head_dim = 1      A3: fp16 A5: fp32
    compress_ratio: int = 4
    indexer_scale_dim: int = 0
    indexer_scale_dtype: torch.dtype = torch.float16

    @property
    def indexer_scale_size_bytes(self) -> int:
        indexer_scale_page_size = self.block_size * self.indexer_scale_dim * get_dtype_size(
            self.indexer_scale_dtype)
        return indexer_scale_page_size

    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        page_size = self.block_size * self.head_size * get_dtype_size(
            self.dtype)
        page_size += self.indexer_scale_size_bytes
        if self.page_size_padded is not None:
            assert (
            self.page_size_padded + page_size >= page_size
        ), f"pad_size >= 0 is required, however got {self.page_size_padded}"
            return self.page_size_padded + page_size
        return page_size

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        max_model_len = vllm_config.model_config.max_model_len
        return cdiv(max_model_len, self.block_size * self.compress_ratio) * self.page_size_bytes