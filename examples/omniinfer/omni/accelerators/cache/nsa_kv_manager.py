# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


import os
from collections import defaultdict
from dataclasses import dataclass
from typing_extensions import override

from vllm.logger import init_logger
from vllm.utils import sha256, cdiv
from vllm.config import get_layers_from_vllm_config
from vllm.attention import AttentionType
from vllm.attention.layer import Attention
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request
from .kv_cache_manager import get_manager_for_kv_cache_spec, OmniKVCacheManager, OmniKVCacheBlocks
from .omni_cache import PrefillOmniCache, DecodeOmniCache


logger = init_logger("vllm.v1.omni")


@dataclass
class ConvolutionCompressSpec(FullAttentionSpec):
    conv_winsize: int = 64
    stride: int = 32

    @property
    def type_id(self) -> str:
        return f"conv_compress_{self.conv_winsize}_{self.stride}_{self.block_size}_{self.page_size_bytes}"

    def max_memory_usage_bytes(self, vllm_config) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        max_compress_len = self.calc_compress_len(max_model_len)
        return cdiv(max_compress_len, self.block_size) * self.page_size_bytes

    def calc_compress_len(self, original_len: int):
        if original_len < self.conv_winsize:
            return 0
        return (original_len - self.conv_winsize) // self.stride + 1


def get_nsa_kv_cache_spec(self: GPUModelRunner) -> dict[str, ConvolutionCompressSpec]:
    """Generates the KVCacheSpec by parsing the kv cache format from
    each attention module in the static forward context.
    Returns:
        A dictionary mapping layer names to their KV cache format.
    """
    layers = get_layers_from_vllm_config(self.vllm_config, Attention)
    block_size = self.vllm_config.cache_config.block_size
    use_mla = self.vllm_config.model_config.use_mla
    kv_cache_spec: dict[str, ConvolutionCompressSpec] = {}
    for layer_name, attn_module in layers.items():
        if attn_module.attn_type == AttentionType.DECODER:
            kv_cache_spec[layer_name] = ConvolutionCompressSpec(
                block_size=block_size,
                num_kv_heads=attn_module.num_kv_heads,
                head_size=attn_module.head_size,
                dtype=self.kv_cache_dtype,
                use_mla=use_mla,
                conv_winsize=64,
                stride=32,
            )
        else:
            raise NotImplementedError("Omni attention supports decoder-only models.")
    return kv_cache_spec


class ConvolutionCompressKVManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: ConvolutionCompressSpec, *args, **kwargs):
        super().__init__(kv_cache_spec, *args, **kwargs)

    @override
    def get_num_blocks_to_allocate(
            self, request_id: str, num_tokens: int,
            new_computed_blocks: list[KVCacheBlock]) -> int:
        num_required_compress_blocks = cdiv(
            self.kv_cache_spec.calc_compress_len(num_tokens),
            self.block_size,
        )
        num_new_compress_blocks = num_required_compress_blocks - len(self.req_to_blocks[request_id])

        return num_new_compress_blocks

    @override
    def save_new_computed_blocks(
            self, request_id: str,
            new_computed_blocks: list[KVCacheBlock]) -> None:
        return

    @override
    def allocate_new_blocks(self, request_id: str, num_tokens: int) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        num_required_compress_blocks = cdiv(
            self.kv_cache_spec.calc_compress_len(num_tokens),
            self.block_size,
        )
        num_new_compress_blocks = num_required_compress_blocks - len(req_blocks)
        if num_new_compress_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_compress_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    @override
    def cache_blocks(self, request: Request, block_hashes: list[BlockHashType],
                     num_tokens: int) -> None:
        return

    def find_longest_cache_hit(self, block_hashes: list[BlockHashType],
                               max_length: int) -> list[KVCacheBlock]:
        raise NotImplementedError("Method find_longest_cache_hit is not implemented yet for ConvolutionCompressKVManager")

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        return

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> int:
        return 0


class NSAHostDeviceKVCacheManager(OmniKVCacheManager):

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
    ) -> None:
        assert len(kv_cache_config.kv_cache_groups) == 1, (
            "NSAHostDeviceKVCacheManager does not support hybrid models with more than 1 "
            "kv cache group")

        kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = kv_cache_spec.block_size
        self.num_gpu_blocks = kv_cache_config.num_blocks
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.caching_hash_fn = sha256 if caching_hash_algo == "sha256" else hash
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        # calculate number of blocks for host cache
        num_layers = len(kv_cache_config.kv_cache_groups[0].layer_names)
        num_kv_heads = kv_cache_spec.num_kv_heads
        head_size = kv_cache_spec.head_size
        dtype = kv_cache_spec.dtype

        if os.getenv("ROLE") == "prefill":
            num_host_blocks = PrefillOmniCache.calc_cache_shape_for_prefill(
                num_layers=num_layers,
                block_size=self.block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=dtype
            )[1]
            self.block_pools: list[BlockPool] = [
                BlockPool(num_host_blocks, enable_caching, enable_kv_cache_events),
            ]
            logger.warning(f"**NSAHostDeviceKVCacheManager**: For prefill, {num_host_blocks} blocks are available for host cache.")
        else:
            num_host_blocks = DecodeOmniCache.calc_cache_shape_for_decode(
                num_layers=num_layers,
                block_size=self.block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=dtype
            )[1]
            self.block_pools: list[BlockPool] = [
                BlockPool(self.num_gpu_blocks, enable_caching, enable_kv_cache_events),
                BlockPool(num_host_blocks, enable_caching, enable_kv_cache_events),
            ]
            logger.warning(f"**NSAHostDeviceKVCacheManager**: For decode, {num_host_blocks} blocks are available for host cache and {self.num_gpu_blocks} blocks for device cache.")

        self.hybrid_managers: list[SingleTypeKVCacheManager] = []
        for block_pool in self.block_pools:
            self.hybrid_managers.append(
                get_manager_for_kv_cache_spec(
                    kv_cache_spec=kv_cache_spec,
                    use_eagle=self.use_eagle,
                    num_kv_cache_groups=1,
                    caching_hash_fn=self.caching_hash_fn,
                    block_pool=block_pool,
                )
            )

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: defaultdict[
            str, list[BlockHashType]] = defaultdict(list)

    # @override
    # def get_computed_blocks(self,
    #                         request: Request) -> tuple[OmniKVCacheBlocks, int]:
    #     kv_cache_blocks, num_computed_tokens = super().get_computed_blocks(request)
    #     blocks: list[KVCacheBlock] = kv_cache_blocks.blocks
    #     multi_group_blocks = OmniKVCacheBlocks([blocks for _ in range(len(self.hybrid_managers))])
    #     return multi_group_blocks, num_computed_tokens
