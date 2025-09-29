from typing import Dict, Tuple

import torch
from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class CacheEngine:

    def __init__(self,attn_backends: list[type[AttentionBackend]],kv_cache_config: KVCacheConfig,
                 gpu_cache: Dict[str, Tuple[torch.Tensor, ...]], cpu_cache: Dict[str, Tuple[torch.Tensor, ...]]):
        self.gpu_cache = gpu_cache
        self.cpu_cache = cpu_cache
        self.attn_backends = attn_backends
        self.kv_cache_config = kv_cache_config

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                self.attn_backends[i].swap_blocks(self.cpu_cache[layer_name], self.gpu_cache[layer_name], src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                self.attn_backends[i].swap_blocks(self.gpu_cache[layer_name], self.cpu_cache[layer_name], src_to_dst)