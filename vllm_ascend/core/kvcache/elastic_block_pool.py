from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable
from typing import Any, Optional, Type

from .kv_cache_manager import KVCacheManager


logger = logging.getLogger(__name__)


def build_elastic_block_pool_class(block_pool_mod: Any,
                                   manager_cls: Type[KVCacheManager] =
                                   KVCacheManager) -> type:
    if hasattr(block_pool_mod, "ElasticBlockPool"):
        return getattr(block_pool_mod, "ElasticBlockPool")

    BlockPool = getattr(block_pool_mod, "BlockPool")
    KVCacheBlock = getattr(block_pool_mod, "KVCacheBlock")

    class ElasticBlockPool(BlockPool):  # type: ignore[misc]

        def __init__(
            self,
            num_gpu_blocks: int,
            block_size: int,
            cell_size: int,
            num_layers: int,
            enable_caching: bool,
            enable_kv_cache_events: bool = False,
        ) -> None:
            assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
            assert not enable_caching, "Caching is not supported in ElasticBlockPool"
            assert not enable_kv_cache_events, (
                "KV cache events are not supported in ElasticBlockPool")

            self.num_gpu_blocks = num_gpu_blocks
            self.enable_kv_cache_events = enable_kv_cache_events
            self.kv_event_queue: list[Any] = []

            self.kv_cache_manager = manager_cls(
                num_blocks=num_gpu_blocks,
                block_size=block_size,
                cell_size=cell_size,
                num_layers=num_layers,
            )

            self.null_block = None

        def get_cached_block(self, *args: Any,
                             **kwargs: Any) -> Optional[list[Any]]:
            return None

        def cache_full_blocks(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError(
                "Caching is not supported in ElasticBlockPool")

        def get_new_blocks(self, num_blocks: int) -> list[Any]:
            if num_blocks > self.get_num_free_blocks():
                raise ValueError(
                    f"Cannot get {num_blocks} free blocks from the pool")

            block_ids = self.kv_cache_manager.alloc(num_blocks)
            if block_ids is None or len(block_ids) != num_blocks:
                # logger.error("Failed to allocate blocks: need=%d, got=%s",
                #              num_blocks, block_ids)
                return []

            return [KVCacheBlock(bid) for bid in block_ids]

        def kv_resize(self, size: int) -> bool:
            if size == 0:
                return True
            return self.kv_cache_manager.resize(size)

        def get_kv_size(self) -> int:
            return self.kv_cache_manager.get_kv_size()

        def touch(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Prefix caching is not supported in ElasticBlockPool")

        def free_blocks(self, ordered_blocks: Iterable[Any]) -> None:
            block_ids = [
                block.block_id  # type: ignore[attr-defined]
                for block in ordered_blocks
            ]
            if block_ids:
                self.kv_cache_manager.free(block_ids)

        def reset_prefix_cache(self) -> bool:
            logger.error(f"Prefix caching is not supported in ElasticBlockPool")
            return False
        
        def get_num_free_blocks(self) -> int:
            return self.kv_cache_manager.available_size()

        def get_usage(self) -> float:
            return 1.0 - (self.get_num_free_blocks() / self.num_gpu_blocks)

        def take_events(self) -> list[Any]:
            return []

    setattr(block_pool_mod, "ElasticBlockPool", ElasticBlockPool)
    return ElasticBlockPool


def inject_elastic_block_pool(block_pool_mod: Any) -> type:
    return build_elastic_block_pool_class(block_pool_mod)


def inject_elastic_block_pool_by_import(
    module_name: str = "vllm.v1.core.block_pool",
) -> type:
    block_pool_mod = importlib.import_module(module_name)
    return inject_elastic_block_pool(block_pool_mod)
