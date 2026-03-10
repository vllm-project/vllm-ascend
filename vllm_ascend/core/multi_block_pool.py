from functools import partial
from typing import Iterable, Optional

from vllm.distributed.kv_events import AllBlocksCleared, KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.request import Request

logger = init_logger(__name__)


class MultiBlockPool(BlockPool):
    """MultiBlockPool for sequence dimension parallelism(cp/dcp and dynamic cp).
        The basic functionality is consistent with the original parent class,
        but it extends multiple block pools, and different block pool IDs are
        recognized during allocation and reuse.

        Args:
            num_gpu_blocks: The number of blocks in the pool.
            enable_caching: Whether to enable prefix caching.
            hash_block_size: The block size of which the block hashes are computed.
                The actual block size usually equals hash_block_size, but in cases
                where different KV cache groups have different block sizes, the
                actual block size can be a multiple of hash_block_size.
            enable_kv_cache_events: Whether to enable kv cache events.
            metrics_collector: Optional metrics collector for tracking block residency.
    """

    def __init__(
        self,
        num_gpu_blocks: list[int],
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        self.num_pools = len(num_gpu_blocks)
        # Create global cached_block_hash_to_block and kv_event_queue.
        # The blocks and free_block_queue is useless in the global pool.
        # Set global pool id to -1.
        super().__init__(
            # The num_gpu_blocks of MultiBlockPool is equal to the sum of num_gpu_blocks of all the pools in it
            sum(num_gpu_blocks),
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )
        # Change the method of BlockPool, make sure all the operations of
        # cached_block_hash_to_block & kv_event_queue will call the global obj.
        # Make sure the method is changed before create local pool.
        BlockPool._maybe_evict_cached_block = (partial(
            BlockPool._maybe_evict_cached_block, self))
        # Create local block pools by the `BlockPool` class.
        self.block_pools = []
        for pool_id in range(self.num_pools):
            self.block_pools.append(
                BlockPool(num_gpu_blocks[pool_id], enable_caching,
                          hash_block_size, enable_kv_cache_events,
                          metrics_collector))

    def get_cached_block(
            self, block_hash: BlockHash,
            kv_cache_group_ids: list[int]) -> Optional[list[KVCacheBlock]]:
        # Use the global cached_block_hash_to_block.
        # Call the method of parent class directly.
        return super().get_cached_block(block_hash, kv_cache_group_ids)

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        # Use the global cached_block_hash_to_block and kv_event_queue.
        # Call the method of parent class directly.
        return super().cache_full_blocks(request, blocks, num_cached_blocks,
                                         num_full_blocks, block_size,
                                         kv_cache_group_id)

    def get_new_blocks(self, num_blocks: int, pool_id: int = 0):
        """Get new blocks by pool ids.

        Args:
            num_blocks: The number of blocks to allocate.
            pool_id: Specify the pool ID to allocate. If None,
                blocks will be allocated across all pools by default.

        Returns:
            A list of new block.
        """
        return self.block_pools[pool_id].get_new_blocks(num_blocks)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        for local_pool in self.block_pools:
            local_pool.evict_blocks(block_ids)

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        # This method will not be called in the global block pool,
        # and the local block pool in the block_pools will call the global
        # cached_block_hash_to_block & kv_event_queue as an alternative.
        return False

    def touch(self, blocks: tuple[list[KVCacheBlock], ...]) -> None:
        for pool_id, blocks_per_group in enumerate(blocks):
            for block in blocks_per_group:
                if block.ref_cnt == 0 and not block.is_null:
                    self.block_pools[pool_id].free_block_queue.remove(block)
                block.ref_cnt += 1

    def free_blocks(self,
                    ordered_blocks: Iterable[KVCacheBlock],
                    pool_id: int = 0) -> None:
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.block_pools[pool_id].free_block_queue.append_n([
            block for block in blocks_list
            if block.ref_cnt == 0 and not block.is_null
        ])

    def reset_prefix_cache(self) -> bool:
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        # Process global cached_block_hash_to_block only.
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap(
        )

        # Remove all hashes from all blocks.
        for local_pool in self.block_pools:
            for block in local_pool.blocks:
                block.reset_hash()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            # Process global kv_event_queue only.
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        # Return free blocks num in all block pools.
        num_free_blocks = 0
        for local_pool in self.block_pools:
            num_free_blocks += local_pool.get_num_free_blocks()
        return num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        return super().get_usage()

    def take_events(self) -> list[KVCacheEvent]:
        # Use the global kv_event_queue.
        # Call the method of parent class directly.
        return super().take_events()
