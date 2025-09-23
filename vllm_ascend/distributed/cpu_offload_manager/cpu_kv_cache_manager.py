import time
from collections import defaultdict
from typing import Optional

from vllm.utils import logger, sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (BlockHash, KVCacheBlock,
                                         PrefixCachingMetrics)
from vllm.v1.core.single_type_kv_cache_manager import \
    get_manager_for_kv_cache_spec
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request


class CPUCacheStats:

    def __init__(self, enable_prefix_caching: bool, log_stats: bool = False):
        self.enable_prefix_caching = enable_prefix_caching
        self.log_stats = log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None
        self.cpu_prefix_cache_metrics = PrefixCachingMetrics()
        self.time_sec = int(time.time())

    def log(self):
        current_time_sec = int(time.time())
        # Log the prefix cache hit rate every 10 seconds.
        if current_time_sec - self.time_sec >= 10:
            self.time_sec = current_time_sec
            logger.info("CPU Prefix cache hit rate: %.1f%%",
                        self.cpu_prefix_cache_metrics.hit_rate * 100)

    def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        """Get (and reset) the prefix cache stats.
        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def update(self, num_tokens, num_computed_tokens):
        # Note the function is called by scheduler
        if self.log_stats and self.enable_prefix_caching:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1
            self.prefix_cache_stats.queries += num_tokens
            self.prefix_cache_stats.hits += num_computed_tokens

    def set_cache_stats(self, num_tokens, num_computed_tokens):
        assert self.prefix_cache_stats is not None
        self.prefix_cache_stats.hits = num_computed_tokens
        self.prefix_cache_stats.queries = num_tokens
        self.prefix_cache_stats.requests = 1


class CPUKVCacheManager:

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        num_cpu_blocks: int,
        caching_hash_algo: str = "builtin",
        use_eagle: bool = False,
        enable_kv_cache_events: bool = False,
    ) -> None:
        self.block_size = kv_cache_spec.block_size
        self.num_cpu_blocks = num_cpu_blocks
        self.caching_hash_fn = sha256 if caching_hash_algo == "sha256" else hash
        self.use_eagle = use_eagle
        self.block_pool = BlockPool(self.num_cpu_blocks, True,
                                    enable_kv_cache_events)
        self.single_type_manager = get_manager_for_kv_cache_spec(
            kv_cache_spec=kv_cache_spec,
            block_pool=self.block_pool,
            kv_cache_group_id=0,
        )
        # Record kv block hashes, avoid redundant computation.
        self.req_to_block_hashes: defaultdict[
            str, list[BlockHash]] = defaultdict(list)
        # Record blocks touched in get_matched_num_and_touch().
        self.req_to_computed_blocks: defaultdict[
            str, list[KVCacheBlock]] = defaultdict(list)
        # Record the request that failed to allocate.
        self.req_failed_to_allocate: defaultdict[str, bool] = defaultdict(bool)
        self.req_to_num_tokens: defaultdict[str, int] = defaultdict(int)
        self.cpu_cache_stats = CPUCacheStats(enable_prefix_caching=True,
                                             log_stats=True)
        # Record request that will be free after finish sending
        self.req_to_free: defaultdict[str, Request] = defaultdict(Request)

    def get_matched_num_and_touch(self, request: Request) -> tuple[int, bool]:
        # When the request requires prompt logprobs, we skip prefix caching.
        if (request.sampling_params.prompt_logprobs is not None):
            return 0, False
        request_id = request.request_id
        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request_id]
        if not block_hashes:
            block_hashes = request.block_hashes
            self.req_to_block_hashes[request_id] = block_hashes
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks = self.single_type_manager.find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.single_type_manager.kv_cache_spec,
            use_eagle=self.use_eagle,
        )
        num_computed_tokens = len(computed_blocks[0]) * self.block_size
        self.req_to_computed_blocks[request_id] = computed_blocks[0]
        # We should touch these blocks in the concurrent scenarios.
        self.block_pool.touch(computed_blocks)

        # cup prefix cache status set and log
        assert self.cpu_cache_stats is not None and self.cpu_cache_stats.prefix_cache_stats is not None
        self.cpu_cache_stats.set_cache_stats(request.num_tokens,
                                             num_computed_tokens)
        self.cpu_cache_stats.cpu_prefix_cache_metrics.observe(
            self.cpu_cache_stats.prefix_cache_stats)
        self.cpu_cache_stats.log()

        return num_computed_tokens, False

    def _release_ahead_touch(self, request_id: str):
        computed_blocks = self.req_to_computed_blocks[request_id]
        if computed_blocks:
            self.single_type_manager.block_pool.free_blocks(
                reversed(computed_blocks))
            self.req_to_computed_blocks.pop(request_id, None)

    def allocate_slots(self, req_to_num_tokens: dict[str, int],
                       unallocated_req_ids: set[str]) -> dict[str, list[int]]:
        for request_id in unallocated_req_ids:
            self._free_slots(request_id)
        req_to_new_blocks = {}
        for request_id, num_tokens in req_to_num_tokens.items():
            if self.req_failed_to_allocate[request_id]:
                continue
            new_computed_blocks = self.req_to_computed_blocks[request_id]
            num_blocks_to_allocate = (
                self.single_type_manager.get_num_blocks_to_allocate(
                    request_id=request_id,
                    num_tokens=num_tokens,
                    new_computed_blocks=new_computed_blocks,
                ))
            if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
                self._release_ahead_touch(request_id)
                self.req_failed_to_allocate[request_id] = True
                continue
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.single_type_manager.save_new_computed_blocks(
                request_id, new_computed_blocks)
            # Allocate new blocks but do not cache now.
            new_blocks = self.single_type_manager.allocate_new_blocks(
                request_id, num_tokens)
            self.req_to_num_tokens[request_id] = num_tokens
            # No need to release ref_cnt because we use officially.
            self.req_to_computed_blocks.pop(request_id, None)
            req_to_new_blocks[request_id] = [
                block.block_id for block in new_computed_blocks + new_blocks
            ]
        return req_to_new_blocks

    def record_request_cache_and_free_slots(self, request: Request):
        logger.debug(
            f"record_request_cache_and_free_slots for request {request.request_id} in cpu_kv_cache_manager"
        )
        self.req_to_free[request.request_id] = request

    def cache_and_free_slots(self, request_id: str):
        logger.debug(
            f"Cache and free slots for request {request_id} in cpu_kv_cache_manager"
        )
        if request_id not in self.req_to_free:
            logger.Error(
                f"request {request_id} not in req_to_free, maybe bug!")
            return
        request = self.req_to_free[request_id]
        if not self.req_failed_to_allocate[request_id]:
            self.single_type_manager.cache_blocks(
                request,
                self.req_to_num_tokens[request_id],
            )
        self._free_slots(request_id)
        logger.debug(
            f"delete request {request_id} in cpu_kv_cache_manager req_to_free")
        del self.req_to_free[request_id]

    def _free_slots(self, request_id: str):
        # This function is designed to be reentrant.
        self._release_ahead_touch(request_id)
        self.single_type_manager.free(request_id)
        self.req_to_block_hashes.pop(request_id, None)
        self.req_to_computed_blocks.pop(request_id, None)
        self.req_failed_to_allocate.pop(request_id, None)
        self.req_to_num_tokens.pop(request_id, None)
