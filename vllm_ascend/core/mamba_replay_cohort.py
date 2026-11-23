# SPDX-License-Identifier: Apache-2.0
"""Keep reusable local hybrid prefix pages together in the eviction queue."""

from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock, get_block_hash, get_group_id

logger = init_logger(__name__)


class MambaReplayCohortPool(BlockPool):
    """Give a complete aligned prefix one extra opportunity for reuse."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mamba_page_sizes: dict[int, int] = {}
        self._replay_groups: list[int] = []
        self._deferred: set[int] = set()

    def enable_mamba_retention(self, mamba_page_sizes: dict[int, int], attention_groups: list[int]) -> None:
        self._mamba_page_sizes = mamba_page_sizes
        self._replay_groups = [*mamba_page_sizes, *attention_groups]

    def _replay_cohort(self, block: KVCacheBlock) -> list[KVCacheBlock]:
        key = block.block_hash
        position = block.block_hash_num_tokens
        if key is None or position is None or block.block_id in self._deferred:
            return []
        group_id = get_group_id(key)
        page = self._mamba_page_sizes.get(group_id)
        if page is None or position % page:
            return []
        blocks = self.get_cached_block(get_block_hash(key), self._replay_groups)
        if not blocks or all(candidate is not block for candidate in blocks):
            return []
        # Active members cannot be evicted and need no queue updates.
        free = [member for member in blocks if member.ref_cnt == 0]
        if any(member.block_id in self._deferred for member in free):
            return []
        return free

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        if not self._replay_groups or num_blocks <= 0:
            return super().get_new_blocks(num_blocks)
        available = self.get_num_free_blocks()
        if num_blocks >= available or available > self.num_gpu_blocks // 4:
            return super().get_new_blocks(num_blocks)

        queue = self.free_block_queue
        current = queue.fake_free_list_head.next_free_block
        victims: set[int] = set()
        deferred: dict[int, KVCacheBlock] = {}
        budget = min(available - num_blocks, len(self._replay_groups))
        # Select before moving any node so siblings later in the list do not
        # change the traversal, and already selected victims stay untouched.
        while len(victims) < num_blocks and len(deferred) < budget and current is not queue.fake_free_list_tail:
            assert current is not None
            block_id = current.block_id
            if block_id not in deferred:
                cohort = self._replay_cohort(current)
                members = {block.block_id: block for block in cohort}
                if members and len(deferred.keys() | members.keys()) <= budget and not victims.intersection(members):
                    deferred.update(members)
                else:
                    victims.add(block_id)
            current = current.next_free_block

        if deferred:
            blocks = list(deferred.values())
            for block in blocks:
                queue.remove(block)
            queue.append_n(blocks)
            self._deferred.update(deferred)
            logger.info_once("Deferred an aligned local PMU replay cohort")
        return super().get_new_blocks(num_blocks)

    def _remove_cached_block_hashes(self, block: KVCacheBlock) -> list[BlockHashWithGroupId]:
        self._deferred.discard(block.block_id)
        return super()._remove_cached_block_hashes(block)

    def reset_prefix_cache(self) -> bool:
        reset = super().reset_prefix_cache()
        if reset:
            self._deferred.clear()
        return reset
