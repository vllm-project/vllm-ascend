from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request

from vllm_ascend.jiuwen_vllm_affinity.kv_cache_plugin.v1.request import request_get_sharing_cache_salt


class KVCacheManagerEx(KVCacheManager):
    def release_kv_cache(self, session_id: str, block_hashes: list[BlockHash]) -> int:
        return self.coordinator.aging_block(session_id, block_hashes)

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append."""
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.coordinator.remove_skipped_blocks(request.request_id, request.num_computed_tokens)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len,
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not any(new_computed_block_list), "Computed blocks should be empty when prefix caching is disabled"

        if new_computed_block_list is not self.empty_kv_cache_blocks.blocks:
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.save_new_computed_blocks_with_session(
                request.request_id,
                new_computed_block_list,
                request_get_sharing_cache_salt(request),
            )

        new_blocks = self.coordinator.allocate_new_blocks_with_session(
            request.request_id,
            num_tokens_need_slot,
            num_encoder_tokens,
            request_get_sharing_cache_salt(request),
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_computed_tokens +
        # num_new_tokens, but must exclude "non-committable" tokens (e.g.,
        # draft tokens that could be rejected). Therefore, we cap the number
        # at `request.num_tokens`, ensuring only "finalized" tokens are cached.
        num_tokens_to_cache = min(num_computed_tokens + num_new_tokens, request.num_tokens)
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)


def register_kv_cache_manager():
    KVCacheManager.allocate_slots = KVCacheManagerEx.allocate_slots
    KVCacheManager.release_kv_cache = KVCacheManagerEx.release_kv_cache
