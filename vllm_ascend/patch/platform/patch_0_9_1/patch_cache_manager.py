from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import Request


def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
    """Cache the blocks for the request, if enabled."""
    if self.enable_caching:
        block_hashes = self.req_to_block_hashes[request.request_id]
        self.coordinator.cache_blocks(request, block_hashes,
                                      num_computed_tokens)


KVCacheManager.cache_blocks = cache_blocks
