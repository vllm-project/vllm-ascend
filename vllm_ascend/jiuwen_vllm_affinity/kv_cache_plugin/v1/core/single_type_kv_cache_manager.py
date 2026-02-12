from collections.abc import Sequence

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

from vllm_ascend.jiuwen_vllm_affinity.kv_cache_plugin.v1.core.kv_cache_session_manager import (
    KvCacheSessionManager,
)

logger = init_logger(__name__)


class SingleTypeKVCacheManagerEx(SingleTypeKVCacheManager):
    def __init__(
        self,
        kv_cache_spec,
        block_pool,
        kv_cache_group_id,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        super().__init__(kv_cache_spec, block_pool, kv_cache_group_id, dcp_world_size, pcp_world_size)
        self.kv_cache_session_manager = KvCacheSessionManager()

    def aging_block(self, session_id, block_hashes) -> int:
        aging_blocks = []
        for block_hash in block_hashes:
            if cached_block := self.block_pool.get_cached_block(block_hash, [self.kv_cache_group_id]):
                aging_blocks.append(cached_block[0])
            else:
                break
        aging_blocks = self.kv_cache_session_manager.release_blocks(aging_blocks, session_id)
        return self.block_pool.aging_block(aging_blocks)

    def save_new_computed_blocks_with_session(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        session_id: str | None,
    ) -> None:
        """ """
        if request_id not in self.num_cached_block and session_id is not None:
            self.kv_cache_session_manager.add_blocks(new_computed_blocks, session_id)
        self.save_new_computed_blocks(request_id, new_computed_blocks)

    def allocate_new_blocks_with_session(
        self, request_id: str, num_tokens: int, session_id: str | None
    ) -> list[KVCacheBlock]:
        """ """
        blocks = self.allocate_new_blocks(request_id, num_tokens)
        if len(blocks) > 0 and session_id is not None:
            self.kv_cache_session_manager.reset_blocks(blocks, session_id)
        logger.debug("new block cnt %s", len(blocks))
        return blocks


def replace_single_type_kv_cache_manager_init():
    origin_init = SingleTypeKVCacheManager.__init__

    def new_init(
        self,
        kv_cache_spec,
        block_pool,
        kv_cache_group_id,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        origin_init(
            self,
            kv_cache_spec,
            block_pool,
            kv_cache_group_id,
            dcp_world_size,
            pcp_world_size,
        )
        self.kv_cache_session_manager = KvCacheSessionManager()

    SingleTypeKVCacheManager.__init__ = new_init


def register_single_type_kv_cache_manager():
    replace_single_type_kv_cache_manager_init()
    SingleTypeKVCacheManager.aging_block = SingleTypeKVCacheManagerEx.aging_block
    SingleTypeKVCacheManager.allocate_new_blocks_with_session = (
        SingleTypeKVCacheManagerEx.allocate_new_blocks_with_session
    )
    SingleTypeKVCacheManager.save_new_computed_blocks_with_session = (
        SingleTypeKVCacheManagerEx.save_new_computed_blocks_with_session
    )
