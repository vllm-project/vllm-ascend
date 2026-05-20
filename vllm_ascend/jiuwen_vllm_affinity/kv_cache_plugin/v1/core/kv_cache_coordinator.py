from collections.abc import Sequence

from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import CrossAttentionManager


class KVCacheCoordinatorEx(KVCacheCoordinator):
    def aging_block(self, session_id, block_hashes) -> int:
        num = 0
        for manager in self.single_type_managers:
            num += manager.aging_block(session_id, block_hashes)
        return num

    def save_new_computed_blocks_with_session(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        session_id: str | None = None,
    ) -> None:
        for i, manager in enumerate(self.single_type_managers):
            manager.save_new_computed_blocks_with_session(request_id, new_computed_blocks[i], session_id)

    def allocate_new_blocks_with_session(
        self,
        request_id: str,
        num_tokens: int,
        num_encoder_tokens: int = 0,
        session_id: str | None = None,
    ) -> tuple[list[KVCacheBlock], ...]:
        """ """
        return tuple(
            manager.allocate_new_blocks_with_session(
                request_id,
                (num_encoder_tokens if isinstance(manager, CrossAttentionManager) else num_tokens),
                session_id,
            )
            for manager in self.single_type_managers
        )


def register_kv_cache_coordinator():
    KVCacheCoordinator.aging_block = KVCacheCoordinatorEx.aging_block
    KVCacheCoordinator.allocate_new_blocks_with_session = KVCacheCoordinatorEx.allocate_new_blocks_with_session
    KVCacheCoordinator.save_new_computed_blocks_with_session = (
        KVCacheCoordinatorEx.save_new_computed_blocks_with_session
    )
