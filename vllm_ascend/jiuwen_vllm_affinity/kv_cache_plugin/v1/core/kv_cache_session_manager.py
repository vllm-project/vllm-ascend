from collections.abc import Sequence

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)


class KvCacheSessionManager:
    """
    Docstring for KvCacheSessionManager
    """

    def __init__(self):
        self.block_to_sessions: list[list[str]] = []

    def add_blocks(self, blocks: Sequence[KVCacheBlock], session_id: str | None = None):
        logger.debug("add %s blocks with session %s", len(blocks), session_id)
        if session_id is None:
            return
        for blk in blocks:
            if blk.block_id < 0:
                continue
            block_num = len(self.block_to_sessions)
            if block_num <= blk.block_id:
                self.block_to_sessions.extend([[] for _ in range(blk.block_id - block_num + 1)])
            if session_id not in self.block_to_sessions[blk.block_id]:
                self.block_to_sessions[blk.block_id].append(session_id)

    def reset_blocks(self, blocks: list[KVCacheBlock], session_id: str | None = None):
        logger.debug("reset %s blocks with session %s", len(blocks), session_id)
        for blk in blocks:
            if blk.block_id < 0:
                continue
            block_num = len(self.block_to_sessions)
            if block_num <= blk.block_id:
                self.block_to_sessions.extend([[] for _ in range(blk.block_id - block_num + 1)])
            self.block_to_sessions[blk.block_id] = [session_id] if session_id else []

    def release_blocks(self, blocks: list[KVCacheBlock], session_id: str | None = None) -> list[KVCacheBlock]:
        logger.debug("release %s blocks with session %s", len(blocks), session_id)
        aging_blocks: list[KVCacheBlock] = []
        if session_id is None:
            return aging_blocks
        block_num = len(self.block_to_sessions)
        for blk in blocks:
            if blk.block_id < 0 or block_num <= blk.block_id:
                continue
            if session_id in self.block_to_sessions[blk.block_id]:
                self.block_to_sessions[blk.block_id].remove(session_id)
                if len(self.block_to_sessions[blk.block_id]) == 0:
                    aging_blocks.append(blk)

        return aging_blocks
