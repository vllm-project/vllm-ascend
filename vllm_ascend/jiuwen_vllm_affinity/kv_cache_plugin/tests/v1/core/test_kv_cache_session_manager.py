from vllm.v1.core.kv_cache_utils import KVCacheBlock

from vllm_ascend.jiuwen_vllm_affinity.kv_cache_plugin.v1.core.kv_cache_session_manager import KvCacheSessionManager


def test_kv_cache_session_manager_basic():
    block_count = 3
    blocks: list[KVCacheBlock] = [KVCacheBlock(idx) for idx in range(block_count)]
    sessions = ["s1", "s2"]
    session_mgr = KvCacheSessionManager()
    session_mgr.add_blocks([], sessions[0])
    session_mgr.reset_blocks(blocks, sessions[0])
    released = session_mgr.release_blocks(blocks, sessions[0])
    assert len(released) == block_count


def test_kv_cache_session_manager():
    block_count = 3
    blocks: list[KVCacheBlock] = [KVCacheBlock(idx) for idx in range(block_count)]
    sessions = ["s1", "s2"]
    session_mgr = KvCacheSessionManager()

    session_mgr.add_blocks(blocks, sessions[0])
    released = session_mgr.release_blocks(blocks, sessions[0])
    assert len(released) == block_count

    session_mgr.add_blocks(blocks, sessions[0])
    session_mgr.add_blocks(blocks[:2], sessions[1])
    released = session_mgr.release_blocks(blocks, sessions[0])
    assert len(released) == 1
    assert released[0].block_id == 2
    released = session_mgr.release_blocks(blocks[:2], sessions[1])
    assert len(released) == 2
    released_block_ids = {b.block_id for b in released}
    assert released_block_ids == {0, 1}
    released = session_mgr.release_blocks(blocks, sessions[0])
    assert len(released) == 0
    released = session_mgr.release_blocks(blocks, sessions[1])
    assert len(released) == 0

    session_mgr.add_blocks(blocks, sessions[0])
    session_mgr.add_blocks(blocks, sessions[1])
    session_mgr.reset_blocks(blocks, sessions[0])
    released = session_mgr.release_blocks(blocks, sessions[0])
    assert len(released) == block_count

    session_mgr.add_blocks(blocks, sessions[0])
    session_mgr.add_blocks(blocks, sessions[1])
    released = session_mgr.release_blocks(blocks, sessions[0])
    assert len(released) == 0
    released = session_mgr.release_blocks(blocks, sessions[1])
    assert len(released) == block_count
