from vllm.v1.core.kv_cache_utils import KVCacheBlock

from vllm_ascend.jiuwen_vllm_affinity.kv_cache_plugin.v1.core.two_phase_block_queue import TwoPhaseBlockQueue


def create_two_phase_queue(queue_len: int) -> TwoPhaseBlockQueue:
    blocks: list[KVCacheBlock] = [KVCacheBlock[idx] for idx in range(queue_len)]
    return TwoPhaseBlockQueue(blocks)


def test_two_phase_queue():
    block_count = 6
    block_queue = create_two_phase_queue(block_count)
    blocks = block_queue.get_all_free_blocks()
    for idx in range(block_count // 2):
        blk = block_queue.popleft()
        assert blk.block_id == idx
        block_queue.append(blk)
    for idx in range(0, block_count, 2):
        block_queue.aging_block(blocks[idx])
        assert blocks[idx].block_id == idx
    block_ids = [3, 5, 0, 2, 4, 1]
    for idx in range(block_count):
        blk = block_queue.popleft()
        assert blk.block_id == block_ids[idx]
    blocks = block_queue.get_all_free_blocks()
    assert len(blocks) == 0
