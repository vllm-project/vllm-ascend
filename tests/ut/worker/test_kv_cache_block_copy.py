import torch
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy

from vllm_ascend.worker.utils import copy_kv_cache_blocks_inplace


def test_copy_kv_cache_blocks_with_segmented_storage():
    """Each state segment must use its own leading block dimension."""
    num_blocks = 4
    conv_elements_per_block = 2
    ssm_elements_per_block = 3
    raw_storage = torch.arange(
        num_blocks * (conv_elements_per_block + ssm_elements_per_block),
        dtype=torch.float32,
    )
    conv_end = num_blocks * conv_elements_per_block
    conv_state = raw_storage[:conv_end].view(num_blocks, conv_elements_per_block)
    ssm_state = raw_storage[conv_end:].view(num_blocks, ssm_elements_per_block)
    conv_before = conv_state.clone()
    ssm_before = ssm_state.clone()

    copy_kv_cache_blocks_inplace(
        [[conv_state, ssm_state]],
        num_blocks,
        [KVCacheBlockCopy(src_block_id=1, dst_block_id=3)],
    )

    torch.testing.assert_close(conv_state[3], conv_before[1])
    torch.testing.assert_close(ssm_state[3], ssm_before[1])
    torch.testing.assert_close(conv_state[:3], conv_before[:3])
    torch.testing.assert_close(ssm_state[:3], ssm_before[:3])


def test_copy_kv_cache_blocks_deduplicates_shared_views():
    cache = torch.arange(12, dtype=torch.float32).view(4, 3)
    before = cache.clone()

    copy_kv_cache_blocks_inplace(
        [cache, cache],
        4,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    torch.testing.assert_close(cache[2], before[0])


def test_copy_kv_cache_blocks_skips_none_entries():
    cache = torch.arange(12, dtype=torch.float32).view(4, 3)
    before = cache.clone()

    copy_kv_cache_blocks_inplace(
        [None, [None, cache]],
        4,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    torch.testing.assert_close(cache[2], before[0])


def test_copy_kv_cache_blocks_copies_complete_physical_pages():
    """A physical page can contain multiple kernel-level cache blocks."""
    num_blocks = 4
    kernel_blocks_per_page = 3
    cache = torch.arange(
        num_blocks * kernel_blocks_per_page * 2,
        dtype=torch.float32,
    ).view(num_blocks * kernel_blocks_per_page, 2)
    before = cache.clone().view(num_blocks, -1)

    copy_kv_cache_blocks_inplace(
        [cache],
        num_blocks,
        [KVCacheBlockCopy(src_block_id=1, dst_block_id=3)],
    )

    after = cache.view(num_blocks, -1)
    torch.testing.assert_close(after[3], before[1])
    torch.testing.assert_close(after[:3], before[:3])


def test_copy_kv_cache_blocks_snapshots_cyclic_sources():
    cache = torch.arange(12, dtype=torch.float32).view(4, 3)
    before = cache.clone()

    copy_kv_cache_blocks_inplace(
        [cache],
        4,
        [
            KVCacheBlockCopy(src_block_id=1, dst_block_id=2),
            KVCacheBlockCopy(src_block_id=2, dst_block_id=1),
        ],
    )

    torch.testing.assert_close(cache[1], before[2])
    torch.testing.assert_close(cache[2], before[1])
