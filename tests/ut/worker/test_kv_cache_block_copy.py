import pytest
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


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.bfloat16, torch.float32])
def test_copy_kv_cache_blocks_preserves_padded_rows_and_storage_offset(dtype):
    num_blocks = 5
    page_stride = 11
    storage_offset = 7
    elements_per_block = 6
    raw_storage = torch.arange(storage_offset + num_blocks * page_stride + 9, dtype=torch.int32).to(dtype)
    cache = torch.as_strided(
        raw_storage,
        size=(num_blocks, 2, 3),
        stride=(page_stride, 3, 1),
        storage_offset=storage_offset,
    )
    assert not cache.is_contiguous()
    assert cache[0].is_contiguous()
    before = raw_storage.clone()
    copies = [
        KVCacheBlockCopy(src_block_id=1, dst_block_id=2),
        KVCacheBlockCopy(src_block_id=2, dst_block_id=1),
        KVCacheBlockCopy(src_block_id=1, dst_block_id=3),
        KVCacheBlockCopy(src_block_id=4, dst_block_id=4),
    ]
    expected = before.clone()
    for copy in copies:
        src_start = storage_offset + copy.src_block_id * page_stride
        dst_start = storage_offset + copy.dst_block_id * page_stride
        expected[dst_start : dst_start + elements_per_block] = before[src_start : src_start + elements_per_block]

    copy_kv_cache_blocks_inplace([cache], num_blocks, copies)

    torch.testing.assert_close(raw_storage, expected, rtol=0, atol=0)


def test_copy_kv_cache_blocks_avoids_full_cache_staging(monkeypatch):
    raw_storage = torch.arange(32, dtype=torch.float32)
    cache = raw_storage.view(4, 8)[:, 2:5]
    before = raw_storage.clone()
    expected = before.clone().view(4, 8)
    expected[1, 2:5] = before.view(4, 8)[2, 2:5]
    expected[2, 2:5] = before.view(4, 8)[1, 2:5]

    def fail_full_cache_staging(*args, **kwargs):
        raise RuntimeError("Simulated NPU full-cache staging exceeds available memory")

    monkeypatch.setattr(torch, "index_select", fail_full_cache_staging)
    monkeypatch.setattr(torch.Tensor, "index_copy_", fail_full_cache_staging)
    copy_kv_cache_blocks_inplace(
        [cache, cache],
        4,
        [
            KVCacheBlockCopy(src_block_id=1, dst_block_id=2),
            KVCacheBlockCopy(src_block_id=2, dst_block_id=1),
        ],
    )

    torch.testing.assert_close(raw_storage.view(4, 8), expected, rtol=0, atol=0)


def test_copy_kv_cache_blocks_copies_strided_kernel_blocks_per_page():
    num_blocks = 4
    kernel_blocks_per_page = 3
    elements_per_kernel_block = 6
    row_stride = 2 * elements_per_kernel_block + 5
    storage_offset = 7
    num_kernel_blocks = num_blocks * kernel_blocks_per_page
    raw_storage = torch.arange(storage_offset + num_kernel_blocks * row_stride + 9, dtype=torch.float32)
    k_cache = torch.as_strided(
        raw_storage,
        size=(num_kernel_blocks, 2, 3),
        stride=(row_stride, 3, 1),
        storage_offset=storage_offset,
    )
    v_cache = torch.as_strided(
        raw_storage,
        size=(num_kernel_blocks, 2, 3),
        stride=(row_stride, 3, 1),
        storage_offset=storage_offset + elements_per_kernel_block,
    ).transpose(1, 2)
    assert not k_cache.is_contiguous()
    assert not v_cache[0].is_contiguous()
    before = raw_storage.clone()
    copies = [
        KVCacheBlockCopy(src_block_id=1, dst_block_id=2),
        KVCacheBlockCopy(src_block_id=2, dst_block_id=1),
        KVCacheBlockCopy(src_block_id=1, dst_block_id=3),
    ]
    expected = before.clone()
    for copy in copies:
        for kernel_index in range(kernel_blocks_per_page):
            src_kernel = copy.src_block_id * kernel_blocks_per_page + kernel_index
            dst_kernel = copy.dst_block_id * kernel_blocks_per_page + kernel_index
            src_start = storage_offset + src_kernel * row_stride
            dst_start = storage_offset + dst_kernel * row_stride
            payload_size = 2 * elements_per_kernel_block
            expected[dst_start : dst_start + payload_size] = before[src_start : src_start + payload_size]

    copy_kv_cache_blocks_inplace([k_cache, v_cache], num_blocks, copies)

    torch.testing.assert_close(raw_storage, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_blocks", [1, 2, 4])
def test_copy_kv_cache_blocks_copies_combined_kv_with_leading_pair_axis(num_blocks):
    kernel_blocks_per_page = 3
    elements_per_kernel_block = 6
    row_stride = 2 * elements_per_kernel_block + 5
    storage_offset = 7
    num_kernel_blocks = num_blocks * kernel_blocks_per_page
    raw_storage = torch.arange(storage_offset + num_kernel_blocks * row_stride + 9, dtype=torch.float32)
    cache = torch.as_strided(
        raw_storage,
        size=(2, num_kernel_blocks, 2, 1, 3),
        stride=(elements_per_kernel_block, row_stride, 3, 3, 1),
        storage_offset=storage_offset,
    )
    cyclic_copies = (
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=0)]
        if num_blocks == 1
        else [
            KVCacheBlockCopy(src_block_id=0, dst_block_id=1),
            KVCacheBlockCopy(src_block_id=1, dst_block_id=0),
        ]
    )
    fanout_copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=dst) for dst in range(num_blocks)]
    for copies in (cyclic_copies, fanout_copies):
        before = raw_storage.clone()
        expected = before.clone()
        for copy in copies:
            for kernel_index in range(kernel_blocks_per_page):
                src_kernel = copy.src_block_id * kernel_blocks_per_page + kernel_index
                dst_kernel = copy.dst_block_id * kernel_blocks_per_page + kernel_index
                src_start = storage_offset + src_kernel * row_stride
                dst_start = storage_offset + dst_kernel * row_stride
                payload_size = 2 * elements_per_kernel_block
                expected[dst_start : dst_start + payload_size] = before[src_start : src_start + payload_size]

        copy_kv_cache_blocks_inplace([cache], num_blocks, copies)

        torch.testing.assert_close(raw_storage, expected, rtol=0, atol=0)
