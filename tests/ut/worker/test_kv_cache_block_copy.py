import weakref

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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("reverse_views", [False, True])
@pytest.mark.parametrize("layout", ["narrow", "strided", "offset"])
@pytest.mark.parametrize(
    "block_pairs",
    [[], [(0, 2)], [(0, 2), (2, 0)], [(0, 2), (2, 3), (0, 1)], [(0, 0)]],
    ids=["empty", "one_way", "cycle", "chain_and_fanout", "self_copy"],
)
def test_copy_kv_cache_blocks_with_overlapping_views(dtype, reverse_views, layout, block_pairs):
    cache = torch.arange(32, dtype=dtype).view(4, 8)
    if layout == "narrow":
        views = [cache[:, :2], cache]
        copied_columns = list(range(8))
    elif layout == "strided":
        # Equal pointers and shapes do not imply equal byte coverage.
        views = [cache[:, :4], cache[:, ::2]]
        copied_columns = [0, 1, 2, 3, 4, 6]
    else:
        # Overlapping views can also start at different storage offsets.
        views = [cache[:, :4], cache[:, 2:6]]
        copied_columns = list(range(6))
    if reverse_views:
        views.reverse()
    before = cache.clone()
    expected = before.clone()
    for src, dst in block_pairs:
        expected[dst, copied_columns] = before[src, copied_columns]

    copy_kv_cache_blocks_inplace(
        views,
        cache.shape[0],
        [KVCacheBlockCopy(src_block_id=src, dst_block_id=dst) for src, dst in block_pairs],
    )

    # Check untouched blocks and padding as well as each destination payload.
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


@pytest.mark.parametrize("attention_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("reverse_views", [False, True])
def test_copy_kv_cache_blocks_with_overlaid_byte_and_attention_views(attention_dtype, reverse_views):
    num_blocks = 4
    page_bytes = 32
    storage_offset = 8
    payload_bytes = 16
    raw = torch.arange(storage_offset + num_blocks * page_bytes + 8, dtype=torch.uint8)
    state = torch.as_strided(raw, (num_blocks, 4), (page_bytes, 1), storage_offset)
    element_size = torch.empty((), dtype=attention_dtype).element_size()
    attention = torch.as_strided(
        raw.view(attention_dtype),
        (num_blocks, payload_bytes // element_size),
        (page_bytes // element_size, 1),
        storage_offset // element_size,
    )
    assert state.data_ptr() == attention.data_ptr()
    views = [state, attention]
    if reverse_views:
        views.reverse()
    before = raw.clone()
    expected = before.clone()
    block_pairs = [(0, 2), (2, 0), (0, 3)]
    for src, dst in block_pairs:
        src_start = storage_offset + src * page_bytes
        dst_start = storage_offset + dst * page_bytes
        expected[dst_start : dst_start + payload_bytes] = before[src_start : src_start + payload_bytes]

    copy_kv_cache_blocks_inplace(
        views,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=src, dst_block_id=dst) for src, dst in block_pairs],
    )

    # Cache copies must preserve bits, including FP8 storage and page padding.
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)


def test_copy_kv_cache_blocks_bounds_staging_for_disjoint_regions(monkeypatch):
    num_regions, num_blocks, page_elements = 8, 4, 8
    raw = torch.arange(num_regions * num_blocks * page_elements, dtype=torch.float32)
    regions = raw.view(num_regions, num_blocks, page_elements)
    views = [view for region in regions for view in (region[:, :2], region)]
    before = regions.clone()
    expected = before.clone()
    expected[:, 0] = before[:, 2]
    expected[:, 2] = before[:, 0]
    # Regions share one backing allocation but do not overlap. Only source
    # blocks for one region's two views should be live at a time.
    copies_per_region = 2
    staging_budget = copies_per_region * (2 + page_elements) * raw.element_size()
    snapshots = []
    stack = torch.stack

    def check_staging_budget(*args, **kwargs):
        result = stack(*args, **kwargs)
        snapshots.append(weakref.ref(result))
        live_bytes = sum(t.numel() * t.element_size() for ref in snapshots if (t := ref()) is not None)
        assert live_bytes <= staging_budget
        return result

    monkeypatch.setattr(torch, "stack", check_staging_budget)
    copy_kv_cache_blocks_inplace(
        views,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2), KVCacheBlockCopy(src_block_id=2, dst_block_id=0)],
    )

    torch.testing.assert_close(regions, expected, rtol=0, atol=0)


def test_copy_kv_cache_blocks_skips_none_entries():
    cache = torch.arange(12, dtype=torch.float32).view(4, 3)
    before = cache.clone()

    copy_kv_cache_blocks_inplace(
        [None, [None, cache]],
        4,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    torch.testing.assert_close(cache[2], before[0])


def test_copy_kv_cache_blocks_skips_empty_views():
    cache = torch.arange(32, dtype=torch.float32).view(4, 8)
    before = cache.clone()
    expected = before.clone()
    expected[2] = before[0]

    copy_kv_cache_blocks_inplace(
        [cache[:, :0], cache],
        cache.shape[0],
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


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
