# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KV block zeroer tests.

The first section is migrated from vLLM's tests/v1/worker/test_kv_block_zeroer.py.
Keep it aligned with upstream except for NPU APIs, Ascend's separate K/V tensors and
one-dimensional launch. Ascend-specific coverage follows it.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker import utils as worker_utils
from vllm_ascend.worker.utils import AscendKVBlockZeroer


class _BlockFirstBackend:
    @staticmethod
    def get_kv_cache_block_dim(*args, **kwargs):
        return 0


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
@pytest.mark.parametrize(
    "spec",
    [
        SlidingWindowSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            sliding_window=4,
        ),
        ChunkedLocalAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            attention_chunk_size=4,
        ),
    ],
    ids=["sliding-window", "chunked-local"],
)
def test_attention_blocks_are_zeroed(spec):
    device = torch.device("npu")
    # Ascend stores K and V in separate tensors instead of one KV tensor.
    storages = (
        torch.ones((4, 1, 2, 2), dtype=torch.uint8, device=device),
        torch.ones((4, 1, 2, 2), dtype=torch.uint8, device=device),
    )
    layer_name = "draft.self_attn"
    zeroer = AscendKVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(_BlockFirstBackend, [layer_name], spec, 0)  # type: ignore[arg-type]
        ],
        # Ascend receives one kernel block-size list per KV cache group.
        kernel_block_sizes=[[2]],
        cache_dtype="fp8",
        static_forward_context={
            layer_name: SimpleNamespace(kv_cache=storages),
        },
    )

    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()

    for storage in storages:
        expected = torch.ones_like(storage)
        expected[1] = 0
        assert torch.equal(storage, expected)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device("npu")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    # Build the minimal zeroer state directly so the test can focus on the
    # in-flight copy behavior without constructing model attention groups.
    zeroer = AscendKVBlockZeroer.__new__(AscendKVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        page_size_el // page_size_el,  # max_chunks = 1
        page_size_el,  # blk_size
        1,  # n_segs
    )

    # Ascend has no torch.cuda._sleep equivalent. Compile first, then use an NPU
    # stream/event dependency to keep both nonblocking copies in flight.
    zeroer.zero_block_ids([0])
    torch.accelerator.synchronize()
    storage.fill_(1)
    torch.accelerator.synchronize()

    blocker_stream = torch.npu.Stream()
    stream = torch.npu.Stream()
    with torch.npu.stream(blocker_stream):
        blocker = torch.full((2048, 2048), 1 / 2048, dtype=torch.float16, device=device)
        blocker_result = torch.mm(blocker, blocker)
        for _ in range(15):
            blocker_result = torch.mm(blocker_result, blocker)
        blocker_done = blocker_stream.record_event()

    with torch.npu.stream(stream):
        stream.wait_event(blocker_done)
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. Each call must stage from its own pinned source so the
        # first copy is not corrupted before it runs.
        zeroer.zero_block_ids([1])
        zeroer.zero_block_ids([2])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
def test_non_uniform_page_sizes():
    """Two segments with different page sizes (e.g. MLA + DSA indexer)."""
    device = torch.device("npu")
    num_blocks = 4
    page_size_a = 10496  # int32 elements
    page_size_b = 2112

    storage_a = torch.ones((num_blocks, page_size_a), dtype=torch.int32, device=device)
    storage_b = torch.ones((num_blocks, page_size_b), dtype=torch.int32, device=device)

    zeroer = AscendKVBlockZeroer.__new__(AscendKVBlockZeroer)
    zeroer.device = device

    seg_page_sizes = [page_size_a, page_size_b]
    max_ps = max(seg_page_sizes)

    blk_size = min(1 << (max_ps - 1).bit_length(), 1024)

    zeroer._meta = (
        torch.tensor(
            [storage_a.data_ptr(), storage_b.data_ptr()],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        (max_ps + blk_size - 1) // blk_size,
        blk_size,
        2,
    )

    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        zeroer.zero_block_ids([1, 2])
    stream.synchronize()

    for storage in (storage_a, storage_b):
        assert torch.all(storage[0] == 1)
        assert torch.all(storage[1] == 0)
        assert torch.all(storage[2] == 0)
        assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
def test_packed_segment_zeros_only_its_last_block_page():
    """A packed KV segment steps by block stride but clears only its page."""
    device = torch.device("npu")
    num_blocks = 4
    block_stride_el = 12
    page_size_el = 4
    page_offset_el = 3
    backing = torch.ones((num_blocks, block_stride_el), dtype=torch.int32, device=device)

    zeroer = AscendKVBlockZeroer.__new__(AscendKVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor(
            [backing.data_ptr() + page_offset_el * backing.element_size()],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor([block_stride_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,
        page_size_el,
        1,
    )

    zeroer.zero_block_ids([num_blocks - 1])
    torch.accelerator.synchronize()

    expected = torch.ones_like(backing)
    expected[-1, page_offset_el : page_offset_el + page_size_el] = 0
    assert torch.equal(backing, expected)


def test_large_dsv4_launch_geometry(monkeypatch):
    """Keep the failing DSV4 shape efficient and within launch limits."""
    device = torch.device("cpu")
    n_blocks, n_segs = 6870, 181
    layer_names = [f"layer.{i}" for i in range(n_segs)]
    page_sizes = [9344 if i % 2 == 0 else 292 for i in range(n_segs)]
    spec = SlidingWindowSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int32,
        sliding_window=1,
    )
    storages = {name: torch.ones((1, page_size), dtype=torch.int32) for name, page_size in zip(layer_names, page_sizes)}
    zeroer = AscendKVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(
                _BlockFirstBackend,  # type: ignore[arg-type]
                [name],
                spec,
                group_id,
            )
            for group_id, name in enumerate(layer_names)
        ],
        # Ascend receives one kernel block-size list per KV cache group.
        kernel_block_sizes=[[1] for _ in range(n_segs)],
        cache_dtype="auto",
        static_forward_context={
            # Reuse one tensor as K/V so pointer deduplication keeps the
            # upstream test's 181-segment geometry.
            name: SimpleNamespace(kv_cache=(storage, storage))
            for name, storage in storages.items()
        },
    )

    assert zeroer._meta is not None
    _, _, seg_page_sizes, max_chunks, blk_size, n_segs = zeroer._meta
    assert seg_page_sizes.tolist() == page_sizes
    assert (max_chunks, blk_size, n_segs) == (10, 1024, 181)

    captured_grids = []

    class FakeKernel:
        def __getitem__(self, grid):
            captured_grids.append(grid)
            return lambda *args, **kwargs: None

    vectorcore_num = 48
    monkeypatch.setattr(worker_utils, "_zero_kv_blocks_kernel", FakeKernel())
    monkeypatch.setattr(
        worker_utils,
        "async_tensor_h2d",
        lambda values, **kwargs: torch.tensor(values, dtype=torch.int64),
    )
    # Ascend uses a fixed one-dimensional persistent grid.
    monkeypatch.setattr(worker_utils, "get_vectorcore_num", lambda: vectorcore_num)

    zeroer.zero_block_ids(list(range(n_blocks)))

    old_max_chunks = max(page_sizes) // 4
    assert math.prod((n_blocks, n_segs, old_max_chunks)) > 2**31 - 1
    assert captured_grids == [(vectorcore_num,)]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
def test_warmup_respects_available_block_count():
    """An empty KV cache must not be warmed with out-of-range block IDs."""
    device = torch.device("npu")
    page_size_el = 4
    storage = torch.ones((1, page_size_el), dtype=torch.int32, device=device)

    zeroer = AscendKVBlockZeroer.__new__(AscendKVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,
        page_size_el,
        1,
    )

    zeroer.warmup(0)
    torch.accelerator.synchronize()

    assert torch.all(storage == 1)


# -----------------------------------------------------------------------------
# vLLM-Ascend-specific coverage
# -----------------------------------------------------------------------------


def _torch_zero_kv_blocks_golden(
    storages: list[torch.Tensor],
    block_ids: list[int],
    page_sizes: list[int],
) -> list[torch.Tensor]:
    """Build expected zeroed pages with ordinary Torch indexing."""
    outputs = [storage.clone() for storage in storages]
    for output, page_size in zip(outputs, page_sizes):
        for block_id in block_ids:
            output[block_id, :page_size] = 0
    return outputs


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
@pytest.mark.parametrize(
    ("num_blocks", "page_sizes", "paddings", "block_ids"),
    [
        (1, [1], [0], [0]),
        (3, [1024], [0], [1]),
        (3, [1025], [7], [2]),
        (4, [9344, 292, 1], [3, 4, 7], [0, 1, 2, 3]),
        (130, [33, 7], [2, 5], [129, 0, 64, 3]),
    ],
    ids=[
        "minimum-page-and-block",
        "maximum-single-chunk-page",
        "first-multi-chunk-page-with-tail",
        "heterogeneous-segments-with-many-masked-chunks",
        "first-middle-last-blocks-in-nonmonotonic-order",
    ],
)
def test_zero_kv_blocks_matches_torch_golden(num_blocks, page_sizes, paddings, block_ids):
    """Compare the Triton kernel with a Torch golden across boundary layouts."""
    device = torch.device("npu")
    storages = []
    for page_size, padding in zip(page_sizes, paddings):
        block_stride = page_size + padding
        storage = torch.arange(
            1,
            num_blocks * block_stride + 1,
            dtype=torch.int32,
            device=device,
        ).reshape(num_blocks, block_stride)
        storages.append(storage)

    block_strides = [storage.stride(0) for storage in storages]
    expected = _torch_zero_kv_blocks_golden(storages, block_ids, page_sizes)
    max_page_size = max(page_sizes)
    block_size = min(1 << (max_page_size - 1).bit_length(), 1024)

    zeroer = AscendKVBlockZeroer.__new__(AscendKVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor(
            [storage.data_ptr() for storage in storages],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor(block_strides, dtype=torch.int64, device=device),
        torch.tensor(page_sizes, dtype=torch.int64, device=device),
        (max_page_size + block_size - 1) // block_size,
        block_size,
        len(storages),
    )

    zeroer.zero_block_ids(block_ids)
    torch.accelerator.synchronize()

    for storage, expected_storage in zip(storages, expected):
        assert torch.equal(storage, expected_storage)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
def test_init_adapts_kv_tuple_and_virtual_kernel_blocks():
    """Validate fake layers for separate K/V and virtual block splitting."""
    device = torch.device("npu")
    num_logical_blocks = 3
    logical_block_size = 4
    kernel_block_size = 2
    ratio = logical_block_size // kernel_block_size
    num_kernel_blocks = num_logical_blocks * ratio
    layer_name = "model.layers.0.self_attn"
    spec = FullAttentionSpec(
        block_size=logical_block_size,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.bfloat16,
    )
    storages = (
        torch.ones(
            (num_kernel_blocks, kernel_block_size, 4),
            dtype=torch.bfloat16,
            device=device,
        ),
        torch.ones(
            (num_kernel_blocks, kernel_block_size, 2),
            dtype=torch.bfloat16,
            device=device,
        ),
    )
    zeroer = AscendKVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(_BlockFirstBackend, [layer_name], spec, 0)  # type: ignore[arg-type]
        ],
        kernel_block_sizes=[[kernel_block_size]],
        cache_dtype="auto",
        static_forward_context={
            layer_name: SimpleNamespace(kv_cache=storages),
        },
    )

    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()

    for storage in storages:
        expected = torch.ones_like(storage)
        expected[ratio : 2 * ratio] = 0
        assert torch.equal(storage, expected)
