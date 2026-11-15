# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse Triton transfer, layout and invalidation accuracy tests."""

from contextlib import contextmanager

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import is_950
from vllm_ascend.hisparse.backend import AscendHiSparseBackend
from vllm_ascend.hisparse.cache_ops import (
    HiSparseNPUOps,
    pack_attention_rows,
    scatter_packed_kv,
)
from vllm_ascend.hisparse.host_pool import allocate_mapped_host

pytestmark = pytest.mark.skipif(
    not is_950(),
    reason="HiSparse Triton tests require Ascend 950 (A5)",
)


@pytest.mark.parametrize("capture", [False, True])
def test_c8_bytes_reuse_existing_scatter_with_padding_and_page_gaps(capture):
    rows, pages, block, width = 5, 4, 128, 656
    raw = torch.full((pages, 2, block, width), 165, dtype=torch.uint8, device="npu")
    cache = raw[:, 1]
    values = torch.arange(rows * width, dtype=torch.int64).to(torch.uint8).reshape(rows, width).npu()
    slots_cpu = torch.tensor([257, -1, 0, 128, 511], dtype=torch.int64)
    slots = slots_cpu.npu()
    valid = torch.tensor([4], dtype=torch.int32, device="npu")

    def forward():
        scatter_packed_kv(values, values[:, :0], cache, slots, valid)

    graph = None
    try:
        forward()
        torch.npu.synchronize()
        if capture:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                forward()
        for count in (4, 2):
            raw.fill_(165)
            values.bitwise_xor_(1)
            valid.fill_(count)
            if graph is None:
                forward()
            else:
                graph.replay()
            torch.npu.synchronize()
            expected = torch.full_like(raw, 165).cpu()
            payload = values.cpu()
            for row, slot in enumerate(slots_cpu.tolist()):
                if row < count and slot >= 0:
                    expected[slot // block, 1, slot % block] = payload[row]
            torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()


@pytest.mark.parametrize("rows,topk,row_bytes", [(1, 17, 656), (4, 2048, 656), (16, 2048, 1152)])
@pytest.mark.parametrize("capture", [False, True])
def test_compact_refill_tiles_current_counts_and_preserves_page_gaps(rows, topk, row_bytes, capture):
    block = 128
    blocks = (rows * topk + block - 1) // block + 1
    host = allocate_mapped_host((blocks * block, row_bytes), torch.uint8)
    mapping = host._hisparse_registration
    host.copy_((torch.arange(host.shape[0], dtype=torch.int32) % 251).to(torch.uint8)[:, None].expand_as(host))
    backing = torch.full((blocks, 2, block, row_bytes), 203, dtype=torch.uint8, device="npu")
    hot = backing[:, 1]
    ops = HiSparseNPUOps(mapping)
    original_source = torch.arange(rows * topk, dtype=torch.int32).view(rows, topk)
    original_destination = original_source + block
    original_source[:, 4] = -1
    original_source[:, 6] = host.shape[0] + 7
    original_destination[:, 2] = -1
    original_destination[:, 5] = blocks * block + 3
    source, destination = original_source.to("npu"), original_destination.to("npu")
    counts = torch.zeros(rows, dtype=torch.int32, device="npu")

    def forward():
        ops.hisparse_gather_compact(host, hot, source, destination, counts)

    graph = None
    try:
        forward()
        torch.npu.synchronize()
        if capture:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                forward()
        for count in (0, 1, 7, 8, 9, 15, 16, 17, topk, 0):
            backing.fill_(203)
            current = torch.full((rows,), count, dtype=torch.int32)
            if rows > 1:
                current[1] = -3
            if rows > 2 and count == topk:
                current[2] = topk + 3
            counts.copy_(current)
            source.copy_(original_source if count else torch.full_like(original_source, -1))
            destination.copy_(original_destination if count else torch.full_like(original_destination, -1))
            if graph is None:
                forward()
            else:
                graph.replay()
            torch.npu.synchronize()
            expected = torch.full(backing.shape, 203, dtype=torch.uint8)
            for row in range(rows):
                length = min(topk, max(0, int(current[row])))
                src = original_source[row, :length].long()
                dst = original_destination[row, :length].long()
                valid = (src >= 0) & (dst >= 0) & (dst < blocks * block)
                payload = torch.zeros((length, row_bytes), dtype=torch.uint8)
                from_host = valid & (src < host.shape[0])
                payload[from_host] = host[src[from_host]]
                expected[dst[valid] // block, 1, dst[valid] % block] = payload[valid]
            torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        mapping.close()


def test_plan_gather_respects_request_state_padding():
    host = allocate_mapped_host((32, 16), torch.float32)
    mapping = host._hisparse_registration
    host.copy_(torch.arange(host.numel(), dtype=torch.float32).view_as(host))
    hot = torch.full((8, 4, 16), -17.0, device="npu")
    ops = HiSparseNPUOps(mapping)
    source = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]], device="npu", dtype=torch.int32)
    destination = source + 4
    mask = torch.ones_like(source, dtype=torch.int32)
    state = torch.tensor([0, -1], device="npu", dtype=torch.int32)
    attention = torch.empty_like(source)
    try:
        ops.hisparse_gather_plan(host, hot, source, destination, mask, state, attention, 8)
        torch.npu.synchronize()
        cache = hot.cpu().view(32, 16)
        torch.testing.assert_close(cache[5:13], host[1:9])
        torch.testing.assert_close(attention[1].cpu(), torch.full((8,), -1, dtype=torch.int32))
    finally:
        torch.npu.synchronize()
        mapping.close()


@contextmanager
def _invalidation_graph(forward):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        yield graph, stream
    finally:
        torch.npu.synchronize()
        graph.reset()


@pytest.mark.parametrize("hot_capacity", [4, 10240])
def test_invalidate_hot_capacity_graph(hot_capacity):
    dgi = torch.arange(2 * hot_capacity, dtype=torch.int32, device="npu").view(2, hot_capacity)
    group_table = torch.tensor([dgi.data_ptr()], dtype=torch.int64, device=dgi.device)
    initial = dgi.cpu()
    states = torch.tensor([1, 0], dtype=torch.int32, device="npu")
    requests = torch.tensor([0, 1, 1, -1], dtype=torch.int32, device="npu")
    written = torch.tensor([2 * hot_capacity - 1, 0, hot_capacity - 1, hot_capacity], dtype=torch.int64, device="npu")
    expected = initial.clone()
    expected[1, -1] = expected[0, 0] = expected[0, -1] = -1

    def forward():
        HiSparseNPUOps.hisparse_invalidate_written_slots(group_table, states, requests, written, *dgi.shape)

    forward()
    torch.npu.synchronize()
    torch.testing.assert_close(dgi.cpu(), expected)
    with _invalidation_graph(forward) as (graph, stream):
        with torch.npu.graph(graph, stream=stream):
            forward()
        for _ in range(10):
            dgi.copy_(initial)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(dgi.cpu(), expected)


@pytest.mark.parametrize(
    "rows,hot_capacity,cross_stream,expect_sorted",
    [(4, 16, False, False), (1537, 8193, False, True), (1537, 8193, True, True)],
)
def test_common_group_invalidation_changed_graph_replay(rows, hot_capacity, cross_stream, expect_sorted, monkeypatch):
    """Exercise both invalidation routes with changing writes in eager and Graph."""
    from types import SimpleNamespace

    from vllm.v1.hisparse.runtime import HiSparseSlotInvalidator

    groups = 3
    initial = torch.arange(3 * hot_capacity, dtype=torch.int32).view(3, hot_capacity)
    tables = [(initial + group * 7).npu() for group in range(groups)]
    states = torch.tensor([2, 0, 1], dtype=torch.int32, device="npu")
    requests = torch.full((rows,), -1, dtype=torch.int32, device="npu")
    requests[:3] = torch.arange(3, dtype=requests.dtype, device=requests.device)
    written = torch.zeros(rows, dtype=torch.int64, device="npu")
    leaders = [
        SimpleNamespace(index_group=SimpleNamespace(device_global_indices=table), cache_ops=HiSparseNPUOps)
        for table in tables
    ]
    invalidator = HiSparseSlotInvalidator(states)
    consumer = torch.Stream(device="npu") if cross_stream else None
    ready = torch.Event() if cross_stream else None

    def forward():
        if consumer is None:
            invalidator(leaders, written, requests)
            return
        current = torch.accelerator.current_stream()
        consumer.wait_stream(current)
        with consumer:
            invalidator(leaders, written, requests)
            ready.record(consumer)
        current.wait_event(ready)

    def check_writes(run):
        for values in (
            [2 * hot_capacity + 3, 4, hot_capacity + 3],
            [3 * hot_capacity - 1, 8, hot_capacity + 7],
            [-1] * 3,
        ):
            written[:3].copy_(torch.tensor(values, dtype=written.dtype))
            for group, table in enumerate(tables):
                table.copy_(initial + group * 7)
            run()
            torch.npu.synchronize()
            for group, table in enumerate(tables):
                expected = initial + group * 7
                for row, value in zip([2, 0, 1], values):
                    expected[row][expected[row] == value] = -1
                torch.testing.assert_close(table.cpu(), expected)

    check_writes(forward)

    sort = torch.sort
    sort_calls = 0

    def count_sort(*args, **kwargs):
        nonlocal sort_calls
        sort_calls += 1
        return sort(*args, **kwargs)

    with _invalidation_graph(forward) as (graph, stream):
        monkeypatch.setattr(torch, "sort", count_sort)
        with torch.npu.graph(graph, stream=stream):
            forward()
        assert sort_calls == int(expect_sorted)
        check_writes(graph.replay)


@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("capture", [False, True])
def test_page_strided_indexer_write_preserves_bits_and_padding(capture, slot_dtype):
    from types import SimpleNamespace

    from vllm.v1.kv_cache_interface import KVCacheTensor, create_kv_cache_views
    from vllm.v1.kv_cache_layout import KVCacheLayout

    from vllm_ascend.attention.indexer import AscendHiSparseIndexerBackend
    from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
    from vllm_ascend.hisparse.binding import _quantized_indexer_views

    pages, block, width, page_stride = 3, 128, 128, 419840
    size = (pages - 1) * page_stride + block * width + block * 4
    raw = torch.full((size,), 90, dtype=torch.uint8, device="npu")
    spec = AscendSFAIndexerCacheSpec(
        block_size=block,
        num_kv_heads=1,
        head_size=width,
        dtype=torch.float8_e4m3fn,
        cache_dtype_str="fp8",
        scale_dim=1,
        scale_dtype=torch.float32,
        cache_sparse_li_c8=True,
        cache_sparse_li_c4=False,
        sfa_dcp_replicated_indexer_size=1,
    )
    tensor_config = KVCacheTensor(
        size=size, layers=["indexer"], layer_stride=spec.page_size_bytes, block_stride=page_stride
    )
    (packed,) = create_kv_cache_views(raw.view(torch.int8), spec, pages, KVCacheLayout.BLNHC, tensor_config)
    key, scale = _quantized_indexer_views(packed, spec)
    writer = AscendHiSparseIndexerBackend.__new__(AscendHiSparseIndexerBackend)
    torch.nn.Module.__init__(writer)
    writer.k_cache = SimpleNamespace(kv_cache=(key, scale))
    writer.enable_sparse_li_c8 = True
    writer.enable_sparse_li_c4 = False
    values = torch.zeros((5, width), dtype=torch.uint8, device="npu").view(torch.float8_e4m3fn)
    scales = torch.zeros(5, dtype=torch.float32, device="npu")
    slots = torch.full((5,), -1, dtype=slot_dtype, device="npu")

    def write():
        writer.write_cache(values, scales, slots)

    write()
    torch.npu.synchronize()
    graph = None
    if capture:
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            write()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            write()
    try:
        for iteration, logical in enumerate([[-1, 0, 127, 128, 383], [0, 129, -1, 382, 255], [128, 256, 1, -1, 127]]):
            payload = (torch.arange(5 * width, dtype=torch.int64) + iteration * 17).to(torch.uint8).view(5, width)
            scale_bits = torch.tensor(
                [0x3F800000, 0x80000000, 0x7FC00123, 0x7F800000, iteration], dtype=torch.int64
            ).to(torch.int32)
            values.view(torch.uint8).copy_(payload)
            scales.view(torch.int32).copy_(scale_bits)
            slots.copy_(torch.tensor(logical, dtype=slot_dtype))
            raw.fill_(90)
            torch.npu.synchronize()
            graph.replay() if graph is not None else write()
            torch.npu.synchronize()
            expected = torch.full((size,), 90, dtype=torch.uint8)
            scale_bytes = scale_bits.view(torch.uint8).view(5, 4)
            for row, slot in enumerate(logical):
                if slot < 0:
                    continue
                page, offset = divmod(slot, block)
                start = page * page_stride + offset * width
                expected[start : start + width] = payload[row]
                start = page * page_stride + block * width + offset * 4
                expected[start : start + 4] = scale_bytes[row]
            torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)
    finally:
        if graph is not None:
            graph.reset()


@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 576), (torch.float8_e4m3fn, 656)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_rows", [0, 1, 4, 9])
def test_backend_resident_payload_graph_replay(dtype, width, index_dtype, num_rows):
    stride = 3 * width
    raw = torch.arange(7 * width * dtype.itemsize, dtype=torch.int64).to(torch.uint8).to("npu:0")
    source = raw.view(dtype).as_strided((2, 2, width), (stride, width, 1), storage_offset=width)
    rows = torch.empty(num_rows, dtype=index_dtype, device="npu:0")
    destination = torch.empty((num_rows, width), dtype=dtype, device="npu:0")
    rows.copy_((torch.arange(num_rows, dtype=index_dtype, device="npu:0") % 5) - 1)
    backend = AscendHiSparseBackend()
    backend.gather_resident_rows(source, destination, rows)
    torch.npu.synchronize()
    if not num_rows:
        assert destination.numel() == 0
        return
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            backend.gather_resident_rows(source, destination, rows)
        for offset in (0, 2):
            values = (torch.arange(num_rows, dtype=index_dtype) + offset) % 5 - 1
            if index_dtype == torch.int64:
                values[0] = -(1 << 34)
            rows.copy_(values.to("npu:0"))
            raw.bitwise_xor_(1)
            graph.replay()
            torch.npu.synchronize()
            safe = values.long().clamp_min(0)
            expected = source.view(torch.uint8).cpu()[safe // 2, safe % 2]
            assert torch.equal(destination.view(torch.uint8).cpu(), expected)
    finally:
        torch.npu.synchronize()
        graph.reset()


@pytest.mark.parametrize("groups", [1, 3])
@pytest.mark.parametrize("recycled_count", [1, 8, 9, 17])
def test_recycled_group_invalidation_preserves_other_states_and_blocks(groups, recycled_count, monkeypatch):
    """Use the model's block128/Hot10240 geometry at every bucket boundary."""
    from types import SimpleNamespace

    from vllm_ascend.hisparse import recycled_invalidate as recycled

    monkeypatch.setattr(recycled, "get_tensor_model_parallel_rank", lambda: 0)
    block_size, hot, state_count = 128, 10240, 3
    host_rows = (recycled_count + 5) * block_size
    initial = torch.arange(state_count * hot, dtype=torch.int32).view(state_count, hot) % host_rows
    initial[:, ::31] = -1
    tables = [(initial + group * 7).to("npu") for group in range(groups)]
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(kernel_config=SimpleNamespace(enable_jit_warmup=True)),
        backend=AscendHiSparseBackend(),
        kernel_block_size=block_size,
        leader_runtimes=[SimpleNamespace(index_group=SimpleNamespace(device_global_indices=t)) for t in tables],
        _pending_invalid_block_ids=[],
    )
    owner = recycled.RecycledGroupInvalidator.bind(worker)
    assert owner is not None
    torch.npu.synchronize()
    for group, table in enumerate(tables):
        torch.testing.assert_close(table.cpu(), initial + group * 7, rtol=0, atol=0)
    states = torch.tensor([2, 0], dtype=torch.int64, device="npu")
    for offset in (0, 2):
        blocks = list(reversed(range(offset, offset + recycled_count))) + [offset]
        worker._pending_invalid_block_ids[:] = blocks
        for group, table in enumerate(tables):
            table.copy_(initial + group * 7)
        assert owner.try_call(worker, worker._pending_invalid_block_ids, states)
        torch.npu.synchronize()
        block_set = torch.tensor(sorted(set(blocks)), dtype=torch.int32)
        for group, table in enumerate(tables):
            expected = initial + group * 7
            for state in (2, 0):
                tags = expected[state]
                hit = (tags >= 0) & torch.isin(tags // block_size, block_set)
                tags[hit] = -1
            torch.testing.assert_close(table.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("queries", [1, 3])
@pytest.mark.parametrize("capture", [False, True])
def test_packed_kv_scatter_and_attention_gather_model_geometry(queries, capture):
    """Check BF16 latent512/RoPE64, block128 and TopK2048 with shared-page gaps."""
    block, width, latent_width, topk = 128, 576, 512, 2048
    pages = 3
    backing = torch.full((pages, 2, block, width), -3, dtype=torch.bfloat16, device="npu")
    cache = backing[:, 1]
    values = torch.empty((4, width), dtype=torch.bfloat16, device="npu")
    slots = torch.tensor([0, -1, 128, 383], dtype=torch.int64, device="npu")
    valid = torch.empty(1, dtype=torch.int32, device="npu")
    pattern = torch.tensor([-1, 0, 128, 383, pages * block], dtype=torch.int32)
    ids = pattern.repeat((topk + pattern.numel() - 1) // pattern.numel())[:topk]
    indices = ids.expand(queries, -1).contiguous().to("npu")
    latent = torch.empty((queries, topk, 1, latent_width), dtype=torch.bfloat16, device="npu")
    rope = torch.empty((queries, topk, 1, width - latent_width), dtype=torch.bfloat16, device="npu")

    def forward():
        scatter_packed_kv(values[:, :latent_width], values[:, latent_width:], cache, slots, valid)
        pack_attention_rows(cache, indices, latent, rope, latent_width)

    def check(run):
        for count in (2, 4):
            payload = ((torch.arange(4 * width).view(4, width) + count) % 97).to(torch.bfloat16)
            values.copy_(payload)
            valid.fill_(count)
            backing.fill_(-3)
            run()
            torch.npu.synchronize()
            expected = torch.full(backing.shape, -3, dtype=torch.bfloat16)
            for token, slot in enumerate((0, -1, 128, 383)):
                if token < count and slot >= 0:
                    expected[slot // block, 1, slot % block] = payload[token]
            torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
            selected = expected[:, 1].reshape(-1, width)[ids.clamp(0, pages * block - 1).long()]
            selected[(ids < 0) | (ids >= pages * block)] = 0
            torch.testing.assert_close(
                latent.cpu(), selected[:, :latent_width].expand(queries, -1, -1).unsqueeze(2), rtol=0, atol=0
            )
            torch.testing.assert_close(
                rope.cpu(), selected[:, latent_width:].expand(queries, -1, -1).unsqueeze(2), rtol=0, atol=0
            )

    valid.fill_(0)
    values.zero_()
    if capture:
        with _invalidation_graph(forward) as (graph, stream):
            with torch.npu.graph(graph, stream=stream):
                forward()
            check(graph.replay)
    else:
        check(forward)
