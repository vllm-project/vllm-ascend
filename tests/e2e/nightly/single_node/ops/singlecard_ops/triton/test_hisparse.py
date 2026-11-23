# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse Triton transfer, layout and invalidation accuracy tests."""

from contextlib import contextmanager

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import is_950
from vllm_ascend.hisparse.cache_ops import (
    HiSparseNPUOps,
    gather_resident_rows,
    pack_attention_rows,
    scatter_packed_kv,
)
from vllm_ascend.hisparse.host_pool import allocate_mapped_host
from vllm_ascend.utils import vllm_version_is

pytestmark = [
    pytest.mark.skipif(vllm_version_is("0.33.0"), reason="HiSparse e2e is not supported on vLLM v0.33.0"),
    pytest.mark.skipif(not is_950(), reason="HiSparse Triton tests require Ascend 950 (A5)"),
]


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
        # Cover invalid indices, the 128-column tile boundary, and reset to empty.
        test_counts = (0, 1, 7, topk, 0) if topk < 128 else (0, 1, 7, 127, 128, 129, topk, 0)
        for count in test_counts:
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
        for _ in range(2):
            dgi.copy_(initial)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(dgi.cpu(), expected)


@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 576), (torch.float8_e4m3fn, 656)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_rows", [1, 8, 9])
def test_gather_resident_rows_graph_replay(dtype, width, index_dtype, num_rows):
    stride = 3 * width
    raw = torch.arange(7 * width * dtype.itemsize, dtype=torch.int64).to(torch.uint8).to("npu:0")
    source = raw.view(dtype).as_strided((2, 2, width), (stride, width, 1), storage_offset=width)
    rows = torch.empty(num_rows, dtype=index_dtype, device="npu:0")
    destination = torch.empty((num_rows, width), dtype=dtype, device="npu:0")
    rows.copy_((torch.arange(num_rows, dtype=index_dtype, device="npu:0") % 5) - 1)
    gather_resident_rows(source, destination, rows)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            gather_resident_rows(source, destination, rows)
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
