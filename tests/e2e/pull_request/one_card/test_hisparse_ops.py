# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-NPU HiSparse state and mapped-payload tests against a CPU oracle."""

from contextlib import contextmanager

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.hisparse.cache_ops import (
    HiSparseNPUOps,
    ResolverWorkspace,
    pack_attention_rows,
    scatter_packed_kv,
)
from vllm_ascend.hisparse.host_pool import MappedHostTensor, allocate_mapped_host

pytest.importorskip("vllm.v1.hisparse.backend")
from vllm_ascend.hisparse.backend import AscendHiSparseBackend  # noqa: E402


@pytest.fixture(autouse=True)
def _require_mapped_host_device():
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("HiSparse mapped-host operators currently target Ascend 950")
    torch.npu.set_device(0)
    # Focused -k selections must load their native dependency independently;
    # they cannot rely on the private-Host-pool test running first.
    AscendHiSparseBackend().check_available()


def test_native_resolver_validates_ub_before_allocating_workspace():
    # This is the same validator called by workspace construction and launch.
    validate = torch.ops._C_ascend.hisparse_resolver_validate_scratch
    validate(10240, 2048)
    with pytest.raises(RuntimeError, match="UB bytes"):
        validate(32768, 32768)
    with pytest.raises(RuntimeError, match="int16 LRU slots"):
        validate(16, 17)


@pytest.mark.parametrize("descriptor_count", [0, 1, 16, 17, 312])
@pytest.mark.parametrize("graph_producer", [False, True])
def test_private_host_pool_retains_pinned_storage_for_dma_writeback(descriptor_count, graph_producer):
    from vllm.v1.hisparse import runtime

    backend = AscendHiSparseBackend()
    capacity = max(32768, (descriptor_count + 1) * 2048)
    payload, registered = runtime.allocate_pinned_host_pool(capacity)
    graph = None
    try:
        assert payload.numel() == capacity
        assert payload.is_pinned() and registered.is_pinned()
        assert registered.data_ptr() % 4096 == 0
        mapping = registered._hisparse_registration
        assert mapping.address_of(payload[656:1312]) == mapping.device_ptr + 656
        assert payload.untyped_storage().data_ptr() == registered.untyped_storage().data_ptr()
        cpu_source = torch.arange(capacity, dtype=torch.int32).to(torch.uint8)
        source = cpu_source.to("npu")
        staging = source.clone()
        if graph_producer:
            source.copy_(staging)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                source.copy_(staging)
        stream = torch.npu.Stream()
        source_offsets = [i * 2048 for i in range(descriptor_count)]
        destination_offsets = [(descriptor_count - i - 1) * 2048 + 656 for i in range(descriptor_count)]
        copy_sizes = [656 * (1 + i % 2) for i in range(descriptor_count)]
        src = torch.tensor([source.data_ptr() + i for i in source_offsets], dtype=torch.int64, pin_memory=True)
        dst = torch.tensor([payload.data_ptr() + i for i in destination_offsets], dtype=torch.int64, pin_memory=True)
        sizes = torch.tensor(copy_sizes, dtype=torch.int64, pin_memory=True)
        originals = tuple(t.clone() for t in (src, dst, sizes))
        for change in (1, 3, 7):
            cpu_source.bitwise_xor_(change)
            staging.copy_(cpu_source, non_blocking=True)
            if graph is None:
                source.copy_(staging)
            else:
                graph.replay()
            expected = torch.full((capacity,), 255, dtype=torch.uint8)
            for source_offset, destination_offset, size in zip(source_offsets, destination_offsets, copy_sizes):
                expected[destination_offset : destination_offset + size] = cpu_source[
                    source_offset : source_offset + size
                ]
            for optimized in (False, True):
                payload.fill_(-1)
                stream.wait_stream(torch.npu.current_stream())
                with stream:
                    if optimized:
                        backend.swap_blocks_batch(src, dst, sizes)
                        if 0 < descriptor_count <= 16:
                            # The queued command owns scalar descriptors even
                            # after the caller reuses its CPU metadata bank.
                            for descriptor in (src, dst, sizes):
                                descriptor.zero_()
                    else:
                        # Existing callers keep the four-argument schema and
                        # original API selection unless they opt in explicitly.
                        torch.ops._C_ascend.swap_blocks_batch(src, dst, sizes, 1)
                    complete = torch.npu.Event()
                    complete.record()
                complete.synchronize()
                torch.testing.assert_close(payload.view(torch.uint8), expected, rtol=0, atol=0)
                for descriptor, original in zip((src, dst, sizes), originals):
                    descriptor.copy_(original)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        backend.unpin_tensor(registered)


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


def reference(topk, requests, states, source, resident, hot_table, dgi, lru, block):
    dgi = dgi.clone()
    lru = lru.clone()
    out = torch.full_like(topk, -1)
    counts = torch.zeros(len(topk), dtype=torch.int32)
    sh = torch.full_like(topk, -9)
    sd = torch.full_like(topk, -9)
    sc = torch.zeros(len(topk), dtype=torch.int32)
    stats = torch.zeros(2, dtype=torch.int64)
    for request in sorted(set(requests.tolist())):
        if request < 0:
            continue
        rows = [i for i, r in enumerate(requests.tolist()) if r == request]
        state = int(states[request])
        union = []
        references = []
        for row in rows:
            for col, token in enumerate(topk[row].tolist()):
                if token < 0:
                    continue
                page, offset = divmod(token, block)
                res = int(resident[state, page])
                if res:
                    out[row, col] = res * block + offset
                else:
                    host = int(source[request, page]) * block + offset
                    references.append((row, col, host))
                    if host not in union:
                        union.append(host)
                counts[row] += 1
        order = lru[state].tolist()
        hits = [slot for slot in order if int(dgi[state, slot]) in union]
        evict = [slot for slot in order if slot not in hits]
        mapping = {int(dgi[state, slot]): slot for slot in hits}
        misses: list[tuple[int, int]] = []
        for host in union:
            if host not in mapping:
                slot = evict[len(misses)]
                mapping[host] = slot
                dgi[state, slot] = host
                misses.append((host, slot))
        if union:
            lru[state] = torch.tensor(evict[len(misses) :] + evict[: len(misses)] + hits, dtype=torch.int16)
        for row, col, host in references:
            slot = mapping[host]
            out[row, col] = int(hot_table[request, slot // block]) * block + slot % block
        for n, (host, slot) in enumerate(misses):
            row = rows[0] + n // topk.size(1)
            col = n % topk.size(1)
            sh[row, col] = host
            sd[row, col] = int(hot_table[request, slot // block]) * block + slot % block
            sc[row] += 1
        stats += torch.tensor([len(references) - len(misses), len(misses)])
    return out, counts, sh, sd, sc, dgi, lru, stats


@pytest.mark.parametrize("device_row_count", [False, True])
@pytest.mark.parametrize("compact_attention", [False, True])
def test_resolver_strides_padding_graph(device_row_count, compact_attention):
    raw = torch.empty(8192, dtype=torch.uint8)
    offset = (-raw.data_ptr()) % 4096
    registered = raw[offset : offset + 4096]
    mapping = MappedHostTensor(registered)
    host = registered.view(torch.float32).view(64, 16)
    host.copy_(torch.arange(host.numel(), dtype=torch.float32).view_as(host))
    # Deliberate gaps between physical blocks exercise fused layer strides.
    backing = torch.full((10, 2, 4, 16), -17.0, device="npu")
    hot = backing[:, 1]
    topk = torch.tensor(
        [
            [0, 4, 8, 12, -1, -1, -1, -1],
            [5, 9, 13, 12, -1, -1, -1, -1],
            [0, 1, -1, 2, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    requests = torch.tensor([0, 0, 1, -1], dtype=torch.int32)
    states = torch.tensor([1, 0], dtype=torch.int32)
    source = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)
    resident = torch.tensor([[0, 0, 0, 0], [5, 0, 0, 0]], dtype=torch.int32)
    hot_table = torch.tensor([[1, 2, 6, 7], [3, 4, 8, 9]], dtype=torch.int32)
    dgi = torch.tensor([[20, -1, -1, -1, -1, -1, -1, -1], [8, 99, -1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    lru = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [3, 0, 5, 1, 7, 4, 2, 6]], dtype=torch.int16)
    dgi = torch.cat((dgi, torch.full((2, 8), -1, dtype=torch.int32)), dim=1)
    lru = torch.cat((lru, torch.arange(8, 16, dtype=torch.int16)[None, :].expand(2, -1)), dim=1)
    expected = reference(topk, requests, states, source, resident, hot_table, dgi, lru, 4)
    workspace = ResolverWorkspace(4, 16, 16, torch.device("npu:0"))
    stride_rows = hot.stride(0) // hot.size(2)
    capacity = (hot.size(0) - 1) * stride_rows + hot.size(1)
    ops = HiSparseNPUOps(
        mapping, workspace, attention_capacity=capacity if compact_attention else 0, attention_block_stride=stride_rows
    )
    topk_n, requests_n, states_n, source_n, resident_n, hot_table_n, dgi_n, lru_n = [
        t.npu() for t in (topk, requests, states, source, resident, hot_table, dgi, lru)
    ]
    out = torch.empty_like(topk_n)
    attn = torch.empty_like(topk_n)
    counts = torch.empty(4, dtype=torch.int32, device="npu")
    sh, sd = torch.full_like(out, -9), torch.full_like(out, -9)
    sc = torch.empty_like(counts)
    stats = torch.zeros(2, dtype=torch.int64, device="npu")
    valid_rows = torch.tensor(3, dtype=torch.int32, device="npu") if device_row_count else None
    if device_row_count:
        # Stale graph padding looks like another valid request row. Only the
        # device scalar separates it from current query rows.
        requests_n[-1] = 1

    def forward():
        ops.hisparse_resolve_residency(
            host,
            hot,
            hot_table_n,
            topk_n,
            out,
            dgi_n,
            lru_n,
            states_n,
            16,
            16,
            None,
            stats,
            attn,
            8,
            requests_n,
            source_n,
            4,
            None,
            counts,
            sh,
            sd,
            sc,
            resident_n,
            4,
            0,
            valid_rows,
        )
        ops.hisparse_gather_compact(host, hot, sh, sd, sc)

    forward()
    torch.npu.synchronize()
    actual = (out, counts, sh, sd, sc, dgi_n, lru_n, stats)
    for name, a, e in zip(
        ("indices", "counts", "swap_host", "swap_device", "swap_counts", "dgi", "lru", "stats"), actual, expected
    ):
        torch.testing.assert_close(a.cpu(), e, msg=name)

    def attention_reference(physical):
        mapped = torch.where(physical >= 0, physical // 4 * 8 + physical % 4, -1)
        if compact_attention:
            for row in mapped:
                valid = [value for value in row.tolist() if value >= 0]
                row.copy_(torch.tensor(valid + [-1] * (row.numel() - len(valid)), dtype=row.dtype))
        return mapped

    expected_attn = attention_reference(expected[0])
    torch.testing.assert_close(attn.cpu(), expected_attn)
    for row in range(4):
        for col in range(int(expected[4][row])):
            src, dst = int(expected[2][row, col]), int(expected[3][row, col])
            torch.testing.assert_close(hot[dst // 4, dst % 4].cpu(), host[src])
    torch.testing.assert_close(backing[:, 0].cpu(), torch.full((10, 4, 16), -17.0))
    latent = torch.empty((4, 8, 1, 12), device="npu")
    rope = torch.empty((4, 8, 1, 4), device="npu")
    pack_attention_rows(hot, out, latent, rope, 12)
    torch.npu.synchronize()
    hot_cpu = hot.cpu()
    for row in range(4):
        for col in range(8):
            slot = int(expected[0][row, col])
            value = torch.zeros(16) if slot < 0 else hot_cpu[slot // 4, slot % 4]
            torch.testing.assert_close(latent[row, col, 0].cpu(), value[:12])
            torch.testing.assert_close(rope[row, col, 0].cpu(), value[12:])
    new_latent = torch.arange(36, dtype=torch.float32).view(3, 1, 1, 12).npu()
    new_rope = torch.arange(12, dtype=torch.float32).view(3, 1, 1, 4).npu()
    slots = torch.tensor([0, -1, 2], dtype=torch.int64, device="npu")
    valid = torch.tensor([2], dtype=torch.int32, device="npu")
    scatter_packed_kv(new_latent, new_rope, hot, slots, valid)
    torch.npu.synchronize()
    torch.testing.assert_close(hot[0, 0].cpu(), torch.cat((new_latent[0].flatten().cpu(), new_rope[0].flatten().cpu())))
    torch.testing.assert_close(hot[0, 2].cpu(), hot_cpu[0, 2])
    # Capture warmed kernels, reset state outside capture, then verify replay
    # after changing the mapped CPU payload. No host selection work runs here.
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(2):
            forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        forward()
    dgi_n.copy_(dgi.npu())
    lru_n.copy_(lru.npu())
    stats.zero_()
    sh.fill_(-9)
    sd.fill_(-9)
    host.add_(5000)
    graph.replay()
    torch.npu.synchronize()
    for name, a, e in zip(
        ("indices", "counts", "swap_host", "swap_device", "swap_counts", "dgi", "lru", "stats"), actual, expected
    ):
        torch.testing.assert_close(a.cpu(), e, msg="graph " + name)
    for row in range(4):
        for col in range(int(expected[4][row])):
            src, dst = int(expected[2][row, col]), int(expected[3][row, col])
            torch.testing.assert_close(hot[dst // 4, dst % 4].cpu(), host[src])
    if device_row_count:
        for active_rows in [0, 1, 2, 3]:
            valid_rows.fill_(active_rows)
            dgi_n.copy_(dgi.npu())
            lru_n.copy_(lru.npu())
            stats.zero_()
            sh.fill_(-9)
            sd.fill_(-9)
            oracle_requests = requests.clone()
            oracle_requests[active_rows:] = -1
            oracle = reference(topk, oracle_requests, states, source, resident, hot_table, dgi, lru, 4)
            graph.replay()
            torch.npu.synchronize()
            for actual_value, expected_value in zip(actual, oracle):
                torch.testing.assert_close(actual_value.cpu(), expected_value)
            torch.testing.assert_close(attn.cpu(), attention_reference(oracle[0]))
    graph.reset()
    mapping.close()


@pytest.mark.parametrize("query_rows,hot_size", [(2, 4096), (4, 10240), (3, 4096)])
@pytest.mark.parametrize("cross_stream", [False, True])
def test_topk_2048_speculative_union_graph(query_rows, hot_size, cross_stream):
    topk, width, block = 2048, 576, 128
    positions = query_rows * topk
    union = min(positions, hot_size)
    # The 3 x 2048 case has more positions than the 4096-row mask/scan
    # capacity. Its duplicates cross tiles of the iterative miss scan.
    blocks = hot_size // block + 1
    host = allocate_mapped_host((hot_size + block, width), torch.bfloat16)
    mapping = host._hisparse_registration
    host.copy_((torch.arange(host.size(0)) % 97).to(torch.bfloat16)[:, None].expand_as(host))
    hot = torch.zeros((blocks, block, width), dtype=torch.bfloat16, device="npu")
    logical = torch.arange(positions, dtype=torch.int32).view(query_rows, topk) % union
    indices = logical.npu()
    requests = torch.zeros(query_rows, dtype=torch.int32, device="npu")
    states = torch.zeros(1, dtype=torch.int32, device="npu")
    table = torch.arange(1, blocks, dtype=torch.int32, device="npu")[None, :]
    resident = torch.zeros_like(table)
    dgi = torch.full((1, hot_size), -1, dtype=torch.int32, device="npu")
    lru = torch.arange(hot_size, dtype=torch.int32, device="npu").to(torch.int16)[None, :]
    workspace = ResolverWorkspace(query_rows, hot_size, union, torch.device("npu:0"))
    ops = HiSparseNPUOps(mapping, workspace)
    out = torch.empty_like(indices)
    counts = torch.empty(query_rows, dtype=torch.int32, device="npu")
    sh, sd, sc = torch.empty_like(out), torch.empty_like(out), torch.empty_like(counts)
    stats = torch.zeros(2, dtype=torch.int64, device="npu")

    stream = torch.Stream(device="npu") if cross_stream else None
    ready = torch.Event() if cross_stream else None

    def resolve_and_gather():
        ops.hisparse_resolve_residency(
            host,
            hot,
            table,
            indices,
            out,
            dgi,
            lru,
            states,
            hot_size,
            union,
            None,
            stats,
            None,
            block,
            requests,
            table,
            block,
            None,
            counts,
            sh,
            sd,
            sc,
            resident,
            block,
            0,
            None,
        )
        ops.hisparse_gather_compact(host, hot, sh, sd, sc)

    def forward():
        if stream is None:
            resolve_and_gather()
            return
        current = torch.accelerator.current_stream()
        stream.wait_stream(current)
        with stream:
            resolve_and_gather()
            ready.record(stream)
        current.wait_event(ready)

    forward()
    torch.npu.synchronize()
    expected = logical + block
    expected_swaps = torch.tensor(
        [min(topk, max(0, union - row * topk)) for row in range(query_rows)], dtype=torch.int32
    )
    cold_hits = positions - union
    torch.testing.assert_close(out.cpu(), expected)
    torch.testing.assert_close(sc.cpu(), expected_swaps)
    torch.testing.assert_close(stats.cpu(), torch.tensor([cold_hits, union], dtype=torch.int64))
    torch.testing.assert_close(hot[1:].cpu().reshape(hot_size, width)[:union], host[block : block + union])
    expected_lru = torch.cat((torch.arange(union, hot_size), torch.arange(union))).to(torch.int16)[None, :]
    torch.testing.assert_close(lru.cpu(), expected_lru)
    forward()
    torch.npu.synchronize()
    torch.testing.assert_close(sc.cpu(), torch.zeros(query_rows, dtype=torch.int32))
    torch.testing.assert_close(stats.cpu(), torch.tensor([cold_hits + positions, union], dtype=torch.int64))
    capture_stream = torch.npu.Stream()
    capture_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(capture_stream):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=capture_stream):
        forward()
    dgi.fill_(-1)
    lru.copy_(torch.arange(hot_size, dtype=torch.int16)[None, :])
    stats.zero_()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected)
    torch.testing.assert_close(sc.cpu(), expected_swaps)
    torch.testing.assert_close(stats.cpu(), torch.tensor([cold_hits, union], dtype=torch.int64))
    for _ in range(10):
        graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(sc.cpu(), torch.zeros(query_rows, dtype=torch.int32))
    torch.testing.assert_close(stats.cpu(), torch.tensor([cold_hits + 10 * positions, union], dtype=torch.int64))
    torch.testing.assert_close(hot[1:].cpu().reshape(hot_size, width)[:union], host[block : block + union])
    torch.testing.assert_close(lru.cpu(), expected_lru)
    graph.reset()
    mapping.close()


@pytest.mark.parametrize("capture", [False, True])
def test_random_request_unions(capture):
    """Vary query counts, request order, stable state, duplicates and residency."""
    rows, topk, hot_size, block, width, requests_count = 40, 8, 16, 4, 16, 8
    host = allocate_mapped_host((132, width), torch.float32)
    mapping = host._hisparse_registration
    host.copy_(torch.arange(host.numel(), dtype=torch.float32).view_as(host))
    # 1..32 are hot pages and 33..64 are resident pages; layer 0 is a sentinel.
    backing = torch.full((65, 2, block, width), -17.0, device="npu")
    hot = backing[:, 1]
    workspace = ResolverWorkspace(rows, hot_size, hot_size, torch.device("npu:0"))
    ops = HiSparseNPUOps(mapping, workspace)
    indices = torch.empty((rows, topk), dtype=torch.int32, device="npu")
    requests = torch.empty(rows, dtype=torch.int32, device="npu")
    states = torch.empty(requests_count, dtype=torch.int32, device="npu")
    source = torch.arange(1, 33, dtype=torch.int32).view(requests_count, 4)
    hot_table = source.npu()
    source_n = source.npu()
    resident = torch.empty_like(hot_table)
    dgi = torch.empty((requests_count, hot_size), dtype=torch.int32, device="npu")
    lru = torch.empty_like(dgi, dtype=torch.int16)
    out, sh, sd = (torch.empty_like(indices) for _ in range(3))
    counts = torch.empty(rows, dtype=torch.int32, device="npu")
    sc = torch.empty_like(counts)
    stats = torch.empty(2, dtype=torch.int64, device="npu")
    miss = torch.empty_like(indices, dtype=torch.uint8)

    def forward():
        ops.hisparse_resolve_residency(
            host,
            hot,
            hot_table,
            indices,
            out,
            dgi,
            lru,
            states,
            hot_size,
            hot_size,
            miss,
            stats,
            None,
            block,
            requests,
            source_n,
            block,
            None,
            counts,
            sh,
            sd,
            sc,
            resident,
            block,
            0,
            None,
        )
        ops.hisparse_gather_compact(host, hot, sh, sd, sc)

    graph = None
    try:
        for seed in range(16):
            generator = torch.Generator().manual_seed(seed)
            state_cpu = torch.randperm(requests_count, generator=generator).to(torch.int32)
            requests_cpu = torch.full((rows,), -1, dtype=torch.int32)
            indices_cpu = torch.randint(-1, 16, (rows, topk), generator=generator, dtype=torch.int32)
            resident_cpu = torch.zeros((requests_count, 4), dtype=torch.int32)
            dgi_cpu = torch.full((requests_count, hot_size), -1, dtype=torch.int32)
            lru_cpu = torch.stack([torch.randperm(hot_size, generator=generator) for _ in range(requests_count)]).to(
                torch.int16
            )
            cache_cpu = torch.full(hot.shape, -17.0)
            cursor = 0
            for request in torch.randperm(requests_count, generator=generator).tolist():
                query_rows = 1 + (request + seed) % 4
                requests_cpu[cursor : cursor + query_rows] = request
                cursor += query_rows + 1  # Adjacent padded rows share no state.
                state = int(state_cpu[request])
                for page in range(4):
                    is_resident = request % 3 == 0 or (request % 3 == 2 and page < 2)
                    if is_resident:
                        physical = 33 + 4 * request + page
                        resident_cpu[state, page] = physical
                        cache_cpu[physical] = host[
                            int(source[request, page]) * block : int(source[request, page]) * block + block
                        ]
                selected = torch.randperm(hot_size, generator=generator)[:8] + 4 + request * hot_size
                slots = torch.randperm(hot_size, generator=generator)[:8]
                dgi_cpu[state, slots] = selected.to(torch.int32)
                for slot, host_id in zip(slots.tolist(), selected.tolist()):
                    physical = int(source[request, slot // block])
                    cache_cpu[physical, slot % block] = host[host_id]
            expected = reference(
                indices_cpu, requests_cpu, state_cpu, source, resident_cpu, source, dgi_cpu, lru_cpu, block
            )
            for destination, value in [
                (indices, indices_cpu),
                (requests, requests_cpu),
                (states, state_cpu),
                (resident, resident_cpu),
                (dgi, dgi_cpu),
                (lru, lru_cpu),
                (hot, cache_cpu),
            ]:
                destination.copy_(value)
            sh.fill_(-9)
            sd.fill_(-9)
            stats.zero_()
            if capture and graph is None:
                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    forward()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    forward()
                dgi.copy_(dgi_cpu)
                lru.copy_(lru_cpu)
                hot.copy_(cache_cpu)
                sh.fill_(-9)
                sd.fill_(-9)
                stats.zero_()
            if graph is None:
                forward()
            else:
                graph.replay()
            torch.npu.synchronize()
            for label, actual, wanted in zip(
                ("indices", "counts", "swap_host", "swap_device", "swap_counts", "dgi", "lru", "stats"),
                (out, counts, sh, sd, sc, dgi, lru, stats),
                expected,
            ):
                torch.testing.assert_close(actual.cpu(), wanted, msg=f"seed {seed}: {label}")
            # Every selected token, whether resident, cached or copied, has
            # exactly the canonical host payload. Sentinel layer stays intact.
            cache = hot.cpu()
            for row in range(rows):
                request = int(requests_cpu[row])
                if request < 0:
                    continue
                for col, token in enumerate(indices_cpu[row].tolist()):
                    if token < 0:
                        continue
                    physical = int(expected[0][row, col])
                    host_id = int(source[request, token // block]) * block + token % block
                    torch.testing.assert_close(cache[physical // block, physical % block], host[host_id])
            torch.testing.assert_close(backing[:, 0].cpu(), torch.full((65, block, width), -17.0))
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        mapping.close()


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


@pytest.mark.parametrize("capture", [False, True])
def test_page_strided_indexer_write_preserves_bits_and_padding(capture):
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
    slots = torch.full((5,), -1, dtype=torch.int32, device="npu")

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
            slots.copy_(torch.tensor(logical, dtype=torch.int32))
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


@pytest.mark.parametrize("graph_producer,shared_group", [(False, True), (True, False)])
def test_direct_target_mirror_survives_legal_draft_writes(graph_producer, shared_group, record_property):
    from types import SimpleNamespace

    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.hisparse import runtime
    from vllm.v1.hisparse.runtime import HiSparseCacheHandle, PagedCacheView
    from vllm.v1.hisparse.types import SparseKVRowMirror
    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    from vllm_ascend.hisparse.host_mirror import HostMirrorConnector, _DirectMirrorSources

    width, block, gpu_blocks, host_blocks = 656, 128, 5, 3
    page = block * width
    stride = 3 * page
    names = ("target0", "target1", "draft")
    layer_offsets = (0, page, 2 * page if shared_group else 0)
    source_groups = (0, 0, 0 if shared_group else 1)
    pool = BlockPool(gpu_blocks, enable_caching=False, hash_block_size=block)
    target_pages = pool.get_new_blocks(2)
    draft_pages = target_pages if shared_group else pool.get_new_blocks(2)
    target_ids = [page.block_id for page in target_pages]
    draft_ids = [page.block_id for page in draft_pages]
    assert (set(target_ids).isdisjoint(draft_ids)) is not shared_group
    spans = ((126, 2), (2, 1))
    mirrors = tuple(
        SparseKVRowMirror(
            (target_ids[i] * block + offset,)
            if shared_group
            else (target_ids[i] * block + offset, draft_ids[i] * block + offset),
            (i + 1) * block + offset,
            count,
        )
        for i, (offset, count) in enumerate(spans)
    )
    backend = AscendHiSparseBackend()
    host_rows = host_blocks * block
    payload, registered = runtime.allocate_pinned_host_pool(len(names) * host_rows * width)
    graph = None
    worker = None
    try:
        assert payload.is_pinned() and registered.is_pinned()
        mapping = registered._hisparse_registration
        # Include both page and layer in the CPU oracle. Plain uint8 arange
        # repeats across these aligned pages and could miss a wrong page ID.
        byte = torch.arange(gpu_blocks * stride, dtype=torch.int32)
        cpu_source = (byte + (byte // stride) * 19 + ((byte % stride) // page) * 31).to(torch.uint8)
        source = cpu_source.to("npu")
        staging = source.clone()
        states = torch.zeros(1, dtype=torch.int32, device=source.device)
        written = torch.tensor(
            [m.destination_start + j for m in mirrors for j in range(m.num_rows)],
            dtype=torch.int32,
            device=source.device,
        )
        requests = torch.zeros_like(written)
        owners, handles = [], []
        for index, (group, offset) in enumerate(zip(source_groups, layer_offsets)):
            span = slice(index * host_rows * width, (index + 1) * host_rows * width)
            view = PagedCacheView.bind(
                source,
                dtype=torch.uint8,
                row_width=width,
                byte_offset=offset,
                block_stride=stride,
                num_blocks=gpu_blocks,
                block_size=block,
            )
            owner = SimpleNamespace(
                resident_source_index=group,
                request_state_indices=states,
                hot=view,
                host_cache=payload[span].view(host_rows, width),
                eager_host_mirror=True,
                is_group_leader=index != 1,
                cache_ops=HiSparseNPUOps(mapping),
            )
            handle = HiSparseCacheHandle(owner)
            handle.view = view
            handle.slot_mapping = torch.tensor(
                [m.source_starts[group] + j for m in mirrors for j in range(m.num_rows)],
                dtype=torch.int32,
                device=source.device,
            )
            handle.draft_layer = index == 2
            owners.append(owner)
            handles.append(handle)
        for leader, followers in ((owners[0], [owners[1]]), (owners[2], [])):
            index_group = SimpleNamespace(
                leader=leader,
                followers=followers,
                device_global_indices=torch.full((1, 16), -1, dtype=torch.int32, device=source.device),
            )
            for owner in [leader, *followers]:
                owner.index_group = index_group
        worker = HiSparseConnectorWorker.__new__(HiSparseConnectorWorker)
        worker.backend, worker._initialized = backend, False
        worker.vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
            num_lookahead_tokens=1,
            kernel_config=SimpleNamespace(enable_jit_warmup=False),
        )
        worker.initialize(handles, list(names), source, 1, host_blocks, source.device, [registered])
        sources = _DirectMirrorSources.bind(worker, SimpleNamespace(draft_attn_layer_names={"draft"}))
        assert sources is not None
        connector = HiSparseConnector.__new__(HiSparseConnector)
        connector.connector_worker = worker
        inner = ActiveKVConnector.__new__(ActiveKVConnector)
        inner.kv_connector, inner._disabled = connector, False
        output_stream = torch.Stream(device=source.device)
        proxy = HostMirrorConnector(inner, worker, output_stream, sources)
        values = torch.full((written.numel(), width), 237, dtype=torch.uint8, device=source.device)
        valid = torch.tensor([written.numel()], dtype=torch.int32, device=source.device)
        if graph_producer:
            source.copy_(staging)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                source.copy_(staging)
        pending_observed = 0
        for change in (1, 3, 7):
            proxy.begin_execution(True)
            proxy._eligible_forward = True  # Fixed decode metadata for this transport-only test.
            worker._set_row_mirrors(mirrors)
            for handle in handles:
                handle.num_actual_tokens = handle.num_decode_tokens = written.numel()
                handle.decode_batch = handle.host_mirror_required = True
                handle.mirror_slot_mapping, handle.req_id_per_token = written, requests
            payload.fill_(-1)
            cpu_source.bitwise_xor_(change)
            staging.copy_(cpu_source, non_blocking=True)
            source.copy_(staging) if graph is None else graph.replay()
            expected = torch.full_like(payload, -1).view(torch.uint8).view(len(names), host_rows, width)
            for i, (offset, count) in enumerate(spans):
                for layer in (0, 1):
                    start = target_ids[i] * stride + layer_offsets[layer] + offset * width
                    end = start + count * width
                    dest = mirrors[i].destination_start
                    expected[layer, dest : dest + count] = cpu_source[start:end].view(count, width)
            proxy.finish_forward()  # Real Target-only descriptor DMA from Resident pages.
            assert proxy._overlap_draft and proxy._pending_event is worker.host_write_event
            draft = handles[2]
            draft.num_actual_tokens = written.numel()
            destination, slots, count = draft.write_target(written.numel(), written.numel())
            assert count == written.numel()
            # Submit only the actual draft layer's writable rows; do not join
            # or overwrite the full slab before this same-round write.
            scatter_packed_kv(values, values[:, :0], destination, slots, valid)
            pending_observed += int(not worker.host_write_event.query())
            proxy.finalize_step()
            output_done = torch.Event()
            output_done.record(output_stream)
            output_done.synchronize()
            torch.testing.assert_close(payload.view(torch.uint8).view_as(expected), expected, rtol=0, atol=0)
            for draft_block, (offset, size) in zip(draft_ids, spans):
                actual = destination[draft_block, offset : offset + size].cpu()
                torch.testing.assert_close(actual, torch.full_like(actual, 237), rtol=0, atol=0)
        assert torch.npu.is_initialized()
        record_property("actual_NPU_initialized", True)
        record_property("direct_mirror_submissions", 3)
        record_property("shared_resident_group", shared_group)
        record_property("host_event_pending_after_draft_submit", pending_observed)
    finally:
        torch.npu.synchronize()
        dma_stream = getattr(worker, "dma_stream", None)
        if dma_stream is not None:
            dma_stream.synchronize()
        if graph is not None:
            graph.reset()
        backend.unpin_tensor(registered)
        pool.free_blocks(target_pages if shared_group else [*target_pages, *draft_pages])
