# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""arch22 native HiSparse contracts against the A5 foundation's CPU oracle."""

from contextlib import contextmanager

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.hisparse.cache_ops import HiSparseNPUOps, ResolverWorkspace
from vllm_ascend.hisparse.host_pool import allocate_mapped_host
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


@pytest.fixture(autouse=True)
def _require_arch22():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    if get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3):
        pytest.skip("requires the arch22 kernel")
    if not hasattr(torch.ops._C_ascend, "hisparse_resolve_residency"):
        import vllm_ascend.vllm_ascend_C  # type: ignore[import-not-found, import-untyped]  # noqa: F401
    assert hasattr(torch.ops._C_ascend, "hisparse_resolve_residency"), "rebuild the arch22 extension"


def _reference_hisparse_resolve_residency(topk, requests, states, source, resident, hot_table, dgi, lru, block):
    """Compute the HiSparse resolver's expected outputs and cache state on CPU."""
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


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_random_request_unions(capture, compact):
    """Vary query counts, request order, stable state, duplicates and residency."""
    rows, topk, hot_size, block, width, requests_count = 40, 8, 16, 4, 16, 8
    host = allocate_mapped_host((132, width), torch.float32)
    mapping = host._hisparse_registration
    host.copy_(torch.arange(host.numel(), dtype=torch.float32).view_as(host))
    # 1..32 are hot pages and 33..64 are resident pages; layer 0 is a sentinel.
    backing = torch.full((65, 2, block, width), -17.0, device="npu")
    hot = backing[:, 1]
    workspace = ResolverWorkspace(rows, hot_size, hot_size, torch.device("npu", torch.npu.current_device()))
    ops = HiSparseNPUOps(mapping, workspace, attention_capacity=65 * 2 * block if compact else 0)
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
    attention = torch.empty_like(indices) if compact else None
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
            stats,
            attention,
            2 * block if compact else block,
            requests,
            source_n,
            block,
            counts,
            sh,
            sd,
            sc,
            resident,
            block,
            None,
            miss_mask=miss,
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
            expected = _reference_hisparse_resolve_residency(
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
            if attention is not None:
                mapped = torch.where(expected[0] >= 0, expected[0] // block * (2 * block) + expected[0] % block, -1)
                compacted = torch.full_like(mapped, -1)
                for row in range(rows):
                    selected = mapped[row][mapped[row] >= 0]
                    compacted[row, : selected.numel()] = selected
                torch.testing.assert_close(attention.cpu(), compacted)
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


@pytest.mark.parametrize("query_rows,hot_size", [(2, 4096), (4, 10240), (3, 4096)])
@pytest.mark.parametrize("cross_stream", [False, True])
@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 576), (torch.int8, 656)])
def test_topk_2048_speculative_union_graph(query_rows, hot_size, cross_stream, dtype, width):
    topk, block = 2048, 128
    positions = query_rows * topk
    union = min(positions, hot_size)
    # The 3 x 2048 case has more positions than the 4096-row mask/scan
    # capacity. Its duplicates cross tiles of the iterative miss scan.
    blocks = hot_size // block + 1
    host = allocate_mapped_host((hot_size + block, width), dtype)
    mapping = host._hisparse_registration
    host.copy_((torch.arange(host.size(0)) % 97).to(dtype)[:, None].expand_as(host))
    hot = torch.zeros((blocks, block, width), dtype=dtype, device="npu")
    logical = torch.arange(positions, dtype=torch.int32).view(query_rows, topk) % union
    indices = logical.npu()
    requests = torch.zeros(query_rows, dtype=torch.int32, device="npu")
    states = torch.zeros(1, dtype=torch.int32, device="npu")
    table = torch.arange(1, blocks, dtype=torch.int32, device="npu")[None, :]
    resident = torch.zeros_like(table)
    dgi = torch.full((1, hot_size), -1, dtype=torch.int32, device="npu")
    lru = torch.arange(hot_size, dtype=torch.int32, device="npu").to(torch.int16)[None, :]
    workspace = ResolverWorkspace(query_rows, hot_size, union, torch.device("npu", torch.npu.current_device()))
    ops = HiSparseNPUOps(mapping, workspace, attention_capacity=blocks * block if dtype == torch.int8 else 0)
    out = torch.empty_like(indices)
    attention = torch.empty_like(indices) if dtype == torch.int8 else None
    counts = torch.empty(query_rows, dtype=torch.int32, device="npu")
    sh, sd, sc = torch.empty_like(out), torch.empty_like(out), torch.empty_like(counts)
    stats_backing = torch.full((6,), -73, dtype=torch.int64, device="npu")
    stats = stats_backing[1:3]
    stats.zero_()

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
            stats,
            attention,
            block,
            requests,
            table,
            block,
            counts,
            sh,
            sd,
            sc,
            resident,
            block,
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
    if attention is not None:
        torch.testing.assert_close(attention.cpu(), expected)
        torch.testing.assert_close(counts.cpu(), torch.full((query_rows,), topk, dtype=torch.int32))
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
    if attention is not None:
        torch.testing.assert_close(attention.cpu(), expected)
    torch.testing.assert_close(sc.cpu(), expected_swaps)
    torch.testing.assert_close(stats.cpu(), torch.tensor([cold_hits, union], dtype=torch.int64))
    for _ in range(10):
        graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(sc.cpu(), torch.zeros(query_rows, dtype=torch.int32))
    torch.testing.assert_close(stats.cpu(), torch.tensor([cold_hits + 10 * positions, union], dtype=torch.int64))
    torch.testing.assert_close(hot[1:].cpu().reshape(hot_size, width)[:union], host[block : block + union])
    torch.testing.assert_close(lru.cpu(), expected_lru)
    assert stats_backing[0].item() == -73
    assert torch.all(stats_backing[3:].cpu() == -73)
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


@pytest.mark.parametrize("hot_capacity", [4, 10240, 32768, 32769])
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


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("hot_capacity", [1, 7, 2049, 10240])
@pytest.mark.parametrize("write_repeats", [1, 8])
def test_batched_invalidation_dynamic_graph(dtype, hot_capacity, write_repeats):
    """Duplicate writes and changing states must not race or pollute neighbors."""
    torch.manual_seed(71)
    initial = torch.randint(-1, 128, (3, 4, hot_capacity), dtype=torch.int32)
    initial[:, :, 0] = 2**31 - 1
    if hot_capacity > 1:
        initial[:, :, 1] = 3
    backing = [torch.full((table.numel() + 9,), -73, dtype=torch.int32, device="npu") for table in initial]
    tables = [storage[1:-8].view_as(table) for storage, table in zip(backing, initial)]
    for table, original in zip(tables, initial):
        table.copy_(original)
    groups = torch.tensor([table.data_ptr() for table in tables], dtype=torch.int64, device="npu")
    states = torch.tensor([3, 1, 0, -1], dtype=torch.int32, device="npu")
    requests_cpu = torch.tensor([0, 0, 1, 0, 2, 3, -1, 9], dtype=torch.int32)
    writes_cpu = torch.tensor([3, 3, 2**31 - 1, 5, -1, 8, 9, 10], dtype=dtype)
    if dtype == torch.int64:
        writes_cpu[4] = 2**32 + 3  # Must not wrap to the valid tag 3.
        writes_cpu[-3:] = torch.tensor([2**34 + 3, -(2**34), 2**31], dtype=dtype)
    requests_cpu = requests_cpu.repeat(write_repeats)
    writes_cpu = writes_cpu.repeat(write_repeats)
    requests, writes = requests_cpu.npu(), writes_cpu.npu()

    def forward():
        HiSparseNPUOps.hisparse_invalidate_written_slots(groups, states, requests, writes, 4, hot_capacity)

    with _invalidation_graph(forward) as (graph, stream):
        with torch.npu.graph(graph, stream=stream):
            forward()
        for state_cpu in (torch.tensor([3, 1, 0, -1]), torch.tensor([0, 0, 2, 8])):
            states.copy_(state_cpu)
            for table, original in zip(tables, initial):
                table.copy_(original)
            graph.replay()
            torch.npu.synchronize()
            expected = initial.clone()
            for request, value in zip(requests_cpu.tolist(), writes_cpu.tolist()):
                if 0 <= request < 4 and 0 <= state_cpu[request] < 4 and 0 <= value <= 2**31 - 1:
                    row = expected[:, state_cpu[request]]
                    row[row == value] = -1
            for storage, table, wanted in zip(backing, tables, expected):
                torch.testing.assert_close(table.cpu(), wanted)
                assert storage[0].item() == -73 and torch.all(storage[-8:].cpu() == -73)


@pytest.mark.parametrize("empty", ["groups", "states", "writes", "state_count", "region"])
@pytest.mark.parametrize("empty_view", [False, True])
def test_invalidation_empty_inputs(empty, empty_view):
    table = torch.arange(7, dtype=torch.int32, device="npu")[None]
    groups = torch.tensor([table.data_ptr()], dtype=torch.int64, device="npu")
    states = torch.zeros(1, dtype=torch.int32, device="npu")
    requests, written = states.clone(), states.clone()

    def make_empty(tensor):
        return tensor[:0] if empty_view else torch.empty(0, dtype=tensor.dtype, device=tensor.device)

    if empty == "groups":
        groups = make_empty(groups)
    elif empty == "states":
        states = make_empty(states)
    elif empty == "writes":
        requests, written = make_empty(requests), make_empty(written)
    state_count, region = (0 if empty == "state_count" else 1), (0 if empty == "region" else 7)
    invalidate = HiSparseNPUOps.hisparse_invalidate_written_slots
    invalidate(groups, states, requests, written, state_count, region)
    torch.testing.assert_close(table.cpu(), torch.arange(7, dtype=torch.int32)[None])
    # Empty work must not hide an invalid tensor shape.
    with pytest.raises(ValueError, match="group_table"):
        invalidate(groups.view(-1, 1), states, requests, written, state_count, region)


@pytest.mark.parametrize("count", [64, 65])
def test_invalidation_write_count(count):
    """The shared Triton kernel has no native write-count ceiling."""
    table = torch.arange(count, dtype=torch.int32, device="npu")[None]
    groups = torch.tensor([table.data_ptr()], dtype=torch.int64, device="npu")
    states = torch.zeros(1, dtype=torch.int32, device="npu")
    requests = torch.zeros(count, dtype=torch.int32, device="npu")
    written = torch.arange(count, dtype=torch.int32, device="npu")
    HiSparseNPUOps.hisparse_invalidate_written_slots(groups, states, requests, written, 1, count)
    assert torch.all(table.cpu() == -1)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("request_count", [1, 4, 128])
@pytest.mark.parametrize("initial", [(1 << 32) - 2, (1 << 33) - 2, (1 << 53) + 1])
def test_native_resolver_int64_stats_carry(capture, request_count, initial):
    """Counters preserve carry and graph storage on Core's generic side stream."""
    block = 8
    logical = torch.tensor([[0, 0, 0, 1, 2, 3]], dtype=torch.int32, device="npu").repeat(request_count, 1)
    table = torch.arange(1, request_count + 1, dtype=torch.int32, device="npu")[:, None]
    tags = torch.full((request_count, block), -1, dtype=torch.int32, device="npu")
    order = torch.arange(block, dtype=torch.int32, device="npu").to(torch.int16).repeat(request_count, 1)
    out, resolved = torch.empty_like(logical), torch.empty_like(logical)
    errors = torch.empty((request_count, 8), dtype=torch.int32, device="npu")
    backing = torch.full((6,), -73, dtype=torch.int64, device="npu")
    stats = backing[1:3]

    def resolve():
        torch.ops._C_ascend.hisparse_resolve_residency(
            logical,
            table,
            None,
            None,
            None,
            None,
            None,
            tags,
            order,
            out,
            resolved,
            errors,
            None,
            None,
            None,
            None,
            None,
            None,
            stats,
            block,
            block,
            block,
            block,
            0,
            block,
            block,
            0,
        )

    side_stream = torch.Stream(device=logical.device)

    def forward():
        current = torch.accelerator.current_stream()
        side_stream.wait_stream(current)
        with side_stream:
            resolve()
        current.wait_stream(side_stream)

    graph = None
    try:
        if capture:
            forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                forward()
            torch.npu.synchronize()
            # Captured temporaries must survive release of the eager cache.
            torch.npu.empty_cache()
        stats.fill_(initial)
        for iteration in range(1, 4):
            # One cached row and three missing rows yield three hits/misses per request.
            tags.fill_(-1)
            tags[:, 0] = 0
            order.copy_(torch.arange(block, dtype=torch.int16)[None, :].expand(request_count, -1))
            errors.fill_(123)
            if graph is None:
                forward()
            else:
                graph.replay()
            torch.npu.synchronize()
            expected = torch.full((2,), initial + iteration * request_count * 3, dtype=torch.int64)
            torch.testing.assert_close(stats.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(errors[:, :3].cpu(), torch.tensor([[0, 3, 3]]).repeat(request_count, 1).int())
            torch.testing.assert_close(out.cpu(), logical.cpu() + table.cpu() * block)
            torch.testing.assert_close(resolved.cpu(), logical.cpu())
            assert backing[0].item() == -73 and torch.all(backing[3:].cpu() == -73)
    finally:
        if graph is not None:
            graph.reset()


def test_native_resolver_empty_state_tables():
    logical = torch.zeros((1, 4), dtype=torch.int32, device="npu")
    table = torch.empty((0, 1), dtype=torch.int32, device="npu")
    states = torch.empty(0, dtype=torch.int32, device="npu")
    tags = torch.empty((0, 4), dtype=torch.int32, device="npu")
    order = torch.empty((0, 4), dtype=torch.int16, device="npu")
    out, resolved = torch.empty_like(logical), torch.empty_like(logical)
    errors = torch.empty((1, 8), dtype=torch.int32, device="npu")
    counts = torch.empty(1, dtype=torch.int32, device="npu")
    stats = torch.zeros(2, dtype=torch.int64, device="npu")
    torch.ops._C_ascend.hisparse_resolve_residency(
        logical,
        table,
        None,
        states,
        table,
        table,
        None,
        tags,
        order,
        out,
        resolved,
        errors,
        None,
        counts,
        None,
        None,
        None,
        None,
        stats,
        8,
        4,
        4,
        4,
        0,
        4,
        4,
        0,
    )
    assert torch.all(out.cpu() == -1) and torch.all(resolved.cpu() == -1)
    assert torch.all(counts.cpu() == 0) and torch.all(stats.cpu() == 0)
    assert torch.all(errors.cpu() == 0)


@pytest.mark.parametrize("state_count", [1, 4])
@pytest.mark.parametrize("mask_dtype", [torch.uint8, torch.int32])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("resident_base", [256, 131073])
@pytest.mark.parametrize("cols", [9, 1024])
def test_resident_fast_path_dynamic_graph(state_count, mask_dtype, compact, resident_base, cols):
    """Replay mixed tables with unselected/selected cold pages and padded rows."""
    torch.manual_seed(19)
    rows, topk, hot, block = 4, 2048, 4096, 128
    source_cpu = torch.arange(1, state_count * cols + 1, dtype=torch.int32).view(state_count, cols)
    hot_table = torch.arange(1, state_count * (hot // block) + 1, dtype=torch.int32).view(state_count, -1).npu()
    states_cpu = torch.arange(state_count - 1, -1, -1, dtype=torch.int32)
    requests_cpu = torch.tensor([0, 0, 0, 0] if state_count == 1 else [0, 0, 1, 1], dtype=torch.int32)
    indices_cpu = torch.randint(-1, (cols - 1) * block, (rows, topk), dtype=torch.int32)
    indices_cpu[:, :5] = torch.tensor([-1, -(2**31), 2**31 - 1, cols * block, 0])
    indices_cpu[-1].fill_(-1)
    indices, requests, states, source = (x.npu() for x in (indices_cpu, requests_cpu, states_cpu, source_cpu))
    resident = torch.empty_like(source)
    dgi = torch.empty((state_count, hot), dtype=torch.int32, device="npu")
    lru_cpu = torch.arange(hot, dtype=torch.int16).repeat(state_count, 1)
    lru = lru_cpu.npu()
    out, attention, resolved, swap_host, swap_device = (torch.empty_like(indices) for _ in range(5))
    counts, swap_counts = (torch.empty(rows, dtype=torch.int32, device="npu") for _ in range(2))
    mask = torch.empty_like(indices, dtype=mask_dtype)
    stats = torch.zeros(2, dtype=torch.int64, device="npu")
    errors = torch.empty((rows, 8), dtype=torch.int32, device="npu")
    valid_rows = torch.tensor([rows], dtype=torch.int32, device="npu")
    capacity = (resident_base + state_count * cols) * 256

    def forward():
        torch.ops._C_ascend.hisparse_resolve_residency(
            indices,
            hot_table,
            requests,
            states,
            source,
            resident,
            valid_rows,
            dgi,
            lru,
            out,
            resolved,
            errors,
            attention,
            counts,
            swap_host,
            swap_device,
            swap_counts,
            mask,
            stats,
            (state_count * cols + 1) * block,
            block,
            block,
            block,
            0,
            256,
            hot,
            capacity if compact else 0,
        )

    resident.fill_(resident_base)
    dgi.fill_(-1)
    with _invalidation_graph(forward) as (graph, stream):
        with torch.npu.graph(graph, stream=stream):
            forward()
        for cold_page, cold_row, live_rows in (
            (None, None, rows),
            (cols - 1, None, rows),
            (cols - 1, 1, rows),
            (cols - 1, 1, 1),
            (0, None, rows),
            (None, None, 2),
            (None, None, rows),
        ):
            resident_cpu = torch.arange(resident_base, resident_base + state_count * cols, dtype=torch.int32).view(
                state_count, cols
            )
            if cold_page is not None:
                resident_cpu[:, cold_page] = 0
            current_indices = indices_cpu.clone()
            if cold_row is not None:
                assert cold_page is not None
                current_indices[cold_row, 0] = cold_page * block
            indices.copy_(current_indices)
            oracle_indices = current_indices.clone()
            oracle_indices[(oracle_indices < 0) | (oracle_indices >= cols * block)] = -1
            resident.copy_(resident_cpu)
            valid_rows.fill_(live_rows)
            dgi.fill_(-1)
            lru.copy_(lru_cpu)
            stats.zero_()
            mask.fill_(1)
            graph.replay()
            torch.npu.synchronize()
            live_indices = oracle_indices.clone()
            live_indices[live_rows:] = -1
            expected = _reference_hisparse_resolve_residency(
                live_indices,
                requests_cpu,
                states_cpu,
                source_cpu,
                resident_cpu,
                hot_table.cpu(),
                torch.full_like(dgi.cpu(), -1),
                lru_cpu,
                block,
            )
            expected_out, expected_count, sh, sd, sc, tags, order, expected_stats = expected
            expected_attention = torch.where(expected_out >= 0, expected_out // block * 256 + expected_out % block, -1)
            if compact:
                for row in range(rows):
                    selected = expected_attention[row][expected_attention[row] >= 0]
                    expected_attention[row].fill_(-1)
                    expected_attention[row, : len(selected)] = selected
            for actual, wanted in (
                (out, expected_out),
                (attention, expected_attention),
                (counts, expected_count),
                (swap_counts, sc),
                (dgi, tags),
                (lru, order),
                (stats, expected_stats),
            ):
                torch.testing.assert_close(actual.cpu(), wanted)
            expected_resolved = torch.full_like(oracle_indices, -1)
            expected_mask = torch.zeros_like(oracle_indices, dtype=mask_dtype)
            seen = set()
            for row in range(rows):
                request = int(requests_cpu[row])
                for col, token in enumerate(live_indices[row].tolist()):
                    if token < 0:
                        continue
                    page, offset = divmod(token, block)
                    host = int(source_cpu[request, page]) * block + offset
                    expected_resolved[row, col] = host
                    if resident_cpu[states_cpu[request], page] == 0 and (request, host) not in seen:
                        expected_mask[row, col] = 1
                        seen.add((request, host))
                n = int(sc[row])
                torch.testing.assert_close(swap_host[row, :n].cpu(), sh[row, :n])
                torch.testing.assert_close(swap_device[row, :n].cpu(), sd[row, :n])
            torch.testing.assert_close(resolved.cpu(), expected_resolved)
            torch.testing.assert_close(mask.cpu(), expected_mask)
            assert not errors[:, 0].cpu().any()


@pytest.mark.parametrize("num_tokens", [4096, 8192])
def test_prefill_invalidation_sorted_graph(num_tokens):
    """Large writes use the same request/slot semantics as the CPU oracle."""
    torch.manual_seed(42)
    groups, num_states, hot_capacity = 3, 5, 10240
    initial = torch.randint(-1, 20000, (groups, num_states, hot_capacity), dtype=torch.int32)
    initial[0].fill_(-1)
    initial[1, 1, 0] = 2**31 - 1
    initial[2, 2, 0] = 2**31 - 2
    tables = [table.to("npu") for table in initial]
    pointers = torch.tensor([table.data_ptr() for table in tables], dtype=torch.int64, device="npu")
    states = torch.tensor([2, 1, -1, 4], dtype=torch.int32)
    requests = torch.randint(-1, 5, (num_tokens,), dtype=torch.int32)
    written = torch.randint(-1, 20000, (num_tokens,), dtype=torch.int64)
    requests[:5] = torch.tensor([1, 0, 2, 3, 1])
    written[:5] = torch.tensor([2**31 - 1, 2**31 - 2, 1, 2**34, -(2**34)])
    device_states, device_requests, device_written = (value.to("npu") for value in (states, requests, written))

    def forward():
        HiSparseNPUOps.hisparse_invalidate_written_slots(
            pointers, device_states, device_requests, device_written, num_states, hot_capacity
        )

    def check():
        expected = initial.clone()
        for request, slot in zip(requests.tolist(), written.tolist()):
            if 0 <= request < len(states) and 0 <= states[request] < num_states and 0 <= slot <= 2**31 - 1:
                row = expected[:, states[request]]
                row[row == slot] = -1
        for group, table in enumerate(tables):
            torch.testing.assert_close(table.cpu(), expected[group], rtol=0, atol=0)

    forward()
    check()
    with _invalidation_graph(forward) as (graph, stream):
        with torch.npu.graph(graph, stream=stream):
            forward()
        for _ in range(3):
            for group, table in enumerate(tables):
                table.copy_(initial[group])
            requests = (requests + 1) % 5 - 1
            device_requests.copy_(requests)
            graph.replay()
            torch.npu.synchronize()
            check()


@pytest.mark.parametrize("valid_count", [0, 1023, 2048])
def test_resident_parallel_rows_preserve_output_neighbors(valid_count):
    """One request spans AIVs; strided scalar and row outputs keep guards intact."""
    rows, topk, hot, block = 17, 2048, 10240, 128
    logical_cpu = torch.arange(topk, dtype=torch.int32).repeat(rows, 1)
    logical_cpu[:, valid_count:] = -1
    logical = logical_cpu.npu()
    requests = torch.zeros(rows, dtype=torch.int32, device="npu")
    source = torch.arange(1, 33, dtype=torch.int32, device="npu")[None]
    resident = source + 131072
    hot_table = torch.arange(1, hot // block + 1, dtype=torch.int32, device="npu")[None]
    tags = torch.full((1, hot), -1, dtype=torch.int32, device="npu")
    order_cpu = torch.arange(hot, dtype=torch.int16)[None]
    order = order_cpu.npu()
    out, resolved, sh, sd = [torch.empty_like(logical) for _ in range(4)]
    attention_backing = torch.full((rows, topk + 8), -73, dtype=torch.int32, device="npu")
    attention = attention_backing[:, :topk]
    count_backing = torch.full((rows * 2,), -73, dtype=torch.int32, device="npu")
    counts = count_backing[::2]
    swaps = torch.empty(rows, dtype=torch.int32, device="npu")
    errors = torch.empty((rows, 8), dtype=torch.int32, device="npu")
    live = torch.tensor([rows], dtype=torch.int32, device="npu")

    def forward():
        torch.ops._C_ascend.hisparse_resolve_residency(
            logical,
            hot_table,
            requests,
            None,
            source,
            resident,
            live,
            tags,
            order,
            out,
            resolved,
            errors,
            attention,
            counts,
            sh,
            sd,
            swaps,
            None,
            None,
            33 * block,
            block,
            block,
            block,
            0,
            256,
            8192,
            131106 * 256,
        )

    with _invalidation_graph(forward) as (graph, stream):
        with torch.npu.graph(graph, stream=stream):
            forward()
        for live_count in (rows, 3, rows):
            live.fill_(live_count)
            for _ in range(5):
                graph.replay()
            torch.npu.synchronize()
            expected_resolved = logical_cpu + block
            expected_out = logical_cpu + 131073 * block
            expected_attention = expected_out // block * 256 + expected_out % block
            for expected in (expected_resolved, expected_out, expected_attention):
                expected[:, valid_count:] = -1
                expected[live_count:] = -1
            torch.testing.assert_close(resolved.cpu(), expected_resolved)
            torch.testing.assert_close(out.cpu(), expected_out)
            torch.testing.assert_close(attention.cpu(), expected_attention)
            expected_counts = torch.zeros(rows, dtype=torch.int32)
            expected_counts[:live_count] = valid_count
            torch.testing.assert_close(counts.cpu(), expected_counts)
            assert torch.all(attention_backing[:, topk:] == -73)
            assert torch.all(count_backing[1::2] == -73)
            assert torch.all(swaps == 0) and torch.all(errors == 0)
            assert torch.all(tags == -1)
            torch.testing.assert_close(order.cpu(), order_cpu)


@pytest.mark.parametrize("count", [1, 2047])
def test_batch_dma_to_mapped_host(count):
    """The existing D2H ABI writes the same allocation that AIV reads."""
    if not hasattr(torch.ops._C_ascend, "swap_blocks_batch"):
        import vllm_ascend.vllm_ascend_C  # noqa: F401
    host = allocate_mapped_host((count, 576), torch.bfloat16)
    source = (torch.arange(count, device="npu") % 97).to(torch.bfloat16)[:, None].expand(count, 576).contiguous()
    row_bytes = 576 * source.element_size()
    src = torch.arange(count, dtype=torch.int64) * row_bytes + source.data_ptr()
    dst = torch.arange(count, dtype=torch.int64) * row_bytes + host.data_ptr()
    sizes = torch.full((count,), row_bytes, dtype=torch.int64)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    try:
        with torch.npu.stream(stream):
            torch.ops._C_ascend.swap_blocks_batch(src, dst, sizes, 1)
        stream.synchronize()
        torch.testing.assert_close(host, source.cpu())
    finally:
        torch.npu.synchronize()
        host._hisparse_registration.close()


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_union_overflow_preserves_state(capture, compact):
    """An oversized MTP union must fail before evicting an existing Hot row."""
    logical = torch.arange(8, dtype=torch.int32, device="npu").view(2, 4)
    requests = torch.zeros(2, dtype=torch.int32, device="npu")
    table = torch.ones((1, 1), dtype=torch.int32, device="npu")
    tags = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32, device="npu")
    lru = torch.arange(4, dtype=torch.int16)[None, :].npu()
    out, resolved, sh, sd = (torch.empty_like(logical) for _ in range(4))
    attention = torch.empty_like(logical) if compact else None
    counts, swaps = (torch.empty(2, dtype=torch.int32, device="npu") for _ in range(2))
    errors = torch.zeros((2, 8), dtype=torch.int32, device="npu")
    stats = torch.zeros(2, dtype=torch.int64, device="npu")
    original_tags, original_lru = tags.clone(), lru.clone()

    def forward():
        torch.ops._C_ascend.hisparse_resolve_residency(
            logical,
            table,
            requests,
            None,
            None,
            None,
            None,
            tags,
            lru,
            out,
            resolved,
            errors,
            attention,
            counts,
            sh,
            sd,
            swaps,
            None,
            stats,
            32,
            4,
            4,
            4,
            0,
            4,
            4,
            32 if compact else 0,
        )

    graph = None
    try:
        forward()
        torch.npu.synchronize()
        if capture:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                forward()
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(tags, original_tags)
        torch.testing.assert_close(lru, original_lru)
        assert errors[0, 0].item() != 0
        assert torch.all(out == -1).item()
        if attention is not None:
            assert torch.all(attention == -1).item()
        assert torch.all(counts == 0).item()
        assert torch.all(swaps == 0).item()
        assert torch.all(stats == 0).item()
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
