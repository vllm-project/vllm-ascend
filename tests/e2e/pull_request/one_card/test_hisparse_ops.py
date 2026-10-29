# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-NPU HiSparse state and mapped-payload tests against a CPU oracle."""

import json

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.hisparse import (
    HiSparseNPUOps,
    MappedHostTensor,
    ResolverWorkspace,
    allocate_mapped_host,
    pack_attention_rows,
    scatter_packed_kv,
)


@pytest.fixture(autouse=True)
def _require_mapped_host_device():
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("HiSparse mapped-host operators currently target Ascend 950")
    torch.npu.set_device(0)


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
        misses = []
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


def test_resolver_strides_padding_graph():
    torch.npu.set_device(0)
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
    ops = HiSparseNPUOps(mapping, workspace)
    topk_n, requests_n, states_n, source_n, resident_n, hot_table_n, dgi_n, lru_n = [
        t.npu() for t in (topk, requests, states, source, resident, hot_table, dgi, lru)
    ]
    out = torch.empty_like(topk_n)
    attn = torch.empty_like(topk_n)
    counts = torch.empty(4, dtype=torch.int32, device="npu")
    sh, sd = torch.full_like(out, -9), torch.full_like(out, -9)
    sc = torch.empty_like(counts)
    stats = torch.zeros(2, dtype=torch.int64, device="npu")

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
        )
        ops.hisparse_gather_compact(host, hot, sh, sd, sc)

    forward()
    torch.npu.synchronize()
    actual = (out, counts, sh, sd, sc, dgi_n, lru_n, stats)
    for name, a, e in zip(
        ("indices", "counts", "swap_host", "swap_device", "swap_counts", "dgi", "lru", "stats"), actual, expected
    ):
        torch.testing.assert_close(a.cpu(), e, msg=name)
    expected_attn = torch.where(expected[0] >= 0, expected[0] // 4 * 8 + expected[0] % 4, -1)
    torch.testing.assert_close(attn.cpu(), expected_attn)
    for row in range(4):
        for col in range(int(expected[4][row])):
            src, dst = int(expected[2][row, col]), int(expected[3][row, col])
            torch.testing.assert_close(hot[dst // 4, dst % 4].cpu(), host[src])
    torch.testing.assert_close(backing[:, 0].cpu(), torch.full((10, 4, 16), -17.0))
    print(json.dumps({"resolver_mtp_union_lru_compact_gather": "PASS"}), flush=True)
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
    print(json.dumps({"pack_scatter_strides_padding": "PASS"}), flush=True)
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
    print(json.dumps({"acl_graph_replay_mapped_host_update": "PASS"}), flush=True)
    for _ in range(100):
        graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected[0])
    torch.testing.assert_close(dgi_n.cpu(), expected[5])
    torch.testing.assert_close(lru_n.cpu(), expected[6])
    print(json.dumps({"repeated_graph_replay": "PASS", "iterations": 100}), flush=True)
    graph.reset()
    mapping.close()


@pytest.mark.parametrize("query_rows,hot_size", [(2, 4096), (4, 10240)])
def test_topk_2048_speculative_union_graph(query_rows, hot_size):
    topk, width, block = 2048, 576, 128
    union = query_rows * topk
    blocks = hot_size // block + 1
    host = allocate_mapped_host((hot_size + block, width), torch.bfloat16)
    mapping = host._hisparse_registration
    host.copy_((torch.arange(host.size(0)) % 97).to(torch.bfloat16)[:, None].expand_as(host))
    hot = torch.zeros((blocks, block, width), dtype=torch.bfloat16, device="npu")
    indices = torch.arange(union, dtype=torch.int32, device="npu").view(query_rows, topk)
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

    def forward():
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
        )
        ops.hisparse_gather_compact(host, hot, sh, sd, sc)

    forward()
    torch.npu.synchronize()
    expected = torch.arange(block, block + union, dtype=torch.int32).view(query_rows, topk)
    torch.testing.assert_close(out.cpu(), expected)
    torch.testing.assert_close(sc.cpu(), torch.full((query_rows,), topk, dtype=torch.int32))
    torch.testing.assert_close(stats.cpu(), torch.tensor([0, union], dtype=torch.int64))
    torch.testing.assert_close(hot[1:].cpu().reshape(hot_size, width)[:union], host[block : block + union])
    expected_lru = torch.cat((torch.arange(union, hot_size), torch.arange(union))).to(torch.int16)[None, :]
    torch.testing.assert_close(lru.cpu(), expected_lru)
    forward()
    torch.npu.synchronize()
    torch.testing.assert_close(sc.cpu(), torch.zeros(query_rows, dtype=torch.int32))
    torch.testing.assert_close(stats.cpu(), torch.tensor([union, union], dtype=torch.int64))
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        forward()
    dgi.fill_(-1)
    lru.copy_(torch.arange(hot_size, dtype=torch.int16)[None, :])
    stats.zero_()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), expected)
    torch.testing.assert_close(sc.cpu(), torch.full((query_rows,), topk, dtype=torch.int32))
    torch.testing.assert_close(stats.cpu(), torch.tensor([0, union], dtype=torch.int64))
    for _ in range(10):
        graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(sc.cpu(), torch.zeros(query_rows, dtype=torch.int32))
    torch.testing.assert_close(stats.cpu(), torch.tensor([10 * union, union], dtype=torch.int64))
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
            graph.replay() if capture else forward()
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


def test_plan_gather_and_written_slot_invalidation():
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
        dgi = torch.tensor([[4, 7, 9, -1], [11, 13, 15, -1]], dtype=torch.int32, device="npu")
        states = torch.tensor([1, 0], dtype=torch.int32, device="npu")
        reqs = torch.tensor([0, 1, 1, -1], dtype=torch.int32, device="npu")
        written = torch.tensor([13, 4, 9, 11], dtype=torch.int64, device="npu")
        ops.hisparse_invalidate_written_slots(dgi, states, reqs, written)
        torch.npu.synchronize()
        torch.testing.assert_close(dgi.cpu(), torch.tensor([[-1, 7, -1, -1], [11, -1, 15, -1]], dtype=torch.int32))
    finally:
        torch.npu.synchronize()
        mapping.close()


def test_upstream_prefill_staging_plan():
    from vllm.v1.hisparse.runtime import build_hisparse_prefill_staging_plan

    host = allocate_mapped_host((32, 16), torch.float32)
    mapping = host._hisparse_registration
    host.copy_(torch.arange(host.numel(), dtype=torch.float32).view_as(host))
    table = torch.tensor([[2, 1, 0], [3, 0, 0]], dtype=torch.int32, device="npu")
    lengths = torch.tensor([5, 4], dtype=torch.int32, device="npu")
    try:
        plan = build_hisparse_prefill_staging_plan(table, lengths, 4, 3)
        assert plan.miss_mask.dtype == torch.int32
        hot = torch.full((plan.row_ids.size(1) // 4, 4, 16), -17.0, device="npu")
        ops = HiSparseNPUOps(mapping)
        ops.hisparse_gather_plan(host, hot, plan.row_ids, plan.dst_rows, plan.miss_mask, None, None, 0)
        torch.npu.synchronize()
        source = plan.row_ids.cpu()[0]
        expected = torch.full((source.numel(), 16), -17.0)
        expected[source >= 0] = host[source[source >= 0].long()]
        torch.testing.assert_close(hot.cpu().view_as(expected), expected)
    finally:
        torch.npu.synchronize()
        mapping.close()


def test_invalidate_large_hot_capacity_graph():
    dgi = torch.arange(20480, dtype=torch.int32, device="npu").view(2, 10240)
    initial = dgi.cpu()
    states = torch.tensor([1, 0], dtype=torch.int32, device="npu")
    requests = torch.tensor([0, 1, 1, -1], dtype=torch.int32, device="npu")
    written = torch.tensor([20479, 0, 10239, 10240], dtype=torch.int64, device="npu")
    expected = initial.clone()
    expected[1, -1] = expected[0, 0] = expected[0, -1] = -1

    def forward():
        HiSparseNPUOps.hisparse_invalidate_written_slots(dgi, states, requests, written)

    forward()
    torch.npu.synchronize()
    torch.testing.assert_close(dgi.cpu(), expected)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph, stream=stream):
            forward()
        for _ in range(10):
            dgi.copy_(initial)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(dgi.cpu(), expected)
    finally:
        torch.npu.synchronize()
        graph.reset()
