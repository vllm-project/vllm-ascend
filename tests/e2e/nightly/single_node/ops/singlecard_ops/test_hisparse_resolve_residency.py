# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse residency resolver single-op accuracy tests on A2/A3 and A5.

The operator under test is the native ACLNN kernel registered by
csrc/attention/hisparse_resolve_residency
(torch.ops._C_ascend.hisparse_resolve_residency). The A2/A3 arch22 kernel and
the A5 SIMT kernel share one ABI, so both run against the same pure-torch CPU
reference of the resolution semantics; no vllm_ascend Python plumbing is
involved beyond loading the native library.
"""

import ctypes
from contextlib import contextmanager
from functools import partial

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.hisparse.cache_ops import HiSparseNPUOps, ResolverWorkspace
from vllm_ascend.hisparse.host_pool import _ACL_HAC_AIV, allocate_mapped_host
from vllm_ascend.utils import bootstrap_custom_op_env, vllm_version_is

DEVICE_TYPE = get_ascend_device_type()
A5_ONLY = pytest.mark.skipif(
    vllm_version_is("0.33.0") or DEVICE_TYPE != AscendDeviceType.A5,
    reason="The A5 resolver is a SIMT-only kernel and HiSparse e2e is not supported on vLLM v0.33.0",
)
_ACL_ERROR_RT_FEATURE_NOT_SUPPORT = 207000


def _arch22_only(test):
    """Skip outside A2/A3 and load the arch22 extension before the test."""
    test = pytest.mark.skipif(
        vllm_version_is("0.33.0") or DEVICE_TYPE not in (AscendDeviceType.A2, AscendDeviceType.A3),
        reason="The arch22 resolver kernel targets Ascend A2/A3 and HiSparse e2e is not supported on vLLM v0.33.0",
    )(test)
    return pytest.mark.usefixtures("arch22_kernel")(test)


@pytest.fixture
def arch22_kernel():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    if not hasattr(torch.ops._C_ascend, "hisparse_resolve_residency"):
        import vllm_ascend.vllm_ascend_C  # type: ignore[import-not-found, import-untyped]  # noqa: F401
    assert hasattr(torch.ops._C_ascend, "hisparse_resolve_residency"), "rebuild the arch22 extension"


@pytest.fixture
def mapped_host_allocator(arch22_kernel):
    """Prefer native mapping; require working MemFabric when it is unsupported."""
    device = torch.npu.current_device()
    acl = ctypes.CDLL("libascendcl.so")
    query = acl.aclrtHostMemMapCapabilities
    query.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    query.restype = ctypes.c_int
    capability = ctypes.c_int()
    status = query(device, _ACL_HAC_AIV, ctypes.byref(capability))
    if status == _ACL_ERROR_RT_FEATURE_NOT_SUPPORT or (status == 0 and capability.value == 0):
        # Select the existing backend explicitly for these tests. Allocation
        # failures must propagate, including a missing/unusable MemFabric SDK.
        return partial(allocate_mapped_host, memfabric=True)
    if status != 0:
        raise RuntimeError(f"AIV Host mapping capability query failed unexpectedly: status={status}")
    return allocate_mapped_host


if not vllm_version_is("0.33.0"):
    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: E402,F401

    resolve = torch.ops._C_ascend.hisparse_resolve_residency

SEED = 45


def reference(topk, requests, states, source, resident, hot_table, dgi, lru, block):
    """Pure-torch CPU golden model of the resolver semantics.

    Per request: resident tokens map through the resident block table,
    misses form a request union; LRU slots whose device-global index is
    in the union are protected, the oldest non-protected slots serve the
    misses in union order, and the LRU order rotates the evicted slots
    behind the hits.
    """
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
            lru[state] = torch.tensor(evict[len(misses) :] + evict[: len(misses)] + hits, dtype=lru.dtype)
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


@contextmanager
def _replay_graph(forward):
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


def run_resolver(
    logical_cpu,
    hot_table_cpu,
    requests_cpu,
    states_cpu,
    source_cpu,
    resident_cpu,
    dgi_cpu,
    lru_cpu,
    *,
    block,
    union_rows,
    host_rows=64,
    attention_capacity=0,
    attention_block_stride=0,
    valid_rows=None,
):
    device = torch.device("npu:0")
    if isinstance(valid_rows, int):
        valid_rows = torch.tensor([valid_rows], dtype=torch.int32, device=device)
    tensors = [t.npu() for t in (logical_cpu, hot_table_cpu, requests_cpu, states_cpu, source_cpu, resident_cpu)]
    logical, hot_table, requests, states, source, resident = tensors
    dgi = dgi_cpu.npu()
    lru = lru_cpu.npu()
    rows, topk = logical.shape
    errors = torch.zeros((rows, 8), dtype=torch.int32, device=device)
    output = torch.empty_like(logical)
    resolved = torch.empty_like(logical)
    counts = torch.empty(rows, dtype=torch.int32, device=device)
    sh = torch.full_like(output, -9)
    sd = torch.full_like(output, -9)
    sc = torch.empty_like(counts)
    stats = torch.zeros(2, dtype=torch.int64, device=device)
    attention = None
    if attention_capacity:
        attention = torch.empty_like(logical)
    # The tiling stage requires attentionBlockStride >= hotBlockSize even
    # when no attention output is requested.
    if attention_block_stride < block:
        attention_block_stride = block
    resolve(
        logical,
        hot_table,
        requests,
        states,
        source,
        resident,
        valid_rows,
        dgi,
        lru,
        output,
        resolved,
        errors,
        attention,
        counts,
        sh,
        sd,
        sc,
        None,
        stats,
        host_rows,
        block,
        block,
        block,
        0,
        attention_block_stride,
        union_rows,
        attention_capacity,
    )
    torch.npu.synchronize()
    return (output, counts, sh, sd, sc, dgi, lru, stats), attention


def assert_matches_reference(actual, expected, msg=""):
    for name, a, e in zip(
        ("indices", "counts", "swap_host", "swap_device", "swap_counts", "dgi", "lru", "stats"), actual, expected
    ):
        torch.testing.assert_close(a.cpu(), e, msg=f"{msg}{name}")


@A5_ONLY
@pytest.mark.e2e_coverage(
    arch="",
    feature="",
    parallel="",
    deploy="",
    hardware="A5",
    quantization="",
    graph_mode="eager",
)
def test_hisparse_resolve_residency_matches_torch_reference_mixed():
    """Two requests, two states, resident hits, LRU hits, misses, padding."""
    block = 4
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
    dgi = torch.cat(
        (
            torch.tensor([[20, -1, -1, -1, -1, -1, -1, -1], [8, 99, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
            torch.full((2, 8), -1, dtype=torch.int32),
        ),
        dim=1,
    )
    lru = torch.cat(
        (
            torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [3, 0, 5, 1, 7, 4, 2, 6]], dtype=torch.int16),
            torch.arange(8, 16, dtype=torch.int16)[None, :].expand(2, -1),
        ),
        dim=1,
    )
    expected = reference(topk, requests, states, source, resident, hot_table, dgi, lru, block)
    actual, _ = run_resolver(topk, hot_table, requests, states, source, resident, dgi, lru, block=block, union_rows=16)
    assert_matches_reference(actual, expected)


@A5_ONLY
@pytest.mark.e2e_coverage(
    arch="",
    feature="",
    parallel="",
    deploy="",
    hardware="A5",
    quantization="",
    graph_mode="eager",
)
@pytest.mark.parametrize("attention_capacity", [0, 32])
def test_hisparse_resolve_residency_matches_torch_reference_all_resident(attention_capacity):
    """Every token is resident, so no union forms and no slot moves."""
    block = 4
    rows, topk = 3, 8
    logical = torch.arange(rows * topk, dtype=torch.int32).view(rows, topk) % 16
    requests = torch.tensor([0, 0, 1], dtype=torch.int32)
    states = torch.tensor([0, 1], dtype=torch.int32)
    source = torch.arange(1, 17, dtype=torch.int32).view(2, 8)
    resident = torch.arange(1, 17, dtype=torch.int32).view(2, 8)
    hot_table = torch.arange(1, 17, dtype=torch.int32).view(2, 8)
    dgi = torch.full((2, 16), -1, dtype=torch.int32)
    lru = torch.arange(16, dtype=torch.int16)[None, :].expand(2, 16).contiguous()
    expected = reference(logical, requests, states, source, resident, hot_table, dgi, lru, block)
    actual, attention = run_resolver(
        logical,
        hot_table,
        requests,
        states,
        source,
        resident,
        dgi,
        lru,
        block=block,
        union_rows=16,
        attention_capacity=attention_capacity,
        attention_block_stride=8,
    )
    if attention_capacity:
        physical = actual[0].cpu()
        mapped = torch.where(physical >= 0, physical // block * 8 + physical % block, -1)
        # Rows beyond the attention capacity are reported as invalid (-1).
        mapped = torch.where(mapped < attention_capacity, mapped, -1)
        torch.testing.assert_close(attention.cpu(), mapped)
        # In compact mode Counts reports how many tokens survived the
        # attention-capacity mapping, not the per-row valid token count.
        torch.testing.assert_close(actual[1].cpu(), (mapped >= 0).sum(dim=1).to(torch.int32), msg="counts")
        for name, a, e in zip(
            ("indices", "swap_host", "swap_device", "swap_counts", "dgi", "lru", "stats"),
            (actual[0], *actual[2:]),
            (expected[0], *expected[2:]),
        ):
            torch.testing.assert_close(a.cpu(), e, msg=name)
    else:
        assert_matches_reference(actual, expected)


@A5_ONLY
@pytest.mark.e2e_coverage(
    arch="",
    feature="",
    parallel="",
    deploy="",
    hardware="A5",
    quantization="",
    graph_mode="eager",
)
def test_hisparse_resolve_residency_matches_torch_reference_single_warp_scan():
    """One state, a full LRU turn and duplicates; tiles<=32 register scan."""
    block = 16
    rows, topk, hot_size = 4, 32, 512
    torch.manual_seed(SEED)
    logical = torch.randint(0, hot_size, (rows, topk), dtype=torch.int32)
    logical[2, :4] = -1  # padding tokens
    requests = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    states = torch.tensor([0], dtype=torch.int32)
    pages = hot_size // block
    source = torch.arange(1, pages + 1, dtype=torch.int32)[None, :]
    resident = torch.zeros(1, pages, dtype=torch.int32)
    resident[0, : pages // 4] = 100 + torch.arange(pages // 4, dtype=torch.int32)
    hot_table = torch.arange(1, pages + 1, dtype=torch.int32)[None, :]
    dgi = torch.full((1, hot_size), -1, dtype=torch.int32)
    # Half of the LRU window already holds part of the upcoming union.
    prefill = torch.unique(logical % hot_size)[: hot_size // 4]
    dgi[0, : len(prefill)] = prefill
    lru = torch.cat((torch.arange(hot_size - len(prefill), hot_size), torch.arange(hot_size - len(prefill))), dim=0).to(
        torch.int16
    )[None, :]
    expected = reference(logical, requests, states, source, resident, hot_table, dgi, lru, block)
    actual, _ = run_resolver(
        logical,
        hot_table,
        requests,
        states,
        source,
        resident,
        dgi,
        lru,
        block=block,
        union_rows=min(hot_size, rows * topk),
        host_rows=hot_size * 4,
    )
    assert_matches_reference(actual, expected)


@A5_ONLY
@pytest.mark.e2e_coverage(
    arch="",
    feature="",
    parallel="",
    deploy="",
    hardware="A5",
    quantization="",
    graph_mode="eager",
)
def test_hisparse_resolve_residency_matches_torch_reference_iterative_scan():
    """A union spanning 64 tiles exercises the multi-batch prefix path."""
    block = 128
    rows, topk, hot_size = 4, 256, 2048
    union_size = 768  # more than rows*topk/2, duplicates across the union
    logical = torch.arange(rows * topk, dtype=torch.int32).view(rows, topk) % union_size
    requests = torch.zeros(rows, dtype=torch.int32)
    states = torch.zeros(1, dtype=torch.int32)
    pages = hot_size // block
    source = torch.arange(1, pages + 1, dtype=torch.int32)[None, :]
    resident = torch.zeros(1, pages, dtype=torch.int32)
    hot_table = torch.arange(1, pages + 1, dtype=torch.int32)[None, :]
    dgi = torch.full((1, hot_size), -1, dtype=torch.int32)
    lru = torch.arange(hot_size, dtype=torch.int16)[None, :]
    expected = reference(logical, requests, states, source, resident, hot_table, dgi, lru, block)
    actual, _ = run_resolver(
        logical,
        hot_table,
        requests,
        states,
        source,
        resident,
        dgi,
        lru,
        block=block,
        union_rows=union_size,
        host_rows=hot_size * 4,
    )
    assert_matches_reference(actual, expected)


@_arch22_only
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_hisparse_resolve_residency_random_request_unions(capture, compact, mapped_host_allocator):
    """Vary query counts, request order, stable state, duplicates and residency."""
    rows, topk, hot_size, block, width, requests_count = 40, 8, 16, 4, 16, 8
    host = mapped_host_allocator((132, width), torch.float32)
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


@_arch22_only
@pytest.mark.parametrize("query_rows,hot_size", [(2, 4096), (4, 10240), (3, 4096)])
@pytest.mark.parametrize("cross_stream", [False, True])
@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 576), (torch.int8, 656)])
def test_hisparse_resolve_residency_topk_2048_speculative_union_graph(
    query_rows, hot_size, cross_stream, dtype, width, mapped_host_allocator
):
    topk, block = 2048, 128
    positions = query_rows * topk
    union = min(positions, hot_size)
    # The 3 x 2048 case has more positions than the 4096-row mask/scan
    # capacity. Its duplicates cross tiles of the iterative miss scan.
    blocks = hot_size // block + 1
    host = mapped_host_allocator((hot_size + block, width), dtype)
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


@_arch22_only
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("request_count", [1, 4, 128])
@pytest.mark.parametrize("initial", [(1 << 32) - 2, (1 << 33) - 2, (1 << 53) + 1])
def test_hisparse_resolve_residency_int64_stats_carry(capture, request_count, initial):
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


@_arch22_only
def test_hisparse_resolve_residency_empty_state_tables():
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


@_arch22_only
@pytest.mark.parametrize("state_count", [1, 4])
@pytest.mark.parametrize("mask_dtype", [torch.uint8, torch.int32])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("resident_base", [256, 131073])
@pytest.mark.parametrize("cols", [9, 1024])
def test_hisparse_resolve_residency_resident_fast_path_dynamic_graph(
    state_count,
    mask_dtype,
    compact,
    resident_base,
    cols,
):
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
    with _replay_graph(forward) as (graph, stream):
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
            expected = reference(
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


@_arch22_only
@pytest.mark.parametrize("valid_count", [0, 1023, 2048])
def test_hisparse_resolve_residency_resident_parallel_rows_preserve_output_neighbors(valid_count):
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

    with _replay_graph(forward) as (graph, stream):
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


@_arch22_only
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_hisparse_resolve_residency_union_overflow_preserves_state(capture, compact):
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


@_arch22_only
@pytest.mark.parametrize(
    "label,cols,hot,rows,topk,resident_pages,miss_mask",
    [
        # cols <= 512 keeps both page rows in UB: the union translates a tile per
        # Gather, the miss count runs as vector compares and the emit happens in
        # the same pass as the slot assignment.
        ("vector_union", 480, 10240, 8, 512, 2, False),
        ("vector_union_all_cold", 480, 10240, 8, 512, 0, False),
        # A wider table no longer falls back to the scalar walk: the union loads
        # it in windows. At this Hot a window holds 3072 columns, so 4096 takes
        # two and 8192 (a 1M-token context) takes three. 3073 leaves a single
        # column in the last window, and 4099 over a 1024-column budget leaves
        # three, which also covers the width rounding.
        ("vector_union_wide_two_windows", 4096, 10240, 8, 512, 0, False),
        ("vector_union_wide_partial_window", 3073, 10240, 8, 512, 0, False),
        ("vector_union_wide_1m_context", 8192, 10240, 8, 512, 0, False),
        ("vector_union_wide_eight_windows", 8192, 6144, 8, 512, 0, False),
        ("vector_union_wide_odd_tail", 4099, 6144, 8, 512, 0, False),
        # The Python glue passes None for the miss mask, so the kwarg no longer
        # selects the Publish pass; this row only proves it is still accepted.
        ("vector_union_wide_mask_kwarg", 4096, 10240, 8, 512, 0, True),
        # A Hot that leaves no room for even one eight-column window falls back.
        ("small_hot", 480, 4096, 4, 512, 0, False),
        # hot too small for the emit tiles: the separate Publish pass runs.
        ("tiny_hot", 480, 512, 2, 128, 0, False),
        # The kwarg is accepted but ignored by the glue, which always passes None.
        ("mask_kwarg", 480, 10240, 8, 512, 0, True),
        # The vector union needs whole 64-lane mask repeats: SFA top-k is a
        # multiple of 128, so 128 keeps the vector path and 17 must fall back.
        ("vector_topk128", 480, 10240, 8, 128, 0, False),
        ("odd_topk_fallback", 480, 10240, 8, 17, 0, False),
        # Above 48 * kTile the int16 permutation reaches into the miss-scan
        # scratch, so the scan has to fall back to the scalar walk.
        ("large_hot_scan_fallback", 480, 16384, 4, 512, 0, False),
    ],
)
def test_hisparse_resolve_residency_union_tiles_and_fallbacks(
    label, cols, hot, rows, topk, resident_pages, miss_mask, mapped_host_allocator
):
    """Check the vector union, its page windows, and every scalar fallback.

    The production geometry (block 128, a 480-page request row) is what the
    vector union is tuned for. The wide rows load the page table in several
    masked windows; the rest disable one guard at a time so the fallbacks stay
    bit-identical to the vector path.
    """
    block, width, requests_count = 128, 16, 2
    rows_per_request = rows // requests_count
    hot_blocks = hot // block

    host = mapped_host_allocator(((cols + 1) * block, width), torch.float32)
    mapping = host._hisparse_registration
    host.copy_(torch.arange(host.numel(), dtype=torch.float32).view_as(host))
    # Layer 0 is the sentinel layer; layer 1 is the Hot cache the kernel writes.
    backing = torch.full((hot_blocks + 1, 2, block, width), -17.0, device="npu")
    hot_cache = backing[:, 1]
    workspace = ResolverWorkspace(rows, hot, rows * topk, torch.device("npu", torch.npu.current_device()))
    ops = HiSparseNPUOps(mapping, workspace)

    generator = torch.Generator().manual_seed(11)
    source = torch.arange(1, cols + 1, dtype=torch.int32).repeat(requests_count, 1)
    hot_table = torch.arange(1, hot_blocks + 1, dtype=torch.int32).repeat(requests_count, 1)
    indices_cpu = torch.randint(-1, cols * block, (rows, topk), generator=generator, dtype=torch.int32)
    requests_cpu = torch.arange(requests_count, dtype=torch.int32).repeat_interleave(rows_per_request)
    states_cpu = torch.arange(requests_count, dtype=torch.int32)
    # Sentinel 0 keeps a page cold; any positive page resolves to device memory.
    resident_cpu = torch.zeros((requests_count, cols), dtype=torch.int32)
    resident_cpu[:, :resident_pages] = 1 + torch.arange(resident_pages, dtype=torch.int32)
    dgi_cpu = torch.full((requests_count, hot), -1, dtype=torch.int32)
    lru_cpu = torch.stack([torch.randperm(hot, generator=generator) for _ in range(requests_count)]).to(torch.int16)

    indices = indices_cpu.npu()
    requests = requests_cpu.npu()
    states = states_cpu.npu()
    source_n = source.npu()
    hot_table_n = hot_table.npu()
    resident = resident_cpu.npu()
    dgi = dgi_cpu.npu()
    lru = lru_cpu.npu()
    out, sh, sd = (torch.empty_like(indices) for _ in range(3))
    counts = torch.empty(rows, dtype=torch.int32, device="npu")
    sc = torch.empty_like(counts)
    stats = torch.zeros(2, dtype=torch.int64, device="npu")
    mask = torch.empty_like(indices, dtype=torch.uint8) if miss_mask else None

    try:
        ops.hisparse_resolve_residency(
            host,
            hot_cache,
            hot_table_n,
            indices,
            out,
            dgi,
            lru,
            states,
            hot,
            rows * topk,
            stats,
            None,
            block,
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
            miss_mask=mask,
        )
        torch.npu.synchronize()
        want_out, want_counts, want_sh, want_sd, want_sc, want_dgi, want_lru, want_stats = reference(
            indices_cpu, requests_cpu, states_cpu, source, resident_cpu, hot_table, dgi_cpu, lru_cpu, block
        )
        for name, actual, wanted in (
            ("indices", out, want_out),
            ("counts", counts, want_counts),
            ("swap_counts", sc, want_sc),
            ("dgi", dgi, want_dgi),
            ("lru", lru, want_lru),
            ("stats", stats, want_stats),
        ):
            torch.testing.assert_close(actual.cpu(), wanted, msg=f"{label}: {name}")
        # The copy-back pads its last store to a 32 B block, so only the prefix
        # announced by swap_counts belongs to the contract.
        sh_cpu, sd_cpu = sh.cpu(), sd.cpu()
        for row, length in enumerate(want_sc.tolist()):
            torch.testing.assert_close(sh_cpu[row, :length], want_sh[row, :length], msg=f"{label}: swap_host row {row}")
            torch.testing.assert_close(
                sd_cpu[row, :length], want_sd[row, :length], msg=f"{label}: swap_device row {row}"
            )
    finally:
        # The registration lives in the ACL runtime, not in the tensor, and PTA
        # recycles pinned allocations: an undrained, unclosed mapping makes the
        # next aclrtHostRegister of the same range fail.
        torch.npu.synchronize()
        mapping.close()
