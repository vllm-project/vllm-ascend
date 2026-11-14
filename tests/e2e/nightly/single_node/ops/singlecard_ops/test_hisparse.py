# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse residency resolver single-op accuracy tests.

The operator under test is the native ACLNN kernel registered by
csrc/attention/hi_sparse_resolve_residency
(torch.ops._C_ascend.hisparse_resolve_residency). Accuracy is checked
against a pure-torch CPU reference of the resolution semantics; no
vllm_ascend Python plumbing is involved beyond loading the native
library.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import is_950
from vllm_ascend.utils import bootstrap_custom_op_env

pytestmark = pytest.mark.skipif(not is_950(), reason="HiSparse resolver requires Ascend 950 (A5)")

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: E402, F401

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
            lru[state] = torch.tensor(evict[len(misses):] + evict[:len(misses)] + hits, dtype=lru.dtype)
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


def run_resolver(
    logical_cpu, hot_table_cpu, requests_cpu, states_cpu, source_cpu, resident_cpu, dgi_cpu, lru_cpu,
    *, block, union_rows, host_rows=64, attention_capacity=0, attention_block_stride=0,
):
    device = torch.device("npu:0")
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
        logical, hot_table, requests, states, source, resident, None,
        dgi, lru, output, resolved, errors, attention, counts, sh, sd, sc, None, stats,
        host_rows, block, block, block, 0, attention_block_stride, union_rows, attention_capacity,
    )
    torch.npu.synchronize()
    return (output, counts, sh, sd, sc, dgi, lru, stats), attention


def assert_matches_reference(actual, expected, msg=""):
    for name, a, e in zip(
        ("indices", "counts", "swap_host", "swap_device", "swap_counts", "dgi", "lru", "stats"), actual, expected
    ):
        torch.testing.assert_close(a.cpu(), e, msg=f"{msg}{name}")


def test_resolver_matches_torch_reference_mixed():
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
    actual, _ = run_resolver(
        topk, hot_table, requests, states, source, resident, dgi, lru, block=block, union_rows=16
    )
    assert_matches_reference(actual, expected)


@pytest.mark.parametrize("attention_capacity", [0, 32])
def test_resolver_matches_torch_reference_all_resident(attention_capacity):
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
        logical, hot_table, requests, states, source, resident, dgi, lru,
        block=block, union_rows=16, attention_capacity=attention_capacity,
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


def test_resolver_matches_torch_reference_single_warp_scan():
    """One state, a full LRU turn and duplicates; tiles<=32 register scan."""
    block = 16
    rows, topk, hot_size = 4, 32, 512
    union_size = 192
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
    lru = torch.cat(
        (torch.arange(hot_size - len(prefill), hot_size), torch.arange(hot_size - len(prefill))), dim=0
    ).to(torch.int16)[None, :]
    expected = reference(logical, requests, states, source, resident, hot_table, dgi, lru, block)
    actual, _ = run_resolver(
        logical, hot_table, requests, states, source, resident, dgi, lru,
        block=block, union_rows=min(hot_size, rows * topk), host_rows=hot_size * 4,
    )
    assert_matches_reference(actual, expected)


def test_resolver_matches_torch_reference_iterative_scan():
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
        logical, hot_table, requests, states, source, resident, dgi, lru,
        block=block, union_rows=union_size, host_rows=hot_size * 4,
    )
    assert_matches_reference(actual, expected)
