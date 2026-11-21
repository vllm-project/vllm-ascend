# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP query, metadata and output helpers on every supported NPU."""

from unittest.mock import Mock

import pytest
import torch
import torch_npu  # noqa: F401

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.dcp_test_utils import (
    assert_dcp_close,
    local_visible_lengths,
    query_visible_lengths,
)
from vllm_ascend.ops.triton.dcp.dcp_a2a import fused_dcp_lse_combine, pack_dcp_output_lse
from vllm_ascend.ops.triton.query_gather_prep import prep_query_head_major
from vllm_ascend.ops.triton.sharded_indexer import (
    build_local_key_lengths,
    build_local_lengths,
)
from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton


@pytest.mark.parametrize(
    "rows,heads,nope,rope",
    [
        (0, 12, 512, 0),
        (1, 12, 512, 0),
        (17, 12, 512, 64),
        (65, 12, 129, 17),
        (129, 12, 512, 0),
        (7, 12, 0, 64),
        (1, 0, 512, 64),
        (3, 12, 0, 0),
    ],
)
@pytest.mark.parametrize("feature_stride", [1, 2])
def test_query_prep_strided_views_and_graph_replay(rows, heads, nope, rope, feature_stride):
    backing = torch.randn(rows, heads, (nope + rope) * feature_stride, dtype=torch.bfloat16, device="npu")
    query, rope_query = backing[..., ::feature_stride].split((nope, rope), -1)
    expected = torch.cat((query, rope_query), -1).transpose(0, 1).contiguous()
    actual = prep_query_head_major(query, rope_query)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if expected.numel() == 0:
        return
    for _ in range(3):
        prep_query_head_major(query, rope_query)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = prep_query_head_major(query, rope_query)
    backing.fill_(0.5)
    graph.replay()
    torch.testing.assert_close(actual, torch.full_like(expected, 0.5), atol=0, rtol=0)
    graph.reset()


@pytest.mark.parametrize("rows,nope,rope", [(1, 512, 0), (65, 129, 17)])
def test_query_prep_broadcast_positional_heads(rows, nope, rope):
    query = torch.randn(rows, 12, nope, dtype=torch.bfloat16, device="npu")
    positional = torch.randn(rows, 1, rope, dtype=query.dtype, device="npu").expand(rows, 12, rope)
    expected = torch.cat((query, positional), -1).transpose(0, 1).contiguous()
    torch.testing.assert_close(prep_query_head_major(query, positional), expected, atol=0, rtol=0)


@pytest.mark.parametrize("world,interleave", [(2, 1), (8, 128), (8, 768)])
@pytest.mark.parametrize("requests", [4, 32, 33])
def test_local_lengths_changed_graph_inputs(world, interleave, requests):
    seq = torch.tensor([0, 7, 131072, 1048576] + [1048576] * (requests - 4), dtype=torch.int32).npu()
    ends = torch.tensor([0, 2, 6, 14] + [14] * (requests - 4), dtype=torch.int32).npu()
    keys = torch.empty_like(seq)
    visible = torch.empty((17, 1), dtype=torch.int32, device="npu")
    rank = world - 1
    for _ in range(3):
        build_local_lengths(seq, ends, keys, visible, world, rank, interleave)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        build_local_lengths(seq, ends, keys, visible, world, rank, interleave)
    for new_ends in ([0, 2, 6, 14], [1, 1, 5, 12], [0, 0, 0, 0]):
        ends.copy_(torch.tensor(new_ends + [new_ends[-1]] * (requests - 4), dtype=torch.int32).npu())
        graph.replay()
        rows = torch.arange(17)
        request_ids = torch.searchsorted(ends.cpu(), rows, right=True).clamp(max=requests - 1)
        global_visible = query_visible_lengths(seq.cpu(), ends.cpu(), rows, request_ids)
        key_only = torch.empty_like(keys)
        build_local_key_lengths(seq, key_only, world, rank, interleave)
        assert torch.equal(key_only.cpu(), keys.cpu())
        assert torch.equal(keys.cpu(), local_visible_lengths(seq.cpu(), world, rank, interleave))
        assert torch.equal(visible.cpu().flatten(), local_visible_lengths(global_visible, world, rank, interleave))
    graph.reset()


@pytest.mark.parametrize(
    "tokens,heads,world,scatter_dim,dtype",
    [
        (1, 96, 8, 1, torch.bfloat16),
        (16, 96, 8, 1, torch.bfloat16),
        (256, 96, 2, 1, torch.bfloat16),
        (8, 12, 2, 0, torch.float16),
        (17, 96, 2, 1, torch.float32),
    ],
)
def test_pack_from_softmax_statistics(tokens, heads, world, scatter_dim, dtype):
    torch.manual_seed(29)
    output = torch.randn(tokens, heads, 512, dtype=dtype, device="npu")
    maxima = torch.randn(heads, tokens, 1, device="npu").transpose(0, 1)
    totals = torch.rand(tokens, heads, 2, device="npu")[..., :1] + 0.01
    totals[0, 0] = 0
    maxima[0, 0] = -torch.inf
    expected = pack_dcp_output_lse(output, maxima + totals.log(), world, scatter_dim)
    actual = pack_dcp_output_lse(output, maxima, world, scatter_dim, softmax_sum=totals)
    # The encoded mantissa is not numerically ordered. Compare its decoded
    # result through the real merge, allowing only normal FP32 log rounding.
    expected_out = fused_dcp_lse_combine(expected, 512, scatter_dim)
    actual_out = fused_dcp_lse_combine(actual, 512, scatter_dim)
    assert_dcp_close(actual_out, expected_out)


@pytest.mark.parametrize("rows", [0, 17])
def test_local_lengths_empty_requests(rows):
    empty = torch.empty(0, dtype=torch.int32, device="npu")
    visible = torch.full((rows, 1), 123, dtype=torch.int32, device="npu")
    build_local_lengths(empty, empty, empty, visible, 8, 7, 768)
    assert (visible.cpu() == 0).all()


@pytest.mark.parametrize(
    "rows,topk,world,rank,interleave,dtype",
    [
        (1, 2048, 16, 15, 768, torch.int32),
        (65, 2048, 8, 7, 1536, torch.int32),
        (6, 31, 3, 2, 127, torch.int32),
        (1, 1, 3, 0, 768, torch.int64),
        (6, 2048, 16, 15, 768, torch.int64),
        (6, 2048, 8, 7, 1536, torch.int64),
        (2, 1025, 3, 2, 127, torch.int64),
        (1, 4097, 8, 7, 1536, torch.int32),
        (1, 2049, 3, 2, 127, torch.int64),
        (2, 4096, 8, 7, 1, torch.int32),
        (1, 4096, 16, 15, 768, torch.int32),
        (2, 2048, 8, 7, 128, torch.int64),
        (2, 31, 1, 0, 768, torch.int64),
        (2, 31, 5, 4, 1, torch.int64),
        (3, 31, 17, 16, 769, torch.int32),
        (1, 127, 3, 2, 2**32 + 15, torch.int64),
        (2, 31, 8, 0, 2**30, torch.int32),
        (2, 31, 3, 0, 2**31 + 1, torch.int32),
    ],
)
def test_remap_exact_integer_boundaries_and_graph_replay(rows, topk, world, rank, interleave, dtype):
    torch.manual_seed(41)
    maximum = torch.iinfo(dtype).max
    boundaries = [0, 1, 2**24, 2**31 - 1, maximum, interleave, interleave * world]
    if dtype == torch.int64:
        boundaries += [2**40, 2**62]
    # Cross this rank's ownership boundaries at large integer magnitudes.
    for anchor in boundaries.copy():
        start = (anchor // (world * interleave) * world + rank) * interleave
        boundaries += [start, start + interleave]
    # These INT32 values set the high bit in reciprocal partial products.
    values = [-1, -2, 171821997, 725140015]
    if dtype == torch.int64:
        values += [-(2**31), -(2**32), -(2**40), torch.iinfo(dtype).min]
    values += [x + delta for x in boundaries for delta in (-1, 0, 1) if 0 <= x + delta <= maximum]
    source = torch.randint(0, maximum, (rows, 1, topk), dtype=dtype)
    edge_values = torch.tensor(values, dtype=dtype)
    source.flatten()[: min(source.numel(), edge_values.numel())] = edge_values[: source.numel()]
    if topk == 1:
        source.fill_(rank * interleave)
    original = source
    data = torch.stack((source, source), -1).npu()[..., 1]

    def expected():
        result = torch.full_like(source, -1)
        for ids, target in zip(source.reshape(rows, topk).tolist(), result.reshape(rows, topk)):
            owned = [
                x // (world * interleave) * interleave + x % interleave
                for x in ids
                if x >= 0 and x // interleave % world == rank
            ]
            target[: len(owned)] = torch.tensor(owned, dtype=dtype)
        return result

    for _ in range(3):
        actual = remap_sparse_indices_triton(data, world, rank, interleave)
    assert torch.equal(actual.cpu(), expected())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = remap_sparse_indices_triton(data, world, rank, interleave)
    small = original.clamp_min(0).remainder(1 << 20)
    replays = [original.flip(-1), small, torch.full_like(original, -1), original]
    if dtype == torch.int64 and topk > 1024:
        mixed = original.clone()
        mixed[..., 1024:] = small[..., 1024:]
        replays.insert(2, mixed)
    for source in replays:
        data.copy_(source)
        graph.replay()
        assert torch.equal(actual.cpu(), expected())
        assert torch.equal(data.cpu(), source)
        assert actual.dtype == dtype
    graph.reset()


@pytest.mark.parametrize("tokens,world,heads", [(1, 1, 12), (17, 2, 6), (129, 8, 12), (512, 8, 12)])
@pytest.mark.parametrize("with_local", [False, True])
def test_output_pack_and_combine_changed_graph_inputs(tokens, world, heads, with_local):
    torch.manual_seed(43)
    # Both feature and statistics strides differ from the timing fixtures.
    partial = torch.randn(tokens, world * heads, 1024, dtype=torch.bfloat16, device="npu")[..., ::2]
    maxima = torch.randn(world * heads, tokens, 2, device="npu").transpose(0, 1)[..., :1]
    sums = torch.rand(world * heads, tokens, 2, device="npu").transpose(0, 1)[..., :1] + 0.25

    def forward():
        packed = pack_dcp_output_lse(partial, maxima, world, 1, softmax_sum=sums)
        local_output = partial[:, :heads] if with_local else None
        local_lse = maxima[:, :heads] + sums[:, :heads].log() if with_local else None
        return packed, fused_dcp_lse_combine(packed, 512, 1, local_output=local_output, local_lse=local_lse)

    for _ in range(3):
        forward()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        packed, merged = forward()
    for empty in (False, True, False):
        maxima[0] = -torch.inf if empty else 1.0
        partial[0] = torch.nan if empty else 0.5
        graph.replay()
        p = partial.cpu().view(tokens, world, heads, 512).permute(1, 2, 0, 3)
        lse = (maxima.cpu() + sums.cpu().log()).view(tokens, world, heads).permute(1, 2, 0)
        torch.testing.assert_close(packed[..., :512].cpu(), p, atol=0, rtol=0, equal_nan=True)
        if with_local:
            p, lse = torch.cat((p, p[:1]), 0), torch.cat((lse, lse[:1]), 0)
        valid = torch.isfinite(lse)
        weights = torch.nan_to_num(lse.softmax(0), nan=0.0)
        expected = (torch.where(valid[..., None], p.float(), 0) * weights[..., None]).sum(0).transpose(0, 1)
        assert_dcp_close(merged, expected)
    graph.reset()


@pytest.mark.parametrize("tokens,local_heads", [(1, 2), (17, 2), (191, 32), (512, 32), (257, 64)])
@pytest.mark.parametrize("from_stats", [False, True])
def test_exchange_statistics_with_identity_transport(monkeypatch, tokens, local_heads, from_stats):
    from vllm_ascend.ops.triton.dcp import sfa_dcp_exchange as exchange

    # Single-device transport simulation; pack and merge execute real kernels.
    monkeypatch.setattr(exchange.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(exchange.dist, "all_to_all_single", lambda dst, src, group: dst.copy_(src))
    output = torch.randn(tokens, 2 * local_heads, 512, dtype=torch.bfloat16, device="npu")
    maxima = torch.randn(tokens, 2 * local_heads, 1, device="npu")
    sums = torch.rand_like(maxima) + 0.1 if from_stats else None
    group = object()

    def run():
        return exchange.exchange(output, maxima, group, softmax_sum=sums)

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = run()
    for invalid in (False, True):
        if sums is not None:
            sums.mul_(0.75)
        if invalid:
            maxima[0].fill_(-torch.inf)
            output[0].fill_(torch.nan)
        graph.replay()
        lse = maxima.cpu().double()
        if sums is not None:
            lse += sums.cpu().double().log()
        lse = lse.reshape(tokens, 2, local_heads).permute(1, 0, 2)
        valid = lse.isfinite()
        weights = torch.nan_to_num(lse.masked_fill(~valid, -torch.inf).softmax(0), nan=0)
        parts = output.cpu().double().reshape(tokens, 2, local_heads, 512).permute(1, 0, 2, 3)
        expected = (parts.masked_fill(~valid[..., None], 0) * weights[..., None]).sum(0)
        assert_dcp_close(actual, expected)
    graph.reset()


@pytest.mark.parametrize("mla", [False, True])
def test_dcp_registry_warmup_prevents_runtime_compile_and_replays(monkeypatch, mla):
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.triton_utils import triton

    from vllm_ascend.ops.triton.dcp import dcp_a2a, dcp_a2a_batched, dcp_warmup, sfa_dcp_exchange, sfa_dcp_merge
    from vllm_ascend.ops.triton.dcp.dcp_jit import DcpJitKernel

    # Isolate native in-memory JIT state from earlier tests. Disk-cache reuse
    # is allowed: _do_compile still detects a new dispatch specialization.
    owners, compiles = [], []
    modules = (dcp_a2a, dcp_a2a_batched, dcp_warmup, sfa_dcp_exchange, sfa_dcp_merge)
    for owner in dcp_warmup._DCP_KERNELS:
        original = owner.kernel
        kernel = triton.jit(original.fn, do_not_specialize=original.do_not_specialize)
        spy = Mock(wraps=kernel._do_compile)
        monkeypatch.setattr(kernel, "_do_compile", spy)
        compiles.append(spy)
        owners.append(DcpJitKernel(kernel, owner.dynamic_scalars))
        for module in modules:
            if getattr(module, original.fn.__name__, None) is original:
                monkeypatch.setattr(module, original.fn.__name__, kernel)
    monkeypatch.setattr(dcp_warmup, "_DCP_KERNELS", tuple(owners))
    monkeypatch.setattr("vllm.distributed.is_global_first_rank", lambda: True)
    monkeypatch.setenv("VLLM_TRITON_JIT_WARMUP_NUM_THREADS", "1")
    # Single-device identity transport; packing and merging remain real NPU
    # kernels, as in test_exchange_statistics_with_identity_transport above.
    monkeypatch.setattr(sfa_dcp_exchange.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(sfa_dcp_exchange.dist, "all_to_all_single", lambda dst, src, group: dst.copy_(src))
    torch.npu.set_device(0)
    registry = JitWarmupRegistry(None)
    with registry.activate():
        dcp_warmup.register_dcp_warmup(
            dtype=torch.bfloat16,
            local_heads=8,
            head_dim=512,
            scatter_size=2,
            pcp_size=1,
            max_tokens=2049,
            from_stats=not mla,
            mla=mla,
        )
    registry.warmup()
    assert sum(spy.call_count for spy in compiles) > 0

    def unexpected_compile(*args, **kwargs):
        pytest.fail("DCP startup warmup missed a runtime Triton specialization")

    for owner in owners:
        monkeypatch.setattr(owner.kernel, "_do_compile", unexpected_compile)

    group = object()
    captured = []
    for tokens in (1, 17, 129, 2049):
        for head_major in (False, True) if mla else (False,):
            partial = torch.randn(tokens, 16, 512, dtype=torch.bfloat16, device="npu")
            lse = torch.randn(tokens, 16, 1, device="npu")
            sums = None if mla else torch.rand_like(lse) + 0.25
            local = torch.randn(tokens, 8, 512, dtype=torch.bfloat16, device="npu") if mla else None
            local_lse = torch.randn(tokens, 8, 1, device="npu") if mla else None
            if head_major:
                partial = partial.transpose(0, 1).contiguous().transpose(0, 1)
                lse = lse.transpose(0, 1).contiguous().transpose(0, 1)
                local = local.transpose(0, 1).contiguous().transpose(0, 1)
                local_lse = local_lse.transpose(0, 1).contiguous().transpose(0, 1)

            def forward(partial=partial, lse=lse, sums=sums, local=local, local_lse=local_lse):
                exchanged = sfa_dcp_exchange.exchange(partial, lse, group, softmax_sum=sums)
                if not mla:
                    return exchanged, None
                packed = pack_dcp_output_lse(partial, lse, 2, 1)
                combined = fused_dcp_lse_combine(packed, 512, 1, local_output=local, local_lse=local_lse)
                return exchanged, combined

            # Every eager call and capture is already behind the compile guard.
            forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                results = forward()
            captured.append((graph, results, partial, lse, sums, local, local_lse))

    # Revisit older graphs after larger T changed the physical receive stride.
    for graph, results, partial, lse, sums, local, local_lse in reversed(captured):
        partial.add_(0.125)
        lse.add_(0.5)
        if sums is not None:
            sums.mul_(0.75)
        partial[0].fill_(torch.nan)
        lse[0].fill_(-torch.inf)
        if local is not None:
            local.add_(0.25)
            local[0].fill_(torch.nan)
            local_lse[0].fill_(-torch.inf)
        graph.replay()
        tokens = partial.shape[0]
        stats = lse.cpu().double()
        if sums is not None:
            stats += sums.cpu().double().log()
        stats = stats.reshape(tokens, 2, 8).permute(1, 0, 2)
        parts = partial.cpu().double().reshape(tokens, 2, 8, 512).permute(1, 0, 2, 3)
        for actual, with_local in ((results[0], False), (results[1], True)):
            if actual is None:
                continue
            values, weights_lse = parts, stats
            if with_local:
                values = torch.cat((values, local.cpu().double().unsqueeze(0)))
                weights_lse = torch.cat((weights_lse, local_lse.cpu().double().squeeze(-1).unsqueeze(0)))
            valid = weights_lse.isfinite()
            weights = torch.nan_to_num(weights_lse.masked_fill(~valid, -torch.inf).softmax(0), nan=0.0)
            expected = (values.masked_fill(~valid[..., None], 0) * weights[..., None]).sum(0)
            assert_dcp_close(actual, expected)
        graph.reset()
    torch.npu.synchronize()
