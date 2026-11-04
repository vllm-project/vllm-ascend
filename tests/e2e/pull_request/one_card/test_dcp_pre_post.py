# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 DCP transport/metadata against the stable PyTorch comparator."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.context_parallel.sharded_indexer import (
    local_to_global,
    local_visible_lengths,
    merge_topk_candidates,
    query_visible_lengths,
)
from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.triton.dcp.dcp_a2a import fused_dcp_lse_combine, pack_dcp_output_lse
from vllm_ascend.ops.triton.query_gather_prep import prep_query_head_major
from vllm_ascend.ops.triton.sharded_indexer import (
    build_local_lengths,
    merge_packed_candidates,
    pack_local_candidates,
    select_local_candidates,
)
from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton


@pytest.fixture(autouse=True)
def a5_device():
    torch.npu.set_device(0)
    if not is_950():
        pytest.skip("A5 specialized DCP kernels")


@pytest.mark.parametrize(
    "rows,world,interleave,topk", [(1, 2, 1, 31), (7, 8, 128, 2048), (16, 2, 768, 2048), (64, 8, 1, 2048)]
)
@pytest.mark.parametrize("equal_scores", [False, True])
def test_candidate_transport(rows, world, interleave, topk, equal_scores):
    torch.manual_seed(17)
    packed, ids, values = [], [], []
    for rank in range(world):
        local = torch.randint(0, 131072, (rows, 1, topk), dtype=torch.int32)
        scores = torch.randn(rows, 1, topk).bfloat16().float()
        if equal_scores:
            scores.fill_(1)
        local[..., -3:] = -1
        scores[..., 0] = float("nan")
        scores[..., 1] = float("inf")
        scores[..., 2] = -float("inf")
        ids.append(local_to_global(local, world, rank, interleave))
        values.append(scores)
        # Non-contiguous views exercise the transport's independent strides.
        local_device = torch.stack((local, local), -1).npu()[..., 0]
        score_device = torch.stack((scores, scores), -1).npu()[..., 1]
        packed.append(pack_local_candidates(local_device, score_device, world, rank, interleave))
    expected = merge_topk_candidates(torch.cat(ids, -1), torch.cat(values, -1), topk)
    actual = merge_packed_candidates(torch.cat(packed, 0), rows, world, topk)
    assert torch.equal(actual.cpu(), expected)
    native_ids, native_scores = torch.cat(ids, -1).npu(), torch.cat(values, -1).npu()
    for rank in range(world):
        local = merge_packed_candidates(torch.cat(packed, 0), rows, world, topk, local_rank=rank, interleave=interleave)
        reference = remap_sparse_indices_triton(actual, world, rank, interleave)
        assert torch.equal(local.cpu(), reference.cpu())
        native_local = merge_topk_candidates(
            native_ids, native_scores, topk, local_rank=rank, world_size=world, interleave=interleave
        )
        assert torch.equal(native_local.cpu(), reference.cpu())


def test_local_candidate_selection_graph_replay():
    ids = torch.arange(2048, dtype=torch.int32, device="npu").reshape(1, 1, -1)
    scores = torch.ones_like(ids, dtype=torch.float32)
    for _ in range(3):
        select_local_candidates(ids, scores, 8, 7, 128)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = select_local_candidates(ids, scores, 8, 7, 128)
    for value in (-1, 7 * 128, 0):
        ids.fill_(value)
        graph.replay()
        expected = remap_sparse_indices_triton(ids, 8, 7, 128)
        assert torch.equal(actual.cpu(), expected.cpu())
    ids.fill_(7 * 128)
    scores.fill_(torch.nan)
    graph.replay()
    assert (actual.cpu() == -1).all()
    graph.reset()


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


@pytest.mark.parametrize("local_rank", [None, 7])
def test_candidate_merge_writes_supplied_output_slice(local_rank):
    rows, world, topk, interleave = 17, 8, 2048, 768
    ids = torch.randint(0, 131072, (rows, 1, topk), dtype=torch.int32, device="npu")
    scores = torch.randn_like(ids, dtype=torch.float32)
    packed = torch.cat([pack_local_candidates(ids, scores, world, rank, interleave) for rank in range(world)], 0)
    global_ids = merge_packed_candidates(packed, rows, world, topk)
    expected = (
        global_ids if local_rank is None else remap_sparse_indices_triton(global_ids, world, local_rank, interleave)
    )
    storage = torch.full((rows + 2, 1, topk), -123, dtype=torch.int32, device="npu")
    out = storage[1:-1]
    actual = merge_packed_candidates(packed, rows, world, topk, local_rank=local_rank, interleave=interleave, out=out)
    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (storage[[0, -1]].cpu() == -123).all()


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
        assert torch.equal(keys.cpu(), local_visible_lengths(seq.cpu(), world, rank, interleave))
        assert torch.equal(visible.cpu().flatten(), local_visible_lengths(global_visible, world, rank, interleave))
    graph.reset()


@pytest.mark.parametrize(
    "rows,topk,interleave", [(1, 1, 1), (7, 31, 128), (16, 2048, 768), (256, 2048, 1), (17, 4096, 128)]
)
def test_remap_repeated_and_empty(rows, topk, interleave):
    torch.manual_seed(23)
    source = torch.randint(-1, 1048576, (rows, 1, topk), dtype=torch.int32)
    source[0].fill_(-1)
    if rows > 1:
        source[1].fill_(7 * interleave)
    expected = torch.full_like(source, -1)
    for row in range(rows):
        idx = source[row, 0]
        valid = (idx >= 0) & ((idx // interleave) % 8 == 7)
        local = idx[valid] // (8 * interleave) * interleave + idx[valid] % interleave
        expected[row, 0, : local.numel()] = local
    data = source.npu()
    for _ in range(30):
        assert torch.equal(remap_sparse_indices_triton(data, 8, 7, interleave).cpu(), expected)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = remap_sparse_indices_triton(data, 8, 7, interleave)
    for _ in range(30):
        graph.replay()
    assert torch.equal(actual.cpu(), expected)
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
    torch.testing.assert_close(
        actual_out,
        expected_out,
        atol=2e-5 if dtype == torch.float32 else 0.008,
        rtol=2e-5 if dtype == torch.float32 else 0.008,
    )


@pytest.mark.parametrize("rows", [0, 17])
def test_local_lengths_empty_requests(rows):
    empty = torch.empty(0, dtype=torch.int32, device="npu")
    visible = torch.full((rows, 1), 123, dtype=torch.int32, device="npu")
    build_local_lengths(empty, empty, empty, visible, 8, 7, 768)
    assert (visible.cpu() == 0).all()


@pytest.mark.parametrize("topk", [1, 31, 128, 257, 2048, 4096])
def test_remap_preserves_large_integer_payloads(topk):
    torch.manual_seed(41)
    # Local indices exceed the exact FP32 integer range. Sort position keys,
    # but preserve payload bits through the UB gather.
    source = torch.randint(1 << 29, (1 << 30) + 10000, (9, 1, topk), dtype=torch.int64)
    source[0] = -1
    expected = torch.full_like(source, -1)
    for row in range(9):
        ids = source[row, 0]
        own = (ids >= 0) & (ids % 8 == 7)
        local = ids[own] // 8
        expected[row, 0, : local.numel()] = local
    data = torch.stack((source, source), -1).npu()[..., 1]
    actual = remap_sparse_indices_triton(data, 8, 7, 1)
    assert torch.equal(actual.cpu(), expected)


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
        torch.testing.assert_close(merged.cpu().float(), expected, atol=0.008, rtol=0.008)
    graph.reset()
