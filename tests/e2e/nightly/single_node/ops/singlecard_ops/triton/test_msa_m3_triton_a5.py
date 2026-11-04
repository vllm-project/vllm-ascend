# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the A5 implementation rather than the portable A2/A3 reference."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.msa import msa_m3_index_topk as ops

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="Requires Ascend A5")

BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 16


def _index_inputs(query_len, *, prefill):
    # Per-rank production geometry: one 128-wide index head, TopK16, TP8.
    seq_lens = torch.tensor([129, 4097] if prefill else [0, 129, 4097], dtype=torch.int32, device="cpu")
    requests = len(seq_lens)
    blocks = (int(seq_lens.max()) + BLOCK_SIZE - 1) // BLOCK_SIZE
    table = torch.randperm(requests * blocks, dtype=torch.int32, device="cpu").reshape(requests, blocks)
    cache = torch.empty((requests * blocks, BLOCK_SIZE, HEAD_DIM), dtype=torch.bfloat16, device="cpu")
    for request in range(requests):
        for block in range(blocks):
            cache[table[request, block]].fill_(block + 1)
    query = torch.ones((requests * query_len, 1, HEAD_DIM), dtype=torch.bfloat16, device="cpu")
    return query, cache, table, seq_lens


def _expected_index_scores(seq_lens, query_len, init_blocks, local_blocks):
    expected = []
    for length in seq_lens.tolist():
        for token in range(query_len):
            position = length - query_len + token
            visible = max(0, (position + BLOCK_SIZE) // BLOCK_SIZE)
            scores = torch.arange(1, visible + 1, dtype=torch.float32, device="cpu") * HEAD_DIM
            scores[:init_blocks] = 1e30
            if local_blocks:
                scores[max(0, visible - local_blocks) :] = 1e29
            expected.append(scores)
    return expected


def _check_index_result(indices, counts, expected):
    indices = indices.cpu()
    assert indices.dtype == torch.int32
    if counts is not None:
        counts = counts.cpu()
        assert counts.dtype == torch.int32
        assert torch.equal(counts, (indices >= 0).sum(-1).to(torch.int32))
    for row, scores in enumerate(expected):
        ids = indices[0, row]
        valid = ids[ids >= 0].long()
        assert len(valid) == min(TOPK, len(scores))
        assert len(valid.unique()) == len(valid)
        assert torch.all(valid < len(scores))
        if counts is not None:
            assert torch.equal(valid, valid.sort().values)
        assert torch.all(ids[ids < 0] == -1)
        torch.testing.assert_close(
            scores[valid].sort().values, scores.topk(len(valid)).values.sort().values, rtol=0, atol=0
        )


@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 1), (1, 2)])
@torch.inference_mode()
def test_prefill_index_topk(init_blocks, local_blocks):
    query_len = 4
    query, cache, table, lengths = _index_inputs(query_len, prefill=True)
    starts = torch.arange(0, query.shape[0] + 1, query_len, dtype=torch.int32, device="cpu")
    prefix = lengths - query_len
    score = ops.minimax_m3_index_score(
        query.npu(),
        cache.npu(),
        table.npu(),
        starts.npu(),
        lengths.npu(),
        prefix.npu(),
        query_len,
        int(lengths.max()),
        1,
        sm_scale=1.0,
    )
    storage = torch.full((1, query.shape[0] + 2, TOPK * 2), -77, dtype=torch.int32, device="npu")
    output = storage[..., ::2]
    indices = ops.minimax_m3_index_topk(
        score,
        starts.npu(),
        prefix.npu(),
        query_len,
        TOPK,
        init_blocks,
        local_blocks,
        out=output,
    )
    assert indices.data_ptr() == output.data_ptr()
    _check_index_result(indices, None, _expected_index_scores(lengths, query_len, init_blocks, local_blocks))
    assert torch.all(storage[..., 1::2] == -77)
    assert torch.all(output[:, query.shape[0] :] == -77)


@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("replay", [False, True])
@torch.inference_mode()
def test_decode_index_counts_and_output_buffer(query_len, replay):
    query, cache, table, lengths = _index_inputs(query_len, prefill=False)
    query, cache, table, device_lengths = query.npu(), cache.npu(), table.npu(), lengths.npu()
    storage = torch.full((1, query.shape[0] + 2, TOPK * 2), -77, dtype=torch.int32, device="npu")
    output = storage[..., ::2]

    def run():
        return ops.minimax_m3_index_decode(
            query,
            cache,
            table,
            device_lengths,
            4097,
            TOPK,
            0,
            1,
            1,
            query_len,
            out=output,
            sort_indices=True,
        )

    indices, counts = run()
    assert indices.data_ptr() == output.data_ptr()
    _check_index_result(indices, counts, _expected_index_scores(lengths, query_len, 0, 1))
    if replay:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            indices, counts = run()
        lengths = torch.tensor([0, 128, 4096], dtype=torch.int32, device="cpu")
        device_lengths.copy_(lengths)
        graph.replay()
        _check_index_result(indices, counts, _expected_index_scores(lengths, query_len, 0, 1))
    assert torch.all(storage[..., 1::2] == -77)
    assert torch.all(output[:, query.shape[0] :] == -77)


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("layout", ["page-first", "kv-first", "split"])
@torch.inference_mode()
def test_sparse_attention_matches_cpu_reference(decode, layout):
    query_len, heads, blocks = 4, 8, 4
    lengths = torch.tensor([129, 385], dtype=torch.int32, device="cpu")
    table = torch.randperm(2 * blocks, dtype=torch.int32, device="cpu").reshape(2, blocks)
    query = torch.randn((2 * query_len, heads, HEAD_DIM), dtype=torch.bfloat16, device="cpu")
    cache = torch.randn((2 * blocks, 2, BLOCK_SIZE, 1, HEAD_DIM), dtype=torch.bfloat16, device="cpu")
    indices = torch.full((1, 2 * query_len, TOPK), -1, dtype=torch.int32, device="cpu")
    indices[..., :2] = torch.tensor([0, 2], dtype=torch.int32, device="cpu")
    expected = torch.empty_like(query)
    for request, length in enumerate(lengths.tolist()):
        for token in range(query_len):
            row = request * query_len + token
            position = length - query_len + token
            positions = [
                p for block in (0, 2) for p in range(block * BLOCK_SIZE, (block + 1) * BLOCK_SIZE) if p <= position
            ]
            logical = torch.tensor(positions, device="cpu")
            page, offset = table[request, logical // BLOCK_SIZE].long(), logical % BLOCK_SIZE
            key, value = cache[page, 0, offset, 0].float(), cache[page, 1, offset, 0].float()
            scores = query[row].float() @ key.T * HEAD_DIM**-0.5
            expected[row] = (scores.softmax(-1) @ value).to(expected.dtype)

    device_cache = cache.npu()
    if layout == "kv-first":
        device_cache = device_cache.transpose(0, 1)
    elif layout == "split":
        device_cache = (device_cache[:, 0], device_cache[:, 1])
    actual = torch.empty_like(query, device="npu")
    if decode:
        ops.minimax_m3_sparse_attn_decode(
            query.npu(),
            device_cache,
            indices.npu(),
            table.npu(),
            lengths.npu(),
            1,
            HEAD_DIM**-0.5,
            actual,
            query_len,
        )
    else:
        starts = torch.arange(0, query.shape[0] + 1, query_len, dtype=torch.int32, device="npu")
        ops.minimax_m3_sparse_attn(
            query.npu(),
            device_cache,
            indices.npu(),
            table.npu(),
            starts,
            lengths.npu(),
            (lengths - query_len).npu(),
            query_len,
            1,
            HEAD_DIM**-0.5,
            actual,
        )
    error = (actual.cpu().float() - expected.float()).abs()
    # Same BF16 tolerances as the existing MiniMax sparse-attention suite.
    assert torch.isfinite(error).all()
    assert error.mean() <= 2.5e-4
    assert error.max() <= 1.7e-2
