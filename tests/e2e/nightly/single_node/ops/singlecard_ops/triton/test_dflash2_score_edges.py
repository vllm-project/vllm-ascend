# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch

from vllm_ascend.models.qwen3_dflash2 import _score_edges, _score_edges_reference
from vllm_ascend.ops.triton.spec_decode.dflash2_score_edges import dflash2_score_edges_fake


def _inputs(batch, steps, top_k, rank, dtype, strided=False):
    torch.manual_seed(42)
    vocab = 37
    predecessor = torch.randn(vocab, rank, dtype=dtype, device="npu") * 0.1
    successor = torch.randn_like(predecessor) * 0.1
    candidates = torch.randint(vocab, (batch, steps, top_k), device="npu")
    unary = torch.randn(batch, steps, top_k, dtype=torch.float32, device="npu") * 0.1
    hidden = torch.randn(batch, steps, rank, dtype=dtype, device="npu") * 0.1
    # Both out-of-vocabulary prompt tokens and negative dummy anchors are safe.
    anchors = torch.arange(batch, device="npu", dtype=torch.int64) * vocab - 1
    if strided:
        candidates = candidates.transpose(1, 2).contiguous().transpose(1, 2)
        unary = unary.transpose(1, 2).contiguous().transpose(1, 2)
        hidden = hidden.transpose(1, 2).contiguous().transpose(1, 2)
        anchors = torch.stack((anchors, anchors), dim=1)[:, 0]
    return predecessor, successor, candidates, unary, hidden, anchors


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch,steps,top_k,rank", [(1, 1, 1, 16), (3, 4, 3, 19), (8, 8, 8, 64), (2, 3, 32, 256)])
@pytest.mark.parametrize("strided", [False, True])
def test_fused_edges_match_reference(dtype, batch, steps, top_k, rank, strided):
    inputs = _inputs(batch, steps, top_k, rank, dtype, strided)
    # Direct custom-op invocation ensures the kernel is exercised, not a fallback.
    actual = torch.ops.vllm.ascend_dflash2_score_edges(*inputs)
    expected = _score_edges_reference(*inputs, top_k)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(_score_edges(*inputs, top_k), actual, rtol=0, atol=0)


@pytest.mark.parametrize("batch,steps,top_k", [(0, 4, 3), (2, 0, 3), (2, 4, 0)])
def test_fused_edges_empty_and_fake_contract(batch, steps, top_k):
    inputs = _inputs(batch, steps, top_k, 16, torch.bfloat16)
    actual = torch.ops.vllm.ascend_dflash2_score_edges(*inputs)
    fake = dflash2_score_edges_fake(*inputs)
    assert actual.shape == fake.shape == (batch, steps, top_k, top_k)
    assert actual.dtype == fake.dtype == torch.float32


@pytest.mark.parametrize(
    "dtype,top_k,rank", [(torch.float32, 3, 19), (torch.bfloat16, 33, 16), (torch.bfloat16, 3, 257)]
)
def test_unsupported_tiles_use_reference(dtype, top_k, rank):
    inputs = _inputs(2, 3, top_k, rank, dtype)
    torch.testing.assert_close(_score_edges(*inputs, top_k), _score_edges_reference(*inputs, top_k), rtol=0, atol=0)


def test_strided_codebooks_use_reference():
    inputs = list(_inputs(2, 3, 3, 19, torch.bfloat16))
    for index in (0, 1):
        inputs[index] = inputs[index].t().contiguous().t()
        assert not inputs[index].is_contiguous()
    torch.testing.assert_close(_score_edges(*inputs, 3), _score_edges_reference(*inputs, 3), rtol=0, atol=0)


def test_edges_aclgraph_replay_reads_updated_inputs():
    inputs = _inputs(3, 4, 3, 19, torch.bfloat16, strided=True)
    # Warm up compilation and allocations before capturing on a separate stream.
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            _score_edges(*inputs, 3)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        actual = _score_edges(*inputs, 3)
    for anchor in (0, 36, 99):
        inputs[-1].fill_(anchor)
        inputs[-2].add_(0.01)
        graph.replay()
        torch.testing.assert_close(actual, _score_edges_reference(*inputs, 3), rtol=2e-4, atol=2e-5)


def test_edges_custom_op_in_fullgraph_compile():
    # Dynamo must see the registered fake implementation, not trace the launch.
    compiled = torch.compile(_score_edges, backend="eager", fullgraph=True, dynamic=True)
    for batch in (1, 3):
        inputs = _inputs(batch, 4, 3, 19, torch.bfloat16)
        torch.testing.assert_close(compiled(*inputs, 3), _score_edges_reference(*inputs, 3), rtol=2e-4, atol=2e-5)
