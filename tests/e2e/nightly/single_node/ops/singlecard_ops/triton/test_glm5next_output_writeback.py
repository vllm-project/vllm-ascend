# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.kda.output_writeback import write_recurrent_output


@pytest.mark.parametrize(
    "source_tokens,output_tokens,valid_tokens,heads,dtype",
    [
        pytest.param(7, 13, 5, 16, torch.float16, id="ragged-fp16"),
        pytest.param(7, 13, 5, 32, torch.bfloat16, id="ragged-bf16"),
        pytest.param(7, 13, 5, 64, torch.float32, id="ragged-fp32"),
        pytest.param(0, 13, 0, 16, torch.bfloat16, id="empty-source"),
        pytest.param(0, 0, 0, 16, torch.bfloat16, id="empty-output"),
        pytest.param(1, 1, 1, 16, torch.bfloat16, id="single-token"),
    ],
)
@torch.inference_mode()
def test_recurrent_writeback_matches_reference(source_tokens, output_tokens, valid_tokens, heads, dtype):
    torch.manual_seed(1024)
    source = torch.randn((1, source_tokens, heads, 128), device="npu", dtype=dtype)
    source[:, valid_tokens:] = float("nan")
    # A non-aligned view and guards also exercise the final partial store.
    storage = torch.full((output_tokens * heads * 128 + 34,), -7, device="npu", dtype=dtype)
    output = storage[17:-17].view(1, output_tokens, heads, 128)
    output.fill_(float("nan"))
    ends = torch.tensor([0, 0, valid_tokens], device="npu", dtype=torch.int32)
    expected = torch.zeros(output.shape, dtype=dtype)
    expected[:, :valid_tokens].copy_(source[:, :valid_tokens].cpu())

    write_recurrent_output(source, output, ends)

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(storage[:17].cpu(), torch.full((17,), -7, dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(storage[-17:].cpu(), torch.full((17,), -7, dtype=dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_recurrent_writeback_dp_padding_graph_replay(dtype):
    torch.manual_seed(1024)
    source_tokens, output_tokens, heads = 63, 1771, 16
    # Reproduce the DP fault: put the source at the allocation end, so an
    # invalid fully masked load cannot hide inside its own mapped storage.
    storage = torch.full((1024 * 1024,), -9, device="npu", dtype=dtype)
    source = storage[-source_tokens * heads * 128 :].view(1, source_tokens, heads, 128)
    output = torch.empty((1, output_tokens, heads, 128), device="npu", dtype=dtype)
    ends = torch.tensor([0, source_tokens], device="npu", dtype=torch.int32)
    source.normal_()
    write_recurrent_output(source, output, ends)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        write_recurrent_output(source, output, ends)

    for valid_tokens in [source_tokens, source_tokens // 2, 0]:
        source.normal_()
        source[:, valid_tokens:] = float("nan")
        ends[-1:].fill_(valid_tokens)
        output.fill_(float("nan"))
        expected = torch.zeros(output.shape, dtype=dtype)
        expected[:, :valid_tokens].copy_(source[:, :valid_tokens].cpu())

        graph.replay()

        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
