# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.flash_attention_output import flash_attention_gate, flash_attention_output


@pytest.mark.parametrize("num_tokens", [4, 8, 17])
@pytest.mark.parametrize("rank", range(8))
@torch.inference_mode()
def test_flash_attention_output_tp_shard_and_empty_mask(num_tokens, rank):
    shard_rows = (num_tokens + 7) // 8
    live = torch.ones(num_tokens, dtype=torch.bool, device="npu")
    live[1::2] = False
    shard_live = live[rank * shard_rows : (rank + 1) * shard_rows]
    result = torch.randn(shard_rows, 7168, dtype=torch.bfloat16, device="npu")
    expected = torch.zeros_like(result)
    valid_rows = shard_live.shape[0]
    expected[:valid_rows] = result[:valid_rows].masked_fill(~shard_live[:, None], 0)
    result[valid_rows:] = float("nan")
    result[:valid_rows].masked_fill_(~shard_live[:, None], float("inf"))
    output = torch.empty_like(result)
    flash_attention_output(result, shard_live, output)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        flash_attention_output(result, shard_live, output)
    output.fill_(float("nan"))
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_flash_attention_gate_bf16_rounding():
    # Cover every finite BF16 gate, including saturation and sigmoid rounding.
    gates = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    gates = gates[torch.isfinite(gates)].reshape(-1, 256).to("npu")
    projected = torch.linspace(0.1, 3.0, gates.numel(), dtype=torch.float32).reshape_as(gates).to(gates)
    live = torch.ones(gates.shape[0], dtype=torch.bool, device="npu")
    expected = projected * torch.sigmoid(gates)
    actual = flash_attention_gate(projected, gates, live)

    assert actual.data_ptr() == projected.data_ptr()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [8, 16, 32, 64])
@torch.inference_mode()
def test_flash_attention_gate_graph_output_buffer(num_tokens):
    torch.manual_seed(123)
    # K3 TP8 O projection: 12 local heads, V128, hidden size 7168.
    projected = torch.randn(num_tokens, 1536, dtype=torch.bfloat16, device="npu")
    gates = torch.randn_like(projected)
    live = torch.ones(num_tokens, dtype=torch.bool, device="npu")
    weight = torch.randn(7168, 1536, dtype=torch.bfloat16, device="npu") * 0.01
    output = torch.empty(num_tokens, 7168, dtype=torch.bfloat16, device="npu")

    def forward():
        flash_attention_gate(projected, gates, live)
        return torch.mm(projected, weight.t(), out=output)

    assert forward().data_ptr() == output.data_ptr()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        forward()

    # Reuse the captured graph as asynchronous scheduling changes visibility.
    for mode in ("all_live", "alternating", "all_inactive"):
        live.fill_(mode != "all_inactive")
        if mode == "alternating":
            live[1::2] = False
        projected.normal_()
        gates.normal_(std=20)
        projected.masked_fill_(~live[:, None], float("nan"))
        gates.masked_fill_(~live[:, None], float("inf"))
        expected = F.linear(projected * torch.sigmoid(gates), weight)
        expected.masked_fill_(~live[:, None], 0)
        output.fill_(float("nan"))

        graph.replay()
        torch.npu.synchronize()

        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output[~live]) == 0
