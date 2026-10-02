# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.flash_attention_output import flash_attention_gate, flash_attention_output


@pytest.mark.parametrize("tokens,hidden", [(4, 128), (32, 1536), (64, 1536), (17, 129)])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_gate_matches_materialized_sigmoid(tokens, hidden, masked, dtype):
    torch.manual_seed(42)
    projected = torch.randn(tokens, hidden + 16, device="npu", dtype=dtype)[:, :hidden]
    gate = torch.randn(tokens, hidden + 8, device="npu", dtype=dtype)[:, :hidden]
    live = torch.arange(tokens, device="npu") % 2 == 0
    if masked:
        projected[~live] = torch.nan
        gate[~live] = torch.inf
    expected = projected * torch.sigmoid(gate)
    if masked:
        expected[~live] = 0
    actual = flash_attention_gate(projected, gate, live if masked else None)
    torch.testing.assert_close(actual, expected)


@torch.inference_mode()
def test_output_masks_nan_and_graph_padding():
    result = torch.randn(3, 129, device="npu", dtype=torch.bfloat16)
    result[1] = torch.nan
    live = torch.tensor([1, 0, 1], device="npu", dtype=torch.int32)
    output = torch.empty(5, 137, device="npu", dtype=result.dtype)[:, :129]
    expected = torch.zeros_like(output)
    expected[0], expected[2] = result[0], result[2]
    assert flash_attention_output(result, live, output) is output
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("tp", [1, 4, 8, 16])
@pytest.mark.parametrize("tokens,active_tokens", [(1, 1), (8, 8), (32, 20), (128, 111)])
@torch.inference_mode()
def test_kimi_k3_gate_tp_and_token_shapes(tp, tokens, active_tokens):
    # Kimi-K3 has 96 MLA heads with v_head_dim=128; g_proj is column-sharded.
    hidden = 96 // tp * 128
    projected = torch.randn(tokens, hidden, device="npu", dtype=torch.bfloat16)
    projected[active_tokens:] = 0
    gate = torch.randn_like(projected)
    expected = projected * torch.sigmoid(gate)

    assert flash_attention_gate(projected, gate) is projected
    torch.testing.assert_close(projected, expected)
    assert torch.count_nonzero(projected[active_tokens:]) == 0


@torch.inference_mode()
def test_kimi_k3_gate_graph_replay_updates_values():
    projected = torch.empty(32, 1536, device="npu", dtype=torch.bfloat16)
    gate = torch.empty_like(projected)
    projected.fill_(1)
    gate.fill_(0)
    flash_attention_gate(projected, gate)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        flash_attention_gate(projected, gate)

    for value, gate_value in [(1.0, -1.0), (-2.0, 2.0)]:
        projected.fill_(value)
        gate.fill_(gate_value)
        graph.replay()
        expected = torch.full_like(projected, value) * torch.sigmoid(torch.full_like(gate, gate_value))
        torch.testing.assert_close(projected, expected)
