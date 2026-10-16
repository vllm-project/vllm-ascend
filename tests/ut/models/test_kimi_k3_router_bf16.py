# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from vllm_ascend.models import kimi_k3 as k3


def make_gate():
    with (
        patch("vllm_ascend.ops.linear.get_replicated_op", return_value=(None, 0, 1)),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1),
    ):
        gate = k3.KimiK3RouterLinear(8, 16, prefix="model.layers.0.mlp.gate")
        assert type(gate) is k3.KimiK3RouterLinear
        return gate


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_constructor_and_checkpoint_loader(dtype):
    gate = make_gate()
    weight = torch.randn(16, 8, dtype=dtype)
    gate.weight.weight_loader(gate.weight, weight)
    assert gate.weight.dtype == torch.bfloat16
    assert gate.bias is None
    assert not hasattr(gate, "weight_fp32")
    torch.testing.assert_close(gate.weight, weight.to(torch.bfloat16))


@pytest.mark.parametrize("tokens", [0, 1, 4, 32])
def test_actual_moe_forward_routes_through_bf16_gate(tokens):
    gate = make_gate()
    gate.weight.weight_loader(gate.weight, torch.randn(16, 8))
    x = torch.randn(tokens, 8, dtype=torch.bfloat16)
    moe = k3.AscendKimiMoE.__new__(k3.AscendKimiMoE)
    nn.Module.__init__(moe)
    moe.gate = gate
    moe.experts = MagicMock(return_value=x)
    ptr = gate.weight.data_ptr()
    native_mm = torch.mm

    def cpu_mm(a, b, *, out_dtype):
        assert a.data_ptr() == x.data_ptr()
        assert a.dtype == b.dtype == torch.bfloat16
        assert out_dtype == torch.float32
        assert b.data_ptr() == ptr
        return native_mm(a.float(), b.float())

    with patch.object(torch, "mm", side_effect=cpu_mm) as mm:
        result = moe.forward(x)
        assert mm.call_count == 1
    assert gate.weight.data_ptr() == ptr
    args = moe.experts.call_args.kwargs
    assert args["router_logits"].dtype == torch.float32
    torch.testing.assert_close(args["hidden_states"], x)
    torch.testing.assert_close(args["router_logits"], native_mm(x.bfloat16().float(), gate.weight.float().T))
    torch.testing.assert_close(result, x)


def test_repeat_forward_reuses_weight():
    gate = make_gate()
    ptr = gate.weight.data_ptr()
    with patch.object(torch, "mm", return_value=torch.zeros(4, 16)) as mm:
        gate(torch.zeros(4, 8, dtype=torch.bfloat16))
        gate(torch.zeros(4, 8, dtype=torch.bfloat16))
    assert mm.call_count == 2
    assert all(call.args[1].data_ptr() == ptr for call in mm.call_args_list)
