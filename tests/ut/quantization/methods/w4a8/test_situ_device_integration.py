# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.device.device_op import A5DeviceAdaptor
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp
from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType

METHOD_MODULE = "vllm_ascend.quantization.methods.w4a8.w4a8"


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("capacity", [0, 4])
@pytest.mark.parametrize("weight_format", [2, 29, 50])
def test_w4a8_situ_native_op_selection(as_list, group_type, capacity, weight_format):
    x = torch.empty((capacity, 64), dtype=torch.int8)
    weight = torch.empty((2, 64, 16), dtype=torch.int32)
    scale = torch.empty((2, 128), dtype=torch.int64)
    assist = torch.empty((2, 128))
    if as_list:
        weight, scale, assist = list(weight.unbind()), list(scale.unbind()), list(assist.unbind())
    groups = torch.tensor([0, capacity], dtype=torch.int32)
    expected = (torch.empty((capacity, 64), dtype=torch.int8), torch.empty(capacity))
    op = Mock(return_value=expected)
    op.list.return_value = expected
    with (
        patch(f"{METHOD_MODULE}.torch_npu.get_npu_format", return_value=weight_format, create=True),
        patch.object(
            torch.ops._C_ascend,
            "gmm1_situ_quant" if weight_format == 2 else "gmm1_situ_quant_weight_nz",
            op,
            create=True,
        ),
    ):
        result = AscendW4A8DynamicFusedMoEMethod._apply_gmm1_situ_quant(
            x=x,
            weight=weight,
            weight_scale=scale,
            weight_assist_matrix=assist,
            x_scale=torch.ones(capacity),
            group_list=groups,
            group_list_type=group_type,
            beta=4.0,
            linear_beta=25.0,
        )
    assert result[0] is expected[0] and result[1] is expected[1]
    selected = op.list if as_list else op
    selected.assert_called_once()
    kwargs = selected.call_args.kwargs
    assert kwargs["weight"] is weight and kwargs["weight_scale"] is scale
    assert kwargs["weight_assist_matrix"] is assist
    assert kwargs["group_list"].dtype == torch.int64
    torch.testing.assert_close(kwargs["group_list"], groups.to(torch.int64))
    assert kwargs["group_list_type"] == group_type
    assert kwargs["dequant_mode"] == kwargs["quant_mode"] == 0
    assert (kwargs["beta"], kwargs["linear_beta"]) == (4.0, 25.0)


@pytest.mark.parametrize("as_list", [False, True])
def test_a5_device_preserves_mxfp_contract(as_list):
    weight = torch.empty((2, 128, 32), dtype=torch.uint8)
    scale = torch.empty((2, 128, 1, 2), dtype=torch.float8_e8m0fnu)
    if as_list:
        weight, scale = list(weight.unbind()), list(scale.unbind())
    x_scale = torch.empty((4, 2), dtype=torch.float8_e8m0fnu)
    groups = torch.tensor([2, 2], dtype=torch.int64)
    expected = (torch.empty((4, 64), dtype=torch.float8_e4m3fn), torch.empty((4, 1, 2)))
    op = Mock(return_value=expected)
    op.list.return_value = expected
    with patch.object(torch.ops._C_ascend, "grouped_matmul_situ_quant_weight_nz", op, create=True):
        result = A5DeviceAdaptor.npu_grouped_matmul_situ_quant(
            x=torch.empty((4, 64), dtype=torch.float8_e4m3fn),
            weight=weight,
            weight_scale=scale,
            x_scale=x_scale,
            group_list=groups,
            group_list_type=1,
            beta=4.0,
            linear_beta=25.0,
            mxfp_quant_dtype=QuantType.W4A8MXFP,
        )
    kwargs = (op.list if as_list else op).call_args.kwargs
    assert result[0] is expected[0] and result[1] is expected[1]
    assert kwargs["dequant_mode"] == kwargs["quant_mode"] == 1
    assert kwargs["weight_assist_matrix"] is None
    assert kwargs["group_list"] is groups
    assert kwargs["x_scale"] is x_scale


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize(
    "case,k,n",
    [
        ("a3", 64, 128),
        pytest.param("a3", 65, 128, id="a3-k-tail"),
        pytest.param("a3", 64, 130, id="a3-n-tail"),
        pytest.param("a3", 72, 144, id="a3-kn-tails"),
        ("a2", 64, 128),
        ("a5", 64, 128),
        ("no_assist", 64, 128),
        ("silu", 64, 128),
    ],
)
def test_w4a8_gmm1_act_quant_dispatch(as_list, group_type, case, k, n):
    device = {"a2": AscendDeviceType.A2, "a5": AscendDeviceType.A5}.get(case, AscendDeviceType.A3)
    x, x_scale = torch.empty((4, k), dtype=torch.int8), torch.ones(4)
    weight = torch.empty((2, k, (n + 7) // 8), dtype=torch.int32)
    scale, assist = torch.empty((2, n), dtype=torch.int64), torch.empty((2, n))
    weights = list(weight.unbind()) if as_list else [weight]
    scales = list(scale.unbind()) if as_list else [scale]
    assists = list(assist.unbind()) if as_list else [assist]
    method = AscendW4A8DynamicFusedMoEMethod.__new__(AscendW4A8DynamicFusedMoEMethod)
    method.use_expert_weight_list = as_list
    method._quant_hidden_states = Mock(return_value=(x, x_scale))
    method._get_mlp_weights = Mock(return_value=(weights, scales, weights, scales))
    method._get_mlp_scale_bias = Mock(return_value=(None, None) if case == "no_assist" else (assists, assists))
    groups = torch.tensor([1, 2] if group_type == 1 else [1, 3], dtype=torch.int32)
    inputs = SimpleNamespace(
        hidden_states=x,
        dynamic_scale=x_scale,
        lora_context=None,
        quant=MoEQuantParams(quant_type=QuantType.W4A8),
        layer=object(),
        group_list=groups,
        group_list_type=group_type,
        activation=MoEActivation.SILU if case == "silu" else MoEActivation.SITU,
        swiglu_limit=0.0,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    expected = (torch.empty((4, n // 2), dtype=torch.int8), torch.empty(4))
    # Three routed rows in a four-row buffer. Padding must not affect the SiTU values sent to quantization.
    projected = torch.linspace(-5, 5, 4 * n, dtype=torch.bfloat16).reshape(4, n)
    projected[-1].fill_(torch.nan)
    with (
        patch("vllm_ascend.ops.fused_moe.moe_mlp.maybe_record_event") as event,
        patch.object(method, "apply_gmm2") as gmm2,
        patch(f"{METHOD_MODULE}.get_current_hardware_profile", return_value=get_hardware_profile(device)),
        patch.object(method, "_apply_gmm1_situ_quant", return_value=expected) as fused,
        patch(f"{METHOD_MODULE}.torch_npu.npu_grouped_matmul", return_value=[projected]) as gmm,
        patch(f"{METHOD_MODULE}.torch_npu.npu_dynamic_quant", return_value=expected, create=True) as act_quant,
        patch.object(
            torch.ops._C_ascend, "grouped_matmul_swiglu_quant_v2", return_value=expected, create=True
        ) as swiglu,
        patch(f"{METHOD_MODULE}.dispose_tensor") as dispose,
    ):
        out, before_gmm2 = apply_moe_mlp(inputs, method)
    assert out is gmm2.return_value and before_gmm2 is event.return_value
    gmm2.assert_called_once_with(inputs, *expected)
    dispose.assert_called_once_with(x)
    if case == "silu":
        swiglu.assert_called_once()
        fused.assert_not_called()
        gmm.assert_not_called()
        act_quant.assert_not_called()
        return
    swiglu.assert_not_called()
    if case in ("a3", "no_assist"):
        fused.assert_called_once()
        gmm.assert_not_called()
        act_quant.assert_not_called()
        kwargs = fused.call_args.kwargs
        assert kwargs["x"] is x
        assert kwargs["weight"] is (weights if as_list else weights[0])
        assert kwargs["weight_scale"] is (scales if as_list else scales[0])
        if case == "no_assist":
            assert kwargs["weight_assist_matrix"] is None
        else:
            assert kwargs["weight_assist_matrix"] is (assists if as_list else assists[0])
        assert kwargs["group_list_type"] == 1
        torch.testing.assert_close(kwargs["group_list"], torch.tensor([1, 2], dtype=torch.int32))
    else:
        fused.assert_not_called()
        gmm.assert_called_once()
        act_quant.assert_called_once()
        gate, up = projected[:3].float().chunk(2, dim=-1)
        ref = 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * (25 * torch.tanh(up / 25))
        torch.testing.assert_close(act_quant.call_args.args[0][:3], ref.bfloat16())
        assert act_quant.call_args.kwargs["dst_type"] == torch.int8
        torch.testing.assert_close(gmm.call_args.kwargs["group_list"], torch.tensor([1, 2], dtype=torch.int32))
