# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType

METHOD_MODULE = "vllm_ascend.quantization.methods.w4a8.w4a8"
DEVICE_MODULE = "vllm_ascend.device.device_op"


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("group_type", [0, 1])
@pytest.mark.parametrize("capacity", [0, 4])
@pytest.mark.parametrize("weight_format", [2, 29, 50])
def test_a3_device_situ_contract(as_list, group_type, capacity, weight_format):
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
        patch(f"{DEVICE_MODULE}.get_current_hardware_profile", return_value=get_hardware_profile(AscendDeviceType.A3)),
        patch(f"{DEVICE_MODULE}.torch_npu.get_npu_format", return_value=weight_format, create=True),
        patch.object(
            torch.ops._C_ascend,
            "grouped_matmul_situ_quant" if weight_format == 2 else "grouped_matmul_situ_quant_weight_nz",
            op,
            create=True,
        ),
    ):
        result = BaseDeviceAdaptor.npu_grouped_matmul_situ_quant(
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
    assert result[0] is expected[0] and result[1] is expected[1] and result[2] is None
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


def test_unsupported_device_rejects_int8_fusion():
    device = AscendDeviceType._310P
    with (
        patch(f"{DEVICE_MODULE}.get_current_hardware_profile", return_value=get_hardware_profile(device)),
        pytest.raises(RuntimeError, match="requires Ascend A2/A3"),
    ):
        BaseDeviceAdaptor.npu_grouped_matmul_situ_quant(
            x=torch.empty((1, 64), dtype=torch.int8),
            weight=None,
            weight_scale=None,
            x_scale=None,
            group_list=None,
            group_list_type=1,
            beta=4.0,
            linear_beta=25.0,
        )


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
@pytest.mark.parametrize("case", ["a3", "a2", "a5", "no_assist", "unaligned"])
def test_w4a8_uses_device_adaptor_or_split_fallback(as_list, group_type, case):
    device = {"a2": AscendDeviceType.A2, "a5": AscendDeviceType.A5}.get(case, AscendDeviceType.A3)
    k, n = (32 if case == "unaligned" else 64), 128
    x, x_scale = torch.empty((4, k), dtype=torch.int8), torch.ones(4)
    weight = torch.empty((2, k, n // 8), dtype=torch.int32)
    scale, assist = torch.empty((2, n), dtype=torch.int64), torch.empty((2, n))
    weights = list(weight.unbind()) if as_list else [weight]
    scales = list(scale.unbind()) if as_list else [scale]
    assists = list(assist.unbind()) if as_list else [assist]
    if case == "no_assist":
        assists = None
    method = AscendW4A8DynamicFusedMoEMethod.__new__(AscendW4A8DynamicFusedMoEMethod)
    method.use_expert_weight_list = as_list
    method.is_per_channel_weight = True
    method._quant_hidden_states = Mock(return_value=(x, x_scale))
    method._get_mlp_weights = Mock(return_value=(weights, scales, weights, scales))
    method._get_mlp_scale_bias = Mock(return_value=(assists, assists))
    groups = torch.tensor([1, 3] if group_type == 1 else [1, 4], dtype=torch.int32)
    inputs = SimpleNamespace(
        hidden_states=x,
        dynamic_scale=x_scale,
        layer=object(),
        group_list=groups,
        group_list_type=group_type,
        activation=MoEActivation.SITU,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    expected = (torch.empty((4, 64), dtype=torch.int8), torch.empty(4))
    with (
        patch(f"{METHOD_MODULE}.get_current_hardware_profile", return_value=get_hardware_profile(device)),
        patch(f"{METHOD_MODULE}.DeviceOperator.npu_grouped_matmul_situ_quant", return_value=(*expected, None)) as fused,
        patch(f"{METHOD_MODULE}.torch_npu.npu_grouped_matmul", return_value=[torch.empty((4, n))]) as gmm,
        patch.object(torch.ops._C_ascend, "dequant_situ_quant", return_value=expected, create=True) as activation,
        patch(f"{METHOD_MODULE}.dispose_tensor") as dispose,
    ):
        result = method.apply_gmm1_act_quant(inputs)
    assert result[0] is expected[0] and result[1] is expected[1]
    dispose.assert_called_once_with(x)
    if case in ("a2", "a3"):
        fused.assert_called_once()
        gmm.assert_not_called()
        activation.assert_not_called()
        kwargs = fused.call_args.kwargs
        assert kwargs["weight"] is (weights if as_list else weights[0])
        assert kwargs["weight_assist_matrix"] is (assists if as_list else assists[0])
        assert kwargs["group_list_type"] == 1
        torch.testing.assert_close(kwargs["group_list"], torch.tensor([1, 3], dtype=torch.int32))
    else:
        fused.assert_not_called()
        gmm.assert_called_once()
        activation.assert_called_once()


@pytest.mark.parametrize(
    "expert_list,lora,weight_format,experts,width,up_width,capacity",
    [
        (False, None, 29, 7, 3584, 6144, 2048),
        (False, None, 29, 7, 3584, 6144, 7168),
        (False, None, 29, 14, 7168, 7168, 128),
        (True, None, 29, 7, 3584, 6144, 16),
        (False, object(), 29, 7, 3584, 6144, 16),
        (False, None, 2, 7, 3584, 6144, 16),
        (False, None, 29, 7, 3584, 6144, 2049),
        (False, None, 29, 7, 3584, 6144, 7169),
        (False, None, 29, 7, 3584, 6144, 32768),
        (False, None, 29, 14, 7168, 7168, 129),
    ],
)
@pytest.mark.parametrize("overlap", [False, True])
def test_fused_mlp_uses_stacked_nz_weights(
    expert_list, lora, weight_format, experts, width, up_width, capacity, overlap
):
    op = Mock()
    x = torch.empty((capacity, width), dtype=torch.int8, device="meta")
    x_scale = torch.ones((capacity, 1))
    up = torch.empty((experts, width, up_width // 8), dtype=torch.int32, device="meta")
    down = torch.empty((experts, up_width // 2, width // 8), dtype=torch.int32, device="meta")
    layer = SimpleNamespace(
        w13_weight=up,
        w2_weight=down,
        w13_scale_bias=torch.ones((experts, up_width)),
        w2_scale_bias=torch.ones((experts, width)),
        w13_weight_scale=torch.ones((experts, up_width), dtype=torch.int64),
        w2_weight_scale=torch.ones((experts, 1, width), dtype=torch.int64),
    )
    inputs = SimpleNamespace(
        dynamic_eplb=False,
        lora_context=lora,
        activation=MoEActivation.SITU,
        group_list_type=0,
        group_list=torch.full((experts,), 16, dtype=torch.int32),
        layer=layer,
        hidden_states=x,
        dynamic_scale=x_scale,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    scheme = AscendW4A8DynamicFusedMoEMethod.__new__(AscendW4A8DynamicFusedMoEMethod)
    scheme.use_expert_weight_list = expert_list
    scheme._quant_hidden_states = Mock(return_value=(x, x_scale))
    with (
        patch(f"{METHOD_MODULE}.get_current_hardware_profile", return_value=get_hardware_profile(AscendDeviceType.A3)),
        patch(
            f"{METHOD_MODULE}.get_ascend_config",
            return_value=SimpleNamespace(multistream_overlap_shared_expert=overlap),
        ),
        patch(f"{METHOD_MODULE}.torch_npu.get_npu_format", return_value=weight_format, create=True),
        patch.object(torch.ops._C_ascend, "grouped_matmul_situ_quant_mlp", op, create=True),
        patch(f"{METHOD_MODULE}.dispose_tensor") as dispose,
    ):
        result = scheme.try_apply_fused_mlp(inputs)
    capacity_limit = 7168 if experts == 7 else 128
    if overlap or expert_list or lora is not None or weight_format != 29 or capacity > capacity_limit:
        assert result is None
        op.assert_not_called()
        dispose.assert_not_called()
    else:
        assert result is op.return_value
        args = op.call_args.args
        assert args[0] is x and args[3] is up and args[6] is down
        assert args[1].shape == (capacity,) and args[2].dtype == torch.int64
        assert args[9:] == (0, 4.0, 25.0)
        dispose.assert_called_once_with(x)
