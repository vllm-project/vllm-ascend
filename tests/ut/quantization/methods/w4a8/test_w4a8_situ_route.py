# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the production method's dispatch and fallback without NPU math."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.quantization.methods.w4a8 import w4a8


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("experts", [1, 3])
@pytest.mark.parametrize("expert_list", [False, True])
@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize("enabled", [False, True])
def test_situ_route(device, experts, expert_list, group_list_type, enabled):
    method = object.__new__(w4a8.AscendW4A8DynamicFusedMoEMethod)
    method.use_expert_weight_list = expert_list
    method.is_per_channel_weight = True
    method.fuse_situ_msd = enabled
    x, scale = torch.zeros(4, 64, dtype=torch.int8), torch.ones(4)
    weight = torch.zeros(experts, 64, 16, dtype=torch.int32)
    weight_scale = torch.zeros(experts, 128, dtype=torch.int64)
    assist = torch.zeros(experts, 128)
    weights = [w.clone() for w in weight.unbind()] if expert_list else [weight]
    scales = list(weight_scale.unbind()) if expert_list else [weight_scale]
    assists = list(assist.unbind()) if expert_list else [assist]
    method._quant_hidden_states = Mock(return_value=(x, scale))
    method._get_mlp_weights = Mock(return_value=(weights, scales, weights, scales))
    method._get_mlp_scale_bias = Mock(return_value=(assists, assists))
    counts = torch.tensor([4] + [0] * (experts - 1), dtype=torch.int32)
    inp = SimpleNamespace(
        hidden_states=x,
        dynamic_scale=scale,
        layer=object(),
        activation=MoEActivation.SITU,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        group_list=counts.cumsum(0).to(torch.int32) if group_list_type == 0 else counts,
        group_list_type=group_list_type,
    )
    expected = (torch.empty(4, 64, dtype=torch.int8), torch.ones(4))
    fused = Mock(return_value=expected)
    fused.list = Mock(return_value=expected)
    split_quant = Mock(return_value=expected)
    split_gmm = Mock(return_value=[torch.empty(4, 128, dtype=torch.bfloat16)])
    with (
        patch.object(w4a8, "get_current_hardware_profile", return_value=get_hardware_profile(device)),
        patch.object(w4a8, "dispose_tensor") as dispose,
        patch.object(w4a8.torch_npu, "npu_grouped_matmul", split_gmm, create=True),
        patch.object(torch.ops._C_ascend, "grouped_matmul_situ_quant_weight_nz", fused, create=True),
        patch.object(torch.ops._C_ascend, "dequant_situ_quant", split_quant, create=True),
    ):
        actual = method.apply_gmm1_act_quant(inp)
    assert actual is expected or all(a is b for a, b in zip(actual, expected))
    dispose.assert_called_once_with(x)
    if device == AscendDeviceType.A3 and enabled:
        selected, unused = (fused.list, fused) if expert_list else (fused, fused.list)
        selected.assert_called_once()
        unused.assert_not_called()
        split_gmm.assert_not_called()
        split_quant.assert_not_called()
        args = selected.call_args.kwargs
        assert args["weight"] is (weights if expert_list else weight)
        assert args["weight_assist_matrix"] is (assists if expert_list else assist)
        assert args["group_list"].dtype == torch.int64
        assert args["group_list_type"] == 1
        torch.testing.assert_close(args["group_list"], counts.to(torch.int64))
        assert args["beta"] == 4.0 and args["linear_beta"] == 25.0
        assert (args["dequant_mode"], args["dequant_dtype"], args["quant_mode"]) == (0, 0, 0)
    else:
        fused.assert_not_called()
        fused.list.assert_not_called()
        split_gmm.assert_called_once()
        split_quant.assert_called_once()


@pytest.mark.parametrize(
    "unsupported",
    ["no_assist", "k_alignment", "n_alignment", "float_scale", "group_type", "weight_offset", "weight_stride"],
)
def test_unsupported_msd_input_uses_existing_path(unsupported):
    method = object.__new__(w4a8.AscendW4A8DynamicFusedMoEMethod)
    method.is_per_channel_weight = True
    method.fuse_situ_msd = True
    k, n = (96 if unsupported == "k_alignment" else 64), (192 if unsupported == "n_alignment" else 128)
    args = dict(
        hidden_states=torch.zeros(4, k, dtype=torch.int8),
        pertoken_scale=torch.ones(4),
        weights=[torch.zeros(2, k, n // 8, dtype=torch.int32)],
        scales=[torch.zeros(2, n, dtype=torch.float32 if unsupported == "float_scale" else torch.int64)],
        assist=None if unsupported == "no_assist" else [torch.zeros(2, n)],
        group_list=torch.tensor([4, 0]),
        group_list_type=2 if unsupported == "group_type" else 1,
    )
    if unsupported == "weight_offset":
        args["weights"] = [torch.zeros(3, k, n // 8, dtype=torch.int32)[1:]]
    elif unsupported == "weight_stride":
        args["weights"] = [torch.zeros(2, k, n // 8, 2, dtype=torch.int32)[..., 0]]
    with patch.object(w4a8, "get_current_hardware_profile", return_value=get_hardware_profile(AscendDeviceType.A3)):
        assert not method._can_fuse_situ_msd(**args)
