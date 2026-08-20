from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from tests.ut.quantization.conftest_quantization import create_mock_ascend_config, create_mock_vllm_config
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.quantization.methods.w4a8_mxfp4 import AscendW4A8MXFPDynamicFusedMoEMethod


def _make_scheme(dynamic_eplb=False):
    with (
        patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_ep_group", return_value=Mock()),
        patch(
            "vllm_ascend.quantization.methods.w4a8_mxfp4.get_current_vllm_config",
            return_value=create_mock_vllm_config(),
        ),
        patch(
            "vllm_ascend.quantization.methods.w4a8_mxfp4.get_ascend_config",
            return_value=create_mock_ascend_config(dynamic_eplb=dynamic_eplb),
        ),
    ):
        return AscendW4A8MXFPDynamicFusedMoEMethod()


def _make_layer():
    layer = nn.Module()
    layer.w13_weight = nn.Parameter(torch.randint(0, 255, (2, 128, 64), dtype=torch.uint8), requires_grad=False)
    layer.w2_weight = nn.Parameter(torch.randint(0, 255, (2, 128, 32), dtype=torch.uint8), requires_grad=False)
    layer.w13_weight_scale = nn.Parameter(
        torch.randint(0, 255, (2, 128, 4), dtype=torch.uint8), requires_grad=False
    )
    layer.w2_weight_scale = nn.Parameter(
        torch.randint(0, 255, (2, 128, 2), dtype=torch.uint8), requires_grad=False
    )
    layer.swiglu_limit = 0.0
    return layer


@patch(
    "vllm_ascend.quantization.methods.w4a8_mxfp4.torch_npu.npu_format_cast",
    side_effect=lambda tensor, *_, **__: tensor,
)
def test_process_weights_preserves_reversible_stacked_layout(_mock_format_cast):
    scheme = _make_scheme()
    layer = _make_layer()

    scheme.process_weights_after_loading(layer)

    assert tuple(layer.w13_weight.shape) == (2, 64, 128)
    assert tuple(layer.w2_weight.shape) == (2, 32, 128)
    assert tuple(layer.w13_weight_scale.shape) == (2, 2, 128, 2)
    assert tuple(layer.w2_weight_scale.shape) == (2, 1, 128, 2)


@patch("vllm_ascend.quantization.methods.w4a8_mxfp4.get_forward_context")
@patch("vllm_ascend.quantization.methods.w4a8_mxfp4.select_experts")
@patch(
    "vllm_ascend.quantization.methods.w4a8_mxfp4.torch_npu.npu_format_cast",
    side_effect=lambda tensor, *_, **__: tensor,
)
def test_apply_uses_stacked_checkpoint_layout_for_mega_moe(
    _mock_format_cast,
    mock_select,
    mock_get_forward_context,
):
    scheme = _make_scheme()
    layer = _make_layer()
    scheme.process_weights_after_loading(layer)
    topk_weights = torch.ones(4, 2)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
    mock_select.return_value = (topk_weights, topk_ids)
    moe_comm_method = Mock()
    mock_get_forward_context.return_value = SimpleNamespace(
        moe_comm_type=MoECommType.FUSED_MC2,
        moe_comm_method=moe_comm_method,
    )

    scheme.apply(
        layer,
        torch.randn(4, 128, dtype=torch.bfloat16),
        torch.randn(4, 2),
        top_k=2,
        renormalize=True,
        num_experts=2,
        enable_force_load_balance=False,
    )

    fused_input = moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
    assert tuple(fused_input.weights.w1.shape) == (2, 128, 64)
    assert tuple(fused_input.weights.w2.shape) == (2, 128, 32)
    assert tuple(fused_input.weights.w1_scale.shape) == (2, 128, 2, 2)
    assert tuple(fused_input.weights.w2_scale.shape) == (2, 128, 1, 2)
    assert fused_input.weights.w1.is_contiguous()
    assert fused_input.weights.w2.is_contiguous()
    assert fused_input.weights.w1_scale.is_contiguous()
    assert fused_input.weights.w2_scale.is_contiguous()
