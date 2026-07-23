from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.quantization.methods import (
    w4a4_mxfp4,
    w4a8,
    w4a8_mxfp4,
    w4a16,
    w4a16_mxfp4,
    w8a8_dynamic,
    w8a8_mxfp8,
)


ROUTED_COMPUTE_CASES = [
    (w4a4_mxfp4, w4a4_mxfp4.AscendW4A4MXFP4DynamicFusedMoEMethod),
    (w4a8, w4a8.AscendW4A8DynamicFusedMoEMethod),
    (w4a8_mxfp4, w4a8_mxfp4.AscendW4A8MXFPDynamicFusedMoEMethod),
    (w4a16, w4a16.AscendW4A16FusedMoEMethod),
    (w4a16_mxfp4, w4a16_mxfp4.AscendW4A16MXFP4FusedMoEMethod),
    (w8a8_dynamic, w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod),
    (w8a8_mxfp8, w8a8_mxfp8.AscendW8A8MXFP8DynamicFusedMoEMethod),
]


def _build_method(scheme_cls):
    method = scheme_cls.__new__(scheme_cls)
    method.dynamic_eplb = False
    if isinstance(method, w4a8.AscendW4A8DynamicFusedMoEMethod):
        method.is_per_channel_weight = False
    if isinstance(method, w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod):
        method.in_dtype = torch.float32
    return method


def _build_layer():
    return SimpleNamespace(
        w13_weight=torch.empty(4, 8, 16),
        w2_weight=torch.empty(4, 16, 8),
        w13_weight_packed=torch.empty(4, 8, 4, dtype=torch.int32),
        w2_weight_packed=torch.empty(4, 16, 2, dtype=torch.int32),
        w13_weight_scale=torch.empty(4, 8, 16),
        w2_weight_scale=torch.empty(4, 16, 8),
        w13_weight_scale_fp32=torch.empty(4, 8, 16),
        w13_weight_offset=torch.empty(4, 8, 16),
        w2_weight_offset=torch.empty(4, 16, 8),
        swiglu_limit=1000000,
    )


@pytest.mark.parametrize(("scheme_module", "scheme_cls"), ROUTED_COMPUTE_CASES)
def test_apply_routed_never_reselects_experts(monkeypatch, scheme_module, scheme_cls):
    select_experts = MagicMock(side_effect=AssertionError("v2 routed compute must not select experts"))
    monkeypatch.setattr(scheme_module, "select_experts", select_experts)
    if hasattr(scheme_module, "torch_npu"):
        monkeypatch.setattr(
            scheme_module.torch_npu,
            "float4_e2m1fn_x2",
            torch.float8_e4m3fn,
            raising=False,
        )
    routed_out = torch.randn(3, 8)
    comm_method = MagicMock()
    comm_method.fused_experts.return_value = routed_out
    if hasattr(scheme_module, "_EXTRA_CTX"):
        monkeypatch.setattr(
            scheme_module,
            "_EXTRA_CTX",
            SimpleNamespace(moe_comm_type=MoECommType.ALLGATHER, moe_comm_method=comm_method),
        )
    if scheme_module is w4a8_mxfp4:
        monkeypatch.setattr(
            scheme_module,
            "get_forward_context",
            lambda: SimpleNamespace(moe_comm_method=comm_method),
        )

    method = _build_method(scheme_cls)
    x = torch.randn(3, 8)
    topk_weights = torch.randn(3, 2)
    topk_ids = torch.randint(0, 4, (3, 2), dtype=torch.int32)

    result = method.apply_routed(_build_layer(), x, topk_weights, topk_ids)

    assert result is routed_out
    select_experts.assert_not_called()
    fused_input = comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
    assert fused_input.topk_ids is topk_ids
