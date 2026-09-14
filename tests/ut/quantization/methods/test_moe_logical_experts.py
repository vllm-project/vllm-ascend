from types import SimpleNamespace

import pytest

from vllm_ascend.ops.fused_moe.moe_utils import get_moe_num_logical_experts
from vllm_ascend.quantization.methods.base import AscendLinearScheme, AscendMoEScheme
from vllm_ascend.quantization.methods.w4a4.w4a4_mxfp4 import AscendW4A4MXFP4DynamicFusedMoEMethod
from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import AscendW4A8MXFPDynamicFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicFusedMoEMethod
from vllm_ascend.quantization.methods.wna16.w4a16 import AscendW4A16FusedMoEMethod
from vllm_ascend.quantization.methods.wna16.w4a16_mxfp4 import AscendW4A16MXFP4FusedMoEMethod


def test_get_moe_num_logical_experts_uses_vllm_config_field():
    layer = SimpleNamespace(moe_config=SimpleNamespace(num_logical_experts=128))

    assert get_moe_num_logical_experts(layer, num_experts=130, global_redundant_expert_num=2) == 128


def test_get_moe_num_logical_experts_falls_back_for_older_configs():
    layer = SimpleNamespace(moe_config=SimpleNamespace())

    assert (
        get_moe_num_logical_experts(
            layer,
            num_experts=133,
            global_redundant_expert_num=2,
            num_shared_experts=3,
        )
        == 128
    )


def test_routed_moe_interface_is_not_exposed_on_linear_schemes():
    assert "get_eplb_weight_views" not in AscendLinearScheme.__dict__
    assert "get_eplb_weight_views" in AscendMoEScheme.__dict__


@pytest.mark.parametrize(
    "scheme_cls",
    [
        AscendW8A8DynamicFusedMoEMethod,
        AscendW4A8DynamicFusedMoEMethod,
        AscendW4A4MXFP4DynamicFusedMoEMethod,
        AscendW8A8MXFP8DynamicFusedMoEMethod,
    ],
)
def test_v2_eplb_supported_quantization(scheme_cls):
    assert scheme_cls.supports_eplb is True


@pytest.mark.parametrize(
    "scheme_cls",
    [
        AscendW4A16FusedMoEMethod,
        AscendW4A16MXFP4FusedMoEMethod,
        AscendW4A8MXFPDynamicFusedMoEMethod,
    ],
)
def test_v2_eplb_unsupported_quantization_is_explicit(scheme_cls):
    assert scheme_cls.supports_eplb is False
