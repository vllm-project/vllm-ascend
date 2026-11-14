"""UT for vllm_ascend/ops/fused_moe/dataclass/moe_quant.py (MoEQuantParams / MoEMxfpParams)."""

import torch
import torch_npu
import pytest

from vllm_ascend.ops.fused_moe.dataclass.moe_quant import (
    MoEMxfpParams,
    MoEQuantParams,
    _build_mxfp_params,
    build_quant_params,
)
from vllm_ascend.quantization.quant_type import QuantType


MXFP_TYPES = (QuantType.W8A8MXFP, QuantType.W4A4MXFP, QuantType.W4A8MXFP, QuantType.W4A16MXFP)
DISPATCH_TYPES = (
    QuantType.W8A8,
    QuantType.W4A8,
    QuantType.W8A8MXFP,
    QuantType.W4A4MXFP,
    QuantType.W4A8MXFP,
    QuantType.W8A8FP,
)


def _expected_flags(quant_type: QuantType) -> dict:
    return {
        "is_quant": quant_type != QuantType.NONE,
        "is_mxfp": quant_type in MXFP_TYPES,
        "is_w4a4_mxfp": quant_type == QuantType.W4A4MXFP,
        "is_int_quant": quant_type in (QuantType.W8A8, QuantType.W4A8),
        "is_fp8": quant_type == QuantType.W8A8FP,
        "dispatch_with_quant": quant_type in DISPATCH_TYPES,
    }


def _expected_dst_type(quant_type: QuantType):
    if quant_type == QuantType.W4A4MXFP:
        return torch_npu.float4_e2m1fn_x2
    if quant_type in (QuantType.W8A8MXFP, QuantType.W4A8MXFP, QuantType.W4A16MXFP, QuantType.W8A8FP):
        return torch.float8_e4m3fn
    if quant_type in (QuantType.W8A8, QuantType.W4A8):
        return torch.int8
    return None


def _expected_scale_type(quant_type: QuantType):
    if quant_type in (QuantType.W8A8MXFP, QuantType.W4A4MXFP, QuantType.W4A8MXFP, QuantType.W4A16MXFP):
        return torch.float8_e8m0fnu
    if quant_type in DISPATCH_TYPES:
        return torch.float32
    return None


@pytest.mark.parametrize("quant_type", list(QuantType))
def test_property_matrix(quant_type):
    params = MoEQuantParams(quant_type=quant_type)
    expected = _expected_flags(quant_type)
    for property_name, expected_value in expected.items():
        assert getattr(params, property_name) is expected_value, property_name
    assert params.get_dst_type is _expected_dst_type(quant_type)
    assert params.get_scale_type is _expected_scale_type(quant_type)


def test_w4a4_mxfp_dst_type_is_float4_e2m1fn_x2():
    params = MoEQuantParams(quant_type=QuantType.W4A4MXFP)
    assert params.get_dst_type is torch_npu.float4_e2m1fn_x2


@pytest.mark.parametrize(
    ("quant_type", "per_channel", "expected"),
    [
        (QuantType.W4A8, True, True),
        (QuantType.W4A8, False, False),
        (QuantType.W8A8, True, False),
        (QuantType.W4A4MXFP, True, False),
    ],
)
def test_use_w4a8_per_channel_gmm_swiglu(quant_type, per_channel, expected):
    params = MoEQuantParams(quant_type=quant_type, is_per_channel_weight=per_channel)
    assert params.use_w4a8_per_channel_gmm_swiglu is expected


def test_moe_mxfp_params_defaults():
    params = MoEMxfpParams()
    assert params.act_quant_type is None
    assert params.weight_quant_type is None
    assert params.scale_dtype is None
    assert params.per_token_scale_dtype is None
    assert params.use_bf16 is True


@pytest.mark.parametrize("quant_type", [QuantType.NONE, QuantType.W8A8, QuantType.W4A16])
def test_build_mxfp_params_returns_none_for_non_mxfp(quant_type):
    assert (
        _build_mxfp_params(
            quant_type=quant_type,
            mxfp_act_quant_type=torch.float8_e4m3fn,
            mxfp_weight_quant_type=torch.float8_e4m3fn,
        )
        is None
    )


@pytest.mark.parametrize("quant_type", list(MXFP_TYPES))
def test_build_mxfp_params_requires_explicit_args(quant_type):
    with pytest.raises(ValueError, match="primitive MXFP params are required"):
        _build_mxfp_params(quant_type=quant_type)


def test_build_mxfp_params_defaults_use_bf16_when_unset():
    params = _build_mxfp_params(
        quant_type=QuantType.W4A4MXFP,
        mxfp_act_quant_type=torch.float8_e4m3fn,
        mxfp_weight_quant_type=torch.float8_e4m3fn,
    )
    assert isinstance(params, MoEMxfpParams)
    assert params.act_quant_type is torch.float8_e4m3fn
    assert params.weight_quant_type is torch.float8_e4m3fn
    assert params.scale_dtype is None
    assert params.per_token_scale_dtype is None
    assert params.use_bf16 is True


def test_build_mxfp_params_honors_explicit_use_bf16_false():
    params = _build_mxfp_params(
        quant_type=QuantType.W8A8MXFP,
        mxfp_scale_dtype=torch.float32,
        mxfp_use_bf16=False,
    )
    assert params.use_bf16 is False
    assert params.scale_dtype is torch.float32


def test_build_quant_params_passthrough_for_non_mxfp():
    params = build_quant_params(
        QuantType.W8A8,
        1,
        None,
        None,
        None,
        None,
        None,
        True,
    )
    assert params.quant_type is QuantType.W8A8
    assert params.comm_quant_mode == 1
    assert params.mxfp is None
    assert params.is_per_channel_weight is True


def test_build_quant_params_attaches_mxfp_leaf():
    params = build_quant_params(
        QuantType.W4A8MXFP,
        None,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        torch.float8_e8m0fnu,
        torch.float8_e8m0fnu,
        True,
        False,
    )
    assert params.is_per_channel_weight is False
    assert isinstance(params.mxfp, MoEMxfpParams)
    assert params.mxfp.act_quant_type is torch.float8_e4m3fn
    assert params.mxfp.weight_quant_type is torch.float8_e4m3fn
    assert params.mxfp.scale_dtype is torch.float8_e8m0fnu
    assert params.mxfp.per_token_scale_dtype is torch.float8_e8m0fnu
    assert params.mxfp.use_bf16 is True


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
