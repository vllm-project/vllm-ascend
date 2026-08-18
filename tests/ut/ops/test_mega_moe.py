from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ops.fused_moe.mega_moe import MegaMoEBackend, _view_mxfp_scales_as_e8m0
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.quant_type import QuantType


class _TestMoEActivation(Enum):
    SILU = 1
    SWIGLUOAI = 2
    SWIGLUSTEP = 3


def _make_moe_config():
    moe_config = MagicMock(spec=FusedMoEConfig)
    moe_config.num_experts = 8
    moe_config.experts_per_token = 2
    return moe_config


def _make_fused_input(
    *,
    swiglu_limit=0.0,
    log2phy=None,
    scale_dtype=torch.uint8,
    activation="silu",
    dynamic_eplb=False,
    global_redundant_expert_num=0,
    w1=None,
):
    return build_fused_experts_input(
        hidden_states=torch.randn(4, 8),
        topk_weights=torch.randn(4, 2),
        topk_ids=torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]]),
        w1=torch.randn(2, 16, 8) if w1 is None else w1,
        w2=torch.randn(2, 8, 8),
        quant_type=QuantType.MXFP8,
        dynamic_eplb=dynamic_eplb,
        log2phy=log2phy,
        global_redundant_expert_num=global_redundant_expert_num,
        activation=activation,
        mxfp_act_quant_type=torch.float8_e4m3fn,
        mxfp_weight_quant_type=torch.float8_e4m3fn,
        mxfp_scale_dtype=torch.uint8,
        mxfp_per_token_scale_dtype=torch.uint8,
        w1_scale=torch.ones(2, 16, 1, 2, dtype=scale_dtype),
        w2_scale=torch.ones(2, 8, 1, 2, dtype=scale_dtype),
        swiglu_limit=swiglu_limit,
    )


def test_mega_moe_backend_builds_buffer_and_operator_args():
    fused_input = _make_fused_input(log2phy=torch.tensor([3, 2, 1, 0]))
    get_symm_buffer = MagicMock(return_value=object())
    mega_moe = MagicMock(return_value=(torch.randn(4, 8), torch.tensor([1, 3], dtype=torch.int32)))

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe._get_mega_moe_ops", return_value=(get_symm_buffer, mega_moe)),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_ascend_config",
            return_value=SimpleNamespace(mega_moe_max_tokens=65536),
        ),
        patch("vllm_ascend.ops.fused_moe.mega_moe.get_ep_group", return_value=SimpleNamespace(device_group=object())),
    ):
        backend = MegaMoEBackend(_make_moe_config())
        backend.fused_experts(fused_input)

    get_symm_buffer.assert_called_once()
    assert get_symm_buffer.call_args.kwargs["intermediate_hidden"] == 16
    assert get_symm_buffer.call_args.kwargs["num_max_tokens_per_rank"] == 65536
    assert get_symm_buffer.call_args.kwargs["dispatch_quant_mode"] == 4

    mega_moe.assert_called_once()
    mega_moe_kwargs = mega_moe.call_args.kwargs
    assert torch.equal(mega_moe_kwargs["topk_ids"], fused_input.routing.log2phy[fused_input.topk_ids].to(torch.int32))
    assert mega_moe_kwargs["activation"] == "swiglu"
    assert mega_moe_kwargs["activation_clamp"] is None
    assert "activation_params" not in mega_moe_kwargs
    assert mega_moe_kwargs["weight1_type"] == torch.float8_e4m3fn
    assert mega_moe_kwargs["weight2_type"] == torch.float8_e4m3fn
    assert mega_moe_kwargs["l1_weights_sf"][0].dtype == torch.float8_e8m0fnu
    assert mega_moe_kwargs["l2_weights_sf"][0].dtype == torch.float8_e8m0fnu


def test_view_mxfp_scales_as_e8m0_reinterprets_uint8_storage():
    scale = torch.arange(64, dtype=torch.uint8).reshape(2, 16, 1, 2)

    normalized_scale = _view_mxfp_scales_as_e8m0([scale], "w1_scale")[0]

    assert normalized_scale.dtype == torch.float8_e8m0fnu
    assert normalized_scale.shape == scale.shape
    assert normalized_scale.stride() == scale.stride()
    assert normalized_scale.numel() == scale.numel()
    assert normalized_scale.data_ptr() == scale.data_ptr()


def test_view_mxfp_scales_as_e8m0_requires_native_torch_dtype():
    scale = torch.ones(2, 16, 1, 2, dtype=torch.uint8)

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe._TORCH_FLOAT8_E8M0FNU_DTYPE", None),
        pytest.raises(RuntimeError, match=r"torch\.float8_e8m0fnu"),
    ):
        _view_mxfp_scales_as_e8m0([scale], "w1_scale")


def test_mega_moe_backend_passes_positive_swiglu_limit_as_clamp():
    fused_input = _make_fused_input(swiglu_limit=7.0)
    get_symm_buffer = MagicMock(return_value=object())
    mega_moe = MagicMock(return_value=(torch.randn(4, 8), torch.tensor([1, 3], dtype=torch.int32)))

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe._get_mega_moe_ops", return_value=(get_symm_buffer, mega_moe)),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_ascend_config",
            return_value=SimpleNamespace(mega_moe_max_tokens=512),
        ),
        patch("vllm_ascend.ops.fused_moe.mega_moe.get_ep_group", return_value=SimpleNamespace(device_group=object())),
    ):
        backend = MegaMoEBackend(_make_moe_config())
        backend.fused_experts(fused_input)

    assert mega_moe.call_args.kwargs["activation_clamp"] == 7.0


def test_mega_moe_backend_accepts_enum_activation():
    assert MegaMoEBackend._normalize_activation(_TestMoEActivation.SILU) == "swiglu"


@pytest.mark.parametrize("activation", [_TestMoEActivation.SWIGLUOAI, _TestMoEActivation.SWIGLUSTEP, "situglu"])
def test_mega_moe_backend_rejects_activation_with_different_semantics(activation):
    with pytest.raises(ValueError, match="without changing its semantics"):
        MegaMoEBackend._normalize_activation(activation)


@pytest.mark.parametrize(
    ("input_kwargs", "message"),
    [
        ({"dynamic_eplb": True}, "dynamic EPLB"),
        ({"global_redundant_expert_num": 1}, "redundant physical experts"),
    ],
)
def test_mega_moe_backend_rejects_unsupported_eplb_layouts(input_kwargs, message):
    backend = MegaMoEBackend(_make_moe_config())
    with pytest.raises(RuntimeError, match=message):
        backend.fused_experts(_make_fused_input(**input_kwargs))


def test_mega_moe_backend_rejects_transposed_grouped_matmul_layout():
    w1 = torch.randn(2, 8, 16).transpose(1, 2)
    fused_input = _make_fused_input(w1=w1)

    backend = MegaMoEBackend(_make_moe_config())
    with pytest.raises(ValueError, match="requires contiguous w1"):
        backend.fused_experts(fused_input)


def test_mega_moe_backend_validates_fp4_packed_dimensions():
    fused_input = build_fused_experts_input(
        hidden_states=torch.randn(4, 128),
        topk_weights=torch.randn(4, 2),
        topk_ids=torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]]),
        w1=torch.randint(0, 255, (2, 128, 64), dtype=torch.uint8),
        w2=torch.randint(0, 255, (2, 128, 32), dtype=torch.uint8),
        quant_type=QuantType.W4A8MXFP,
        dynamic_eplb=False,
        mxfp_act_quant_type=torch.float8_e4m3fn,
        mxfp_weight_quant_type=torch.uint8,
        mxfp_scale_dtype=torch.uint8,
        mxfp_per_token_scale_dtype=torch.uint8,
        w1_scale=torch.ones(2, 128, 2, 2, dtype=torch.uint8),
        w2_scale=torch.ones(2, 128, 1, 2, dtype=torch.uint8),
    )

    with patch(
        "vllm_ascend.ops.fused_moe.mega_moe.get_ascend_config",
        return_value=SimpleNamespace(mega_moe_max_tokens=512),
    ):
        key = MegaMoEBackend(_make_moe_config())._make_buffer_key(fused_input)

    assert key.hidden_size == 128
    assert key.intermediate_hidden == 128


def test_mega_moe_backend_rejects_input_over_configured_capacity():
    fused_input = _make_fused_input()
    with patch(
        "vllm_ascend.ops.fused_moe.mega_moe.get_ascend_config",
        return_value=SimpleNamespace(mega_moe_max_tokens=3),
    ):
        backend = MegaMoEBackend(_make_moe_config())
        with pytest.raises(ValueError, match="exceeds the configured capacity"):
            backend._make_buffer_key(fused_input)


def test_mega_moe_backend_rejects_packed_uint64_scales():
    fused_input = _make_fused_input(scale_dtype=torch.uint64)

    backend = MegaMoEBackend(_make_moe_config())
    try:
        backend.fused_experts(fused_input)
    except RuntimeError as exc:
        assert "Do not pass packed INT/Fused-MC2 UINT64 scales into MegaMoE" in str(exc)
    else:
        raise AssertionError("Expected uint64 scales to be rejected.")
