from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ops.fused_moe.mega_moe import MegaMoEBackend
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


def _make_fused_input(*, swiglu_limit=0.0, log2phy=None, scale_dtype=torch.uint8):
    return build_fused_experts_input(
        hidden_states=torch.randn(4, 8),
        topk_weights=torch.randn(4, 2),
        topk_ids=torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]]),
        w1=torch.randn(2, 16, 8),
        w2=torch.randn(2, 8, 8),
        quant_type=QuantType.MXFP8,
        dynamic_eplb=False,
        log2phy=log2phy,
        activation="silu",
        mxfp_act_quant_type=torch.float8_e4m3fn,
        mxfp_weight_quant_type=torch.float8_e4m3fn,
        mxfp_scale_dtype=torch.uint8,
        mxfp_per_token_scale_dtype=torch.uint8,
        w1_scale=torch.ones(2, 16, 1, dtype=scale_dtype),
        w2_scale=torch.ones(2, 8, 1, dtype=scale_dtype),
        swiglu_limit=swiglu_limit,
    )


def test_mega_moe_backend_builds_buffer_and_operator_args():
    fused_input = _make_fused_input(log2phy=torch.tensor([3, 2, 1, 0]))
    get_symm_buffer = MagicMock(return_value=object())
    mega_moe = MagicMock(return_value=(torch.randn(4, 8), torch.tensor([1, 3], dtype=torch.int32)))

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe._get_mega_moe_ops", return_value=(get_symm_buffer, mega_moe)),
        patch("vllm_ascend.ops.fused_moe.mega_moe.FLOAT8_E8M0FNU_DTYPE", torch.uint8),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_ascend_config",
            return_value=SimpleNamespace(mega_moe_max_tokens=65536),
        ),
        patch("vllm_ascend.ops.fused_moe.mega_moe.get_ep_group", return_value=SimpleNamespace(device_group=object())),
    ):
        backend = MegaMoEBackend(_make_moe_config())
        backend.fused_experts(fused_input)

    get_symm_buffer.assert_called_once()
    assert get_symm_buffer.call_args.kwargs["intermediate_hidden"] == 8
    assert get_symm_buffer.call_args.kwargs["num_max_tokens_per_rank"] == 512
    assert get_symm_buffer.call_args.kwargs["dispatch_quant_mode"] == 4

    mega_moe.assert_called_once()
    mega_moe_kwargs = mega_moe.call_args.kwargs
    assert torch.equal(mega_moe_kwargs["topk_ids"], fused_input.routing.log2phy[fused_input.topk_ids].to(torch.int32))
    assert mega_moe_kwargs["activation"] == "swiglu"
    assert mega_moe_kwargs["activation_clamp"] is None
    assert "activation_params" not in mega_moe_kwargs
    assert mega_moe_kwargs["weight1_type"] == torch.float8_e4m3fn
    assert mega_moe_kwargs["weight2_type"] == torch.float8_e4m3fn


def test_mega_moe_backend_passes_positive_swiglu_limit_as_clamp():
    fused_input = _make_fused_input(swiglu_limit=7.0)
    get_symm_buffer = MagicMock(return_value=object())
    mega_moe = MagicMock(return_value=(torch.randn(4, 8), torch.tensor([1, 3], dtype=torch.int32)))

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe._get_mega_moe_ops", return_value=(get_symm_buffer, mega_moe)),
        patch("vllm_ascend.ops.fused_moe.mega_moe.FLOAT8_E8M0FNU_DTYPE", torch.uint8),
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
    assert MegaMoEBackend._normalize_activation(_TestMoEActivation.SWIGLUOAI) == "swiglu"
    assert MegaMoEBackend._normalize_activation(_TestMoEActivation.SWIGLUSTEP) == "swiglu"


def test_mega_moe_backend_rejects_packed_uint64_scales():
    fused_input = _make_fused_input(scale_dtype=torch.uint64)

    with patch("vllm_ascend.ops.fused_moe.mega_moe.FLOAT8_E8M0FNU_DTYPE", torch.uint8):
        backend = MegaMoEBackend(_make_moe_config())
        try:
            backend.fused_experts(fused_input)
        except RuntimeError as exc:
            assert "Do not pass packed INT/Fused-MC2 UINT64 scales into MegaMoE" in str(exc)
        else:
            raise AssertionError("Expected uint64 scales to be rejected.")
