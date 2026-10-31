# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.catccos_adapter import (
    CatccosLayerCapability,
    evaluate_catccos_layer,
    validate_catccos_operands,
    validate_catccos_weight_shapes,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType


def _moe_config(**overrides):
    values = {
        "in_dtype": torch.bfloat16,
        "hidden_dim": 2048,
        "intermediate_size_per_partition": 768,
        "ep_size": 4,
        "num_experts": 128,
        "experts_per_token": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _quant_method(**overrides):
    values = {"quant_type": QuantType.NONE, "dynamic_eplb": False}
    values.update(overrides)
    return SimpleNamespace(**values)


@patch(
    "vllm_ascend.ops.fused_moe.catccos_adapter.catccos_backend_enabled",
    return_value=True,
)
def test_qwen3_shape_is_supported(_mock_enabled):
    capability = evaluate_catccos_layer(
        _moe_config(),
        _quant_method(),
        "silu",
        device_type=AscendDeviceType.A5,
        library_exists=True,
    )
    assert capability == CatccosLayerCapability(True)


@patch(
    "vllm_ascend.ops.fused_moe.catccos_adapter.catccos_backend_enabled",
    return_value=True,
)
def test_unsupported_layer_fails_closed(_mock_enabled):
    capability = evaluate_catccos_layer(
        _moe_config(num_experts=127),
        _quant_method(),
        "silu",
        device_type=AscendDeviceType.A5,
        library_exists=True,
    )
    assert not capability.supported
    assert "divisible by EP size" in capability.reason


@patch(
    "vllm_ascend.ops.fused_moe.catccos_adapter.catccos_backend_enabled",
    return_value=True,
)
def test_deepseek_v4_flash_configuration_is_supported(_mock_enabled):
    capability = evaluate_catccos_layer(
        _moe_config(
            hidden_dim=7168,
            intermediate_size_per_partition=2048,
            ep_size=4,
            num_experts=256,
            experts_per_token=8,
            dp_size=4,
        ),
        _quant_method(quant_type=QuantType.W4A8MXFP),
        "silu",
        n_shared_experts=1,
        swiglu_limit=10.0,
        dp_size=4,
        device_type=AscendDeviceType.A5,
        library_exists=True,
    )
    assert capability == CatccosLayerCapability(True)


@patch(
    "vllm_ascend.ops.fused_moe.catccos_adapter.catccos_backend_enabled",
    return_value=True,
)
def test_deepseek_v4_quantization_adapter_is_unwrapped(_mock_enabled):
    quant_scheme = _quant_method(quant_type=QuantType.W4A8MXFP)
    quant_adapter = SimpleNamespace(quant_method=quant_scheme)
    capability = evaluate_catccos_layer(
        _moe_config(
            hidden_dim=7168,
            intermediate_size_per_partition=2048,
            ep_size=4,
            num_experts=256,
            experts_per_token=8,
        ),
        quant_adapter,
        "silu",
        n_shared_experts=1,
        swiglu_limit=10.0,
        device_type=AscendDeviceType.A5,
        library_exists=True,
    )
    assert capability == CatccosLayerCapability(True)


def test_qwen3_operator_weight_shapes_are_supported():
    validate_catccos_weight_shapes(
        torch.empty(32, 1536, 2048, device="meta"),
        torch.empty(32, 2048, 768, device="meta"),
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"expert_map_path": "/tmp/map"}, "contiguous expert placement"),
        ({"mix_placement": True}, "contiguous expert placement"),
        ({"log2phy": torch.empty(1, device="meta")}, "contiguous expert placement"),
        ({"swiglu_limit": -1.0}, "swiglu_limit"),
        ({"apply_router_weight_on_input": True}, "router weights"),
        ({"has_bias": True}, "expert bias"),
        ({"zero_expert_type": "default"}, "zero experts"),
        ({"n_shared_experts": 2}, "shared expert"),
        ({"is_sequence_parallel": True}, "sequence parallelism"),
        ({"pcp_size": 2}, "PCP size 1"),
        ({"speculative_model": True}, "speculative MoE"),
        ({"multistream_overlap_gate": True}, "multistream"),
    ],
)
@patch("vllm_ascend.ops.fused_moe.catccos_adapter.catccos_backend_enabled", return_value=True)
def test_unsupported_catccos_semantics_fail_closed(_mock_enabled, overrides, reason):
    capability = evaluate_catccos_layer(
        _moe_config(),
        _quant_method(),
        "silu",
        device_type=AscendDeviceType.A5,
        library_exists=True,
        **overrides,
    )
    assert not capability.supported
    assert reason in capability.reason


@patch("vllm_ascend.ops.fused_moe.catccos_adapter.catccos_backend_enabled", return_value=True)
def test_catccos_rejects_single_rank_expert_parallelism(_mock_enabled):
    capability = evaluate_catccos_layer(
        _moe_config(ep_size=1),
        _quant_method(),
        "silu",
        device_type=AscendDeviceType.A5,
        library_exists=True,
    )
    assert not capability.supported
    assert "expert parallelism" in capability.reason


def _operator_operands():
    x = torch.empty((2, 2048), dtype=torch.bfloat16, device="meta")
    expert_idx = torch.empty((2, 8), dtype=torch.int64, device="meta")
    gate_weight = torch.empty((2, 8), dtype=torch.bfloat16, device="meta")
    w1 = torch.empty((32, 1536, 2048), dtype=torch.float8_e4m3fn, device="meta")
    w1_scale = torch.empty((32, 1536, 64), dtype=torch.float8_e8m0fnu, device="meta")
    w2 = torch.empty((32, 2048, 768), dtype=torch.float8_e4m3fn, device="meta")
    w2_scale = torch.empty((32, 2048, 24), dtype=torch.float8_e8m0fnu, device="meta")
    return x, expert_idx, gate_weight, w1, w1_scale, w2, w2_scale


def test_catccos_operand_contract_accepts_qwen3_shapes():
    validate_catccos_operands(*_operator_operands(), ep_size=4)


def test_catccos_operand_contract_rejects_bad_scale_shape():
    operands = list(_operator_operands())
    operands[4] = torch.empty((32, 1536, 63), dtype=torch.float8_e8m0fnu, device="meta")
    with pytest.raises(ValueError, match="w1_scale"):
        validate_catccos_operands(*operands, ep_size=4)


def _mxfp4_operator_operands():
    x = torch.empty((2, 2048), dtype=torch.bfloat16, device="meta")
    expert_idx = torch.empty((2, 8), dtype=torch.int64, device="meta")
    gate_weight = torch.empty((2, 8), dtype=torch.float32, device="meta")
    w1 = torch.empty((32, 1024, 1536), dtype=torch.uint8, device="meta")
    w1_scale = torch.empty((32, 32, 1536, 2), dtype=torch.uint8, device="meta")
    w2 = torch.empty((32, 384, 2048), dtype=torch.uint8, device="meta")
    w2_scale = torch.empty((32, 12, 2048, 2), dtype=torch.uint8, device="meta")
    return x, expert_idx, gate_weight, w1, w1_scale, w2, w2_scale


def test_catccos_mxfp4_weight_shapes_are_supported():
    operands = _mxfp4_operator_operands()
    validate_catccos_weight_shapes(operands[3], operands[5], weight_format="mxfp4")
    validate_catccos_operands(*operands, ep_size=4, weight_format="mxfp4")


def test_catccos_mxfp4_operand_contract_rejects_bad_scale_shape():
    operands = list(_mxfp4_operator_operands())
    operands[4] = torch.empty((32, 31, 1536, 2), dtype=torch.uint8, device="meta")
    with pytest.raises(ValueError, match="w1_scale"):
        validate_catccos_operands(*operands, ep_size=4, weight_format="mxfp4")
