from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.fused_moe.mega_moe_adapter import (
    CannMegaMoeApiCapability,
    CannMegaMoeLayerCapability,
    evaluate_cann_mega_moe_layer,
    get_model_cann_mega_moe_capability,
    probe_cann_mega_moe_api,
    resolve_cann_mega_moe_activation,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType


@pytest.fixture
def a5_api_capability():
    return CannMegaMoeApiCapability(
        available=True,
        supports_situ=True,
        supports_comm_context_preload=True,
    )


def _moe_config(**overrides):
    values = {
        "in_dtype": torch.bfloat16,
        "hidden_dim": 3584,
        "intermediate_size_per_partition": 3072,
        "experts_per_token": 16,
        "ep_size": 32,
        "num_experts": 896,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _quant_method(quant_type=QuantType.W4A8MXFP, group_size=32):
    return SimpleNamespace(quant_type=quant_type, group_size=group_size)


def _compatible_mega_moe(*args, **kwargs):
    return args, kwargs


def _situ_mega_moe(
    x,
    topk_ids,
    topk_weights,
    l1_weights,
    l2_weights,
    sym_buffer,
    *,
    l1_weights_sf=None,
    l2_weights_sf=None,
    l1_bias=None,
    l2_bias=None,
    x_active_mask=None,
    activation="swiglu",
    activation_clamp=None,
    activation_params=None,
    weight1_type=None,
    weight2_type=None,
):
    return x, topk_ids


def test_api_probe_separates_base_backend_from_a5_extensions():
    ops_module = SimpleNamespace(
        mega_moe=_compatible_mega_moe,
        get_symm_buffer_for_mega_moe=object(),
    )

    probe_cann_mega_moe_api.cache_clear()
    with patch(
        "vllm_ascend.ops.fused_moe.mega_moe_adapter.import_module",
        side_effect=[ops_module, ImportError("comm context is unavailable")],
    ):
        capability = probe_cann_mega_moe_api()
    probe_cann_mega_moe_api.cache_clear()

    assert capability.available
    assert not capability.supports_situ
    assert not capability.supports_comm_context_preload


def test_api_probe_recognizes_explicit_a5_situ_contract():
    ops_module = SimpleNamespace(
        mega_moe=_situ_mega_moe,
        get_symm_buffer_for_mega_moe=object(),
    )
    comm_context_module = SimpleNamespace(
        comm_context_op_builder=SimpleNamespace(load=lambda: None),
    )

    probe_cann_mega_moe_api.cache_clear()
    with patch(
        "vllm_ascend.ops.fused_moe.mega_moe_adapter.import_module",
        side_effect=[ops_module, comm_context_module],
    ):
        capability = probe_cann_mega_moe_api()
    probe_cann_mega_moe_api.cache_clear()

    assert capability.available
    assert capability.supports_situ
    assert capability.supports_comm_context_preload


def test_a5_capability_uses_instantiated_layer_contract(a5_api_capability):
    capability = evaluate_cann_mega_moe_layer(
        _moe_config(),
        _quant_method(),
        MoEActivation.SITU,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        device_type=AscendDeviceType.A5,
        api_capability=a5_api_capability,
    )

    assert capability.supported
    assert capability.quant_type == QuantType.W4A8MXFP
    assert capability.activation is not None
    assert capability.activation.name == "situglu"
    assert capability.activation.alpha == 25.0
    assert capability.activation.beta == 4.0


def test_a5_capability_unwraps_the_runtime_quant_scheme(a5_api_capability):
    wrapped_quant_method = SimpleNamespace(quant_method=_quant_method())

    capability = evaluate_cann_mega_moe_layer(
        _moe_config(),
        wrapped_quant_method,
        MoEActivation.SILU,
        device_type=AscendDeviceType.A5,
        api_capability=a5_api_capability,
    )

    assert capability.supported
    assert capability.quant_type == QuantType.W4A8MXFP


@pytest.mark.parametrize(
    ("config_overrides", "quant_overrides", "reason"),
    [
        ({}, {"group_size": 64}, "group_size=32"),
        ({"hidden_dim": 3600}, {}, "hidden size"),
        ({"experts_per_token": 33}, {}, "top-k"),
        ({"num_experts": 895}, {}, "divisible by EP size"),
    ],
)
def test_a5_capability_rejects_unsupported_runtime_layout(
    a5_api_capability,
    config_overrides,
    quant_overrides,
    reason,
):
    capability = evaluate_cann_mega_moe_layer(
        _moe_config(**config_overrides),
        _quant_method(**quant_overrides),
        MoEActivation.SILU,
        device_type=AscendDeviceType.A5,
        api_capability=a5_api_capability,
    )

    assert not capability.supported
    assert reason in capability.reason


def test_a3_keeps_unquantized_megamoe_support(a5_api_capability):
    capability = evaluate_cann_mega_moe_layer(
        _moe_config(num_experts=128, ep_size=8),
        _quant_method(QuantType.NONE, group_size=None),
        MoEActivation.SILU,
        device_type=AscendDeviceType.A3,
        api_capability=a5_api_capability,
    )

    assert capability.supported
    assert capability.quant_type == QuantType.NONE


def test_a3_rejects_mxfp4_instead_of_entering_incompatible_operator(a5_api_capability):
    capability = evaluate_cann_mega_moe_layer(
        _moe_config(num_experts=128, ep_size=8),
        _quant_method(),
        MoEActivation.SILU,
        device_type=AscendDeviceType.A3,
        api_capability=a5_api_capability,
    )

    assert not capability.supported
    assert "W4A8MXFP" in capability.reason


def test_situ_requires_linear_beta():
    activation = resolve_cann_mega_moe_activation(MoEActivation.SITU, situ_beta=1.0)

    assert activation is None


def test_model_capability_requires_every_registered_moe_layer():
    model = torch.nn.Module()
    model.first_moe = torch.nn.Module()
    model.second_moe = torch.nn.Module()
    supported = CannMegaMoeLayerCapability(True, "", QuantType.W4A8MXFP)
    unsupported = CannMegaMoeLayerCapability(False, "unsupported activation", QuantType.W4A8MXFP)
    model.first_moe.cann_mega_moe_capability = supported
    model.second_moe.cann_mega_moe_capability = unsupported

    assert get_model_cann_mega_moe_capability(model) is unsupported


def test_model_capability_ignores_checkpoint_metadata_without_registered_layers():
    model = torch.nn.Module()
    model.quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized",
    }

    capability = get_model_cann_mega_moe_capability(model)

    assert not capability.supported
    assert "no registered" in capability.reason
