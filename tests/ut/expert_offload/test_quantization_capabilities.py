# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.expert_offload.config import ExpertOffloadConfig
from vllm_ascend.expert_offload.expert_offload_manager import LayerOffloadBinding
from vllm_ascend.expert_offload.quantization import MXFP4ExpertAdapter


def layer_and_scheme():
    layer = SimpleNamespace(
        layer_name="model.layers.0.mlp.experts",
        local_num_experts=6,
        top_k=2,
        activation="silu",
        weight_loader=Mock(),
        moe_config=SimpleNamespace(device="cpu", is_sequence_parallel=False, has_bias=False),
        w13_weight=torch.zeros(6, 128, 32, dtype=torch.uint8),
        w2_weight=torch.zeros(6, 64, 32, dtype=torch.uint8),
        w13_weight_scale=torch.zeros(6, 128, 2, dtype=torch.uint8),
        w2_weight_scale=torch.zeros(6, 64, 2, dtype=torch.uint8),
    )
    return layer, SimpleNamespace(group_size=32, process_weights_after_loading=Mock())


@pytest.mark.parametrize("activation", ["silu", "situ", SimpleNamespace(value="silu")])
@pytest.mark.parametrize("semantic_scales", [False, True])
def test_supported_packed_contract(activation, semantic_scales):
    layer, scheme = layer_and_scheme()
    layer.activation = activation
    if semantic_scales:
        for name in ("w13_weight_scale", "w2_weight_scale"):
            setattr(layer, name, getattr(layer, name).view(torch.float8_e8m0fnu))
    MXFP4ExpertAdapter(layer, scheme).validate_layer(layer)
    scheme.process_weights_after_loading.assert_not_called()


@pytest.mark.parametrize("model_type", ["deepseek_v4", "other_moe"])
@pytest.mark.parametrize("name", MXFP4ExpertAdapter.tensor_names)
def test_model_name_does_not_hide_missing_tensor(model_type, name):
    layer, scheme = layer_and_scheme()
    layer.model_type = model_type
    delattr(layer, name)
    with pytest.raises(ValueError, match=name):
        MXFP4ExpertAdapter(layer, scheme).validate_layer(layer)


@pytest.mark.parametrize("name", MXFP4ExpertAdapter.tensor_names)
@pytest.mark.parametrize("invalid", ["dtype", "shape"])
def test_rejects_incompatible_weight_or_scale_before_host_allocation(monkeypatch, name, invalid):
    layer, scheme = layer_and_scheme()
    tensor = getattr(layer, name)
    setattr(layer, name, tensor.float() if invalid == "dtype" else tensor[:5])
    store = Mock()
    monkeypatch.setattr("vllm_ascend.expert_offload.expert_offload_manager.HostExpertStore", store)
    with pytest.raises(ValueError, match=name):
        LayerOffloadBinding(layer, scheme, ExpertOffloadConfig(enabled=True, num_device_experts=2))
    store.assert_not_called()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("activation", "gelu", "activation"),
        ("weight_loader", None, "weight_loader"),
    ],
)
def test_rejects_missing_moe_capability(field, value, match):
    layer, scheme = layer_and_scheme()
    setattr(layer, field, value)
    with pytest.raises(ValueError, match=match):
        MXFP4ExpertAdapter(layer, scheme).validate_layer(layer)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("group_size", 16, "group_size=32"),
        ("process_weights_after_loading", None, "post-processing"),
    ],
)
def test_rejects_incompatible_scheme(field, value, match):
    layer, scheme = layer_and_scheme()
    setattr(scheme, field, value)
    with pytest.raises(ValueError, match=match):
        MXFP4ExpertAdapter(layer, scheme)
