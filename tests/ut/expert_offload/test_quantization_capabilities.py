# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.expert_offload.config import ExpertOffloadConfig
from vllm_ascend.expert_offload.expert_offload_manager import LayerOffloadBinding
from vllm_ascend.expert_offload.quantization import MXFP4ExpertAdapter, MXFP8ExpertAdapter
from vllm_ascend.quantization.quant_type import QuantType


def layer_and_scheme(adapter=MXFP4ExpertAdapter):
    is_mxfp8 = adapter is MXFP8ExpertAdapter
    weight_dtype = torch.float8_e4m3fn if is_mxfp8 else torch.uint8
    packing_factor = 1 if is_mxfp8 else 2
    layer = SimpleNamespace(
        layer_name="model.layers.0.mlp.experts",
        local_num_experts=6,
        top_k=2,
        activation="silu",
        moe_config=SimpleNamespace(device="cpu", is_sequence_parallel=False, has_bias=False),
        w13_weight=torch.zeros(6, 128, 64 // packing_factor, dtype=weight_dtype),
        w2_weight=torch.zeros(6, 64, 64 // packing_factor, dtype=weight_dtype),
        w13_weight_scale=torch.zeros(6, 128, 2, dtype=torch.uint8),
        w2_weight_scale=torch.zeros(6, 64, 2, dtype=torch.uint8),
    )
    return layer, SimpleNamespace(
        quant_type=QuantType.W8A8MXFP if is_mxfp8 else QuantType.W4A8MXFP,
        group_size=32,
        process_weights_after_loading=Mock(),
    )


@pytest.mark.parametrize("adapter", [MXFP4ExpertAdapter, MXFP8ExpertAdapter])
@pytest.mark.parametrize("name,size", [("w13_weight", 192), ("w2_weight", 96), ("w13_weight", 0), ("w2_weight", 0)])
def test_rejects_unsupported_alignment_before_host_allocation(monkeypatch, adapter, name, size):
    layer, scheme = layer_and_scheme(adapter)
    tensor = getattr(layer, name)
    setattr(layer, name, torch.zeros(tensor.shape[0], size, tensor.shape[2], dtype=tensor.dtype))
    store = Mock()
    monkeypatch.setattr("vllm_ascend.expert_offload.expert_offload_manager.HostExpertStore", store)
    with pytest.raises(ValueError, match="divisible by 64"):
        LayerOffloadBinding(layer, scheme, ExpertOffloadConfig(enabled=True, num_device_experts=2))
    store.assert_not_called()


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
