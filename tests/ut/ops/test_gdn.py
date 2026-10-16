#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from types import SimpleNamespace

import torch
from torch import nn
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)

from vllm_ascend.ops.gdn import (
    _PACKED_CONV_WEIGHT_NAME,
    _get_base_conv1d,
    _get_packed_conv_weights,
    initialize_packed_conv_weight,
)


class _RecordingQuantMethod(QuantizeMethodBase):
    def __init__(self, events: list[str] | None = None):
        self.events = events if events is not None else []

    def create_weights(self, layer, *args, **kwargs):
        return None

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        self.events.append("process")


def _make_layer(weight: torch.Tensor, *, lora: bool = False, events=None) -> nn.Module:
    layer = nn.Module()
    layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
    base_conv = nn.Module()
    base_conv.weight = nn.Parameter(weight)
    base_conv.quant_method = _RecordingQuantMethod(events)
    if lora:
        layer.conv1d = nn.Module()
        layer.conv1d.base_layer = base_conv
    else:
        layer.conv1d = base_conv
    return layer


def _source_weight() -> torch.Tensor:
    return torch.arange(18 * 4, dtype=torch.float32).reshape(18, 1, 4)


def test_packed_weight_is_registered_on_base_conv():
    layer = _make_layer(_source_weight())

    initialize_packed_conv_weight(layer)

    base_conv = _get_base_conv1d(layer)
    packed = base_conv.get_parameter(_PACKED_CONV_WEIGHT_NAME)
    assert isinstance(packed, nn.Parameter)
    assert not packed.requires_grad
    assert packed.shape == (4, 18)
    assert packed.dtype == torch.bfloat16
    assert packed.device == base_conv.weight.device
    assert packed.is_contiguous()
    assert _PACKED_CONV_WEIGHT_NAME not in layer._parameters


def test_updated_packed_value_keeps_data_ptr():
    events: list[str] = []
    layer = _make_layer(_source_weight(), events=events)
    initialize_packed_conv_weight(layer)
    base_conv = _get_base_conv1d(layer)

    base_conv.quant_method.process_weights_after_loading(base_conv)
    packed = base_conv.get_parameter(_PACKED_CONV_WEIGHT_NAME)
    packed_ptr = packed.data_ptr()
    expected = base_conv.weight.squeeze(1).transpose(0, 1).to(packed.dtype)
    torch.testing.assert_close(packed, expected)

    with torch.no_grad():
        base_conv.weight.fill_(7.0)
    base_conv.quant_method.process_weights_after_loading(base_conv)

    assert packed.data_ptr() == packed_ptr
    torch.testing.assert_close(
        packed,
        base_conv.weight.squeeze(1).transpose(0, 1).to(packed.dtype),
    )
    assert events == ["process", "process"]


def test_packed_getter_resolves_lora_base_layer():
    layer = _make_layer(_source_weight(), lora=True)
    initialize_packed_conv_weight(layer)

    packed = _get_packed_conv_weights(layer)
    assert packed is layer.conv1d.base_layer.get_parameter(_PACKED_CONV_WEIGHT_NAME)
    assert _PACKED_CONV_WEIGHT_NAME not in layer.conv1d._parameters


def test_meta_weight_keeps_meta_placeholder():
    layer = _make_layer(torch.empty(18, 1, 4, device="meta"))
    initialize_packed_conv_weight(layer)

    packed = _get_packed_conv_weights(layer)
    assert packed.is_meta
    assert packed.shape == (4, 18)
    assert packed.data_ptr() == 0


def test_layerwise_reload_repacks_updated_source_in_place():
    layer = _make_layer(_source_weight())
    model = nn.Sequential(layer)
    initialize_packed_conv_weight(layer)
    base_conv = _get_base_conv1d(layer)
    base_conv.quant_method.process_weights_after_loading(base_conv)
    source_ptr = base_conv.weight.data_ptr()
    packed_ptr = base_conv.get_parameter(_PACKED_CONV_WEIGHT_NAME).data_ptr()

    loaded = torch.full_like(base_conv.weight, 11.0)
    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    base_conv.weight.weight_loader(base_conv.weight, loaded)
    finalize_layerwise_reload(model, model_config=None)

    assert base_conv.weight.data_ptr() == source_ptr
    assert base_conv.get_parameter(_PACKED_CONV_WEIGHT_NAME).data_ptr() == packed_ptr
    torch.testing.assert_close(base_conv.weight, loaded)
    torch.testing.assert_close(
        base_conv.get_parameter(_PACKED_CONV_WEIGHT_NAME),
        loaded.squeeze(1).transpose(0, 1).to(torch.bfloat16),
    )
