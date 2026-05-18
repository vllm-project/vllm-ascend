#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
# Todo: Once https://github.com/vllm-project/vllm/pull/23553 is merged in vllm. Remove this model register.
import types

import torch


# Module-level registry populated by layers during __init__.
# Each entry is (global_layer_idx, layer).
_moe_layer_entries: list[tuple[int, "torch.nn.Module"]] = []


def register_moe_layer(global_idx: int, layer: "torch.nn.Module") -> None:
    """Register a MoE layer with EPLB. Called during layer initialization.

    Only real layers call this; PPMissingLayer (nn.Identity) won't,
    so the registry naturally contains only layers on this PP rank.
    """
    _moe_layer_entries.append((global_idx, layer))


def _clear_moe_layer_entries() -> None:
    """Reset the registry before building a new model."""
    _moe_layer_entries.clear()


def _get_expert_map(self, layer_id):
    return self._moe_layer_map[layer_id]._expert_map


def _get_log2phy_map(self, layer_id):
    return self._moe_layer_map[layer_id].get_log2phy_map()


def _get_all_moe_loads(self):
    loads = [
        layer.moe_load
        for _, layer in self._moe_layers
    ]
    return torch.stack(loads, dim=0) if loads else torch.empty(0)


def _clear_all_moe_loads(self):
    for _, layer in self._moe_layers:
        layer.clear_moe_load()


def model_register(model):
    if hasattr(model, "language_model"):
        model = model.language_model

    # Sort by global index ascending
    _moe_layer_entries.sort(key=lambda x: x[0])
    model._moe_layers = list(_moe_layer_entries)
    model._moe_layer_map = dict(_moe_layer_entries)

    model.get_expert_map = types.MethodType(_get_expert_map, model)
    model.get_log2phy_map = types.MethodType(_get_log2phy_map, model)
    model.get_all_moe_loads = types.MethodType(_get_all_moe_loads, model)
    model.clear_all_moe_loads = types.MethodType(_clear_all_moe_loads, model)

    # Clear after use to avoid leaking between model loads
    _clear_moe_layer_entries()
