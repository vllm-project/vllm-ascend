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
import re
import types
from typing import Optional

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


class MoeLayerRegistry:
    """Per-model view of registered MoE layers, with lookup by global index."""

    def __init__(self, entries: list[tuple[int, "torch.nn.Module"]]):
        # Sort by global index ascending
        entries.sort(key=lambda x: x[0])
        self._entries = entries
        self._global_to_local: dict[int, int] = {
            idx: i for i, (idx, _) in enumerate(entries)
        }

    @property
    def num_layers(self) -> int:
        return len(self._entries)

    def get_global_index(self, local_idx: int) -> int:
        """local MoE index (0, 1, ...) -> global layer index."""
        return self._entries[local_idx][0]

    def get_layer(self, global_idx: int) -> Optional["torch.nn.Module"]:
        """Get layer by global index, returns None if not on this rank."""
        local_idx = self._global_to_local.get(global_idx)
        if local_idx is None:
            return None
        return self._entries[local_idx][1]

    def has_layer(self, global_idx: int) -> bool:
        return global_idx in self._global_to_local

    def iter_layers(self):
        """Iterate (global_idx, layer) pairs in order."""
        yield from self._entries


def _get_expert_map(self, layer_id):
    return self._moe_layer_registry.get_layer(layer_id).mlp.experts.expert_map


def _get_log2phy_map(self, layer_id):
    return self._moe_layer_registry.get_layer(layer_id).mlp.experts.get_log2phy_map()


def _get_all_moe_loads(self):
    loads = [
        layer.mlp.experts.moe_load
        for _, layer in self._moe_layer_registry.iter_layers()
    ]
    return torch.stack(loads, dim=0) if loads else torch.empty(0)


def _clear_all_moe_loads(self):
    for _, layer in self._moe_layer_registry.iter_layers():
        layer.mlp.experts.clear_moe_load()


def model_register(model):
    if hasattr(model, "language_model"):
        model = model.language_model

    # Build registry from layers that called register_moe_layer() during init
    registry = MoeLayerRegistry(list(_moe_layer_entries))
    model._moe_layer_registry = registry

    model.get_expert_map = types.MethodType(_get_expert_map, model)
    model.get_log2phy_map = types.MethodType(_get_log2phy_map, model)
    model.get_all_moe_loads = types.MethodType(_get_all_moe_loads, model)
    model.clear_all_moe_loads = types.MethodType(_clear_all_moe_loads, model)

    # Clear after use to avoid leaking between model loads
    _clear_moe_layer_entries()
