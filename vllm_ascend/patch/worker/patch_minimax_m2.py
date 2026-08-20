#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# MiniMax-M2 on Ascend: fused attention.
#

from collections.abc import Iterable

import torch
from vllm.model_executor.models.minimax_m2 import MiniMaxM2Attention, MiniMaxM2Model

from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_slice


# ---------------------------------------------------------------------------
# MiniMaxM2Attention: fused qkv split, rmsnorm, and rope on NPU.
# ---------------------------------------------------------------------------
def _patch_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    cos, sin = get_cos_and_sin_slice()
    q, k, v = torch.ops.vllm.split_qkv_tp_rmsnorm_rope(
        input=qkv,
        q_weight=self.q_norm.weight,
        k_weight=self.k_norm.weight,
        q_hidden_size=self.q_size,
        kv_hidden_size=self.kv_size,
        head_dim=self.head_dim,
        rotary_dim=getattr(self.rotary_emb, "rotary_dim", self.head_dim),
        eps=self.q_norm.variance_epsilon,
        tp_world=self.q_norm.tp_world,
        cos=cos,
        sin=sin,
    )
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


MiniMaxM2Attention.forward = _patch_forward


# ---------------------------------------------------------------------------
# MiniMaxM2Model: skip surplus decoder layers on a reduced config.
# ---------------------------------------------------------------------------
def _filter_reduced_layer_weights(
    self: "MiniMaxM2Model",
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Skip decoder layers that exceed the configured reduced stack.

    The MiniMax-M2.7 4-card e2e case loads the full 62-layer checkpoint with
    a 16-layer config (``num_hidden_layers=16``, ``num_hidden_layers_orig=62``).
    Layers 16..61 have no destination module, so they must be filtered before
    ``AutoWeightsLoader`` sees them.

    Note: ``num_hidden_layers_orig`` is a vllm-ascend convention injected via
    ``--hf-overrides`` (not an HF-native config field); it is always paired
    with ``num_hidden_layers`` in the e2e config.
    """
    num_layers = getattr(self.config, "num_hidden_layers", None)
    orig_layers = getattr(self.config, "num_hidden_layers_orig", None)
    if not isinstance(num_layers, int) or not isinstance(orig_layers, int) or orig_layers <= num_layers:
        yield from weights
        return

    for name, loaded_weight in weights:
        parts = name.split(".")
        if len(parts) > 1 and parts[0] == "layers" and parts[1].isdigit():
            if int(parts[1]) >= num_layers:
                continue
        yield name, loaded_weight


MiniMaxM2Model._filter_reduced_layer_weights = _filter_reduced_layer_weights

_original_load_weights = MiniMaxM2Model.load_weights


def _patched_load_weights(
    self: "MiniMaxM2Model",
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    weights = self._filter_reduced_layer_weights(weights)
    return _original_load_weights(self, weights)


MiniMaxM2Model.load_weights = _patched_load_weights
