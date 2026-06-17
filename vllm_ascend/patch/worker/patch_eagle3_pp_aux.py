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
"""
Patch: Propagate Eagle3 aux hidden states through PP pipeline.

In Eagle3 speculative decoding with Pipeline Parallelism (PP), auxiliary
hidden states are collected from specific target model layers (e.g., layers
2, N/2, N-3). When these layers span multiple PP stages, the last PP rank
(where the drafter runs) only sees a subset of aux states, causing
combine_hidden_states to fail with k-axis shape mismatch.

This patch wraps the inner model's forward and make_empty_intermediate_tensors
to transparently pass aux hidden states through IntermediateTensors across PP
stages. Each PP stage carries forward all aux states from previous stages,
and the last PP rank merges them into a single list for the drafter.

Currently supports:
- DeepseekV2Model (used by Kimi K2/K2.6, DeepSeek-V2/V3)
- EagleModelMixin-based models (MiniMaxM2, Llama, Qwen2, etc.)
"""

import logging
from itertools import islice

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionMetadata

logger = logging.getLogger(__name__)

_AUX_KEY_PREFIX = "aux_layer_"


def _extract_aux_from_intermediate(
    intermediate_tensors: "IntermediateTensors | None",
) -> list[torch.Tensor]:
    if intermediate_tensors is None:
        return []
    aux_keys = sorted(
        (k for k in intermediate_tensors.tensors if k.startswith(_AUX_KEY_PREFIX)),
        key=lambda k: int(k.split("_")[-1]),
    )
    return [intermediate_tensors.tensors[k] for k in aux_keys]


def _make_deepseek_v2_forward():
    def pp_eagle3_forward(
        self,
        input_ids: "torch.Tensor | None",
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata: "AttentionMetadata",
        intermediate_tensors: "IntermediateTensors | None" = None,
        inputs_embeds: "torch.Tensor | None" = None,
    ):
        pp_group = get_pp_group()

        prev_aux_list = _extract_aux_from_intermediate(intermediate_tensors)

        if pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("Either input_ids or inputs_embeds must be provided to DeepseekV2Model.forward")
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        llama_4_scaling: torch.Tensor | None = None
        if llama_4_scaling_config is not None:
            from vllm.model_executor.models.deepseek_v2 import _get_llama_4_scaling

            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config["original_max_position_embeddings"],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )

        aux_hidden_states: list[torch.Tensor] = list(prev_aux_list)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual if residual is not None else hidden_states)
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                kv_caches[idx - self.start_layer],
                attn_metadata,
                llama_4_scaling,
            )

        if not pp_group.is_last_rank:
            result = IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
            for i, t in enumerate(aux_hidden_states):
                result.tensors[f"{_AUX_KEY_PREFIX}{i}"] = t
            return result

        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    return pp_eagle3_forward


def _make_eagle_mixin_forward():
    def pp_eagle3_forward(
        self,
        input_ids: "torch.Tensor | None",
        positions: torch.Tensor,
        intermediate_tensors: "IntermediateTensors | None" = None,
        inputs_embeds: "torch.Tensor | None" = None,
    ):
        pp_group = get_pp_group()

        prev_aux_list = _extract_aux_from_intermediate(intermediate_tensors)

        if pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state(list(prev_aux_list), 0, hidden_states, residual)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(aux_hidden_states, idx + 1, hidden_states, residual)

        if not pp_group.is_last_rank:
            result = IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
            for i, t in enumerate(aux_hidden_states):
                result.tensors[f"{_AUX_KEY_PREFIX}{i}"] = t
            return result

        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    return pp_eagle3_forward


def _patch_make_empty_intermediate_tensors(inner_model: nn.Module) -> None:
    if getattr(inner_model, "_eagle3_pp_aux_make_empty_patched", False):
        return

    original_make_empty = inner_model.make_empty_intermediate_tensors

    def pp_make_empty_intermediate_tensors(batch_size, dtype, device):
        result = original_make_empty(batch_size, dtype, device)
        aux_layers = getattr(inner_model, "aux_hidden_state_layers", ())
        # A non-first PP rank only receives aux hidden states produced by
        # earlier pipeline stages. Local aux states are appended during forward.
        num_incoming_aux_layers = sum(layer_idx < inner_model.start_layer for layer_idx in aux_layers)
        hidden_size = inner_model.config.hidden_size
        for i in range(num_incoming_aux_layers):
            result.tensors[f"{_AUX_KEY_PREFIX}{i}"] = torch.zeros(
                (batch_size, hidden_size),
                dtype=dtype,
                device=device,
            )
        return result

    inner_model.make_empty_intermediate_tensors = pp_make_empty_intermediate_tensors
    inner_model._eagle3_pp_aux_make_empty_patched = True


def patch_eagle3_pp_aux_propagation(inner_model: nn.Module) -> bool:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
    from vllm.model_executor.models.interfaces import EagleModelMixin

    if isinstance(inner_model, DeepseekV2Model):
        make_forward = _make_deepseek_v2_forward
    elif isinstance(inner_model, EagleModelMixin):
        make_forward = _make_eagle_mixin_forward
    else:
        logger.warning(
            "Eagle3 PP aux propagation is only supported for DeepseekV2Model "
            "or EagleModelMixin-based models, got %s. Skipping patch.",
            type(inner_model).__name__,
        )
        return False

    if not getattr(inner_model, "_eagle3_pp_aux_forward_patched", False):
        inner_model.forward = make_forward().__get__(inner_model, type(inner_model))
        inner_model._eagle3_pp_aux_forward_patched = True
    _patch_make_empty_intermediate_tensors(inner_model)

    logger.info(
        "Applied Eagle3 PP aux propagation patch to %s (aux_layers=%s, start_layer=%d, end_layer=%d).",
        type(inner_model).__name__,
        inner_model.aux_hidden_state_layers,
        inner_model.start_layer,
        inner_model.end_layer,
    )
    return True
