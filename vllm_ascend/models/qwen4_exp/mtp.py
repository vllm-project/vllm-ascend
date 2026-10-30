# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend wrapper for Qwen3.8-Flash-Next MTP."""

from typing import Any

import regex as re
import torch
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.models.qwen4_exp.amd import mtp as upstream_mtp
from vllm.sequence import IntermediateTensors

# Import the target wrapper first so the upstream decoder resolves Ascend QSA
# and HyperConnection components before the MTP module is initialized.
from . import model as _model  # noqa: F401

_MTP_LAYER_PREFIX = re.compile(r"^mtp\.layers\.(\d+)(?=\.)")


def _remap_modelslim_mtp_layer_ids(
    quant_config: Any,
    mtp_start_layer_idx: int,
    num_mtp_layers: int,
) -> None:
    """Map checkpoint-local Qwen MTP layer ids to runtime layer ids.

    ModelSlim descriptions number the standalone draft layers from zero, while
    Qwen4Exp constructs them after the target layers. Keep this Qwen-specific
    namespace adaptation next to the model and match the same ``mtp.layers``
    strings used by the upstream weight loader.
    """
    quant_description = getattr(quant_config, "quant_description", None)
    if not isinstance(quant_description, dict):
        return

    def remap_name(name: str) -> str:
        match = _MTP_LAYER_PREFIX.match(name)
        if match is None:
            return name
        local_layer_idx = int(match.group(1))
        if local_layer_idx >= num_mtp_layers:
            return name
        runtime_layer_idx = mtp_start_layer_idx + local_layer_idx
        return _MTP_LAYER_PREFIX.sub(
            f"mtp.layers.{runtime_layer_idx}",
            name,
            count=1,
        )

    quant_config.quant_description = {remap_name(name): description for name, description in quant_description.items()}


_upstream_make_draft_vllm_config = upstream_mtp._make_draft_vllm_config


def _make_draft_vllm_config(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
) -> VllmConfig:
    draft_vllm_config = _upstream_make_draft_vllm_config(
        vllm_config,
        mtp_start_layer_idx,
    )
    num_mtp_layers = getattr(
        vllm_config.model_config.hf_text_config,
        "mtp_num_hidden_layers",
        1,
    )
    _remap_modelslim_mtp_layer_ids(
        draft_vllm_config.quant_config,
        mtp_start_layer_idx,
        num_mtp_layers,
    )
    return draft_vllm_config


upstream_mtp._make_draft_vllm_config = _make_draft_vllm_config


class AscendQwen4ExpMultiTokenPredictor(upstream_mtp.Qwen4ExpMultiTokenPredictor):
    """Qwen4Exp predictor adapted to Ascend's local PP drafter layout."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        hc_count = self.hc_count
        hidden_size = self.hidden_size
        # The draft is text-only; M-RoPE rows are identical for text tokens.
        if positions.ndim > 1:
            positions = positions[0]

        # Ascend PP loads the local MTP drafter on the last target stage. It is
        # therefore the first logical draft stage even though the global PP
        # coordinator reports that it is not the first target stage.
        if get_pp_group().is_first_rank or intermediate_tensors is None:
            assert hidden_states is not None
            if inputs_embeds is None:
                assert input_ids is not None
                inputs_embeds = self.embed_input_ids(input_ids)
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            inputs_embeds = self.fc_embedding(inputs_embeds)

            num_tokens = hidden_states.shape[0]
            hidden_states = hidden_states.view(num_tokens, hc_count, hidden_size)
            hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view(num_tokens, hc_count, hidden_size)
            hidden_states = self.fc_hidden(hidden_states.reshape(-1, hidden_size)).view(
                num_tokens, hc_count, hidden_size
            )
            hidden_states = inputs_embeds.unsqueeze(-2) + hidden_states
            hidden_states = hidden_states.flatten(-2)
        else:
            hidden_states = intermediate_tensors["hidden_states"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer = self.layers[current_step_idx]
        hidden_states, block_output, injection = layer(
            hidden_states=hidden_states,
            prev_block_output=None,
            prev_injection=None,
            positions=positions,
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
        )
        if not get_pp_group().is_last_rank:
            hidden_states = layer.mlp_hyper_connection.combine(hidden_states, block_output, injection)
            return IntermediateTensors({"hidden_states": hidden_states})

        multi_hidden, sample_hidden_states, _ = self.hyper_connection_mixer.combine_and_mix(
            hidden_states, block_output, injection
        )
        return sample_hidden_states, multi_hidden


# Qwen4ExpMTP resolves this module global when constructing its predictor.
upstream_mtp.Qwen4ExpMultiTokenPredictor = AscendQwen4ExpMultiTokenPredictor


class AscendQwen4ExpMTP(upstream_mtp.Qwen4ExpMTP):
    """Qwen4Exp MTP model bound to the Ascend-patched decoder."""

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del multimodal_embeddings, is_multimodal
        return self.model.embed_input_ids(input_ids)


__all__ = ["AscendQwen4ExpMTP", "AscendQwen4ExpMultiTokenPredictor"]
