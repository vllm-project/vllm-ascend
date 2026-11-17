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

"""Ascend wrappers for the upstream Qwen4Exp model implementation."""

from itertools import islice

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder
from vllm.models.qwen4_exp.amd import (
    hyperconnection as upstream_hc,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.qwen3_5_vl import (
    AscendQwen3_5VLProcessor,
    _AscendProcessorMixin,
    _AscendVLPreprocessMixin,
)
from vllm_ascend.ops.triton.qwen4_exp.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)

from . import ple_embedding  # noqa: F401 - register the OOT host embedding before model construction
from .ple import patch_upstream_ple_short_conv

# Keep the upstream model and weight loader authoritative. HyperConnection's
# CUDA glue functions are module globals, so replace them before importing the
# model classes that instantiate GatedResidual.
upstream_hc.grouped_gemma_rmsnorm = grouped_gemma_rmsnorm
upstream_hc.hc_combine = hc_combine
upstream_hc.hc_combine_norm = hc_combine_norm
upstream_hc.hc_gate_mix = hc_gate_mix
upstream_hc.hc_silu = hc_silu

# PLE's depthwise F.conv1d otherwise lowers to ACLop Conv2D when torch-npu
# internal formats are enabled. Scope the graph-safe option override to the
# upstream PLE custom op instead of changing it globally for the worker.
patch_upstream_ple_short_conv()

from vllm.models.qwen4_exp.amd import model as upstream_model  # noqa: E402

from .qsa import AscendQwen4ExpQSAAttention  # noqa: E402

upstream_model.Qwen4ExpQSAAttention = AscendQwen4ExpQSAAttention


class AscendQwen4ExpModel(upstream_model.Qwen4ExpModel, EagleModelMixin):
    """Qwen4Exp backbone with Ascend graph and DFlash2 auxiliary outputs."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # With max_num_seqs=1 the PLE query-start buffer has a static length of
        # two. Marking that length dynamic makes TorchDynamo reject the static
        # specialization used by PLE during FULL_DECODE_ONLY graph capture.
        dynamic_arg_dims = getattr(self, "_dynamic_arg_dims", None)
        if dynamic_arg_dims is not None:
            self._dynamic_arg_dims = {
                name: dims for name, dims in dynamic_arg_dims.items() if name != "query_start_loc"
            }

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            if self.use_hc_sequence_parallel:
                hidden_states = upstream_model.sp_shard(hidden_states)
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        full_num_tokens = positions.shape[-1]
        block_output = None
        injection = None
        last_layer = None
        aux_hidden_states: list[torch.Tensor] = []
        for layer_idx, layer in islice(enumerate(self.layers), self.start_layer, self.end_layer):
            last_layer = layer
            hidden_states, block_output, injection = layer(
                hidden_states=hidden_states,
                prev_block_output=block_output,
                prev_injection=injection,
                positions=positions,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
            )
            if layer_idx + 1 in self.aux_hidden_state_layers:
                # This DFlash2 checkpoint consumes the mean of the completed
                # layer's HC streams. Keep the delayed-combine backbone state
                # untouched so the ordinary target output follows the same path.
                combined = layer.mlp_hyper_connection.combine(hidden_states, block_output, injection)
                aux_hidden_states.append(combined.view(-1, self.config.hc_count, self.config.hidden_size).mean(dim=-2))
            if deepstack_input_embeds is not None and layer_idx < len(deepstack_input_embeds):
                deepstack_embed = deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                if self.use_hc_sequence_parallel:
                    deepstack_embed = upstream_model.sp_shard(deepstack_embed)
                deepstack_embed = (
                    deepstack_embed.unsqueeze(-2)
                    .expand(
                        *deepstack_embed.shape[:-1],
                        self.config.hc_count,
                        self.config.hidden_size,
                    )
                    .flatten(-2)
                )
                hidden_states = layer.mlp_hyper_connection.combine(hidden_states, block_output, injection)
                block_output = None
                injection = None
                hidden_states = hidden_states + deepstack_embed

        if not get_pp_group().is_last_rank:
            if last_layer is not None and block_output is not None:
                hidden_states = last_layer.mlp_hyper_connection.combine(hidden_states, block_output, injection)
            return IntermediateTensors({"hidden_states": hidden_states})

        final_mixer = self.hyper_connection_mixer
        assert final_mixer is not None
        multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(hidden_states, block_output, injection)
        if self.use_hc_sequence_parallel:
            if aux_hidden_states:
                hidden_size = self.config.hidden_size
                packed_hidden_states = torch.cat([sample_hidden_states, *aux_hidden_states], dim=-1)
                packed_hidden_states = upstream_model.sp_all_gather(packed_hidden_states)[:full_num_tokens]
                split_sizes = [hidden_size] * (1 + len(aux_hidden_states))
                sample_hidden_states, *aux_hidden_states = packed_hidden_states.split(split_sizes, dim=-1)
                sample_hidden_states = sample_hidden_states.contiguous()
            elif self._mtp_hidden_buffer is not None:
                hidden_size = self.config.hidden_size
                packed_hidden_states = torch.cat([sample_hidden_states, multi_hidden], dim=-1)
                packed_hidden_states = upstream_model.sp_all_gather(packed_hidden_states)[:full_num_tokens]
                sample_hidden_states, multi_hidden = packed_hidden_states.split(
                    [hidden_size, self.config.hc_count * hidden_size], dim=-1
                )
                sample_hidden_states = sample_hidden_states.contiguous()
            else:
                sample_hidden_states = upstream_model.sp_all_gather(sample_hidden_states)[:full_num_tokens]
        if self._mtp_hidden_buffer is not None:
            num_tokens = multi_hidden.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        if aux_hidden_states:
            return sample_hidden_states, aux_hidden_states
        return sample_hidden_states


# Qwen4ExpForCausalLM resolves this module global when it creates the backbone.
upstream_model.Qwen4ExpModel = AscendQwen4ExpModel


class AscendQwen4ExpForCausalLM(upstream_model.Qwen4ExpForCausalLM, SupportsEagle3):
    """Qwen3.8-Flash-Next text model using Ascend platform operators."""

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config: VllmConfig):
        # vLLM uses these specs to align attention pages. PLE has a private
        # dense bank; only GDN's complete conv+SSM state shares QSA pages.
        return tuple(spec for spec in super().get_mamba_specs_from_config(vllm_config) if len(spec.shapes) == 2)

    @staticmethod
    def get_model_state_cls():
        from .model_state import AscendQwen4ExpModelState

        return AscendQwen4ExpModelState


class AscendQwen4ExpProcessingInfo(_AscendProcessorMixin, upstream_model.Qwen4ExpProcessingInfo):
    """Preserve Qwen4Exp config handling while emitting raw image payloads."""

    pass


@MULTIMODAL_REGISTRY.register_processor(
    AscendQwen3_5VLProcessor,
    info=AscendQwen4ExpProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendQwen4ExpForConditionalGeneration(_AscendVLPreprocessMixin, upstream_model.Qwen4ExpForConditionalGeneration):
    """Qwen3.8-Flash-Next multimodal model using Ascend operators."""

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config: VllmConfig):
        return AscendQwen4ExpForCausalLM.get_mamba_specs_from_config(vllm_config)

    @staticmethod
    def get_model_state_cls():
        from .model_state import AscendQwen4ExpModelState

        return AscendQwen4ExpModelState


__all__ = [
    "AscendQwen4ExpForCausalLM",
    "AscendQwen4ExpForConditionalGeneration",
    "AscendQwen4ExpModel",
]
