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

from collections.abc import Callable
from itertools import islice
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder
from vllm.models.qwen4_exp.amd import hyperconnection as upstream_hc
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)
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
from .hc_fused import AscendTextGatedResidual, prepare_model_textgr_weights
from .ple import AscendPCPPLELayer, patch_upstream_ple_short_conv
from .topk_buffer import share_target_topk_buffers

if TYPE_CHECKING:
    from torch import Tensor

# Keep the upstream model and weight loader authoritative. HyperConnection's
# CUDA glue functions are module globals, so replace them before importing the
# model classes that instantiate GatedResidual.
upstream_hc.grouped_gemma_rmsnorm = grouped_gemma_rmsnorm
upstream_hc.hc_combine = hc_combine
upstream_hc.hc_combine_norm = hc_combine_norm
upstream_hc.hc_gate_mix = hc_gate_mix
upstream_hc.hc_silu = hc_silu
upstream_hc.GatedResidual = AscendTextGatedResidual

# PLE's depthwise F.conv1d otherwise lowers to ACLop Conv2D when torch-npu
# internal formats are enabled. Scope the graph-safe option override to the
# upstream PLE custom op instead of changing it globally for the worker.
patch_upstream_ple_short_conv()

from vllm.models.qwen4_exp.amd import model as upstream_model  # noqa: E402
from vllm.models.qwen4_exp.amd import ple_layer as upstream_ple  # noqa: E402
from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLELayer  # noqa: E402

from .qsa import AscendQwen4ExpQSAAttention  # noqa: E402

# vLLM owns Qwen4Exp PP, HC and SP control flow. Bind only the device-specific
# collectives and operators used by that upstream implementation.
upstream_model.sp_all_gather = sp_all_gather
upstream_model.sp_reduce_scatter = sp_reduce_scatter
upstream_model.sp_shard = sp_shard
upstream_ple.sp_all_gather = sp_all_gather
upstream_ple.sp_reduce_scatter = sp_reduce_scatter
upstream_ple.sp_shard = sp_shard
upstream_model.Qwen4ExpQSAAttention = AscendQwen4ExpQSAAttention
upstream_model.GatedResidual = AscendTextGatedResidual


def _build_ascend_ple(config, vllm_config: VllmConfig, **kwargs) -> Qwen4ExpPLELayer:
    # Choose the backend before PLE allocates its host table and registers
    # the layer in the static forward context.
    if vllm_config.parallel_config.prefill_context_parallel_size > 1:
        return AscendPCPPLELayer(config, vllm_config=vllm_config, **kwargs)
    return Qwen4ExpPLELayer(config, vllm_config=vllm_config, **kwargs)


# The upstream decoder resolves this module global when it creates each PLE.
upstream_model.Qwen4ExpPLELayer = _build_ascend_ple


class AscendQwen4ExpModel(upstream_model.Qwen4ExpModel, EagleModelMixin):
    """Qwen4Exp backbone with Ascend graph and DFlash2 auxiliary outputs."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        parallel = vllm_config.parallel_config
        use_pcp = parallel.prefill_context_parallel_size > 1
        if use_pcp:
            if parallel.decode_context_parallel_size != 1:
                raise NotImplementedError("Qwen4Exp PCP requires DCP=1")

            from vllm_ascend.ops.head_parallel import get_head_parallel_size

            text_config = vllm_config.model_config.hf_text_config
            head_shards = get_head_parallel_size(parallel)
            if text_config.linear_num_key_heads % head_shards or text_config.linear_num_value_heads % head_shards:
                raise ValueError("Qwen4Exp GDN key and value heads must be divisible by TP * PCP")
        # PCP has separate cache and graph paths. Keep its scratch buffers private.
        self._share_qsa_topk_buffers = not use_pcp and not parallel.use_ubatching
        with share_target_topk_buffers(self._share_qsa_topk_buffers):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        if use_pcp:
            from vllm_ascend.attention.gdn import shard_gdn_heads_over_pcp

            for layer in self.layers[self.start_layer : self.end_layer]:
                if hasattr(layer, "linear_attn"):
                    shard_gdn_heads_over_pcp(layer.linear_attn, parallel)
        # With max_num_seqs=1 the PLE query-start buffer has a static length of
        # two. Marking that length dynamic makes TorchDynamo reject the static
        # specialization used by PLE during FULL_DECODE_ONLY graph capture.
        dynamic_arg_dims = getattr(self, "_dynamic_arg_dims", None)
        if dynamic_arg_dims is not None:
            self._dynamic_arg_dims = {
                name: dims for name, dims in dynamic_arg_dims.items() if name != "query_start_loc"
            }

    def process_weights_after_loading(self) -> None:
        parent_hook = getattr(super(), "process_weights_after_loading", None)
        if parent_hook is not None:
            parent_hook()
        prepare_model_textgr_weights(self)

    def _apply(self, fn: "Callable[[Tensor], Tensor]", recurse: bool = True) -> Self:
        if not recurse or not self._share_qsa_topk_buffers:
            return super()._apply(fn, recurse=recurse)

        # Module._apply normally converts each registered buffer independently.
        # Reuse conversions of the exact Top-K tensors within this backbone.
        topk_buffers = {}
        for module in self.modules():
            if isinstance(module, AscendQwen4ExpQSAAttention):
                buffer = module.topk_indices_buffer
                # Keep input tensors alive so their IDs cannot be reused during conversion.
                topk_buffers[id(buffer)] = buffer
        if not topk_buffers:
            return super()._apply(fn, recurse=recurse)

        converted: dict[int, Tensor] = {}

        def convert(tensor: "Tensor") -> "Tensor":
            key = id(tensor)
            if key not in topk_buffers:
                return fn(tensor)
            if key not in converted:
                converted[key] = fn(tensor)
            return converted[key]

        return super()._apply(convert, recurse=recurse)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        """Reuse the upstream forward except for DFlash2 auxiliary capture."""
        pp_group = get_pp_group()
        if not pp_group.is_first_rank and self.use_hc_sequence_parallel and intermediate_tensors is not None:
            hidden_states = intermediate_tensors["hidden_states"]
            if hidden_states.shape[0] == positions.shape[-1]:
                intermediate_tensors = IntermediateTensors(
                    {**intermediate_tensors.tensors, "hidden_states": sp_shard(hidden_states)}
                )

        if self.aux_hidden_state_layers:
            # Upstream does not expose completed decoder-layer HC states. Only
            # DFlash2 uses the local loop that captures and returns them.
            output = self._forward_with_dflash2_aux(
                input_ids,
                positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        else:
            output = super().forward(
                input_ids,
                positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        if not pp_group.is_last_rank and self.use_hc_sequence_parallel:
            assert isinstance(output, IntermediateTensors)
            hidden_states = sp_all_gather(output["hidden_states"])
            output = IntermediateTensors({**output.tensors, "hidden_states": hidden_states[: positions.shape[-1]]})
        return output

    def _forward_with_dflash2_aux(
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
                # layer's HC streams. Reuse the materialized BF16 state in the
                # backbone instead of combining the same pending output again.
                combined = layer.mlp_hyper_connection.combine(hidden_states, block_output, injection)
                aux_hidden_states.append(combined.view(-1, self.config.hc_count, self.config.hidden_size).mean(dim=-2))
                hidden_states = combined
                block_output = None
                injection = None
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
                if block_output is not None:
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
        if block_output is None:
            multi_hidden, sample_hidden_states, _ = final_mixer.mix(hidden_states)
        else:
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

    pp_intermediate_tensors_are_full_tokens = True

    def process_weights_after_loading(self) -> None:
        parent_hook = getattr(super(), "process_weights_after_loading", None)
        if parent_hook is not None:
            parent_hook()
        prepare_model_textgr_weights(self)

    @classmethod
    def get_gdn_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        upstream_shape = super().get_gdn_mamba_state_shape_from_config
        if vllm_config.parallel_config.prefill_context_parallel_size == 1:
            return upstream_shape(vllm_config)
        from vllm_ascend.ops.head_parallel import get_head_sharded_state_shape

        return get_head_sharded_state_shape(vllm_config, upstream_shape)

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

    pp_intermediate_tensors_are_full_tokens = True

    def process_weights_after_loading(self) -> None:
        parent_hook = getattr(super(), "process_weights_after_loading", None)
        if parent_hook is not None:
            parent_hook()
        prepare_model_textgr_weights(self)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return AscendQwen4ExpForCausalLM.get_mamba_state_shape_from_config(vllm_config)

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
