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
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.model_executor.models.utils import extract_layer_index, maybe_prefix
from vllm.models.qwen4_exp.amd import (
    hyperconnection as upstream_hc,
)
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm_ascend.ops.triton.qwen4_exp.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)

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
from vllm.models.qwen4_exp.amd import ple_layer as upstream_ple  # noqa: E402

from .qsa import AscendQwen4ExpQSAAttention  # noqa: E402

upstream_model.Qwen4ExpQSAAttention = AscendQwen4ExpQSAAttention


class AscendQwen4ExpPLELayer(upstream_model.Qwen4ExpPLELayer):
    """Keep token-local PLE work sharded around the full-token short convolution."""

    def __init__(
        self,
        *args,
        use_sequence_parallel: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_sequence_parallel = use_sequence_parallel

    def _sequence_parallel_embeddings(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        ple_embedding = self.ple_embedding
        ngram_ids = input_ids.new_empty(
            (input_ids.shape[0], ple_embedding.ngram_heads),
            dtype=torch.int64,
        )
        torch.ops.vllm.qwen4_exp_amd_ple_ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context,
            ngram_ids,
            ple_embedding.layer_name,
        )
        local_ngram_ids = sp_shard(ngram_ids)
        embedding = ple_embedding.ngram_embedding
        output_dtype = (
            embedding.weight.dtype
            if embedding.supports_prefetch
            else embedding.params_dtype
        )
        output = local_ngram_ids.new_empty(
            (local_ngram_ids.shape[0], ple_embedding.embedding_dim),
            dtype=output_dtype,
        )
        if embedding.supports_prefetch:
            torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding_pinned(
                local_ngram_ids,
                output,
                ple_embedding.layer_name,
            )
        else:
            torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding(
                local_ngram_ids,
                output,
                ple_embedding.layer_name,
            )
        return output

    def _use_local_ngram_lookup(self) -> bool:
        if not is_forward_context_available():
            return True
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, list):
            attn_metadata = attn_metadata[0]
        if not isinstance(attn_metadata, dict):
            return True
        ple_metadata = attn_metadata.get(self.prefix)
        if not isinstance(
            ple_metadata,
            upstream_ple.PleShortConvAttentionMetadata,
        ):
            return True
        return ple_metadata.num_prefills > 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_sequence_parallel:
            return super().forward(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )

        input_ids = input_ids.reshape(-1)
        if self._use_local_ngram_lookup():
            embeddings = self._sequence_parallel_embeddings(
                input_ids,
                query_start_loc,
                ngram_context,
            )
        else:
            embeddings = self.ple_embedding(
                input_ids,
                query_start_loc,
                ngram_context,
            )
            embeddings = sp_shard(embeddings)
        embeddings = self.ple_embedding.ngram_embedding.dequantize(
            embeddings, hidden_states.dtype
        )
        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
        gated_output, conv_input = upstream_ple.ple_gate(
            key,
            value,
            hidden_states,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.norm_key.eps,
        )

        # PLE short-conv and the immediately following attention both need a
        # full-token view. Gather the token-local residual branch and conv
        # input together so the decoder can reuse the same full-token result
        # instead of launching a second all-gather before attention.
        gated_output.add_(hidden_states)
        gated_output_size = gated_output.shape[-1]
        gathered_ple = sp_all_gather(
            torch.cat((gated_output, conv_input), dim=-1)
        )[: input_ids.shape[0]]
        gated_output, conv_input = gathered_ple.split(
            (gated_output_size, conv_input.shape[-1]), dim=-1
        )
        conv_output = torch.zeros_like(conv_input)
        torch.ops.vllm.qwen4_exp_ple_short_conv(
            conv_input,
            conv_output,
            torch.zeros_like(conv_input),
            self.prefix,
        )
        gated_output.add_(conv_output)
        return gated_output


class AscendQwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """Qwen4Exp MoE with Ascend sequence parallel support."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)


class AscendQwen4ExpDecoderLayer(upstream_model.Qwen4ExpDecoderLayer):
    """Qwen4Exp decoder layer whose MoE owns the local SP region."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.use_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.ple = None
        ple_layer_ids = config.ple_layer_ids
        if (self.layer_idx + 1) in ple_layer_ids:
            ple_layer_ids_sorted = sorted(set(ple_layer_ids))
            ple_dense_layer_id_map = {
                abs_id: idx for idx, abs_id in enumerate(ple_layer_ids_sorted)
            }
            ple_dense_layer_id = ple_dense_layer_id_map[self.layer_idx + 1]
            self.ple = AscendQwen4ExpPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=ple_dense_layer_id,
                prefix=f"{prefix}.ple",
                use_sequence_parallel=self.use_sequence_parallel,
            )

        if layer_type == "linear_attention":
            self.linear_attn = upstream_model.QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=not self.use_sequence_parallel,
            )
        elif layer_type in upstream_model._QSA_LAYER_TYPES:
            use_qsa = getattr(config, "indexer_n_heads", None) is not None
            if not use_qsa:
                self.self_attn = upstream_model.Qwen3NextAttention(
                    config,
                    model_config=vllm_config.model_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=quant_config,
                    reduce_results=not self.use_sequence_parallel,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                self.self_attn = upstream_model.Qwen4ExpQSAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=quant_config,
                    reduce_results=not self.use_sequence_parallel,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        self.mlp = AscendQwen4ExpSparseMoeBlock(
            vllm_config=vllm_config, prefix=f"{prefix}.mlp"
        )

        hc_config = upstream_model.HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = upstream_model.GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "attn_hyper_connection"),
        )
        self.mlp_hyper_connection = upstream_model.GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor | None,
        prev_injection: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_hc = self.attn_hyper_connection
        ple_output_is_full_tokens = False
        if self.ple is not None:
            if prev_block_output is not None and prev_injection is not None:
                hidden_states = attn_hc.combine(
                    hidden_states, prev_block_output, prev_injection
                )
                prev_block_output = prev_injection = None

            if input_ids is None or query_start_loc is None or ngram_context is None:
                raise RuntimeError("PLE inputs were not prepared")
            hidden_states = self.ple(
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
            )
            ple_output_is_full_tokens = self.use_sequence_parallel
        if prev_block_output is not None and prev_injection is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(
                hidden_states, prev_block_output, prev_injection
            )
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)


        if self.use_sequence_parallel:
            if ple_output_is_full_tokens:
                # HC mixing is token-local. Keep its attention input full and
                # restore only the delayed residual state to the local shard.
                hidden_states = sp_shard(hidden_states)
                injection = sp_shard(injection)
            else:
                block_input = sp_all_gather(block_input)[: positions.shape[-1]]

        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states=block_input)
        elif self.layer_type in upstream_model._QSA_LAYER_TYPES:
            attn_out = self.self_attn(
                hidden_states=block_input,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")

        if self.use_sequence_parallel:
            attn_out = sp_reduce_scatter(attn_out)


        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(
            hidden_states, attn_out, injection
        )
        mlp_out = self.mlp(
            block_input,
            already_sequence_parallel=self.use_sequence_parallel,
        )
        if self.use_sequence_parallel and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None and is_padding.shape[0] == mlp_out.shape[0]:
                mlp_out = torch.where(
                    is_padding.unsqueeze(-1),
                    torch.zeros((), dtype=mlp_out.dtype, device=mlp_out.device),
                    mlp_out,
                )
        return hidden_states, mlp_out, injection


# Keep the overrides local to Ascend model registration.
# HC and token-local PLE state remain sequence-parallel across decoder
# layers; Attention temporarily gathers full-token input and restores
# the local shard with ReduceScatter, while MoE consumes the existing shard.
upstream_model.Qwen4ExpSparseMoeBlock = AscendQwen4ExpSparseMoeBlock
upstream_model.Qwen4ExpDecoderLayer = AscendQwen4ExpDecoderLayer


class AscendQwen4ExpModel(upstream_model.Qwen4ExpModel):
    """Qwen4Exp backbone with Ascend-safe graph shape annotations."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.use_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
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
    ) -> torch.Tensor | IntermediateTensors:
        pp_group = get_pp_group()
        if pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        if (
            self.use_sequence_parallel
            and hidden_states.shape[0] == positions.shape[-1]
        ):
            if is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding,
                    hidden_states,
                )
            hidden_states = sp_shard(hidden_states)
        elif self.use_sequence_parallel and is_forward_context_available():
            # A non-first PP stage receives an already sharded intermediate,
            # while its ForwardContext still carries the full-token padding
            # mask prepared by the runner. Shard that mask to the same TP row
            # range so MoE never processes SP padding as real tokens.
            forward_context = get_forward_context()
            is_padding = forward_context.is_padding
            if is_padding is not None and is_padding.shape[0] != hidden_states.shape[0]:
                mask_input = hidden_states.new_empty(
                    (is_padding.shape[0], 1), dtype=hidden_states.dtype
                )
                forward_context.is_padding = sp_padding_mask(
                    is_padding,
                    mask_input,
                )
        if pp_group.is_first_rank:
            hidden_states = hidden_states.repeat(1, self.config.hc_count)

        block_output = None
        injection = None
        last_layer = None
        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
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
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                deepstack_embed = (
                    deepstack_embed.unsqueeze(-2)
                    .expand(
                        *deepstack_embed.shape[:-1],
                        self.config.hc_count,
                        self.config.hidden_size,
                    )
                    .flatten(-2)
                )
                hidden_states = layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
                block_output = None
                injection = None
                if self.use_sequence_parallel:
                    deepstack_embed = sp_shard(deepstack_embed)
                hidden_states = hidden_states + deepstack_embed

        if not pp_group.is_last_rank:
            if last_layer is not None and block_output is not None:
                hidden_states = last_layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
            if self.use_sequence_parallel:
                hidden_states = sp_all_gather(hidden_states)[
                    : positions.shape[-1]
                ]
            return IntermediateTensors({"hidden_states": hidden_states})

        final_mixer = self.hyper_connection_mixer
        assert final_mixer is not None
        multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(
            hidden_states, block_output, injection
        )
        if self.use_sequence_parallel:
            full_num_tokens = positions.shape[-1]
            multi_hidden_size = multi_hidden.shape[-1]
            gathered_hidden = sp_all_gather(
                torch.cat((multi_hidden, sample_hidden_states), dim=-1)
            )[:full_num_tokens]
            multi_hidden, sample_hidden_states = gathered_hidden.split(
                (multi_hidden_size, self.config.hidden_size), dim=-1
            )
        if self._mtp_hidden_buffer is not None:
            num_tokens = multi_hidden.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        return sample_hidden_states


# Qwen4ExpForCausalLM resolves this module global when it creates the backbone.
upstream_model.Qwen4ExpModel = AscendQwen4ExpModel


class AscendQwen4ExpForCausalLM(upstream_model.Qwen4ExpForCausalLM):
    """Qwen3.8-Flash-Next text model using Ascend platform operators."""

    # Decoder stages gather their SP shard before crossing a PP boundary.
    # The runner must therefore transport full-token intermediate tensors.
    pp_intermediate_tensors_are_full_tokens = True

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config: VllmConfig):
        # vLLM uses these specs to align attention pages. PLE has a private
        # dense bank; only GDN's complete conv+SSM state shares QSA pages.
        return tuple(spec for spec in super().get_mamba_specs_from_config(vllm_config) if len(spec.shapes) == 2)

    @staticmethod
    def get_model_state_cls():
        from .model_state import AscendQwen4ExpModelState

        return AscendQwen4ExpModelState


class AscendQwen4ExpForConditionalGeneration(upstream_model.Qwen4ExpForConditionalGeneration):
    """Qwen3.8-Flash-Next multimodal model using Ascend operators."""

    pp_intermediate_tensors_are_full_tokens = True

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
    "AscendQwen4ExpDecoderLayer",
    "AscendQwen4ExpModel",
    "AscendQwen4ExpPLELayer",
    "AscendQwen4ExpSparseMoeBlock",
]
