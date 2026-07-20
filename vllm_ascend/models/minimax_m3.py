# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The MiniMax AI team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only MiniMaxM3 model."""

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
)
from vllm.logger import logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_ascend.attention.msa_m3 import MiniMaxM3SparseAttention


def _sparse_attention_layer_ids(config: PretrainedConfig) -> set[int]:
    cfg = getattr(config, "sparse_attention_config", None)
    if not cfg:
        return set()
    freq = cfg.get("sparse_attention_freq")
    if freq is None:
        return set()
    return {i for i, f in enumerate(freq) if f != 0}


def _get_text_config(vllm_config: VllmConfig) -> PretrainedConfig:
    return vllm_config.model_config.hf_text_config


def _get_max_position_embeddings(config: PretrainedConfig) -> int:
    max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
    max_model_len = getattr(config, "max_model_len", None)
    if isinstance(max_model_len, int):
        max_position_embeddings = max(max_position_embeddings, max_model_len)
    return max_position_embeddings


def _get_rope_parameters(config: PretrainedConfig) -> dict[str, Any] | None:
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None:
        rope_parameters = dict(rope_parameters)
    else:
        rope_parameters = {
            "rope_theta": getattr(config, "rope_theta", 10000),
            "partial_rotary_factor": getattr(config, "partial_rotary_factor", 1.0),
        }
    return rope_parameters


class MiniMaxM3SwiGLUOAI(nn.Module):
    """MiniMax-M3 SwiGLU-OAI activation for packed gate/up outputs."""

    def __init__(self, alpha: float, beta: float, limit: float):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.limit = float(limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.npu.npu_clipped_swiglu(
            x,
            dim=-1,
            alpha=self.alpha,
            limit=self.limit,
            bias=self.beta,
            interleaved=False,
        )


class MiniMaxM3MLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        intermediate_size: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        hidden_act = config.hidden_act
        if intermediate_size is None:
            intermediate_size = config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act == "swigluoai":
            self.act_fn = MiniMaxM3SwiGLUOAI(
                alpha=config.swiglu_alpha,
                beta=getattr(config, "swiglu_beta", 1.0),
                limit=config.swiglu_limit,
            )
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}. Only swigluoai is supported.")

    def forward(
        self,
        x,
    ):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniMaxM3MoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_shared_experts = getattr(config, "n_shared_experts", None)

        if self.tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than the "
                f"number of experts {config.num_local_experts}."
            )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(torch.empty(config.num_local_experts, dtype=torch.float32))
            self.e_score_correction_bias.weight_loader = MiniMaxM3MoE.ebias_weight_loader
        else:
            self.e_score_correction_bias = None

        if self.n_shared_experts:
            intermediate_size = config.intermediate_size * self.n_shared_experts
            self.shared_experts = MiniMaxM3MLP(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
                intermediate_size=intermediate_size,
            )
        else:
            self.shared_experts = None

        self.gate = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            renormalize=True,
            activation="swigluoai_uninterleave",
            swiglu_limit=config.swiglu_limit,
            swiglu_alpha=config.swiglu_alpha,
            swiglu_beta=getattr(config, "swiglu_beta", 1.0),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            router_logits_dtype=self.gate.out_dtype,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=True,
        )

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)

        return final_hidden_states.view(num_tokens, hidden_dim)


class MiniMaxM3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_dim: int,
        rope_parameters: dict[str, Any] | None = None,
        attn_window_size: int | None = None,
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if rope_parameters is not None and "partial_rotary_factor" not in rope_parameters:
            rope_parameters["partial_rotary_factor"] = rotary_dim / self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=rms_norm_eps)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            per_layer_sliding_window=attn_window_size,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def _qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_shape = q.shape
        k_shape = k.shape
        q = q.reshape(-1, self.head_dim).contiguous()
        k = k.reshape(-1, self.head_dim).contiguous()
        q = self.q_norm(q).reshape(q_shape)
        k = self.k_norm(k).reshape(k_shape)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        v = v.contiguous()

        q, k = self._qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class MiniMaxM3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = _get_max_position_embeddings(config)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])

        self.layer_idx = layer_idx

        sparse_attention_config = getattr(config, "sparse_attention_config", None)

        if sparse_attention_config is not None:
            is_sparse_attention_layer = layer_idx in _sparse_attention_layer_ids(config)
            disable_index_value = sparse_attention_config["sparse_disable_index_value"][layer_idx] == 1
        else:
            is_sparse_attention_layer = False
            disable_index_value = False

        attn_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rotary_dim=config.rotary_dim,
            rope_parameters=_get_rope_parameters(config),
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        if is_sparse_attention_layer:
            self.self_attn = MiniMaxM3SparseAttention(
                **attn_kwargs,
                sparse_cfg=sparse_attention_config,
                disable_index_value=disable_index_value,
            )
        else:
            self.self_attn = MiniMaxM3Attention(**attn_kwargs)

        moe_layer_freq = getattr(config, "moe_layer_freq", None)
        # ``is_layer_sparse`` here means "this layer's MLP is a sparse MoE",
        # not anything about attention sparsity. The name is kept (instead of
        # the clearer ``is_layer_moe``) to match the convention used by the
        # rest of sglang -- ``OperationsStrategy``, ``LayerScatterModes``,
        # ``LayerCommunicator``, ``gpt_oss``, ``falcon_h1`` etc all access
        # ``layer.is_layer_sparse``.
        self.is_layer_sparse = moe_layer_freq[layer_idx] != 0 if moe_layer_freq is not None else True

        if self.is_layer_sparse:
            self.block_sparse_moe = MiniMaxM3MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            self.mlp = MiniMaxM3MLP(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                intermediate_size=config.dense_intermediate_size,
            )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if self.is_layer_sparse:
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile
class MiniMaxM3Model(nn.Module, EagleModelMixin):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # config = vllm_config.model_config.hf_config
        config = _get_text_config(vllm_config)
        text_config = config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self._enable_eagle3_aux_hidden_states = (
            vllm_config.speculative_config is not None and vllm_config.speculative_config.method == "eagle3"
        )

        self.vocab_size = text_config.vocab_size
        self.num_hidden_layers = text_config.num_hidden_layers
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            text_config.num_hidden_layers,
            lambda prefix: MiniMaxM3DecoderLayer(
                config,
                prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(aux_hidden_states, idx + 1, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states

        return hidden_states

    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if self._enable_eagle3_aux_hidden_states:
            EagleModelMixin._set_aux_hidden_state_layers(self, layers)
        else:
            EagleModelMixin._set_aux_hidden_state_layers(self, ())

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self._set_aux_hidden_state_layers(layers)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping: list[tuple[str, str, int | str]] = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".indexer_proj", ".index_q_proj", "index_q"),
            (".indexer_proj", ".index_k_proj", "index_k"),
            (".qkv_proj", ".index_q_proj", "index_q"),
            (".qkv_proj", ".index_k_proj", "index_k"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = self.get_expert_mapping()

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        loaded_tensors = 0
        skipped_tensors = 0
        skipped_reasons: dict[str, int] = {}

        def mark_loaded(param_name: str) -> None:
            nonlocal loaded_tensors
            loaded_params.add(param_name)
            loaded_tensors += 1

        def mark_skipped(reason: str) -> None:
            nonlocal skipped_tensors
            skipped_tensors += 1
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

        for name, loaded_weight in weights:
            if name.startswith("model."):
                name = name[len("model.") :]
            if "mtp." in name:
                mark_skipped("mtp")
                continue
            if "weight_scale_inv" in name:
                name = name.replace("weight_scale_inv", "weight_scale")
            elif "scale_inv" in name:
                name = name.replace("scale_inv", "scale")
            if "rotary_emb.inv_freq" in name:
                mark_skipped("rotary_emb")
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                mark_skipped("spec_decode")
                continue  # skip spec decode layers for main model

            # Sparse layers fold index_q/index_k into fused qkv_proj (handled below).
            # Other index_* weights (e.g. index_v/index_o) load explicitly here.
            if ".index_" in name and ".index_q_proj" not in name and ".index_k_proj" not in name:
                if name.endswith(".bias") and name not in params_dict:
                    mark_skipped("missing_bias")
                    continue
                if is_pp_missing_parameter(name, self):
                    mark_skipped("pp_missing")
                    continue
                if name not in params_dict:
                    mark_skipped("missing_index_param")
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                mark_loaded(name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Routed experts (w1/w2/w3) are handled below; don't let
                # stacked dense/shared-expert mappings rewrite them.
                if ("block_sparse_moe.experts." in name) and name not in params_dict:
                    continue
                param_name_full = name.replace(weight_name, param_name)
                if param_name_full.endswith(".bias") and param_name_full not in params_dict:
                    mark_skipped("missing_bias")
                    continue
                if is_pp_missing_parameter(param_name_full, self):
                    mark_skipped("pp_missing")
                    continue
                if param_name_full.endswith((".k_scale", ".v_scale")):
                    remapped_name = maybe_remap_kv_scale_name(param_name_full, params_dict)
                    if remapped_name is not None and remapped_name in params_dict:
                        param = params_dict[remapped_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        mark_loaded(remapped_name)
                        break
                if param_name_full not in params_dict:
                    continue

                param = params_dict[param_name_full]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                mark_loaded(param_name_full)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        mark_skipped("pp_missing")
                        break
                    if name_mapped not in params_dict:
                        continue

                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        mark_loaded(name_mapped)
                        break
                else:
                    if is_expert_weight:
                        mark_skipped("nonlocal_or_unmapped_expert")
                        continue

                    if name.endswith(".bias") and name not in params_dict:
                        mark_skipped("missing_bias")
                        continue

                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        mark_skipped("kv_scale_remap_missing")
                        continue

                    if is_pp_missing_parameter(name, self):
                        mark_skipped("pp_missing")
                        continue
                    if name not in params_dict:
                        mark_skipped("missing_param")
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    if getattr(weight_loader, "supports_moe_loading", False):
                        if loaded_weight.shape == param.shape:
                            default_weight_loader(param, loaded_weight)
                            mark_loaded(name)
                            continue
                        raise ValueError(
                            f"FusedMoE parameter {name!r} reached the "
                            "fallback loader with an incompatible shape: "
                            f"checkpoint={tuple(loaded_weight.shape)}, "
                            f"parameter={tuple(param.shape)}. Add an expert "
                            "mapping for this checkpoint weight instead."
                        )
                    weight_loader(param, loaded_weight)
                    mark_loaded(name)
        logger.warning(
            "MiniMax M3 text load_weights loaded %d checkpoint tensors into "
            "%d parameter names; skipped %d tensors by reason: %s",
            loaded_tensors,
            len(loaded_params),
            skipped_tensors,
            skipped_reasons,
        )
        return loaded_params


class MiniMaxM3SparseForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "indexer_proj": ["index_q_proj", "index_k_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts": ["experts.0.w1", "experts.0.w2", "experts.0.w3"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.model.": "model.",
            "language_model.lm_head.": "lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _get_text_config(vllm_config)
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        if hasattr(vllm_config.model_config, "max_model_len"):
            self.config.max_model_len = vllm_config.model_config.max_model_len
        self.model = MiniMaxM3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        raw_tensors = 0
        text_tensors = 0
        skipped_multimodal_tensors = 0

        def text_weights() -> Iterable[tuple[str, torch.Tensor]]:
            nonlocal raw_tensors, text_tensors, skipped_multimodal_tensors
            for name, weight in weights:
                raw_tensors += 1
                if "vision_tower" in name or "multi_modal_projector" in name or "patch_merge_mlp" in name:
                    skipped_multimodal_tensors += 1
                    continue
                text_tensors += 1
                yield name, weight

        loaded_params = loader.load_weights(text_weights(), mapper=self.hf_to_vllm_mapper)
        logger.warning(
            "MiniMax M3 top-level load_weights saw %d checkpoint tensors, "
            "passed %d text tensors, skipped %d multimodal tensors, "
            "returned %d loaded parameter names",
            raw_tensors,
            text_tensors,
            skipped_multimodal_tensors,
            len(loaded_params),
        )
        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()


def get_spec_layer_idx_from_weight_name(config: PretrainedConfig, weight_name: str) -> int | None:
    if hasattr(config, "num_mtp_modules") and (config.num_mtp_modules > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_mtp_modules):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None
