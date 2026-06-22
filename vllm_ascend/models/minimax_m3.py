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

from collections.abc import Iterable, Mapping, Sequence
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import BatchFeature, PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model

from vllm_ascend.attention.msa_m3 import MiniMaxM3SparseAttention
from vllm_ascend.models.minimax_m3_vit import MiniMaxVLVisionModel


logger = init_logger(__name__)


def _sparse_attention_layer_ids(config: PretrainedConfig) -> set[int]:
    cfg = getattr(config, "sparse_attention_config", None)
    if not cfg:
        return set()
    freq = cfg.get("sparse_attention_freq")
    if freq is None:
        return set()
    return {i for i, f in enumerate(freq) if f != 0}


def is_minimax_m3_sparse_model(config: PretrainedConfig | None) -> bool:
    return bool(_sparse_attention_layer_ids(config)) if config is not None else False


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
        d = x.shape[-1] // 2
        gate = torch.clamp(x[..., :d], max=self.limit)
        up = torch.clamp(x[..., d:], min=-self.limit, max=self.limit)
        return gate * torch.sigmoid(self.alpha * gate) * (up + self.beta)

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
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only swigluoai is supported."
            )

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
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.use_routing_bias = getattr(config, "use_routing_bias", False)
        if self.use_routing_bias:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.num_local_experts, dtype=torch.float32)
            )
            self.e_score_correction_bias.weight_loader = (
                MiniMaxM3MoE.ebias_weight_loader
            )
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
            activation="swigluoai",
            swiglu_limit=config.swiglu_limit,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            router_logits_dtype=self.gate.out_dtype,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=True,
        )
        self.experts.swiglu_alpha = config.swiglu_alpha
        self.experts.swiglu_beta = getattr(config, "swiglu_beta", 1.0)
        self.experts.swigluoai_uninterleave = True

    @staticmethod
    def ebias_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight.to(torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

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

        if (
            rope_parameters is not None
            and "partial_rotary_factor" not in rope_parameters
        ):
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

    def _qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            is_sparse_attention_layer = (
                layer_idx in _sparse_attention_layer_ids(config)
            )
            disable_index_value = (
                sparse_attention_config["sparse_disable_index_value"][layer_idx] == 1
            )
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
        self.is_layer_sparse = (
            moe_layer_freq[layer_idx] != 0 if moe_layer_freq is not None else True
        )

        
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
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

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
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states

        return hidden_states

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
                name = name[len("model."):]
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
            if (
                ".index_" in name
                and ".index_q_proj" not in name
                and ".index_k_proj" not in name
            ):
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
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    mark_skipped("missing_bias")
                    continue
                if is_pp_missing_parameter(name, self):
                    mark_skipped("pp_missing")
                    continue
                if name.endswith((".k_scale", ".v_scale")):
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is not None and remapped_name in params_dict:
                        param = params_dict[remapped_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        mark_loaded(remapped_name)
                        break
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                mark_loaded(name)
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
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
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
        self.model = MiniMaxM3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
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
        
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
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
                if (
                    "vision_tower" in name
                    or "multi_modal_projector" in name
                    or "patch_merge_mlp" in name
                ):
                    skipped_multimodal_tensors += 1
                    continue
                text_tensors += 1
                yield name, weight

        loaded_params = loader.load_weights(
            text_weights(), mapper=self.hf_to_vllm_mapper
        )
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


def _get_minimax_m3_image_size(image_processor: object) -> ImageSize:
    patch_size = int(getattr(image_processor, "patch_size", 14))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    factor = patch_size * merge_size

    max_pixels = getattr(image_processor, "max_pixels", None)
    if not isinstance(max_pixels, int):
        size = getattr(image_processor, "size", None)
        if isinstance(size, Mapping):
            height = int(size.get("height") or size.get("shortest_edge") or 672)
            width = int(size.get("width") or size.get("longest_edge") or height)
            max_pixels = height * width
        elif isinstance(size, Sequence) and len(size) >= 2:
            max_pixels = int(size[0]) * int(size[1])
        else:
            max_pixels = 672 * 672

    side = max(factor, int(max_pixels**0.5) // factor * factor)
    return ImageSize(width=side, height=side)


def _get_minimax_m3_num_image_tokens(
    image_processor: object,
    *,
    image_width: int,
    image_height: int,
) -> int:
    merge_size = int(getattr(image_processor, "merge_size", 2))
    if hasattr(image_processor, "get_number_of_image_patches"):
        num_patches = image_processor.get_number_of_image_patches(
            image_height, image_width
        )
    else:
        patch_size = int(getattr(image_processor, "patch_size", 14))
        grid_h = image_height // patch_size
        grid_w = image_width // patch_size
        num_patches = grid_h * grid_w
    return int(num_patches) // (merge_size**2)


class MiniMaxM3VLProcessingInfo(BaseProcessingInfo):
    IMAGE_TOKEN = "]<]image[>["
    VIDEO_TOKEN = "]<]video[>["
    VISION_START_TOKEN = "]<]start of image[>["
    VISION_END_TOKEN = "]<]end of image[>["

    def get_hf_processor(self, **kwargs: object):
        # Do not type-check the processor class here: released MiniMax-M3
        # checkpoints may expose remote-code MiniMaxVLProcessor, while
        # transformers main uses MiniMaxM3VLProcessor.
        return self.ctx.get_hf_processor(**kwargs)

    def get_image_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).image_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def get_image_size_with_most_features(self) -> ImageSize:
        return _get_minimax_m3_image_size(self.get_image_processor())

    def get_max_image_tokens(self) -> int:
        image_processor = self.get_image_processor()
        size = self.get_image_size_with_most_features()
        return _get_minimax_m3_num_image_tokens(
            image_processor,
            image_width=size.width,
            image_height=size.height,
        )


class MiniMaxM3VLDummyInputsBuilder(
    BaseDummyInputsBuilder[MiniMaxM3VLProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.info.IMAGE_TOKEN * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
        }


class MiniMaxM3VLMultiModalProcessor(
    BaseMultiModalProcessor[MiniMaxM3VLProcessingInfo]
):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        image_token = getattr(
            hf_processor, "image_token", MiniMaxM3VLProcessingInfo.IMAGE_TOKEN
        )
        start_token = getattr(
            hf_processor,
            "VISION_START_TOKEN",
            MiniMaxM3VLProcessingInfo.VISION_START_TOKEN,
        )
        end_token = getattr(
            hf_processor,
            "VISION_END_TOKEN",
            MiniMaxM3VLProcessingInfo.VISION_END_TOKEN,
        )

        image_token_id = vocab[image_token]
        start_token_id = vocab[start_token]
        end_token_id = vocab[end_token]
        merge_length = int(getattr(image_processor, "merge_size", 2)) ** 2

        def get_image_replacement(item_idx: int):
            grid_thw = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod().item()) // merge_length
            full = [start_token_id] + [image_token_id] * num_tokens + [
                end_token_id
            ]
            return PromptUpdateDetails.select_token_id(full, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_image_replacement,
            )
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw")
        if image_grid_thw is None:
            image_grid_sizes = torch.empty(0, dtype=torch.long)
        else:
            image_grid_sizes = image_grid_thw.prod(-1)

        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes
            ),
            "image_grid_thw": MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            ),
        }


class MiniMaxM3VLModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        text_config = vllm_config.model_config.hf_text_config
        vision_config = getattr(config, "vision_config", None)
        if vision_config is None:
            raise ValueError("MiniMax-M3 VL requires config.vision_config.")

        if isinstance(vision_config, dict):
            vision_config = PretrainedConfig.from_dict(vision_config)

        projector_hidden_size = getattr(config, "projector_hidden_size", None)
        self.vision_tower = MiniMaxVLVisionModel(
            config=vision_config,
            text_hidden_size=text_config.hidden_size,
            projector_hidden_size=projector_hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )
        self.language_model = MiniMaxM3SparseForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )


@MULTIMODAL_REGISTRY.register_processor(
    MiniMaxM3VLMultiModalProcessor,
    info=MiniMaxM3VLProcessingInfo,
    dummy_inputs=MiniMaxM3VLDummyInputsBuilder,
)
class MiniMaxM3SparseForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP, SupportsEagle3
):
    supports_encoder_tp_data = True

    packed_modules_mapping = {
        **MiniMaxM3SparseForCausalLM.packed_modules_mapping,
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "model.language_model.model.",
            "language_model.model.": "model.language_model.model.",
            "language_model.lm_head.": "model.language_model.lm_head.",
            "model.vision_tower.": "model.vision_tower.",
            "vision_tower.": "model.vision_tower.",
            "multi_modal_projector.": (
                "model.vision_tower.multi_modal_projector."
            ),
            "patch_merge_mlp.": "model.vision_tower.patch_merge_mlp.",
            "lm_head.": "model.language_model.lm_head.",
        },
        orig_to_new_substr={
            ".mlp.fc1.": ".fc1.",
            ".mlp.fc2.": ".fc2.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "]<]image[>["
        if modality == "video":
            return "]<]video[>["
        raise ValueError(f"Unsupported modality: {modality!r}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model_config = vllm_config.model_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.use_data_parallel = (
            self.multimodal_config is not None
            and self.multimodal_config.mm_encoder_tp_mode == "data"
        )

        with self._mark_composite_model(
            vllm_config,
            language_targets=MiniMaxM3SparseForCausalLM,
            tower_targets={"image": MiniMaxVLVisionModel, "video": MiniMaxVLVisionModel},
        ):
            self.model = MiniMaxM3VLModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )

        self.vision_tower = self.model.vision_tower
        self.language_model = self.model.language_model
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @property
    def lm_head(self) -> nn.Module:
        return self.language_model.lm_head

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        image_embeds = kwargs.pop("image_embeds", None)
        if pixel_values is None and image_embeds is None:
            return None
        if pixel_values is not None:
            return {
                "type": "pixel_values",
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
        return {
            "type": "image_embeds",
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
        }

    def _parse_and_validate_video_input(self, **kwargs: object) -> dict | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        video_embeds = kwargs.pop("video_embeds", None)
        if pixel_values_videos is None and video_embeds is None:
            return None
        if pixel_values_videos is not None:
            return {
                "type": "pixel_values_videos",
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
            }
        return {
            "type": "video_embeds",
            "video_embeds": video_embeds,
            "video_grid_thw": video_grid_thw,
        }

    def _process_image_input(self, image_input: dict) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw is not None and grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.vision_tower.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.vision_tower.dtype)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.vision_tower,
                    pixel_values,
                    grid_thw_list,
                    rope_type="rope_3d",
                )
            image_embeds = self.vision_tower(
                pixel_values=pixel_values,
                grid_thw=grid_thw_list,
            )

        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(self, video_input: dict) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw is not None and grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.vision_tower.dtype)
        else:
            pixel_values = video_input["pixel_values_videos"].type(
                self.vision_tower.dtype
            )
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.vision_tower,
                    pixel_values,
                    grid_thw_list,
                    rope_type="rope_3d",
                )
            video_embeds = self.vision_tower(
                pixel_values=pixel_values,
                grid_thw=grid_thw_list,
            )

        merge_size = self.vision_tower.spatial_merge_size
        sizes = (grid_thw.prod(-1) // (merge_size * merge_size)).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> dict[str, dict]:
        mm_input_by_modality: dict[str, dict] = {}
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                image_input = self._parse_and_validate_image_input(**kwargs)
                if image_input is not None:
                    mm_input_by_modality["image"] = image_input
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                video_input = self._parse_and_validate_video_input(**kwargs)
                if video_input is not None:
                    mm_input_by_modality["video"] = video_input
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: list[torch.Tensor] = []
        for modality, multimodal_input in mm_input_by_modality.items():
            if modality == "image":
                multimodal_embeddings.extend(
                    self._process_image_input(multimodal_input)
                )
            elif modality == "video":
                multimodal_embeddings.extend(
                    self._process_video_input(multimodal_input)
                )
        return tuple(multimodal_embeddings)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return SupportsMultiModal.embed_input_ids(
            self,
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        raw_tensors = 0
        prefix_counts: dict[str, int] = {}

        def counted_weights() -> Iterable[tuple[str, torch.Tensor]]:
            nonlocal raw_tensors
            for name, weight in weights:
                raw_tensors += 1
                if name.startswith("language_model."):
                    bucket = "language_model"
                elif name.startswith("vision_tower."):
                    bucket = "vision_tower"
                elif name.startswith("multi_modal_projector."):
                    bucket = "multi_modal_projector"
                elif name.startswith("patch_merge_mlp."):
                    bucket = "patch_merge_mlp"
                else:
                    bucket = name.split(".", 1)[0]
                prefix_counts[bucket] = prefix_counts.get(bucket, 0) + 1
                yield name, weight

        logger.warning("MiniMax M3 VL load_weights entered")
        loaded_params = loader.load_weights(
            counted_weights(), mapper=self.hf_to_vllm_mapper
        )
        logger.warning(
            "MiniMax M3 VL load_weights saw %d checkpoint tensors by prefix %s; "
            "returned %d loaded parameter names",
            raw_tensors,
            prefix_counts,
            len(loaded_params),
        )
        return loaded_params


def get_spec_layer_idx_from_weight_name(
    config: PretrainedConfig, weight_name: str
) -> int | None:
    if hasattr(config, "num_mtp_modules") and (config.num_mtp_modules > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_mtp_modules):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None
