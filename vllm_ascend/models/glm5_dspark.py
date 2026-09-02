# SPDX-License-Identifier: Apache-2.0
"""GLM-5 dense MLA draft model for DSpark speculative decoding on Ascend."""

from collections.abc import Iterable
from copy import copy

import torch
import torch.nn as nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention, DeepseekV2MLP
from vllm.model_executor.models.qwen3_dflash import _get_dflash_fc_input_size
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

from vllm_ascend.models.llama_eagle3 import get_rotation_matrix, get_rotation_path
from vllm_ascend.models.qwen3_dspark import DSparkConfidenceHead, process_weight

_GLM5_DSPARK_MLA_FIELDS = (
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
)


def _split_interleaved_cos_sin_cache(rotary_emb: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert vLLM's interleaved RoPE cache to separate Ascend cos/sin caches."""
    cos_sin_cache = rotary_emb.cos_sin_cache
    rotary_dim = cos_sin_cache.shape[-1] // 2
    cos_cache, sin_cache = cos_sin_cache.view(-1, 2, rotary_dim).repeat(1, 1, 2).chunk(2, dim=1)
    return cos_cache.squeeze(1), sin_cache.squeeze(1)


def _select_cos_sin(
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos_cache[positions].unsqueeze(1).unsqueeze(2)
    sin = sin_cache[positions].unsqueeze(1).unsqueeze(2)
    return cos, sin


class Glm5DSparkMLAAttention(DeepseekV2MLAAttention):
    """Dense MLA used by the self-contained GLM-5 DSpark draft."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        quant_config,
        prefix: str,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        # Ascend's cache writer and the proposer both use the inner attention
        # layer name, implementation, and cache directly.
        self.attn = self.mla_attn.mla_attn

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return super().forward(positions, hidden_states, llama_4_scaling=None)


class Glm5DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        start_layer_id: int,
        quant_config,
        prefix: str,
    ) -> None:
        super().__init__()
        layer_prefix = maybe_prefix(prefix, f"layers.{start_layer_id + layer_idx}")
        self.self_attn = Glm5DSparkMLAAttention(
            vllm_config=vllm_config,
            config=config,
            quant_config=quant_config,
            prefix=maybe_prefix(layer_prefix, "self_attn"),
        )
        self.mlp = DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=maybe_prefix(layer_prefix, "mlp"),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Glm5DSparkModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        draft_model_config = vllm_config.speculative_config.draft_model_config
        if draft_model_config is None:
            raise ValueError("GLM-5 MLA DSpark requires a draft model config.")
        self.config = draft_model_config.hf_config
        self.quant_config = None

        missing_fields = [name for name in _GLM5_DSPARK_MLA_FIELDS if getattr(self.config, name, None) is None]
        if missing_fields:
            raise ValueError(f"GLM-5 MLA DSpark config is missing fields: {missing_fields}")

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.context_proj = ReplicatedLinear(
            _get_dflash_fc_input_size(vllm_config),
            self.config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "context_proj"),
        )
        self.context_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                Glm5DSparkDecoderLayer(
                    vllm_config=vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    start_layer_id=start_layer_id,
                    quant_config=self.quant_config,
                    prefix=prefix,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        draft_vocab_size = getattr(self.config, "draft_vocab_size", None) or self.config.vocab_size
        self.markov_head = DSparkMarkovHead(
            self.config.vocab_size,
            draft_vocab_size,
            self.config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=self.quant_config,
        )
        self.enable_confidence_head = bool(
            getattr(self.config, "enable_confidence_head", hasattr(self.config, "markov_head_type"))
        )
        if self.enable_confidence_head:
            self.confidence_head = DSparkConfidenceHead(
                config=self.config,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )

        self._rope_cos_cache, self._rope_sin_cache = _split_interleaved_cos_sin_cache(
            self.layers[0].self_attn.rotary_emb
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.context_norm(self.context_proj(hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: list[torch.Tensor | None] | None = None,
    ) -> None:
        if context_states.numel() == 0 or context_slot_mapping is None:
            return
        if len(context_slot_mapping) != len(self.layers):
            raise ValueError(
                "context_slot_mapping must contain one entry per GLM-5 draft layer: "
                f"got {len(context_slot_mapping)} entries for {len(self.layers)} layers"
            )

        cos, sin = _select_cos_sin(self._rope_cos_cache, self._rope_sin_cache, context_positions)
        for layer, slot_mapping in zip(self.layers, context_slot_mapping, strict=True):
            if slot_mapping is None or slot_mapping.numel() == 0:
                continue
            attn = layer.self_attn
            qkv_lora = attn.fused_qkv_a_proj(context_states)[0]
            kv_no_split = qkv_lora[..., attn.q_lora_rank :].contiguous()
            attn.attn.impl.exec_kv_prefill(
                kv_no_split,
                cos,
                sin,
                attn.attn.kv_cache,
                slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.final_norm(hidden_states, residual)
        return hidden_states

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        return [True] * len(self.layers)


class Glm5DSparkForCausalLM(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        if self.draft_model_config is None:
            raise ValueError("GLM-5 MLA DSpark requires a draft model config.")
        self.config = self.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        draft_vllm_config = copy(vllm_config)
        draft_vllm_config.quant_config = None
        draft_vocab_size = getattr(self.config, "draft_vocab_size", None) or self.config.vocab_size
        with set_current_vllm_config(draft_vllm_config):
            self.model = Glm5DSparkModel(
                vllm_config=draft_vllm_config,
                start_layer_id=target_layer_num,
                prefix=maybe_prefix(prefix, "model"),
            )
            self.lm_head = ParallelLMHead(
                draft_vocab_size,
                self.config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(draft_vocab_size)
        self.enable_confidence_head = self.model.enable_confidence_head
        self.has_own_embed_tokens = True
        self.has_own_lm_head = True
        if draft_vocab_size != vllm_config.model_config.get_vocab_size():
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None
        self.rotation_path = get_rotation_path(vllm_config) if vllm_config.quant_config is not None else None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def get_draft_attn_causal(self) -> list[bool]:
        return self.model.get_draft_attn_causal()

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(context_states, context_positions, context_slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def apply_markov_bias_gathered(
        self,
        markov_embed: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.markov_head.apply_bias_gathered(
            markov_embed,
            logits,
            values,
            index,
            self.logits_processor.scale,
        )

    def confidence_logits(self, hidden_states: torch.Tensor, markov_embeds: torch.Tensor) -> torch.Tensor:
        if not self.enable_confidence_head:
            raise RuntimeError("The DSpark confidence head is disabled.")
        return self.model.confidence_head(hidden_states, markov_embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        optional_weights = ("embed_tokens", "lm_head", "confidence_head")
        included_weights: set[str] = set()
        includes_draft_id_mapping = False
        rotation_weight = get_rotation_matrix(self.rotation_path) if self.rotation_path is not None else None
        normalized_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            # t2d is only needed for training. At inference time DSpark maps
            # sampled draft IDs back to the target vocabulary with d2t.
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            if name == "norm.weight":
                name = "final_norm.weight"
            if "draft_id_to_target_id" not in name and not name.startswith("lm_head."):
                name = f"model.{name}"
            if rotation_weight is not None and "context_proj.weight" in name:
                weight = process_weight(weight, rotation_weight)
            included_weights.update(optional_weight for optional_weight in optional_weights if optional_weight in name)
            normalized_weights.append((name, weight))

        self.has_own_embed_tokens = "embed_tokens" in included_weights
        self.has_own_lm_head = "lm_head" in included_weights
        if "confidence_head" not in included_weights:
            self.enable_confidence_head = False

        loader = AutoWeightsLoader(self)
        mapper = self.hf_to_vllm_mapper
        if not includes_draft_id_mapping:
            mapper |= WeightsMapper(orig_to_new_substr={"draft_id_to_target_id": None})
        return loader.load_weights(normalized_weights, mapper=mapper)
