# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend DSpark head, context KV and checkpoint-loading contracts."""

import typing
from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import PPMissingLayer, process_eagle_weight

from vllm_ascend.models.common.deepseek import DeepseekMixtureOfExperts, DeepseekMoE
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.utils import enable_dsa_cp


def _apply_dsv4_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    x: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    cos, sin = get_cos_and_sin_dsa(positions)
    layer_name = rotary_emb.layername
    cos_t = cos[layer_name]
    sin_t = sin[layer_name]
    if inverse:
        sin_t = -sin_t
    return rotary_emb(x, cos_t, sin_t)


def _get_dspark_num_mtp_layers(config: PretrainedConfig) -> int:
    num_layers = getattr(config, "n_mtp_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "dspark_num_mtp_layers", 3)
    return int(num_layers or 3)


class DeepseekDSparkModelMixin:
    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.dsa_attn.swa_cache_layer.prefix for layer in self.layers.values()]

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.main_norm(self.main_proj(aux_hidden_states))

    def _project_shared_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn: type[nn.Module] | None = None,
    ) -> torch.Tensor:
        assert attn is not None
        kv = attn.kv_norm(attn.wkv(hidden_states))
        k_nope, k_pe = kv.split([attn.nope_head_dim, attn.rope_head_dim], dim=-1)
        k_pe = _apply_dsv4_rope(attn.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        return torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, attn.head_dim).contiguous()

    def _store_standard_swa_kv(
        self,
        shared_kv: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        attn: type[nn.Module] | None = None,
    ) -> None:
        if slot_mapping is None or slot_mapping.numel() == 0:
            return

        assert attn is not None
        swa_cache_layer = attn.dsa_attn.swa_cache_layer
        swa_kv_cache = getattr(swa_cache_layer, "kv_cache", None)
        if swa_kv_cache is None:
            return
        while isinstance(swa_kv_cache, (list, tuple)) and len(swa_kv_cache) == 1:
            swa_kv_cache = swa_kv_cache[0]

        from vllm_ascend.attention.dsa_attn_kv_plan import get_dsa_attn_kv_plan

        if slot_mapping.ndim == 1:
            slot_mapping = get_dsa_attn_kv_plan(self.vllm_config).format_dsa_slot_mapping(
                slot_mapping, swa_cache_layer.block_size
            )
        get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(swa_kv_cache, shared_kv, slot_mapping)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: list[torch.Tensor | None] | None = None,
    ) -> None:
        if context_states.numel() == 0 or context_slot_mapping is None:
            return
        for layer_idx, layer in enumerate(self.layers.values()):
            layer_context_slot_mapping = None if context_slot_mapping is None else context_slot_mapping[layer_idx]
            if context_positions.numel() == 0:
                return
            attn = layer.self_attn
            shared_kv = self._project_shared_kv(context_states, context_positions, attn)
            self._store_standard_swa_kv(shared_kv, layer_context_slot_mapping, attn)

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = torch.nn.functional.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor, logits_processor: LogitsProcessor) -> torch.Tensor:
        return self.markov_head.bias(markov_embed, logits_processor)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        return logits_processor(lm_head, self.norm(hidden_states))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )


class DeepseekDSparkForCausalLMMixin(DeepseekMixtureOfExperts):
    def set_moe_parameters(self) -> None:
        self.expert_weights: typing.MutableSequence[typing.Sequence[torch.Tensor]] = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekMoE] = []
        example_moe = None
        for layer in self.model.layers.values():
            if isinstance(layer, PPMissingLayer):
                continue

            if isinstance(layer.mlp, DeepseekMoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
        )

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Full-vocab draft: base logits, no d2t scatter.
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids  # full-vocab: draft ids are target ids

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        del spec_step_idx
        return self.model.compute_logits(
            hidden_states,
            self.lm_head,
            self.logits_processor,
        )

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_bias(markov_embed, self.logits_processor)

    def compute_confidence(self, head_hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states,
            context_positions,
            context_slot_mapping,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{i}.*`` draft weights from the target checkpoint.

        Non-MTP weights belong to the target model and are skipped, except for
        standalone embedding/head weights used by the Ascend draft loader.
        """
        expert_mapping = self.model.get_expert_mapping()

        # (param_name, checkpoint shard name, shard_id) for non-expert
        # stacked parameters. Ascend keeps wq_a and wkv as separate parameters.
        stacked_params_mapping = [
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
            ("shared_experts.gate_up_proj", "shared_experts.gate_proj", 0),
            ("shared_experts.gate_up_proj", "shared_experts.up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        for name, loaded_weight in weights:
            if name == "embed.weight" and not self.rotation_path:
                name = "model.embed_tokens.weight"
            elif name == "head.weight" and not self.rotation_path:
                name = "lm_head.weight"
            elif name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                name = f"model.{name}"
            else:
                mapped_name = self._remap_dspark_name(name)
                if mapped_name is None:
                    continue
                name = mapped_name

            # Detect whether the checkpoint ships its own embed_tokens / lm_head
            # for the draft model.
            process_eagle_weight(self, name)

            # Expert scale parameters use Ascend's ``weight_scale`` convention.
            if name.endswith(".scale"):
                name = name.replace(".scale", ".weight_scale")

            # The multimodal checkpoint also contains one vision-router bias
            # for each MTP/DSpark layer.  DSpark runs only during text decode,
            # so draft MoE gates intentionally do not expose ``bias_vl``.
            # Do not alias it to the text correction bias: that would change
            # text routing whenever speculative decoding is enabled.
            if name.endswith(".e_score_correction_bias_vl") and name not in params_dict:
                logger.info_once("Ignoring vision-only router bias while loading the text-only DSpark drafter")
                continue

            if ".experts." in name:
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(typing.Callable[..., bool], param.weight_loader)
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                continue

            # Stacked rules only apply to decoder-layer weights. Head-stack
            # parameters load directly through the fallback below.
            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or f".{weight_name}." not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name)
                break
            else:
                if "attn_sink" in name:
                    if enable_dsa_cp():
                        narrow = loaded_weight
                    else:
                        narrow = loaded_weight[head_start:head_end]
                    with torch.no_grad():
                        params_dict[name].copy_(narrow)
                    loaded_params.add(name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _remap_dspark_name(self, name: str) -> str | None:
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        stage = int(m.group(1))
        rest = m.group(2)

        if stage == self.model.num_dspark_layers - 1 and rest.startswith("confidence_head."):
            return f"model.{rest}"

        if stage == 0 and rest == "embed.weight":
            return "model.embed_tokens.weight"
        if stage == self.model.num_dspark_layers - 1 and rest == "head.weight":
            return "lm_head.weight"
        if rest.startswith(("hc_head_fn", "hc_head_base", "hc_head_scale")):
            return f"model.{rest}"

        first_layer_idx = self.config.num_hidden_layers
        last_layer_idx = first_layer_idx + self.model.num_dspark_layers - 1
        if rest.startswith(("main_proj.", "main_norm.")):
            layer_idx = first_layer_idx
        elif rest.startswith(("norm.", "markov_head.")):
            layer_idx = last_layer_idx
        else:
            layer_idx = first_layer_idx + stage
        name = f"model.layers.{layer_idx}.{rest}"

        replacements = (
            (".attn.", ".self_attn."),
            (".ffn_norm.", ".post_attention_layernorm."),
            (".attn_norm.", ".input_layernorm."),
            (".ffn.", ".mlp."),
            (".w1.", ".gate_proj."),
            (".w2.", ".down_proj."),
            (".w3.", ".up_proj."),
            (".mlp.gate.bias", ".mlp.gate.e_score_correction_bias"),
        )
        for checkpoint_name, param_name in replacements:
            name = name.replace(checkpoint_name, param_name)
        return name
