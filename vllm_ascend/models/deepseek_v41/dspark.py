# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 dSPark draft model for Ascend."""

import typing
from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn
import vllm.envs as envs
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsEagle3
from vllm.model_executor.models.qwen3_dspark import DSparkConfidenceHead, DSparkMarkovHead
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix, process_eagle_weight

from vllm_ascend.attention.context_parallel.dsa_v41_cp import get_v41_cp_classes
from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheBackend, scatter_cache_sk
from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)
from vllm_ascend.models.deepseek_v41.model import (
    AscendDeepseekV41SWACache,
    DeepseekV41Attention,
    DeepseekV41DecoderLayer,
    DeepseekV41LayerRole,
    DeepseekV41SWAAttention,
)
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.utils import enable_dsa_cp, normalize_deepseek_v41_config

from .model import DeepseekV41MixtureOfExperts, DeepseekV41MoE


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


class DeepseekV41DSparkSWACache(AscendDeepseekV41SWACache):
    """DeepSeek V4.1 DSpark draft SWA cache layer.

    ``DeepseekV41DraftSWASpec`` exists only to identify the draft cache.
    """

    # TODO: Extract DeepseekV41DraftSWASpec construction from this cache subclass.
    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        return AscendSlidingWindowMLASpec(
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=spec.dtype,
            sliding_window=spec.sliding_window,
            cache_dtype_str=spec.cache_dtype_str,
            model_version=spec.model_version,
        )

    def get_attn_backend(self):
        return DeepseekV41CacheBackend


class DeepseekV41DSparkAttention(DeepseekV41SWAAttention):
    """DeepSeek V4.1 DSpark draft attention layer."""

    swa_cache_cls = DeepseekV41DSparkSWACache

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dsv41_backend = DeviceOperator.get_deepseek_v41_backend()
        self.softmax_scale = self.scale
        self.shared_state = None
        prefix = kwargs["prefix"]
        # Returns (metadata_builder_cls, impl_cls); select the CP-aware implementation.
        self.v41_impl = get_v41_cp_classes()[1](
            prefix=prefix,
            role=DeepseekV41LayerRole(
                layer_idx=int(prefix.split(".")[-2]),
                compress_ratio=0,
                kv_source_layer=None,
                index_source_layer=None,
                is_kv_source=False,
                is_index_source=False,
                is_candidate_source=False,
                uses_candidate_filter=False,
                engram_slot=None,
            ),
            topology=None,
            long_kv_source_prefix=None,
            index_k_source_prefix=None,
        )
        self.v41_layer_name = f"{prefix}.v41_attn"
        context = kwargs["vllm_config"].compilation_config.static_forward_context
        context[self.v41_layer_name] = self

    forward = DeepseekV41Attention.forward


class DeepseekV41DSparkDecoderLayer(DeepseekV41DecoderLayer):
    """V4.1 delayed-mHC block with a draft-only SWA attention backend."""

    attention_cls = DeepseekV41DSparkAttention


class DeepseekV41DSparkModel(torch.nn.Module):
    """Three serial draft blocks matching the checkpoint's ``mtp.*`` tree."""

    def __init__(self, *, vllm_config, prefix="") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        draft_model_config = vllm_config.speculative_config.draft_model_config
        config = normalize_deepseek_v41_config(draft_model_config.hf_text_config)
        self.config = config
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.block_size = int(config.dspark_block_size)
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.num_dspark_layers = _get_dspark_num_mtp_layers(config)
        self.mtp_start_layer_idx = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = torch.nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + idx): DeepseekV41DSparkDecoderLayer(
                    vllm_config,
                    prefix=f"mtp.{idx}",
                    config=config,
                    is_draft_layer=True,
                )
                for idx in range(self.num_dspark_layers)
            }
        )

        first_layer = self.layers[str(self.mtp_start_layer_idx)]
        self.use_sequence_parallel_moe = vllm_config.parallel_config.use_sequence_parallel_moe
        # V4.1 keeps the checkpoint quantization contract on the composite
        # config.  Its DSpark main projection is FP8 in the release checkpoint,
        # so constructing a BF16 linear drops the scale parameter and makes the
        # draft loader fail on ``main_proj.scale``.
        checkpoint_config = getattr(draft_model_config, "hf_config", None)
        model_quant_config = getattr(checkpoint_config, "quantization_config", None)
        main_proj_quant_config = (
            vllm_config.quant_config
            if model_quant_config is not None
            and model_quant_config.get("quant_method") == "fp8"
            else None
        )
        self.main_proj = ColumnParallelLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=main_proj_quant_config,
            prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx}.main_proj"),
            gather_output=True,
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        first_layer.main_proj = self.main_proj
        first_layer.main_norm = self.main_norm

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        last_layer_idx = self.mtp_start_layer_idx + self.num_dspark_layers - 1
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            getattr(config, "draft_vocab_size", None) or config.vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, f"layers.{last_layer_idx}.markov_head"),
        )
        self.confidence_head = DSparkConfidenceHead(
            input_dim=config.hidden_size + config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "confidence_head"),
            bias=False,
            with_markov=True,
        )
        last_layer = self.layers[str(last_layer_idx)]
        last_layer.norm = self.norm
        last_layer.markov_head = self.markov_head

        self.needs_moe_input_ids = any(
            layer.mlp.gate.tid2eid is not None or layer.mlp.gate.bias_vl is not None for layer in self.layers.values()
        )

    def _store_standard_swa_kv(self, shared_kv, slot_mapping, attn=None):
        if slot_mapping is None or slot_mapping.numel() == 0:
            return
        cache = attn.dsa_attn.swa_cache_layer
        values = shared_kv.squeeze(1)
        if attn.dsv41_backend is not None:
            # A5's cache is physically packed and must be updated through the
            # device-routed writer.  The draft proposer already supplies linear
            # physical slots, so keep them in that form for the native kernel.
            attn.dsv41_backend.write_attention_cache(
                cache.kv_cache[0],
                slot_mapping,
                values,
                kind="win",
            )
            return
        if slot_mapping.ndim == 1:
            valid = slot_mapping >= 0
            physical = slot_mapping.clamp_min(0)
            slot_mapping = torch.stack((physical // cache.block_size, physical % cache.block_size), dim=-1).to(
                torch.int32
            )
            slot_mapping.masked_fill_(~valid.unsqueeze(-1), -1)
        scatter_cache_sk(cache.kv_cache[0], slot_mapping, values)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids).unsqueeze(-2).repeat(1, self.hc_mult, 1)
        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel_moe:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding,
                    hidden_states,
                )
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)
        pre_mix = hidden_states.new_zeros(hidden_states.shape[0], self.hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        last_layer = None
        moe_input_ids = input_ids
        if self.needs_moe_input_ids:
            moe_input_ids = torch.where(input_ids == -1, 0, input_ids)
        for layer in self.layers.values():
            last_layer = layer
            hidden_states, pre_mix = layer(
                positions,
                hidden_states,
                pre_mix,
                llama_4_scaling=None,
                input_ids=moe_input_ids,
            )
        assert last_layer is not None, "Hyper-connection collapse requires at least one decoder layer"
        hidden_states = last_layer.hc_collapse(hidden_states, pre_mix)
        if self.use_sequence_parallel_moe:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        return hidden_states

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.dsa_attn.swa_cache_layer.prefix for layer in self.layers.values()]

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.main_norm(self.main_proj(aux_hidden_states))

    def _project_shared_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn: type[nn.Module],
    ) -> torch.Tensor:
        kv = attn.kv_norm(attn.wkv(hidden_states))
        k_nope, k_pe = kv.split([attn.nope_head_dim, attn.rope_head_dim], dim=-1)
        k_pe = _apply_dsv4_rope(attn.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        return torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, attn.head_dim).contiguous()

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


@support_torch_compile
class DSparkDeepseekV41ForCausalLM(torch.nn.Module, DeepseekV41MixtureOfExperts, SupportsEagle3):
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, *, vllm_config, prefix="") -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_text_config

        from vllm_ascend.utils import get_rotation_path

        self.rotation_path = get_rotation_path(vllm_config) if vllm_config.quant_config is not None else None
        self.model = DeepseekV41DSparkModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.set_moe_parameters()
        from vllm_ascend.ascend_forward_context import MoECommType
        from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

        self.moe_comm_methods = {kind: get_moe_comm_method(kind) for kind in MoECommType}

    def _remap_dspark_name(self, name: str) -> str | None:
        mapped = self._remap_checkpoint_name(name)
        if mapped is None:
            return None
        # DeepSeek V4.1 names the low-rank Markov matrices after their operations,
        # while the runtime uses explicit embedding/projection parameter names.
        mapped = mapped.replace(".markov_head.embed.weight", ".markov_head.markov_w1.weight")
        mapped = mapped.replace(".markov_head.head.weight", ".markov_head.markov_w2.weight")
        mapped = mapped.replace("model.confidence_head.weight", "model.confidence_head.proj.weight")
        return mapped

    def set_moe_parameters(self) -> None:
        self.expert_weights: typing.MutableSequence[typing.Sequence[torch.Tensor]] = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV41MoE] = []
        example_moe = None
        for layer in self.model.layers.values():
            if isinstance(layer, PPMissingLayer):
                continue

            if isinstance(layer.mlp, DeepseekV41MoE):
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

    def _remap_checkpoint_name(self, name: str) -> str | None:
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
