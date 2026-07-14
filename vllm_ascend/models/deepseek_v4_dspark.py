# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 DSpark draft model for Ascend."""

import copy
import typing
from collections.abc import Iterable, Sequence
from pathlib import Path

import regex as re
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.deepseek_v4 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MixtureOfExperts,
    DeepseekV4MoE,
)
from vllm_ascend.models.deepseek_v4_draft import remap_dspark_mtp_weight_name
from vllm_ascend.ops.dsa import unfold_kvcache
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.23.0"):
    from vllm.model_executor.layers.fused_moe import (
        fused_moe_make_expert_params_mapping,
    )

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.scale$")
_LAYER_ID_RE = re.compile(r"model\.layers\.(\d+)\.")


def _linear_output(layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    output = layer(hidden_states)
    return output[0] if isinstance(output, tuple) else output


def _draft_quant_config(vllm_config: VllmConfig):
    assert vllm_config.speculative_config is not None
    draft_config = vllm_config.speculative_config.draft_model_config.hf_config
    if getattr(draft_config, "dspark_mtp_dequantized_to_bf16", False):
        return None
    return vllm_config.quant_config


def _get_dspark_num_mtp_layers(config: PretrainedConfig) -> int:
    return int(getattr(config, "n_mtp_layers", None) or getattr(config, "dspark_num_mtp_layers", None) or 3)


def _load_dspark_quarot_rotation(
    vllm_config: VllmConfig,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    quant_description = getattr(vllm_config.quant_config, "quant_description", None)
    if not isinstance(quant_description, dict):
        return None
    try:
        relative_path = quant_description["optional"]["quarot"]["rotation_map"]["global_rotation"]
    except KeyError:
        return None

    rotation_path = Path(vllm_config.model_config.model) / relative_path
    rotation = load_file(rotation_path)["global_rotation"].to(device=device, dtype=torch.float32)
    logger.info_once("Loaded DSpark QuaRot rotation from %s", rotation_path)
    return rotation


def _apply_dspark_quarot_rotation(
    hidden_states: torch.Tensor,
    rotation: torch.Tensor | None,
    *,
    transpose: bool,
) -> torch.Tensor:
    if rotation is None:
        return hidden_states
    if rotation.device != hidden_states.device:
        raise RuntimeError("DSpark QuaRot rotation must be loaded on the same device as hidden states")
    matrix = rotation.t() if transpose else rotation
    return torch.matmul(hidden_states.float(), matrix).to(hidden_states.dtype)


def _hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    shape, dtype = hidden_states.shape, hidden_states.dtype
    hidden_flat = hidden_states.flatten(1).float()
    rsqrt = torch.rsqrt(hidden_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = torch.nn.functional.linear(hidden_flat, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    return torch.sum(pre.unsqueeze(-1) * hidden_flat.view(shape), dim=1).to(dtype)


def _scatter_context_kv(cache: torch.Tensor | list, kv: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    from vllm_ascend.device.device_op import DeviceOperator

    cache = unfold_kvcache(cache)
    slot_mapping = DeviceOperator.format_dsa_slot_mapping(slot_mapping, cache.shape[1])
    DeviceOperator.dsa_kv_compress_scatter(cache, kv, slot_mapping)


def _make_expert_params_mapping(model: nn.Module, num_experts: int) -> list[tuple[str, str, int, str]]:
    kwargs = dict(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=num_experts,
        num_redundant_experts=0,
    )
    if vllm_version_is("0.23.0"):
        return FusedMoE.make_expert_params_mapping(model, **kwargs)
    return fused_moe_make_expert_params_mapping(model, **kwargs)


class DSparkMarkovHead(nn.Module):
    def __init__(self, config: PretrainedConfig, prefix: str) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size,
            config.dspark_markov_rank,
            prefix=f"{prefix}.markov_w1",
        )
        self.markov_w2 = ParallelLMHead(
            config.vocab_size,
            config.dspark_markov_rank,
            org_num_embeddings=config.vocab_size,
            prefix=f"{prefix}.markov_w2",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.markov_w2, markov_embed)


class DeepseekV4DSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hc_mult = config.hc_mult
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_dspark_layers = _get_dspark_num_mtp_layers(config)
        self.mtp_start_layer_idx = config.num_hidden_layers

        draft_vllm_config = copy.copy(vllm_config)
        draft_vllm_config.quant_config = _draft_quant_config(vllm_config)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=draft_vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + index): DeepseekV2DecoderLayer(
                    draft_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx + index}"),
                    config=config,
                    is_draft_layer=True,
                )
                for index in range(self.num_dspark_layers)
            }
        )

        first_layer_idx = self.mtp_start_layer_idx
        last_layer_idx = first_layer_idx + self.num_dspark_layers - 1
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, f"layers.{first_layer_idx}.main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers[str(first_layer_idx)].main_proj = self.main_proj
        self.layers[str(first_layer_idx)].main_norm = self.main_norm

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            config,
            maybe_prefix(prefix, f"layers.{last_layer_idx}.markov_head"),
        )
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False)
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
        last_layer = self.layers[str(last_layer_idx)]
        last_layer.norm = self.norm
        last_layer.markov_head = self.markov_head
        last_layer.hc_head_fn = self.hc_head_fn
        last_layer.hc_head_base = self.hc_head_base
        last_layer.hc_head_scale = self.hc_head_scale

        device_config = getattr(vllm_config, "device_config", None)
        rotation_device = getattr(device_config, "device", current_platform.device_type)
        self.register_buffer(
            "_dspark_quarot_rotation",
            _load_dspark_quarot_rotation(vllm_config, rotation_device),
            persistent=False,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return _apply_dspark_quarot_rotation(
            self.embed_tokens(input_ids),
            self._dspark_quarot_rotation,
            transpose=True,
        )

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.main_norm(_linear_output(self.main_proj, aux_hidden_states))

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.dsa_attn.swa_cache_layer.prefix for layer in self.layers.values()]

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_hidden_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: Sequence[torch.Tensor | None] | None = None,
    ) -> None:
        if context_slot_mappings is not None and len(context_slot_mappings) != len(self.layers):
            raise ValueError("DSpark context slot mappings must match the number of draft layers")
        main_hidden_states = self.combine_hidden_states(main_hidden_states)
        cos, sin = get_cos_and_sin_dsa(context_positions)
        for index, layer in enumerate(self.layers.values()):
            slot_mapping = None if context_slot_mappings is None else context_slot_mappings[index]
            attn = layer.self_attn
            kv = attn.kv_norm(_linear_output(attn.wkv, main_hidden_states))
            kv = kv.view(-1, 1, attn.head_dim)
            layer_name = attn.rotary_emb.layername
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos[layer_name],
                sin[layer_name],
                rotary_mode="interleave",
                partial_slice=[attn.nope_head_dim, attn.head_dim],
            )
            if slot_mapping is not None:
                _scatter_context_kv(
                    attn.dsa_attn.swa_cache_layer.kv_cache,
                    kv,
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
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        for layer in self.layers.values():
            hidden_states, _ = layer(positions, hidden_states, None)
        return _hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.config.rms_norm_eps,
            self.config.hc_eps,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        if self._dspark_quarot_rotation is not None:
            hidden_states = hidden_states / self.norm.weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = _apply_dspark_quarot_rotation(
            hidden_states,
            self._dspark_quarot_rotation,
            transpose=False,
        )
        return logits_processor(lm_head, hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return _make_expert_params_mapping(self, self.config.n_routed_experts)

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.layers.values():
            finalize = getattr(layer.mlp, "finalize_mega_moe_weights", None)
            if finalize is not None:
                finalize()


@support_torch_compile
class DeepSeekV4DSparkMTP(nn.Module, DeepseekV2MixtureOfExperts):
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = _draft_quant_config(vllm_config)
        self.model = DeepseekV4DSparkModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.expert_weights: list[Sequence[torch.Tensor]] = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, DeepseekV4MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return self.model.get_draft_kv_cache_layer_names()

    def precompute_and_store_context_kv(
        self,
        main_hidden_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: Sequence[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(main_hidden_states, context_positions, context_slot_mappings)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        del hidden_states, intermediate_tensors, spec_step_idx
        assert input_ids is not None
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0) -> torch.Tensor:
        del spec_step_idx
        return self.model.compute_logits(hidden_states, self.lm_head, self.logits_processor)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = (
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
            ("shared_experts.gate_up_proj", "shared_experts.gate_proj", 0),
            ("shared_experts.gate_up_proj", "shared_experts.up_proj", 1),
        )
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        missing_mtp_params: set[str] = set()
        start_layer_idx = self.config.num_hidden_layers
        last_layer_idx = start_layer_idx + self.model.num_dspark_layers - 1

        tp_size = get_tensor_model_parallel_world_size()
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = get_tensor_model_parallel_rank() * heads_per_rank
        head_end = head_start + heads_per_rank
        expert_mapping = self.model.get_expert_mapping()
        expert_scale_suffix = (
            ".weight_scale" if getattr(self.config, "expert_dtype", "fp4") == "fp4" else ".weight_scale_inv"
        )

        for name, loaded_weight in weights:
            if name in ("embed.weight", "head.weight"):
                continue
            mapped_name = remap_dspark_mtp_weight_name(name, self.config.num_hidden_layers)
            if mapped_name is None:
                continue
            name = mapped_name
            if name.startswith(f"model.layers.{last_layer_idx}.hc_head_"):
                name = name.replace(f"model.layers.{last_layer_idx}.", "model.", 1)
            if name.endswith(".scale"):
                suffix = expert_scale_suffix if _EXPERT_SCALE_RE.search(name) else ".weight_scale"
                name = name.removesuffix(".scale") + suffix

            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if ".experts." in name or f".{weight_name}." not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                param = params_dict.get(mapped)
                if param is None:
                    missing_mtp_params.add(mapped)
                else:
                    param.weight_loader(param, loaded_weight, stacked_shard_id)
                    loaded_params.add(mapped)
                break
            else:
                if ".experts." in name:
                    matched_expert_mapping = False
                    if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for param_name, weight_name, expert_id, expert_shard_id in expert_mapping:
                        if weight_name not in name:
                            continue
                        matched_expert_mapping = True
                        mapped = name.replace(weight_name, param_name)
                        param = params_dict.get(mapped)
                        if param is None:
                            continue
                        weight_loader = typing.cast(typing.Callable[..., bool], param.weight_loader)
                        if weight_loader(
                            param,
                            loaded_weight,
                            mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        ):
                            loaded_params.add(mapped)
                            break
                    if not matched_expert_mapping:
                        missing_mtp_params.add(name)
                    continue
                if "attn_sink" in name:
                    param = params_dict.get(name)
                    if param is None:
                        missing_mtp_params.add(name)
                    else:
                        with torch.no_grad():
                            param[:heads_per_rank].copy_(loaded_weight[head_start:head_end])
                        loaded_params.add(name)
                    continue
                param = params_dict.get(name)
                if param is None:
                    missing_mtp_params.add(name)
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if missing_mtp_params:
            raise ValueError(f"DSpark checkpoint weights did not match model parameters: {sorted(missing_mtp_params)}")

        loaded_layer_ids = {
            int(match.group(1)) for name in loaded_params if (match := _LAYER_ID_RE.search(name)) is not None
        }
        for layer_idx in range(start_layer_idx, last_layer_idx + 1):
            if layer_idx not in loaded_layer_ids:
                raise ValueError(f"DSpark layer {layer_idx} weights missing from checkpoint")
        required_params = {
            f"model.layers.{start_layer_idx}.main_proj.weight",
            f"model.layers.{start_layer_idx}.main_norm.weight",
            f"model.layers.{last_layer_idx}.norm.weight",
            "model.hc_head_fn",
            "model.hc_head_base",
            "model.hc_head_scale",
            f"model.layers.{last_layer_idx}.markov_head.markov_w1.weight",
            f"model.layers.{last_layer_idx}.markov_head.markov_w2.weight",
        }
        missing_required = sorted(required_params - loaded_params)
        if missing_required:
            raise ValueError(f"DSpark required weights missing from checkpoint load: {missing_required}")
        self.model.finalize_mega_moe_weights()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params


DSparkDeepseekV4ForCausalLM = DeepSeekV4DSparkMTP
