# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 DSpark draft model for Ascend.

DSpark weights are stored under the target checkpoint's ``mtp.*`` namespace,
but the draft path is a block drafter rather than the ordinary serial MTP
module. The target model provides selected layer hidden states; this model
projects them into the draft attention context and emits a full draft block.
"""

import typing
from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from vllm_ascend import envs
from vllm_ascend.models.deepseek_v4 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MixtureOfExperts,
    DeepseekV4Attention,
    DeepseekV4MoE,
)
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.utils import vllm_version_is

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.scale$")
_LAYER_ID_RE = re.compile(r"model\.layers\.(\d+)\.")

if not vllm_version_is("0.23.0"):
    from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping


def _make_deepseek_v4_expert_params_mapping(
    model: nn.Module,
    num_experts: int,
    num_redundant_experts: int = 0,
) -> list[tuple[str, str, int, str]]:
    if vllm_version_is("0.23.0"):
        return FusedMoE.make_expert_params_mapping(
            model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
            num_redundant_experts=num_redundant_experts,
        )
    return fused_moe_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=num_experts,
        num_redundant_experts=num_redundant_experts,
    )


def _hc_head_torch(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    shape, dtype = x.size(), x.dtype
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=1)
    return y.to(dtype)


def _linear_output(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = layer(x)
    return out[0] if isinstance(out, tuple) else out


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


def _draft_quant_config(vllm_config: VllmConfig):
    assert vllm_config.speculative_config is not None
    draft_config = vllm_config.speculative_config.draft_model_config.hf_config
    if getattr(draft_config, "dspark_mtp_dequantized_to_bf16", False):
        return None
    return vllm_config.quant_config


def _get_dspark_num_mtp_layers(config: PretrainedConfig) -> int:
    num_layers = getattr(config, "n_mtp_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "dspark_num_mtp_layers", 3)
    return int(num_layers or 3)


class DeepseekV4DSparkAttention(DeepseekV4Attention):
    """DSpark sliding-window attention with an internal eager context cache.

    The draft attention reuses the target model's DSA path
    (``self.dsa_attn`` -> ``AscendDSAImpl._forward_decode``) and only diverges
    via the per-layer dspark attention metadata (non-causal visible slot list,
    a.k.a. ``dspark_swa_indices``) built by the DSA metadata builder. The
    eager projection of the target's context KV (``precompute_context_kv``) is
    the only DSpark-specific code that stays here.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Draft layers are SWA-only (no compressor/indexer); pin both the
        # module and the DSA wrapper so _build_kv_cache skips the compress path.
        self.compress_ratio = 1
        self.dsa_attn.compress_ratio = 1

    def _project_shared_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        kv = self.kv_norm(_linear_output(self.wkv, hidden_states))
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k_pe = _apply_dsv4_rope(self.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        return torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, self.head_dim).contiguous()

    def _store_standard_swa_kv(
        self,
        shared_kv: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        if slot_mapping is None or slot_mapping.numel() == 0:
            return

        swa_cache_layer = self.dsa_attn.swa_cache_layer
        swa_kv_cache = getattr(swa_cache_layer, "kv_cache", None)
        if swa_kv_cache is None:
            return
        while isinstance(swa_kv_cache, (list, tuple)) and len(swa_kv_cache) == 1:
            swa_kv_cache = swa_kv_cache[0]

        from vllm_ascend.device.device_op import DeviceOperator

        slot_mapping = slot_mapping.to(device=shared_kv.device, dtype=torch.int32)
        valid = slot_mapping >= 0 if slot_mapping.ndim == 1 else torch.all(slot_mapping >= 0, dim=-1)
        if not bool(torch.any(valid).item()):
            return
        if not bool(torch.all(valid).item()):
            shared_kv = shared_kv[valid]
            slot_mapping = slot_mapping[valid]
        if slot_mapping.ndim == 1:
            slot_mapping = DeviceOperator.format_dsa_slot_mapping(slot_mapping, swa_cache_layer.block_size)
        DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, shared_kv, slot_mapping)

    def precompute_context_kv(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        if positions.numel() == 0:
            return
        shared_kv = self._project_shared_kv(main_x, positions)
        self._store_standard_swa_kv(shared_kv, context_slot_mapping)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Reuse the target model's DSA attention path. The per-layer dspark
        # attention metadata (block_table / query_start_loc / seq_lens /
        # token_to_req_indices / dspark_swa_indices) is resolved by the DSA
        # backend from the forward context (keyed by swa_cache_layer.prefix),
        # so forward() takes no attention kwargs -- matching
        # DeepseekV4Attention.forward.
        return self.dsa_attn(positions, hidden_states)


class DeepseekV4DSparkDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            config=config,
            topk_indices_buffer=None,
            is_draft_layer=True,
            attn_cls=DeepseekV4DSparkAttention,
        )
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # MHC pre/post reuse the upstream NPU fused ops ``npu_hc_pre_v2`` /
        # ``npu_hc_post`` inherited from DeepseekV2DecoderLayer instead of the
        # torch reference. The torch reference fused the previous step's post
        # with the current step's pre into one call; here they are two NPU ops,
        # numerically equivalent at the bf16 ulp level (the NPU op hardcodes the
        # post alpha of 2.0; equivalence was verified against a torch reference
        # before that reference was removed).
        if residual is None:
            residual = hidden_states
            hidden_states, post_mix, res_mix = self.hc_pre(
                hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
        else:
            assert post_mix is not None and res_mix is not None
            residual = self.hc_post(hidden_states, residual, post_mix, res_mix)
            hidden_states, post_mix, res_mix = self.hc_pre(
                residual, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)

        residual = self.hc_post(hidden_states, residual, post_mix, res_mix)
        hidden_states, post_mix, res_mix = self.hc_pre(
            residual, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, input_ids)
        return hidden_states, residual, post_mix, res_mix


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
        self.hidden_size = config.hidden_size
        self.block_size = int(config.dspark_block_size)
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.num_dspark_layers = _get_dspark_num_mtp_layers(config)
        self.mtp_start_layer_idx = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=_draft_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + idx): DeepseekV4DSparkDecoderLayer(
                    vllm_config,
                    prefix=f"mtp.{idx}",
                )
                for idx in range(self.num_dspark_layers)
            }
        )

        first_layer = self.layers[str(self.mtp_start_layer_idx)]
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx}.main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        first_layer.main_proj = self.main_proj
        first_layer.main_norm = self.main_norm

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        last_layer_idx = self.mtp_start_layer_idx + self.num_dspark_layers - 1
        self.markov_head = DSparkMarkovHead(
            config,
            maybe_prefix(prefix, f"layers.{last_layer_idx}.markov_head"),
        )
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        last_layer = self.layers[str(last_layer_idx)]
        last_layer.norm = self.norm
        last_layer.markov_head = self.markov_head
        last_layer.hc_head_fn = self.hc_head_fn
        last_layer.hc_head_base = self.hc_head_base
        last_layer.hc_head_scale = self.hc_head_scale

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.dsa_attn.swa_cache_layer.prefix for layer in self.layers.values()]

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.main_norm(_linear_output(self.main_proj, aux_hidden_states))

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: list[torch.Tensor | None] | None = None,
    ) -> None:
        if context_states.numel() == 0:
            return
        for layer_idx, layer in enumerate(self.layers.values()):
            layer_context_slot_mapping = None if context_slot_mapping is None else context_slot_mapping[layer_idx]
            layer.self_attn.precompute_context_kv(
                context_states,
                context_positions,
                context_slot_mapping=layer_context_slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids).unsqueeze(-2).repeat(1, self.hc_mult, 1)
        residual = post_mix = res_mix = None
        for layer in self.layers.values():
            hidden_states, residual, post_mix, res_mix = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                post_mix=post_mix,
                res_mix=res_mix,
                input_ids=input_ids,
            )
        head_hidden = self.compute_head_hidden(hidden_states, residual, post_mix, res_mix)
        return head_hidden

    def compute_head_hidden(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residual is not None and post_mix is not None and res_mix is not None:
            # Final MHC post of the last decoder layer's output, reusing the
            # inherited hc_post (DeepseekV2DecoderLayer.hc_post -> npu_hc_post).
            last_layer = self.layers[str(self.mtp_start_layer_idx + self.num_dspark_layers - 1)]
            hidden_states = last_layer.hc_post(hidden_states, residual, post_mix, res_mix)
        if hidden_states.dim() == 2:
            return hidden_states
        return _hc_head_torch(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.config.rms_norm_eps,
            self.config.hc_eps,
        )

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.markov_head.bias(markov_embed)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        head_hidden = self.compute_head_hidden(hidden_states)
        return logits_processor(lm_head, self.norm(head_hidden))

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return _make_deepseek_v4_expert_params_mapping(
            self,
            num_experts=self.config.n_routed_experts,
        )

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.layers.values():
            finalize = getattr(layer.mlp, "finalize_mega_moe_weights", None)
            if finalize is not None:
                finalize()


@support_torch_compile
class DeepSeekV4DSparkMTP(nn.Module, DeepseekV2MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = _draft_quant_config(vllm_config)
        self.has_own_embed_tokens = self.quant_config is not None
        self.has_own_lm_head = self.quant_config is not None
        self.model = DeepseekV4DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.expert_weights: typing.MutableSequence[typing.Sequence[torch.Tensor]] = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV4MoE] = []
        example_moe = None
        for layer in self.model.layers.values():
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, DeepseekV2DecoderLayer)
            if isinstance(layer.mlp, DeepseekV4MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
        )

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
        return self.model.markov_bias(markov_embed)

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
        stacked_params_mapping = [
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
            ("shared_experts.gate_up_proj", "shared_experts.gate_proj", 0),
            ("shared_experts.gate_up_proj", "shared_experts.up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        missing_mtp_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank
        head_end = head_start + heads_per_rank
        expert_mapping = self.model.get_expert_mapping()
        expert_scale_suffix = (
            ".weight_scale" if getattr(self.config, "expert_dtype", "fp4") == "fp4" else ".weight_scale_inv"
        )
        start_layer_idx = self.config.num_hidden_layers
        last_layer_idx = start_layer_idx + self.model.num_dspark_layers - 1

        for name, loaded_weight in weights:
            if name == "embed.weight" and not self.has_own_embed_tokens:
                embed_name = "model.embed_tokens.weight"
                param = params_dict[embed_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(embed_name)
                continue
            if name == "head.weight" and not self.has_own_lm_head:
                head_name = "lm_head.weight"
                param = params_dict[head_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(head_name)
                continue
            mapped_name = self._remap_dspark_name(name)
            if mapped_name is None:
                continue
            name = mapped_name
            if name.startswith(f"model.layers.{last_layer_idx}.hc_head_"):
                canonical_name = name.replace(f"model.layers.{last_layer_idx}.", "model.", 1)
                if canonical_name in params_dict:
                    name = canonical_name
            if name.endswith(".scale"):
                suffix = expert_scale_suffix if _EXPERT_SCALE_RE.search(name) else ".weight_scale"
                name = name.removesuffix(".scale") + suffix
                if name not in params_dict:
                    continue
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if ".experts." in name or f".{weight_name}." not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    missing_mtp_params.add(mapped)
                    break
                param = params_dict[mapped]
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
                        if mapped not in params_dict:
                            continue
                        param = params_dict[mapped]
                        weight_loader = typing.cast(typing.Callable[..., bool], param.weight_loader)
                        success = weight_loader(
                            param,
                            loaded_weight,
                            mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            loaded_params.add(mapped)
                            break
                    if not matched_expert_mapping:
                        missing_mtp_params.add(name)
                    continue
                if "attn_sink" in name:
                    if name not in params_dict:
                        missing_mtp_params.add(name)
                        continue
                    narrow = loaded_weight[head_start:head_end]
                    with torch.no_grad():
                        params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                if name not in params_dict:
                    missing_mtp_params.add(name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if missing_mtp_params:
            raise ValueError(
                "DSpark speculative decoding checkpoint weights did not match model parameters: "
                f"{sorted(missing_mtp_params)}"
            )

        loaded_layer_ids: set[int] = set()
        for param_name in loaded_params:
            match = _LAYER_ID_RE.search(param_name)
            if match:
                loaded_layer_ids.add(int(match.group(1)))
        for layer_idx in range(start_layer_idx, start_layer_idx + self.model.num_dspark_layers):
            if layer_idx not in loaded_layer_ids:
                raise ValueError(f"DSpark speculative decoding layer {layer_idx} weights missing from checkpoint.")
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
            raise ValueError(
                f"DSpark speculative decoding required weights missing from checkpoint load: {missing_required}"
            )
        self.model.finalize_mega_moe_weights()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _remap_dspark_name(self, name: str) -> str | None:
        if name == "mtp.0.embed.weight":
            return "model.embed_tokens.weight"
        if name == "mtp.2.head.weight":
            return "lm_head.weight"
        match = re.match(r"mtp\.(\d+)\.(.*)", name)
        if match is None:
            return None
        stage_idx = int(match.group(1))
        layer_idx = self.config.num_hidden_layers + stage_idx
        rest = match.group(2)
        if rest.startswith("confidence_head."):
            return None
        name = f"model.layers.{layer_idx}.{rest}"
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".w1.", ".gate_proj.")
        name = name.replace(".w2.", ".down_proj.")
        name = name.replace(".w3.", ".up_proj.")
        name = name.replace(".mlp.gate.bias", ".mlp.gate.e_score_correction_bias")
        return name


DSparkDeepseekV4ForCausalLM = DeepSeekV4DSparkMTP
