# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 DSpark draft model for Ascend.

DSpark weights are stored under the target checkpoint's ``mtp.*`` namespace,
but the draft path is a block drafter rather than the ordinary serial MTP
module. The target model provides selected layer hidden states; this model
projects them into the draft attention context and emits a full draft block.
"""

import os
import typing
from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.deepseek_v4 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MixtureOfExperts,
    DeepseekV4Attention,
    _apply_dsv4_rope,
    _apply_dsv4_rope_tail,
    _grouped_wo_a_projection,
    _hc_head_torch,
    _linear_output,
    _make_deepseek_v4_expert_params_mapping,
    _wo_a_weight_for_eager_projection,
)
from vllm_ascend.ops.dspark_attention import dspark_attention

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.scale$")
_LAYER_ID_RE = re.compile(r"model\.layers\.(\d+)\.")

_FP8_E4M3FN_SUBNORMAL_STEP = 2.0**-9
_FP8_E4M3FN_MIN_NORMAL = 2.0**-6
_FP8_E4M3FN_SUBNORMAL_NORMAL_MIDPOINT = (7 * _FP8_E4M3FN_SUBNORMAL_STEP + _FP8_E4M3FN_MIN_NORMAL) * 0.5


def _draft_quant_config(vllm_config: VllmConfig):
    assert vllm_config.speculative_config is not None
    draft_config = vllm_config.speculative_config.draft_model_config.hf_config
    if getattr(draft_config, "dspark_mtp_dequantized_to_bf16", False):
        return None
    return vllm_config.quant_config


def _dspark_cache_capacity(vllm_config: VllmConfig, block_size: int, window_size: int | None = None) -> int:
    if window_size is not None:
        return max(block_size, int(window_size) + block_size)
    model_config = getattr(vllm_config, "model_config", None)
    max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
    return max(block_size, max_model_len + block_size)


def _dspark_max_request_slots(vllm_config: VllmConfig) -> int:
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    return max(1, int(getattr(scheduler_config, "max_num_seqs", 1) or 1))


def _get_dspark_num_mtp_layers(config: PretrainedConfig) -> int:
    num_layers = getattr(config, "n_mtp_layers", None)
    if num_layers is None:
        num_layers = getattr(config, "dspark_num_mtp_layers", 3)
    return int(num_layers or 3)


def _dspark_standard_dsa_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_DSPARK_USE_STANDARD_DSA", "0") == "1"


def _fp8_e4m3fn_quantized_abs(abs_scaled: torch.Tensor) -> torch.Tensor:
    subnormal = torch.floor(abs_scaled / _FP8_E4M3FN_SUBNORMAL_STEP + 0.5).clamp(0, 7) * _FP8_E4M3FN_SUBNORMAL_STEP

    normal_exp = torch.floor(torch.log2(abs_scaled.clamp_min(_FP8_E4M3FN_MIN_NORMAL))).clamp(-6, 8)
    normal_base = torch.exp2(normal_exp)
    mantissa = torch.floor((abs_scaled / normal_base - 1.0) * 8.0 + 0.5)
    carry = mantissa >= 8
    normal_exp = torch.where(carry, normal_exp + 1.0, normal_exp).clamp(-6, 8)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    mantissa = torch.where(
        normal_exp >= 8,
        mantissa.clamp(0, 6),
        mantissa.clamp(0, 7),
    )
    normal = (1.0 + mantissa / 8.0) * torch.exp2(normal_exp)

    return torch.where(
        abs_scaled < _FP8_E4M3FN_SUBNORMAL_NORMAL_MIDPOINT,
        subnormal,
        normal,
    )


def _fp8_e4m3fn_qdq(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if x.numel() == 0:
        return x

    orig_shape = x.shape
    last_dim = orig_shape[-1]
    assert last_dim % block_size == 0
    x_view = x.float().reshape(-1, last_dim)
    blocks = x_view.reshape(-1, last_dim // block_size, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.pow(
        torch.full((), 2.0, dtype=torch.float32, device=x.device),
        torch.ceil(torch.log2(amax / 448.0)),
    )
    scaled = (blocks / scale).clamp(-448.0, 448.0)

    quantized_abs = _fp8_e4m3fn_quantized_abs(scaled.abs())
    qdq = torch.where(scaled < 0, -quantized_abs, quantized_abs) * scale
    return qdq.reshape(orig_shape).to(x.dtype)


def _fp8_qdq_nope_dims(
    kv: torch.Tensor,
    nope_head_dim: int,
    block_size: int = 64,
) -> torch.Tensor:
    if nope_head_dim <= 0:
        return kv
    kv_nope = _fp8_e4m3fn_qdq(kv[..., :nope_head_dim], block_size)
    return torch.cat([kv_nope, kv[..., nope_head_dim:]], dim=-1)


class DeepseekV4DSparkAttention(DeepseekV4Attention):
    """DSpark sliding-window attention with an internal eager context cache."""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs["vllm_config"]
        config = kwargs["config"]
        super().__init__(*args, **kwargs)
        self.compress_ratio = 0
        self.dsa_attn.compress_ratio = 0
        self.block_size = int(config.dspark_block_size)
        cache_capacity = _dspark_cache_capacity(
            vllm_config,
            self.block_size,
            self.window_size if self.window_size is not None else None,
        )
        max_request_slots = _dspark_max_request_slots(vllm_config)
        cache_shape = (max_request_slots, cache_capacity, self.n_local_heads, self.head_dim)
        self.register_buffer(
            "_dspark_k_cache",
            torch.empty(
                cache_shape,
                dtype=vllm_config.model_config.dtype,
                device=current_platform.device_type,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_dspark_v_cache",
            torch.empty(
                cache_shape,
                dtype=vllm_config.model_config.dtype,
                device=current_platform.device_type,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_dspark_cache_valid",
            torch.zeros((max_request_slots, cache_capacity), dtype=torch.bool, device=current_platform.device_type),
            persistent=False,
        )
        self.register_buffer(
            "_dspark_cache_positions",
            torch.full(
                (max_request_slots, cache_capacity),
                -1,
                dtype=torch.int32,
                device=current_platform.device_type,
            ),
            persistent=False,
        )
        self._dspark_cache_capacity = cache_capacity
        self._dspark_max_request_slots = max_request_slots

    def _ensure_dspark_cache(self, length: int, like: torch.Tensor) -> None:
        del like
        if length > self._dspark_cache_capacity:
            raise ValueError(
                "DSpark attention cache position exceeds preallocated capacity: "
                f"length={length}, capacity={self._dspark_cache_capacity}"
            )

    def reset_request_slots(self, request_slots: torch.Tensor | None) -> None:
        if request_slots is None or request_slots.numel() == 0:
            return
        slots = torch.unique(request_slots.to(torch.long))
        if slots.numel() == 0:
            return
        assert self._dspark_cache_valid is not None
        assert self._dspark_cache_positions is not None
        self._dspark_cache_valid[slots] = False
        self._dspark_cache_positions[slots] = -1

    def _project_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._expand_private_kv(self._project_shared_kv(hidden_states, positions))

    def _project_shared_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        kv = self.kv_norm(_linear_output(self.wkv, hidden_states))
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k_pe = _apply_dsv4_rope(self.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        return torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, self.head_dim).contiguous()

    def _expand_private_kv(self, shared_kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv = shared_kv.squeeze(1)
        kv = _fp8_qdq_nope_dims(kv, self.nope_head_dim)
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k = torch.cat(
            [
                k_nope.unsqueeze(1).expand(-1, self.n_local_heads, -1),
                k_pe.unsqueeze(1).expand(-1, self.n_local_heads, -1),
            ],
            dim=-1,
        ).contiguous()
        v = kv.unsqueeze(1).expand(-1, self.n_local_heads, -1).contiguous()
        return k, v

    def _store_standard_swa_kv(
        self,
        shared_kv: torch.Tensor,
        slot_mapping: torch.Tensor | None,
    ) -> None:
        if not _dspark_standard_dsa_enabled():
            return
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
        if slot_mapping.ndim == 1:
            slot_mapping = DeviceOperator.format_dsa_slot_mapping(slot_mapping, swa_cache_layer.block_size)
        DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, shared_kv, slot_mapping)

    _store_standard_swa_context_kv = _store_standard_swa_kv

    def _run_dspark_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        request_slots: torch.Tensor | None,
    ) -> torch.Tensor:
        if positions.numel() == 0:
            return torch.empty_like(q)
        if request_slots is None:
            request_slots = torch.zeros_like(positions, dtype=torch.int32)
        if request_slots.numel() != positions.numel():
            raise ValueError(
                "DSpark request_slots length must match query positions: "
                f"request_slots={request_slots.numel()}, positions={positions.numel()}"
            )

        assert self._dspark_k_cache is not None
        assert self._dspark_v_cache is not None
        assert self._dspark_cache_valid is not None
        assert self._dspark_cache_positions is not None
        return dspark_attention(
            q,
            self._dspark_k_cache,
            self._dspark_v_cache,
            self._dspark_cache_positions,
            self._dspark_cache_valid,
            k,
            v,
            request_slots,
            positions,
            self.attn_sink[: self.n_local_heads],
            self.block_size,
            int(self.window_size),
            float(self.scale),
            shared_kv=True,
        )

    def precompute_context_kv(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
        request_slots: torch.Tensor | None = None,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        if positions.numel() == 0:
            return
        shared_kv = self._project_shared_kv(main_x, positions)
        k, v = self._expand_private_kv(shared_kv)
        max_pos = int(positions.max().item())
        self._ensure_dspark_cache(min(max_pos + 1, self._dspark_cache_capacity), k)
        assert self._dspark_k_cache is not None
        assert self._dspark_v_cache is not None
        assert self._dspark_cache_valid is not None
        assert self._dspark_cache_positions is not None
        if request_slots is None:
            request_slots = torch.zeros_like(positions, dtype=torch.int32)
        slots_long = request_slots.to(torch.long)
        if slots_long.numel() != positions.numel():
            raise ValueError(
                "DSpark request_slots length must match context positions: "
                f"request_slots={slots_long.numel()}, positions={positions.numel()}"
            )
        if int(slots_long.max().item()) >= self._dspark_max_request_slots:
            raise ValueError(
                "DSpark request slot exceeds preallocated cache slots: "
                f"slot={int(slots_long.max().item())}, capacity={self._dspark_max_request_slots}"
            )
        pos_long = positions.to(torch.long)
        cache_indices = pos_long % self._dspark_cache_capacity
        self._dspark_k_cache[slots_long, cache_indices] = k
        self._dspark_v_cache[slots_long, cache_indices] = v
        self._dspark_cache_positions[slots_long, cache_indices] = positions.to(torch.int32)
        self._dspark_cache_valid[slots_long, cache_indices] = True
        self._store_standard_swa_kv(shared_kv, context_slot_mapping)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _dspark_standard_dsa_enabled():
            return self.dsa_attn(positions, hidden_states, llama_4_scaling)

        del llama_4_scaling
        qr = self.q_norm(_linear_output(self.wq_a, hidden_states))
        kv = self.kv_norm(_linear_output(self.wkv, hidden_states))

        q = _linear_output(self.wq_b, qr).view(-1, self.n_local_heads, self.head_dim)
        q = self.q_norm_without_weight(q)
        q_nope, q_pe = q.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = _apply_dsv4_rope(self.rotary_emb, positions, q_pe)
        k_pe = _apply_dsv4_rope(self.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        shared_kv = torch.cat([k_nope, k_pe], dim=-1).view(-1, 1, self.head_dim).contiguous()
        kv = _fp8_qdq_nope_dims(shared_kv.squeeze(1), self.nope_head_dim)
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)

        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat(
            [
                k_nope.unsqueeze(1).expand(-1, self.n_local_heads, -1),
                k_pe.unsqueeze(1).expand(-1, self.n_local_heads, -1),
            ],
            dim=-1,
        ).contiguous()
        v = kv.unsqueeze(1).expand(-1, self.n_local_heads, -1).contiguous()
        self._store_standard_swa_kv(shared_kv, slot_mapping)
        attn_out = self._run_dspark_attention(q, k, v, positions, request_slots)

        attn_out = _apply_dsv4_rope_tail(
            self.rotary_emb,
            positions,
            attn_out,
            inverse=True,
        )
        group_dim = self.n_local_heads * self.head_dim // self.n_local_groups
        attn_out = attn_out.reshape(-1, self.n_local_groups, group_dim)
        attn_out = _fp8_e4m3fn_qdq(attn_out, 128)
        wo_a = _wo_a_weight_for_eager_projection(
            self.wo_a.weight,
            self.n_local_groups,
            self.o_lora_rank,
            group_dim,
        )
        z = _grouped_wo_a_projection(attn_out, wo_a).flatten(1)
        return _linear_output(self.wo_b, z)


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
            quant_config_override=_draft_quant_config(vllm_config),
            use_quant_config_override=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        llama_4_scaling: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del residual, llama_4_scaling
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions,
            hidden_states,
            None,
            request_slots=request_slots,
            slot_mapping=slot_mapping,
        )
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, input_ids)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        return hidden_states


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
            params_dtype=torch.float32,
            org_num_embeddings=config.vocab_size,
            prefix=f"{prefix}.markov_w2",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.markov_w2, markov_embed)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embed(token_ids)
        return self.bias(embed), embed


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
                    prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx + idx}"),
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

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.dsa_attn.swa_cache_layer.prefix for layer in self.layers.values()]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | tuple[torch.Tensor | None, ...] | None = None,
        context_request_slots: torch.Tensor | None = None,
    ) -> None:
        if context_states.numel() == 0:
            return
        main_x = self.main_norm(_linear_output(self.main_proj, context_states))
        for layer_idx, layer in enumerate(self.layers.values()):
            layer_context_slot_mapping = context_slot_mapping
            if isinstance(context_slot_mapping, (list, tuple)):
                layer_context_slot_mapping = context_slot_mapping[layer_idx]
            layer.self_attn.precompute_context_kv(
                main_x,
                context_positions,
                request_slots=context_request_slots,
                context_slot_mapping=layer_context_slot_mapping,
            )

    def reset_request_slots(self, request_slots: torch.Tensor | None) -> None:
        for layer in self.layers.values():
            layer.self_attn.reset_request_slots(request_slots)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hidden_states
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        for layer in self.layers.values():
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                input_ids=input_ids,
                request_slots=request_slots,
                slot_mapping=slot_mapping,
            )
        return self.compute_head_hidden(hidden_states)

    def compute_head_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
        self.set_moe_parameters_from_layers(self.config, self.model.layers.values())

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del intermediate_tensors, spec_step_idx
        assert input_ids is not None
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            hidden_states=hidden_states,
            request_slots=request_slots,
            slot_mapping=slot_mapping,
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

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | tuple[torch.Tensor | None, ...] | None = None,
        context_request_slots: torch.Tensor | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states,
            context_positions,
            context_slot_mapping,
            context_request_slots,
        )

    def reset_request_slots(self, request_slots: torch.Tensor | None) -> None:
        self.model.reset_request_slots(request_slots)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("mlp.gate_up_proj", "mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "mlp.up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

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
            if name == "embed.weight":
                embed_name = "model.embed_tokens.weight"
                param = params_dict[embed_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(embed_name)
                continue
            if name == "head.weight":
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
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(mapped)
                break
            else:
                if ".experts." in name:
                    if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for param_name, weight_name, expert_id, expert_shard_id in expert_mapping:
                        if weight_name not in name:
                            continue
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
                    continue
                if "attn_sink" in name:
                    if name not in params_dict:
                        continue
                    narrow = loaded_weight[head_start:head_end]
                    with torch.no_grad():
                        params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

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
        return name


DSparkDeepseekV4ForCausalLM = DeepSeekV4DSparkMTP
