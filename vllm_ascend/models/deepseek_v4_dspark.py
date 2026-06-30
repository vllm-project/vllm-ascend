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
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.deepseek_v4 import (
    DeepseekV2MixtureOfExperts,
    DeepseekV4Attention,
    DeepseekV4MoE,
    _hc_post_torch,
    _hc_pre_torch,
    _use_torch_hc_fallback,
)
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.23.0"):
    from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping

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


def _linear(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = layer(x)
    return out[0] if isinstance(out, tuple) else out


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


def _headwise_scores(q: torch.Tensor, k_ctx: torch.Tensor) -> torch.Tensor:
    return (q.float().unsqueeze(0) * k_ctx.float()).sum(dim=-1).transpose(0, 1)


def _headwise_weighted_sum(
    probs: torch.Tensor,
    v_ctx: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return (probs.transpose(0, 1).unsqueeze(-1) * v_ctx.float()).sum(dim=0).to(dtype)


def _grouped_wo_a_projection(
    attn_out: torch.Tensor,
    wo_a: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(attn_out.transpose(0, 1), wo_a.transpose(1, 2)).transpose(0, 1)


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


def _apply_dsv4_rope_tail(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    x: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    rotary_dim = rotary_emb.rotary_dim
    if x.shape[-1] == rotary_dim:
        return _apply_dsv4_rope(rotary_emb, positions, x, inverse=inverse)
    x_pass, x_rot = x[..., :-rotary_dim], x[..., -rotary_dim:]
    x_rot = _apply_dsv4_rope(rotary_emb, positions, x_rot, inverse=inverse)
    return torch.cat([x_pass, x_rot], dim=-1)


def _hc_head(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    shape, dtype = x.size(), x.dtype
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=1)
    return y.to(dtype)


class DeepseekV4DSparkAttention(DeepseekV4Attention):
    """DSpark sliding-window attention with an internal eager context cache."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compress_ratio = 0
        self.dsa_attn.compress_ratio = 0
        self._dspark_k_cache: torch.Tensor | None = None
        self._dspark_v_cache: torch.Tensor | None = None
        self._dspark_cache_valid: torch.Tensor | None = None

    def _ensure_dspark_cache(self, length: int, like: torch.Tensor) -> None:
        current = 0
        if self._dspark_k_cache is not None:
            current = self._dspark_k_cache.shape[0]
        if current >= length:
            return
        new_len = max(length, max(16, current * 2))
        shape = (new_len, self.n_local_heads, self.head_dim)
        k_cache = torch.empty(shape, dtype=like.dtype, device=like.device)
        v_cache = torch.empty(shape, dtype=like.dtype, device=like.device)
        valid = torch.zeros(new_len, dtype=torch.bool, device=like.device)
        if self._dspark_k_cache is not None:
            assert self._dspark_v_cache is not None
            assert self._dspark_cache_valid is not None
            k_cache[:current].copy_(self._dspark_k_cache)
            v_cache[:current].copy_(self._dspark_v_cache)
            valid[:current].copy_(self._dspark_cache_valid)
        self._dspark_k_cache = k_cache
        self._dspark_v_cache = v_cache
        self._dspark_cache_valid = valid

    def _project_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv = self.kv_norm(_linear(self.wkv, hidden_states))
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k_pe = _apply_dsv4_rope(self.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        kv = torch.cat([k_nope, k_pe], dim=-1)
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

    def precompute_context_kv(
        self,
        main_x: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if positions.numel() == 0:
            return
        k, v = self._project_kv(main_x, positions)
        max_pos = int(positions.max().item())
        self._ensure_dspark_cache(max_pos + 1, k)
        assert self._dspark_k_cache is not None
        assert self._dspark_v_cache is not None
        assert self._dspark_cache_valid is not None
        if int(positions.min().item()) == 0:
            self._dspark_cache_valid.zero_()
        pos_long = positions.to(torch.long)
        self._dspark_k_cache.index_copy_(0, pos_long, k)
        self._dspark_v_cache.index_copy_(0, pos_long, v)
        self._dspark_cache_valid[pos_long] = True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del llama_4_scaling
        qr = self.q_norm(_linear(self.wq_a, hidden_states))
        kv = self.kv_norm(_linear(self.wkv, hidden_states))

        q = _linear(self.wq_b, qr).view(-1, self.n_local_heads, self.head_dim)
        q = self.q_norm_without_weight(q)
        q_nope, q_pe = q.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = _apply_dsv4_rope(self.rotary_emb, positions, q_pe)
        k_pe = _apply_dsv4_rope(self.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        kv = torch.cat([k_nope, k_pe], dim=-1)
        kv = _fp8_qdq_nope_dims(kv, self.nope_head_dim)
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

        if positions.numel() == 0:
            attn_out = torch.empty_like(q)
        else:
            pos_long = positions.to(torch.long)
            block_start = int(pos_long.min().item())
            context_end = block_start - 1
            context_start = max(0, context_end + 1 - self.window_size)
            if context_end >= context_start and self._dspark_cache_valid is not None:
                assert self._dspark_k_cache is not None
                assert self._dspark_v_cache is not None
                valid = self._dspark_cache_valid[context_start : context_end + 1]
                k_ctx = self._dspark_k_cache[context_start : context_end + 1][valid]
                v_ctx = self._dspark_v_cache[context_start : context_end + 1][valid]
                k_ctx = torch.cat([k_ctx, k], dim=0)
                v_ctx = torch.cat([v_ctx, v], dim=0)
            else:
                k_ctx = k
                v_ctx = v

            scores = torch.einsum("qhd,khd->qhk", q.float(), k_ctx.float()) * self.scale
            sink = self.attn_sink[: self.n_local_heads].float().view(1, -1, 1)
            scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
            exp_scores = torch.exp(scores - scores_max)
            probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max))
            attn_out = torch.einsum("qhk,khd->qhd", probs, v_ctx.float()).to(q.dtype)

        attn_out = _apply_dsv4_rope_tail(
            self.rotary_emb,
            positions,
            attn_out,
            inverse=True,
        )
        group_dim = self.n_local_heads * self.head_dim // self.n_local_groups
        attn_out = attn_out.reshape(-1, self.n_local_groups, group_dim)
        attn_out = _fp8_e4m3fn_qdq(attn_out, 128)
        wo_a_weight = self.wo_a.weight
        if wo_a_weight.ndim == 3:
            # Ascend's wo_a loader stores weights as [group, group_dim, rank]
            # for the main DSA path. DSpark's eager projection needs the
            # original [group, rank, group_dim] layout.
            wo_a = wo_a_weight.transpose(1, 2).contiguous()
        else:
            wo_a = wo_a_weight.view(self.n_local_groups, self.o_lora_rank, group_dim)
        z = _grouped_wo_a_projection(attn_out, wo_a).flatten(1)
        return _linear(self.wo_b, z)


class DeepseekV4DSparkDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = _draft_quant_config(vllm_config)
        parallel_config = vllm_config.parallel_config
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.split(sep=".")[-1])
        self.norm_eps = config.rms_norm_eps
        self.self_attn = DeepseekV4DSparkAttention(
            vllm_config=vllm_config,
            config=config,
            max_position_embeddings=config.rope_parameters["original_max_position_embeddings"],
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=None,
        )
        self.mlp = DeepseekV4MoE(
            config=config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            is_draft_layer=True,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.use_torch_hc = _use_torch_hc_fallback(vllm_config)

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_torch_hc:
            return _hc_pre_torch(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                self.hc_mult,
                self.hc_sinkhorn_iters,
                self.norm_eps,
                self.hc_eps,
            )
        return torch.ops._C_ascend.npu_hc_pre_v2(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.norm_eps,
            self.hc_eps,
        )

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_torch_hc:
            return _hc_post_torch(x, residual, post, comb)
        return torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(dim=0),
            residual.unsqueeze(dim=0),
            post.unsqueeze(dim=0),
            comb.unsqueeze(dim=0),
        ).squeeze(dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, None)
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

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.markov_w1(token_ids)
        logits = self.logits_processor(self.markov_w2, embed)
        return logits, embed


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int, prefix: str) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(
            input_dim,
            1,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            return_bias=False,
            prefix=f"{prefix}.proj",
        )

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat([hidden, markov_embed], dim=-1)
        return _linear(self.proj, hidden.float()).squeeze(-1)


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
        self.num_dspark_layers = int(getattr(config, "dspark_num_mtp_layers", 3))
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
        self.confidence_head = DSparkConfidenceHead(
            config.hidden_size + config.dspark_markov_rank,
            maybe_prefix(prefix, f"layers.{last_layer_idx}.confidence_head"),
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
        last_layer.confidence_head = self.confidence_head
        last_layer.hc_head_fn = self.hc_head_fn
        last_layer.hc_head_base = self.hc_head_base
        last_layer.hc_head_scale = self.hc_head_scale
        self._last_input_ids: torch.Tensor | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        del context_slot_mapping
        if context_states.numel() == 0:
            return
        main_x = self.main_norm(_linear(self.main_proj, context_states))
        for layer in self.layers.values():
            layer.self_attn.precompute_context_kv(main_x, context_positions)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hidden_states
        self._last_input_ids = input_ids
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        for layer in self.layers.values():
            hidden_states = layer(hidden_states, positions, input_ids)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        logits_processor: LogitsProcessor,
    ) -> torch.Tensor:
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.view(-1, self.hc_mult, self.hidden_size)
        dense_hidden = _hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.config.rms_norm_eps,
            self.config.hc_eps,
        )
        base_logits = logits_processor(lm_head, self.norm(dense_hidden))

        if self._last_input_ids is None or hidden_states.shape[0] % self.block_size != 0:
            return base_logits

        batch = hidden_states.shape[0] // self.block_size
        base_logits = base_logits.view(batch, self.block_size, -1)
        input_ids = self._last_input_ids[: batch * self.block_size].view(batch, self.block_size)
        prev_ids = input_ids[:, 0]
        logits_out = []
        for idx in range(self.block_size):
            markov_logits, _ = self.markov_head(prev_ids)
            logits_i = base_logits[:, idx, :] + markov_logits
            logits_out.append(logits_i)
            prev_ids = logits_i.argmax(dim=-1)
        return torch.stack(logits_out, dim=1).reshape(batch * self.block_size, -1)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        if vllm_version_is("0.23.0"):
            return FusedMoE.make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
            )
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
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
        self.expert_weights: typing.MutableSequence[typing.Sequence[torch.Tensor]] = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV4MoE] = []
        example_moe = None
        for layer in self.model.layers.values():
            if isinstance(layer.mlp, DeepseekV4MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)

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
    ) -> torch.Tensor:
        del intermediate_tensors, spec_step_idx
        assert input_ids is not None
        return self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            hidden_states=hidden_states,
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

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
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
            f"model.layers.{last_layer_idx}.confidence_head.proj.weight",
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
