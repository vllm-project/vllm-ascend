# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import typing
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch_npu
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_ascend.utils import enable_dsa_cp

from .deepseek_v4 import (
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

def _draft_quant_config(vllm_config: VllmConfig):
    return vllm_config.quant_config


def _get_dspark_num_layers(config: PretrainedConfig) -> int:
    for attr in ("dspark_num_layers", "n_mtp_layers", "dspark_num_mtp_layers"):
        value = getattr(config, attr, None)
        if value:
            return int(value)
    return 3


def _dspark_cache_capacity(vllm_config: VllmConfig, block_size: int, window_size: int) -> int:
    model_config = getattr(vllm_config, "model_config", None)
    max_model_len = int(getattr(model_config, "max_model_len", 0) or 0)
    return max(block_size, window_size + block_size, max_model_len + block_size)


def _dspark_max_request_slots(vllm_config: VllmConfig) -> int:
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    return max(1, int(getattr(scheduler_config, "max_num_seqs", 1) or 1))


def _recipe_sparse_sharedkv_attention(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    block_size: int,
    window_size: int,
    sparse_attn_ops,
) -> torch.Tensor | None:
    if q.device.type == "cpu" or sparse_attn_ops is None:
        return None
    attn_op, metadata_op = sparse_attn_ops

    cu_seqlens_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device)
    cu_seqlens_kv = torch.tensor([0, packed_kv.shape[0]], dtype=torch.int32, device=q.device)
    metadata = metadata_op(
        num_heads_q=int(q.shape[1]),
        num_heads_kv=1,
        head_dim=int(q.shape[2]),
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=None,
        seqused_kv=None,
        batch_size=1,
        max_seqlen_q=int(q.shape[0]),
        max_seqlen_kv=int(packed_kv.shape[0]),
        ori_topk=0,
        cmp_topk=0,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=int(window_size + block_size - 1),
        ori_win_right=int(block_size - 1),
        layout_q="TND",
        layout_kv="TND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device=str(q.device),
    )
    return attn_op(
        q,
        ori_kv=packed_kv,
        cmp_kv=None,
        ori_sparse_indices=None,
        cmp_sparse_indices=None,
        ori_block_table=None,
        cmp_block_table=None,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_kv,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=None,
        sinks=attn_sink[: q.shape[1]].float().contiguous(),
        metadata=metadata,
        softmax_scale=float(softmax_scale),
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=int(window_size + block_size - 1),
        ori_win_right=int(block_size - 1),
        layout_q="TND",
        layout_kv="TND",
        return_softmax_lse=False,
    )[0]


def _torch_sharedkv_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    scores = torch.einsum("qhd,kd->qhk", q.float(), kv.float()) * softmax_scale
    sink = attn_sink[: q.shape[1]].float().view(1, q.shape[1], 1)
    scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
    exp_scores = torch.exp(scores - scores_max)
    probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max))
    return torch.einsum("qhk,kd->qhd", probs, kv.float()).to(q.dtype)


class DeepseekV4DSparkAttention(DeepseekV4Attention):
    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs["vllm_config"]
        config = kwargs["config"]
        super().__init__(*args, **kwargs)
        self.compress_ratio = 0
        self.dsa_attn.compress_ratio = 0
        self.block_size = int(getattr(config, "dspark_block_size", getattr(config, "n_predict", 5)) or 5)
        self.window_size = int(self.window_size)
        cache_capacity = _dspark_cache_capacity(vllm_config, self.block_size, self.window_size)
        max_request_slots = _dspark_max_request_slots(vllm_config)
        cache_shape = (max_request_slots, cache_capacity, self.head_dim)
        self.register_buffer("_dspark_kv_cache", torch.empty(cache_shape, dtype=vllm_config.model_config.dtype, device=self.attn_sink.device), persistent=False)
        self.register_buffer("_dspark_cache_valid", torch.zeros((max_request_slots, cache_capacity), dtype=torch.bool, device=self.attn_sink.device), persistent=False)
        self.register_buffer("_dspark_cache_positions", torch.full((max_request_slots, cache_capacity), -1, dtype=torch.int32, device=self.attn_sink.device), persistent=False)
        self._dspark_cache_capacity = cache_capacity
        self._dspark_max_request_slots = max_request_slots
        self._dspark_sparse_attn_ops = None
        self._dspark_sparse_attn_unavailable_warned = False

    def _get_dspark_sparse_attn_ops(self):
        if self._dspark_sparse_attn_ops is not None:
            return self._dspark_sparse_attn_ops
        try:
            attn_op = torch.ops.custom.npu_sparse_attn_sharedkv
            metadata_op = torch.ops.custom.npu_sparse_attn_sharedkv_metadata
        except (AttributeError, RuntimeError) as exc:
            if not self._dspark_sparse_attn_unavailable_warned:
                logger.warning("Failed to get custom DSpark sparse sharedkv ops: %s", exc)
                self._dspark_sparse_attn_unavailable_warned = True
            return None
        self._dspark_sparse_attn_ops = attn_op, metadata_op
        return self._dspark_sparse_attn_ops

    def reset_request_slots(self, request_slots: torch.Tensor | None) -> None:
        if request_slots is None or request_slots.numel() == 0:
            return
        slots = torch.unique(request_slots.to(device=self._dspark_cache_valid.device, dtype=torch.long))
        self._dspark_cache_valid[slots] = False
        self._dspark_cache_positions[slots] = -1

    def _project_shared_kv(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        kv = self.kv_norm(_linear_output(self.wkv, hidden_states))
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k_pe = _apply_dsv4_rope(self.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        return torch.cat([k_nope, k_pe], dim=-1).contiguous()

    def precompute_context_kv(self, main_x: torch.Tensor, positions: torch.Tensor, request_slots: torch.Tensor | None) -> None:
        if positions.numel() == 0:
            return
        shared_kv = self._project_shared_kv(main_x, positions)
        if request_slots is None:
            request_slots = torch.zeros_like(positions, dtype=torch.int32)
        slots_long = request_slots.to(device=shared_kv.device, dtype=torch.long)
        pos_long = positions.to(device=shared_kv.device, dtype=torch.long)
        cache_indices = pos_long % self._dspark_cache_capacity
        self._dspark_kv_cache[slots_long, cache_indices] = shared_kv
        self._dspark_cache_positions[slots_long, cache_indices] = positions.to(torch.int32)
        self._dspark_cache_valid[slots_long, cache_indices] = True

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        request_slots: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del llama_4_scaling, slot_mapping
        qr = self.q_norm(_linear_output(self.wq_a, hidden_states))
        kv = self.kv_norm(_linear_output(self.wkv, hidden_states))
        q = _linear_output(self.wq_b, qr).view(-1, self.n_local_heads, self.head_dim)
        q = self.q_norm_without_weight(q)
        q_nope, q_pe = q.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        k_nope, k_pe = kv.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = _apply_dsv4_rope(self.rotary_emb, positions, q_pe)
        k_pe = _apply_dsv4_rope(self.rotary_emb, positions, k_pe.unsqueeze(1)).squeeze(1)
        shared_kv = torch.cat([k_nope, k_pe], dim=-1).contiguous()
        q = torch.cat([q_nope, q_pe], dim=-1)
        if request_slots is None:
            request_slots = torch.zeros_like(positions, dtype=torch.int32)

        out = torch.empty_like(q)
        pos_long = positions.to(torch.long)
        slots_long = request_slots.to(torch.long)
        for block_offset in range(0, positions.numel(), self.block_size):
            block_end = min(block_offset + self.block_size, positions.numel())
            block_pos = pos_long[block_offset:block_end]
            block_start = int(block_pos.min().item())
            context_end = block_start - 1
            context_start = max(0, context_end + 1 - self.window_size)
            request_slot = int(slots_long[block_offset].item())
            ctx_positions = torch.arange(context_start, context_end + 1, dtype=torch.long, device=q.device)
            if ctx_positions.numel() > 0:
                cache_indices = ctx_positions % self._dspark_cache_capacity
                cached_positions = self._dspark_cache_positions[request_slot, cache_indices].to(torch.long)
                valid = self._dspark_cache_valid[request_slot, cache_indices] & (cached_positions == ctx_positions)
                ctx_kv = self._dspark_kv_cache[request_slot, cache_indices][valid]
            else:
                ctx_kv = shared_kv.new_empty((0, shared_kv.shape[-1]))
            packed_kv = torch.cat([ctx_kv, shared_kv[block_offset:block_end]], dim=0).contiguous()
            op_out = _recipe_sparse_sharedkv_attention(
                q[block_offset:block_end].contiguous(),
                packed_kv[:, None, :].contiguous(),
                self.attn_sink,
                float(self.scale),
                self.block_size,
                self.window_size,
                self._get_dspark_sparse_attn_ops(),
            )
            if op_out is None:
                op_out = _torch_sharedkv_attention(q[block_offset:block_end], packed_kv, self.attn_sink, float(self.scale))
            out[block_offset:block_end] = op_out

        out = _apply_dsv4_rope_tail(self.rotary_emb, positions, out, inverse=True)
        group_dim = self.n_local_heads * self.head_dim // self.n_local_groups
        out = out.reshape(-1, self.n_local_groups, group_dim)
        if hasattr(self.wo_a, "weight_scale") and self.wo_a.weight.dtype == torch.float8_e4m3fn:
            out, out_scale = torch_npu.npu_dynamic_mx_quant(out, dst_type=torch.float8_e4m3fn)
            # Recipe A5 MXFP8 o-proj path: 32 is the MX scale group size on
            # the innermost dimension; the permutations compute grouped
            # [groups, tokens, dim] x [groups, dim, rank] -> [tokens, groups, rank].
            z = torch_npu.npu_transpose_quant_batchmatmul(
                out,
                self.wo_a.weight,
                dtype=torch.bfloat16,
                bias=None,
                group_sizes=(0, 0, 32),
                x1_scale=out_scale.view(torch.float8_e8m0fnu),
                x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            ).flatten(1)
        else:
            wo_a = _wo_a_weight_for_eager_projection(self.wo_a.weight, self.n_local_groups, self.o_lora_rank, group_dim)
            z = _grouped_wo_a_projection(out, wo_a).flatten(1)
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
        hidden_states = self.self_attn(positions, hidden_states, None, request_slots=request_slots, slot_mapping=slot_mapping)
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
        self.markov_w1 = VocabParallelEmbedding(config.vocab_size, config.dspark_markov_rank, prefix=f"{prefix}.markov_w1")
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


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int, prefix: str) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(input_dim, 1, bias=False, params_dtype=torch.float32, quant_config=None, prefix=f"{prefix}.proj")

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        return _linear_output(self.proj, torch.cat([hidden, markov_embed], dim=-1).float()).squeeze(-1)


class DeepseekV4DSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.block_size = int(getattr(config, "dspark_block_size", getattr(config, "n_predict", 5)) or 5)
        self.noise_token_id = int(getattr(config, "dspark_noise_token_id", getattr(config, "ptd_token_id", 0)) or 0)
        self.target_layer_ids = list(getattr(config, "dspark_target_layer_ids", []) or [])
        self.num_dspark_layers = _get_dspark_num_layers(config)
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.embed_tokens = None
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
            config.hidden_size * max(1, len(self.target_layer_ids)),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=_draft_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx}.main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        first_layer.main_proj = self.main_proj
        first_layer.main_norm = self.main_norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        last_layer_idx = self.mtp_start_layer_idx + self.num_dspark_layers - 1
        self.markov_head = DSparkMarkovHead(config, maybe_prefix(prefix, f"layers.{last_layer_idx}.markov_head"))
        self.confidence_head = DSparkConfidenceHead(
            config.hidden_size + config.dspark_markov_rank,
            maybe_prefix(prefix, f"layers.{last_layer_idx}.confidence_head"),
        )
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False)
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
        last_layer = self.layers[str(last_layer_idx)]
        last_layer.norm = self.norm
        last_layer.markov_head = self.markov_head
        last_layer.confidence_head = self.confidence_head
        last_layer.hc_head_fn = self.hc_head_fn
        last_layer.hc_head_base = self.hc_head_base
        last_layer.hc_head_scale = self.hc_head_scale

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise AttributeError("DSpark draft model requires shared target embed_tokens.")
        return self.embed_tokens(input_ids)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping=None,
        context_request_slots: torch.Tensor | None = None,
    ) -> None:
        del context_slot_mapping
        if context_states.numel() == 0:
            return
        main_x = self.main_norm(_linear_output(self.main_proj, context_states))
        for layer in self.layers.values():
            layer.self_attn.precompute_context_kv(main_x, context_positions, context_request_slots)

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
            if self.embed_tokens is None:
                raise AttributeError("DSpark draft model requires shared target embed_tokens.")
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        for layer in self.layers.values():
            hidden_states = layer(positions=positions, hidden_states=hidden_states, input_ids=input_ids, request_slots=request_slots, slot_mapping=slot_mapping)
        return hidden_states

    def compute_head_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 2:
            return hidden_states
        return _hc_head_torch(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.config.rms_norm_eps, self.config.hc_eps)

    def compute_logits(self, hidden_states: torch.Tensor, lm_head: ParallelLMHead, logits_processor: LogitsProcessor) -> torch.Tensor:
        return logits_processor(lm_head, self.norm(self.compute_head_hidden(hidden_states)))

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.markov_head.bias(markov_embed)

    def confidence(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.confidence_head(hidden, markov_embed)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return _make_deepseek_v4_expert_params_mapping(self, num_experts=self.config.n_routed_experts)

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.layers.values():
            finalize = getattr(layer.mlp, "finalize_mega_moe_weights", None)
            if finalize is not None:
                finalize()


@support_torch_compile
class DeepSeekV4DSparkMTP(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = _draft_quant_config(vllm_config)
        self.has_own_embed_tokens = False
        self.model = DeepseekV4DSparkModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.set_moe_parameters_from_layers(iter(self.model.layers.values()))

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
        return self.model(input_ids=input_ids, positions=positions, inputs_embeds=inputs_embeds, hidden_states=hidden_states, request_slots=request_slots, slot_mapping=slot_mapping)

    def compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0) -> torch.Tensor | None:
        del spec_step_idx
        if self.lm_head is None:
            raise AttributeError("DSpark draft model requires shared target lm_head.")
        return self.model.compute_logits(hidden_states, self.lm_head, self.logits_processor)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_bias(markov_embed)

    def confidence(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.confidence(hidden, markov_embed)

    def precompute_and_store_context_kv(self, context_states, context_positions, context_slot_mapping=None, context_request_slots=None) -> None:
        self.model.precompute_and_store_context_kv(context_states, context_positions, context_slot_mapping, context_request_slots)

    def reset_request_slots(self, request_slots: torch.Tensor | None) -> None:
        self.model.reset_request_slots(request_slots)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1)]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        skipped_params: set[str] = set()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank
        head_end = head_start + heads_per_rank
        expert_mapping = self.model.get_expert_mapping()
        start_layer_idx = self.config.num_hidden_layers
        last_layer_idx = start_layer_idx + self.model.num_dspark_layers - 1

        for name, loaded_weight in weights:
            if name in ("embed.weight", "head.weight"):
                continue
            mapped_name = self._map_dspark_weight_name(name)
            if mapped_name is None:
                continue
            name = mapped_name

            if name.startswith(f"model.layers.{last_layer_idx}.hc_head_"):
                canonical_name = name.replace(f"model.layers.{last_layer_idx}.", "model.", 1)
                if canonical_name in params_dict:
                    name = canonical_name
            if name.endswith(".scale"):
                name = name.replace(".scale", ".weight_scale")

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
                        success = weight_loader(param, loaded_weight, mapped, shard_id=expert_shard_id, expert_id=expert_id, return_success=True)
                        if success:
                            loaded_params.add(mapped)
                            break
                    continue
                if "attn_sink" in name:
                    if name not in params_dict:
                        skipped_params.add(name)
                        continue
                    narrow = loaded_weight[head_start:head_end] if not enable_dsa_cp() else loaded_weight
                    with torch.no_grad():
                        params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if name not in params_dict:
                    skipped_params.add(name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

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
            raise ValueError(f"DSpark speculative decoding required weights missing from checkpoint load: {missing_required}")
        missing_params = set(params_dict) - loaded_params
        optional_keys = (
            "q_norm_without_weight.",
            "rotary_emb.inv_freq",
        )
        missing_params = {name for name in missing_params if not any(key in name for key in optional_keys)}
        if missing_params:
            logger.warning("DSpark weights not initialized from checkpoint: %s", sorted(missing_params))
        if skipped_params:
            logger.warning("DSpark checkpoint weights skipped by name mapping: %s", sorted(skipped_params))
        self.model.finalize_mega_moe_weights()
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _map_dspark_weight_name(self, name: str) -> str | None:
        if "rotary_emb.inv_freq" in name:
            return None
        if not name.startswith("mtp."):
            return None

        parts = name.split(".", 2)
        if len(parts) != 3 or not parts[1].isdigit():
            return None
        stage_idx = int(parts[1])
        if stage_idx >= self.model.num_dspark_layers:
            return None
        layer_idx = self.config.num_hidden_layers + stage_idx
        suffix = parts[2]

        name = f"model.layers.{layer_idx}.{suffix}"
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".w1.", ".gate_proj.")
        name = name.replace(".w2.", ".down_proj.")
        name = name.replace(".w3.", ".up_proj.")
        if name.endswith(".scale"):
            name = name.replace(".scale", ".weight_scale")
        if ".gate.bias" in name:
            name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
        return name


DSparkDeepseekV4ForCausalLM = DeepSeekV4DSparkMTP
