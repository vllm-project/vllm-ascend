# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import math
from collections.abc import Iterable
from dataclasses import fields
from typing import Any, cast

import torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers import mla as mla_module
from vllm.model_executor.layers.attention import mla_attention
from vllm.model_executor.models import deepseek_v2

from vllm_ascend.core.kv_cache_interface import (
    Dots3NoteMLAAttentionSpec,
    Dots3NoteSlidingWindowMLASpec,
)


def _is_dots3_note_config(config: Any) -> bool:
    return "dots3_note" in {
        getattr(config, "model_type", None),
        getattr(config, "original_model_type", None),
    }


def _dots3_note_noaux_tc_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    use_dynamic_rsf: bool = False,
    **_: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    del hidden_states
    logit_scores = gating_output.sigmoid()
    scores_for_choice = logit_scores + e_score_correction_bias.unsqueeze(0)
    _, topk_idx = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=False)
    topk_weight = logit_scores.gather(1, topk_idx)
    if renormalize:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    routed_scaling = 1 / (topk_weight.norm(dim=-1, keepdim=True) + 1e-20) if use_dynamic_rsf else routed_scaling_factor
    return topk_weight * routed_scaling, topk_idx


_original_moe_init = getattr(
    deepseek_v2.DeepseekV2MoE.__init__,
    "_vllm_ascend_dots3_note_original",
    deepseek_v2.DeepseekV2MoE.__init__,
)


def _moe_init(
    self,
    config,
    parallel_config,
    quant_config=None,
    reduce_results: bool = True,
    prefix: str = "",
    apply_routed_scale_to_output: bool = False,
) -> None:
    if not (_is_dots3_note_config(config) and getattr(config, "topk_method", None) == "noaux_tc"):
        _original_moe_init(
            self,
            config=config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=prefix,
            apply_routed_scale_to_output=apply_routed_scale_to_output,
        )
        return

    def route(hidden_states, gating_output, topk, renormalize, **kwargs):
        return _dots3_note_noaux_tc_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            use_dynamic_rsf=getattr(config, "use_dynamic_rsf", False),
            **kwargs,
        )

    original_fused_moe_factory = deepseek_v2.FusedMoEFactory

    def create_dots3_note_fused_moe_factory(*args, **kwargs):
        kwargs["use_grouped_topk"] = False
        kwargs["custom_routing_function"] = route
        kwargs["routed_scaling_factor"] = 1.0
        return original_fused_moe_factory(*args, **kwargs)

    deepseek_v2.FusedMoEFactory = create_dots3_note_fused_moe_factory
    try:
        _original_moe_init(
            self,
            config=config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=prefix,
            apply_routed_scale_to_output=apply_routed_scale_to_output,
        )
    finally:
        deepseek_v2.FusedMoEFactory = original_fused_moe_factory


_original_mla_attention_init = getattr(
    deepseek_v2.DeepseekV2MLAAttention.__init__,
    "_vllm_ascend_dots3_note_original",
    deepseek_v2.DeepseekV2MLAAttention.__init__,
)


def _deepseek_mla_attention_init(
    self,
    vllm_config: VllmConfig,
    config,
    hidden_size: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    q_lora_rank: int | None,
    kv_lora_rank: int,
    max_position_embeddings: int = 8192,
    cache_config=None,
    quant_config=None,
    prefix: str = "",
    topk_indices_buffer: torch.Tensor | None = None,
    input_size: int | None = None,
    reduce_results: bool = True,
    non_causal_multi_token_decode: bool = False,
    sliding_window: int | None = None,
    rope_parameters: dict[str, Any] | None = None,
) -> None:
    """Initialize upstream MLA plus Dots3 Note layer-local extensions."""
    if not _is_dots3_note_config(config):
        _original_mla_attention_init(
            self,
            vllm_config=vllm_config,
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            input_size=input_size,
            reduce_results=reduce_results,
            non_causal_multi_token_decode=non_causal_multi_token_decode,
        )
        return

    sliding_window = getattr(config, "_dots3_note_sliding_window", sliding_window)
    local_config = copy.copy(config)
    local_config.rope_parameters = dict(rope_parameters or getattr(config, "rope_parameters", {}))
    _original_mla_attention_init(
        self,
        vllm_config=vllm_config,
        config=local_config,
        hidden_size=hidden_size,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        max_position_embeddings=max_position_embeddings,
        cache_config=cache_config,
        quant_config=quant_config,
        prefix=prefix,
        topk_indices_buffer=topk_indices_buffer,
        input_size=input_size,
        reduce_results=reduce_results,
        non_causal_multi_token_decode=non_causal_multi_token_decode,
    )

    skip_topk = self.mla_attn.skip_topk
    if sliding_window is not None:
        self.is_v32 = False
        self.indexer = None
        self.indexer_rope_emb = None

    self.apply_mla_qkv_lora_rescale = getattr(config, "apply_mla_qkv_lora_rescale", False)
    self.k_rope_only_layernorm = (
        deepseek_v2.RMSNorm(self.qk_rope_head_dim, eps=config.rms_norm_eps)
        if getattr(config, "k_rope_only_layernorm", False)
        else None
    )
    self.sdpa_gate_type = getattr(config, "sdpa_gate_type", None) or getattr(config, "attention_gate_type", None)
    if self.sdpa_gate_type in ("elementwise", "headwise"):
        gate_output_size = self.num_heads * self.v_head_dim if self.sdpa_gate_type == "elementwise" else self.num_heads
        gate_cls = (
            deepseek_v2.ReplicatedLinear if self.sdpa_gate_type == "headwise" else deepseek_v2.ColumnParallelLinear
        )
        self.g_proj = gate_cls(
            self.hidden_size,
            gate_output_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
    else:
        self.g_proj = None

    modules = mla_module.MLAModules(
        kv_a_layernorm=self.kv_a_layernorm,
        kv_b_proj=self.kv_b_proj,
        rotary_emb=self.rotary_emb,
        o_proj=self.o_proj,
        fused_qkv_a_proj=(self.fused_qkv_a_proj if self.q_lora_rank is not None else None),
        kv_a_proj_with_mqa=(self.kv_a_proj_with_mqa if self.q_lora_rank is None else None),
        q_a_layernorm=(self.q_a_layernorm if self.q_lora_rank is not None else None),
        q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
        q_proj=self.q_proj if self.q_lora_rank is None else None,
        indexer=self.indexer,
        indexer_rotary_emb=self.indexer_rope_emb,
        is_sparse=self.is_v32,
        topk_indices_buffer=topk_indices_buffer,
    )
    modules._vllm_ascend_dots3_note = True
    modules.k_rope_only_layernorm = self.k_rope_only_layernorm
    modules.g_proj = self.g_proj
    modules.sdpa_gate_type = self.sdpa_gate_type
    modules.apply_mla_qkv_lora_rescale = self.apply_mla_qkv_lora_rescale

    static_context = get_current_vllm_config().compilation_config.static_forward_context
    static_context.pop(prefix, None)
    static_context.pop(f"{prefix}.attn", None)
    self.mla_attn = mla_module.MultiHeadLatentAttentionWrapper(
        self.hidden_size,
        self.num_local_heads,
        self.scaling,
        self.qk_nope_head_dim,
        self.qk_rope_head_dim,
        self.v_head_dim,
        self.q_lora_rank,
        self.kv_lora_rank,
        modules,
        cache_config,
        quant_config,
        prefix,
        skip_topk=skip_topk,
        non_causal_multi_token_decode=non_causal_multi_token_decode,
        sliding_window=sliding_window,
    )


_original_decoder_init = getattr(
    deepseek_v2.DeepseekV2DecoderLayer.__init__,
    "_vllm_ascend_dots3_note_original",
    deepseek_v2.DeepseekV2DecoderLayer.__init__,
)


def _decoder_init(
    self,
    vllm_config: VllmConfig,
    prefix: str,
    config=None,
    topk_indices_buffer: torch.Tensor | None = None,
) -> None:
    if config is None:
        config = vllm_config.model_config.hf_config
    if not _is_dots3_note_config(config):
        _original_decoder_init(
            self,
            vllm_config,
            prefix,
            config,
            topk_indices_buffer,
        )
        return

    local_config = copy.copy(config)
    moe_layer_freq = getattr(config, "moe_layer_freq", 1)
    layer_idx = int(prefix.rsplit(".", 1)[-1])
    layer_types = getattr(config, "layer_types", None)
    is_mtp_layer = layer_idx >= getattr(config, "num_hidden_layers", 0)
    if is_mtp_layer and not getattr(config, "mtp_use_moe", False):
        local_config.first_k_dense_replace = layer_idx + 1
    is_sliding = is_mtp_layer or (
        isinstance(layer_types, (list, tuple))
        and layer_idx < len(layer_types)
        and layer_types[layer_idx] == "sliding_attention"
    )
    if is_sliding:
        local_config.num_attention_heads = (
            getattr(config, "swa_num_attention_heads", None) or config.num_attention_heads
        )
        local_config.num_key_value_heads = (
            getattr(config, "swa_num_key_value_heads", None) or local_config.num_attention_heads
        )
        local_config.q_lora_rank = getattr(config, "swa_q_lora_rank", None) or config.q_lora_rank
        local_config.kv_lora_rank = getattr(config, "swa_kv_lora_rank", None) or config.kv_lora_rank
        local_config.qk_nope_head_dim = getattr(config, "swa_qk_nope_head_dim", None) or config.qk_nope_head_dim
        local_config.qk_rope_head_dim = getattr(config, "swa_qk_rope_head_dim", None) or config.qk_rope_head_dim
        local_config.v_head_dim = getattr(config, "swa_v_head_dim", None) or config.v_head_dim
        local_config.sdpa_gate_type = getattr(config, "swa_attention_gate_type", None) or getattr(
            config, "attention_gate_type", None
        )
        sliding_window = getattr(config, "sliding_window_size", None) or getattr(config, "sliding_window", None)
        local_config._dots3_note_sliding_window = None if sliding_window is None else sliding_window - 1
        rope_theta = (
            getattr(config, "swa_rope_theta", None)
            or getattr(config, "rope_local_base_freq", None)
            or getattr(config, "rope_theta", 10000)
        )
        local_config.rope_parameters = {
            "rope_type": "default",
            "rope_theta": rope_theta,
        }
    else:
        local_config.sdpa_gate_type = getattr(config, "attention_gate_type", None)

    if isinstance(moe_layer_freq, (list, tuple)):
        is_moe_layer = layer_idx < len(moe_layer_freq) and bool(moe_layer_freq[layer_idx])
        local_config.moe_layer_freq = 1
        if not is_moe_layer:
            local_config.n_routed_experts = None

    _original_decoder_init(
        self,
        vllm_config,
        prefix,
        local_config,
        topk_indices_buffer,
    )


_original_wrapper_init = getattr(
    mla_module.MultiHeadLatentAttentionWrapper.__init__,
    "_vllm_ascend_dots3_note_original",
    mla_module.MultiHeadLatentAttentionWrapper.__init__,
)
_original_wrapper_forward = getattr(
    mla_module.MultiHeadLatentAttentionWrapper.forward,
    "_vllm_ascend_dots3_note_original",
    mla_module.MultiHeadLatentAttentionWrapper.forward,
)


def _wrapper_init(
    self,
    hidden_size: int,
    num_heads: int,
    scale: float,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    q_lora_rank: int | None,
    kv_lora_rank: int,
    mla_modules,
    cache_config=None,
    quant_config=None,
    prefix: str = "",
    skip_topk: bool = False,
    non_causal_multi_token_decode: bool = False,
    allow_short_prefill_indexer_scoring_skip: bool = False,
    sliding_window: int | None = None,
) -> None:
    _original_wrapper_init(
        self,
        hidden_size=hidden_size,
        num_heads=num_heads,
        scale=scale,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        mla_modules=mla_modules,
        cache_config=cache_config,
        quant_config=quant_config,
        prefix=prefix,
        skip_topk=skip_topk,
        non_causal_multi_token_decode=non_causal_multi_token_decode,
        allow_short_prefill_indexer_scoring_skip=allow_short_prefill_indexer_scoring_skip,
    )
    if not getattr(mla_modules, "_vllm_ascend_dots3_note", False):
        return

    self._vllm_ascend_dots3_note = True
    self.k_rope_only_layernorm = getattr(mla_modules, "k_rope_only_layernorm", None)
    self.g_proj = getattr(mla_modules, "g_proj", None)
    self.sdpa_gate_type = getattr(mla_modules, "sdpa_gate_type", None)
    self.apply_mla_qkv_lora_rescale = getattr(mla_modules, "apply_mla_qkv_lora_rescale", False)
    get_current_vllm_config().compilation_config.static_forward_context.pop(f"{prefix}.attn", None)
    self.mla_attn = mla_attention.MLAAttention(
        num_heads=self.num_heads,
        scale=scale,
        qk_nope_head_dim=self.qk_nope_head_dim,
        qk_rope_head_dim=self.qk_rope_head_dim,
        v_head_dim=self.v_head_dim,
        q_lora_rank=self.q_lora_rank,
        kv_lora_rank=self.kv_lora_rank,
        cache_config=cache_config,
        quant_config=quant_config,
        prefix=f"{prefix}.attn",
        kv_b_proj=self.kv_b_proj,
        dcp_q_replicate=self.dcp_q_replicate,
        use_sparse=self.is_sparse,
        indexer=self.indexer,
        sliding_window=sliding_window,
        dots3_note_model=True,
        non_causal_multi_token_decode=non_causal_multi_token_decode,
    )


def _wrapper_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None = None,
) -> torch.Tensor:
    if not getattr(self, "_vllm_ascend_dots3_note", False):
        return _original_wrapper_forward(self, positions, hidden_states, llama_4_scaling)

    q_c = None
    if self.q_lora_rank is not None:
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_lora = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        q_c = self.q_a_layernorm(q_c)
        if self.apply_mla_qkv_lora_rescale:
            q_c = q_c * math.sqrt(self.hidden_size / self.q_lora_rank)
        q = self.q_b_proj(q_c)[0]
    else:
        kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        q = self.q_proj(hidden_states)[0]

    kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_c_normed = self.kv_a_layernorm(kv_c)
    if self.apply_mla_qkv_lora_rescale:
        kv_c_normed = kv_c_normed * math.sqrt(self.hidden_size / self.kv_lora_rank)
    q = q.reshape(-1, self.num_heads, self.qk_head_dim)
    k_pe = k_pe.unsqueeze(1)
    if self.k_rope_only_layernorm is not None:
        k_pe = self.k_rope_only_layernorm(k_pe)
    if self.rotary_emb is not None:
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(positions, q[..., self.qk_nope_head_dim :], k_pe)
    if self.indexer and self.is_sparse and not self.skip_topk:
        self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
    if llama_4_scaling is not None:
        q *= llama_4_scaling

    attn_out = self.mla_attn(
        q,
        kv_c_normed,
        k_pe,
        output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
    )
    if self.g_proj is not None:
        gate = self.g_proj(hidden_states)[0]
        if self.sdpa_gate_type == "headwise" and gate.shape[-1] != self.num_heads:
            rank = get_tensor_model_parallel_rank()
            gate = gate.narrow(-1, rank * self.num_heads, self.num_heads)
        gate = torch.sigmoid(gate.float()).to(attn_out.dtype)
        if self.sdpa_gate_type == "elementwise":
            attn_out = attn_out * gate.view_as(attn_out)
        else:
            attn_shape = attn_out.shape
            attn_out = attn_out.reshape(-1, self.num_heads, self.v_head_dim) * gate.reshape(-1, self.num_heads, 1)
            attn_out = attn_out.reshape(attn_shape)
    return self.o_proj(attn_out)[0]


_original_mla_init = getattr(
    mla_attention.MLAAttention.__init__,
    "_vllm_ascend_dots3_note_original",
    mla_attention.MLAAttention.__init__,
)
_original_get_kv_cache_spec = getattr(
    mla_attention.MLAAttention.get_kv_cache_spec,
    "_vllm_ascend_dots3_note_original",
    mla_attention.MLAAttention.get_kv_cache_spec,
)


def _mla_init(self, *args, sliding_window=None, dots3_note_model=False, **kwargs) -> None:
    if dots3_note_model:
        kwargs["dots3_note_model"] = True
    _original_mla_init(self, *args, **kwargs)
    if not dots3_note_model:
        return
    self._vllm_ascend_dots3_note = True
    self.sliding_window = sliding_window
    self.impl.sliding_window = sliding_window


def _copy_spec(spec, spec_cls, **updates):
    values = {
        field.name: getattr(spec, field.name) for field in fields(spec_cls) if field.init and hasattr(spec, field.name)
    }
    values.update(updates)
    return spec_cls(**values)


def _get_kv_cache_spec(self, vllm_config: VllmConfig):
    spec = _original_get_kv_cache_spec(self, vllm_config)
    if not getattr(self, "_vllm_ascend_dots3_note", False):
        return spec
    if self.sliding_window is not None:
        return _copy_spec(
            spec,
            Dots3NoteSlidingWindowMLASpec,
            sliding_window=self.sliding_window,
        )
    return _copy_spec(spec, Dots3NoteMLAAttentionSpec)


_original_load_weights = getattr(
    deepseek_v2.DeepseekV2Model.load_weights,
    "_vllm_ascend_dots3_note_original",
    deepseek_v2.DeepseekV2Model.load_weights,
)


def _load_weights(
    self,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    if not _is_dots3_note_config(self.config):
        return _original_load_weights(self, weights)

    moe_layer_freq = getattr(self.config, "moe_layer_freq", 1)
    num_main_layers = (
        len(moe_layer_freq) if isinstance(moe_layer_freq, (list, tuple)) else self.config.num_hidden_layers
    )

    def main_model_weights():
        for name, weight in weights:
            if name.startswith(("model.mtp.", "mtp.")):
                continue
            if name.startswith("model.layers."):
                layer_idx = name.split(".", 3)[2]
                if layer_idx.isdigit() and int(layer_idx) >= num_main_layers:
                    continue
            yield name, weight

    return _original_load_weights(self, main_model_weights())


cast(Any, _moe_init)._vllm_ascend_dots3_note_original = _original_moe_init
cast(Any, _deepseek_mla_attention_init)._vllm_ascend_dots3_note_original = _original_mla_attention_init
cast(Any, _decoder_init)._vllm_ascend_dots3_note_original = _original_decoder_init
cast(Any, _load_weights)._vllm_ascend_dots3_note_original = _original_load_weights
cast(Any, _wrapper_init)._vllm_ascend_dots3_note_original = _original_wrapper_init
cast(Any, _wrapper_forward)._vllm_ascend_dots3_note_original = _original_wrapper_forward
cast(Any, _mla_init)._vllm_ascend_dots3_note_original = _original_mla_init
cast(Any, _get_kv_cache_spec)._vllm_ascend_dots3_note_original = _original_get_kv_cache_spec
deepseek_v2.DeepseekV2MoE.__init__ = _moe_init
deepseek_v2.DeepseekV2MLAAttention.__init__ = _deepseek_mla_attention_init
deepseek_v2.DeepseekV2DecoderLayer.__init__ = _decoder_init
deepseek_v2.DeepseekV2Model.load_weights = _load_weights
mla_module.MultiHeadLatentAttentionWrapper.__init__ = _wrapper_init
mla_module.MultiHeadLatentAttentionWrapper.forward = _wrapper_forward
mla_attention.MLAAttention.__init__ = _mla_init
mla_attention.MLAAttention.get_kv_cache_spec = _get_kv_cache_spec
