# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Experimental M3.1 Flash DSpark adapter using the existing Ascend protocol.

Only draft model differences are implemented here. Target verification,
Markov drafting, confidence, batching and cache metadata use the supplied
vLLM/vLLM-Ascend implementations unchanged.
"""

import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention,
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    _resolve_layer_attention,
)
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkModel

from vllm_ascend.models.minimax_m3.minimax_m3 import MiniMaxM3MLP
from vllm_ascend.models.qwen3_dspark import AscendQwen3DSparkForCausalLM
from vllm_ascend.patch.minimax_m3_dspark import iter_draft_weights


class MiniMaxFlashDraftLayer(DFlashQwen3DecoderLayer):
    def __init__(self, vllm_config, *, config, layer_idx, cache_config=None, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)
        self.self_attn = DFlashQwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=dict(config.rope_parameters),
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=False,
            sliding_window=sliding_window,
            causal=causal,
            is_neox_style=True,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn.q_norm = GemmaRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.self_attn.k_norm = GemmaRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.mlp = MiniMaxM3MLP(
            config,
            intermediate_size=config.dspark_ffn_hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MiniMaxFlashDSparkModel(Qwen3DSparkModel):
    decoder_layer_cls = MiniMaxFlashDraftLayer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = self.config
        for name, choice in (("hidden_norm", config.dspark_hidden_norm_type), ("norm", config.dspark_final_norm_type)):
            if choice not in ("rms", "gemma"):
                raise ValueError(f"Explicit draft norm type required for {name}")
            cls = GemmaRMSNorm if choice == "gemma" else RMSNorm
            setattr(self, name, cls(config.hidden_size, eps=config.rms_norm_eps))

    def _build_fused_kv_buffers(self):
        # Native MXFP8 projections retain their packed weights and scales.
        if self.quant_config is None:
            super()._build_fused_kv_buffers()

    def precompute_and_store_context_kv(self, context_states, context_positions, context_slot_mapping=None):
        # Use the actual norm modules (including Gemma's +1) and assign RoPE's
        # returned output. This remains correct for out-of-place NPU RoPE.
        length = context_states.shape[0]
        normalized = self.hidden_norm(context_states)
        if self.quant_config is None:
            if not hasattr(self, "_num_attn_layers"):
                self._build_fused_kv_buffers()
            layers, nkv, hd = self._num_attn_layers, self._num_kv_heads, self._head_dim
            kv = F.linear(normalized, self._fused_kv_weight, self._fused_kv_bias)
            kv = kv.view(length, layers, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            nkv, hd = attn.num_kv_heads, attn.head_dim
            if self.quant_config is not None:
                qkv, _ = attn.qkv_proj(normalized)
                _, keys, values = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
                keys, values = keys.reshape(length, nkv, hd), values.reshape(length, nkv, hd)
            else:
                keys, values = kv[0, i], kv[1, i]
            keys = attn.k_norm(keys).reshape(length, nkv * hd)
            keys, _ = attn.rotary_emb(context_positions, keys, None)
            if context_slot_mapping is None:
                continue
            slots = context_slot_mapping[i] if isinstance(context_slot_mapping, (list, tuple)) else context_slot_mapping
            if slots is None:
                continue
            inner = attn.attn
            inner.impl.do_kv_cache_update(inner, keys.view(length, nkv, hd), values, inner.kv_cache, slots)


class MiniMaxFlashDSparkForCausalLM(AscendQwen3DSparkForCausalLM):
    model_cls = MiniMaxFlashDSparkModel

    def __init__(self, *, vllm_config, prefix=""):
        config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(config, "ascend_minimax_dspark", None) != 1:
            raise ValueError("Use the MiniMax M3.1 DSpark configuration")
        if not config.use_gemma_norm or config.hidden_act != "swigluoai":
            raise ValueError("Unexpected MiniMax draft decoder semantics")
        if getattr(config, "dspark_hybrid_context_fusion", False):
            raise ValueError("Hybrid context fusion is not implemented by this adapter")
        # DFlash's constructor honors model_cls; the DSpark constructor
        # currently hardcodes Qwen3DSparkModel. All DSpark methods are inherited.
        DFlashQwen3ForCausalLM.__init__(self, vllm_config=vllm_config, prefix=prefix)
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        if config.vocab_size != self.target_vocab_size:
            raise ValueError("The unembedded Flash draft must match target vocabulary")
        self.enable_confidence_head = config.enable_confidence_head
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False

    def load_weights(self, weights):
        normalized = iter_draft_weights(weights, self.config.to_dict())
        return super().load_weights(normalized)
