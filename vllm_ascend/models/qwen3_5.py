# SPDX-License-Identifier: Apache-2.0

"""Ascend model extensions for Qwen3.5.

The classes in this module are registered through vLLM's model registry.  They
keep the Ascend-specific fused attention and local-drafter PP behavior without
mutating vLLM model classes at import time.
"""

from typing import Any

import torch
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP, Qwen3_5MTP
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors

from vllm_ascend.utils import is_310p

try:
    from vllm.model_executor.models.qwen3_next import (
        _all_gather_hidden_and_residual,
    )
except ImportError:
    _UPSTREAM_MTP_LAYER_HANDLES_SP = False
else:
    _UPSTREAM_MTP_LAYER_HANDLES_SP = True


class AscendQwen3NextAttention(Qwen3NextAttention):
    """Qwen3.5 attention using the Ascend fused QKV/RMSNorm/MRoPE op."""

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Qwen3-Next and 310P keep the portable upstream implementation.  The
        # retired patch was not loaded on 310P either.
        if "qwen3_5" not in self.config.model_type or is_310p():
            return super()._project_qkv_gate(qkv, positions)

        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        if cos_sin.device != qkv.device:
            cos_sin = cos_sin.to(qkv.device)
        if cos_sin.dtype != qkv.dtype:
            cos_sin = cos_sin.to(qkv.dtype)

        q, k, v, gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv,
            q_weight=1.0 + self.q_norm.weight,
            k_weight=1.0 + self.k_norm.weight,
            cos_sin=cos_sin,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            eps=self.config.rms_norm_eps,
            mrope_section=self.rotary_emb.mrope_section,
            is_interleaved=self.rotary_emb.mrope_interleaved,
            rope_dim=self.rotary_emb.rotary_dim,
            has_gate=self.attn_output_gate,
        )
        return q, k, v, gate


def _replace_full_attention_layers(model: Any, vllm_config: Any) -> None:
    """Replace local Qwen3.5 full-attention layers before weights are loaded."""

    for layer in model.layers:
        if getattr(layer, "layer_type", None) != "full_attention":
            continue

        old_attention = layer.self_attn
        layer_name = old_attention.attn.layer_name
        if not layer_name.endswith(".attn"):
            raise ValueError(f"Unexpected Qwen3.5 attention layer name: {layer_name}")
        prefix = layer_name.removesuffix(".attn")

        static_context = vllm_config.compilation_config.static_forward_context
        if static_context.get(layer_name) is not old_attention.attn:
            raise ValueError(f"Unexpected attention registration for {layer_name}")

        # Attention registers itself by prefix. Remove the superseded instance
        # before constructing the Ascend replacement with the same layer name.
        del static_context[layer_name]
        del layer.self_attn
        try:
            layer.self_attn = AscendQwen3NextAttention(
                old_attention.config,
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                reduce_results=not layer.use_attn_reduce_scatter_for_moe,
                prefix=prefix,
            )
        except Exception:
            layer.self_attn = old_attention
            static_context[layer_name] = old_attention.attn
            raise


def _forward_local_mtp(
    predictor: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    inputs_embeds: torch.Tensor | None,
    spec_step_idx: int,
) -> torch.Tensor | IntermediateTensors:
    """Run the Ascend local drafter, which is colocated with the last PP stage."""

    if inputs_embeds is None:
        inputs_embeds = predictor.embed_input_ids(input_ids)
    assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
    inputs_embeds = predictor.pre_fc_norm_embedding(inputs_embeds)
    hidden_states = predictor.pre_fc_norm_hidden(hidden_states)
    hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
    hidden_states = predictor.fc(hidden_states)
    residual = None

    current_step_idx = spec_step_idx % predictor.num_mtp_layers
    mtp_layer = predictor.layers[current_step_idx]
    if not _UPSTREAM_MTP_LAYER_HANDLES_SP and mtp_layer.use_attn_reduce_scatter_for_moe:
        assert hidden_states.shape[0] == positions.shape[-1]
        hidden_states = sequence_parallel_chunk(hidden_states)
        assert residual is None

    hidden_states, residual = mtp_layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {
                "hidden_states": hidden_states,
                "residual": residual,
            }
        )

    if _UPSTREAM_MTP_LAYER_HANDLES_SP:
        if mtp_layer.use_attn_reduce_scatter_for_moe:
            hidden_states, residual = _all_gather_hidden_and_residual(
                hidden_states,
                residual,
                positions.shape[-1],
                predictor.config.hidden_size,
            )
        hidden_states, _ = predictor.norm(hidden_states, residual)
        return hidden_states

    hidden_states, _ = predictor.norm(hidden_states, residual)
    if mtp_layer.use_attn_reduce_scatter_for_moe:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[: positions.shape[-1]]
    return hidden_states


class _AscendQwen35ModelMixin:
    def _replace_qwen35_attention(self, vllm_config: Any) -> None:
        language_model = getattr(self, "language_model", self)
        _replace_full_attention_layers(language_model.model, vllm_config)


class AscendQwen3_5ForCausalLM(_AscendQwen35ModelMixin, Qwen3_5ForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._replace_qwen35_attention(vllm_config)


class AscendQwen3_5MoeForCausalLM(
    _AscendQwen35ModelMixin,
    Qwen3_5MoeForCausalLM,
):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._replace_qwen35_attention(vllm_config)


class AscendQwen3_5ForConditionalGeneration(
    _AscendQwen35ModelMixin,
    Qwen3_5ForConditionalGeneration,
):
    def __init__(self, *, vllm_config, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._replace_qwen35_attention(vllm_config)


class AscendQwen3_5MoeForConditionalGeneration(
    _AscendQwen35ModelMixin,
    Qwen3_5MoeForConditionalGeneration,
):
    def __init__(self, *, vllm_config, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._replace_qwen35_attention(vllm_config)


class _AscendQwen35MTPMixin:
    def _prepare_ascend_mtp(self, vllm_config: Any) -> None:
        _replace_full_attention_layers(self.model, vllm_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # intermediate_tensors intentionally are not consumed: Ascend's local
        # drafter runs on the last PP stage and combines local target states.
        del intermediate_tensors
        return _forward_local_mtp(
            self.model,
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            int(kwargs.get("spec_step_idx", 0)),
        )


class AscendQwen3_5MTP(_AscendQwen35MTPMixin, Qwen3_5MTP):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._prepare_ascend_mtp(vllm_config)


class AscendQwen3_5MoeMTP(_AscendQwen35MTPMixin, Qwen3_5MoeMTP):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._prepare_ascend_mtp(vllm_config)
