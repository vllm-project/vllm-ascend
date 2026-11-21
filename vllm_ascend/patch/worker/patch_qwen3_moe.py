import torch

from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

from vllm_ascend.model_executor.fusion.kernels.fused_norm_quant import (
    maybe_fused_norm_quant,
)


def _patched_qwen3_moe_decoder_layer_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = maybe_fused_norm_quant(
            self.input_layernorm, hidden_states, residual, self.self_attn.qkv_proj
        )

    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
    )

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual


Qwen3MoeDecoderLayer.forward = _patched_qwen3_moe_decoder_layer_forward