"""Explicit Q/K RMSNorm + RoPE fusion shared by attention implementations."""

import torch
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding, AscendYaRNRotaryEmbedding
from vllm_ascend.utils import enable_sp


def maybe_fused_qknorm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm: torch.nn.Module,
    k_norm: torch.nn.Module,
    rotary_emb: torch.nn.Module,
    q_size: int,
    kv_size: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return fused Q, K, V for eligible Qwen3 attention, or None for fallback."""
    if not get_ascend_config().ascend_compilation_config.fuse_qknorm_rope:
        return None
    if qkv.dtype != torch.bfloat16 or qkv.ndim != 2 or positions.ndim != 1 or positions.dtype != torch.int64:
        return None
    if qkv.shape[0] != positions.shape[0]:
        return None
    if q_size <= 0 or kv_size <= 0 or head_dim <= 0:
        return None
    if q_size % head_dim or kv_size % head_dim or qkv.shape[-1] != q_size + 2 * kv_size:
        return None
    if type(rotary_emb) not in (AscendRotaryEmbedding, AscendYaRNRotaryEmbedding):
        return None
    if not rotary_emb.is_neox_style or rotary_emb.head_size != head_dim:
        return None
    if rotary_emb.use_mtp and enable_sp():
        return None
    rope_dim = rotary_emb.rotary_dim
    cache = rotary_emb.cos_sin_cache
    if rope_dim <= 0 or rope_dim > head_dim or cache.shape[-1] != rope_dim:
        return None
    if cache.device != qkv.device or cache.dtype != qkv.dtype:
        return None
    if not isinstance(q_norm, RMSNorm) or not isinstance(k_norm, RMSNorm):
        return None
    if not q_norm.has_weight or not k_norm.has_weight or not q_norm.pass_weight or not k_norm.pass_weight:
        return None
    if q_norm.variance_size_override is not None or k_norm.variance_size_override is not None:
        return None
    if q_norm.weight.shape != (head_dim,) or k_norm.weight.shape != (head_dim,):
        return None
    if q_norm.weight.dtype != qkv.dtype or k_norm.weight.dtype != qkv.dtype:
        return None
    eps = q_norm.variance_epsilon
    if eps not in (1e-6, 1e-5) or k_norm.variance_epsilon != eps:
        return None
    if head_dim != 128:
        return None
    q_bias = q_norm.bias if getattr(q_norm, "bias_loaded", False) else None
    k_bias = k_norm.bias if getattr(k_norm, "bias_loaded", False) else None
    if (q_bias is None) != (k_bias is None):
        return None
    if q_bias is not None and k_bias is not None:
        if q_bias.shape != (head_dim,) or k_bias.shape != (head_dim,):
            return None
        if q_bias.dtype != qkv.dtype or k_bias.dtype != qkv.dtype:
            return None
        if q_bias.device != qkv.device or k_bias.device != qkv.device:
            return None
    return DeviceOperator.split_qkv_rmsnorm_rope(
        input=qkv,
        q_weight=q_norm.weight,
        k_weight=k_norm.weight,
        q_hidden_size=q_size,
        kv_hidden_size=kv_size,
        head_dim=head_dim,
        eps=eps,
        q_bias=q_bias,
        k_bias=k_bias,
        cos_sin_cache=cache,
        positions=positions,
    )
