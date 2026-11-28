from __future__ import annotations

import torch

from vllm_ascend.attention.kimi_c8_static import FP8_E4M3_MAX


def run_quant_mla(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_q: torch.Tensor,
    metadata: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    return_softmax_lse: bool = True,
    softmax_scale: float = 1.0,
    attn_mask: torch.Tensor | None = None,
    mask_mode: int | None = None,
):
    if q.dtype != torch.float8_e4m3fn or k_cache.dtype != torch.float8_e4m3fn:
        raise ValueError("Kimi C8 quant FA requires E4M3 query and cache")
    if mask_mode is None:
        mask_mode = 0 if max_seqlen_q == 1 else 3
    if mask_mode not in (0, 3):
        raise ValueError("quant MLA mask_mode must be 0 (no mask) or 3 (causal)")
    if mask_mode == 3 and attn_mask is None:
        raise ValueError("causal quant MLA requires attn_mask")
    if mask_mode == 0 and attn_mask is not None:
        raise ValueError("quant MLA mask_mode=0 cannot accept attn_mask")
    result = torch.ops.cann_ops_transformer.quant_flash_mla(
        q,
        k_cache,
        q_descale,
        k_descale,
        block_table,
        cache_seqlens,
        1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        metadata=metadata,
        head_dim_v=512,
        softmax_scale=softmax_scale,
        mask_mode=mask_mode,
        attn_mask=attn_mask,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        layout_q="TND",
        layout_kv="PA_NZ",
        layout_out="TND",
        return_softmax_lse=return_softmax_lse,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("quant MLA must return (attention_output, softmax_lse)")
    return result


def merge_and_quantize_query(q_nope: torch.Tensor, q_pe: torch.Tensor):
    """Build recipe-compatible merged E4M3 Q and per-token/head descale."""
    if q_nope.ndim != 3 or q_pe.ndim != 3 or q_nope.shape[:2] != q_pe.shape[:2]:
        raise ValueError("Q components must be [tokens, heads, dim] with matching prefix")
    merged = torch.cat((q_nope.float(), q_pe.float()), dim=-1)
    scale = torch.clamp(merged.abs().amax(dim=-1), min=1e-8) / FP8_E4M3_MAX
    quant = (merged / scale.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return quant, scale.float()
