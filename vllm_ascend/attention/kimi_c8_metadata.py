from __future__ import annotations

import torch


def build_quant_mla_metadata_kwargs(
    cache_seqlens,
    cu_seqlens_q,
    seqused_q,
    *,
    num_heads_q: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    head_dim_qk: int = 576,
    head_dim_v: int = 512,
    global_seqlens: torch.Tensor | None = None,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 128,
):
    if head_dim_qk != 576 or head_dim_v != 512:
        raise ValueError("Kimi C8 metadata requires qk=576 and v=512")
    if num_heads_q <= 0:
        raise ValueError("num_heads_q must be positive")
    kwargs = dict(
        cache_seqlens=cache_seqlens,
        num_heads_q=num_heads_q,
        num_heads_kv=1,
        quant_mode=1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        head_dim_qk=576,
        head_dim_v=512,
        mask_mode=0 if enable_dcp or max_seqlen_q == 1 else 3,
        layout_q="TND",
    )
    if enable_dcp:
        if (
            cp_world <= 0
            or not 0 <= cp_rank < cp_world
            or cp_kv_cache_interleave_size <= 0
            or cp_kv_cache_interleave_size % 128
        ):
            raise ValueError("native DCP requires valid world/rank and 128-aligned actual interleave")
        if (
            not isinstance(global_seqlens, torch.Tensor)
            or global_seqlens.ndim != 1
            or global_seqlens.shape != cache_seqlens.shape
            or global_seqlens.dtype != torch.int32
        ):
            raise ValueError("native DCP global_seqlens must be int32 with the same request shape as cache_seqlens")
        kwargs.update(
            global_seqlens=global_seqlens,
            enable_dcp=True,
            cp_world=cp_world,
            cp_rank=cp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )
    return kwargs
