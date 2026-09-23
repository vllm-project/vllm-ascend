# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal eager adapter for the external FlashMLA metadata/main operator pair.

The runner owns the cache. This adapter keeps its views and page identities;
only per-batch metadata is materialized. Both operators use the current stream.
"""

from dataclasses import dataclass

import torch

from vllm_ascend.attention.utils import MLA_FLASH_SUPPORTED_Q_HEADS

FLASH_MLA_BLOCK_SIZE = 128
FLASH_MLA_QK_DIM = 576
FLASH_MLA_V_DIM = 512
FLASH_MLA_MASK_SIZE = 2048


@dataclass
class FlashMLAMetadata:
    num_tokens: int
    num_heads: int
    schedule: torch.Tensor
    cu: torch.Tensor
    used_q: torch.Tensor
    cache_lens: torch.Tensor
    block_table: torch.Tensor
    slots: torch.Tensor
    token_live: torch.Tensor
    positions: torch.Tensor
    attn_mask: torch.Tensor | None
    causal: bool
    is_prefill: bool


def init_flash_mla_metadata(builder, impl) -> None:
    if impl.num_heads not in MLA_FLASH_SUPPORTED_Q_HEADS or impl.num_kv_heads != 1:
        raise ValueError("FlashMLA requires actual local Q heads in {8, 12, 64, 96} and one KV head")
    builder.flash_num_heads = impl.num_heads
    builder.flash_attn_mask = torch.triu(
        torch.ones((FLASH_MLA_MASK_SIZE, FLASH_MLA_MASK_SIZE), dtype=torch.int8, device=builder.device), diagonal=1
    )


def build_flash_mla_metadata(builder, common) -> FlashMLAMetadata:
    # Lazy import: the default backend does not require the external package.
    from cann_ops_transformer.ops import flash_mla_with_kvcache_metadata

    batch = common.num_reqs
    tokens = max(common.num_actual_tokens, common.num_input_tokens)
    args = {"dtype": torch.int32, "device": builder.device}
    # One zero-used request accounts for physical padding, without exposing it
    # as a real query or allowing it to write any cache slot.
    cu = torch.empty(batch + 2, **args)
    cu[: batch + 1].copy_(common.query_start_loc[: batch + 1])
    cu[-1].fill_(tokens)
    used_q = torch.zeros(batch + 1, **args)
    used_q[:batch].copy_(cu[1 : batch + 1] - cu[:batch])
    used_q[:batch].masked_fill_(common.seq_lens[:batch] <= 0, 0)
    cache_lens = torch.zeros(batch + 1, **args)
    cache_lens[:batch].copy_(common.seq_lens[:batch])
    table = common.block_table_tensor[:batch]
    block_table = torch.zeros((batch + 1, table.shape[1]), **args)
    block_table[:batch].copy_(table)
    boundaries = torch.zeros(tokens + 1, **args)
    live_rows = (used_q > 0).to(torch.int32)
    boundaries.scatter_add_(0, cu[:-1].long(), live_rows)
    boundaries.scatter_add_(0, (cu[:-1] + used_q).long(), -live_rows)
    token_live = boundaries.cumsum(0)[:tokens] > 0
    slots = torch.full((tokens,), -1, dtype=torch.int64, device=builder.device)
    source_slots = common.slot_mapping[:tokens]
    slots[: source_slots.shape[0]].copy_(source_slots)
    slots.masked_fill_(~token_live, -1)
    positions = torch.zeros(tokens, dtype=torch.int64, device=builder.device)
    source_positions = common.positions[:tokens]
    positions[: source_positions.shape[0]].copy_(source_positions)
    schedule = flash_mla_with_kvcache_metadata(
        cache_lens,
        builder.flash_num_heads,
        1,
        cu_seqlens_q=cu,
        seqused_q=used_q,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        head_dim_qk=FLASH_MLA_QK_DIM,
        head_dim_v=FLASH_MLA_V_DIM,
        mask_mode=3 if common.causal else 0,
        layout_q="TND",
    )
    return FlashMLAMetadata(
        num_tokens=tokens,
        num_heads=builder.flash_num_heads,
        schedule=schedule,
        cu=cu,
        used_q=used_q,
        cache_lens=cache_lens,
        block_table=block_table,
        slots=slots,
        token_live=token_live,
        positions=positions,
        attn_mask=builder.flash_attn_mask if common.causal else None,
        causal=common.causal,
        is_prefill=common.max_query_len > builder.decode_threshold,
    )


def validate_flash_cache(cache: torch.Tensor) -> None:
    """Validate the physical view without repacking or changing its ownership."""
    if not isinstance(cache, torch.Tensor) or cache.ndim != 4 or cache.shape[1:] != (128, 1, 576):
        raise ValueError("FlashMLA requires token-fused PA_BBND cache [P,128,1,576]")
    page_span = (FLASH_MLA_BLOCK_SIZE - 1) * cache.stride(1) + FLASH_MLA_QK_DIM
    if cache.stride(-1) != 1 or cache.stride(1) < FLASH_MLA_QK_DIM or cache.stride(0) < page_span:
        raise ValueError("FlashMLA requires non-overlapping BBND pages and contiguous channels")
    if cache.dtype != torch.bfloat16:
        raise ValueError("This FlashMLA integration requires unquantized BF16 cache")


def run_flash_mla(query: torch.Tensor, cache: torch.Tensor, flash: FlashMLAMetadata, scale: float):
    from cann_ops_transformer.ops import flash_mla_with_kvcache

    validate_flash_cache(cache)
    if query.shape != (flash.num_tokens, flash.num_heads, FLASH_MLA_QK_DIM):
        raise ValueError("Actual Q shape does not match FlashMLA metadata geometry")
    if query.dtype != cache.dtype or query.device != cache.device:
        raise ValueError("FlashMLA Q and cache must have the same BF16 dtype and device")
    return flash_mla_with_kvcache(
        query,
        cache,
        block_table=flash.block_table,
        cache_seqlens=flash.cache_lens,
        cu_seqlens_q=flash.cu,
        seqused_q=flash.used_q,
        attn_mask=flash.attn_mask,
        metadata=flash.schedule,
        head_dim_v=FLASH_MLA_V_DIM,
        softmax_scale=scale,
        mask_mode=3 if flash.causal else 0,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_out="NTD",
        return_softmax_lse=False,
    )
