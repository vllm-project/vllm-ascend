# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External FlashMLA boundary retained from the public handoff.

See docs/flash_mla_versions.json for the pinned sources used for integration.

Persistent cache ownership remains with PR #16456. Capacity fields size local
buffers; the public operator's max_seqlen attributes always use -1.
"""

from dataclasses import dataclass

import torch
from vllm.utils.math_utils import cdiv

from vllm_ascend.attention.context_parallel.common_cp import get_flash_dcp_local_seq_lens
from vllm_ascend.attention.utils import MLA_FLASH_SUPPORTED_Q_HEADS
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, DeviceMetadataTask


@dataclass
class AscendFlashAttentionMetadata:
    """Stable FA/MLA buffers populated by the existing device metadata executor."""

    query: torch.Tensor
    schedule: torch.Tensor
    cu: torch.Tensor
    used_q: torch.Tensor
    cache_lens: torch.Tensor
    block_table: torch.Tensor
    slots: torch.Tensor
    live_boundaries: torch.Tensor
    token_live: torch.Tensor
    positions: torch.Tensor
    attn_mask: torch.Tensor
    max_query_len: int
    max_seq_len: int
    is_prefill: bool
    dcp_size: int = 1
    split_kv: bool = False
    current_schedule: torch.Tensor | None = None
    current_cache: torch.Tensor | None = None
    current_block_table: torch.Tensor | None = None
    current_slots: torch.Tensor | None = None
    causal: bool = True


def _flash_attention_schedule(builder, flash, *, current=False, meta=False):
    """Use the installed operator's own Meta implementation to size buffers."""

    def tensor(value):
        return torch.empty_like(value, device="meta") if meta else value

    lengths = flash.used_q if current else flash.cache_lens
    mask_mode = 3 if current else (0 if flash.split_kv or flash.dcp_size > 1 else flash.causal * 3)
    heads = builder.flash_num_heads if current else builder.flash_num_heads * flash.dcp_size
    from cann_ops_transformer.ops import flash_mla_with_kvcache_metadata

    return flash_mla_with_kvcache_metadata(
        tensor(lengths),
        heads,
        1,
        cu_seqlens_q=tensor(flash.cu),
        seqused_q=tensor(flash.used_q),
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        head_dim_qk=576,
        head_dim_v=512,
        mask_mode=mask_mode,
        layout_q="TND",
    )


def _init_flash_attention_metadata(builder, impl) -> None:
    builder.flash_num_heads, builder.flash_num_kv_heads = impl.num_heads, impl.num_kv_heads
    if impl.num_heads not in MLA_FLASH_SUPPORTED_Q_HEADS:
        raise ValueError("FlashMLA requires actual local Q heads in {8, 12, 64, 96}")
    builder._flash_buffers = {}
    builder._flash_attn_mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8, device=builder.device), diagonal=1)


def _build_flash_attention_metadata(builder, common) -> AscendFlashAttentionMetadata:
    batch = common.num_reqs
    tokens = max(common.num_actual_tokens, common.num_input_tokens)
    table = common.block_table_tensor[:batch]
    dcp_size = getattr(builder, "dcp_size", 1)
    split_kv = common.causal and dcp_size > 1
    key = batch, tokens, table.shape[1], common.causal
    # Retaining every eager prefill shape also retains its DCP-gathered query.
    # Decode metadata needs stable addresses for graph replay.
    # Noncausal DSpark query groups can exceed the target decode threshold
    # while still using their captured graph, so preserve those buffers.
    # Keep eager buffers owned by the current metadata instead of the builder.
    is_eager_prefill = common.causal and common.max_query_len > builder.decode_threshold
    buffers = {} if is_eager_prefill else builder._flash_buffers
    if key not in buffers:
        # The final zero-used request owns physical graph/SP padding tokens.
        rows = batch + 1
        int_args = {"dtype": torch.int32, "device": builder.device}
        float_args = {"dtype": builder.kv_cache_spec.dtype, "device": builder.device}
        head_dim = 576
        shape = (tokens, builder.flash_num_heads * dcp_size, head_dim)
        block_size = builder.kernel_block_size or builder.kv_cache_spec.block_size
        buffers[key] = AscendFlashAttentionMetadata(
            query=torch.empty(shape, **float_args),
            schedule=torch.empty(0, **int_args),
            cu=torch.zeros(rows + 1, **int_args),
            used_q=torch.zeros(rows, **int_args),
            cache_lens=torch.zeros(rows, **int_args),
            block_table=torch.zeros((rows, table.shape[1]), **int_args),
            slots=torch.full((tokens,), -1, dtype=torch.int64, device=builder.device),
            live_boundaries=torch.zeros(tokens + 1, **int_args),
            token_live=torch.zeros(tokens, dtype=torch.bool, device=builder.device),
            positions=torch.zeros(tokens, dtype=torch.int64, device=builder.device),
            attn_mask=builder._flash_attn_mask,
            max_query_len=tokens,
            max_seq_len=table.shape[1] * block_size,
            is_prefill=common.max_query_len > builder.decode_threshold,
            dcp_size=dcp_size,
            split_kv=split_kv,
            causal=common.causal,
        )
        flash = buffers[key]
        flash.schedule = torch.empty_like(_flash_attention_schedule(builder, flash, meta=True), device=builder.device)
        if split_kv:
            flash.current_schedule = torch.empty_like(
                _flash_attention_schedule(builder, flash, current=True, meta=True), device=builder.device
            )
            # FlashMLA consumes paged KV only. This temporary current chunk
            # does not change persistent cache allocation or page identities.
            current_block_size = 128
            pages = cdiv(tokens, current_block_size) + rows
            flash.current_cache = torch.empty((pages, current_block_size, 1, head_dim), **float_args)
            flash.current_block_table = torch.zeros((rows, cdiv(tokens, current_block_size)), **int_args)
            flash.current_slots = torch.full_like(flash.slots, -1)
    flash = buffers[key]
    flash.is_prefill = common.max_query_len > builder.decode_threshold

    def build_metadata() -> None:
        # Read current device lengths after async rejection correction. Never
        # derive visibility from the CPU mirror, including DSpark query rebuilds.
        flash.cu[: batch + 1].copy_(common.query_start_loc[: batch + 1])
        flash.cu[batch + 1].fill_(tokens)
        flash.used_q[:batch].copy_(flash.cu[1 : batch + 1] - flash.cu[:batch])
        flash.used_q[:batch].masked_fill_(common.seq_lens[:batch] <= 0, 0)
        flash.used_q[batch:].zero_()
        lengths = common.seq_lens[:batch]
        if split_kv:
            lengths = (lengths - flash.used_q[:batch]).clamp_min(0)
        if dcp_size > 1:
            lengths = get_flash_dcp_local_seq_lens(
                lengths, dcp_size, builder.vllm_config.parallel_config.cp_kv_cache_interleave_size
            )[:, builder.dcp_rank]
        flash.cache_lens[:batch].copy_(lengths)
        flash.cache_lens[batch:].zero_()
        flash.block_table[:batch].copy_(table)
        flash.block_table[batch:].zero_()
        flash.slots.fill_(-1)
        slots = common.slot_mapping[:tokens]
        flash.slots[: slots.shape[0]].copy_(slots)
        flash.live_boundaries.zero_()
        live_rows = (flash.used_q > 0).to(torch.int32)
        flash.live_boundaries.scatter_add_(0, flash.cu[:-1].long(), live_rows)
        flash.live_boundaries.scatter_add_(0, (flash.cu[:-1] + flash.used_q).long(), -live_rows)
        flash.token_live.copy_(flash.live_boundaries.cumsum(0)[:tokens] > 0)
        flash.slots.masked_fill_(~flash.token_live, -1)
        flash.positions.zero_()
        positions = common.positions[:tokens]
        flash.positions[: positions.shape[0]].copy_(positions)
        flash.schedule.copy_(_flash_attention_schedule(builder, flash))
        if split_kv:
            if flash.current_cache is not None:
                block_size = flash.current_cache.shape[1]
                page_counts = (flash.used_q + block_size - 1) // block_size
                page_starts = page_counts.cumsum(0) - page_counts
                columns = torch.arange(flash.current_block_table.shape[1], device=builder.device)
                flash.current_block_table.copy_(page_starts[:, None] + columns)
                flash.current_block_table.masked_fill_(columns >= page_counts[:, None], 0)
                indices = torch.arange(tokens, device=builder.device)
                requests = torch.searchsorted(flash.cu[1:], indices.to(torch.int32), right=True)
                flash.current_slots.copy_(page_starts[requests] * block_size + indices - flash.cu[requests])
                flash.current_slots.masked_fill_(~flash.token_live, -1)
            flash.current_schedule.copy_(_flash_attention_schedule(builder, flash, current=True))

    if builder._device_metadata_enabled:
        builder._device_metadata_tasks = (
            DeviceMetadataTask(DeviceMetadataStage.ATTENTION, build_metadata, id(flash.schedule)),
        )
    else:
        build_metadata()
    return flash


def validate_flash_cache(cache: torch.Tensor) -> None:
    """Check the view, never clone, repack, or change a persistent cache."""
    if not isinstance(cache, torch.Tensor) or cache.ndim != 4 or cache.shape[1:] != (128, 1, 576):
        raise ValueError("FlashMLA requires the token-fused BBND [P,128,1,576] cache from PR #16456")
    page_span = 127 * cache.stride(1) + 576
    if cache.stride(-1) != 1 or cache.stride(1) < 576 or cache.stride(0) < page_span:
        raise ValueError("FlashMLA requires non-overlapping BBND pages and contiguous channels")


def run_flash_mla(query, cache, flash, scale, *, current=False):
    from cann_ops_transformer.ops import flash_mla_with_kvcache

    validate_flash_cache(cache)
    heads = query.shape[1]
    if heads not in MLA_FLASH_SUPPORTED_Q_HEADS:
        raise ValueError(f"Unsupported actual FlashMLA Q heads: {heads}")
    mask_mode = 3 if current else (0 if flash.split_kv or flash.dcp_size > 1 else flash.causal * 3)
    return flash_mla_with_kvcache(
        query,
        cache,
        block_table=flash.current_block_table if current else flash.block_table,
        cache_seqlens=flash.used_q if current else flash.cache_lens,
        cu_seqlens_q=flash.cu,
        seqused_q=flash.used_q,
        attn_mask=flash.attn_mask if mask_mode else None,
        metadata=flash.current_schedule if current else flash.schedule,
        head_dim_v=512,
        softmax_scale=scale,
        mask_mode=mask_mode,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_out="NTD",
        return_softmax_lse=flash.dcp_size > 1 or flash.split_kv,
    )
