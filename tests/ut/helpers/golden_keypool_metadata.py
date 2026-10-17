# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU reference for fused KeyPool metadata and causal CP buffer contracts."""

from dataclasses import dataclass

import torch

from vllm_ascend.attention.context_parallel.common_cp import get_cp_local_query_key_lens


def build_keypool_metadata(common, buffers, tokens, source, logical, ratio, *, query_metadata=None, local_start=0):
    if common.positions.device.type == "npu":
        from vllm_ascend.ops.triton.glm5_next_cann_keypool import (  # type: ignore[import-untyped]
            build_keypool_metadata as fused_metadata,
        )

        return fused_metadata(common, buffers, tokens, source, logical, ratio)
    n = common.num_reqs
    positions, slots = buffers.positions[:tokens], buffers.slot_mapping[:tokens]
    seq, raw = buffers.seq_lens[:n], buffers.raw_seq_lens[:n]
    query, ends = buffers.query_start_loc[: n + 1], buffers.cum_query_lens[:n]
    starts, tails = buffers.start_pos[:n], buffers.pool_tail[:n]
    positions[:source].copy_(common.positions[:source])
    positions[source:].zero_()
    slot = common.slot_mapping[:source]
    slots.fill_(-1)
    slots[:source].copy_(
        torch.where(
            (slot >= 0) & ((positions[:source] + 1) % ratio == 0),
            slot // logical * (logical // ratio) + slot % logical // ratio,
            -1,
        )
    )
    raw.copy_(common.seq_lens[:n])
    seq.copy_(raw // ratio)
    query.copy_(common.query_start_loc[: n + 1])
    query.masked_fill_(query < 0, common.num_actual_tokens).clamp_max_(common.num_actual_tokens)
    ends.copy_(query[1:])
    starts.copy_(raw - query.diff())
    tails.copy_(raw % ratio)
    buffers.start_offsets[:n].copy_(starts % ratio)
    completed = ((positions + 1) % ratio == 0) & (buffers.token_indices[:tokens] < query[-1])
    pooled = completed.to(torch.int32).cumsum(0, dtype=torch.int32).sub_(1).clamp_min_(0).long()
    # Incomplete tokens write to a discarded sentinel; completed rows are unique.
    compact = buffers.pooled_slot_mapping[: tokens + 1]
    compact.fill_(-1)
    if tokens:
        compact.scatter_(0, torch.where(completed, pooled + 1, 0), slots)
    if query_metadata is not None:
        local_ends, local_keys = get_cp_local_query_key_lens(
            query, query[1:], raw, local_start, local_start + query_metadata.positions.numel()
        )
        query_metadata.cum_query_lens.copy_(local_ends)
        query_metadata.seq_lens.copy_(local_keys // ratio)
        query_metadata.pool_tail.copy_(local_keys % ratio)
    return positions, slots, seq, raw, query, ends, starts, tails


@dataclass
class KeypoolTailInputs:
    history_indices: torch.Tensor
    write_indices: torch.Tensor
    state_indices: torch.Tensor
    retained_indices: torch.Tensor | None
    page_rows: int


def build_tail_inputs(common, pool, capacity, page_rows, retained_width, buffers):
    if common.seq_lens.device.type == "npu":
        from vllm_ascend.ops.triton.glm5_next_cann_keypool import (  # type: ignore[import-untyped]
            build_tail_inputs as fused_inputs,
        )

        return fused_inputs(common, pool, capacity, page_rows, retained_width, buffers)
    n = common.num_reqs
    query = common.query_start_loc[: n + 1].clamp(0, common.num_actual_tokens)
    query = torch.where(common.query_start_loc[: n + 1] < 0, common.num_actual_tokens, query)
    lengths = query.diff()
    starts = common.seq_lens[:n] - lengths
    pages = common.block_table_tensor[:n, 0]
    columns = torch.arange(max(pool, retained_width), device=common.seq_lens.device)
    history = buffers["history"][:n]
    history.copy_(
        pages.clamp_min(0)[:, None] * page_rows + ((starts - starts % pool)[:, None] + columns[:pool]) % capacity
    )
    width = retained_width if retained_width else pool
    writes, state = buffers["writes"][:n, :width], buffers["state"][:n, :width]
    retained = None
    if retained_width:
        raw = query[1:, None] - width + columns[:width]
        # Shape the prefix as contiguous rows to avoid per-layer flatten copies.
        retained = buffers["retained"].view(-1)[: n * width].view(n, width)
        retained.copy_(raw.clamp_min(0))
        slots = torch.index_select(common.slot_mapping, 0, retained.flatten()).view(n, width)
        writes.copy_(slots // capacity * page_rows + slots % capacity)
        writes.masked_fill_(raw < query[:-1, None], -1)
    else:
        end = starts % pool + lengths
        requests = torch.arange(n, device=common.seq_lens.device)
        state.copy_((requests + 1 + (end >= pool) * n)[:, None] * pool + columns[:pool])
        positions = (starts + lengths - end % pool)[:, None] + columns[:pool]
        writes.copy_(pages[:, None] * page_rows + positions % capacity)
        writes.masked_fill_(
            (lengths[:, None] == 0) | (columns[:pool] >= (end % pool)[:, None]) | (pages[:, None] < 0), -1
        )
    return KeypoolTailInputs(history, writes, state, retained, page_rows)
