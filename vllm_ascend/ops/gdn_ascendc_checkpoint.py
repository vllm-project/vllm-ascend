# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Export FP32 checkpoints with the existing AscendC recurrence interface."""

from dataclasses import dataclass

import torch

GDN_CHUNK_SIZE = 64


@dataclass
class AscendCRecurrenceSegment:
    rows: torch.Tensor
    tokens: torch.Tensor
    chunks: torch.Tensor
    cu_seqlens: tuple[int, ...]
    chunk_indices: tuple[int, ...]


@dataclass
class AscendCCheckpointPlan:
    prefix: AscendCRecurrenceSegment
    tail: AscendCRecurrenceSegment
    num_chunks: int


def build_ascendc_checkpoint_plan(
    cu_seqlens: tuple[int, ...], offsets: list[int], device: torch.device
) -> AscendCCheckpointPlan:
    """Plan both recurrence calls on the CPU once for all GDN layers."""
    if len(offsets) + 1 != len(cu_seqlens) or cu_seqlens[0] != 0:
        raise ValueError("Checkpoint offsets must match the packed sequence rows")
    prefix_rows, prefix_tokens, prefix_chunks = [], [], []
    tail_rows, tail_tokens, tail_chunks = [], [], []
    prefix_cu, tail_cu = [0], [0]
    prefix_indices, tail_indices = [], []
    chunk_base = 0
    for row, (start, end, offset) in enumerate(zip(cu_seqlens, cu_seqlens[1:], offsets)):
        length = end - start
        if length < 0 or offset < 0 or (offset and (offset >= length or offset % GDN_CHUNK_SIZE)):
            raise ValueError("Checkpoint must be an internal 64-token boundary")
        split = offset or length
        count = (length + GDN_CHUNK_SIZE - 1) // GDN_CHUNK_SIZE
        prefix_count = (split + GDN_CHUNK_SIZE - 1) // GDN_CHUNK_SIZE
        if split:
            segment_row = len(prefix_rows)
            prefix_rows.append(row)
            prefix_tokens.extend(range(start, start + split))
            prefix_chunks.extend(range(chunk_base, chunk_base + prefix_count))
            prefix_cu.append(prefix_cu[-1] + split)
            prefix_indices.extend(value for i in range(prefix_count) for value in (segment_row, i))
        if offset:
            segment_row = len(tail_rows)
            tail_rows.append(row)
            tail_tokens.extend(range(start + offset, end))
            tail_chunks.extend(range(chunk_base + prefix_count, chunk_base + count))
            tail_cu.append(tail_cu[-1] + length - offset)
            tail_indices.extend(value for i in range(count - prefix_count) for value in (segment_row, i))
        chunk_base += count

    def segment(rows, tokens, chunks, cu, indices):
        return AscendCRecurrenceSegment(
            rows=torch.tensor(rows, dtype=torch.int64, device=device),
            tokens=torch.tensor(tokens, dtype=torch.int64, device=device),
            chunks=torch.tensor(chunks, dtype=torch.int64, device=device),
            cu_seqlens=tuple(cu),
            chunk_indices=tuple(indices),
        )

    return AscendCCheckpointPlan(
        segment(prefix_rows, prefix_tokens, prefix_chunks, prefix_cu, prefix_indices),
        segment(tail_rows, tail_tokens, tail_chunks, tail_cu, tail_indices),
        chunk_base,
    )


def ascendc_checkpoint_recurrence(k, w, u, g, initial_state, checkpoint_output, plan):
    """Run two batched recurrence segments inside one model forward.

    Inputs and H/V outputs use AscendC's head-first layout. A segment's final
    state retains FP32 precision, unlike the BF16 per-chunk H output.
    """
    if initial_state.dtype != torch.float32 or checkpoint_output.dtype != torch.float32:
        raise ValueError("AscendC checkpoint segments require FP32 recurrent state")
    h = k.new_empty((1, u.shape[1], plan.num_chunks, k.shape[-1], u.shape[-1]))
    v_new = torch.empty_like(u)
    final = initial_state.clone()

    def run(segment, states):
        return torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h_vllm(
            k.index_select(2, segment.tokens),
            w.index_select(2, segment.tokens),
            u.index_select(2, segment.tokens),
            g=g.index_select(2, segment.tokens),
            gk=None,
            initial_state=states.index_select(0, segment.rows),
            output_final_state=True,
            chunk_size=GDN_CHUNK_SIZE,
            save_new_value=True,
            cu_seqlens=segment.cu_seqlens,
            chunk_indices=segment.chunk_indices,
            use_exp2=False,
            transpose_state_layout=False,
        )

    prefix = plan.prefix
    if prefix.rows.numel():
        prefix_h, prefix_v, prefix_final = run(prefix, initial_state)
        h.index_copy_(2, prefix.chunks, prefix_h)
        v_new.index_copy_(2, prefix.tokens, prefix_v)
        final.index_copy_(0, prefix.rows, prefix_final)
    tail = plan.tail
    if tail.rows.numel():
        checkpoint_output.index_copy_(0, tail.rows, final.index_select(0, tail.rows).transpose(-1, -2))
        tail_h, tail_v, tail_final = run(tail, final)
        h.index_copy_(2, tail.chunks, tail_h)
        v_new.index_copy_(2, tail.tokens, tail_v)
        final.index_copy_(0, tail.rows, tail_final)
    return h, v_new, final
