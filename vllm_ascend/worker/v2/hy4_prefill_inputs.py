# SPDX-License-Identifier: Apache-2.0
"""Eager prompt-only input preparation."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import _load_ptr

from .hy4_prefill_slot_mappings import compute_prefill_slot_mappings
from vllm_ascend.ops.triton.hy4_vocab_lookup import rank_local_vocab_index


@triton.jit
def _prepare_prefill_row(
    row, input_ids, positions, seq_lens, logits_indices, next_tokens, history,
    mapping, starts, all_ids, prefill_lens, num_computed,
    rope_positions, rope_source, rope_delta, vocab_indices,
    VOCAB_START: tl.constexpr, VOCAB_END: tl.constexpr,
    ADDED_START: tl.constexpr, ADDED_END: tl.constexpr, ADDED_LOCAL: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
    ROPE_DIMS: tl.constexpr, ROPE_STRIDE: tl.constexpr, ROPE_SOURCE_STRIDE: tl.constexpr,
    ALL_STRIDE: tl.constexpr, NEXT_STRIDE: tl.constexpr,
    LOOKAHEAD: tl.constexpr, LOOKAHEAD_BLOCK: tl.constexpr,
    REQS: tl.constexpr, MAX_REQS: tl.constexpr, BLOCK: tl.constexpr,
    DEPTH: tl.constexpr, HISTORY_BLOCK: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK)
    if row == REQS:
        tail = row + lanes
        tl.store(seq_lens + tail, 0, tail < MAX_REQS)
    else:
        req = tl.load(mapping + row)
        start = tl.load(starts + row)
        end = tl.load(starts + row + 1)
        computed = tl.load(num_computed + req)
        prompt_end = tl.load(prefill_lens + req)
        seq_len = computed + end - start
        req_ids = all_ids + req.to(tl.int64) * ALL_STRIDE
        if ROPE_DIMS:
            delta = tl.load(rope_delta + req)
            rope_row = rope_source + req.to(tl.int64) * ROPE_DIMS * ROPE_SOURCE_STRIDE
        for offset in range(0, end - start, BLOCK):
            lane = offset + lanes
            mask = start + lane < end
            token = tl.load(req_ids + computed + lane, mask=mask)
            tl.store(input_ids + start + lane, token, mask)
            if VOCAB_PAD >= 0:
                rank_index, _ = rank_local_vocab_index(
                    token, VOCAB_START, VOCAB_END, ADDED_START, ADDED_END, ADDED_LOCAL, VOCAB_PAD,
                )
                tl.store(vocab_indices + start + lane, rank_index, mask)
            tl.store(positions + start + lane, computed + lane, mask)
            if ROPE_DIMS:
                for dim in tl.static_range(ROPE_DIMS):
                    if computed < prompt_end:
                        pos = tl.load(rope_row + dim * ROPE_SOURCE_STRIDE + computed + lane, mask=mask)
                    else:
                        pos = computed + lane + delta
                    tl.store(rope_positions + dim * ROPE_STRIDE + start + lane, pos, mask)
        tl.store(seq_lens + row, seq_len)
        tl.store(logits_indices + row, end - 1)
        lookahead = tl.arange(0, LOOKAHEAD_BLOCK)
        token = tl.load(req_ids + seq_len + lookahead,
                        (lookahead < LOOKAHEAD) & (seq_len + lookahead < prompt_end), 0)
        tl.store(next_tokens + lookahead * NEXT_STRIDE + req, token, lookahead < LOOKAHEAD)
        if DEPTH:
            previous = tl.arange(0, HISTORY_BLOCK)
            past = computed - 1 - previous
            token = tl.load(req_ids + tl.maximum(past, 0), (previous < DEPTH) & (past >= 0), -1)
            tl.store(history + row * DEPTH + previous, token, previous < DEPTH)


@triton.jit
def _hy4_prepare_prefill_kernel(
    input_ids, positions, seq_lens, logits_indices, next_tokens, history,
    mapping, starts, all_ids, prefill_lens, num_computed,
    ALL_STRIDE: tl.constexpr, NEXT_STRIDE: tl.constexpr,
    LOOKAHEAD: tl.constexpr, LOOKAHEAD_BLOCK: tl.constexpr,
    REQS: tl.constexpr, MAX_REQS: tl.constexpr, BLOCK: tl.constexpr,
    DEPTH: tl.constexpr, HISTORY_BLOCK: tl.constexpr,
):
    _prepare_prefill_row(
        tl.program_id(0), input_ids, positions, seq_lens, logits_indices, next_tokens, history,
        mapping, starts, all_ids, prefill_lens, num_computed,
        None, None, None, None, 0, 0, 0, 0, 0, -1, 0, 0, 0,
        ALL_STRIDE, NEXT_STRIDE, LOOKAHEAD, LOOKAHEAD_BLOCK, REQS, MAX_REQS, BLOCK, DEPTH, HISTORY_BLOCK,
    )


@triton.jit
def _gather_prefill_block_table(
    group, row, mapping, source_tables, batch_tables, strides, num_blocks, window_reach,
    block_sizes, num_computed, NUM_BLOCKS_STRIDE: tl.constexpr, REQS: tl.constexpr,
    MASK_SWA: tl.constexpr, BLOCK: tl.constexpr,
):
    # vLLM BlockTables gather, expressed without early returns so Ascend
    # Triton can inline it into the entry kernel.
    stride = tl.load(strides + group)
    target = _load_ptr(batch_tables + group, tl.int32) + row * stride
    lanes = tl.arange(0, BLOCK)
    if row >= REQS:
        for pad_start in tl.range(0, stride, BLOCK):
            pad_offsets = pad_start + lanes
            tl.store(target + pad_offsets, 0, pad_offsets < stride)
    else:
        req = tl.load(mapping + row)
        count = tl.load(num_blocks + group * NUM_BLOCKS_STRIDE + req)
        source = _load_ptr(source_tables + group, tl.int32) + req * stride
        masked_blocks = 0
        if MASK_SWA:
            reach = tl.load(window_reach + group)
            first_pos = tl.load(num_computed + req)
            block_size = tl.load(block_sizes + group)
            masked_blocks = tl.where(reach > 0, tl.maximum(first_pos - reach, 0) // block_size, 0)
        for i in tl.range(0, count, BLOCK):
            offsets = i + lanes
            values = tl.load(source + offsets, (offsets >= masked_blocks) & (offsets < count), other=0)
            tl.store(target + offsets, values, offsets < count)


@triton.jit
def _hy4_prepare_prefill_entry_kernel(
    input_ids, positions, seq_lens, logits_indices, next_tokens, history,
    mapping, starts, all_ids, prefill_lens, num_computed,
    source_tables, batch_tables, table_strides, num_blocks, window_reach,
    block_sizes, kernel_block_sizes, slot_mappings, slot_mapping_enabled, dcp_sharded,
    rope_positions, rope_source, rope_delta, vocab_indices,
    VOCAB_START: tl.constexpr, VOCAB_END: tl.constexpr,
    ADDED_START: tl.constexpr, ADDED_END: tl.constexpr, ADDED_LOCAL: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
    ROPE_DIMS: tl.constexpr, ROPE_STRIDE: tl.constexpr, ROPE_SOURCE_STRIDE: tl.constexpr,
    GATHER_ROWS: tl.constexpr, GROUPS: tl.constexpr, NUM_BLOCKS_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr, SLOT_CAPACITY: tl.constexpr, SLOT_WINDOW: tl.constexpr,
    MASK_SWA: tl.constexpr, PAD_ID: tl.constexpr,
    ALL_STRIDE: tl.constexpr, NEXT_STRIDE: tl.constexpr,
    LOOKAHEAD: tl.constexpr, LOOKAHEAD_BLOCK: tl.constexpr,
    REQS: tl.constexpr, MAX_REQS: tl.constexpr, BLOCK: tl.constexpr,
    DEPTH: tl.constexpr, HISTORY_BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    row = tl.program_id(1)
    if group == GROUPS:
        if row <= REQS:
            _prepare_prefill_row(
                row, input_ids, positions, seq_lens, logits_indices, next_tokens, history,
                mapping, starts, all_ids, prefill_lens, num_computed,
                rope_positions, rope_source, rope_delta, vocab_indices,
                VOCAB_START, VOCAB_END, ADDED_START, ADDED_END, ADDED_LOCAL, VOCAB_PAD,
                ROPE_DIMS, ROPE_STRIDE, ROPE_SOURCE_STRIDE,
                ALL_STRIDE, NEXT_STRIDE, LOOKAHEAD, LOOKAHEAD_BLOCK, REQS, MAX_REQS, BLOCK, DEPTH, HISTORY_BLOCK,
            )
    else:
        if row < GATHER_ROWS:
            _gather_prefill_block_table(
                group, row, mapping, source_tables, batch_tables, table_strides, num_blocks,
                window_reach, block_sizes, num_computed, NUM_BLOCKS_STRIDE, REQS, MASK_SWA, BLOCK,
            )
        if row < REQS or row == tl.num_programs(1) - 1:
            # Derive positions without reading the output of another program.
            compute_prefill_slot_mappings(
                SLOT_CAPACITY, mapping, starts, positions, source_tables, table_strides,
                block_sizes, kernel_block_sizes, slot_mappings, SLOT_STRIDE, 0,
                CP_SIZE=1, CP_INTERLEAVE=1, PAD_ID=PAD_ID, TRITON_BLOCK_SIZE=BLOCK,
                BLOCK_TABLE_WINDOW_SIZE=SLOT_WINDOW, slot_mapping_enabled=slot_mapping_enabled,
                HAS_SLOT_MAPPING_ENABLED=True, dcp_sharded=dcp_sharded,
                num_computed=num_computed, PREFILL_REQS=REQS,
            )


def prepare_hy4_prefill_inputs(
    buffers, states, idx_mapping, query_starts, *, lookback_depth=0, tables=None, num_tokens=0, rope=None,
    vocab=(),
):
    rows = idx_mapping.shape[0]
    logits_indices = torch.empty(rows, dtype=torch.int64, device=idx_mapping.device)
    lookahead = states.next_prefill_tokens.shape[0]
    history = None
    if lookback_depth:
        history = torch.empty((rows, lookback_depth), device=idx_mapping.device, dtype=torch.int32)
    input_args = (
        buffers.input_ids, buffers.positions, buffers.seq_lens, logits_indices,
        states.next_prefill_tokens, history, idx_mapping, query_starts, states.all_token_ids.gpu,
        states.prefill_len.gpu, states.num_computed_tokens.gpu,
    )
    input_geometry = dict(
        ALL_STRIDE=states.all_token_ids.gpu.stride(0), NEXT_STRIDE=states.next_prefill_tokens.stride(0),
        LOOKAHEAD=lookahead, LOOKAHEAD_BLOCK=triton.next_power_of_2(max(1, lookahead)),
        REQS=rows, MAX_REQS=buffers.seq_lens.shape[0], BLOCK=1024,
        DEPTH=lookback_depth, HISTORY_BLOCK=triton.next_power_of_2(max(1, lookback_depth)),
    )
    attention_inputs = None
    vocab_indices = None
    if tables is None:
        _hy4_prepare_prefill_kernel[(rows + 1,)](*input_args, **input_geometry)
    else:
        groups = tables.num_kv_cache_groups
        gather_rows = max(rows, tables._num_dirty_rows)
        if rope is None:
            rope_tensors, rope_dims, rope_stride, rope_source_stride = (None, None, None), 0, 0, 0
        else:
            rope_tensors = (rope.positions, rope.prefill_positions.gpu, rope.prefill_delta.gpu)
            rope_dims = rope.num_dims
            rope_stride = rope.positions.stride(0)
            rope_source_stride = rope.max_model_len
        if vocab:
            vocab_indices = torch.empty(num_tokens, dtype=torch.int32, device=idx_mapping.device)
        vocab_start, vocab_end, added_start, added_end, added_local, vocab_pad = vocab or (0, 0, 0, 0, 0, -1)
        _hy4_prepare_prefill_entry_kernel[(groups + 1, max(rows + 1, gather_rows))](
            *input_args, tables.block_table_ptrs, tables.input_block_table_ptrs, tables.block_table_strides,
            tables.num_blocks.gpu, tables.sliding_window_reach, tables.block_sizes_tensor,
            tables.kernel_block_sizes_tensor, tables.slot_mappings,
            tables.slot_mapping_enabled.view(torch.uint8), tables.dcp_sharded.view(torch.uint8),
            *rope_tensors, vocab_indices,
            VOCAB_START=vocab_start, VOCAB_END=vocab_end, ADDED_START=added_start,
            ADDED_END=added_end, ADDED_LOCAL=added_local, VOCAB_PAD=vocab_pad,
            ROPE_DIMS=rope_dims, ROPE_STRIDE=rope_stride, ROPE_SOURCE_STRIDE=rope_source_stride,
            GATHER_ROWS=gather_rows, GROUPS=groups, NUM_BLOCKS_STRIDE=tables.num_blocks.gpu.stride(0),
            SLOT_STRIDE=tables.slot_mappings.stride(0), SLOT_CAPACITY=tables.slot_mappings.shape[1],
            SLOT_WINDOW=tables._block_table_window_size, MASK_SWA=tables.has_sliding_window_groups,
            PAD_ID=PAD_SLOT_ID,
            **input_geometry,
        )
        tables._num_dirty_rows = rows
        attention_inputs = (
            tuple(table[:rows] for table in tables.input_block_tables),
            tables.slot_mappings[:, :num_tokens],
        )
    return logits_indices, history, attention_inputs, vocab_indices
