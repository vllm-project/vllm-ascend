# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_gdn_metadata_kernel(
    block_table,
    seq_lens,
    query,
    accepted,
    layout,
    spec_states,
    non_spec_states,
    conv_indices,
    sequence_masks,
    spec_query,
    non_spec_query,
    spec_lengths,
    non_spec_lengths,
    accepted_out,
    spec_tokens,
    non_spec_tokens,
    initial_state,
    prefill_query,
    conv_output,
    NUM_REQS: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    NUM_SPEC: tl.constexpr,
    NUM_DECODES: tl.constexpr,
    NUM_DECODE_TOKENS: tl.constexpr,
    HAS_PREFILL: tl.constexpr,
    SPEC_WIDTH: tl.constexpr,
    MIXED_SPEC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TABLE_ROW_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    SEQ_LEN_STRIDE: tl.constexpr,
    ACCEPTED_STRIDE: tl.constexpr,
    PAD_BLOCK_ID: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
    CONV_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
    CLEAR_BLOCK: tl.constexpr,
):
    """Materialize aligned-cache metadata, stable token partitions and padding.

    Request programs consume actual query boundaries. Mixed speculative batches
    use a CPU partition (signed group row, group token offset); homogeneous
    batches need no layout transfer. Each request writes its own token range;
    independent clear programs zero the shared output in the same launch.
    """
    row = tl.program_id(0)
    if row < NUM_ROWS:
        active = row < NUM_REQS
        begin = tl.load(query + tl.minimum(row, NUM_REQS))
        end = tl.load(query + tl.minimum(row + 1, NUM_REQS))
        length = end - begin
        source_begin = begin
        seq_len = tl.load(seq_lens + row * SEQ_LEN_STRIDE, active, 0)
        start_block = tl.maximum((seq_len - 1) // BLOCK_SIZE, 0)
        if MIXED_SPEC:
            group_row = tl.load(layout + row * 2)
            is_spec = group_row >= 0
            dst = tl.where(is_spec, group_row, -group_row - 1)
            begin = tl.load(layout + row * 2 + 1)
        else:
            is_spec: tl.constexpr = NUM_SPEC > 0
            dst = row
        if NUM_SPEC:
            tl.store(sequence_masks + row, active & is_spec)
        if is_spec:
            cols = tl.arange(0, STATE_BLOCK)
            states = tl.load(
                block_table + row * TABLE_ROW_STRIDE + (start_block + cols) * TABLE_COL_STRIDE,
                active & (cols < SPEC_WIDTH),
                PAD_BLOCK_ID,
            )
            tl.store(spec_states + dst * SPEC_WIDTH + cols, states, cols < SPEC_WIDTH)
            if not MIXED_SPEC:
                tl.store(spec_tokens + begin + cols, begin + cols, active & (cols < length))
            first_state = tl.load(
                block_table + row * TABLE_ROW_STRIDE + start_block * TABLE_COL_STRIDE, active, PAD_BLOCK_ID
            )
            tl.store(conv_indices + dst, first_state)
            tl.store(accepted_out + dst, tl.load(accepted + row * ACCEPTED_STRIDE, active, 1))
            tl.store(spec_query + dst + 1, begin + length)
            tl.store(spec_lengths + dst + 1, length)
            if dst == 0:
                tl.store(spec_query, 0)
                tl.store(spec_lengths, 0)
        else:
            state = tl.load(block_table + row * TABLE_ROW_STRIDE + start_block * TABLE_COL_STRIDE, active, PAD_BLOCK_ID)
            tl.store(non_spec_states + dst, state)
            tl.store(non_spec_query + dst + 1, begin + length)
            tl.store(non_spec_lengths + dst + 1, length)
            if dst == 0:
                tl.store(non_spec_query, 0)
                tl.store(non_spec_lengths, 0)
            if HAS_PREFILL:
                tl.store(initial_state + dst, seq_len > length)
                if dst >= NUM_DECODES:
                    tl.store(prefill_query + dst - NUM_DECODES + 1, begin + length - NUM_DECODE_TOKENS)
                    if dst == NUM_DECODES:
                        tl.store(prefill_query, 0)
        if MIXED_SPEC:
            for start in range(0, length, BLOCK):
                offsets = start + tl.arange(0, BLOCK)
                tokens = source_begin + offsets
                tl.store(spec_tokens + begin + offsets, tokens, (offsets < length) & is_spec)
                tl.store(non_spec_tokens + begin + offsets, tokens, (offsets < length) & ~is_spec)
    elif CONV_NUMEL:
        offsets = (row - NUM_ROWS) * CLEAR_BLOCK + tl.arange(0, CLEAR_BLOCK)
        tl.store(conv_output + offsets, 0, offsets < CONV_NUMEL)
