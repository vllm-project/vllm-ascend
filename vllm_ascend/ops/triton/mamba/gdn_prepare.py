# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["NUM_REQS", "NUM_ROWS", "NUM_DECODES", "NUM_DECODE_TOKENS"])
def _prepare_gdn_kernel(
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
    NUM_REQS,
    NUM_ROWS,
    NUM_DECODES,
    NUM_DECODE_TOKENS,
    HAS_SPEC: tl.constexpr,
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
    Batch counts stay dynamic; phase flags specialize optional tensor accesses.
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
        is_spec: tl.constexpr = HAS_SPEC
        if MIXED_SPEC:
            group_row = tl.load(layout + row * 2)
            is_spec = group_row >= 0
            dst = tl.where(is_spec, group_row, -group_row - 1)
            begin = tl.load(layout + row * 2 + 1)
        else:
            dst = row
        if HAS_SPEC:
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


def prepare_gdn(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query: torch.Tensor,
    accepted: torch.Tensor | None,
    layout: torch.Tensor | None,
    spec_states: torch.Tensor,
    non_spec_states: torch.Tensor,
    conv_indices: torch.Tensor,
    sequence_masks: torch.Tensor,
    spec_query: torch.Tensor,
    non_spec_query: torch.Tensor,
    spec_lengths: torch.Tensor,
    non_spec_lengths: torch.Tensor,
    accepted_out: torch.Tensor,
    spec_tokens: torch.Tensor,
    non_spec_tokens: torch.Tensor,
    initial_state: torch.Tensor | None,
    prefill_query: torch.Tensor | None,
    conv_output: torch.Tensor | None,
    *,
    num_reqs: int,
    num_spec_decodes: int,
    num_decodes: int,
    num_decode_tokens: int,
    has_prefill: bool,
    spec_width: int,
    mixed_spec: bool,
    block_size: int,
    pad_block_id: int,
) -> None:
    """Populate GDN metadata buffers through a single Triton launch.

    Output buffers may include graph-padding rows beyond the live requests.
    Optional tensors can be None when their corresponding phase is disabled.
    Launch sizing and tensor strides are internal to this entry point.
    """
    rows = max(num_reqs, spec_states.shape[0], non_spec_states.shape[0])
    conv_numel = conv_output.numel() if conv_output is not None else 0
    block = 256
    clear_block = 1024
    _prepare_gdn_kernel[(rows + (conv_numel + clear_block - 1) // clear_block,)](
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
        NUM_REQS=num_reqs,
        NUM_ROWS=rows,
        NUM_DECODES=num_decodes,
        NUM_DECODE_TOKENS=num_decode_tokens,
        HAS_SPEC=num_spec_decodes > 0,
        HAS_PREFILL=has_prefill,
        SPEC_WIDTH=spec_width,
        MIXED_SPEC=mixed_spec,
        BLOCK_SIZE=block_size,
        TABLE_ROW_STRIDE=block_table.stride(0),
        TABLE_COL_STRIDE=block_table.stride(1),
        SEQ_LEN_STRIDE=seq_lens.stride(0),
        ACCEPTED_STRIDE=accepted.stride(0) if accepted is not None else 1,
        PAD_BLOCK_ID=pad_block_id,
        STATE_BLOCK=1 << (spec_width - 1).bit_length(),
        CONV_NUMEL=conv_numel,
        BLOCK=block,
        CLEAR_BLOCK=clear_block,
    )
