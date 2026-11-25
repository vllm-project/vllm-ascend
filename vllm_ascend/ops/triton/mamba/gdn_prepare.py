# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.logger import logger
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["NUM_REQS", "NUM_ROWS", "NUM_DECODES", "NUM_DECODE_TOKENS", "LIVE_CONV_NUMEL"])
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
    group_pointers,
    NUM_REQS,
    NUM_ROWS,
    NUM_DECODES,
    NUM_DECODE_TOKENS,
    LIVE_CONV_NUMEL,
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
    GROUPED: tl.constexpr,
):
    """Materialize aligned-cache metadata, stable token partitions and padding.

    Request programs consume actual query boundaries. Mixed speculative batches
    use a CPU partition (signed group row, group token offset); homogeneous
    batches need no layout transfer. Each request writes its own token range;
    independent clear programs zero the shared output in the same launch.
    Batch counts stay dynamic; phase flags specialize optional tensor accesses.
    """
    if GROUPED:
        tl.static_assert(HAS_SPEC and not MIXED_SPEC and not HAS_PREFILL)
        # Pure spec batches share query/length/acceptance inputs; state IDs
        # and buffers captured by each cache group remain independently owned.
        group = group_pointers + tl.program_id(1) * 9
        block_table = tl.load(group).to(block_table.dtype)
        spec_states = tl.load(group + 1).to(spec_states.dtype)
        conv_indices = tl.load(group + 2).to(conv_indices.dtype)
        sequence_masks = tl.load(group + 3).to(sequence_masks.dtype)
        spec_query = tl.load(group + 4).to(spec_query.dtype)
        spec_lengths = tl.load(group + 5).to(spec_lengths.dtype)
        accepted_out = tl.load(group + 6).to(accepted_out.dtype)
        spec_tokens = tl.load(group + 7).to(spec_tokens.dtype)
        conv_output = tl.load(group + 8).to(conv_output.dtype)
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
        clear_numel = LIVE_CONV_NUMEL if GROUPED else CONV_NUMEL
        tl.store(conv_output + offsets, 0, offsets < clear_numel)


class GroupedSpecGDN:
    """Combine pure spec metadata launches without changing graph-owned buffers."""

    def __init__(self) -> None:
        self._pending = []
        self._addresses: tuple[tuple[int, ...], ...] | None = None
        self._pointers: torch.Tensor | None = None

    def reset(self) -> None:
        self._pending.clear()

    def stage(self, grid, arguments, geometry, group_tensors, shared_inputs) -> None:
        self._pending.append((grid, arguments, geometry, group_tensors, shared_inputs))

    def launch(self, *, clear_conv_tokens: int | None = None) -> None:
        pending, self._pending = self._pending, []
        if not pending:
            return
        grid, arguments, geometry, tensors, shared = pending[0]
        if len(pending) == 1 or any(
            current_grid != grid or current_geometry != geometry or current_shared != shared
            for current_grid, _, current_geometry, _, current_shared in pending[1:]
        ):
            logger.info_once(
                "Flash HiSparse GDN metadata uses separate launches: groups=%d requests=%d query=%d",
                len(pending),
                geometry["NUM_REQS"],
                geometry["SPEC_WIDTH"],
                scope="process",
            )
            for grid, arguments, geometry, _, _ in pending:
                _prepare_gdn_kernel[(grid,)](*arguments, None, **geometry, LIVE_CONV_NUMEL=0, GROUPED=False)
            return
        addresses = tuple(tuple(tensor.data_ptr() for tensor in item[3]) for item in pending)
        if self._addresses != addresses:
            pointer_map = torch.tensor(addresses, dtype=torch.int64, device="cpu", pin_memory=True)
            self._pointers = pointer_map.to(tensors[0].device, non_blocking=True)
            self._addresses = addresses
        logger.info_once(
            "Flash HiSparse GDN metadata grouped: groups=%d requests=%d query=%d",
            len(pending),
            geometry["NUM_REQS"],
            geometry["SPEC_WIDTH"],
            scope="process",
        )
        conv_numel = geometry["CONV_NUMEL"]
        if clear_conv_tokens is not None:
            # The unpadded GLM Q6 graph passes only the live projection rows
            # to causal_conv1d_update. Keep full clears on separate launches.
            conv_numel = tensors[-1][:clear_conv_tokens].numel()
            grid = geometry["NUM_ROWS"] + triton.cdiv(conv_numel, geometry["CLEAR_BLOCK"])
        _prepare_gdn_kernel[(grid, len(pending))](
            *arguments, self._pointers, **geometry, LIVE_CONV_NUMEL=conv_numel, GROUPED=True
        )


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
    grouped_spec: GroupedSpecGDN | None = None,
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
    grid = rows + (conv_numel + clear_block - 1) // clear_block
    arguments = (
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
    )
    geometry = dict(
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
    if grouped_spec is not None and conv_output is not None and num_spec_decodes and not (mixed_spec or has_prefill):
        assert accepted is not None
        group_tensors = (
            block_table,
            spec_states,
            conv_indices,
            sequence_masks,
            spec_query,
            spec_lengths,
            accepted_out,
            spec_tokens,
            conv_output,
        )
        shared_inputs = (seq_lens.data_ptr(), query.data_ptr(), accepted.data_ptr())
        grouped_spec.stage(grid, arguments, geometry, group_tensors, shared_inputs)
    else:
        _prepare_gdn_kernel[(grid,)](*arguments, None, **geometry, LIVE_CONV_NUMEL=0, GROUPED=False)
