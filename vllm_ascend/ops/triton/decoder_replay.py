# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

REPLAY_BLOCK_SIZE = 1024


@triton.jit
def _gather_replay_batch_kernel(
    query_start_loc,
    dropped_before,
    seq_lens,
    positions,
    slot_mappings,
    rows_out,
    query_start_loc_out,
    positions_out,
    slot_mappings_out,
    slot_stride: tl.constexpr,
    replay_start_out,
    num_tokens_padded,
    WINDOW: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    begin = tl.load(query_start_loc + req)
    end = tl.load(query_start_loc + req + 1)
    kept_begin = begin - tl.load(dropped_before + req)
    kept_end = end - tl.load(dropped_before + req + 1)
    kept_len = kept_end - kept_begin
    if req == 0:
        tl.store(query_start_loc_out, kept_begin)
    tl.store(query_start_loc_out + req + 1, kept_end)
    # Untrimmed requests also need the preceding window for their queries.
    history = tl.where(kept_len < end - begin, 0, WINDOW - 1)
    replay_start = tl.maximum(tl.load(seq_lens + req) - kept_len - history, 0)
    tl.store(replay_start_out + req, replay_start)

    first_row = end - kept_len
    for start in range(kept_begin, kept_end, BLOCK):
        out_rows = start + tl.arange(0, BLOCK)
        valid = out_rows < kept_end
        source_rows = first_row + out_rows - kept_begin
        tl.store(rows_out + out_rows, source_rows, mask=valid)
        pos = tl.load(positions + source_rows, mask=valid, other=0)
        tl.store(positions_out + out_rows, pos, mask=valid)
        for group in tl.static_range(NUM_GROUPS):
            slots = tl.load(slot_mappings[group] + source_rows, mask=valid, other=-1)
            tl.store(slot_mappings_out + group * slot_stride + out_rows, slots, mask=valid)

    if req == tl.num_programs(0) - 1:
        # Padding gathers row zero, but must never write a cache slot.
        first_pos = tl.load(positions)
        for start in range(kept_end, num_tokens_padded, BLOCK):
            out_rows = start + tl.arange(0, BLOCK)
            valid = out_rows < num_tokens_padded
            tl.store(rows_out + out_rows, 0, mask=valid)
            tl.store(positions_out + out_rows, first_pos, mask=valid)
            for group in tl.static_range(NUM_GROUPS):
                tl.store(slot_mappings_out + group * slot_stride + out_rows, -1, mask=valid)


def gather_replay_batch(
    query_start_loc: torch.Tensor,
    dropped_before: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    slot_mappings: tuple[torch.Tensor, ...],
    rows_out: torch.Tensor,
    query_start_loc_out: torch.Tensor,
    positions_out: torch.Tensor,
    slot_mappings_out: torch.Tensor,
    replay_start_out: torch.Tensor,
    num_tokens_padded: int,
    window: int,
) -> None:
    """Gather a replay batch into caller-owned buffers for either model runner.

    ``dropped_before`` contains cumulative discarded query rows at each request
    boundary, calculated on the host. Device query boundaries preserve decode
    lengths adjusted by the runner. Slot mappings are contiguous token arrays,
    one per cache group; output mappings use their own row stride. Callers keep
    decode and prompt-logprob rows and handle empty/dummy batches separately.
    """
    _gather_replay_batch_kernel[(query_start_loc.numel() - 1,)](
        query_start_loc,
        dropped_before,
        seq_lens,
        positions,
        slot_mappings,
        rows_out,
        query_start_loc_out,
        positions_out,
        slot_mappings_out,
        slot_mappings_out.stride(0),
        replay_start_out,
        num_tokens_padded,
        WINDOW=window,
        NUM_GROUPS=len(slot_mappings),
        BLOCK=REPLAY_BLOCK_SIZE,
    )
