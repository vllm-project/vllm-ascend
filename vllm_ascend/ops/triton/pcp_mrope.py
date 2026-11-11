# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""PCP-aware M-RoPE position preparation."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_pcp_mrope_positions_kernel(
    positions_ptr,
    positions_stride,
    prefill_positions_ptr,
    prefill_positions_req_stride,
    prefill_positions_dim_stride,
    prefill_delta_ptr,
    input_positions_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    prefill_len_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_DIMS: tl.constexpr,
):
    row = tl.program_id(0)

    req_idx = tl.load(idx_mapping_ptr + row)
    query_start = tl.load(query_start_loc_ptr + row)
    query_end = tl.load(query_start_loc_ptr + row + 1)
    query_len = query_end - query_start

    start = tl.load(input_positions_ptr + query_start).to(tl.int64)
    prefill_len = tl.load(prefill_len_ptr + req_idx).to(tl.int64)
    is_prefill = start < prefill_len
    delta = tl.load(prefill_delta_ptr + req_idx).to(tl.int64)

    for i in range(0, query_len, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < query_len

        if is_prefill:
            for dim in tl.static_range(NUM_DIMS):
                source = tl.load(
                    prefill_positions_ptr
                    + req_idx * prefill_positions_req_stride
                    + dim * prefill_positions_dim_stride
                    + start
                    + offsets,
                    mask=mask,
                ).to(tl.int64)
                tl.store(
                    positions_ptr + dim * positions_stride + query_start + offsets,
                    source,
                    mask=mask,
                )
        else:
            source = tl.load(
                input_positions_ptr + query_start + offsets, mask=mask
            ).to(tl.int64) + delta
            for dim in tl.static_range(NUM_DIMS):
                tl.store(
                    positions_ptr + dim * positions_stride + query_start + offsets,
                    source,
                    mask=mask,
                )


def prepare_pcp_mrope_positions(
    rope_state,
    input_batch,
    prefill_lens,
) -> torch.Tensor:
    """Fill the preallocated M-RoPE buffer for a PCP-local batch."""
    num_tokens = input_batch.num_tokens_after_padding
    positions = rope_state.positions[:, :num_tokens]

    num_rows = input_batch.num_reqs
    if num_rows == 0 or num_tokens == 0:
        positions.zero_()
        return positions


    idx_mapping = input_batch.idx_mapping[:num_rows]
    query_start_loc = input_batch.query_start_loc[: num_rows + 1]

    prefill_positions = rope_state.prefill_positions.gpu
    _prepare_pcp_mrope_positions_kernel[(num_rows,)](
        positions,
        positions.stride(0),
        prefill_positions,
        rope_state.num_dims * prefill_positions.stride(0),
        prefill_positions.stride(0),
        rope_state.prefill_delta.gpu,
        input_batch.positions,
        idx_mapping,
        query_start_loc,
        prefill_lens,
        BLOCK_SIZE=1024,
        NUM_DIMS=rope_state.num_dims,
    )

    actual_tokens = input_batch.num_tokens
    if actual_tokens < num_tokens:
        positions[:, actual_tokens:] = 0
    return positions
