# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _aligned_mamba_block_table_kernel(
    block_table,
    seq_lens,
    output,
    table_row_stride,
    table_col_stride,
    seq_stride,
    block_size: tl.constexpr,
    num_columns: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    seq_len = tl.load(seq_lens + request * seq_stride)
    start = tl.maximum((seq_len - 1) // block_size, 0)
    indices = tl.load(
        block_table + request * table_row_stride + (start + columns) * table_col_stride,
        mask=columns < num_columns,
        other=0,
    )
    tl.store(output + request * num_columns + columns, indices, mask=columns < num_columns)


def aligned_mamba_block_table(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    num_speculative_blocks: int,
) -> torch.Tensor:
    """Gather recurrent state pages with one launch, using device seq_lens."""
    if block_table.ndim != 2 or seq_lens.ndim != 1 or block_table.shape[0] != seq_lens.shape[0]:
        raise ValueError("Expected a block-table row per sequence length")
    if block_table.device != seq_lens.device:
        raise ValueError("Block table and sequence lengths must be on the same device")
    if block_size <= 0 or num_speculative_blocks < 0:
        raise ValueError("Expected positive block_size and nonnegative num_speculative_blocks")
    num_columns = 1 + num_speculative_blocks
    if block_table.shape[1] < num_columns:
        raise ValueError("Block table has fewer columns than the recurrent state window")
    if block_table.device.type == "cpu":
        start = torch.div(seq_lens - 1, block_size, rounding_mode="floor").clamp_min_(0)
        offsets = torch.arange(num_columns, device=block_table.device)
        return torch.gather(block_table, 1, (start[:, None] + offsets).to(torch.int64))
    output = block_table.new_empty((block_table.shape[0], num_columns))
    if output.numel():
        _aligned_mamba_block_table_kernel[(block_table.shape[0],)](
            block_table,
            seq_lens,
            output,
            block_table.stride(0),
            block_table.stride(1),
            seq_lens.stride(0),
            block_size,
            num_columns,
            BLOCK=triton.next_power_of_2(num_columns),
        )
    return output
