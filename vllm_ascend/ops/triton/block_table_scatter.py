# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _block_table_scatter_kernel(
    packed_ptr,
    metadata_ptr,
    dst_ptr,
    dst_stride,
    BLOCK_SIZE: tl.constexpr,
):
    segment_idx = tl.program_id(0)
    metadata_offset = segment_idx * 4
    row_idx = tl.load(metadata_ptr + metadata_offset).to(tl.int64)
    dst_start = tl.load(metadata_ptr + metadata_offset + 1).to(tl.int64)
    length = tl.load(metadata_ptr + metadata_offset + 2).to(tl.int64)
    packed_start = tl.load(metadata_ptr + metadata_offset + 3).to(tl.int64)

    offsets = tl.arange(0, BLOCK_SIZE)
    for base in range(0, length, BLOCK_SIZE):
        indices = base + offsets
        mask = indices < length
        values = tl.load(
            packed_ptr + packed_start + indices,
            mask=mask,
            other=0,
        )
        tl.store(
            dst_ptr + row_idx * dst_stride + dst_start + indices,
            values,
            mask=mask,
        )


def scatter_block_table(
    packed_block_ids: torch.Tensor,
    metadata: torch.Tensor,
    dst_block_table: torch.Tensor,
    segment_count: int,
) -> None:
    if segment_count <= 0:
        return
    _block_table_scatter_kernel[(segment_count,)](
        packed_block_ids,
        metadata,
        dst_block_table,
        dst_block_table.stride(0),
        BLOCK_SIZE=1024,
    )
