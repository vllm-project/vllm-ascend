# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit
def _block_table_scatter_kernel(
    packed_block_ids,
    metadata,
    block_table,
    block_table_stride,
    BLOCK_SIZE: tl.constexpr,
):
    segment_idx = tl.program_id(0)
    metadata_offset = segment_idx * 4
    row_idx = tl.load(metadata + metadata_offset)
    dst_start = tl.load(metadata + metadata_offset + 1)
    length = tl.load(metadata + metadata_offset + 2)
    packed_start = tl.load(metadata + metadata_offset + 3)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < length
    values = tl.load(packed_block_ids + packed_start + offsets, mask=mask)
    dst = block_table + row_idx * block_table_stride + dst_start + offsets
    tl.store(dst, values, mask=mask)


def scatter_block_table(
    packed_block_ids,
    metadata,
    block_table,
    segment_count: int,
) -> None:
    if segment_count == 0:
        return
    _block_table_scatter_kernel[(segment_count,)](
        packed_block_ids,
        metadata,
        block_table,
        block_table.stride(0),
        BLOCK_SIZE=1024,
    )
