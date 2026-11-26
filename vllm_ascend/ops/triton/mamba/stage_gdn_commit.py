# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy/pad one group's recovery metadata in one device launch."""

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["batch", "sampled_rows"])
def _stage_gdn_commit_kernel(
    accepted,
    indices,
    offsets,
    mapping,
    table,
    computed,
    out_accepted,
    out_indices,
    out_offsets,
    out_mapping,
    out_table,
    out_computed,
    batch,
    sampled_rows,
    accepted_stride: tl.constexpr,
    indices_stride: tl.constexpr,
    offsets_stride: tl.constexpr,
    mapping_stride: tl.constexpr,
    table_row_stride: tl.constexpr,
    table_col_stride: tl.constexpr,
    computed_stride: tl.constexpr,
    BUCKET: tl.constexpr,
    CAPACITY: tl.constexpr,
    WIDTH: tl.constexpr,
    MAPPED: tl.constexpr,
    ALIGN: tl.constexpr,
    COPY_ACCEPTED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if COPY_ACCEPTED:
        value = tl.load(accepted + i * accepted_stride, i < sampled_rows, 0)
        tl.store(out_accepted + i, value, i < CAPACITY)
    page = tl.load(indices + i * indices_stride, i < batch, 0)
    tl.store(out_indices + i, page, i < BUCKET)
    # Empty padded rows retain the last offset, hence a zero query length.
    offset = tl.load(offsets + tl.minimum(i, batch) * offsets_stride, i < BUCKET + 1, 0)
    tl.store(out_offsets + i, offset, i < BUCKET + 1)
    row = i
    if MAPPED:
        row = tl.load(mapping + i * mapping_stride, i < batch, 0)
    tl.store(out_mapping + i, tl.where(i < batch, row, 0), i < BUCKET)
    if ALIGN:
        tokens = tl.load(computed + i * computed_stride, i < sampled_rows, 0)
        tl.store(out_computed + i, tokens, i < CAPACITY)
        r, c = i // WIDTH, i % WIDTH
        block = tl.load(table + r * table_row_stride + c * table_col_stride, r < sampled_rows, 0)
        tl.store(out_table + i, block, i < CAPACITY * WIDTH)


def stage_gdn_commit(source, target, accepted, out_accepted, *, copy_accepted=True):
    """Preserve group-specific page IDs and sampler mapping; no host reads."""
    align = source.table is not None
    width = source.table.shape[1] if align else 1
    capacity = out_accepted.numel()
    bucket = target.indices.numel()
    count = max(capacity * width if align else capacity, bucket + 1)
    _stage_gdn_commit_kernel[(triton.cdiv(count, 256),)](
        accepted,
        source.indices,
        source.offsets,
        source.mapping,
        source.table,
        source.computed,
        out_accepted,
        target.indices,
        target.offsets,
        target.mapping,
        target.table,
        target.computed,
        source.indices.numel(),
        accepted.numel(),
        accepted.stride(0),
        source.indices.stride(0),
        source.offsets.stride(0),
        source.mapping.stride(0) if source.mapping is not None else 0,
        source.table.stride(0) if align else 0,
        source.table.stride(1) if align else 0,
        source.computed.stride(0) if align else 0,
        BUCKET=bucket,
        CAPACITY=capacity,
        WIDTH=width,
        MAPPED=source.mapping is not None,
        ALIGN=align,
        COPY_ACCEPTED=copy_accepted,
        BLOCK=256,
        num_warps=4,
    )
