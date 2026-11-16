# SPDX-License-Identifier: Apache-2.0
"""Pack selected QSA KV blocks in logical order for batch-invariant GBSA."""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _pack_selected_blocks(
    key, value, source_table, indices, counts, visible_lens,
    packed_key, packed_value, packed_indices, packed_lens,
    key_page_stride, key_token_stride, key_head_stride, key_dim_stride,
    value_page_stride, value_token_stride, value_head_stride, value_dim_stride,
    idx_head_stride, idx_row_stride, idx_group_stride,
    count_head_stride, count_row_stride, table_row_stride, table_col_stride,
    ROWS: tl.constexpr, GROUPS: tl.constexpr, CAPACITY: tl.constexpr,
    HEADS: tl.constexpr, DIM: tl.constexpr, DIM_TILE: tl.constexpr,
    GROUP_TILE: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    head = tl.program_id(2)
    count = tl.load(counts + head * count_head_stride + row * count_row_stride)
    visible = tl.load(visible_lens + row)
    # QSA's prepared indices end with the partial tail group, when there is one.
    last = tl.load(indices + head * idx_head_stride + row * idx_row_stride +
                   tl.maximum(count - 1, 0) * idx_group_stride, mask=count > 0, other=-1)
    if tile == 0 and head == 0:
        tail = visible % 4
        missing = tl.where((count > 0) & (tail > 0) & (last == visible // 4), 4 - tail, 0)
        tl.store(packed_lens + row, count * 4 - missing)

    groups = tile * GROUP_TILE + tl.arange(0, GROUP_TILE)
    tl.store(packed_indices + (head * ROWS + row) * GROUPS + groups,
             tl.where(groups < count, groups, -1), mask=groups < GROUPS)

    tokens = tile * (4 * GROUP_TILE) + tl.arange(0, 4 * GROUP_TILE)
    slot = tokens // 4
    group = tl.load(indices + head * idx_head_stride + row * idx_row_stride + slot * idx_group_stride,
                    mask=(slot < count) & (slot < GROUPS), other=-1)
    # Keep masked lanes within the source allocation as Triton may form pointer
    # offsets before applying the load mask.
    safe_group = tl.where(group >= 0, group, 0)
    logical = safe_group * 4 + tokens % 4
    valid = (group >= 0) & (logical < visible)
    page = tl.load(source_table + row * table_row_stride + (logical // 128) * table_col_stride,
                   mask=valid, other=0).to(tl.int64)
    channel = tl.arange(0, DIM_TILE)
    key_row = page * key_page_stride + (logical % 128) * key_token_stride + head * key_head_stride
    value_row = page * value_page_stride + (logical % 128) * value_token_stride + head * value_head_stride
    mask = valid[:, None] & (channel[None, :] < DIM)
    k = tl.load(key + key_row[:, None] + channel[None, :] * key_dim_stride, mask=mask, other=0.0)
    v = tl.load(value + value_row[:, None] + channel[None, :] * value_dim_stride, mask=mask, other=0.0)
    dest = ((row * CAPACITY + tokens) * HEADS + head) * DIM
    dest_mask = (tokens[:, None] < CAPACITY) & (channel[None, :] < DIM)
    tl.store(packed_key + dest[:, None] + channel[None, :], k, mask=dest_mask)
    tl.store(packed_value + dest[:, None] + channel[None, :], v, mask=dest_mask)


def pack_selected_kv(key, value, source_table, indices, counts, visible_lens):
    """Give every query the same physical layout for its selected KV blocks.

    Each QSA group contains four tokens. All KV heads share the selected groups
    and their visibility, as established by prepare_qsa_gbsa_inputs.
    """
    heads, rows, groups = indices.shape
    dim = key.shape[-1]
    pages = triton.cdiv(groups * 4, 128)
    capacity = pages * 128
    shape = (rows * pages, 128, heads, dim)
    packed_key = torch.empty(shape, dtype=key.dtype, device=key.device)
    packed_value = torch.empty_like(packed_key)
    packed_indices = torch.empty(indices.shape, dtype=indices.dtype, device=indices.device)
    packed_lens = torch.empty((rows,), dtype=visible_lens.dtype, device=key.device)
    table = torch.arange(rows * pages, dtype=source_table.dtype, device=key.device).reshape(rows, pages)
    group_tile = 8
    _pack_selected_blocks[(triton.cdiv(capacity, group_tile * 4), rows, heads)](
        key, value, source_table, indices, counts, visible_lens,
        packed_key, packed_value, packed_indices, packed_lens,
        *key.stride(), *value.stride(), *indices.stride(), *counts.stride(), *source_table.stride(),
        ROWS=rows, GROUPS=groups, CAPACITY=capacity, HEADS=heads, DIM=dim,
        DIM_TILE=triton.next_power_of_2(dim), GROUP_TILE=group_tile, num_warps=1,
    )
    return packed_key, packed_value, packed_indices, packed_lens, table
