# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _dequant_one(
    cache,
    scale,
    dense,
    physical,
    dst_token,
    valid_tokens,
    group,
    token,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
    TILE_G: tl.constexpr,
    TILE_T: tl.constexpr,
):
    nz_ptr = tl.make_block_ptr(
        base=cache + physical * (GROUPS * BLOCK * 32),
        shape=(GROUPS, valid_tokens, 32),
        strides=(BLOCK * 32, 32, 1),
        offsets=(group, token, 0),
        block_shape=(TILE_G, TILE_T, 32),
        order=(2, 1, 0),
    )
    scale_ptr = tl.make_block_ptr(
        base=scale,
        shape=(GROUPS, 32),
        strides=(32, 1),
        offsets=(group, 0),
        block_shape=(TILE_G, 32),
        order=(1, 0),
    )
    nd_ptr = tl.make_block_ptr(
        base=dense + dst_token * (GROUPS * 32),
        shape=(valid_tokens, GROUPS, 32),
        strides=(GROUPS * 32, 32, 1),
        offsets=(token, group, 0),
        block_shape=(TILE_T, TILE_G, 32),
        order=(2, 1, 0),
    )

    values = tl.load(nz_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    scales = tl.load(scale_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    converted = (values * scales[:, None, :]).to(dense.dtype.element_ty)
    tl.store(nd_ptr, tl.trans(converted, (1, 0, 2)), boundary_check=(0, 1))


@triton.jit
def _fused_tiles(
    key,
    value,
    table,
    pages,
    k_scale,
    v_scale,
    out_key,
    out_value,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
    TILE_G: tl.constexpr,
    TILE_T: tl.constexpr,
):
    token_tiles: tl.constexpr = (BLOCK + TILE_T - 1) // TILE_T
    group_tiles: tl.constexpr = (GROUPS + TILE_G - 1) // TILE_G
    tiles_per_page: tl.constexpr = token_tiles * group_tiles
    program = tl.program_id(0)
    page = program // tiles_per_page
    local_tile = program % tiles_per_page
    group = (local_tile // token_tiles) * TILE_G
    token = (local_tile % token_tiles) * TILE_T
    valid_tokens = tl.load(pages + 3 * page + 2)
    if token < valid_tokens:
        table_offset = tl.load(pages + 3 * page)
        output_start = tl.load(pages + 3 * page + 1).to(tl.int64)
        physical = tl.load(table + table_offset).to(tl.int64)
        _dequant_one(
            key, k_scale, out_key, physical, output_start, valid_tokens, group, token, BLOCK, GROUPS, TILE_G, TILE_T
        )
        _dequant_one(
            value, v_scale, out_value, physical, output_start, valid_tokens, group, token, BLOCK, GROUPS, TILE_G, TILE_T
        )


def try_fused_c8_prefill(
    key,
    value,
    block_table,
    seq_lens,
    block_size,
    num_kv_heads,
    head_size,
    target_dtype,
    k_scale,
    v_scale,
):
    # PIECEWISE executes cached attention outside the captured partitions.
    # Raw graph capture cannot rebuild host page descriptors on replay.
    scales = k_scale, v_scale
    if (
        key.device.type != "npu"
        or torch.npu.is_current_stream_capturing()
        or target_dtype not in (torch.bfloat16, torch.float16)
        or block_size <= 0
        or block_size % 32
        or head_size <= 0
        or head_size % 32
        or num_kv_heads <= 0
        or not isinstance(seq_lens, (list, tuple))
        or block_table.ndim != 2
        or block_table.shape[0] != len(seq_lens)
        or block_table.stride(1) != 1
        or block_table.dtype not in (torch.int32, torch.int64)
        or block_table.device != key.device
        or value.device != key.device
        or key.numel() != value.numel()
        or key.numel() % (block_size * num_kv_heads * head_size)
        or any(cache.dtype != torch.int8 or not cache.is_contiguous() for cache in (key, value))
        or any(
            scale.ndim == 0
            or scale.numel() != num_kv_heads * head_size
            or scale.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or scale.device != key.device
            for scale in scales
        )
    ):
        return None

    entries = []
    output_start = 0
    max_length = block_table.shape[1] * block_size
    for row, length in enumerate(seq_lens):
        if length < 0 or length > max_length:
            return None
        row_start = row * block_table.stride(0)
        for column, token in enumerate(range(0, length, block_size)):
            entries.append((row_start + column, output_start + token, min(block_size, length - token)))
        output_start += length
    if output_start >= 2**31 or (entries and entries[-1][0] >= 2**31):
        return None

    outputs = tuple(
        torch.empty(
            (output_start, num_kv_heads, head_size),
            dtype=torch.promote_types(target_dtype, scale.dtype),
            device=key.device,
        )
        for scale in scales
    )
    if entries:
        descriptors = torch.tensor(entries, dtype=torch.int32, device=key.device)
        groups = num_kv_heads * (head_size // 32)
        tile_groups, tile_tokens = 8, 64
        grid = (len(entries) * triton.cdiv(groups, tile_groups) * triton.cdiv(block_size, tile_tokens),)
        _fused_tiles[grid](
            key,
            value,
            block_table,
            descriptors,
            k_scale.contiguous(),
            v_scale.contiguous(),
            *outputs,
            block_size,
            groups,
            tile_groups,
            tile_tokens,
        )
    return outputs
