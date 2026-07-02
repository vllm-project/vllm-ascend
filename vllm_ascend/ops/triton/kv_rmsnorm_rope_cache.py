#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def _kv_rmsnorm_rope_cache_by_cache_kernel(
    kv_ptr,
    weight_ptr,
    cos_sin_cache_ptr,
    cos_sin_row_stride,
    positions_ptr,
    slots_ptr,
    kv_cache_rope_ptr,
    kv_cache_nope_ptr,
    out_rope_ptr,
    out_nope_ptr,
    num_tokens,
    cache_block_size: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    total_dim: tl.constexpr,
    eps: tl.constexpr,
    pad_nope_dim: tl.constexpr,
    pad_half_rope_dim: tl.constexpr,
    is_neox_style: tl.constexpr,
    is_output_kv: tl.constexpr,
    cache_mode_is_nz: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_block_size = tl.num_programs(0)
    half_rope_dim = rope_dim // 2

    nope_offsets = tl.arange(0, pad_nope_dim)
    nope_mask = nope_offsets < nope_dim
    half_offsets = tl.arange(0, pad_half_rope_dim)
    half_mask = half_offsets < half_rope_dim

    weight = tl.load(weight_ptr + nope_offsets, mask=nope_mask, other=0.0).to(tl.float32)

    for token_idx in tl.range(pid, num_tokens, row_block_size):
        kv_row_ptr = kv_ptr + token_idx * total_dim
        nope_vals = tl.load(kv_row_ptr + nope_offsets, mask=nope_mask, other=0.0).to(tl.float32)
        variance = tl.sum(nope_vals * nope_vals, axis=0) / nope_dim
        rstd = 1.0 / tl.sqrt(tl.maximum(variance, 0.0) + eps)
        nope_out = nope_vals * rstd * weight

        slot = tl.load(slots_ptr + token_idx).to(tl.int64)
        slot_valid = slot >= 0
        block_idx = slot // cache_block_size
        block_offset = slot - block_idx * cache_block_size
        if cache_mode_is_nz:
            cache_nope_offsets = (
                block_idx * cache_block_size * nope_dim
                + (nope_offsets // 16) * cache_block_size * 16
                + block_offset * 16
                + (nope_offsets % 16)
            )
        else:
            cache_nope_offsets = (block_idx * cache_block_size + block_offset) * nope_dim + nope_offsets
        tl.store(kv_cache_nope_ptr + cache_nope_offsets, nope_out, mask=nope_mask & slot_valid)

        pos = tl.load(positions_ptr + token_idx).to(tl.int64)
        cos_sin_row = cos_sin_cache_ptr + pos * cos_sin_row_stride
        cos_row = tl.load(cos_sin_row + half_offsets, mask=half_mask, other=0.0).to(tl.float32)
        sin_row = tl.load(cos_sin_row + half_rope_dim + half_offsets, mask=half_mask, other=0.0).to(tl.float32)

        if is_neox_style:
            rope_first_offsets = nope_dim + half_offsets
            rope_second_offsets = rope_first_offsets + half_rope_dim
            rope_in_first = tl.load(kv_row_ptr + rope_first_offsets, mask=half_mask, other=0.0).to(tl.float32)
            rope_in_second = tl.load(kv_row_ptr + rope_second_offsets, mask=half_mask, other=0.0).to(tl.float32)
        else:
            pair_offsets = nope_dim + 2 * half_offsets[:, None] + tl.arange(0, 2)[None, :]
            rope_pair = tl.load(kv_row_ptr + pair_offsets, mask=half_mask[:, None], other=0.0).to(tl.float32)
            rope_in_first, rope_in_second = tl.split(rope_pair)
        rope_first = rope_in_first * cos_row - rope_in_second * sin_row
        rope_second = rope_in_second * cos_row + rope_in_first * sin_row

        if cache_mode_is_nz:
            rope_first_dims = half_offsets
            rope_second_dims = half_offsets + half_rope_dim
            cache_rope_first_offsets = (
                block_idx * cache_block_size * rope_dim
                + (rope_first_dims // 16) * cache_block_size * 16
                + block_offset * 16
                + (rope_first_dims % 16)
            )
            cache_rope_second_offsets = (
                block_idx * cache_block_size * rope_dim
                + (rope_second_dims // 16) * cache_block_size * 16
                + block_offset * 16
                + (rope_second_dims % 16)
            )
        else:
            cache_rope_row = (block_idx * cache_block_size + block_offset) * rope_dim
            cache_rope_first_offsets = cache_rope_row + half_offsets
            cache_rope_second_offsets = cache_rope_first_offsets + half_rope_dim
        tl.store(kv_cache_rope_ptr + cache_rope_first_offsets, rope_first, mask=half_mask & slot_valid)
        tl.store(kv_cache_rope_ptr + cache_rope_second_offsets, rope_second, mask=half_mask & slot_valid)

        out_nope_row = token_idx * nope_dim
        out_rope_row = token_idx * rope_dim
        if is_output_kv:
            tl.store(out_nope_ptr + out_nope_row + nope_offsets, nope_out, mask=nope_mask)
            tl.store(out_rope_ptr + out_rope_row + half_offsets, rope_first, mask=half_mask)
            tl.store(out_rope_ptr + out_rope_row + half_offsets + half_rope_dim, rope_second, mask=half_mask)


def kv_rmsnorm_rope_cache_by_cache_triton(
    kv_no_split: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    slots: torch.Tensor,
    kv_cache_rope: torch.Tensor,
    kv_cache_nope: torch.Tensor,
    *,
    epsilon: float,
    rope_dim: int,
    is_neox_style: bool,
    is_output_kv: bool = False,
    cache_mode_is_nz: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not kv_no_split.is_contiguous():
        kv_no_split = kv_no_split.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not positions.is_contiguous():
        positions = positions.contiguous()
    if not slots.is_contiguous():
        slots = slots.contiguous()
    if not cos_sin_cache.is_contiguous():
        cos_sin_cache = cos_sin_cache.contiguous()

    num_tokens = kv_no_split.shape[0]
    nope_dim = weight.shape[0]
    total_dim = kv_no_split.shape[-1]
    if is_output_kv:
        out_rope = torch.empty((num_tokens, 1, 1, rope_dim), device=kv_no_split.device, dtype=kv_no_split.dtype)
        out_nope = torch.empty((num_tokens, 1, 1, nope_dim), device=kv_no_split.device, dtype=kv_no_split.dtype)
    else:
        out_rope = torch.empty((0,), device=kv_no_split.device, dtype=kv_no_split.dtype)
        out_nope = torch.empty((0,), device=kv_no_split.device, dtype=kv_no_split.dtype)
    if num_tokens == 0:
        return kv_cache_rope, kv_cache_nope, out_rope, out_nope

    pad_nope_dim = triton.next_power_of_2(nope_dim)
    pad_half_rope_dim = triton.next_power_of_2(rope_dim // 2)
    grid = (min(num_tokens, get_vectorcore_num()),)
    _kv_rmsnorm_rope_cache_by_cache_kernel[grid](
        kv_no_split,
        weight,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        positions,
        slots,
        kv_cache_rope,
        kv_cache_nope,
        out_rope,
        out_nope,
        num_tokens,
        kv_cache_rope.shape[1],
        nope_dim,
        rope_dim,
        total_dim,
        epsilon,
        pad_nope_dim,
        pad_half_rope_dim,
        is_neox_style,
        is_output_kv,
        cache_mode_is_nz,
    )
    return kv_cache_rope, kv_cache_nope, out_rope, out_nope
