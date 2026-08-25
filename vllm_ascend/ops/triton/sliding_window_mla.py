# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def _paged_sliding_window_gather_kernel(
    latent_cache_ptr,
    latent_block_stride,
    latent_token_stride,
    rope_cache_ptr,
    rope_block_stride,
    rope_token_stride,
    block_table_ptr,
    block_table_req_stride,
    block_table_block_stride,
    query_start_loc_ptr,
    input_positions_ptr,
    latent_out_ptr,
    rope_out_ptr,
    valid_out_ptr,
    total_positions,
    BLOCK_SIZE: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    LATENT_DIM_PADDED: tl.constexpr,
    ROPE_DIM_PADDED: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    latent_offsets = tl.arange(0, LATENT_DIM_PADDED)
    rope_offsets = tl.arange(0, ROPE_DIM_PADDED)

    for linear_idx in tl.range(pid, total_positions, num_programs):
        token_idx = linear_idx // WINDOW_SIZE
        window_idx = linear_idx - token_idx * WINDOW_SIZE

        request_id = tl.full((), 0, dtype=tl.int32)
        for req_idx in range(1, NUM_REQUESTS + 1):
            request_end = tl.load(query_start_loc_ptr + req_idx).to(tl.int32)
            request_id += tl.where(token_idx >= request_end, 1, 0)

        query_position = tl.load(input_positions_ptr + token_idx).to(tl.int64)
        key_position = query_position + window_idx - (WINDOW_SIZE - 1)
        valid = key_position >= 0
        safe_position = tl.maximum(key_position, 0)
        logical_block = safe_position // BLOCK_SIZE
        block_offset = safe_position - logical_block * BLOCK_SIZE
        physical_block = tl.load(
            block_table_ptr
            + request_id.to(tl.int64) * block_table_req_stride
            + logical_block * block_table_block_stride,
            mask=valid,
            other=0,
        ).to(tl.int64)

        latent = tl.load(
            latent_cache_ptr
            + physical_block * latent_block_stride
            + block_offset * latent_token_stride
            + latent_offsets,
            mask=valid & (latent_offsets < LATENT_DIM),
            other=0.0,
        ).to(tl.float32)
        rope = tl.load(
            rope_cache_ptr + physical_block * rope_block_stride + block_offset * rope_token_stride + rope_offsets,
            mask=valid & (rope_offsets < ROPE_DIM),
            other=0.0,
        ).to(tl.float32)

        latent_out_offset = linear_idx * LATENT_DIM
        rope_out_offset = linear_idx * ROPE_DIM
        tl.store(
            latent_out_ptr + latent_out_offset + latent_offsets,
            latent,
            mask=latent_offsets < LATENT_DIM,
        )
        tl.store(
            rope_out_ptr + rope_out_offset + rope_offsets,
            rope,
            mask=rope_offsets < ROPE_DIM,
        )
        tl.store(valid_out_ptr + linear_idx, valid)


def paged_sliding_window_gather(
    latent_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    input_positions: torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather sliding-window MLA paged KV into FP32 workspaces."""
    num_tokens = input_positions.shape[0]
    num_requests = query_start_loc.shape[0] - 1
    block_size = latent_cache.shape[2]
    latent_dim = latent_cache.shape[3]
    rope_dim = rope_cache.shape[3]

    latent_out = torch.empty(
        (num_tokens, window_size, latent_dim),
        dtype=torch.float32,
        device=latent_cache.device,
    )
    rope_out = torch.empty(
        (num_tokens, window_size, rope_dim),
        dtype=torch.float32,
        device=rope_cache.device,
    )
    valid_out = torch.empty(
        (num_tokens, window_size),
        dtype=torch.bool,
        device=latent_cache.device,
    )

    total_positions = num_tokens * window_size
    grid_size = min(get_vectorcore_num(), total_positions)
    _paged_sliding_window_gather_kernel[(grid_size,)](
        latent_cache,
        latent_cache.stride(0),
        latent_cache.stride(2),
        rope_cache,
        rope_cache.stride(0),
        rope_cache.stride(2),
        block_table,
        block_table.stride(0),
        block_table.stride(1),
        query_start_loc,
        input_positions,
        latent_out,
        rope_out,
        valid_out,
        total_positions,
        BLOCK_SIZE=block_size,
        WINDOW_SIZE=window_size,
        NUM_REQUESTS=num_requests,
        LATENT_DIM=latent_dim,
        ROPE_DIM=rope_dim,
        LATENT_DIM_PADDED=triton.next_power_of_2(latent_dim),
        ROPE_DIM_PADDED=triton.next_power_of_2(rope_dim),
    )
    return latent_out, rope_out, valid_out
