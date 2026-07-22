# SPDX-License-Identifier: Apache-2.0
#
# This file is a part of the vllm-ascend project.
#
# Fused Triton kernels for the Domino correction loop, ported from
# the reference Domino correction implementation.
#
# NPU porting rules applied (per npu-development-pitfalls.md):
#  - import via vllm.triton_utils (project forbids direct `import triton`).
#  - no cache_modifier / num_warps / num_stages.
#  - fp32 elementwise/accumulation where supported; batch dot keeps bf16/fp16
#    inputs for Cube compatibility and accumulates into fp32.
#  - tl.tanh (unavailable) replaced by 2*sigmoid(2x)-1.

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

BATCHED_BLOCK_B = 16
BATCHED_BLOCK_V = 256
BATCHED_BLOCK_M = 64
MAX_TRITON_GRID_SIZE = 65535


@triton.jit
def _fused_mlp_bias_argmax_kernel(
    z_ptr,  # [B, M] z projection (z = fc1_weight[:, :H] @ parallel_hidden)
    s_ptr,  # [B, M] s projection (s = fc1_weight[:, H:] @ gru_state)
    fc2_ptr,  # [V, M] fc2 weight (Linear(M, V).weight is [V, M])
    base_ptr,  # [B, V] base_logits
    fc2_bias_ptr,  # [V] or dummy when HAS_FC2_BIAS=False
    out_val_ptr,  # [B, num_v_blocks] per-block best value
    out_idx_ptr,  # [B, num_v_blocks] per-block best index
    M,
    V,
    stride_z_b,
    stride_z_m,
    stride_s_b,
    stride_s_m,
    stride_base_b,
    stride_base_v,
    stride_f2_v,
    stride_f2_m,
    stride_out_b,
    stride_out_v,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_FC2_BIAS: tl.constexpr,
):
    """Fused: SiLU(z + s) @ fc2.T + [bias] + base -> per-block argmax.

    Each program handles one BLOCK_V slice of the vocabulary for one batch row.
    """
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    acc = tl.zeros([BLOCK_V], dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        # SiLU(z + s) computed on the fly.
        z_ptrs = z_ptr + pid_b * stride_z_b + offs_m * stride_z_m
        s_ptrs = s_ptr + pid_b * stride_s_b + offs_m * stride_s_m
        z_block = tl.load(z_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        s_block = tl.load(s_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        preact = z_block + s_block
        mid_block = preact * tl.sigmoid(preact)  # SiLU(preact)

        # GEMV tile: acc[v] += sum_m fc2[v, m] * mid[m]
        fc2_ptrs = fc2_ptr + offs_v[:, None] * stride_f2_v + offs_m[None, :] * stride_f2_m
        fc2_block = tl.load(fc2_ptrs, mask=mask_v[:, None] & mask_m[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(fc2_block * mid_block[None, :], axis=1)

    if HAS_FC2_BIAS:
        bias_ptrs = fc2_bias_ptr + offs_v
        bias_block = tl.load(bias_ptrs, mask=mask_v, other=0.0).to(tl.float32)
        acc += bias_block

    base_ptrs = base_ptr + pid_b * stride_base_b + offs_v * stride_base_v
    base_block = tl.load(base_ptrs, mask=mask_v, other=-float("inf")).to(tl.float32)
    acc += base_block

    acc = tl.where(mask_v, acc, -float("inf"))

    local_max_val = tl.max(acc, axis=0)
    local_max_idx = tl.argmax(acc, axis=0)
    global_max_idx = pid_v * BLOCK_V + local_max_idx

    out_offs = pid_b * stride_out_b + pid_v * stride_out_v
    tl.store(out_val_ptr + out_offs, local_max_val)
    tl.store(out_idx_ptr + out_offs, global_max_idx)


@triton.jit
def _fused_mlp_bias_argmax_batched_kernel(
    z_ptr,  # [B, M]
    s_ptr,  # [B, M]
    fc2_ptr,  # [V, M]
    base_ptr,  # [B, V]
    fc2_bias_ptr,  # [V] or dummy when HAS_FC2_BIAS=False
    out_val_ptr,  # [B, NUM_V_BLOCKS]
    out_idx_ptr,  # [B, NUM_V_BLOCKS]
    B,
    M,
    V,
    stride_z_b,
    stride_z_m,
    stride_s_b,
    stride_s_m,
    stride_base_b,
    stride_base_v,
    stride_f2_v,
    stride_f2_m,
    stride_out_b,
    stride_out_v,
    NUM_V_BLOCKS: tl.constexpr,
    TOTAL_PROGRAMS: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_FC2_BIAS: tl.constexpr,
):
    """Batch-tiled MLP correction using Cube matmul and shared weight tiles."""
    pid = tl.program_id(0)
    for task_id in range(pid, TOTAL_PROGRAMS, GRID_SIZE):
        pid_b = task_id // NUM_V_BLOCKS
        pid_v = task_id % NUM_V_BLOCKS

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_b = offs_b < B
        mask_v = offs_v < V

        acc = tl.zeros([BLOCK_B, BLOCK_V], dtype=tl.float32)
        for m_start in range(0, M, BLOCK_M):
            offs_m = m_start + tl.arange(0, BLOCK_M)
            mask_m = offs_m < M

            z_ptrs = z_ptr + offs_b[:, None] * stride_z_b + offs_m[None, :] * stride_z_m
            s_ptrs = s_ptr + offs_b[:, None] * stride_s_b + offs_m[None, :] * stride_s_m
            input_mask = mask_b[:, None] & mask_m[None, :]
            z_block = tl.load(z_ptrs, mask=input_mask, other=0.0).to(tl.float32)
            s_block = tl.load(s_ptrs, mask=input_mask, other=0.0).to(tl.float32)
            preact = z_block + s_block
            mid_block = preact * tl.sigmoid(preact)

            fc2_ptrs = fc2_ptr + offs_v[:, None] * stride_f2_v + offs_m[None, :] * stride_f2_m
            fc2_block = tl.load(fc2_ptrs, mask=mask_v[:, None] & mask_m[None, :], other=0.0)
            acc += tl.dot(mid_block.to(fc2_block.dtype), tl.trans(fc2_block))

        if HAS_FC2_BIAS:
            bias_block = tl.load(fc2_bias_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            acc += bias_block[None, :]

        base_ptrs = base_ptr + offs_b[:, None] * stride_base_b + offs_v[None, :] * stride_base_v
        output_mask = mask_b[:, None] & mask_v[None, :]
        base_block = tl.load(base_ptrs, mask=output_mask, other=-float("inf")).to(tl.float32)
        acc += base_block
        acc = tl.where(output_mask, acc, -float("inf"))

        local_max_val = tl.max(acc, axis=1)
        local_max_idx = tl.argmax(acc, axis=1)
        global_max_idx = pid_v * BLOCK_V + local_max_idx
        out_offs = offs_b * stride_out_b + pid_v * stride_out_v
        tl.store(out_val_ptr + out_offs, local_max_val, mask=mask_b)
        tl.store(out_idx_ptr + out_offs, global_max_idx, mask=mask_b)


@triton.jit
def _reduce_argmax_kernel(
    out_val_ptr,  # [B, N]
    out_idx_ptr,  # [B, N]
    final_token_ptr,  # [B]
    N,
    stride_val_b,
    stride_val_n,
    stride_idx_b,
    stride_idx_n,
    stride_final_b,
    BLOCK_N: tl.constexpr,
):
    """Reduce per-block argmax results to the final token id per batch row."""
    pid_b = tl.program_id(0)
    best_val = -float("inf")
    best_idx = 0
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        val_ptrs = out_val_ptr + pid_b * stride_val_b + offs_n * stride_val_n
        vals = tl.load(val_ptrs, mask=mask_n, other=-float("inf"))
        local_pos = tl.argmax(vals, axis=0)
        local_val = tl.max(vals, axis=0)
        local_idx = tl.load(out_idx_ptr + pid_b * stride_idx_b + (start_n + local_pos) * stride_idx_n)
        take = local_val > best_val
        best_val = tl.where(take, local_val, best_val)
        best_idx = tl.where(take, local_idx, best_idx)
    tl.store(final_token_ptr + pid_b * stride_final_b, best_idx)


@triton.jit
def _fused_gru_cell_kernel(
    gi_ptr,  # [B, 3*G] input projection, or [V, 3*G] lookup table
    token_ptr,  # [B] token ids when USE_GI_TABLE=True; otherwise unused
    gh_ptr,  # [B, 3*G] hidden projection (h_state @ W_hh.T)
    h_ptr,  # [B, G] input hidden state
    h_out_ptr,  # [B, G] output hidden state
    B,
    G,
    stride_gi_b,
    stride_gi_g,
    stride_token_b,
    stride_gh_b,
    stride_gh_g,
    stride_h_b,
    stride_h_g,
    stride_hout_b,
    stride_hout_g,
    BLOCK_G: tl.constexpr,
    USE_GI_TABLE: tl.constexpr,
):
    """Fused GRU cell (bf16-safe, bypasses Ascend's fp16-only DynamicGRUV2).

    h_new = (1 - z) * n + z * h
    where r = sigmoid(gi_r + gh_r), z = sigmoid(gi_z + gh_z),
          n = tanh(gi_n + r * gh_n), and tanh(x) = 2*sigmoid(2x) - 1.
    """
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    mask_g = offs_g < G

    h_state = tl.load(h_ptr + pid_b * stride_h_b + offs_g * stride_h_g, mask=mask_g, other=0.0).to(tl.float32)

    # gi/gh hold [r | z | n] gates, each of length G. In the correction
    # loop, loading gi directly from the [V, 3*G] lookup table avoids a
    # standalone gather kernel and its temporary [B, 3*G] output.
    if USE_GI_TABLE:
        token_id = tl.load(token_ptr + pid_b * stride_token_b)
        gi_row_ptr = gi_ptr + token_id * stride_gi_b
    else:
        gi_row_ptr = gi_ptr + pid_b * stride_gi_b
    gi_r = tl.load(gi_row_ptr + offs_g * stride_gi_g, mask=mask_g, other=0.0).to(tl.float32)
    gi_z = tl.load(gi_row_ptr + (G + offs_g) * stride_gi_g, mask=mask_g, other=0.0).to(tl.float32)
    gi_n = tl.load(gi_row_ptr + (2 * G + offs_g) * stride_gi_g, mask=mask_g, other=0.0).to(tl.float32)

    gh_r = tl.load(gh_ptr + pid_b * stride_gh_b + offs_g * stride_gh_g, mask=mask_g, other=0.0).to(tl.float32)
    gh_z = tl.load(gh_ptr + pid_b * stride_gh_b + (G + offs_g) * stride_gh_g, mask=mask_g, other=0.0).to(tl.float32)
    gh_n = tl.load(gh_ptr + pid_b * stride_gh_b + (2 * G + offs_g) * stride_gh_g, mask=mask_g, other=0.0).to(tl.float32)

    r = tl.sigmoid(gi_r + gh_r)
    z = tl.sigmoid(gi_z + gh_z)
    # tl.tanh is unavailable on triton-ascend; use the sigmoid identity.
    n = 2.0 * tl.sigmoid(2.0 * (gi_n + r * gh_n)) - 1.0
    h_new = (1.0 - z) * n + z * h_state

    tl.store(
        h_out_ptr + pid_b * stride_hout_b + offs_g * stride_hout_g,
        h_new.to(h_ptr.dtype.element_ty),
        mask=mask_g,
    )


def fused_mlp_bias_argmax(
    z_proj: torch.Tensor,  # [B, M]
    s_proj: torch.Tensor,  # [B, M]
    base: torch.Tensor,  # [B, V]
    fc2_weight: torch.Tensor,  # [V, M]
    fc2_bias: torch.Tensor | None,  # [V] or None
    block_v: int = 512,
    block_m: int = 32,
    block_n: int = 512,
    use_batch_tiling: bool = True,
    out_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused SiLU(z+s) @ fc2.T + [bias] + base -> argmax. Returns [B] int64 tokens.

    Computes the correction-step logits + greedy sampling in two kernel launches
    (one fused GEMV+argmax, one reduce) instead of ~5 eager kernels. When
    ``out_tokens`` is supplied, the reduce kernel writes the result directly to
    that [B] int64 tensor, including a strided ``out[:, i]`` view.
    """
    B, M = z_proj.shape
    V = base.shape[1]
    batch_tiled = (
        use_batch_tiling
        and B > 1
        and z_proj.dtype in (torch.bfloat16, torch.float16)
        and s_proj.dtype == z_proj.dtype
        and fc2_weight.dtype == z_proj.dtype
    )
    effective_block_v = BATCHED_BLOCK_V if batch_tiled else block_v
    num_v_blocks = (V + effective_block_v - 1) // effective_block_v
    out_val = torch.empty(B, num_v_blocks, dtype=torch.float32, device=z_proj.device)
    out_idx = torch.empty(B, num_v_blocks, dtype=torch.int32, device=z_proj.device)
    final = torch.empty(B, dtype=torch.int64, device=z_proj.device) if out_tokens is None else out_tokens
    has_bias = fc2_bias is not None
    bias_ptr = fc2_bias if has_bias else out_val  # dummy, never read when HAS_FC2_BIAS=False

    if batch_tiled:
        num_batch_blocks = (B + BATCHED_BLOCK_B - 1) // BATCHED_BLOCK_B
        total_programs = num_batch_blocks * num_v_blocks
        grid_size = min(total_programs, MAX_TRITON_GRID_SIZE)
        _fused_mlp_bias_argmax_batched_kernel[(grid_size,)](
            z_proj,
            s_proj,
            fc2_weight,
            base,
            bias_ptr,
            out_val,
            out_idx,
            B,
            M,
            V,
            z_proj.stride(0),
            z_proj.stride(1),
            s_proj.stride(0),
            s_proj.stride(1),
            base.stride(0),
            base.stride(1),
            fc2_weight.stride(0),
            fc2_weight.stride(1),
            out_val.stride(0),
            out_val.stride(1),
            NUM_V_BLOCKS=num_v_blocks,
            TOTAL_PROGRAMS=total_programs,
            GRID_SIZE=grid_size,
            BLOCK_B=BATCHED_BLOCK_B,
            BLOCK_V=BATCHED_BLOCK_V,
            BLOCK_M=BATCHED_BLOCK_M,
            HAS_FC2_BIAS=has_bias,
        )
    else:
        grid = (num_v_blocks, B)
        _fused_mlp_bias_argmax_kernel[grid](
            z_proj,
            s_proj,
            fc2_weight,
            base,
            bias_ptr,
            out_val,
            out_idx,
            M,
            V,
            z_proj.stride(0),
            z_proj.stride(1),
            s_proj.stride(0),
            s_proj.stride(1),
            base.stride(0),
            base.stride(1),
            fc2_weight.stride(0),
            fc2_weight.stride(1),
            out_val.stride(0),
            out_val.stride(1),
            BLOCK_V=block_v,
            BLOCK_M=block_m,
            HAS_FC2_BIAS=has_bias,
        )
    _reduce_argmax_kernel[(B,)](
        out_val,
        out_idx,
        final,
        num_v_blocks,
        out_val.stride(0),
        out_val.stride(1),
        out_idx.stride(0),
        out_idx.stride(1),
        final.stride(0),
        BLOCK_N=block_n,
    )
    return final


def fused_gru_cell(
    gi: torch.Tensor,  # [B, 3*G]
    gh: torch.Tensor,  # [B, 3*G]
    h_state: torch.Tensor,  # [B, G]
    block_g: int = 256,
) -> torch.Tensor:
    """Fused GRU cell update. Returns the new hidden state [B, G].

    bf16-safe (computes in fp32, stores back to h_state's dtype). Replaces
    `nn.GRU` whose Ascend backend (`DynamicGRUV2`) is fp16-only.
    """
    B = h_state.shape[0]
    G = h_state.shape[1]
    h_out = torch.empty_like(h_state)
    grid = (B, (G + block_g - 1) // block_g)
    _fused_gru_cell_kernel[grid](
        gi,
        gi,
        gh,
        h_state,
        h_out,
        B,
        G,
        gi.stride(0),
        gi.stride(1),
        gi.stride(0),
        gh.stride(0),
        gh.stride(1),
        h_state.stride(0),
        h_state.stride(1),
        h_out.stride(0),
        h_out.stride(1),
        BLOCK_G=block_g,
        USE_GI_TABLE=False,
    )
    return h_out


def fused_gru_cell_from_tokens(
    gi_table: torch.Tensor,  # [V, 3*G]
    token_ids: torch.Tensor,  # [B]
    gh: torch.Tensor,  # [B, 3*G]
    h_state: torch.Tensor,  # [B, G]
    block_g: int = 256,
) -> torch.Tensor:
    """Update GRU state with a direct token-to-gi-table lookup.

    This is equivalent to ``fused_gru_cell(gi_table[token_ids], gh, h_state)``
    without materializing the gathered ``gi`` tensor.
    """
    B = h_state.shape[0]
    G = h_state.shape[1]
    h_out = torch.empty_like(h_state)
    grid = (B, (G + block_g - 1) // block_g)
    _fused_gru_cell_kernel[grid](
        gi_table,
        token_ids,
        gh,
        h_state,
        h_out,
        B,
        G,
        gi_table.stride(0),
        gi_table.stride(1),
        token_ids.stride(0),
        gh.stride(0),
        gh.stride(1),
        h_state.stride(0),
        h_state.stride(1),
        h_out.stride(0),
        h_out.stride(1),
        BLOCK_G=block_g,
        USE_GI_TABLE=True,
    )
    return h_out
