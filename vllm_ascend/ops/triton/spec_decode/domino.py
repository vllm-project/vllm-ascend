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
#  - fp32 compute internally, store back to the original dtype.
#  - tl.tanh (unavailable) replaced by 2*sigmoid(2x)-1.

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


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
        fc2_block = tl.load(
            fc2_ptrs, mask=mask_v[:, None] & mask_m[None, :], other=0.0
        ).to(tl.float32)
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
        local_idx = tl.load(
            out_idx_ptr + pid_b * stride_idx_b + (start_n + local_pos) * stride_idx_n
        )
        take = local_val > best_val
        best_val = tl.where(take, local_val, best_val)
        best_idx = tl.where(take, local_idx, best_idx)
    tl.store(final_token_ptr + pid_b * stride_final_b, best_idx)


@triton.jit
def _fused_gru_cell_kernel(
    gi_ptr,  # [B, 3*G] input projection (embed @ W_ih.T)
    gh_ptr,  # [B, 3*G] hidden projection (h_state @ W_hh.T)
    h_ptr,  # [B, G] input hidden state
    h_out_ptr,  # [B, G] output hidden state
    B,
    G,
    stride_gi_b,
    stride_gi_g,
    stride_gh_b,
    stride_gh_g,
    stride_h_b,
    stride_h_g,
    stride_hout_b,
    stride_hout_g,
    BLOCK_G: tl.constexpr,
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

    h_state = tl.load(
        h_ptr + pid_b * stride_h_b + offs_g * stride_h_g, mask=mask_g, other=0.0
    ).to(tl.float32)

    # gi/gh hold [r | z | n] gates, each of length G.
    gi_r = tl.load(
        gi_ptr + pid_b * stride_gi_b + offs_g * stride_gi_g, mask=mask_g, other=0.0
    ).to(tl.float32)
    gi_z = tl.load(
        gi_ptr + pid_b * stride_gi_b + (G + offs_g) * stride_gi_g, mask=mask_g, other=0.0
    ).to(tl.float32)
    gi_n = tl.load(
        gi_ptr + pid_b * stride_gi_b + (2 * G + offs_g) * stride_gi_g, mask=mask_g, other=0.0
    ).to(tl.float32)

    gh_r = tl.load(
        gh_ptr + pid_b * stride_gh_b + offs_g * stride_gh_g, mask=mask_g, other=0.0
    ).to(tl.float32)
    gh_z = tl.load(
        gh_ptr + pid_b * stride_gh_b + (G + offs_g) * stride_gh_g, mask=mask_g, other=0.0
    ).to(tl.float32)
    gh_n = tl.load(
        gh_ptr + pid_b * stride_gh_b + (2 * G + offs_g) * stride_gh_g, mask=mask_g, other=0.0
    ).to(tl.float32)

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
) -> torch.Tensor:
    """Fused SiLU(z+s) @ fc2.T + [bias] + base -> argmax. Returns [B] int64 tokens.

    Computes the correction-step logits + greedy sampling in two kernel launches
    (one fused GEMV+argmax, one reduce) instead of ~5 eager kernels.
    """
    B, M = z_proj.shape
    V = base.shape[1]
    num_v_blocks = (V + block_v - 1) // block_v
    out_val = torch.empty(B, num_v_blocks, dtype=torch.float32, device=z_proj.device)
    out_idx = torch.empty(B, num_v_blocks, dtype=torch.int32, device=z_proj.device)
    final = torch.empty(B, dtype=torch.int64, device=z_proj.device)
    has_bias = fc2_bias is not None
    bias_ptr = fc2_bias if has_bias else out_val  # dummy, never read when HAS_FC2_BIAS=False

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
        gh,
        h_state,
        h_out,
        B,
        G,
        gi.stride(0),
        gi.stride(1),
        gh.stride(0),
        gh.stride(1),
        h_state.stride(0),
        h_state.stride(1),
        h_out.stride(0),
        h_out.stride(1),
        BLOCK_G=block_g,
    )
    return h_out
