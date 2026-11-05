# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# ruff: noqa: E501
# mypy: ignore-errors
import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

MAX_ROWS_PER_ITER = 64


@triton.jit(do_not_specialize=["total_rows", "rows_per_vec"])
def fused_qkvzba_split_reshape_cat_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    total_rows,
    rows_per_vec,
    QKVZ_ROW_STRIDE: tl.constexpr,
    BA_ROW_STRIDE: tl.constexpr,
    QKV_ROW_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    BA_OUT_ROW_STRIDE: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
):
    """
    Fused kernel to split and reshape mixed QKVZ and BA tensors.

    This kernel performs the following transformations:
    - Input mixed_qkvz: [num_tokens, num_heads_qk * (Q + K + V + Z)] where each
      head block contains [Q(HEAD_QK), K(HEAD_QK), V(V_DIM_PER_QK), Z(V_DIM_PER_QK)]
    - Input mixed_ba: [num_tokens, num_heads_qk * (B + A)] where each head block
      contains [B(V_HEADS_PER_QK), A(V_HEADS_PER_QK)]
    - Output mixed_qkv: [num_tokens, Q_all | K_all | V_all] concatenated by type
    - Output z: [num_tokens, num_heads_v, head_v]
    - Output b, a: [num_tokens, num_heads_v]
    """
    # Each vector core processes a contiguous chunk of rows
    vec_id = tl.program_id(0)

    V_HEADS_PER_QK: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    V_DIM_PER_QK: tl.constexpr = V_HEADS_PER_QK * HEAD_V
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + V_DIM_PER_QK * 2
    BA_DIM_T: tl.constexpr = V_HEADS_PER_QK * 2

    Q_TOTAL: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    K_TOTAL: tl.constexpr = NUM_HEADS_QK * HEAD_QK

    row_start = vec_id * rows_per_vec
    row_end = min(row_start + rows_per_vec, total_rows)

    row_offset = row_start

    iter_count = (row_end - row_start + ROWS_PER_ITER - 1) // ROWS_PER_ITER

    # ========== Main Iteration Loop ==========
    for _ in tl.range(iter_count):
        row_indices = tl.arange(0, ROWS_PER_ITER) + row_offset
        row_mask = row_indices < row_end

        # ========== Head Iteration Loop ==========
        # Iterate over each Q/K head group to extract and rearrange data
        for head_id in tl.static_range(NUM_HEADS_QK):
            # Byte offset to the current head's data block in mixed_qkvz
            src_head_offset = head_id * QKVZ_DIM_T

            # ----- Q (Query) Extraction -----
            # Source layout: mixed_qkvz[row, head_id * QKVZ_DIM_T + 0:HEAD_QK]
            # Dest layout: mixed_qkv[row, head_id * HEAD_QK : (head_id+1) * HEAD_QK]
            q_range = tl.arange(0, HEAD_QK)
            q_src = row_indices[:, None] * QKVZ_ROW_STRIDE + src_head_offset + q_range[None, :]
            q_dst = row_indices[:, None] * QKV_ROW_STRIDE + head_id * HEAD_QK + q_range[None, :]
            q_data = tl.load(mixed_qkvz + q_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + q_dst, q_data, mask=row_mask[:, None])

            # ----- K (Key) Extraction -----
            # Source layout: mixed_qkvz[row, head_id * QKVZ_DIM_T + HEAD_QK : +HEAD_QK]
            # Dest layout: mixed_qkv[row, Q_TOTAL + head_id * HEAD_QK : ...]
            # K is stored after Q in the source; in dest, K starts after all Q heads
            k_src = row_indices[:, None] * QKVZ_ROW_STRIDE + src_head_offset + HEAD_QK + q_range[None, :]
            k_dst = row_indices[:, None] * QKV_ROW_STRIDE + Q_TOTAL + head_id * HEAD_QK + q_range[None, :]
            k_data = tl.load(mixed_qkvz + k_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + k_dst, k_data, mask=row_mask[:, None])

            # ----- V (Value) Extraction -----
            # Source layout: mixed_qkvz[row, head_id * QKVZ_DIM_T + HEAD_QK*2 : +V_DIM_PER_QK]
            # Dest layout: mixed_qkv[row, Q_TOTAL + K_TOTAL + head_id * V_DIM_PER_QK : ...]
            # V follows Q and K in source; in dest, V starts after all Q and K heads
            v_range = tl.arange(0, V_DIM_PER_QK)
            v_src = row_indices[:, None] * QKVZ_ROW_STRIDE + src_head_offset + HEAD_QK * 2 + v_range[None, :]
            v_dst = (
                row_indices[:, None] * QKV_ROW_STRIDE + Q_TOTAL + K_TOTAL + head_id * V_DIM_PER_QK + v_range[None, :]
            )
            v_data = tl.load(mixed_qkvz + v_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + v_dst, v_data, mask=row_mask[:, None])

            # ----- Z Extraction -----
            # Source layout: mixed_qkvz[row, head_id * QKVZ_DIM_T + HEAD_QK*2 + V_DIM_PER_QK : ...]
            # Dest layout: z[row, head_id * V_DIM_PER_QK : (head_id+1) * V_DIM_PER_QK]
            # Z follows V in source; output z is reshaped to [batch, num_heads_v, head_v]
            z_src = (
                row_indices[:, None] * QKVZ_ROW_STRIDE + src_head_offset + HEAD_QK * 2 + V_DIM_PER_QK + v_range[None, :]
            )
            z_dst = row_indices[:, None] * Z_ROW_STRIDE + head_id * V_DIM_PER_QK + v_range[None, :]
            z_data = tl.load(mixed_qkvz + z_src, mask=row_mask[:, None])
            tl.store(z + z_dst, z_data, mask=row_mask[:, None])

            # ----- B Extraction -----
            # Source layout: mixed_ba[row, head_id * BA_DIM_T : +V_HEADS_PER_QK]
            # Dest layout: b[row, head_id * V_HEADS_PER_QK : (head_id+1) * V_HEADS_PER_QK]
            b_range = tl.arange(0, V_HEADS_PER_QK)
            ba_head_offset = head_id * BA_DIM_T
            b_src = row_indices[:, None] * BA_ROW_STRIDE + ba_head_offset + b_range[None, :]
            b_dst = row_indices[:, None] * BA_OUT_ROW_STRIDE + head_id * V_HEADS_PER_QK + b_range[None, :]
            b_data = tl.load(mixed_ba + b_src, mask=row_mask[:, None])
            tl.store(b + b_dst, b_data, mask=row_mask[:, None])

            # ----- A Extraction -----
            # Source layout: mixed_ba[row, head_id * BA_DIM_T + V_HEADS_PER_QK : ...]
            # Dest layout: a[row, head_id * V_HEADS_PER_QK : ...] (same as b_dst)
            # A follows B in source; output layout is same as B
            a_src = row_indices[:, None] * BA_ROW_STRIDE + ba_head_offset + V_HEADS_PER_QK + b_range[None, :]
            a_data = tl.load(mixed_ba + a_src, mask=row_mask[:, None])
            tl.store(a + b_dst, a_data, mask=row_mask[:, None])

        row_offset += ROWS_PER_ITER


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    batch, seq_len = mixed_qkvz.shape[0], 1
    total_rows = batch * seq_len

    v_heads_per_qk = num_heads_v // num_heads_qk
    v_dim_per_qk = v_heads_per_qk * head_v
    qkvz_dim_t = head_qk * 2 + v_dim_per_qk * 2
    ba_dim_t = v_heads_per_qk * 2

    # row stride
    qkvz_row_stride = num_heads_qk * qkvz_dim_t
    ba_row_stride = num_heads_qk * ba_dim_t
    qkv_row_stride = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    z_row_stride = num_heads_v * head_v
    ba_out_row_stride = num_heads_v

    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        [batch * seq_len, qkv_dim_t],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        [batch * seq_len, num_heads_v, head_v],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )

    num_vectorcore = get_vectorcore_num()

    grid_size = min(num_vectorcore, total_rows)
    grid_size = max(1, grid_size)

    rows_per_vec = triton.cdiv(total_rows, grid_size)

    ub_size = 85 * 1024 // mixed_qkvz.element_size()

    elements_per_row = qkvz_row_stride + ba_row_stride + qkv_row_stride + z_row_stride + ba_out_row_stride * 2

    rows_per_iter = max(1, ub_size // elements_per_row)
    rows_per_iter = triton.next_power_of_2(rows_per_iter)
    rows_per_iter = min(rows_per_iter, rows_per_vec, MAX_ROWS_PER_ITER)

    grid = (grid_size, 1)
    fused_qkvzba_split_reshape_cat_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        total_rows,
        rows_per_vec,
        qkvz_row_stride,
        ba_row_stride,
        qkv_row_stride,
        z_row_stride,
        ba_out_row_stride,
        rows_per_iter,
    )
    return mixed_qkv, z, b, a
