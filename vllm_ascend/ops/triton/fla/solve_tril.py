# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# The kernels below are adapted from
# flashserve/flash-linear-attention-npu/fla/ops/triton/triton_core/solve_tril_fast.py.
# They retain the FLA algorithm while using vLLM-Ascend's BHTC head-first layout.
# ruff: noqa: E501
# mypy: ignore-errors

from typing import Optional

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import extract_slice, insert_slice

from .utils import prepare_chunk_indices

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel_paral(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    LARGE_BLOCK_T: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    T_max = T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if IS_VARLEN:
        A = A + (i_h * T_max + bos) * BT
        Ad = Ad + (i_h * T_max + bos) * 16
    else:
        A = A + (i_b * H + i_h) * T_max * BT
        Ad = Ad + (i_b * H + i_h) * T_max * 16

    N_BLOCKS: tl.constexpr = LARGE_BLOCK_T // 16
    base_t = i_t * LARGE_BLOCK_T

    b_A = tl.zeros((N_BLOCKS, 16, 16), dtype=tl.float32)  # (N_BLOCKS, 16, 16)
    for blkid in range(0, N_BLOCKS):
        row_start_o = base_t + blkid * 16
        col_start_o = row_start_o % BT
        p_A_subrec16 = tl.make_block_ptr(
            A, (T, BT), (BT, 1), (row_start_o, col_start_o), (16, 16), (1, 0)
        )
        b_A_subrec16 = tl.load(p_A_subrec16, boundary_check=(0, 1)).to(
            tl.float32
        )  # (16, 16)
        b_A = insert_slice(
            ful=b_A,
            sub=b_A_subrec16[None, :, :],  # (1, 16, 16)
            offsets=[blkid, 0, 0],
            sizes=[1, 16, 16],
            strides=[1, 1, 1],
        )

    # load multi 16x16 into UB
    local_ori_A = tl.trans(b_A, (1, 0, 2))
    local_ori_A = tl.reshape(local_ori_A, (16, 16 * N_BLOCKS))  # (16, N_BLOCKS*16)

    tmp = tl.arange(0, 16).to(tl.float32)
    rows = tmp[:, None]
    cols = tmp[None, :]
    is_lower = (rows > cols).to(b_A.dtype)
    b_A = -b_A * is_lower

    o_i = tl.arange(0, 16)
    for i in range(1, 16):

        nblks_vec16 = -extract_slice(
            local_ori_A, (i, 0), (1, 16 * N_BLOCKS), (16 * N_BLOCKS, 1)
        )
        b_a = tl.reshape(nblks_vec16, (N_BLOCKS, 16))

        dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2))
        dot_product = tl.sum(dot_tmp, 0)
        b_a = b_a + dot_product  # (N_BLOCKS, 16)

        row_mask = o_i == i  # (16,), True at position i
        update_mask = row_mask[None, :, None]  # (1, 16, 1)
        b_a_expanded = b_a[:, None, :]  # (N_BLOCKS, 1, 16)
        b_A = tl.where(update_mask, b_a_expanded, b_A)  # shape keeps (N_BLOCKS, 16, 16)

    on_diagonal = rows == cols
    b_A = tl.where(on_diagonal, b_A + 1.0, b_A)

    b_A = tl.reshape(b_A, (N_BLOCKS * 16, 16))
    p_Ai = tl.make_block_ptr(
        Ad, (T, 16), (16, 1), (base_t, 0), (N_BLOCKS * 16, 16), (1, 0)
    )
    tl.store(
        p_Ai,
        b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel_paral_v3(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    LARGE_BLOCK_T: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    T_max = T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if IS_VARLEN:
        A = A + (i_h * T_max + bos) * BT
        Ad = Ad + (i_h * T_max + bos) * 16
    else:
        A = A + (i_b * H + i_h) * T_max * BT
        Ad = Ad + (i_b * H + i_h) * T_max * 16

    base_t = i_t * LARGE_BLOCK_T

    NTASKS: tl.constexpr = 2
    N_BLOCKS: tl.constexpr = LARGE_BLOCK_T // 16 // NTASKS

    for taskid in range(0, NTASKS):
        base_t += taskid * (LARGE_BLOCK_T // NTASKS)

        b_A = tl.zeros((N_BLOCKS, 16, 16), dtype=tl.float32)  # (N_BLOCKS, 16, 16)
        for blkid in range(0, N_BLOCKS):
            row_start_o = base_t + blkid * 16
            col_start_o = row_start_o % BT
            # using ptr with mask instead of tl.load(block_ptr)
            offs_rows_in_block = tl.arange(0, 16)
            offs_cols_in_block = tl.arange(0, 16)
            # BHTC: the token dimension is contiguous inside one head.
            ptr_A_subrec16 = (
                A
                + row_start_o * BT
                + col_start_o
                + offs_rows_in_block[:, None] * BT
                + offs_cols_in_block[None, :]
            )
            global_rows = row_start_o + offs_rows_in_block[:, None]
            global_cols = col_start_o + offs_cols_in_block[None, :]
            load_mask = (global_rows < T) & (global_cols < BT)
            b_A_subrec16 = tl.load(ptr_A_subrec16, mask=load_mask, other=0.0).to(
                tl.float32
            )
            b_A = insert_slice(
                ful=b_A,
                sub=b_A_subrec16[None, :, :],  # (1, 16, 16)
                offsets=[blkid, 0, 0],
                sizes=[1, 16, 16],
                strides=[1, 1, 1],
            )

        # load multi 16x16
        local_ori_A = tl.trans(b_A, (1, 0, 2))
        local_ori_A = tl.reshape(local_ori_A, (16, 16 * N_BLOCKS))  # (16, N_BLOCKS*16)

        # change mask into matrix elementwise action
        tmp = tl.arange(0, 16).to(tl.float32)
        rows = tmp[:, None]
        cols = tmp[None, :]
        is_lower = (rows > cols).to(b_A.dtype)
        b_A = -b_A * is_lower

        for i in range(1, 16):

            nblks_vec16 = -extract_slice(
                local_ori_A, (i, 0), (1, 16 * N_BLOCKS), (16 * N_BLOCKS, 1)
            )
            b_a = tl.reshape(nblks_vec16, (N_BLOCKS, 16))

            dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2))
            dot_product = tl.sum(dot_tmp, 0)
            b_a = b_a + dot_product  # (N_BLOCKS, 16)

            b_a_new_expanded = b_a[:, None, :]  # (N_BLOCKS, 1, 16)
            b_A = insert_slice(
                ful=b_A,
                sub=b_a_new_expanded,
                offsets=[0, i, 0],
                sizes=[N_BLOCKS, 1, 16],
                strides=[1, 1, 1],
            )

        on_diagonal = rows == cols
        b_A = tl.where(on_diagonal, b_A + 1.0, b_A)

        b_A = tl.reshape(b_A, (N_BLOCKS * 16, 16))
        p_Ai = tl.make_block_ptr(
            Ad, (T, 16), (16, 1), (base_t, 0), (N_BLOCKS * 16, 16), (1, 0)
        )
        # using ptr with mask instead of tl.load(block_ptr)
        offs_rows_to_store = tl.arange(0, N_BLOCKS * 16)
        offs_cols_to_store = tl.arange(0, 16)
        # BHTC: the token dimension is contiguous inside one head.
        p_Ai = (
            Ad
            + base_t * 16
            + 0
            + offs_rows_to_store[:, None] * 16
            + offs_cols_to_store[None, :]
        )
        global_store_rows = base_t + offs_rows_to_store[:, None]
        store_mask = global_store_rows < T
        tl.store(
            p_Ai,
            b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=store_mask,
        )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tt, i_bh = tl.program_id(0), tl.program_id(1)
    i_t = i_tt
    i_b, i_h = i_bh // H, i_bh % H
    T_max = T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_tt * 2).to(tl.int32), tl.load(
            chunk_indices + i_tt * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if IS_VARLEN:
        A += (i_h * T_max + bos) * BT
        Ad += (i_h * T_max + bos) * 16
        Ai += (i_h * T_max + bos) * 32
    else:
        A += (i_b * H + i_h) * T_max * BT
        Ad += (i_b * H + i_h) * T_max * 16
        Ai += (i_b * H + i_h) * T_max * 32

    p_A_21 = tl.make_block_ptr(
        A, (T, BT), (BT, 1), (i_t * 32 + 16, 0 + i_t % (BT // 32) * 32), (16, 16), (1, 0)
    )
    p_Ad_11 = tl.make_block_ptr(
        Ad, (T, 16), (16, 1), (i_t * 32, 0), (16, 16), (1, 0)
    )
    p_Ad_22 = tl.make_block_ptr(
        Ad, (T, 16), (16, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )
    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, 32), (32, 1), (i_t * 32, 0), (16, 16), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, 32), (32, 1), (i_t * 32 + 16, 16), (16, 16), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, 32), (32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
    )
    tl.store(
        p_Ai_11,
        Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def merge_32x32_to_64x64_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    T_max = T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if IS_VARLEN:
        A += (i_h * T_max + bos) * BT
        Ad += (i_h * T_max + bos) * 32
        Ai += (i_h * T_max + bos) * 64
    else:
        A += (i_b * H + i_h) * T_max * BT
        Ad += (i_b * H + i_h) * T_max * 32
        Ai += (i_b * H + i_h) * T_max * 64

    p_A_21 = tl.make_block_ptr(
        A, (T, BT), (BT, 1), (i_t * 64 + 32, 0 + i_t % (BT // 64) * 64), (32, 32), (1, 0)
    )

    p_Ad_11 = tl.make_block_ptr(
        Ad, (T, 32), (32, 1), (i_t * 64, 0), (32, 32), (1, 0)
    )
    p_Ad_22 = tl.make_block_ptr(
        Ad, (T, 32), (32, 1), (i_t * 64 + 32, 0), (32, 32), (1, 0)
    )

    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, 64), (64, 1), (i_t * 64, 0), (32, 32), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, 64), (64, 1), (i_t * 64 + 32, 32), (32, 32), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, 64), (64, 1), (i_t * 64 + 32, 0), (32, 32), (1, 0)
    )

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
    )
    tl.store(
        p_Ai_11,
        Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )

@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def merge_64x64_to_128x128_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    T_max = T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if IS_VARLEN:
        A += (i_h * T_max + bos) * 128
        Ad += (i_h * T_max + bos) * 64
        Ai += (i_h * T_max + bos) * 128
    else:
        A += (i_b * H + i_h) * T_max * 128
        Ad += (i_b * H + i_h) * T_max * 64
        Ai += (i_b * H + i_h) * T_max * 128

    p_A_21 = tl.make_block_ptr(
        A, (T, 128), (128, 1), (i_t * 128 + 64, 0), (64, 64), (1, 0)
    )

    p_Ad_11 = tl.make_block_ptr(
        Ad, (T, 64), (64, 1), (i_t * 128, 0), (64, 64), (1, 0)
    )
    p_Ad_22 = tl.make_block_ptr(
        Ad, (T, 64), (64, 1), (i_t * 128 + 64, 0), (64, 64), (1, 0)
    )

    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, 128), (128, 1), (i_t * 128, 0), (64, 64), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, 128), (128, 1), (i_t * 128 + 64, 64), (64, 64), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, 128), (128, 1), (i_t * 128 + 64, 0), (64, 64), (1, 0)
    )

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"), Ai_11, input_precision="ieee"
    )
    tl.store(
        p_Ai_11,
        Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )

@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_inverse_kernel_reorder_all_masked(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    T_max = T

    if IS_VARLEN:
        i_n, i_t_val = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        i_t = i_t_val
    else:
        bos, eos = i_b * T, i_b * T + T

    # A, Ad and Ai are BHTC: each head owns a contiguous T_max-by-BT plane.
    if IS_VARLEN:
        A += (i_h * T_max + bos) * 64
        Ad += (i_h * T_max + bos) * 16
        Ai += (i_h * T_max + bos) * 64
    else:
        A += (i_b * H + i_h) * T_max * 64
        Ad += (i_b * H + i_h) * T_max * 16
        Ai += (i_b * H + i_h) * T_max * 64

    # ------------------ Load Ai_22 (Ad block at row i_t*64+16, col 0, 16x16) ------------------
    offs_m = i_t * 64 + 16 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * 16 + offs_n[None, :]
    Ai_22 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    # ------------------ Load A_21 (A block at row i_t*64+16, col 0, 16x16) ------------------
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)  # A has 64 cols
    ptr_A = A + offs_m[:, None] * 64 + offs_n[None, :]
    A_21 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

    tmp = tl.dot(Ai_22, A_21, input_precision="ieee")

    # ------------------ Load Ai_11 (Ad block at row i_t*64, col 0, 16x16) ------------------
    offs_m = i_t * 64 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * 16 + offs_n[None, :]
    Ai_11 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    Ai_21 = -tl.dot(tmp, Ai_11, input_precision="ieee")

    # ------------------ Load Ai_44 (Ad block at row i_t*64+48, col 0, 16x16) ------------------
    offs_m = i_t * 64 + 48 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * 16 + offs_n[None, :]
    Ai_44 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    # ------------------ Load A_43 (A block at row i_t*64+48, col 32, 16x16) ------------------
    offs_n = 32 + tl.arange(0, 16)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + offs_m[:, None] * 64 + offs_n[None, :]
    A_43 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

    tmp = tl.dot(Ai_44, A_43, input_precision="ieee")

    # ------------------ Load Ai_33 (Ad block at row i_t*64+32, col 0, 16x16) ------------------
    offs_m = i_t * 64 + 32 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * 16 + offs_n[None, :]
    Ai_33 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    Ai_43 = -tl.dot(tmp, Ai_33, input_precision="ieee")

    # ------------------ Build Ai_22_32 (32x32) ------------------
    Ai_22_32 = tl.zeros((32, 32), tl.float32)
    Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 16), (1, 1))
    Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16, 16), (1, 1))
    Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16, 16), (1, 1))

    # ------------------ Load A_21_32 (A block at row i_t*64+32, col 0, 32x32) ------------------
    offs_m = i_t * 64 + 32 + tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + offs_m[:, None] * 64 + offs_n[None, :]
    A_21_32 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

    tmp = tl.dot(Ai_22_32, A_21_32, input_precision="ieee")

    # ------------------ Build Ai_11_32 (32x32) ------------------
    Ai_11_32 = tl.zeros((32, 32), tl.float32)
    Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 16), (1, 1))
    Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16), (1, 1))
    Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16), (1, 1))

    Ai_21_32 = -tl.dot(tmp, Ai_11_32, input_precision="ieee")

    # ------------------ Store Ai_11_32 to (i_t*64, 0) ------------------
    offs_m = i_t * 64 + tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * 64 + offs_n[None, :]
    tl.store(
        ptr_Ai,
        Ai_11_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_store,
    )

    # ------------------ Store Ai_22_32 to (i_t*64+32, 32) ------------------
    offs_m = i_t * 64 + 32 + tl.arange(0, 32)
    offs_n = 32 + tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * 64 + offs_n[None, :]
    tl.store(
        ptr_Ai,
        Ai_22_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_store,
    )

    # ------------------ Store Ai_21_32 to (i_t*64+32, 0) ------------------
    offs_n = tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * 64 + offs_n[None, :]
    tl.store(
        ptr_Ai,
        Ai_21_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_store,
    )

    # ------------------ Zero out the upper-right 32x32 block (rows 0~31, cols 32~63) ------------------
    # offs_m = i_t * 64 + tl.arange(0, 32)
    # offs_n = 32 + tl.arange(0, 32)
    # mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < BT)  # BT=64
    # ptr_Ai = Ai + offs_m[:, None] * (H * BT) + offs_n[None, :]
    # zero_block = tl.zeros((32, 32), dtype=ptr_Ai.dtype.element_ty)
    # tl.store(ptr_Ai, zero_block, mask=mask_store)


SOLVE_TRIL_LARGE_BLOCK_SIZE = 608 * 2


def _chunk_indices(
    cu_seqlens: torch.Tensor | None,
    prebuilt: dict[str, torch.Tensor | None] | None,
    chunk_size: int,
) -> torch.Tensor | None:
    if cu_seqlens is None:
        return None
    indices = None if prebuilt is None else prebuilt.get(str(chunk_size))
    return prepare_chunk_indices(cu_seqlens, chunk_size) if indices is None else indices


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices_out: dict[str, torch.Tensor | None] | None = None,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    """Return ``(I + A)^{-1}`` for strict lower-triangular BHTC blocks.

    ``A`` stays in head-first ``[B, H, T, BT]`` memory order from KKT through
    recompute.  ``cu_seqlens`` is a device tensor in varlen mode, and callers
    may supply device-resident chunk indices to avoid per-layer host syncs.
    """
    if A.ndim != 4 or A.shape[-1] not in (16, 32, 64, 128):
        raise ValueError(f"expected BHTC input with BT in (16, 32, 64, 128), got {tuple(A.shape)}")

    output_dtype = A.dtype if output_dtype is None else output_dtype
    B, H, T, BT = A.shape
    Ad = torch.empty(B, H, T, 16, device=A.device, dtype=torch.float if BT != 16 else output_dtype)

    chunk_indices = _chunk_indices(cu_seqlens, chunk_indices_out, SOLVE_TRIL_LARGE_BLOCK_SIZE)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, SOLVE_TRIL_LARGE_BLOCK_SIZE)
    solve_tril_16x16_kernel_paral_v3[NT, B * H](
        A=A,
        Ad=Ad,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        LARGE_BLOCK_T=SOLVE_TRIL_LARGE_BLOCK_SIZE,
    )
    if BT == 16:
        return Ad

    Ai = torch.zeros(B, H, T, 32, device=A.device, dtype=torch.float if BT != 32 else output_dtype)
    chunk_indices = _chunk_indices(cu_seqlens, chunk_indices_out, 32)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, 32)
    merge_16x16_to_32x32_inverse_kernel[NT, B * H](
        A=A,
        Ad=Ad,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
    )
    if BT == 32:
        return Ai

    Ad = Ai
    Ai = torch.zeros(B, H, T, 64, device=A.device, dtype=torch.float if BT != 64 else output_dtype)
    chunk_indices = _chunk_indices(cu_seqlens, chunk_indices_out, 64)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, 64)
    merge_32x32_to_64x64_inverse_kernel[NT, B * H](
        A=A,
        Ad=Ad,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
    )
    if BT == 64:
        return Ai

    Aii = torch.zeros_like(A, dtype=output_dtype)
    chunk_indices = _chunk_indices(cu_seqlens, chunk_indices_out, 128)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, 128)
    merge_64x64_to_128x128_inverse_kernel[NT, B * H](
        A=A,
        Ad=Ai,
        Ai=Aii,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
    )
    return Aii
