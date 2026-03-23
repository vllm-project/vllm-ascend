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

from vllm_ascend.ops.triton.triton_utils import extract_slice, insert_slice
from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def solve_tril_16x16_kernel(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    LARGE_BLOCK_T: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A = A + (bos * H + i_h) * BT
    Ad = Ad + (bos * H + i_h) * 16

    base_t = i_t * LARGE_BLOCK_T

    NTASKS: tl.constexpr = 2
    N_BLOCKS: tl.constexpr = LARGE_BLOCK_T // 16 // NTASKS
    # Number of 16x16 blocks per BT-column group (= BT // 16)
    BT_BLOCKS: tl.constexpr = BT // 16
    # Number of BT-wide groups per task
    N_GROUPS: tl.constexpr = N_BLOCKS // BT_BLOCKS

    # Pre-compute constant masks
    # Since A is strictly lower triangular, its upper triangle is already 0,
    # so we only need the diagonal mask to add identity.
    offs16 = tl.arange(0, 16)
    # identity_diag: 1.0 on diagonal, 0.0 elsewhere (float32)
    identity_diag = (offs16[:, None] == offs16[None, :]).to(tl.float32)

    # BT-wide constant offsets for group loads
    offs_bt_rows = tl.arange(0, BT)
    offs_bt_cols = tl.arange(0, BT)

    for taskid in range(0, NTASKS):
        base_t += taskid * (LARGE_BLOCK_T // NTASKS)

        # --- Vectorized load phase: N_GROUPS BT×BT group loads ---
        # Each group covers BT_BLOCKS consecutive 16x16 diagonal blocks.
        # LARGE_BLOCK_T and task offsets are multiples of BT, so col alignment is clean.
        b_A = tl.zeros((N_BLOCKS, 16, 16), dtype=tl.float32)

        for g in range(0, N_GROUPS):
            grp_row_start = base_t + g * BT
            global_grp_rows = grp_row_start + offs_bt_rows

            ptr_grp = (
                A
                + global_grp_rows[:, None] * H * BT
                + offs_bt_cols[None, :]
            )
            # No mask: out-of-bounds rows load garbage but store phase applies row_mask
            block_BT = tl.load(ptr_grp).to(tl.float32)

            # Extract BT_BLOCKS diagonal 16×16 sub-blocks and insert into b_A
            for b in range(0, BT_BLOCKS):
                sub = extract_slice(block_BT, (b * 16, b * 16), (16, 16), (1, 1))
                blk_idx = g * BT_BLOCKS + b
                b_A = insert_slice(
                    ful=b_A,
                    sub=sub[None, :, :],
                    offsets=[blk_idx, 0, 0],
                    sizes=[1, 16, 16],
                    strides=[1, 1, 1],
                )

        # --- Compute phase ---
        local_ori_A = tl.trans(b_A, (1, 0, 2))
        local_ori_A = tl.reshape(local_ori_A, (16, 16 * N_BLOCKS))
        # A is strictly lower triangular, so upper triangle is already 0.
        # We only need to negate (not mask) since zeros stay zeros under negation.
        b_A = -b_A

        for i in range(1, 16):
            nblks_vec16 = -extract_slice(local_ori_A, (i, 0), (1, 16 * N_BLOCKS), (16 * N_BLOCKS, 1))
            b_a = tl.reshape(nblks_vec16, (N_BLOCKS, 16))

            dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2))
            dot_product = tl.sum(dot_tmp, 0)
            b_a = b_a + dot_product

            b_a_new_expanded = b_a[:, None, :]
            b_A = insert_slice(
                ful=b_A, sub=b_a_new_expanded, offsets=[0, i, 0], sizes=[N_BLOCKS, 1, 16], strides=[1, 1, 1]
            )

        # Add identity (1.0 on diagonal) to complete (I - A_lower)^{-1}
        b_A = b_A + identity_diag
        b_A = tl.reshape(b_A, (N_BLOCKS * 16, 16))

        # --- Store phase: use make_block_ptr for efficient contiguous store ---
        p_Ai_store = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (base_t, 0), (N_BLOCKS * 16, 16), (1, 0))
        tl.store(p_Ai_store, b_A.to(p_Ai_store.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0,))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * 32
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 32

    p_A_21 = tl.make_block_ptr(A, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))
    p_Ad_11 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 32, 0), (16, 16), (1, 0))
    p_Ad_22 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))
    p_Ai_11 = tl.make_block_ptr(Ai, (T, 32), (H * 32, 1), (i_t * 32, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 16), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"),
        Ai_11,
        input_precision="ieee",
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


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T", "NT_CHUNKS"])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MERGE_BATCH: tl.constexpr,
    NT_CHUNKS,
):
    i_t_base, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    offs_16 = tl.arange(0, 16)
    offs_32 = tl.arange(0, 32)

    # Process MERGE_BATCH consecutive BT=64-row blocks
    for batch_idx in range(0, MERGE_BATCH):
        chunk_idx = i_t_base * MERGE_BATCH + batch_idx
        # Clamp to last valid chunk to avoid OOB access on chunk_indices
        safe_chunk_idx = tl.minimum(chunk_idx, NT_CHUNKS - 1)

        if IS_VARLEN:
            i_n = tl.load(chunk_indices + safe_chunk_idx * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + safe_chunk_idx * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_local = eos - bos
        else:
            i_t = chunk_idx
            bos = i_b * T
            eos = bos + T
            T_local = T

        # If this chunk is out of range (padding), set T_local=0 so all masks are False
        T_local = tl.where(chunk_idx < NT_CHUNKS, T_local, 0)

        # Base pointers offset by batch and head
        A_base = A + (bos * H + i_h) * 64
        Ad_base = Ad + (bos * H + i_h) * 16
        Ai_base = Ai + (bos * H + i_h) * 64

        # load Ai_22 (Ad block at row i_t * 64 + 16, col 0, 16 * 16)
        offs_m = i_t * 64 + 16 + offs_16
        mask_Ad = (offs_m[:, None] < T_local) & (offs_16[None, :] < 16)
        ptr_Ad = Ad_base + offs_m[:, None] * (H * 16) + offs_16[None, :]
        Ai_22 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

        # load A_21 (A block at row i_t * 64 + 16, col 0, 16 * 16)
        mask_A = (offs_m[:, None] < T_local) & (offs_16[None, :] < 64)
        ptr_A = A_base + offs_m[:, None] * (H * 64) + offs_16[None, :]
        A_21 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)
        tmp = tl.dot(Ai_22, A_21, input_precision="ieee")

        # load Ai_11 (Ad block at row i_t * 64, col 0, 16 * 16)
        offs_m = i_t * 64 + offs_16
        mask_Ad = (offs_m[:, None] < T_local) & (offs_16[None, :] < 16)
        ptr_Ad = Ad_base + offs_m[:, None] * (H * 16) + offs_16[None, :]
        Ai_11 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

        Ai_21 = -tl.dot(tmp, Ai_11, input_precision="ieee")

        # load Ai_44 (Ad block at row i_t * 64 + 48, col 0, 16 * 16)
        offs_m = i_t * 64 + 48 + offs_16
        mask_Ad = (offs_m[:, None] < T_local) & (offs_16[None, :] < 16)
        ptr_Ad = Ad_base + offs_m[:, None] * (H * 16) + offs_16[None, :]
        Ai_44 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

        # load A_43 (Ad block at row i_t * 64 + 48, col 32, 16 * 16)
        offs_n = 32 + offs_16
        mask_A = (offs_m[:, None] < T_local) & (offs_n[None, :] < 64)
        ptr_A = A_base + offs_m[:, None] * (H * 64) + offs_n[None, :]
        A_43 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)
        tmp = tl.dot(Ai_44, A_43, input_precision="ieee")

        # load Ai_33 (Ad block at row i_t * 64 + 32, col 0, 16 * 16)
        offs_m = i_t * 64 + 32 + offs_16
        mask_Ad = (offs_m[:, None] < T_local) & (offs_16[None, :] < 16)
        ptr_Ad = Ad_base + offs_m[:, None] * (H * 16) + offs_16[None, :]
        Ai_33 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

        Ai_43 = -tl.dot(tmp, Ai_33, input_precision="ieee")

        # build Ai_22_32 (32 * 32)
        Ai_22_32 = tl.zeros((32, 32), tl.float32)
        Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 16), (1, 1))
        Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16, 16), (1, 1))
        Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16, 16), (1, 1))

        # load A_21_32 (A block at row i_t * 64 + 32, col 0, 32 * 32)
        offs_m = i_t * 64 + 32 + offs_32
        mask_A = (offs_m[:, None] < T_local) & (offs_32[None, :] < 64)
        ptr_A = A_base + offs_m[:, None] * (H * 64) + offs_32[None, :]
        A_21_32 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)
        tmp = tl.dot(Ai_22_32, A_21_32, input_precision="ieee")

        # build Ai_11_32 (32 * 32)
        Ai_11_32 = tl.zeros((32, 32), tl.float32)
        Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 16), (1, 1))
        Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16), (1, 1))
        Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16), (1, 1))

        Ai_21_32 = -tl.dot(tmp, Ai_11_32, input_precision="ieee")

        # store Ai_11_32 to (i_t * 64, 0)
        offs_m = i_t * 64 + offs_32
        mask_store = (offs_m[:, None] < T_local) & (offs_32[None, :] < 64)
        ptr_Ai = Ai_base + offs_m[:, None] * (H * 64) + offs_32[None, :]
        tl.store(ptr_Ai, Ai_11_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_store)

        # store Ai_22_32 to (i_t * 64 + 32, 32)
        offs_m = i_t * 64 + 32 + offs_32
        offs_n_32_upper = 32 + offs_32
        mask_store = (offs_m[:, None] < T_local) & (offs_n_32_upper[None, :] < 64)
        ptr_Ai = Ai_base + offs_m[:, None] * (H * 64) + offs_n_32_upper[None, :]
        tl.store(ptr_Ai, Ai_22_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_store)

        # store Ai_21_32 to (i_t * 64 + 32, 0)
        mask_store = (offs_m[:, None] < T_local) & (offs_32[None, :] < 64)
        ptr_Ai = Ai_base + offs_m[:, None] * (H * 64) + offs_32[None, :]
        tl.store(ptr_Ai, Ai_21_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_store)

        # zero out the upper-right 32 * 32 block (rows 0 ~ 31, cols 32 ~ 63)
        offs_m_upper = i_t * 64 + offs_32
        offs_n_upper = 32 + offs_32
        mask_store = (offs_m_upper[:, None] < T_local) & (offs_n_upper[None, :] < BT)
        ptr_Ai = Ai_base + offs_m_upper[:, None] * (H * BT) + offs_n_upper[None, :]
        zero_block = tl.zeros((32, 32), dtype=ptr_Ai.dtype.element_ty)
        tl.store(ptr_Ai, zero_block, mask=mask_store)


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices_large_block: torch.Tensor | None = None,
    chunk_indices_bt: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, BT], where BT should only be 16, 32, or 64.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`.
            If `None`, the output dtype will be the same as the input dtype.

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64]

    B, T, H, BT = A.shape
    Ad = torch.empty(B, T, H, 16, device=A.device, dtype=torch.float if BT != 16 else output_dtype)

    LARGE_BLOCK_T = 768

    if cu_seqlens is not None and chunk_indices_large_block is None:
        chunk_indices_large_block = prepare_chunk_indices(cu_seqlens, LARGE_BLOCK_T)
    chunk_indices = chunk_indices_large_block
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, LARGE_BLOCK_T)

    solve_tril_16x16_kernel[NT, B * H](
        A=A,
        Ad=Ad,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        LARGE_BLOCK_T=LARGE_BLOCK_T,
        num_warps=1,
        num_stages=4,
    )

    if BT == 16:
        return Ad

    Ai = torch.empty(B, T, H, BT, device=A.device, dtype=output_dtype)
    merge_fn = merge_16x16_to_32x32_inverse_kernel if BT == 32 else merge_16x16_to_64x64_inverse_kernel
    if cu_seqlens is not None and chunk_indices_bt is None:
        chunk_indices_bt = prepare_chunk_indices(cu_seqlens, BT)
    chunk_indices = chunk_indices_bt
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    if BT == 64:
        # Process multiple 64-row blocks per kernel to reduce launch overhead
        MERGE_BATCH = 1
        NT_merge = triton.cdiv(NT, MERGE_BATCH)
        merge_fn[NT_merge, B * H](
            A=A,
            Ad=Ad,
            Ai=Ai,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            BT=BT,
            MERGE_BATCH=MERGE_BATCH,
            NT_CHUNKS=NT,
            num_warps=4,
            num_stages=3,
        )
    else:
        merge_fn[NT, B * H](
            A=A,
            Ad=Ad,
            Ai=Ai,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            BT=BT,
            num_warps=4,
            num_stages=3,
        )
    return Ai
