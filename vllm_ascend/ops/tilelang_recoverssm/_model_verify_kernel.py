# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack and launch the A3 shape-generic model Verify kernels."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from ._model_cube import build_pipeline

_CUBE_CORES = 24
_JOBS_PER_ROW = 8
_PIPELINE_DEPTH = 2


@triton.jit
def pack_model_verify(
    Q,
    K,
    A,
    B,
    ALog,
    Bias,
    Starts,
    Packed,
    Gates,
    Slots,
    ActivePages,
    TOKENS: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    A_STRIDE: tl.constexpr,
    B_STRIDE: tl.constexpr,
    JOBS: tl.constexpr,
    COMPACT: tl.constexpr = False,
    PAIR_HEADS: tl.constexpr = False,
    PRECOMPUTED_EXP: tl.constexpr = False,
    SHARE_PAGES: tl.constexpr = False,
    INDEX_STRIDE: tl.constexpr = 1,
    PAGES: tl.constexpr = 0,
    DENSE_QK: tl.constexpr = False,
    PAGE_ROWS: tl.constexpr = 0,
):
    program = tl.program_id(0)
    TILE_JOBS: tl.constexpr = JOBS // (2 if PAIR_HEADS else 1)
    if PAIR_HEADS:
        if program < TILE_JOBS:
            row = program // 4
            pair = program % 4
            begin = tl.load(Starts + row)
            end = tl.load(Starts + row + 1)
            valid = (begin >= 0) & (end <= TOKENS) & (end - begin <= 4)
            plane = tl.arange(0, 8)
            cols = tl.arange(0, 256)
            token = begin + plane % 4
            element = pair * 256 + cols[None, :]
            active = valid & (token < end)
            key = tl.load(K + token[:, None] * K_STRIDE + element, (active & (plane < 4))[:, None], other=0)
            query = tl.load(Q + token[:, None] * Q_STRIDE + element, (active & (plane >= 4))[:, None], other=0)
            raw = tl.where(plane[:, None] < 4, key, query)
            job = program * 2 + cols // 128
            PACK_STRIDE: tl.constexpr = 1024 if DENSE_QK else 2048
            offset = job[None, :] * PACK_STRIDE + plane[:, None] * 128 + cols[None, :] % 128
            tl.store(Packed + offset, raw)
            if not DENSE_QK:
                tl.store(Packed + offset + 1024, 0)
            fp32 = tl.reshape(raw.to(tl.float32), (8, 2, 128))
            norm = tl.rsqrt(tl.sum(fp32 * fp32, axis=2) + 1e-6)
            tl.store(Gates + (program * 2 + tl.arange(0, 2)[None, :]) * 48 + 32 + plane[:, None], norm)
    elif program < TILE_JOBS:
        row, head = program // 8, program % 8
        begin = tl.load(Starts + row)
        end = tl.load(Starts + row + 1)
        valid = (begin >= 0) & (end <= TOKENS) & (end - begin <= 4)
        plane = tl.arange(0, 8 if COMPACT else 16)
        cols = tl.arange(0, 128)
        qk_token = begin + plane % 4
        element = head * 128 + cols[None, :]
        qk_active = valid & (qk_token < end)
        key = tl.load(K + qk_token[:, None] * K_STRIDE + element, (qk_active & (plane < 4))[:, None], other=0)
        query = tl.load(
            Q + qk_token[:, None] * Q_STRIDE + element, (qk_active & (plane >= 4) & (plane < 8))[:, None], other=0
        )
        raw = tl.where(plane[:, None] < 4, key, query)
        tl.store(Packed + program * 2048 + plane[:, None] * 128 + cols[None, :], raw)
        if COMPACT:
            tl.store(Packed + program * 2048 + (plane[:, None] + 8) * 128 + cols[None, :], 0)
        fp32 = raw.to(tl.float32)
        norm = tl.rsqrt(tl.sum(fp32 * fp32, axis=1) + 1e-6)
        tl.store(Gates + program * 48 + 32 + plane, norm)
    if program >= TILE_JOBS:
        row = program - TILE_JOBS
        begin = tl.load(Starts + row)
        end = tl.load(Starts + row + 1)
        valid = (begin >= 0) & (end <= TOKENS) & (end > begin) & (end - begin <= 4)
        if SHARE_PAGES:
            page = tl.load(Slots + row * INDEX_STRIDE)
            tl.store(ActivePages + row, tl.where(valid & (page > 0) & (page < PAGES), page, 0))
        if PAGE_ROWS:
            page = tl.load(Slots + row * INDEX_STRIDE)
            tl.store(ActivePages + row * 2, page)
            tl.store(ActivePages + row * 2 + 1, tl.minimum(tl.maximum(page, 0), PAGES - 1) * PAGE_ROWS)
        lane = tl.arange(0, 128)
        gate_head = lane // 16
        value_head = gate_head * 3 + lane % 4
        token = begin + lane % 16 // 4
        active = valid & (token < end) & (lane % 4 < 3)
        a = tl.load(A + token * A_STRIDE + value_head, active, other=0).to(tl.float32)
        b = tl.load(B + token * B_STRIDE + value_head, active, other=0).to(tl.float32)
        bias = tl.load(Bias + value_head, lane % 4 < 3, other=0).to(tl.float32)
        a_const = tl.load(ALog + value_head, lane % 4 < 3, other=1.0 if PRECOMPUTED_EXP else 0.0).to(tl.float32)
        if not PRECOMPUTED_EXP:
            a_const = tl.exp(a_const)
        x = a + bias
        softplus = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        decay = tl.exp(-a_const * softplus)
        beta = tl.sigmoid(b)
        offset = (row * 8 + gate_head) * 48 + lane % 16
        tl.store(Gates + offset, decay)
        tl.store(Gates + offset + 16, beta)


@lru_cache(maxsize=64)
def _get_kernel(
    rows,
    pages,
    tokens,
    state_stride,
    replay_stride,
    index_stride,
    value_stride,
    gate_stride,
    q_stride,
    k_stride,
    z_stride,
    norm_eps,
):
    return build_pipeline(
        rows,
        pages,
        state_stride,
        replay_stride,
        index_stride,
        value_stride,
        depth=_PIPELINE_DEPTH,
        variant="model",
        gate_stride=gate_stride,
        token_count=tokens,
        prefetch_state=True,
        directional_fix_sync=True,
        directional_input_sync=True,
        q_stride=q_stride,
        k_stride=k_stride,
        z_stride=z_stride,
        norm_eps=norm_eps,
        norm_activation="sigmoid" if norm_eps is not None else "silu",
        vector_rstd=norm_eps is not None,
        address_mode="pack",
    )


def run_model_verify(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    *,
    checkpoint_state,
    replay_cache,
    query_start_loc,
    state_indices,
    out,
    z,
    norm_weight,
    norm_eps,
):
    rows, tokens = state_indices.numel(), q.shape[1]
    pages = checkpoint_state.shape[0]
    jobs = rows * _JOBS_PER_ROW
    a, b = a.view(tokens, 24), b.view(tokens, 24)
    A_log, dt_bias = A_log.float(), dt_bias.float()
    if z is None:
        # Unused operands retain the fixed kernel ABI without allocating extra buffers.
        z = norm_weight = v
    else:
        norm_eps = float(norm_eps)

    packed = torch.empty((jobs, 16, 128), dtype=torch.bfloat16, device=q.device)
    gates = torch.empty((jobs, 48), dtype=torch.float32, device=q.device)
    projection = torch.empty((_CUBE_CORES * _PIPELINE_DEPTH, 16, 384), dtype=torch.float32, device=q.device)
    gram = torch.empty((_CUBE_CORES * _PIPELINE_DEPTH, 16, 16), dtype=torch.float32, device=q.device)
    page_table = torch.empty((rows, 2), dtype=torch.int32, device=q.device)
    if out is None:
        out = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    kernel = _get_kernel(
        rows,
        pages,
        tokens,
        checkpoint_state.stride(0),
        replay_cache.stride(0),
        state_indices.stride(0),
        v.stride(1),
        a.stride(0),
        q.stride(1),
        k.stride(1),
        z.stride(1),
        norm_eps,
    )
    pack_model_verify[(jobs // 2 + rows,)](
        q,
        k,
        a,
        b,
        A_log,
        dt_bias,
        query_start_loc,
        packed,
        gates,
        state_indices,
        page_table,
        tokens,
        q.stride(1),
        k.stride(1),
        a.stride(0),
        b.stride(0),
        jobs,
        PAIR_HEADS=True,
        INDEX_STRIDE=state_indices.stride(0),
        PAGES=pages,
        PAGE_ROWS=checkpoint_state.stride(0) // 128,
        enable_fp_fusion=False,
    )
    kernel.adapter._forward_from_prebuild_lib(
        checkpoint_state,
        page_table,
        packed,
        gates,
        projection,
        gram,
        v,
        replay_cache,
        query_start_loc,
        out,
        a,
        b,
        A_log,
        dt_bias,
        q,
        k,
        z,
        norm_weight,
        stream=torch.npu.current_stream(q.device).npu_stream,
    )
    return out
