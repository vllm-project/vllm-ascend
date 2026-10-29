# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend RecoverSSM verify and batched accepted-state recovery.

Adapted from vLLM c4df37dfcc. Preserve its paged-cache strides and public
verify/commit contracts. Q/K normalization follows the AscendC KDA kernel.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.triton.triton_utils import get_ub_size_bytes, init_device_properties_triton

RECOVERSSM_COMMIT_HEAD_TILE = 2
RECOVERSSM_VALUE_TILE = 64
RECOVERSSM_VALUE_TILE_LARGE = 128
RECOVERSSM_LARGE_TILE_MIN_PROGRAMS = 96
RECOVERSSM_CONV_TILE = 4096
RECOVERSSM_CONV_STRIDED_TILE = 256


@triton.jit
def _kda_l2_normalize(vector):
    # AscendC normalizes nonzero vectors without CUDA's squared-norm bias.
    norm_square = tl.sum(vector * vector)
    norm_input = tl.where(norm_square > 0.0, norm_square, 1.0)
    tl.static_assert(norm_input.dtype == tl.float32)
    inverse = tl.rsqrt(norm_input)
    return tl.where(norm_square > 0.0, vector * inverse, 0.0)


@triton.jit
def _kda_gate(
    raw_g,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
):
    gate_input = raw_g + dt_bias
    if USE_LOWER_BOUND:
        scaled_gate = A * gate_input
        tl.static_assert(scaled_gate.dtype == tl.float32)
        return lower_bound * tl.sigmoid(scaled_gate)
    tl.static_assert(gate_input.dtype == tl.float32)
    softplus_input = 1.0 + tl.exp(gate_input)
    tl.static_assert(softplus_input.dtype == tl.float32)
    log_softplus = tl.log(softplus_input)
    softplus_gate = tl.where(
        gate_input > 20.0,
        gate_input,
        log_softplus,
    )
    return -A * softplus_gate


@triton.jit
def _kda_recurrent_step(
    state,
    k,
    v,
    raw_g,
    raw_beta,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
):
    normalized_k = _kda_l2_normalize(k)
    gate = _kda_gate(
        raw_g,
        dt_bias,
        A,
        lower_bound,
        USE_LOWER_BOUND,
    )

    tl.static_assert(gate.dtype == tl.float32)
    state *= tl.exp(gate)[None, :]
    correction = v - tl.sum(state * normalized_k[None, :], axis=1)
    tl.static_assert(raw_beta.dtype == tl.float32)
    correction *= tl.sigmoid(raw_beta)
    return state + correction[:, None] * normalized_k[None, :], correction


@triton.jit
def _kda_recoverssm_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    raw_g_ptr,
    raw_beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction_cache_ptr,
    kg_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    lower_bound,
    null_block_id,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_g_token,
    stride_beta_token,
    stride_state_block,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_block,
    stride_correction_head,
    stride_correction_pos,
    stride_correction_dim,
    stride_kg_block,
    stride_kg_head,
    stride_kg_pos,
    stride_kg_dim,
    stride_out_token,
    stride_query_start_loc,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    # Advance a register-held checkpoint and record each token's compact
    # correction/key/gate; persistent checkpoint pages remain read-only.
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(2)
    pid_h = tl.program_id(1)

    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(tl.int64)
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    # Provably-full tiles (K a power of two, BV dividing V) drop their
    # predication masks entirely; ragged shapes keep the original masks.
    if K == BK:
        mask_k = None
    else:
        mask_k = offs_k < K
    if V % BV == 0:
        mask_v = None
    else:
        mask_v = offs_v < V
    if (K == BK) and (V % BV == 0):
        mask_state = None
    elif K == BK:
        mask_state = tl.broadcast_to((offs_v < V)[:, None], (BV, BK))
    elif V % BV == 0:
        mask_state = tl.broadcast_to((offs_k < K)[None, :], (BV, BK))
    else:
        mask_state = (offs_v < V)[:, None] & (offs_k < K)[None, :]

    if state_idx <= null_block_id:
        for token_offset in range(query_len):
            tl.store(
                out_ptr + (bos + token_offset) * stride_out_token + pid_h * V + offs_v,
                tl.zeros([BV], dtype=tl.float32),
                mask=mask_v,
            )
        return

    state_ptrs = (
        state_ptr
        + state_idx * stride_state_block
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    state = tl.load(state_ptrs, mask=mask_state, other=0.0 if mask_state is not None else None).to(tl.float32)
    a_log = tl.load(A_log_ptr + pid_h).to(tl.float32)
    tl.static_assert(a_log.dtype == tl.float32)
    A = tl.exp(a_log)

    # token_offset < query_len always holds inside range(query_len); the
    # token_valid masks and the tl.where state select are redundant and dropped.
    dt_bias = tl.load(
        dt_bias_ptr + pid_h * K + offs_k,
        mask=mask_k,
        other=0.0 if mask_k is not None else None,
    ).to(tl.float32)
    for token_offset in range(tl.maximum(query_len - 1, 0)):
        token = bos + token_offset
        q = tl.load(
            q_ptr + token * stride_q_token + pid_h * K + offs_k,
            mask=mask_k,
            other=0.0 if mask_k is not None else None,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + token * stride_k_token + pid_h * K + offs_k,
            mask=mask_k,
            other=0.0 if mask_k is not None else None,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + token * stride_v_token + pid_h * V + offs_v,
            mask=mask_v,
            other=0.0 if mask_v is not None else None,
        ).to(tl.float32)
        raw_g = tl.load(
            raw_g_ptr + token * stride_g_token + pid_h * K + offs_k,
            mask=mask_k,
            other=0.0 if mask_k is not None else None,
        ).to(tl.float32)
        raw_beta = tl.load(
            raw_beta_ptr + token * stride_beta_token + pid_h,
        ).to(tl.float32)

        q = _kda_l2_normalize(q) * (K**-0.5)
        normalized_k = _kda_l2_normalize(k)
        gate = _kda_gate(raw_g, dt_bias, A, lower_bound, USE_LOWER_BOUND)
        tl.static_assert(gate.dtype == tl.float32)
        decay = tl.exp(gate)
        weighted_k = normalized_k * decay
        weighted_q = q * decay
        tl.static_assert(raw_beta.dtype == tl.float32)
        correction = (v - tl.sum(state * weighted_k[None, :], axis=1)) * tl.sigmoid(raw_beta)
        out = tl.sum(state * weighted_q[None, :], axis=1) + correction * tl.sum(normalized_k * q)
        state = state * decay[None, :] + correction[:, None] * normalized_k[None, :]
        tl.store(
            out_ptr + token * stride_out_token + pid_h * V + offs_v,
            out,
            mask=mask_v,
        )

        correction_ptr = (
            correction_cache_ptr
            + state_idx * stride_correction_block
            + pid_h * stride_correction_head
            + token_offset * stride_correction_pos
        )
        tl.store(
            correction_ptr + offs_v * stride_correction_dim,
            correction,
            mask=mask_v,
        )
        if pid_v == 0:
            kg_ptr = kg_cache_ptr + state_idx * stride_kg_block + pid_h * stride_kg_head + token_offset * stride_kg_pos
            tl.store(
                kg_ptr + offs_k * stride_kg_dim,
                k,
                mask=mask_k,
            )
            tl.store(
                kg_ptr + (K + offs_k) * stride_kg_dim,
                raw_g,
                mask=mask_k,
            )
    if query_len > 0:
        token_offset = query_len - 1
        token = bos + token_offset
        q = tl.load(
            q_ptr + token * stride_q_token + pid_h * K + offs_k,
            mask=mask_k,
            other=0.0 if mask_k is not None else None,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + token * stride_k_token + pid_h * K + offs_k,
            mask=mask_k,
            other=0.0 if mask_k is not None else None,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + token * stride_v_token + pid_h * V + offs_v,
            mask=mask_v,
            other=0.0 if mask_v is not None else None,
        ).to(tl.float32)
        raw_g = tl.load(
            raw_g_ptr + token * stride_g_token + pid_h * K + offs_k,
            mask=mask_k,
            other=0.0 if mask_k is not None else None,
        ).to(tl.float32)
        raw_beta = tl.load(
            raw_beta_ptr + token * stride_beta_token + pid_h,
        ).to(tl.float32)

        q = _kda_l2_normalize(q) * (K**-0.5)
        normalized_k = _kda_l2_normalize(k)
        gate = _kda_gate(raw_g, dt_bias, A, lower_bound, USE_LOWER_BOUND)
        tl.static_assert(gate.dtype == tl.float32)
        decay = tl.exp(gate)
        weighted_k = normalized_k * decay
        weighted_q = q * decay
        tl.static_assert(raw_beta.dtype == tl.float32)
        correction = (v - tl.sum(state * weighted_k[None, :], axis=1)) * tl.sigmoid(raw_beta)
        out = tl.sum(state * weighted_q[None, :], axis=1) + correction * tl.sum(normalized_k * q)
        tl.store(
            out_ptr + token * stride_out_token + pid_h * V + offs_v,
            out,
            mask=mask_v,
        )

        correction_ptr = (
            correction_cache_ptr
            + state_idx * stride_correction_block
            + pid_h * stride_correction_head
            + token_offset * stride_correction_pos
        )
        tl.store(
            correction_ptr + offs_v * stride_correction_dim,
            correction,
            mask=mask_v,
        )
        if pid_v == 0:
            kg_ptr = kg_cache_ptr + state_idx * stride_kg_block + pid_h * stride_kg_head + token_offset * stride_kg_pos
            tl.store(
                kg_ptr + offs_k * stride_kg_dim,
                k,
                mask=mask_k,
            )
            tl.store(
                kg_ptr + (K + offs_k) * stride_kg_dim,
                raw_g,
                mask=mask_k,
            )


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "ALIGN_MODE": lambda args: args["block_table_ptr"] is not None,
    }
)
@triton.jit
def _prepare_commit_plan_kernel(
    num_accepted_ptr,
    request_indices_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    num_computed_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    mamba_block_size,
    block_table_width,
    stride_num_accepted,
    stride_request_indices,
    stride_state_indices,
    stride_query_start_loc,
    stride_block_table_row,
    stride_block_table_col,
    stride_num_computed,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    spec_idx = tl.program_id(0)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices).to(tl.int64)
    request_idx = spec_idx
    if HAS_REQUEST_INDICES:
        request_idx = tl.load(request_indices_ptr + spec_idx * stride_request_indices).to(tl.int64)
    num_accepted = tl.load(num_accepted_ptr + request_idx * stride_num_accepted).to(tl.int32)
    bos = tl.load(query_start_loc_ptr + spec_idx * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (spec_idx + 1) * stride_query_start_loc).to(tl.int64)
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)

    final_state_idx = source_state_idx
    boundary_state_idx = null_block_id
    boundary_recovery_len = 0
    if ALIGN_MODE:
        num_computed = tl.load(num_computed_ptr + request_idx * stride_num_computed).to(tl.int32)
        final_num_computed = num_computed + commit_len
        final_state_col = tl.minimum(final_num_computed // mamba_block_size, block_table_width - 1)
        final_state_idx = tl.load(
            block_table_ptr + request_idx * stride_block_table_row + final_state_col * stride_block_table_col
        ).to(tl.int64)
        next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
        crosses_boundary = final_num_computed >= next_boundary
        boundary_recovery_len = next_boundary - num_computed
        boundary_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + (next_boundary // mamba_block_size - 1) * stride_block_table_col,
            mask=crosses_boundary,
            other=null_block_id,
        ).to(tl.int64)
    valid = (source_state_idx > null_block_id) & (commit_len > 0)
    tl.store(commit_lens_ptr + spec_idx, tl.where(valid, commit_len, 0))
    tl.store(
        final_state_indices_ptr + spec_idx,
        tl.where(valid, final_state_idx, null_block_id),
    )
    tl.store(
        boundary_state_indices_ptr + spec_idx,
        tl.where(valid, boundary_state_idx, null_block_id),
    )
    tl.store(
        boundary_recovery_lens_ptr + spec_idx,
        tl.where(valid, boundary_recovery_len, 0),
    )


@triton.jit
def _copy_conv_state_tile(
    source_ptr,
    final_ptr,
    boundary_ptr,
    channel_offsets,
    history_offsets,
    token_stride,
    commit_len,
    boundary_recovery_len,
    mask,
    boundary_valid,
    ALIGN_MODE: tl.constexpr,
):
    final_values = tl.load(
        source_ptr + channel_offsets + (commit_len - 1 + history_offsets) * token_stride,
        mask=mask,
    )
    if ALIGN_MODE:
        boundary_values = tl.load(
            source_ptr + channel_offsets + (boundary_recovery_len - 1 + history_offsets) * token_stride,
            mask=mask & boundary_valid,
        )
        tl.store(
            boundary_ptr + channel_offsets + history_offsets * token_stride,
            boundary_values,
            mask=mask & boundary_valid,
        )
    tl.store(
        final_ptr + channel_offsets + history_offsets * token_stride,
        final_values,
        mask=mask,
    )


@triton.jit
def _compact_conv_state_kernel(
    conv_state_ref_ptr,
    conv_state_base_addrs_ptr,
    conv_state_block_strides_ptr,
    conv_state_dim_strides_ptr,
    conv_state_token_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    conv_dim,
    conv_history_len,
    stride_state_indices,
    BLOCK_D: tl.constexpr,
    BLOCK_HISTORY: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    DIM_STRIDE: tl.constexpr = None,
    USE_1D_CHANNEL_GATHER: tl.constexpr = False,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_l = tl.program_id(2)
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    if source_state_idx <= null_block_id:
        return

    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    base_addr = tl.load(conv_state_base_addrs_ptr + pid_l)
    block_stride = tl.load(conv_state_block_strides_ptr + pid_l)
    if DIM_STRIDE is None:
        dim_stride = tl.load(conv_state_dim_strides_ptr + pid_l)
    else:
        dim_stride = DIM_STRIDE
    token_stride = tl.load(conv_state_token_strides_ptr + pid_l)
    conv_state_ptr = base_addr.to(tl.pointer_type(conv_state_ref_ptr.dtype.element_ty))
    source_ptr = conv_state_ptr + source_state_idx * block_stride
    final_ptr = conv_state_ptr + final_state_idx * block_stride

    boundary_ptr = None
    if ALIGN_MODE:
        boundary_ptr = conv_state_ptr + boundary_state_idx * block_stride

    if USE_1D_CHANNEL_GATHER and (DIM_STRIDE is not None and DIM_STRIDE != 1):
        # A3 Triton 3.2 requires one-dimensional gathers for strided channels.
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        for history in tl.static_range(BLOCK_HISTORY):
            mask = (offs_d < conv_dim) & (history < conv_history_len)
            _copy_conv_state_tile(
                source_ptr,
                final_ptr,
                boundary_ptr,
                offs_d * dim_stride,
                history,
                token_stride,
                commit_len,
                boundary_recovery_len,
                mask,
                boundary_state_idx > null_block_id,
                ALIGN_MODE,
            )
        return
    offs_d = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D))[None, :]
    offs_h = tl.arange(0, BLOCK_HISTORY)[:, None]
    mask = (offs_d < conv_dim) & (offs_h < conv_history_len)
    _copy_conv_state_tile(
        source_ptr,
        final_ptr,
        boundary_ptr,
        offs_d * dim_stride,
        offs_h,
        token_stride,
        commit_len,
        boundary_recovery_len,
        mask,
        boundary_state_idx > null_block_id,
        ALIGN_MODE,
    )


@triton.jit
def _accumulate_kda_commit_step(
    correction_cache_ptr,
    kg_cache_ptr,
    token_offset,
    offs_k,
    offs_v,
    mask_k,
    mask_v,
    stride_correction_cache_pos,
    stride_correction_cache_dim,
    stride_kg_cache_pos,
    stride_kg_cache_dim,
    A,
    dt_bias,
    lower_bound,
    boundary_recovery_len,
    final_decay,
    final_correction,
    boundary_decay,
    boundary_correction,
    K: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    correction_ptr = correction_cache_ptr + token_offset * stride_correction_cache_pos
    kg_ptr = kg_cache_ptr + token_offset * stride_kg_cache_pos
    k = tl.load(
        kg_ptr + offs_k * stride_kg_cache_dim,
        mask=mask_k,
        other=0.0 if mask_k is not None else None,
    ).to(tl.float32)
    correction = tl.load(
        correction_ptr + offs_v * stride_correction_cache_dim,
        mask=mask_v,
        other=0.0 if mask_v is not None else None,
    ).to(tl.float32)
    raw_g = tl.load(
        kg_ptr + (K + offs_k) * stride_kg_cache_dim,
        mask=mask_k,
        other=0.0 if mask_k is not None else None,
    ).to(tl.float32)
    normalized_k = _kda_l2_normalize(k)
    gate = _kda_gate(
        raw_g,
        dt_bias,
        A,
        lower_bound,
        USE_LOWER_BOUND,
    )
    tl.static_assert(gate.dtype == tl.float32)
    decay = tl.exp(gate)
    weighted_k = normalized_k * final_decay
    final_correction = tl.fma(correction[:, None], weighted_k[None, :], final_correction)
    final_decay *= decay
    if ALIGN_MODE:
        before_boundary = token_offset < boundary_recovery_len
        boundary_k = normalized_k * boundary_decay
        boundary_correction = tl.where(
            before_boundary,
            tl.fma(correction[:, None], boundary_k[None, :], boundary_correction),
            boundary_correction,
        )
        boundary_decay *= tl.where(before_boundary, decay, 1.0)
    return final_decay, final_correction, boundary_decay, boundary_correction


@triton.jit
def _commit_kda_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    correction_cache_ref_ptr,
    correction_cache_base_addrs_ptr,
    correction_cache_block_strides_ptr,
    kg_cache_ref_ptr,
    kg_cache_base_addrs_ptr,
    kg_cache_block_strides_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    lower_bound,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_cache_head,
    stride_correction_cache_pos,
    stride_correction_cache_dim,
    stride_kg_cache_head,
    stride_kg_cache_pos,
    stride_kg_cache_dim,
    stride_A_layer,
    stride_A_head,
    stride_dt_bias_layer,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    UNROLL_COMMIT: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    # Walk accepted records backwards, composing decay-weighted rank-one
    # updates before combining them with the unchanged initial checkpoint.
    pid_lh = tl.program_id(2)
    pid_l = pid_lh // NUM_HEADS
    pid_h = pid_lh % NUM_HEADS

    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    if source_state_idx <= null_block_id:
        return
    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    state_base_addr = tl.load(state_base_addrs_ptr + pid_l)
    state_block_stride = tl.load(state_block_strides_ptr + pid_l)
    state_ptr = state_base_addr.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    source_state_ptr = state_ptr + source_state_idx * state_block_stride + pid_h * stride_state_head

    correction_cache_base_addr = tl.load(correction_cache_base_addrs_ptr + pid_l)
    correction_cache_block_stride = tl.load(correction_cache_block_strides_ptr + pid_l)
    correction_cache_ptr = correction_cache_base_addr.to(tl.pointer_type(correction_cache_ref_ptr.dtype.element_ty))
    correction_cache_ptr += source_state_idx * correction_cache_block_stride + pid_h * stride_correction_cache_head
    kg_cache_base_addr = tl.load(kg_cache_base_addrs_ptr + pid_l)
    kg_cache_block_stride = tl.load(kg_cache_block_strides_ptr + pid_l)
    kg_cache_ptr = kg_cache_base_addr.to(tl.pointer_type(kg_cache_ref_ptr.dtype.element_ty))
    kg_cache_ptr += source_state_idx * kg_cache_block_stride + pid_h * stride_kg_cache_head

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = None if K == BK else offs_k < K
    mask_v = None if V % BV == 0 else offs_v < V
    if K == BK and V % BV == 0:
        mask_state = None
    elif K == BK:
        mask_state = tl.broadcast_to((offs_v < V)[:, None], (BV, BK))
    elif V % BV == 0:
        mask_state = tl.broadcast_to((offs_k < K)[None, :], (BV, BK))
    else:
        mask_state = (offs_v < V)[:, None] & (offs_k < K)[None, :]
    state_ptrs = source_state_ptr + offs_v[:, None] * stride_state_v + offs_k[None, :] * stride_state_k
    a_log = tl.load(A_log_ptr + pid_l * stride_A_layer + pid_h * stride_A_head).to(tl.float32)
    tl.static_assert(a_log.dtype == tl.float32)
    A = tl.exp(a_log)

    dt_bias = tl.load(
        dt_bias_ptr + pid_l * stride_dt_bias_layer + pid_h * stride_dt_bias_head + offs_k * stride_dt_bias_dim,
        mask=mask_k,
        other=0.0 if mask_k is not None else None,
    ).to(tl.float32)
    final_decay = tl.full([BK], 1.0, tl.float32)
    final_correction = tl.zeros([BV, BK], tl.float32)
    boundary_decay = tl.full([BK], 1.0, tl.float32)
    boundary_correction = tl.zeros([BV, BK], tl.float32)

    # A3 runtime iteration bounds UB use; A5 retains its faster static expansion.
    if UNROLL_COMMIT:
        for reverse_offset in tl.static_range(SPEC_QUERY_LEN):
            if reverse_offset < commit_len:
                final_decay, final_correction, boundary_decay, boundary_correction = _accumulate_kda_commit_step(
                    correction_cache_ptr,
                    kg_cache_ptr,
                    commit_len - reverse_offset - 1,
                    offs_k,
                    offs_v,
                    mask_k,
                    mask_v,
                    stride_correction_cache_pos,
                    stride_correction_cache_dim,
                    stride_kg_cache_pos,
                    stride_kg_cache_dim,
                    A,
                    dt_bias,
                    lower_bound,
                    boundary_recovery_len,
                    final_decay,
                    final_correction,
                    boundary_decay,
                    boundary_correction,
                    K,
                    USE_LOWER_BOUND,
                    ALIGN_MODE,
                )
    else:
        for reverse_offset in range(tl.minimum(commit_len, SPEC_QUERY_LEN)):
            if reverse_offset < commit_len:
                final_decay, final_correction, boundary_decay, boundary_correction = _accumulate_kda_commit_step(
                    correction_cache_ptr,
                    kg_cache_ptr,
                    commit_len - reverse_offset - 1,
                    offs_k,
                    offs_v,
                    mask_k,
                    mask_v,
                    stride_correction_cache_pos,
                    stride_correction_cache_dim,
                    stride_kg_cache_pos,
                    stride_kg_cache_dim,
                    A,
                    dt_bias,
                    lower_bound,
                    boundary_recovery_len,
                    final_decay,
                    final_correction,
                    boundary_decay,
                    boundary_correction,
                    K,
                    USE_LOWER_BOUND,
                    ALIGN_MODE,
                )
    initial_state = tl.load(state_ptrs, mask=mask_state, other=0.0 if mask_state is not None else None).to(tl.float32)
    state = initial_state * final_decay[None, :] + final_correction
    if ALIGN_MODE:
        boundary_ptrs = (
            state_ptr
            + boundary_state_idx * state_block_stride
            + pid_h * stride_state_head
            + offs_v[:, None] * stride_state_v
            + offs_k[None, :] * stride_state_k
        )
        boundary_state = initial_state * boundary_decay[None, :] + boundary_correction
        tl.store(
            boundary_ptrs,
            boundary_state,
            mask=(boundary_state_idx > null_block_id)
            if mask_state is None
            else mask_state & (boundary_state_idx > null_block_id),
        )

    final_ptrs = (
        state_ptr
        + final_state_idx * state_block_stride
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    tl.store(final_ptrs, state, mask=mask_state)


@triton.jit
def _kda_l2_normalize_heads(vector):
    norm_square = tl.sum(vector * vector, axis=1)
    norm_input = tl.where(norm_square > 0.0, norm_square, 1.0)
    tl.static_assert(norm_input.dtype == tl.float32)
    inverse = tl.rsqrt(norm_input)
    return tl.where(norm_square[:, None] > 0.0, vector * inverse[:, None], 0.0)


@triton.jit
def _commit_kda_state_grouped_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    correction_cache_ref_ptr,
    correction_cache_base_addrs_ptr,
    correction_cache_block_strides_ptr,
    kg_cache_ref_ptr,
    kg_cache_base_addrs_ptr,
    kg_cache_block_strides_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    lower_bound,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_cache_head,
    stride_correction_cache_pos,
    stride_correction_cache_dim,
    stride_kg_cache_head,
    stride_kg_cache_pos,
    stride_kg_cache_dim,
    stride_A_layer,
    stride_A_head,
    stride_dt_bias_layer,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_state_indices,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    BH: tl.constexpr = 2,
):
    # Commit two independent heads per program with BV64, matching the
    # state-element count of a single-head BV128 tile.
    tl.static_assert(not ALIGN_MODE)
    tl.static_assert(K == 128)
    tl.static_assert(V == 128)
    tl.static_assert(BK == 128)
    tl.static_assert(BV == 64)
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    tl.static_assert(NUM_HEADS % BH == 0)
    pid_l = pid_lh // (NUM_HEADS // BH)
    pid_h = pid_lh % (NUM_HEADS // BH) * BH + tl.arange(0, BH)
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    if source_state_idx <= null_block_id:
        return
    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    if final_state_idx <= null_block_id:
        return
    state_base_addr = tl.load(state_base_addrs_ptr + pid_l)
    state_block_stride = tl.load(state_block_strides_ptr + pid_l)
    state_ptr = state_base_addr.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    source_state_ptr = state_ptr + source_state_idx * state_block_stride + pid_h * stride_state_head
    correction_cache_base_addr = tl.load(correction_cache_base_addrs_ptr + pid_l)
    correction_cache_block_stride = tl.load(correction_cache_block_strides_ptr + pid_l)
    correction_cache_ptr = correction_cache_base_addr.to(tl.pointer_type(correction_cache_ref_ptr.dtype.element_ty))
    correction_cache_ptr += source_state_idx * correction_cache_block_stride + pid_h * stride_correction_cache_head
    kg_cache_base_addr = tl.load(kg_cache_base_addrs_ptr + pid_l)
    kg_cache_block_stride = tl.load(kg_cache_block_strides_ptr + pid_l)
    kg_cache_ptr = kg_cache_base_addr.to(tl.pointer_type(kg_cache_ref_ptr.dtype.element_ty))
    kg_cache_ptr += source_state_idx * kg_cache_block_stride + pid_h * stride_kg_cache_head
    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = None if K == BK else offs_k < K
    mask_v = None if V % BV == 0 else offs_v < V
    if K == BK and V % BV == 0:
        mask_state = None
    elif K == BK:
        mask_state = tl.broadcast_to((offs_v < V)[:, None], (BV, BK))
    elif V % BV == 0:
        mask_state = tl.broadcast_to((offs_k < K)[None, :], (BV, BK))
    else:
        mask_state = (offs_v < V)[:, None] & (offs_k < K)[None, :]
    if mask_state is not None:
        mask_state = tl.broadcast_to(mask_state[None, :, :], (BH, BV, BK))
    if mask_k is not None:
        mask_k = tl.broadcast_to(mask_k[None, :], (BH, BK))
    if mask_v is not None:
        mask_v = tl.broadcast_to(mask_v[None, :], (BH, BV))
    state_ptrs = (
        source_state_ptr[:, None, None]
        + offs_v[None, :, None] * stride_state_v
        + offs_k[None, None, :] * stride_state_k
    )
    a_log = tl.load(A_log_ptr + pid_l * stride_A_layer + pid_h * stride_A_head).to(tl.float32)
    tl.static_assert(a_log.dtype == tl.float32)
    A = tl.exp(a_log)
    dt_bias = tl.load(
        dt_bias_ptr
        + pid_l * stride_dt_bias_layer
        + pid_h[:, None] * stride_dt_bias_head
        + offs_k[None, :] * stride_dt_bias_dim,
        mask=mask_k,
        other=0.0 if mask_k is not None else None,
    ).to(tl.float32)
    final_decay = tl.full([BH, BK], 1.0, tl.float32)
    final_correction = tl.zeros([BH, BV, BK], tl.float32)
    for reverse_offset in tl.static_range(SPEC_QUERY_LEN):
        if reverse_offset < commit_len:
            token_offset = commit_len - reverse_offset - 1
            correction_ptr = correction_cache_ptr + token_offset * stride_correction_cache_pos
            kg_ptr = kg_cache_ptr + token_offset * stride_kg_cache_pos
            k = tl.load(
                kg_ptr[:, None] + offs_k[None, :] * stride_kg_cache_dim,
                mask=mask_k,
                other=0.0 if mask_k is not None else None,
            ).to(tl.float32)
            correction = tl.load(
                correction_ptr[:, None] + offs_v[None, :] * stride_correction_cache_dim,
                mask=mask_v,
                other=0.0 if mask_v is not None else None,
            ).to(tl.float32)
            raw_g = tl.load(
                kg_ptr[:, None] + (K + offs_k[None, :]) * stride_kg_cache_dim,
                mask=mask_k,
                other=0.0 if mask_k is not None else None,
            ).to(tl.float32)
            normalized_k = _kda_l2_normalize_heads(k)
            gate = _kda_gate(raw_g, dt_bias, A[:, None], lower_bound, USE_LOWER_BOUND)
            tl.static_assert(gate.dtype == tl.float32)
            decay = tl.exp(gate)
            weighted_k = normalized_k * final_decay
            final_correction = tl.fma(correction[:, :, None], weighted_k[:, None, :], final_correction)
            final_decay *= decay
    initial_state = tl.load(state_ptrs, mask=mask_state, other=0.0 if mask_state is not None else None).to(tl.float32)
    state = initial_state * final_decay[:, None, :] + final_correction
    final_ptrs = (
        state_ptr
        + final_state_idx * state_block_stride
        + pid_h[:, None, None] * stride_state_head
        + offs_v[None, :, None] * stride_state_v
        + offs_k[None, None, :] * stride_state_k
    )
    tl.store(final_ptrs, state, mask=mask_state)


def kda_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kg_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify a KDA speculative window without modifying its checkpoint."""
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("KDA RecoverSSM q must have shape [1, tokens, heads, dim]")
    _, total_tokens, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if k.shape != q.shape or v.shape != (1, total_tokens, num_heads, value_dim):
        raise ValueError("KDA RecoverSSM q, k, and v shapes are incompatible")
    if raw_g.shape != q.shape or raw_beta.shape != (1, total_tokens, num_heads):
        raise ValueError("KDA RecoverSSM gate or beta shape is incompatible")
    if any(tensor.stride()[2:] != (key_dim, 1) for tensor in (q, k, raw_g)):
        raise ValueError("KDA RecoverSSM q, k, and gate heads must be contiguous")
    if v.stride()[2:] != (value_dim, 1) or raw_beta.stride(2) != 1:
        raise ValueError("KDA RecoverSSM v and beta heads must be contiguous")
    num_blocks = checkpoint_state.shape[0]
    if checkpoint_state.shape[1:] != (
        num_heads,
        value_dim,
        key_dim,
    ):
        raise ValueError("KDA RecoverSSM checkpoint shape is incompatible")
    expected_correction_shape = (
        num_blocks,
        num_heads,
        spec_query_len,
        value_dim,
    )
    if correction_cache.shape != expected_correction_shape:
        raise ValueError(f"KDA RecoverSSM correction buffer needs shape {expected_correction_shape}")
    expected_kg_shape = (num_blocks, num_heads, spec_query_len, 2 * key_dim)
    if kg_cache.shape != expected_kg_shape:
        raise ValueError(f"KDA RecoverSSM key/gate buffer needs shape {expected_kg_shape}")
    if correction_cache.dtype != torch.float32:
        raise ValueError("KDA RecoverSSM correction buffer must use float32")
    if kg_cache.dtype != k.dtype:
        raise ValueError("KDA RecoverSSM key/gate buffer must match activation dtype")
    if A_log.shape != (num_heads,) or dt_bias.numel() != num_heads * key_dim:
        raise ValueError("KDA RecoverSSM gate parameters are incompatible")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("KDA RecoverSSM gate parameters must be contiguous")
    batch = state_indices.shape[0]
    if query_start_loc.shape[0] != batch + 1:
        raise ValueError("KDA RecoverSSM query metadata is incompatible")
    if total_tokens > batch * spec_query_len:
        raise ValueError("KDA RecoverSSM speculative decode input exceeds its activation capacity")
    if out is None:
        out = torch.empty_like(v)
    if out.shape != v.shape:
        raise ValueError("KDA RecoverSSM output shape is incompatible")
    if out.stride()[2:] != (value_dim, 1):
        raise ValueError("KDA RecoverSSM output heads must be contiguous")
    device = q.device
    if any(
        tensor.device != device
        for tensor in (
            k,
            v,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            checkpoint_state,
            correction_cache,
            kg_cache,
            query_start_loc,
            state_indices,
            out,
        )
    ):
        raise ValueError("KDA RecoverSSM inputs must be on the same device")
    if total_tokens == 0:
        return out

    block_k = triton.next_power_of_2(key_dim)
    # Grid specialization: large tiles halve redundant K-side loads and win once
    # enough programs keep the NPU busy; below that, smaller tiles keep more
    # programs in flight. Measured crossover: 96 programs (batch*heads*V tiles).
    is_a5 = is_950()
    block_v = min(triton.next_power_of_2(value_dim), RECOVERSSM_VALUE_TILE)
    if is_a5 and block_v < RECOVERSSM_VALUE_TILE_LARGE and batch * num_heads >= RECOVERSSM_LARGE_TILE_MIN_PROGRAMS:
        large_v = min(triton.next_power_of_2(value_dim), RECOVERSSM_VALUE_TILE_LARGE)
        if batch * num_heads * triton.cdiv(value_dim, large_v) >= RECOVERSSM_LARGE_TILE_MIN_PROGRAMS:
            block_v = large_v
    grid = (triton.cdiv(value_dim, block_v), num_heads, batch)
    _kda_recoverssm_verify_kernel[grid](
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        checkpoint_state,
        correction_cache,
        kg_cache,
        out,
        query_start_loc,
        state_indices,
        lower_bound or 0.0,
        NULL_BLOCK_ID,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        raw_g.stride(1),
        raw_beta.stride(1),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        correction_cache.stride(0),
        correction_cache.stride(1),
        correction_cache.stride(2),
        correction_cache.stride(3),
        kg_cache.stride(0),
        kg_cache.stride(1),
        kg_cache.stride(2),
        kg_cache.stride(3),
        out.stride(1),
        query_start_loc.stride(0),
        state_indices.stride(0),
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        USE_LOWER_BOUND=lower_bound is not None,
        num_warps=4,
        num_stages=2,
    )
    return out


def _conv_compaction_block_d(dim_stride):
    # A3 strided gathers need smaller tiles to fit the 192 KiB UB.
    return RECOVERSSM_CONV_TILE if is_950() or dim_stride == 1 else RECOVERSSM_CONV_STRIDED_TILE


@dataclass
class KDARecoverSSMCommitContext:
    conv_states: tuple[torch.Tensor, ...]
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    checkpoints: tuple[torch.Tensor, ...]
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    correction_caches: tuple[torch.Tensor, ...]
    correction_cache_base_addrs: torch.Tensor
    correction_cache_block_strides: torch.Tensor
    kg_caches: tuple[torch.Tensor, ...]
    kg_cache_base_addrs: torch.Tensor
    kg_cache_block_strides: torch.Tensor
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    lower_bound: float | None
    spec_query_len: int
    conv_dim_stride: int | None

    @classmethod
    def create(
        cls,
        layers: Sequence[Any],
        *,
        spec_query_len: int,
        max_num_reqs: int,
        backend: Literal["cute", "triton"] = "triton",
    ) -> "KDARecoverSSMCommitContext":
        """Bind layer-owned page views and allocate reusable commit-plan buffers.

        ``backend`` is accepted for signature compatibility with the upstream
        factory on vLLM main, which selects between its CuTe and Triton commit
        kernels. Ascend only implements the Triton plan, so the value is
        ignored.
        """
        del backend
        if not layers:
            raise ValueError("KDA RecoverSSM commit requires at least one layer")
        if any(len(layer.kv_cache) != 4 for layer in layers):
            raise ValueError("KDA RecoverSSM pages must contain conv, state, correction, and key/gate")
        init_device_properties_triton()

        conv_states = [layer.kv_cache[0] for layer in layers]
        if not is_conv_state_dim_first():
            conv_states = [state.transpose(-1, -2) for state in conv_states]
        checkpoints = [layer.kv_cache[1] for layer in layers]
        correction_caches = [layer.kv_cache[2] for layer in layers]
        kg_caches = [layer.kv_cache[3] for layer in layers]
        A_log = [layer.A_log.reshape(-1) for layer in layers]
        dt_bias = [layer.dt_bias.view(layer.local_num_heads, layer.head_dim) for layer in layers]
        lower_bounds = {layer.gate_lower_bound for layer in layers}
        if len(lower_bounds) != 1:
            raise ValueError("KDA RecoverSSM layers need matching gate bounds")

        state_ref = checkpoints[0]
        if state_ref.ndim != 4:
            raise ValueError("KDA RecoverSSM checkpoint must be four-dimensional")
        num_blocks, num_heads, value_dim, key_dim = state_ref.shape
        for state in checkpoints:
            if (
                state.shape != state_ref.shape
                or state.dtype != state_ref.dtype
                or state.device != state_ref.device
                or state.stride()[1:] != state_ref.stride()[1:]
            ):
                raise ValueError("KDA RecoverSSM layers need matching checkpoint layout")
        expected_correction_shape = (
            num_blocks,
            num_heads,
            spec_query_len,
            value_dim,
        )
        correction_ref = correction_caches[0]
        for correction_cache in correction_caches:
            if (
                correction_cache.shape != expected_correction_shape
                or correction_cache.dtype != torch.float32
                or correction_cache.device != state_ref.device
                or correction_cache.stride()[1:] != correction_ref.stride()[1:]
            ):
                raise ValueError(f"KDA RecoverSSM correction buffers need float32 shape {expected_correction_shape}")
        expected_kg_shape = (num_blocks, num_heads, spec_query_len, 2 * key_dim)
        kg_ref = kg_caches[0]
        for kg_cache in kg_caches:
            if (
                kg_cache.shape != expected_kg_shape
                or kg_cache.dtype != kg_ref.dtype
                or kg_cache.device != state_ref.device
                or kg_cache.stride()[1:] != kg_ref.stride()[1:]
            ):
                raise ValueError(f"KDA RecoverSSM key/gate buffers need shape {expected_kg_shape}")
        if any(param.shape != (num_heads,) for param in A_log):
            raise ValueError("KDA RecoverSSM A_log shape is incompatible")
        if any(param.shape != (num_heads, key_dim) for param in dt_bias):
            raise ValueError("KDA RecoverSSM dt_bias shape is incompatible")

        conv_ref = conv_states[0]
        if conv_ref.ndim != 3:
            raise ValueError("KDA RecoverSSM conv state must be three-dimensional")
        conv_dim, conv_state_len = conv_ref.shape[1:]
        conv_history_len = conv_state_len - spec_query_len + 1
        if conv_history_len <= 0:
            raise ValueError("KDA RecoverSSM conv state is shorter than its window")
        for conv_state in conv_states:
            if (
                conv_state.shape != conv_ref.shape
                or conv_state.dtype != conv_ref.dtype
                or conv_state.device != state_ref.device
                or conv_state.shape[0] != num_blocks
            ):
                raise ValueError("KDA RecoverSSM layers need matching conv state")

        device = state_ref.device

        def _base_addrs(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [tensor.data_ptr() for tensor in tensors],
                dtype=torch.int64,
                device=device,
            )

        def _block_strides(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [tensor.stride(0) for tensor in tensors],
                dtype=torch.int64,
                device=device,
            )

        return cls(
            conv_states=tuple(conv_states),
            conv_state_base_addrs=_base_addrs(conv_states),
            conv_state_block_strides=_block_strides(conv_states),
            conv_state_dim_strides=torch.tensor(
                [state.stride(1) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_state_token_strides=torch.tensor(
                [state.stride(2) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_history_len=conv_history_len,
            checkpoints=tuple(checkpoints),
            state_base_addrs=_base_addrs(checkpoints),
            state_block_strides=_block_strides(checkpoints),
            correction_caches=tuple(correction_caches),
            correction_cache_base_addrs=_base_addrs(correction_caches),
            correction_cache_block_strides=_block_strides(correction_caches),
            kg_caches=tuple(kg_caches),
            kg_cache_base_addrs=_base_addrs(kg_caches),
            kg_cache_block_strides=_block_strides(kg_caches),
            commit_lens=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            final_state_indices=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            boundary_state_indices=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            boundary_recovery_lens=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            A_log=torch.stack(tuple(A_log)).contiguous(),
            dt_bias=torch.stack(tuple(dt_bias)).contiguous(),
            lower_bound=lower_bounds.pop(),
            spec_query_len=spec_query_len,
            conv_dim_stride=(
                conv_ref.stride(1) if all(state.stride(1) == conv_ref.stride(1) for state in conv_states) else None
            ),
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
    ) -> None:
        """Fold accepted KDA and convolution inputs into every layer."""
        batch = state_indices.shape[0]
        if batch == 0:
            return
        if batch > self.commit_lens.shape[0]:
            raise ValueError("KDA RecoverSSM commit batch exceeds its plan capacity")
        if query_start_loc.shape[0] != batch + 1:
            raise ValueError("KDA RecoverSSM commit metadata is incompatible")
        if request_indices is not None and request_indices.shape[0] < batch:
            raise ValueError("KDA RecoverSSM request mapping is too short")
        align_args = (block_table, num_computed_tokens, mamba_block_size)
        if any(arg is not None for arg in align_args) and any(arg is None for arg in align_args):
            raise ValueError("KDA RecoverSSM align metadata is incomplete")
        if mamba_block_size is not None and mamba_block_size < self.spec_query_len:
            raise ValueError("KDA RecoverSSM align block size must cover one speculative window")
        if block_table is not None and block_table.ndim != 2:
            raise ValueError("KDA RecoverSSM block table must be two-dimensional")
        device = self.checkpoints[0].device
        if (
            any(
                tensor.device != device
                for tensor in (
                    num_accepted_tokens,
                    state_indices,
                    query_start_loc,
                )
            )
            or (request_indices is not None and request_indices.device != device)
            or (block_table is not None and block_table.device != device)
            or (num_computed_tokens is not None and num_computed_tokens.device != device)
        ):
            raise ValueError("KDA RecoverSSM commit inputs must be on the same device")

        block_table_stride = (0, 0) if block_table is None else block_table.stride()
        num_computed_stride = 0 if num_computed_tokens is None else num_computed_tokens.stride(0)

        num_layers = len(self.checkpoints)
        conv_ref = self.conv_states[0]
        conv_dim = conv_ref.shape[1]
        block_history = triton.next_power_of_2(self.conv_history_len)
        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            mamba_block_size or 1,
            block_table.shape[1] if block_table is not None else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            block_table_stride[0],
            block_table_stride[1],
            num_computed_stride,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )
        conv_block_d = _conv_compaction_block_d(self.conv_dim_stride)
        _compact_conv_state_kernel[(triton.cdiv(conv_dim, conv_block_d), batch, num_layers)](
            conv_ref,
            self.conv_state_base_addrs,
            self.conv_state_block_strides,
            self.conv_state_dim_strides,
            self.conv_state_token_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            conv_dim,
            self.conv_history_len,
            state_indices.stride(0),
            BLOCK_D=conv_block_d,
            BLOCK_HISTORY=block_history,
            ALIGN_MODE=block_table is not None,
            DIM_STRIDE=self.conv_dim_stride,
            USE_1D_CHANNEL_GATHER=not is_950(),
            num_warps=4,
        )

        state_ref = self.checkpoints[0]
        _, num_heads, value_dim, key_dim = state_ref.shape
        block_k = triton.next_power_of_2(key_dim)
        is_a5 = is_950()
        # Group two adjacent heads in one program to reconstruct and commit their states.
        grouped_commit = (
            is_a5
            and block_table is None
            and state_ref.dtype == torch.float32
            and key_dim == value_dim == 128
            and num_heads in (32, 64)
        )
        value_tile = RECOVERSSM_VALUE_TILE
        if grouped_commit:
            head_groups = num_heads // RECOVERSSM_COMMIT_HEAD_TILE
            commit_kernel = _commit_kda_state_grouped_kernel
            commit_options = {}
        else:
            if is_a5 and block_table is None:
                value_tile *= 2
            # Runtime iteration keeps accumulator/merge matrices independent of window length.
            matrix_buffers = 6 if block_table is not None else 2
            bytes_per_row = matrix_buffers * block_k * torch.float32.itemsize
            # Reserve two rows' worth for FP32 vector temporaries.
            max_rows = max(1, get_ub_size_bytes() // bytes_per_row - 2)
            value_tile = min(value_tile, 1 << (max_rows.bit_length() - 1))
            head_groups = num_heads
            commit_kernel = _commit_kda_state_kernel
            commit_options = {"UNROLL_COMMIT": is_a5}
        block_v = min(triton.next_power_of_2(value_dim), value_tile)
        grid = (triton.cdiv(value_dim, block_v), batch, num_layers * head_groups)
        commit_kernel[grid](
            state_ref,
            self.state_base_addrs,
            self.state_block_strides,
            self.correction_caches[0],
            self.correction_cache_base_addrs,
            self.correction_cache_block_strides,
            self.kg_caches[0],
            self.kg_cache_base_addrs,
            self.kg_cache_block_strides,
            self.A_log,
            self.dt_bias,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            self.lower_bound or 0.0,
            NULL_BLOCK_ID,
            state_ref.stride(1),
            state_ref.stride(2),
            state_ref.stride(3),
            self.correction_caches[0].stride(1),
            self.correction_caches[0].stride(2),
            self.correction_caches[0].stride(3),
            self.kg_caches[0].stride(1),
            self.kg_caches[0].stride(2),
            self.kg_caches[0].stride(3),
            self.A_log.stride(0),
            self.A_log.stride(1),
            self.dt_bias.stride(0),
            self.dt_bias.stride(1),
            self.dt_bias.stride(2),
            state_indices.stride(0),
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            NUM_HEADS=num_heads,
            USE_LOWER_BOUND=self.lower_bound is not None,
            ALIGN_MODE=block_table is not None,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=4,
            num_stages=2,
            **commit_options,
        )


__all__ = ["KDARecoverSSMCommitContext", "kda_recoverssm_verify"]
