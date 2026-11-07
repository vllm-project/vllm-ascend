# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged GDN checkpoints and accepted-prefix recovery on Ascend.

Q/K are normalized by the caller, matching the existing recurrent path.
Verification never overwrites the checkpoint. All records use the effective
post-normalization key and post-activation gates seen by verification.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.triton.mamba.recoverssm import (
    compact_conv_state_kernel as _compact_conv_state_kernel,
)
from vllm_ascend.ops.triton.mamba.recoverssm import (
    prepare_commit_plan_kernel as _prepare_commit_plan_kernel,
)
from vllm_ascend.ops.triton.triton_utils import get_ub_size_bytes, init_device_properties_triton

GDN_PAGE_COUNT: tl.constexpr = 3  # conv, checkpoint, packed FP32 replay
RECOVERSSM_COMMIT_HEAD_TILE = 2
# Conservative UB budgets in BV x BK FP32 matrix equivalents. The GDN
# align path commits final and boundary states in one kernel, so it uses the
# larger boundary budget for the whole launch.
RECOVERSSM_COMMIT_UB_MATRICES = 2
RECOVERSSM_BOUNDARY_COMMIT_UB_MATRICES = 6
# Reserve tile rows for FP32 vectors and other temporary workspace.
RECOVERSSM_COMMIT_UB_HEADROOM_ROWS = 2
RECOVERSSM_VALUE_TILE = 64
RECOVERSSM_VALUE_TILE_LARGE = 128
RECOVERSSM_LARGE_TILE_MIN_PROGRAMS = 96
RECOVERSSM_CONV_TILE = 4096
RECOVERSSM_CONV_STRIDED_TILE = 256


@triton.jit
def _load_gdn_commit_record(
    record_ptr,
    offs_k,
    offs_v,
    mask_k,
    mask_v,
    stride_replay_dim,
    K: tl.constexpr,
    V: tl.constexpr,
):
    # record_ptr already selects the layer, block, value head and token.
    normalized_k = tl.load(
        record_ptr + (V + offs_k) * stride_replay_dim,
        mask=mask_k,
        other=0.0 if mask_k is not None else None,
    ).to(tl.float32)
    correction = tl.load(
        record_ptr + offs_v * stride_replay_dim,
        mask=mask_v,
        other=0.0 if mask_v is not None else None,
    ).to(tl.float32)
    decay = tl.load(record_ptr + (V + K) * stride_replay_dim)
    return correction, normalized_k, decay


@triton.jit
def _gdn_recoverssm_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    gate_ptr,
    beta_ptr,
    state_ptr,
    replay_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    out_ptr,
    null_block_id,
    stride_q_token: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_gate_token: tl.constexpr,
    stride_gate_head: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_beta_head: tl.constexpr,
    state_strides: tl.constexpr,
    replay_strides: tl.constexpr,
    stride_out_token: tl.constexpr,
    stride_out_head: tl.constexpr,
    stride_query_start_loc: tl.constexpr,
    stride_state_indices: tl.constexpr,
    NUM_KEY_HEADS: tl.constexpr,
    NUM_VALUE_HEADS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # Preserve physical page layouts, including padding between blocks or heads.
    # State: [block, head, value, key]; records: [block, head, pos, dim].
    stride_state_block: tl.constexpr = state_strides[0]
    stride_state_head: tl.constexpr = state_strides[1]
    stride_state_v: tl.constexpr = state_strides[2]
    stride_state_k: tl.constexpr = state_strides[3]
    stride_replay_block: tl.constexpr = replay_strides[0]
    stride_replay_head: tl.constexpr = replay_strides[1]
    stride_replay_position: tl.constexpr = replay_strides[2]
    stride_replay_dim: tl.constexpr = replay_strides[3]
    # Advance a register-held checkpoint and record each token's compact
    # correction/key/gate; persistent checkpoint pages remain read-only.
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(2)
    pid_h = tl.program_id(1)

    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(tl.int64)
    if bos == eos:
        return
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
            tl.store(out_ptr + (bos + token_offset) * stride_out_token + pid_h * stride_out_head + offs_v, 0.0, mask_v)
        return

    state_ptrs = (
        state_ptr
        + state_idx * stride_state_block
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    state = tl.load(state_ptrs, mask=mask_state, other=0.0 if mask_state is not None else None).to(tl.float32)

    key_head = pid_h // (NUM_VALUE_HEADS // NUM_KEY_HEADS)
    # token_offset < query_len always holds inside range(query_len); the
    # token_valid masks and the tl.where state select are redundant and dropped.
    for token_offset in range(query_len):
        token = bos + token_offset
        q = tl.load(
            q_ptr + token * stride_q_token + key_head * stride_q_head + offs_k,
            mask_k,
            0.0 if mask_k is not None else None,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + token * stride_k_token + key_head * stride_k_head + offs_k,
            mask_k,
            0.0 if mask_k is not None else None,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + token * stride_v_token + pid_h * stride_v_head + offs_v,
            mask_v,
            0.0 if mask_v is not None else None,
        ).to(tl.float32)
        beta = tl.load(beta_ptr + token * stride_beta_token + pid_h * stride_beta_head).to(tl.float32)

        # Store the same FP32 decay that is multiplied into the state.
        gdn_gate = tl.load(gate_ptr + token * stride_gate_token + pid_h * stride_gate_head).to(tl.float32)
        decay = tl.exp(gdn_gate)
        normalized_k = k

        decayed_state = state * decay
        correction = (v - tl.sum(decayed_state * normalized_k[None, :], axis=1)) * beta
        # GDN output consumes the updated state, including for the last token.
        state = decayed_state + correction[:, None] * normalized_k[None, :]
        out = tl.sum(state * q[None, :], axis=1) * (K**-0.5)
        tl.store(out_ptr + token * stride_out_token + pid_h * stride_out_head + offs_v, out, mask_v)

        record_ptr = (
            replay_ptr
            + state_idx * stride_replay_block
            + pid_h * stride_replay_head
            + token_offset * stride_replay_position
        )
        tl.store(record_ptr + offs_v * stride_replay_dim, correction, mask_v)
        if pid_v == 0:
            tl.store(record_ptr + (V + offs_k) * stride_replay_dim, k, mask_k)
            tl.store(record_ptr + (V + K) * stride_replay_dim, decay)


@triton.jit
def _accumulate_gdn_commit_step(
    replay_ptr,
    token_record_pos,
    offs_k,
    offs_v,
    mask_k,
    mask_v,
    stride_replay_pos,
    stride_replay_dim,
    boundary_recovery_len,
    final_decay,
    final_correction,
    boundary_decay,
    boundary_correction,
    ALIGN_MODE: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    correction, normalized_k, decay = _load_gdn_commit_record(
        record_ptr=replay_ptr + token_record_pos * stride_replay_pos,
        offs_k=offs_k,
        offs_v=offs_v,
        mask_k=mask_k,
        mask_v=mask_v,
        stride_replay_dim=stride_replay_dim,
        K=K,
        V=V,
    )
    weighted_k = normalized_k * final_decay
    final_correction = tl.fma(correction[:, None], weighted_k[None, :], final_correction)
    final_decay *= decay
    if ALIGN_MODE:
        before_boundary = token_record_pos < boundary_recovery_len
        boundary_k = normalized_k * boundary_decay
        boundary_correction = tl.where(
            before_boundary,
            tl.fma(correction[:, None], boundary_k[None, :], boundary_correction),
            boundary_correction,
        )
        boundary_decay *= tl.where(before_boundary, decay, 1.0)
    return final_decay, final_correction, boundary_decay, boundary_correction


@triton.jit(do_not_specialize=["num_layers"])
def _commit_gdn_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    replay_ref_ptr,
    replay_base_addrs_ptr,
    replay_block_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_replay_head,
    stride_replay_pos,
    stride_replay_dim,
    stride_state_indices,
    num_layers,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    UNROLL_COMMIT: tl.constexpr,
):
    # Explicit one-dimensional indexing gives the same request-major order
    # across Ascend Triton backends with different 3-D grid flattening orders.
    pid = tl.program_id(0)
    value_tiles = tl.cdiv(V, BV)
    pid_v = pid % value_tiles
    pid_lh = pid // value_tiles
    layer_head_tiles = num_layers * NUM_HEADS
    pid_b = pid_lh // layer_head_tiles
    pid_lh %= layer_head_tiles
    # Walk accepted records backwards, composing decay-weighted rank-one
    # updates before combining them with the unchanged initial checkpoint.
    pid_l = pid_lh // NUM_HEADS
    pid_h = pid_lh % NUM_HEADS

    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    if source_state_idx <= null_block_id:
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

    replay_base_addr = tl.load(replay_base_addrs_ptr + pid_l)
    replay_block_stride = tl.load(replay_block_strides_ptr + pid_l)
    replay_ptr = replay_base_addr.to(tl.pointer_type(replay_ref_ptr.dtype.element_ty))
    replay_ptr += source_state_idx * replay_block_stride + pid_h * stride_replay_head

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
    final_decay = tl.full([BK], 1.0, tl.float32)
    final_correction = tl.zeros([BV, BK], tl.float32)
    boundary_decay = tl.full([BK], 1.0, tl.float32)
    boundary_correction = tl.zeros([BV, BK], tl.float32)

    # A3 runtime iteration bounds UB use; A5 retains its faster static expansion.
    if UNROLL_COMMIT:
        for token_offset in tl.static_range(SPEC_QUERY_LEN):
            if token_offset < commit_len:
                final_decay, final_correction, boundary_decay, boundary_correction = _accumulate_gdn_commit_step(
                    replay_ptr,
                    commit_len - token_offset - 1,
                    offs_k,
                    offs_v,
                    mask_k,
                    mask_v,
                    stride_replay_pos,
                    stride_replay_dim,
                    boundary_recovery_len,
                    final_decay,
                    final_correction,
                    boundary_decay,
                    boundary_correction,
                    ALIGN_MODE,
                    K,
                    V,
                )
    else:
        for token_offset in range(tl.minimum(commit_len, SPEC_QUERY_LEN)):
            if token_offset < commit_len:
                final_decay, final_correction, boundary_decay, boundary_correction = _accumulate_gdn_commit_step(
                    replay_ptr,
                    commit_len - token_offset - 1,
                    offs_k,
                    offs_v,
                    mask_k,
                    mask_v,
                    stride_replay_pos,
                    stride_replay_dim,
                    boundary_recovery_len,
                    final_decay,
                    final_correction,
                    boundary_decay,
                    boundary_correction,
                    ALIGN_MODE,
                    K,
                    V,
                )
    initial_state = tl.load(state_ptrs, mask=mask_state, other=0.0 if mask_state is not None else None).to(tl.float32)
    state = initial_state * final_decay[None, :] + final_correction
    if ALIGN_MODE:
        boundary_ptrs = (
            state_ptr
            + tl.maximum(boundary_state_idx, 0) * state_block_stride
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


@triton.jit(do_not_specialize=["num_layers"])
def _commit_gdn_state_grouped_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    replay_ref_ptr,
    replay_base_addrs_ptr,
    replay_block_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_replay_head,
    stride_replay_pos,
    stride_replay_dim,
    stride_state_indices,
    num_layers,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    BH: tl.constexpr = 2,
):
    # Commit two independent heads per program.
    tl.static_assert(K == 128)
    tl.static_assert(V == 128)
    tl.static_assert(BK == 128)
    tl.static_assert(BV <= 64)
    pid = tl.program_id(0)
    value_tiles = tl.cdiv(V, BV)
    pid_v = pid % value_tiles
    pid_lh = pid // value_tiles
    layer_head_tiles = num_layers * (NUM_HEADS // BH)
    pid_b = pid_lh // layer_head_tiles
    pid_lh %= layer_head_tiles
    tl.static_assert(NUM_HEADS % BH == 0)
    pid_l = pid_lh // (NUM_HEADS // BH)
    pid_h = pid_lh % (NUM_HEADS // BH) * BH + tl.arange(0, BH)
    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    if source_state_idx <= null_block_id:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    if final_state_idx <= null_block_id:
        return
    if ALIGN_MODE:
        boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
        boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)
    state_base_addr = tl.load(state_base_addrs_ptr + pid_l)
    state_block_stride = tl.load(state_block_strides_ptr + pid_l)
    state_ptr = state_base_addr.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    source_state_ptr = state_ptr + source_state_idx * state_block_stride + pid_h * stride_state_head
    replay_base_addr = tl.load(replay_base_addrs_ptr + pid_l)
    replay_block_stride = tl.load(replay_block_strides_ptr + pid_l)
    replay_ptr = replay_base_addr.to(tl.pointer_type(replay_ref_ptr.dtype.element_ty))
    replay_ptr += source_state_idx * replay_block_stride + pid_h * stride_replay_head

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
    final_decay = tl.full([BH, BK], 1.0, tl.float32)
    final_correction = tl.zeros([BH, BV, BK], tl.float32)
    if ALIGN_MODE:
        boundary_decay = tl.full([BH, BK], 1.0, tl.float32)
        boundary_correction = tl.zeros([BH, BV, BK], tl.float32)
    for token_offset in tl.static_range(SPEC_QUERY_LEN):
        if token_offset < commit_len:
            # Reverse token access order from commit_len.
            token_record_pos = commit_len - token_offset - 1
            record_ptr = replay_ptr + token_record_pos * stride_replay_pos
            correction, normalized_k, decay = _load_gdn_commit_record(
                record_ptr=record_ptr[:, None],
                offs_k=offs_k[None, :],
                offs_v=offs_v[None, :],
                mask_k=mask_k,
                mask_v=mask_v,
                stride_replay_dim=stride_replay_dim,
                K=K,
                V=V,
            )
            weighted_k = normalized_k * final_decay
            final_correction = tl.fma(correction[:, :, None], weighted_k[:, None, :], final_correction)
            final_decay *= decay
            if ALIGN_MODE:
                before_boundary = token_record_pos < boundary_recovery_len
                boundary_k = normalized_k * boundary_decay
                boundary_correction = tl.where(
                    before_boundary,
                    tl.fma(correction[:, :, None], boundary_k[:, None, :], boundary_correction),
                    boundary_correction,
                )
                boundary_decay *= tl.where(before_boundary, decay, 1.0)
    initial_state = tl.load(state_ptrs, mask=mask_state, other=0.0 if mask_state is not None else None).to(tl.float32)
    state = initial_state * final_decay[:, None, :] + final_correction
    if ALIGN_MODE:
        if boundary_state_idx > null_block_id:
            boundary_ptrs = (
                state_ptr
                + boundary_state_idx * state_block_stride
                + pid_h[:, None, None] * stride_state_head
                + offs_v[None, :, None] * stride_state_v
                + offs_k[None, None, :] * stride_state_k
            )
            boundary_state = initial_state * boundary_decay[:, None, :] + boundary_correction
            tl.store(boundary_ptrs, boundary_state, mask=mask_state)
    final_ptrs = (
        state_ptr
        + final_state_idx * state_block_stride
        + pid_h[:, None, None] * stride_state_head
        + offs_v[None, :, None] * stride_state_v
        + offs_k[None, None, :] * stride_state_k
    )
    tl.store(final_ptrs, state, mask=mask_state)


def _inner_stride_signature(tensor):
    return tuple(stride if size > 1 else 0 for size, stride in zip(tensor.shape[1:], tensor.stride()[1:]))


def _validate_state_replay(checkpoint, replay, hv, dv, dk, spec_query_len):
    if checkpoint.ndim != 4 or checkpoint.shape[1:] != (hv, dv, dk):
        raise ValueError("GDN RecoverSSM checkpoint has incompatible shape")
    if checkpoint.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("GDN RecoverSSM checkpoint must be BF16 or FP32")
    if replay.shape != (checkpoint.shape[0], hv, spec_query_len, dv + dk + 1):
        raise ValueError("GDN RecoverSSM replay has incompatible shape")
    if replay.dtype != torch.float32:
        raise ValueError("GDN RecoverSSM replay must be FP32")
    if replay.device != checkpoint.device:
        raise ValueError("GDN RecoverSSM replay and checkpoint must share a device")


def gdn_recoverssm_verify(
    q,
    k,
    v,
    g,
    beta,
    checkpoint_state,
    replay_cache,
    query_start_loc,
    state_indices,
    spec_query_len,
):
    """Return GDN outputs and compact records, leaving the checkpoint intact."""
    if q.ndim != 4 or q.shape[0] != 1 or k.shape != q.shape:
        raise ValueError("GDN RecoverSSM Q/K must have matching [1,T,HK,K] shapes")
    _, tokens, hk, dk = q.shape
    if v.ndim != 4 or v.shape[:2] != (1, tokens):
        raise ValueError("GDN RecoverSSM V must have shape [1,T,HV,V]")
    hv, dv = v.shape[2:]
    if hk < 1 or hv % hk or spec_query_len < 1:
        raise ValueError("GDN RecoverSSM requires integral GQA and a positive window")
    if g.shape != (1, tokens, hv) or beta.shape != g.shape:
        raise ValueError("GDN RecoverSSM requires one gate and beta per value head")
    _validate_state_replay(checkpoint_state, replay_cache, hv, dv, dk, spec_query_len)
    tensors = (
        q,
        k,
        v,
        g,
        beta,
        checkpoint_state,
        replay_cache,
        query_start_loc,
        state_indices,
    )
    if any(t.device != q.device for t in tensors):
        raise ValueError("GDN RecoverSSM inputs must share a device")
    # The current rearrange/L2-norm path guarantees unit channel strides; guard other callers.
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("GDN RecoverSSM Q/K/V channels must be contiguous")
    rows = state_indices.numel()
    if state_indices.ndim != 1 or query_start_loc.shape != (rows + 1,):
        raise ValueError("GDN RecoverSSM request indices and offsets do not match")
    if tokens > rows * spec_query_len:
        raise ValueError("GDN RecoverSSM input exceeds its record capacity")
    if any(t.dtype not in (torch.int32, torch.int64) for t in (state_indices, query_start_loc)):
        raise ValueError("GDN RecoverSSM request metadata must use integer tensors")
    out = torch.empty_like(v, memory_format=torch.contiguous_format)
    if rows and tokens:
        block_v = min(triton.next_power_of_2(dv), RECOVERSSM_VALUE_TILE)
        # Larger A5 tiles reduce repeated K-side loads when the remaining grid
        # still has enough programs to keep the device occupied.
        is_a5 = is_950()
        if is_a5 and block_v < RECOVERSSM_VALUE_TILE_LARGE and rows * hv >= RECOVERSSM_LARGE_TILE_MIN_PROGRAMS:
            large_v = min(triton.next_power_of_2(dv), RECOVERSSM_VALUE_TILE_LARGE)
            if rows * hv * triton.cdiv(dv, large_v) >= RECOVERSSM_LARGE_TILE_MIN_PROGRAMS:
                block_v = large_v
        grid = (triton.cdiv(dv, block_v), hv, rows)
        _gdn_recoverssm_verify_kernel[grid](
            # Inputs, persistent checkpoint, and per-token records.
            q_ptr=q,
            k_ptr=k,
            v_ptr=v,
            gate_ptr=g,
            beta_ptr=beta,
            state_ptr=checkpoint_state,
            replay_ptr=replay_cache,
            query_start_loc_ptr=query_start_loc,
            state_indices_ptr=state_indices,
            out_ptr=out,
            null_block_id=0,
            # Activation and checkpoint layouts.
            stride_q_token=q.stride(1),
            stride_q_head=q.stride(2),
            stride_k_token=k.stride(1),
            stride_k_head=k.stride(2),
            stride_v_token=v.stride(1),
            stride_v_head=v.stride(2),
            stride_gate_token=g.stride(1),
            stride_gate_head=g.stride(2),
            stride_beta_token=beta.stride(1),
            stride_beta_head=beta.stride(2),
            state_strides=checkpoint_state.stride(),
            replay_strides=replay_cache.stride(),
            stride_out_token=out.stride(1),
            stride_out_head=out.stride(2),
            stride_query_start_loc=query_start_loc.stride(0),
            stride_state_indices=state_indices.stride(0),
            # Compile-time shape and input-format specialization.
            NUM_KEY_HEADS=hk,
            NUM_VALUE_HEADS=hv,
            K=dk,
            V=dv,
            BK=triton.next_power_of_2(dk),
            BV=block_v,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@dataclass(frozen=True)
class GDNRecoverSSMLayerCache:
    """Cache tensors and kernel address metadata for one GDN layer."""

    conv: torch.Tensor
    conv_base_addr: int
    conv_block_stride: int
    conv_dim_stride: int
    conv_token_stride: int
    checkpoint: torch.Tensor
    checkpoint_base_addr: int
    checkpoint_block_stride: int
    replay: torch.Tensor
    replay_base_addr: int
    replay_block_stride: int
    num_key_heads: int

    @classmethod
    def from_layer(cls, layer: Any) -> "GDNRecoverSSMLayerCache":
        if len(layer.kv_cache) != GDN_PAGE_COUNT:
            raise ValueError("GDN RecoverSSM requires conv, checkpoint and replay views")
        conv, checkpoint, replay = layer.kv_cache
        if not is_conv_state_dim_first():
            conv = conv.transpose(-1, -2)
        return cls(
            conv=conv,
            conv_base_addr=conv.data_ptr(),
            conv_block_stride=conv.stride(0),
            conv_dim_stride=conv.stride(1),
            conv_token_stride=conv.stride(2),
            checkpoint=checkpoint,
            checkpoint_base_addr=checkpoint.data_ptr(),
            checkpoint_block_stride=checkpoint.stride(0),
            replay=replay,
            replay_base_addr=replay.data_ptr(),
            replay_block_stride=replay.stride(0),
            num_key_heads=layer.num_k_heads // layer.tp_size,
        )

    def state_pages(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.checkpoint, self.replay

    def all_pages(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.conv, *self.state_pages()


def _create_state_commit_cache_address_stride_table(
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...],
) -> torch.Tensor:
    """Create the cache-kind-major address/stride table consumed by the kernel."""
    return (
        torch.tensor(
            [
                [
                    [cache.checkpoint_base_addr, cache.checkpoint_block_stride],
                    [cache.replay_base_addr, cache.replay_block_stride],
                ]
                for cache in layer_caches
            ],
            dtype=torch.int64,
            device=layer_caches[0].checkpoint.device,
        )
        .permute(1, 2, 0)
        .contiguous()
    )


def _create_conv_cache_address_stride_table(
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...],
) -> torch.Tensor:
    """Create the field-major convolution address/stride table for the kernel."""
    return torch.tensor(
        [
            [
                cache.conv_base_addr,
                cache.conv_block_stride,
                cache.conv_dim_stride,
                cache.conv_token_stride,
            ]
            for cache in layer_caches
        ],
        dtype=torch.int64,
        device=layer_caches[0].conv.device,
    ).T.contiguous()


def _get_shared_conv_dim_stride(
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...],
) -> int | None:
    """Return the common dim stride, or None when layers use different strides."""
    reference_stride = layer_caches[0].conv_dim_stride
    if all(cache.conv_dim_stride == reference_stride for cache in layer_caches):
        return reference_stride
    return None


@dataclass(frozen=True)
class GDNRecoverSSMCommitPlan:
    """Reusable per-request buffers populated by the commit-plan kernel."""

    commit_lens: torch.Tensor
    state_commit_lens: torch.Tensor
    boundary_commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor

    @classmethod
    def allocate(cls, max_num_reqs: int, device: torch.device) -> "GDNRecoverSSMCommitPlan":
        # Allocate one reusable contiguous buffer for six per-request plan
        # arrays: commit length, non-boundary/boundary commit lengths, final and
        # boundary state indices, and boundary recovery length. The row slices
        # below are views sharing this allocation. The plan kernel overwrites
        # every row in [:batch] before use, so uninitialized capacity beyond the
        # current batch is safe and avoids an unnecessary device-side fill.
        #
        #                                        request 0   request 1   request 2   request 3
        #                                      ┌───────────┬───────────┬───────────┬───────────┐
        # storage[0] commit_lens               │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[1] state_commit_lens         │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[2] boundary_commit_lens      │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[3] final_state_indices       │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[4] boundary_state_indices    │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[5] boundary_recovery_lens    │           │           │           │           │
        #                                      └───────────┴───────────┴───────────┴───────────┘
        storage = torch.empty((6, max_num_reqs), dtype=torch.int32, device=device)
        return cls(
            commit_lens=storage[0],
            state_commit_lens=storage[1],
            boundary_commit_lens=storage[2],
            final_state_indices=storage[3],
            boundary_state_indices=storage[4],
            boundary_recovery_lens=storage[5],
        )

    @property
    def capacity(self) -> int:
        return self.commit_lens.shape[0]


def _validate_layer_caches(
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...],
    spec_query_len: int,
) -> int:
    reference = layer_caches[0]
    if reference.checkpoint.ndim != 4:
        raise ValueError("GDN RecoverSSM checkpoint must have four dimensions")
    blocks, num_value_heads, value_dim, key_dim = reference.checkpoint.shape
    for cache in layer_caches:
        _validate_state_replay(cache.checkpoint, cache.replay, num_value_heads, value_dim, key_dim, spec_query_len)
        if cache.num_key_heads != reference.num_key_heads:
            raise ValueError("GDN RecoverSSM layers must share key head geometry")
    num_key_heads = reference.num_key_heads
    if num_key_heads < 1 or num_value_heads % num_key_heads:
        raise ValueError("GDN RecoverSSM requires integral GQA")

    channels = 2 * num_key_heads * key_dim + num_value_heads * value_dim
    if (
        reference.conv.ndim != 3
        or reference.conv.shape[:2] != (blocks, channels)
        or reference.conv.device != reference.checkpoint.device
    ):
        raise ValueError("GDN RecoverSSM convolution page has incompatible layout")
    history_len = reference.conv.shape[2] - spec_query_len + 1
    if history_len < 1:
        raise ValueError("GDN RecoverSSM convolution history is too short")

    for current in layer_caches[1:]:
        if any(
            (tensor.shape, tensor.dtype, tensor.device)
            != (reference_tensor.shape, reference_tensor.dtype, reference_tensor.device)
            for tensor, reference_tensor in zip(current.all_pages(), reference.all_pages())
        ):
            raise ValueError("GDN RecoverSSM layers must share shapes, dtypes and device")
        for name, tensor, reference_tensor in zip(
            ("checkpoint", "replay"),
            current.state_pages(),
            reference.state_pages(),
        ):
            if _inner_stride_signature(tensor) != _inner_stride_signature(reference_tensor):
                raise ValueError(f"GDN RecoverSSM layers must share {name} page inner strides")
    return history_len


@dataclass
class GDNRecoverSSMCommitContext:
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...]
    commit_plan: GDNRecoverSSMCommitPlan
    spec_query_len: int
    # Number of convolution-history tokens retained after committing the
    # accepted prefix: conv_window_len - spec_query_len + 1.
    conv_history_len: int
    # Common conv dim stride used as a compile-time kernel constant. None
    # makes the kernel load each layer's stride from
    # _conv_cache_address_stride_table.
    conv_shared_dim_stride: int | None
    # Contiguous device-side int64 table with shape [4, num_layers]. Column i
    # describes layer_caches[i].conv:
    #   [0, i] = conv.data_ptr(), the base address of layer i's conv cache.
    #   [1, i] = conv.stride(0), the cache-block stride in elements.
    #   [2, i] = conv.stride(1), the channel/dim stride in elements.
    #   [3, i] = conv.stride(2), the token/history stride in elements.
    # The compact-conv kernel indexes each row with its device-side layer id.
    _conv_cache_address_stride_table: torch.Tensor = field(repr=False)
    # Device-side int64 table [2, 2, num_layers]: checkpoint/replay,
    # base address/block stride in elements, layer index.
    _state_commit_cache_address_stride_table: torch.Tensor = field(repr=False)

    @classmethod
    def create(
        cls,
        layers: Sequence[Any],
        *,
        spec_query_len: int,
        max_num_reqs: int,
    ) -> "GDNRecoverSSMCommitContext":
        """Bind named layer caches and allocate reusable commit metadata."""
        if not layers or spec_query_len < 1 or max_num_reqs < 1:
            raise ValueError("GDN RecoverSSM requires layers and positive capacities")
        layer_caches = tuple(GDNRecoverSSMLayerCache.from_layer(layer) for layer in layers)
        history_len = _validate_layer_caches(layer_caches, spec_query_len)
        device = layer_caches[0].checkpoint.device
        if device.type == "npu":
            init_device_properties_triton()
        return cls(
            layer_caches=layer_caches,
            commit_plan=GDNRecoverSSMCommitPlan.allocate(max_num_reqs, device),
            spec_query_len=spec_query_len,
            conv_history_len=history_len,
            conv_shared_dim_stride=_get_shared_conv_dim_stride(layer_caches),
            _conv_cache_address_stride_table=_create_conv_cache_address_stride_table(layer_caches),
            _state_commit_cache_address_stride_table=_create_state_commit_cache_address_stride_table(layer_caches),
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        *,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
    ) -> None:
        """Commit each request's accepted prefix into every bound GDN layer."""
        align_args = (block_table, num_computed_tokens, mamba_block_size)
        if any(arg is not None for arg in align_args) and any(arg is None for arg in align_args):
            raise ValueError("GDN RecoverSSM align metadata is incomplete")
        align_mode = block_table is not None
        if align_mode:
            if mamba_block_size < self.spec_query_len:
                raise ValueError("GDN RecoverSSM align block size must cover one speculative window")
            if block_table.ndim != 2 or block_table.shape[1] < 1:
                raise ValueError("GDN RecoverSSM block table must be nonempty and two-dimensional")
            if block_table.dtype not in (torch.int32, torch.int64):
                raise ValueError("GDN RecoverSSM block table must use integer indices")
        batch = state_indices.numel()
        if batch > self.commit_plan.capacity or state_indices.ndim != 1 or query_start_loc.shape != (batch + 1,):
            raise ValueError("GDN RecoverSSM commit request metadata exceeds capacity or has incompatible shape")
        if request_indices is not None and request_indices.shape != (batch,):
            raise ValueError("GDN RecoverSSM sampler mapping must have one row per request")
        if request_indices is None and num_accepted_tokens.numel() < batch:
            raise ValueError("GDN RecoverSSM acceptance counts are too short")
        reference = self.layer_caches[0]
        device = reference.checkpoint.device
        if any(t.device != device for t in (num_accepted_tokens, state_indices, query_start_loc)):
            raise ValueError("GDN RecoverSSM commit tensors must share the cache device")
        if request_indices is not None and request_indices.device != device:
            raise ValueError("GDN RecoverSSM sampler mapping must be on the cache device")
        if align_mode:
            if block_table.device != device or num_computed_tokens.device != device:
                raise ValueError("GDN RecoverSSM align metadata must be on the cache device")
            if request_indices is None and (block_table.shape[0] < batch or num_computed_tokens.numel() < batch):
                raise ValueError("GDN RecoverSSM align metadata must cover the sampler rows")
        if not batch:
            return

        conv = reference.conv
        state = reference.checkpoint
        plan = self.commit_plan
        _, hv, dv, dk = state.shape
        # 1. Resolve accepted lengths and both destinations once per request.
        block_table_stride = (0, 0) if block_table is None else block_table.stride()
        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            plan.commit_lens,
            plan.state_commit_lens,
            plan.boundary_commit_lens,
            plan.final_state_indices,
            plan.boundary_state_indices,
            plan.boundary_recovery_lens,
            0,
            mamba_block_size or 1,
            block_table.shape[1] if align_mode else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            block_table_stride[0],
            block_table_stride[1],
            num_computed_tokens.stride(0) if align_mode else 0,
            SPEC_QUERY_LEN=self.spec_query_len,
            PROMOTE_REQUEST_INDEX_TO_INT64=False,
            num_warps=1,
        )
        # 2. Copy final and optional boundary convolution histories.
        # Strided channel gathers need smaller tiles to fit the A3 UB.
        conv_block = (
            RECOVERSSM_CONV_TILE if is_950() or self.conv_shared_dim_stride == 1 else RECOVERSSM_CONV_STRIDED_TILE
        )
        conv_cache_address_stride_table = self._conv_cache_address_stride_table
        num_layers = len(self.layer_caches)
        _compact_conv_state_kernel[(triton.cdiv(conv.shape[1], conv_block), batch, num_layers)](
            conv,
            conv_cache_address_stride_table[0],
            conv_cache_address_stride_table[1],
            conv_cache_address_stride_table[2],
            conv_cache_address_stride_table[3],
            state_indices,
            plan.commit_lens,
            plan.final_state_indices,
            plan.boundary_state_indices,
            plan.boundary_recovery_lens,
            0,
            conv.shape[1],
            self.conv_history_len,
            state_indices.stride(0),
            BLOCK_D=conv_block,
            BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len),
            ALIGN_MODE=align_mode,
            DIM_STRIDE=self.conv_shared_dim_stride,
            USE_1D_CHANNEL_GATHER=not is_950(),
            num_warps=4,
        )
        # 3. Compose accepted records backwards, then merge the checkpoint.
        replay = reference.replay
        state_commit_cache_address_stride_table = self._state_commit_cache_address_stride_table
        block_k = triton.next_power_of_2(dk)
        matrix_element_size = replay.element_size()
        is_a5 = is_950()
        grouped = is_a5 and dk == dv == 128 and hv >= 2 and hv % RECOVERSSM_COMMIT_HEAD_TILE == 0
        value_tile = RECOVERSSM_VALUE_TILE
        if grouped:
            head_groups = hv // RECOVERSSM_COMMIT_HEAD_TILE
            commit_kernel = _commit_gdn_state_grouped_kernel
            options = {}
        else:
            # Match KDA's larger A5 ordinary-commit tile when the launch does
            # not also carry boundary accumulators.
            if is_a5 and not align_mode:
                value_tile *= 2
            head_groups = hv
            commit_kernel = _commit_gdn_state_kernel
            options = {"UNROLL_COMMIT": is_a5}

        # KDA launches ordinary and boundary commits separately. GDN keeps one
        # launch, so every aligned program is compiled with the final/boundary
        # accumulators and must use the boundary UB budget. The grouped kernel
        # holds two heads' matrices in each program.
        ub_matrices = RECOVERSSM_BOUNDARY_COMMIT_UB_MATRICES if align_mode else RECOVERSSM_COMMIT_UB_MATRICES
        bytes_per_row = ub_matrices * block_k * matrix_element_size
        if grouped:
            bytes_per_row *= RECOVERSSM_COMMIT_HEAD_TILE
        max_rows = max(1, get_ub_size_bytes() // bytes_per_row - RECOVERSSM_COMMIT_UB_HEADROOM_ROWS)
        value_tile = min(value_tile, 1 << (max_rows.bit_length() - 1))
        block_v = min(triton.next_power_of_2(dv), value_tile)
        # Explicit flat indexing keeps request rows outermost on both 3.2/3.6
        # backends, balancing mixed boundary requests without a new static flag.
        grid = (triton.cdiv(dv, block_v) * num_layers * head_groups * batch,)
        commit_kernel[grid](
            state_ref_ptr=state,
            state_base_addrs_ptr=state_commit_cache_address_stride_table[0, 0],
            state_block_strides_ptr=state_commit_cache_address_stride_table[0, 1],
            replay_ref_ptr=replay,
            replay_base_addrs_ptr=state_commit_cache_address_stride_table[1, 0],
            replay_block_strides_ptr=state_commit_cache_address_stride_table[1, 1],
            state_indices_ptr=state_indices,
            commit_lens_ptr=plan.commit_lens,
            final_state_indices_ptr=plan.final_state_indices,
            boundary_state_indices_ptr=plan.boundary_state_indices,
            boundary_recovery_lens_ptr=plan.boundary_recovery_lens,
            null_block_id=0,
            stride_state_head=state.stride(1),
            stride_state_v=state.stride(2),
            stride_state_k=state.stride(3),
            stride_replay_head=replay.stride(1),
            stride_replay_pos=replay.stride(2),
            stride_replay_dim=replay.stride(3),
            stride_state_indices=state_indices.stride(0),
            num_layers=num_layers,
            K=dk,
            V=dv,
            BK=block_k,
            BV=block_v,
            NUM_HEADS=hv,
            ALIGN_MODE=align_mode,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=4,
            enable_fp_fusion=False,
            **options,
        )
