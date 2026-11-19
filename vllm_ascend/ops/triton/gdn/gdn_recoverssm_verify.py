# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify a paged GDN window and emit replay records on Ascend."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.device.device_config import is_950

RECOVERSSM_VALUE_TILE = 64
RECOVERSSM_VALUE_TILE_LARGE = 128
RECOVERSSM_LARGE_TILE_MIN_PROGRAMS = 96


@triton.jit
def _gdn_recoverssm_verify_kernel(
    A_log_ptr,
    a_ptr,
    b_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    replay_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    out_ptr,
    null_block_id,
    scale,
    softplus_beta,
    softplus_threshold,
    stride_a_token: tl.constexpr,
    stride_a_head: tl.constexpr,
    stride_b_token: tl.constexpr,
    stride_b_head: tl.constexpr,
    stride_q_token: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_head: tl.constexpr,
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
    USE_QK_L2NORM: tl.constexpr,
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
    neg_A = -tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    q_start = q_ptr + bos * stride_q_token + key_head * stride_q_head + offs_k
    k_start = k_ptr + bos * stride_k_token + key_head * stride_k_head + offs_k
    qk_other = 0.0 if mask_k is not None else None
    next_q = tl.load(q_start, mask_k, qk_other)
    next_k = tl.load(k_start, mask_k, qk_other)
    for token_offset in range(query_len):
        token = bos + token_offset
        q = next_q.to(tl.float32)
        k = next_k.to(tl.float32)
        # Load the next Q/K before the current state update so the backend
        # can overlap them. Keep the prefetched values in the input dtype.
        next_offset = token_offset + 1
        if next_offset < query_len:
            next_q = tl.load(q_start + next_offset * stride_q_token, mask_k, qk_other)
            next_k = tl.load(k_start + next_offset * stride_k_token, mask_k, qk_other)
        v = tl.load(
            v_ptr + token * stride_v_token + pid_h * stride_v_head + offs_v,
            mask_v,
            0.0 if mask_v is not None else None,
        ).to(tl.float32)
        x = tl.load(a_ptr + token * stride_a_token + pid_h * stride_a_head).to(tl.float32) + dt_bias
        softplus_x = tl.where(
            softplus_beta * x <= softplus_threshold,
            tl.log(1.0 + tl.exp(softplus_beta * x)) / softplus_beta,
            x,
        )
        gdn_gate = neg_A * softplus_x
        beta = tl.sigmoid(tl.load(b_ptr + token * stride_b_token + pid_h * stride_b_head).to(tl.float32))
        if USE_QK_L2NORM:
            q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6)
            k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
        q = q * scale

        # Store the same FP32 decay that is multiplied into the state.
        decay = tl.exp(gdn_gate)
        normalized_k = k

        decayed_state = state * decay
        correction = (v - tl.sum(decayed_state * normalized_k[None, :], axis=1)) * beta
        # GDN output consumes the updated state, including for the last token.
        state = decayed_state + correction[:, None] * normalized_k[None, :]
        out = tl.sum(state * q[None, :], axis=1)
        tl.store(out_ptr + token * stride_out_token + pid_h * stride_out_head + offs_v, out, mask_v)

        record_ptr = (
            replay_ptr
            + state_idx * stride_replay_block
            + pid_h * stride_replay_head
            + token_offset * stride_replay_position
        )
        tl.store(record_ptr + offs_v * stride_replay_dim, correction, mask_v)
        if pid_v == 0:
            tl.store(record_ptr + (V + offs_k) * stride_replay_dim, normalized_k, mask_k)
            tl.store(record_ptr + (V + K) * stride_replay_dim, decay)


def validate_gdn_state_replay(checkpoint, replay, hv, dv, dk, spec_query_len):
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
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    checkpoint_state: torch.Tensor,
    replay_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    scale: float | None = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_qk_l2norm_in_kernel: bool = True,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return GDN outputs and compact records, leaving the checkpoint intact.

    Q/K use shape ``[1, T, HK, K]``; V uses ``[1, T, HV, V]``;
    A/B use ``[T, HV]`` or ``[1, T, HV]``.
    """
    if q.ndim != 4 or q.shape[0] != 1 or k.shape != q.shape:
        raise ValueError("GDN RecoverSSM Q/K must have matching [1,T,HK,K] shapes")
    _, tokens, hk, dk = q.shape
    if v.ndim != 4 or v.shape[:2] != (1, tokens):
        raise ValueError("GDN RecoverSSM V must have shape [1,T,HV,V]")
    hv, dv = v.shape[2:]
    if hk < 1 or hv % hk or spec_query_len < 1:
        raise ValueError("GDN RecoverSSM requires integral GQA and a positive window")
    if a.numel() != tokens * hv or b.numel() != tokens * hv:
        raise ValueError("GDN RecoverSSM gate inputs have incompatible shapes")
    a = a.reshape(tokens, hv)
    b = b.reshape(tokens, hv)
    if a.stride(1) != 1 or b.stride(1) != 1:
        raise ValueError("GDN RecoverSSM gate input heads must be contiguous")
    if A_log.shape != (hv,) or dt_bias.shape != (hv,):
        raise ValueError("GDN RecoverSSM A_log or dt_bias has incompatible shape")
    if softplus_beta <= 0:
        raise ValueError("GDN RecoverSSM softplus_beta must be positive")
    validate_gdn_state_replay(checkpoint_state, replay_cache, hv, dv, dk, spec_query_len)
    tensors = (
        A_log,
        a,
        b,
        dt_bias,
        q,
        k,
        v,
        checkpoint_state,
        replay_cache,
        query_start_loc,
        state_indices,
    )
    if any(t.device != q.device for t in tensors):
        raise ValueError("GDN RecoverSSM inputs must share a device")
    if any(t.stride(-1) != 1 or t.stride(-2) != dk for t in (q, k)):
        raise ValueError("GDN RecoverSSM Q/K heads must be contiguous")
    if v.stride(-1) != 1 or v.stride(-2) != dv:
        raise ValueError("GDN RecoverSSM V heads must be contiguous")
    rows = state_indices.numel()
    if state_indices.ndim != 1 or query_start_loc.shape != (rows + 1,):
        raise ValueError("GDN RecoverSSM request indices and offsets do not match")
    if tokens > rows * spec_query_len:
        raise ValueError("GDN RecoverSSM input exceeds its record capacity")
    if any(t.dtype not in (torch.int32, torch.int64) for t in (state_indices, query_start_loc)):
        raise ValueError("GDN RecoverSSM request metadata must use integer tensors")
    if scale is None:
        scale = dk**-0.5
    if out is None:
        out = torch.empty_like(v, memory_format=torch.contiguous_format)
    elif out.device != q.device:
        raise ValueError("GDN RecoverSSM output must share the input device")
    if out.shape != v.shape or out.stride()[2:] != (dv, 1):
        raise ValueError("GDN RecoverSSM output shape or layout is incompatible")
    if rows and tokens:
        block_v = min(triton.next_power_of_2(dv), RECOVERSSM_VALUE_TILE)
        # Larger A5 tiles reduce repeated K-side loads when the remaining grid
        # still has enough programs to keep the device occupied.
        is_a5 = is_950()
        if is_a5 and block_v < RECOVERSSM_VALUE_TILE_LARGE and rows * hv >= RECOVERSSM_LARGE_TILE_MIN_PROGRAMS:
            block_v = min(triton.next_power_of_2(dv), RECOVERSSM_VALUE_TILE_LARGE)
        grid = (triton.cdiv(dv, block_v), hv, rows)
        _gdn_recoverssm_verify_kernel[grid](
            # Inputs, persistent checkpoint, and per-token records.
            A_log_ptr=A_log,
            a_ptr=a,
            b_ptr=b,
            dt_bias_ptr=dt_bias,
            q_ptr=q,
            k_ptr=k,
            v_ptr=v,
            state_ptr=checkpoint_state,
            replay_ptr=replay_cache,
            query_start_loc_ptr=query_start_loc,
            state_indices_ptr=state_indices,
            out_ptr=out,
            null_block_id=0,
            scale=scale,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            # Activation and checkpoint layouts.
            stride_a_token=a.stride(0),
            stride_a_head=a.stride(1),
            stride_b_token=b.stride(0),
            stride_b_head=b.stride(1),
            stride_q_token=q.stride(1),
            stride_q_head=q.stride(2),
            stride_k_token=k.stride(1),
            stride_k_head=k.stride(2),
            stride_v_token=v.stride(1),
            stride_v_head=v.stride(2),
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
            USE_QK_L2NORM=use_qk_l2norm_in_kernel,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
