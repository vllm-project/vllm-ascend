# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reconstruct accepted GDN checkpoints from paged replay records on Ascend."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.triton.triton_utils import get_ub_size_bytes

RECOVERSSM_COMMIT_HEAD_TILE = 2
# Conservative UB budgets in BV x BK FP32 matrix equivalents. The GDN
# align path commits final and boundary states in one kernel, so it uses the
# larger boundary budget for the whole launch.
RECOVERSSM_COMMIT_UB_MATRICES = 2
RECOVERSSM_BOUNDARY_COMMIT_UB_MATRICES = 6
# Reserve tile rows for FP32 vectors and other temporary workspace.
RECOVERSSM_COMMIT_UB_HEADROOM_ROWS = 2
RECOVERSSM_VALUE_TILE = 64


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
    need_boundary,
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
        # A scalar branch skips the FMA; tl.where evaluates both operands.
        if need_boundary & (token_record_pos < boundary_recovery_len):
            boundary_k = normalized_k * boundary_decay
            boundary_correction = tl.fma(correction[:, None], boundary_k[None, :], boundary_correction)
            boundary_decay *= decay
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
    # The final write already covers an identical prefix at the same page.
    boundary_and_final_are_same = (boundary_state_idx == final_state_idx) & (boundary_recovery_len == commit_len)
    need_boundary = (boundary_state_idx > null_block_id) & ~boundary_and_final_are_same

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
                    need_boundary,
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
                    need_boundary,
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
        if need_boundary:
            boundary_ptrs = (
                state_ptr
                + boundary_state_idx * state_block_stride
                + pid_h * stride_state_head
                + offs_v[:, None] * stride_state_v
                + offs_k[None, :] * stride_state_k
            )
            boundary_state = initial_state * boundary_decay[None, :] + boundary_correction
            tl.store(boundary_ptrs, boundary_state, mask=mask_state)

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
        # The final write already covers an identical prefix at the same page.
        boundary_and_final_are_same = (boundary_state_idx == final_state_idx) & (boundary_recovery_len == commit_len)
        need_boundary = (boundary_state_idx > null_block_id) & ~boundary_and_final_are_same
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
                # Non-crossing requests can still have a positive boundary length.
                # Check the destination as well, and skip unused matrix updates.
                if need_boundary & (token_record_pos < boundary_recovery_len):
                    boundary_k = normalized_k * boundary_decay
                    boundary_correction = tl.fma(correction[:, :, None], boundary_k[:, None, :], boundary_correction)
                    boundary_decay *= decay
    initial_state = tl.load(state_ptrs, mask=mask_state, other=0.0 if mask_state is not None else None).to(tl.float32)
    state = initial_state * final_decay[:, None, :] + final_correction
    if ALIGN_MODE:
        if need_boundary:
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


def gdn_recoverssm_commit(
    state: torch.Tensor,
    replay: torch.Tensor,
    state_commit_cache_address_stride_table: torch.Tensor,
    state_indices: torch.Tensor,
    commit_lens: torch.Tensor,
    final_state_indices: torch.Tensor,
    boundary_state_indices: torch.Tensor,
    boundary_recovery_lens: torch.Tensor,
    *,
    spec_query_len: int,
    align_mode: bool,
) -> None:
    """Reconstruct accepted checkpoints using a prepared commit plan."""
    batch = state_indices.numel()
    num_layers = state_commit_cache_address_stride_table.shape[2]
    _, hv, dv, dk = state.shape
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
        commit_lens_ptr=commit_lens,
        final_state_indices_ptr=final_state_indices,
        boundary_state_indices_ptr=boundary_state_indices,
        boundary_recovery_lens_ptr=boundary_recovery_lens,
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
        SPEC_QUERY_LEN=spec_query_len,
        num_warps=4,
        enable_fp_fusion=False,
        **options,
    )
