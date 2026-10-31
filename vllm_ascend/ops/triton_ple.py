# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend-friendly Qwen PLE prefill short convolution."""

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

PLE_BLOCK_C = 2048
PLE_NUM_WARPS = 8


# Dilated short convolution
@triton.jit(do_not_specialize=["num_reqs", "bs_iters", "has_token_map"])
def _ple_conv_kernel(
    x_ptr,
    state_ptr,
    w_ptr,
    residual_ptr,
    outer_residual_ptr,
    state_idx_ptr,
    qsl_ptr,
    has_init_ptr,
    token_idx_ptr,
    has_token_map,
    num_reqs,
    bs_iters,
    state_idx_stride,
    state_bs,
    state_ws,
    state_cs,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    DILATION: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
    FUSE_RESIDUAL: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    t = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # Mixed batches map logical rows back to the original token order.
    output_t = tl.load(token_idx_ptr + t, mask=has_token_map, other=t).to(tl.int64)

    # Locate the request containing this token.
    lo = tl.full([], 1, tl.int32)
    hi = tl.full([], num_reqs + 1, tl.int32)
    for _ in range(bs_iters):
        mid = (lo + hi) // 2
        qmid = tl.load(qsl_ptr + mid, mask=mid <= num_reqs, other=0)
        pred = qmid <= t
        lo = tl.where(pred, mid + 1, lo)
        hi = tl.where(pred, hi, mid)
    r = tl.minimum(lo - 1, num_reqs - 1)
    q_start = tl.load(qsl_ptr + r)
    j = (t - q_start).to(tl.int32)
    total_real = tl.load(qsl_ptr + num_reqs)
    in_range = t < total_real

    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    sid_safe = tl.where(state_ok, sid, 0)
    has_init = tl.load(has_init_ptr + r, mask=state_ok, other=0) != 0
    read_state = state_ok & has_init
    out_ok = in_range & state_ok

    # state[sid, w, c] may be a strided view into a shared pool.
    base_state = state_ptr + sid_safe * state_bs
    acc = tl.zeros([BLOCK_C], tl.float32)
    for k in tl.static_range(0, KERNEL_SIZE):
        h = j + DILATION * k
        from_state = h <= STATE_LEN - 1
        state_tap = tl.full([BLOCK_C], 0, state_ptr.dtype.element_ty)
        if from_state:
            state_tap = tl.load(
                base_state + h * state_ws + c_offs * state_cs,
                mask=c_mask & read_state,
                other=0.0,
            )
        input_t = q_start + h - STATE_LEN
        input_t = tl.load(
            token_idx_ptr + input_t,
            mask=has_token_map & out_ok & (~from_state),
            other=input_t,
        ).to(tl.int64)
        input_tap = tl.load(
            x_ptr + input_t * C + c_offs,
            mask=c_mask & out_ok & (~from_state),
            other=0.0,
        )
        tap = tl.where(from_state, state_tap, input_tap).to(tl.float32)
        weight = tl.load(
            w_ptr + k * C + c_offs,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)
        acc += weight * tap

    # F.conv1d materializes its output dtype before SiLU.
    conv = acc.to(residual_ptr.dtype.element_ty).to(tl.float32)
    y = conv * tl.sigmoid(conv)
    conv_output = tl.where(out_ok, y, 0.0).to(residual_ptr.dtype.element_ty)
    ple_output = conv_output
    if FUSE_RESIDUAL:
        residual = tl.load(
            residual_ptr + output_t * C + c_offs,
            mask=c_mask,
            other=0.0,
        )
        # Preserve the original eager operation boundaries: short convolution is
        # first accumulated into the BF16/FP16 PLE output, then the outer residual
        # is added to that rounded value.
        ple_output = (residual + conv_output).to(residual_ptr.dtype.element_ty)
        outer_residual = tl.load(
            outer_residual_ptr + output_t * C + c_offs,
            mask=c_mask,
            other=0.0,
        )
        ple_output = outer_residual.to(tl.float32) + ple_output.to(tl.float32)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        residual_ptr + output_t * C + c_offs,
        ple_output,
        mask=c_mask,
    )


# Write back separately so every token reads the old state.
@triton.jit(do_not_specialize=["has_token_map"])
def _ple_conv_writeback_kernel(
    x_ptr,
    state_ptr,
    state_idx_ptr,
    qsl_ptr,
    has_init_ptr,
    token_idx_ptr,
    has_token_map,
    state_idx_stride,
    state_bs,
    state_ws,
    state_cs,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
):
    r = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    if not state_ok:
        return

    q_start = tl.load(qsl_ptr + r)
    q_end = tl.load(qsl_ptr + r + 1)
    qlen = (q_end - q_start).to(tl.int32)
    if qlen <= 0:
        return

    has_init = tl.load(has_init_ptr + r) != 0
    src_ok = state_ok & has_init

    base_state = state_ptr + sid * state_bs
    for i in tl.static_range(0, STATE_LEN):
        m = qlen + i
        from_state = m <= STATE_LEN - 1
        state_value = tl.load(
            base_state + m * state_ws + c_offs * state_cs,
            mask=c_mask & from_state & src_ok,
            other=0.0,
        )
        input_t = q_start + m - STATE_LEN
        input_t = tl.load(
            token_idx_ptr + input_t,
            mask=has_token_map & (~from_state),
            other=input_t,
        ).to(tl.int64)
        input_value = tl.load(
            x_ptr + input_t * C + c_offs,
            mask=c_mask & (~from_state),
            other=0.0,
        )
        value = tl.where(from_state, state_value, input_value)
        tl.store(
            base_state + i * state_ws + c_offs * state_cs,
            value,
            mask=c_mask,
        )


def ple_conv_prefill(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    outer_residual: torch.Tensor | None,
    *,
    dilation: int,
    query_start_loc: torch.Tensor | None = None,
    has_initial_states: torch.Tensor | None = None,
    token_indices: torch.Tensor | None = None,
) -> None:
    """Update state and write activated convolution into residual.
    A non-None outer_residual enables both fused residual additions.
    """
    BLOCK_C = PLE_BLOCK_C
    T, C = inputs.shape
    K = conv_weights.shape[1]
    state_len = (K - 1) * dilation
    if token_indices is not None:
        T = token_indices.numel()
    if conv_state.shape[1] != C or conv_state.shape[2] < state_len:
        raise ValueError(
            f"conv_state must have shape [slots, channels, window], with channels={C} and window >= {state_len}"
        )
    state_bs, state_cs, state_ws = conv_state.stride()

    if query_start_loc is None or has_initial_states is None:
        raise ValueError("query_start_loc and has_initial_states are required for prefill")
    num_reqs = state_indices.numel()
    binary_search_iters = max(num_reqs, 1).bit_length()

    launch_pdl = current_platform.is_arch_support_pdl()
    # Pure-prefill indices can be a strided block-table column view.
    state_idx_stride = state_indices.stride(0)

    # Constexpr flags eliminate accesses to optional None arguments. Without a
    # token map, state_indices is an unused but device-resident placeholder.
    weights_by_tap = conv_weights.transpose(0, 1).contiguous()
    _ple_conv_kernel[(T, triton.cdiv(C, BLOCK_C))](
        inputs,
        conv_state,
        weights_by_tap,
        residual,
        outer_residual,
        state_indices,
        query_start_loc,
        has_initial_states,
        token_indices if token_indices is not None else state_indices,
        token_indices is not None,
        num_reqs,
        binary_search_iters,
        state_idx_stride,
        state_bs,
        state_ws,
        state_cs,
        C=C,
        BLOCK_C=BLOCK_C,
        STATE_LEN=state_len,
        DILATION=dilation,
        KERNEL_SIZE=K,
        FUSE_RESIDUAL=outer_residual is not None,
        NULL_STATE_ID=NULL_BLOCK_ID,
        launch_pdl=launch_pdl,
        num_warps=PLE_NUM_WARPS,
    )
    _ple_conv_writeback_kernel[(num_reqs, triton.cdiv(C, BLOCK_C))](
        inputs,
        conv_state,
        state_indices,
        query_start_loc,
        has_initial_states,
        token_indices if token_indices is not None else state_indices,
        token_indices is not None,
        state_idx_stride,
        state_bs,
        state_ws,
        state_cs,
        C=C,
        BLOCK_C=BLOCK_C,
        STATE_LEN=state_len,
        NULL_STATE_ID=NULL_BLOCK_ID,
        num_warps=PLE_NUM_WARPS,
    )
