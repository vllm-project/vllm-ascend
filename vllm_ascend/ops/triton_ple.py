# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend-friendly Qwen PLE short convolution."""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

PLE_BLOCK_C = 512
PLE_NUM_WARPS = 8


@triton.jit
def _ple_gate_materialized_kernel(
    key,
    value,
    hidden,
    nk,
    nq,
    nc,
    previous,
    current,
    gated,
    normed,
    key_stride,
    value_stride,
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
    STAGE: tl.constexpr,
):
    t, group = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    mask = lane < H
    offset = group * H + lane
    dtype: tl.constexpr = hidden.dtype.element_ty
    if STAGE == 0:
        k = tl.load(key + t * key_stride + offset, mask, 0).to(tl.float32)
        q = tl.load(hidden + t * HC * H + offset, mask, 0).to(tl.float32)
        kw = tl.load(nk + offset, mask, 0).to(tl.float32)
        qw = tl.load(nq + offset, mask, 0).to(tl.float32)
        kn = (k * tl.rsqrt(tl.sum(k * k) / H + eps) * (1.0 + kw)).to(dtype)
        qn = (q * tl.rsqrt(tl.sum(q * q) / H + eps) * (1.0 + qw)).to(dtype)
        products = (kn.to(tl.float32) * qn.to(tl.float32)).to(dtype)
        result = tl.sum(products.to(tl.float32))
    else:
        scalar = tl.load(previous + t * HC + group).to(tl.float32)
        if STAGE == 1:
            result = scalar / tl.sqrt(float(H))
        elif STAGE == 2:
            sign = tl.where(scalar < 0, -1.0, tl.where(scalar > 0, 1.0, 0.0))
            magnitude = tl.sqrt(tl.maximum(tl.abs(scalar), 1e-6)).to(dtype)
            result = sign * magnitude
        elif STAGE == 3:
            result = tl.sigmoid(scalar)
        else:
            v = tl.load(value + t * value_stride + lane, mask, 0).to(tl.float32)
            g = (scalar * v).to(dtype)
            gf = g.to(tl.float32)
            cw = tl.load(nc + offset, mask, 0).to(tl.float32)
            n = gf * tl.rsqrt(tl.sum(gf * gf) / H + eps) * (1.0 + cw)
            tl.store(gated + t * HC * H + offset, g, mask)
            tl.store(normed + t * HC * H + offset, n, mask)
    if STAGE < 4:
        # A separate launch consumes this BF16 value. A cast inside the
        # original fused kernel alone did not preserve these rounding points.
        tl.store(current + t * HC + group, result)


def _ple_gate_bf16(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, width = value.shape
    hc = hidden.shape[-1] // width
    gated, normed = torch.empty_like(hidden), torch.empty_like(hidden)
    if tokens == 0:
        return gated, normed
    scratch = [hidden.new_empty((tokens, hc)) for _ in range(4)]
    for stage in range(5):
        _ple_gate_materialized_kernel[(tokens, hc)](
            key,
            value,
            hidden,
            norm_key_w,
            norm_query_w,
            norm_conv_w,
            scratch[max(stage - 1, 0)],
            scratch[min(stage, 3)],
            gated,
            normed,
            key.stride(0),
            value.stride(0),
            eps,
            H=width,
            HC=hc,
            BLOCK=triton.next_power_of_2(width),
            STAGE=stage,
            num_warps=4,
        )
    return gated, normed


def _ple_gate_bf16_fake(key, value, hidden, norm_key_w, norm_query_w, norm_conv_w, eps):
    return torch.empty_like(hidden), torch.empty_like(hidden)


direct_register_custom_op(
    op_name="ascend_ple_gate_bf16",
    op_func=_ple_gate_bf16,
    mutates_args=[],
    fake_impl=_ple_gate_bf16_fake,
)


def ple_gate(key, value, hidden, norm_key_w, norm_query_w, norm_conv_w, eps):
    if hidden.dtype != torch.bfloat16:
        from vllm.models.qwen4_exp.common.ops.ple import ple_gate as upstream_gate

        return upstream_gate(key, value, hidden, norm_key_w, norm_query_w, norm_conv_w, eps)
    if key.dtype != hidden.dtype or value.dtype != hidden.dtype:
        raise ValueError("key, value, and hidden must have the same dtype")
    if key.stride(1) != 1 or value.stride(1) != 1 or not hidden.is_contiguous():
        raise ValueError("PLE gate requires contiguous hidden and unit key/value inner strides")
    return torch.ops.vllm.ascend_ple_gate_bf16(key, value, hidden, norm_key_w, norm_query_w, norm_conv_w, eps)


@triton.jit
def _ple_add_outer_kernel(residual, outer, token_map, C: tl.constexpr, BLOCK: tl.constexpr, MAPPED: tl.constexpr):
    row = tl.program_id(0)
    if MAPPED:
        row = tl.load(token_map + row).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offset = row * C + col
    first_sum = tl.load(residual + offset, col < C, 0).to(tl.float32)
    outer_value = tl.load(outer + offset, col < C, 0).to(tl.float32)
    tl.store(residual + offset, first_sum + outer_value, col < C)


def _preserve_aliased_outer(residual, outer):
    if outer is not None and residual.dtype == torch.bfloat16:
        if residual.untyped_storage().data_ptr() == outer.untyped_storage().data_ptr():
            return outer.clone()
    return outer


def _add_outer_after_bf16(residual, outer, token_indices):
    if outer is not None and residual.dtype == torch.bfloat16:
        rows = residual.shape[0] if token_indices is None else token_indices.numel()
        if rows:
            _ple_add_outer_kernel[(rows, triton.cdiv(residual.shape[1], PLE_BLOCK_C))](
                residual,
                outer,
                token_indices,
                C=residual.shape[1],
                BLOCK=PLE_BLOCK_C,
                MAPPED=token_indices is not None,
                num_warps=PLE_NUM_WARPS,
            )


@triton.jit
def _store_ple_conv_output(
    acc,
    residual_ptr,
    outer_residual_ptr,
    output_t,
    c_offs,
    c_mask,
    out_ok,
    C: tl.constexpr,
    FUSE_RESIDUAL: tl.constexpr,
):
    # F.conv1d materializes its output dtype before SiLU and residual additions.
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
        ple_output = (residual + conv_output).to(residual_ptr.dtype.element_ty)
        if residual_ptr.dtype.element_ty != tl.bfloat16:
            outer_residual = tl.load(
                outer_residual_ptr + output_t * C + c_offs,
                mask=c_mask,
                other=0.0,
            )
            ple_output = outer_residual.to(tl.float32) + ple_output.to(tl.float32)
        # BF16: store the first sum before a second launch adds outer.
    tl.store(
        residual_ptr + output_t * C + c_offs,
        ple_output,
        mask=c_mask,
    )


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
):
    t = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

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

    _store_ple_conv_output(acc, residual_ptr, outer_residual_ptr, output_t, c_offs, c_mask, out_ok, C, FUSE_RESIDUAL)


@triton.jit(do_not_specialize=["has_token_map"])
def _ple_conv_decode_kernel(
    x_ptr,
    state_ptr,
    w_ptr,
    residual_ptr,
    outer_residual_ptr,
    state_idx_ptr,
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
    DILATION: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    HAS_INIT: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
    FUSE_RESIDUAL: tl.constexpr,
):
    t = tl.program_id(0)
    c_offs = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    output_t = tl.load(token_idx_ptr + t, mask=has_token_map, other=t).to(tl.int64)
    sid = tl.load(state_idx_ptr + t * state_idx_stride).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    sid_safe = tl.where(state_ok, sid, 0)
    if HAS_INIT:
        has_init = tl.load(has_init_ptr + t, mask=state_ok, other=0) != 0
    else:
        has_init = state_ok
    read_state = state_ok & has_init
    base_state = state_ptr + sid_safe * state_bs

    acc = tl.zeros([BLOCK_C], tl.float32)
    for k in tl.static_range(0, KERNEL_SIZE):
        h = DILATION * k
        if h < STATE_LEN:
            tap = tl.load(
                base_state + h * state_ws + c_offs * state_cs,
                mask=c_mask & read_state,
                other=0.0,
            ).to(tl.float32)
        else:
            tap = tl.load(
                x_ptr + output_t * C + c_offs,
                mask=c_mask & state_ok,
                other=0.0,
            ).to(tl.float32)
        weight = tl.load(
            w_ptr + k * C + c_offs,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)
        acc += weight * tap

    _store_ple_conv_output(acc, residual_ptr, outer_residual_ptr, output_t, c_offs, c_mask, state_ok, C, FUSE_RESIDUAL)

    # Decode has one program per request and channel block, so state writeback
    # can share the convolution launch.
    decode_input = tl.load(
        x_ptr + output_t * C + c_offs,
        mask=c_mask & state_ok,
        other=0.0,
    )
    for i in tl.static_range(0, STATE_LEN):
        if i < STATE_LEN - 1:
            next_state = tl.load(
                base_state + (i + 1) * state_ws + c_offs * state_cs,
                mask=c_mask & state_ok & has_init,
                other=0.0,
            )
        else:
            next_state = decode_input
        tl.store(
            base_state + i * state_ws + c_offs * state_cs,
            next_state,
            mask=c_mask & state_ok,
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


# Speculative (MTP) short convolution. Unlike decode, each request may span
# multiple speculative tokens, so we (1) binary-search query_start_loc to map a
# token back to its request, (2) roll the state window back to the last accepted
# token via slot_off, and (3) write state back in a separate kernel so every
# token reads the old state (avoids races between programs of the same request).
@triton.jit(do_not_specialize=["num_reqs", "bs_iters", "has_token_map"])
def _ple_conv_spec_kernel(
    x_ptr,
    state_ptr,
    w_ptr,
    residual_ptr,
    outer_residual_ptr,
    state_idx_ptr,
    qsl_ptr,
    num_acc_ptr,
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
    SPEC_QUERY_LEN: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
    FUSE_RESIDUAL: tl.constexpr,
):
    t = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    output_t = tl.load(token_idx_ptr + t, mask=has_token_map, other=t).to(tl.int64)

    # Locate the request containing this token via binary search over qsl.
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

    # Roll back draft state to the last accepted token.
    num_acc = tl.load(num_acc_ptr + r)
    slot_off = tl.minimum(tl.maximum(num_acc - 1, 0), SPEC_QUERY_LEN - 1).to(tl.int32)

    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    sid_safe = tl.where(state_ok, sid, 0)
    read_state = state_ok
    out_ok = in_range

    base_state = state_ptr + sid_safe * state_bs
    acc = tl.zeros([BLOCK_C], tl.float32)
    for k in tl.static_range(0, KERNEL_SIZE):
        h = j + DILATION * k
        from_state = h <= STATE_LEN - 1
        state_tap = tl.load(
            base_state + (h + slot_off) * state_ws + c_offs * state_cs,
            mask=c_mask & read_state & from_state,
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

    _store_ple_conv_output(acc, residual_ptr, outer_residual_ptr, output_t, c_offs, c_mask, out_ok, C, FUSE_RESIDUAL)


@triton.jit(do_not_specialize=["has_token_map"])
def _ple_conv_spec_writeback_kernel(
    x_ptr,
    state_ptr,
    state_idx_ptr,
    qsl_ptr,
    num_acc_ptr,
    token_idx_ptr,
    has_token_map,
    state_idx_stride,
    state_bs,
    state_ws,
    state_cs,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
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
    num_acc = tl.load(num_acc_ptr + r)
    # Use the same rollback window as the output kernel.
    slot_off = tl.minimum(tl.maximum(num_acc - 1, 0), SPEC_QUERY_LEN - 1).to(tl.int32)
    shift = 1

    src_ok = state_ok
    base_state = state_ptr + sid * state_bs
    for i in tl.static_range(0, STATE_WIDTH):
        m = shift + i
        from_state = m <= STATE_LEN - 1
        # Short or graph-padded spec chunks preserve the unused state tail.
        do_write = i < STATE_LEN + qlen - 1
        state_value = tl.load(
            base_state + (slot_off + m) * state_ws + c_offs * state_cs,
            mask=c_mask & from_state & src_ok,
            other=0.0,
        )
        input_t = q_start + m - STATE_LEN
        input_t = tl.load(
            token_idx_ptr + input_t,
            mask=has_token_map & (~from_state) & do_write,
            other=input_t,
        ).to(tl.int64)
        input_value = tl.load(
            x_ptr + input_t * C + c_offs,
            mask=c_mask & (~from_state) & do_write,
            other=0.0,
        )
        value = tl.where(from_state, state_value, input_value)
        tl.store(
            base_state + i * state_ws + c_offs * state_cs,
            value,
            mask=c_mask & do_write,
        )


def ple_conv_spec(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    outer_residual: torch.Tensor | None,
    *,
    dilation: int,
    query_start_loc: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    has_initial_states: torch.Tensor | None = None,
    spec_query_len: int = 1,
    token_indices: torch.Tensor | None = None,
) -> None:
    """Run speculative short conv with channel-contiguous weights and a
    separate (race-free) state writeback."""
    if query_start_loc is None or num_accepted_tokens is None:
        raise ValueError("query_start_loc and num_accepted_tokens are required for spec decode")
    outer_residual = _preserve_aliased_outer(residual, outer_residual)
    BLOCK_C = PLE_BLOCK_C
    T, C = inputs.shape
    K = conv_weights.shape[1]
    state_len = (K - 1) * dilation
    state_width = state_len + spec_query_len - 1
    if token_indices is not None:
        T = token_indices.numel()
    if conv_state.shape[2] != C or conv_state.shape[1] < state_width:
        raise ValueError(
            f"conv_state must have shape [slots, window, channels], with channels={C} and window >= {state_width}"
        )
    state_bs, state_ws, state_cs = conv_state.stride()
    num_reqs = state_indices.numel()
    binary_search_iters = max(num_reqs, 1).bit_length()

    state_idx_stride = state_indices.stride(0)
    weights_by_tap = conv_weights.transpose(0, 1).contiguous()

    _ple_conv_spec_kernel[(T, triton.cdiv(C, BLOCK_C))](
        inputs,
        conv_state,
        weights_by_tap,
        residual,
        outer_residual,
        state_indices,
        query_start_loc,
        num_accepted_tokens,
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
        SPEC_QUERY_LEN=spec_query_len,
        NULL_STATE_ID=NULL_BLOCK_ID,
        FUSE_RESIDUAL=outer_residual is not None,
        num_warps=PLE_NUM_WARPS,
    )
    _ple_conv_spec_writeback_kernel[(num_reqs, triton.cdiv(C, BLOCK_C))](
        inputs,
        conv_state,
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        token_indices if token_indices is not None else state_indices,
        token_indices is not None,
        state_idx_stride,
        state_bs,
        state_ws,
        state_cs,
        C=C,
        BLOCK_C=BLOCK_C,
        STATE_LEN=state_len,
        SPEC_QUERY_LEN=spec_query_len,
        STATE_WIDTH=state_width,
        NULL_STATE_ID=NULL_BLOCK_ID,
        num_warps=PLE_NUM_WARPS,
    )
    _add_outer_after_bf16(residual, outer_residual, token_indices)


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
    outer_residual = _preserve_aliased_outer(residual, outer_residual)
    BLOCK_C = PLE_BLOCK_C
    T, C = inputs.shape
    K = conv_weights.shape[1]
    state_len = (K - 1) * dilation
    if token_indices is not None:
        T = token_indices.numel()
    if conv_state.shape[2] != C or conv_state.shape[1] < state_len:
        raise ValueError(
            f"conv_state must have shape [slots, window, channels], with channels={C} and window >= {state_len}"
        )
    state_bs, state_ws, state_cs = conv_state.stride()

    if query_start_loc is None or has_initial_states is None:
        raise ValueError("query_start_loc and has_initial_states are required for prefill")
    num_reqs = state_indices.numel()
    binary_search_iters = max(num_reqs, 1).bit_length()

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
    _add_outer_after_bf16(residual, outer_residual, token_indices)


def ple_conv_decode(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    outer_residual: torch.Tensor | None,
    *,
    dilation: int,
    query_start_loc: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    has_initial_states: torch.Tensor | None = None,
    spec_query_len: int = 1,
    token_indices: torch.Tensor | None = None,
) -> None:
    """Run decode with channel-contiguous weights and fused state writeback."""
    outer_residual = _preserve_aliased_outer(residual, outer_residual)
    T, C = inputs.shape
    if token_indices is not None:
        T = token_indices.numel()
    kernel_size = conv_weights.shape[1]
    state_len = (kernel_size - 1) * dilation
    if conv_state.shape[2] != C or conv_state.shape[1] < state_len:
        raise ValueError(
            f"conv_state must have shape [slots, window, channels], with channels={C} and window >= {state_len}"
        )
    state_bs, state_ws, state_cs = conv_state.stride()
    weights_by_tap = conv_weights.transpose(0, 1).contiguous()
    _ple_conv_decode_kernel[(T, triton.cdiv(C, PLE_BLOCK_C))](
        inputs,
        conv_state,
        weights_by_tap,
        residual,
        outer_residual,
        state_indices,
        has_initial_states if has_initial_states is not None else state_indices,
        token_indices if token_indices is not None else state_indices,
        token_indices is not None,
        state_indices.stride(0),
        state_bs,
        state_ws,
        state_cs,
        C=C,
        BLOCK_C=PLE_BLOCK_C,
        STATE_LEN=state_len,
        DILATION=dilation,
        KERNEL_SIZE=kernel_size,
        HAS_INIT=has_initial_states is not None,
        NULL_STATE_ID=NULL_BLOCK_ID,
        FUSE_RESIDUAL=outer_residual is not None,
        num_warps=PLE_NUM_WARPS,
    )
    _add_outer_after_bf16(residual, outer_residual, token_indices)
