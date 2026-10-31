# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-token GDN decode with fused normalization, gates and state update."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

_VALUE_TILE_SIZE = 64
_L2_NORM_EPS = 1e-6
_SOFTPLUS_THRESHOLD = 20.0


@triton.jit
def _round_bfloat16(value):
    # Keep the separate operators' BF16 rounding boundary in registers.
    # A float32 -> bfloat16 -> float32 cast can be folded by the NPU compiler.
    bits = value.to(tl.int32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & -65536
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _fused_gdn_decode_kernel(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    state,
    indices,
    query_start_loc,
    out,
    num_tokens,
    num_requests: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    q_token_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    v_token_stride: tl.constexpr,
    v_head_stride: tl.constexpr,
    a_token_stride: tl.constexpr,
    a_head_stride: tl.constexpr,
    b_token_stride: tl.constexpr,
    b_head_stride: tl.constexpr,
    state_token_stride: tl.constexpr,
    indices_stride: tl.constexpr,
    log_stride: tl.constexpr,
    bias_stride: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    EPS: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    num_value_tiles = tl.cdiv(V, BV)
    first_token = tl.load(query_start_loc)
    last_token = tl.load(query_start_loc + num_requests)
    for task in range(tl.program_id(0), num_tokens * HV * num_value_tiles, tl.num_programs(0)):
        value_tile = task % num_value_tiles
        token_head = task // num_value_tiles
        token, head = token_head // HV, token_head % HV
        key_head = head // (HV // H)
        slot = tl.load(indices + token * indices_stride).to(tl.int64)
        key_offsets = tl.arange(0, BK)
        value_offsets = value_tile * BV + tl.arange(0, BV)
        key_mask = key_offsets < K
        value_mask = value_offsets < V
        output_offsets = (token * HV + head) * V + value_offsets
        if (slot >= 0) & (token >= first_token) & (token < last_token):
            query = tl.load(q + token * q_token_stride + key_head * q_head_stride + key_offsets, key_mask, other=0).to(
                tl.float32
            )
            key = tl.load(k + token * k_token_stride + key_head * k_head_stride + key_offsets, key_mask, other=0).to(
                tl.float32
            )
            # Preserve the intermediate input-dtype rounding of l2norm_fwd.
            query = _round_bfloat16(query * tl.rsqrt(tl.sum(query * query, 0) + EPS))
            key = _round_bfloat16(key * tl.rsqrt(tl.sum(key * key, 0) + EPS))
            query *= K**-0.5
            gate_input = tl.load(a + token * a_token_stride + head * a_head_stride).to(tl.float32)
            gate_bias = tl.load(dt_bias + head * bias_stride).to(tl.float32)
            log_decay = tl.load(A_log + head * log_stride).to(tl.float32)
            gate_input += gate_bias
            softplus = tl.where(gate_input <= THRESHOLD, tl.log(1.0 + tl.exp(gate_input)), gate_input)
            decay = tl.exp(-tl.exp(log_decay) * softplus)
            beta_input = tl.load(b + token * b_token_stride + head * b_head_stride).to(tl.float32)
            # The separate gating op stores sigmoid(b) in b.dtype.
            beta = _round_bfloat16(tl.sigmoid(beta_input))
            state_offsets = slot * state_token_stride + head * V * K + value_offsets[:, None] * K + key_offsets[None, :]
            state_mask = value_mask[:, None] & key_mask[None, :]
            hidden = tl.load(state + state_offsets, state_mask, other=0).to(tl.float32)
            hidden *= decay
            value = tl.load(v + token * v_token_stride + head * v_head_stride + value_offsets, value_mask, other=0).to(
                tl.float32
            )
            delta = (value - tl.sum(hidden * key[None, :], 1)) * beta
            hidden += delta[:, None] * key[None, :]
            output = tl.sum(hidden * query[None, :], 1)
            tl.store(state + state_offsets, hidden, state_mask)
            tl.store(out + output_offsets, output, value_mask)
        else:
            # Graph padding must not read/write the persistent state pool.
            tl.store(out + output_offsets, 0.0, value_mask)


def fused_gdn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Update distinct state slots for one token per request.

    Q/K are [T, H, K], V is [T, HV, V], gates are [T, HV], and
    state is [pool, HV, V, K]. Negative indices are padding; slot zero is valid.
    Query boundaries exclude graph padding tokens. The caller must not map
    two live requests to the same state slot.
    """
    num_tokens, num_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[1:]
    if any(x.dtype != torch.bfloat16 for x in (q, k, v, a, b)):
        raise ValueError("Fused GDN decode requires BF16 query/key/value and gates")
    if k.shape != q.shape or v.shape[0] != num_tokens or value_heads % num_heads:
        raise ValueError("Incompatible GDN query/key/value shapes")
    if a.shape != (num_tokens, value_heads) or b.shape != a.shape:
        raise ValueError("GDN gates must have shape [tokens, value_heads]")
    if state.shape[1:] != (value_heads, value_dim, key_dim) or (
        state.stride(1) != value_dim * key_dim
        or state.stride(2) != key_dim
        or state.stride(3) != 1
        or state.stride(0) < value_heads * value_dim * key_dim
    ):
        raise ValueError("GDN state must have non-overlapping slots with contiguous [value_heads, value_dim, key_dim]")
    if state_indices.ndim != 1 or state_indices.numel() != num_tokens:
        raise ValueError("GDN decode requires one state index per token")
    if (
        query_start_loc.ndim != 1
        or query_start_loc.numel() < (2 if num_tokens else 1)
        or query_start_loc.stride(0) != 1
    ):
        raise ValueError("GDN decode requires contiguous request boundaries")
    if any(x.stride(-1) != 1 for x in (q, k, v)):
        raise ValueError("GDN query/key/value must have contiguous head dimensions")
    if A_log.shape != (value_heads,) or dt_bias.shape != A_log.shape:
        raise ValueError("GDN gate parameters must have shape [value_heads]")
    output = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    if num_tokens == 0:
        return output
    num_tasks = num_tokens * value_heads * triton.cdiv(value_dim, _VALUE_TILE_SIZE)
    _fused_gdn_decode_kernel[(min(get_vectorcore_num(), num_tasks),)](
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state,
        state_indices,
        query_start_loc,
        output,
        num_tokens,
        query_start_loc.numel() - 1,
        num_heads,
        value_heads,
        key_dim,
        value_dim,
        *q.stride()[:2],
        *k.stride()[:2],
        *v.stride()[:2],
        *a.stride(),
        *b.stride(),
        state.stride(0),
        state_indices.stride(0),
        A_log.stride(0),
        dt_bias.stride(0),
        BK=triton.next_power_of_2(key_dim),
        BV=_VALUE_TILE_SIZE,
        EPS=_L2_NORM_EPS,
        THRESHOLD=_SOFTPLUS_THRESHOLD,
        num_warps=4,
        num_stages=1,
        multibuffer=False,
        enable_fp_fusion=False,
    )
    return output
