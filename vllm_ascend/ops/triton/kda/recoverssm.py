# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA RecoverSSM primitives for Ascend Triton.

Adapted from vLLM's Kimi-K3 RecoverSSM implementation at 45e6e6f0fba.
The caller supplies unique positive state indices; block zero is padding.
These primitives do not enable ReplaySSM in the model runner.
"""

import torch
from vllm.triton_utils import tl, triton

# A5 measurements favor 64-value tiles over repeated 8-value programs at
# serving batch sizes. Larger 128-value tiles fail the tested compiler.
RECOVERSSM_VALUE_TILE = 64


@triton.jit
def _kda_l2_normalize(vector):
    # AscendC normalizes nonzero vectors without the CUDA kernel's +1e-6
    # squared-norm bias. The difference matters for trained small-norm Q/K.
    norm_square = tl.sum(vector * vector)
    inverse = tl.rsqrt(tl.where(norm_square > 0.0, norm_square, 1.0))
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
        return lower_bound * tl.sigmoid(A * gate_input)
    softplus_gate = tl.where(
        gate_input > 20.0,
        gate_input,
        tl.log(1.0 + tl.exp(gate_input)),
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

    state *= tl.exp(gate)[None, :]
    correction = v - tl.sum(state * normalized_k[None, :], axis=1)
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
    SPEC_QUERY_LEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(tl.int64)
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    if state_idx <= null_block_id:
        for token_offset in range(query_len):
            token_valid = token_offset < query_len
            tl.store(
                out_ptr + (bos + token_offset) * stride_out_token + pid_h * V + offs_v,
                tl.zeros([BV], dtype=tl.float32),
                mask=token_valid & mask_v,
            )
        return

    # Keep the checkpoint read-only while verifying the entire candidate chain.
    state_ptrs = (
        state_ptr
        + state_idx * stride_state_block
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    state = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)
    A = tl.exp(tl.load(A_log_ptr + pid_h).to(tl.float32))

    for token_offset in range(query_len):
        token_valid = token_offset < query_len
        token = bos + token_offset
        q = tl.load(
            q_ptr + token * stride_q_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + token * stride_k_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + token * stride_v_token + pid_h * V + offs_v,
            mask=token_valid & mask_v,
            other=0.0,
        ).to(tl.float32)
        raw_g = tl.load(
            raw_g_ptr + token * stride_g_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
        ).to(tl.float32)
        raw_beta = tl.load(
            raw_beta_ptr + token * stride_beta_token + pid_h,
            mask=token_valid,
            other=0.0,
        ).to(tl.float32)

        q = _kda_l2_normalize(q) * (K**-0.5)
        dt_bias = tl.load(dt_bias_ptr + pid_h * K + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        updated_state, correction = _kda_recurrent_step(
            state,
            k,
            v,
            raw_g,
            raw_beta,
            dt_bias,
            A,
            lower_bound,
            USE_LOWER_BOUND,
        )
        state = tl.where(token_valid, updated_state, state)

        out = tl.sum(state * q[None, :], axis=1)
        tl.store(
            out_ptr + token * stride_out_token + pid_h * V + offs_v,
            out,
            mask=token_valid & mask_v,
        )

        # Store vector records instead of a full recurrent matrix per token.
        correction_ptr = (
            correction_cache_ptr
            + state_idx * stride_correction_block
            + pid_h * stride_correction_head
            + token_offset * stride_correction_pos
        )
        tl.store(
            correction_ptr + offs_v * stride_correction_dim,
            correction,
            mask=token_valid & mask_v,
        )
        if pid_v == 0:
            kg_ptr = kg_cache_ptr + state_idx * stride_kg_block + pid_h * stride_kg_head + token_offset * stride_kg_pos
            tl.store(
                kg_ptr + offs_k * stride_kg_dim,
                k,
                mask=token_valid & mask_k,
            )
            tl.store(
                kg_ptr + (K + offs_k) * stride_kg_dim,
                raw_g,
                mask=token_valid & mask_k,
            )


@triton.jit
def _commit_kernel(
    checkpoint,
    correction,
    kg,
    a_log,
    dt_bias,
    indices,
    query_starts,
    accepted,
    lower_bound,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    row, request, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    block = tl.load(indices + request).to(tl.int64)
    if block <= 0:
        return
    query_len = tl.load(query_starts + request + 1) - tl.load(query_starts + request)
    count = tl.minimum(tl.maximum(tl.load(accepted + request), 0), query_len)
    if count == 0:
        return
    keys = tl.arange(0, BK)
    values = row * BV + tl.arange(0, BV)
    mask = (values[:, None] < V) & (keys[None, :] < K)
    state_ptr = checkpoint + ((block * H + head) * V + values[:, None]) * K + keys[None, :]
    initial = tl.load(state_ptr, mask, other=0).to(tl.float32)
    a = tl.exp(tl.load(a_log + head).to(tl.float32))
    bias = tl.load(dt_bias + head * K + keys, keys < K, other=0).to(tl.float32)
    decay_acc = tl.full((BK,), 1, tl.float32)
    update_acc = tl.zeros((BV, BK), tl.float32)
    # Traverse only the accepted prefix backwards, accumulating its affine
    # updates before applying them to the original checkpoint once.
    for reverse in range(count):
        token = count - reverse - 1
        record = (block * H + head) * L + token
        key = tl.load(kg + record * (2 * K) + keys, keys < K, other=0).to(tl.float32)
        raw_gate = tl.load(kg + record * (2 * K) + K + keys, keys < K, other=0).to(tl.float32)
        delta = tl.load(correction + record * V + values, values < V, other=0).to(tl.float32)
        key = _kda_l2_normalize(key)
        decay = tl.exp(_kda_gate(raw_gate, bias, a, lower_bound, USE_LOWER_BOUND))
        update_acc += delta[:, None] * key[None, :] * decay_acc[None, :]
        decay_acc *= decay
    tl.store(state_ptr, initial * decay_acc[None, :] + update_acc, mask)


def _validate_cache(checkpoint, correction, kg, a_log, dt_bias, indices, query_starts):
    if checkpoint.ndim != 4:
        raise ValueError("checkpoint must have shape [blocks, heads, value_dim, key_dim]")
    blocks, heads, value_dim, key_dim = checkpoint.shape
    if min(blocks, heads, value_dim, key_dim) <= 0:
        raise ValueError("checkpoint dimensions must be positive")
    if checkpoint.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("checkpoint must use float32 or bfloat16")
    if correction.ndim != 4 or correction.shape[:2] != (blocks, heads):
        raise ValueError("correction must have shape [blocks, heads, capacity, value_dim]")
    capacity = correction.shape[2]
    if capacity <= 0 or correction.shape[3] != value_dim or correction.dtype != torch.float32:
        raise ValueError("correction needs positive capacity, matching value_dim, and float32")
    if kg.shape != (blocks, heads, capacity, 2 * key_dim):
        raise ValueError("key/gate records have incompatible shape")
    if kg.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("key/gate records must use an activation dtype")
    if a_log.numel() != heads or dt_bias.numel() != heads * key_dim:
        raise ValueError("gate parameters have incompatible shape")
    for name, tensor in (("state_indices", indices), ("query_start_loc", query_starts)):
        if tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be a one-dimensional integer tensor")
    if query_starts.numel() != indices.numel() + 1:
        raise ValueError("query_start_loc must have one more entry than state_indices")
    tensors = (checkpoint, correction, kg, a_log, dt_bias, indices, query_starts)
    if any(t.device != checkpoint.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("RecoverSSM caches, parameters, and metadata must be contiguous")
    return heads, key_dim, value_dim, capacity


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
    """Verify packed sequences, leaving the checkpoint unchanged.

    Metadata values must describe valid, nonoverlapping sequences with lengths
    at most spec_query_len, and unique state indices in the cache range. Block
    zero denotes padding. Values are not copied to the host for validation.
    GLM's [1, 1, heads, 1] A_log layout is accepted.
    """
    heads, key_dim, value_dim, capacity = _validate_cache(
        checkpoint_state, correction_cache, kg_cache, A_log, dt_bias, state_indices, query_start_loc
    )
    if spec_query_len != capacity:
        raise ValueError("spec_query_len must match record capacity")
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2:] != (heads, key_dim):
        raise ValueError("q must have shape [1, tokens, heads, key_dim]")
    tokens = q.shape[1]
    if k.shape != q.shape or raw_g.shape != q.shape:
        raise ValueError("key and gate must match query shape")
    if v.shape != (1, tokens, heads, value_dim) or raw_beta.shape != (1, tokens, heads):
        raise ValueError("value or beta shape is incompatible")
    if tokens > state_indices.numel() * capacity:
        raise ValueError("input exceeds record capacity")
    if any(t.stride()[2:] != (key_dim, 1) for t in (q, k, raw_g)):
        raise ValueError("query, key, and gate heads must be contiguous")
    if v.stride()[2:] != (value_dim, 1) or raw_beta.stride(2) != 1:
        raise ValueError("value and beta heads must be contiguous")
    if out is None:
        out = torch.empty_like(v)
    if out.shape != v.shape or out.dtype != v.dtype or out.stride()[2:] != (value_dim, 1):
        raise ValueError("output must match value shape, dtype, and head layout")
    if any(t.device != checkpoint_state.device for t in (q, k, v, raw_g, raw_beta, out)):
        raise ValueError("all tensors must be on the same device")
    if any(t.dtype != kg_cache.dtype for t in (q, k, v, raw_g, raw_beta)):
        raise ValueError("inputs must match key/gate record dtype")
    if checkpoint_state.device.type != "npu":
        raise ValueError("RecoverSSM requires an NPU device")
    if tokens == 0:
        return out
    # Split the value dimension to keep each NPU program's matrix tile small.
    block_v = min(triton.next_power_of_2(value_dim), RECOVERSSM_VALUE_TILE)
    _kda_recoverssm_verify_kernel[(triton.cdiv(value_dim, block_v), state_indices.numel(), heads)](
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log.view(-1),
        dt_bias,
        checkpoint_state,
        correction_cache,
        kg_cache,
        out,
        query_start_loc,
        state_indices,
        lower_bound if lower_bound is not None else 0.0,
        0,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        raw_g.stride(1),
        raw_beta.stride(1),
        *checkpoint_state.stride(),
        *correction_cache.stride(),
        *kg_cache.stride(),
        out.stride(1),
        query_start_loc.stride(0),
        state_indices.stride(0),
        K=key_dim,
        V=value_dim,
        BK=triton.next_power_of_2(key_dim),
        BV=block_v,
        SPEC_QUERY_LEN=capacity,
        USE_LOWER_BOUND=lower_bound is not None,
    )
    return out


def kda_recoverssm_commit(
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kg_cache: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> None:
    """Rebuild accepted prefixes into the same checkpoint slots.

    Counts refer to verified positions (including the leading target token),
    not only accepted drafts. This primitive does not compact convolution
    history or create prefix-cache boundary checkpoints.
    """
    heads, key_dim, value_dim, capacity = _validate_cache(
        checkpoint_state, correction_cache, kg_cache, A_log, dt_bias, state_indices, query_start_loc
    )
    if (
        num_accepted_tokens.shape != state_indices.shape
        or num_accepted_tokens.dtype not in (torch.int32, torch.int64)
        or not num_accepted_tokens.is_contiguous()
        or num_accepted_tokens.device != checkpoint_state.device
    ):
        raise ValueError("accepted counts must be contiguous device integers, one per request")
    if checkpoint_state.device.type != "npu":
        raise ValueError("RecoverSSM requires an NPU device")
    if state_indices.numel() == 0:
        return
    block_v = min(triton.next_power_of_2(value_dim), RECOVERSSM_VALUE_TILE)
    _commit_kernel[(triton.cdiv(value_dim, block_v), state_indices.numel(), heads)](
        checkpoint_state,
        correction_cache,
        kg_cache,
        A_log.view(-1),
        dt_bias,
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        lower_bound if lower_bound is not None else 0.0,
        H=heads,
        K=key_dim,
        V=value_dim,
        L=capacity,
        BK=triton.next_power_of_2(key_dim),
        BV=block_v,
        USE_LOWER_BOUND=lower_bound is not None,
    )
