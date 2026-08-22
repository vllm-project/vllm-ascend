# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Head-tiled Qwen3.5 GDN decode pipeline.

The kernel consumes the two input projections in their native Qwen3.5 layout
and processes one GQA group per program. A group contains one Q/K head and two
V/Z heads. Keeping that unit intact removes the whole-tensor layout barrier
between projection, convolution, gating, recurrent update, and gated RMSNorm.
"""

from functools import lru_cache

import torch
from vllm.triton_utils import tl, triton

QWEN35_NUM_QK_HEADS = 16
QWEN35_NUM_V_HEADS = 32
QWEN35_HEAD_DIM = 128
QWEN35_HIDDEN_DIM = 4096
QWEN35_CONV_KERNEL_SIZE = 4
QWEN35_QK_DIM = QWEN35_NUM_QK_HEADS * QWEN35_HEAD_DIM
QWEN35_V_DIM = QWEN35_NUM_V_HEADS * QWEN35_HEAD_DIM
QWEN35_CONV_DIM = QWEN35_QK_DIM * 2 + QWEN35_V_DIM
QWEN35_QKVZ_DIM = QWEN35_CONV_DIM + QWEN35_V_DIM


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _conv_and_update(
    raw,
    conv_weight,
    conv_state,
    cache_index,
    channel_offsets,
    state_stride,
    valid,
    dim: tl.constexpr,
):
    state_base = cache_index * state_stride + channel_offsets
    state0 = tl.load(conv_state + state_base, mask=valid, other=0.0)
    state1 = tl.load(conv_state + state_base + dim, mask=valid, other=0.0)
    state2 = tl.load(conv_state + state_base + 2 * dim, mask=valid, other=0.0)
    weight0 = tl.load(conv_weight + channel_offsets)
    weight1 = tl.load(conv_weight + dim + channel_offsets)
    weight2 = tl.load(conv_weight + 2 * dim + channel_offsets)
    weight3 = tl.load(conv_weight + 3 * dim + channel_offsets)

    acc = state0.to(tl.float32) * weight0.to(tl.float32)
    acc += state1.to(tl.float32) * weight1.to(tl.float32)
    acc += state2.to(tl.float32) * weight2.to(tl.float32)
    acc += raw.to(tl.float32) * weight3.to(tl.float32)

    tl.store(conv_state + state_base, state1, mask=valid)
    tl.store(conv_state + state_base + dim, state2, mask=valid)
    tl.store(conv_state + state_base + 2 * dim, raw, mask=valid)
    return _silu(acc).to(raw.dtype)


@triton.jit
def _l2_norm(x, eps: tl.constexpr):
    x_fp32 = x.to(tl.float32)
    inv_norm = tl.rsqrt(tl.sum(x_fp32 * x_fp32, axis=0) + eps)
    return (x_fp32 * inv_norm).to(x.dtype)


@triton.jit
def _recurrent_head(
    q,
    k,
    v,
    z,
    beta,
    log_decay,
    state,
    state_index,
    value_head,
    norm_weight,
    output,
    valid,
    head_dim: tl.constexpr,
    num_v_heads: tl.constexpr,
    scale: tl.constexpr,
    norm_eps: tl.constexpr,
):
    rows = tl.arange(0, head_dim)[:, None]
    cols = tl.arange(0, head_dim)[None, :]
    state_offset = (state_index * num_v_heads + value_head) * head_dim * head_dim + rows * head_dim + cols
    state_tile = tl.load(state + state_offset, mask=valid, other=0.0).to(tl.float32)
    state_tile *= tl.exp(log_decay)

    k_fp32 = k.to(tl.float32)
    prediction = tl.sum(state_tile * k_fp32[None, :], axis=1)
    delta = (v.to(tl.float32) - prediction) * beta
    state_tile += delta[:, None] * k_fp32[None, :]

    q_fp32 = q.to(tl.float32) * scale
    recurrent_out = tl.sum(state_tile * q_fp32[None, :], axis=1)
    tl.store(state + state_offset, state_tile, mask=valid)

    # Match the unfused path's model-dtype recurrent output before
    # RMSNormGated, then perform normalization in FP32.
    recurrent_model_dtype = recurrent_out.to(q.dtype)
    recurrent_fp32 = recurrent_model_dtype.to(tl.float32)
    variance = tl.sum(recurrent_fp32 * recurrent_fp32, axis=0) / head_dim
    normed = recurrent_fp32 * tl.rsqrt(variance + norm_eps)
    normed *= tl.load(norm_weight + tl.arange(0, head_dim)).to(tl.float32)
    normed *= _silu(z.to(tl.float32))
    output_offset = value_head * head_dim + tl.arange(0, head_dim)
    output_value = tl.where(valid, normed, 0.0)
    tl.store(output + output_offset, output_value)


@triton.jit
def qwen35_gdn_decode_tile_kernel(
    projected_qkvz,
    projected_ba,
    conv_weight,
    conv_state,
    cache_indices,
    query_start_loc,
    a_log,
    dt_bias,
    recurrent_state,
    recurrent_state_indices,
    norm_weight,
    output,
    tile_token_indices,
    tile_qk_heads,
    use_descriptors: tl.constexpr,
    l2_eps: tl.constexpr,
    norm_eps: tl.constexpr,
    scale: tl.constexpr,
    head_dim: tl.constexpr,
    num_qk_heads: tl.constexpr,
    num_v_heads: tl.constexpr,
    qk_dim: tl.constexpr,
    v_dim: tl.constexpr,
    conv_dim: tl.constexpr,
):
    tile_id = tl.program_id(0)
    if use_descriptors:
        token_index = tl.load(tile_token_indices + tile_id)
        qk_head = tl.load(tile_qk_heads + tile_id)
    else:
        token_index = 0
        qk_head = tile_id
    # Full graph decode pads requests to a capture batch. Padded rows have a
    # repeated query offset and must not update either recurrent state cache.
    valid = tl.load(query_start_loc + token_index + 1) > tl.load(query_start_loc + token_index)
    offsets = tl.arange(0, head_dim)
    value_head0 = qk_head * 2
    value_head1 = value_head0 + 1

    projected_base = token_index * (conv_dim + v_dim)
    q_offsets = projected_base + qk_head * head_dim + offsets
    k_offsets = projected_base + qk_dim + qk_head * head_dim + offsets
    v0_offsets = projected_base + qk_dim * 2 + value_head0 * head_dim + offsets
    v1_offsets = projected_base + qk_dim * 2 + value_head1 * head_dim + offsets
    z0_offsets = projected_base + conv_dim + value_head0 * head_dim + offsets
    z1_offsets = projected_base + conv_dim + value_head1 * head_dim + offsets

    q_raw = tl.load(projected_qkvz + q_offsets)
    k_raw = tl.load(projected_qkvz + k_offsets)
    v0_raw = tl.load(projected_qkvz + v0_offsets)
    v1_raw = tl.load(projected_qkvz + v1_offsets)
    z0 = tl.load(projected_qkvz + z0_offsets)
    z1 = tl.load(projected_qkvz + z1_offsets)

    cache_index = tl.load(cache_indices + token_index).to(tl.int64)
    cache_index = tl.where(valid, cache_index, 0)
    conv_state_stride = 3 * conv_dim
    q_channel_offsets = qk_head * head_dim + offsets
    k_channel_offsets = qk_dim + qk_head * head_dim + offsets
    v0_channel_offsets = qk_dim * 2 + value_head0 * head_dim + offsets
    v1_channel_offsets = qk_dim * 2 + value_head1 * head_dim + offsets
    q = _conv_and_update(
        q_raw,
        conv_weight,
        conv_state,
        cache_index,
        q_channel_offsets,
        conv_state_stride,
        valid,
        conv_dim,
    )
    k = _conv_and_update(
        k_raw,
        conv_weight,
        conv_state,
        cache_index,
        k_channel_offsets,
        conv_state_stride,
        valid,
        conv_dim,
    )
    v0 = _conv_and_update(
        v0_raw,
        conv_weight,
        conv_state,
        cache_index,
        v0_channel_offsets,
        conv_state_stride,
        valid,
        conv_dim,
    )
    v1 = _conv_and_update(
        v1_raw,
        conv_weight,
        conv_state,
        cache_index,
        v1_channel_offsets,
        conv_state_stride,
        valid,
        conv_dim,
    )

    q = _l2_norm(q, l2_eps)
    k = _l2_norm(k, l2_eps)

    ba_base = token_index * 2 * num_v_heads
    b0_raw = tl.load(projected_ba + ba_base + value_head0)
    b1_raw = tl.load(projected_ba + ba_base + value_head1)
    b0 = b0_raw.to(tl.float32)
    b1 = b1_raw.to(tl.float32)
    a0 = tl.load(projected_ba + ba_base + num_v_heads + value_head0).to(tl.float32)
    a1 = tl.load(projected_ba + ba_base + num_v_heads + value_head1).to(tl.float32)
    dt0 = a0 + tl.load(dt_bias + value_head0).to(tl.float32)
    dt1 = a1 + tl.load(dt_bias + value_head1).to(tl.float32)
    softplus0 = tl.where(dt0 <= 20.0, tl.log(1.0 + tl.exp(dt0)), dt0)
    softplus1 = tl.where(dt1 <= 20.0, tl.log(1.0 + tl.exp(dt1)), dt1)
    log_decay0 = -tl.exp(tl.load(a_log + value_head0).to(tl.float32)) * softplus0
    log_decay1 = -tl.exp(tl.load(a_log + value_head1).to(tl.float32)) * softplus1
    beta0 = tl.sigmoid(b0).to(b0_raw.dtype).to(tl.float32)
    beta1 = tl.sigmoid(b1).to(b1_raw.dtype).to(tl.float32)

    state_index = tl.load(recurrent_state_indices + token_index).to(tl.int64)
    state_index = tl.where(valid, state_index, 0)
    token_output = output + token_index * v_dim
    _recurrent_head(
        q,
        k,
        v0,
        z0,
        beta0,
        log_decay0,
        recurrent_state,
        state_index,
        value_head0,
        norm_weight,
        token_output,
        valid,
        head_dim,
        num_v_heads,
        scale,
        norm_eps,
    )
    _recurrent_head(
        q,
        k,
        v1,
        z1,
        beta1,
        log_decay1,
        recurrent_state,
        state_index,
        value_head1,
        norm_weight,
        token_output,
        valid,
        head_dim,
        num_v_heads,
        scale,
        norm_eps,
    )


@lru_cache(maxsize=32)
def _tile_descriptors(device: str, batch: int, num_qk_heads: int):
    token_indices = torch.arange(batch, dtype=torch.int32, device=device).repeat_interleave(num_qk_heads)
    qk_heads = torch.arange(num_qk_heads, dtype=torch.int32, device=device).repeat(batch)
    return token_indices, qk_heads


def gdn_decode_tile(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    recurrent_state_indices: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    num_qk_heads: int,
    num_v_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Run a batched GDN head-tile pipeline for one-token decode inputs."""
    if num_v_heads != num_qk_heads * 2:
        raise ValueError("the current tile maps each Q/K head to exactly two V heads")
    batch = projected_qkvz.shape[0]
    qk_dim = num_qk_heads * head_dim
    v_dim = num_v_heads * head_dim
    conv_dim = qk_dim * 2 + v_dim
    qkvz_dim = conv_dim + v_dim
    if projected_qkvz.shape != (batch, qkvz_dim):
        raise ValueError(f"expected projected_qkvz [{batch}, {qkvz_dim}], got {projected_qkvz.shape}")
    if projected_ba.shape != (batch, num_v_heads * 2):
        raise ValueError(f"expected projected_ba [{batch}, {num_v_heads * 2}], got {projected_ba.shape}")
    if conv_weight.shape != (QWEN35_CONV_KERNEL_SIZE, conv_dim):
        raise ValueError(f"expected conv_weight [{QWEN35_CONV_KERNEL_SIZE}, {conv_dim}], got {conv_weight.shape}")
    if cache_indices.numel() != batch or recurrent_state_indices.numel() != batch:
        raise ValueError("cache and recurrent state indices must have one entry per token")
    if query_start_loc.numel() != batch + 1:
        raise ValueError("query_start_loc must contain one offset per token plus the terminal offset")
    if recurrent_state.dtype != torch.float32:
        raise ValueError(f"expected FP32 recurrent state, got {recurrent_state.dtype}")
    model_dtype = projected_qkvz.dtype
    if model_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported model dtype: {model_dtype}")
    model_dtype_tensors = {
        "projected_ba": projected_ba,
        "conv_weight": conv_weight,
        "conv_state": conv_state,
        "norm_weight": norm_weight,
    }
    for name, tensor in model_dtype_tensors.items():
        if tensor.dtype != model_dtype:
            raise ValueError(f"expected {name} dtype {model_dtype}, got {tensor.dtype}")

    output = torch.empty((batch, v_dim), dtype=projected_qkvz.dtype, device=projected_qkvz.device)
    use_descriptors = batch > 1
    if use_descriptors:
        tile_token_indices, tile_qk_heads = _tile_descriptors(str(projected_qkvz.device), batch, num_qk_heads)
    else:
        # The descriptor pointers are compile-time dead for the validated B=1 path.
        tile_token_indices = cache_indices
        tile_qk_heads = cache_indices
    qwen35_gdn_decode_tile_kernel[(batch * num_qk_heads,)](
        projected_qkvz,
        projected_ba,
        conv_weight,
        conv_state,
        cache_indices,
        query_start_loc,
        a_log,
        dt_bias,
        recurrent_state,
        recurrent_state_indices,
        norm_weight,
        output,
        tile_token_indices,
        tile_qk_heads,
        use_descriptors=use_descriptors,
        l2_eps=1e-6,
        norm_eps=norm_eps,
        scale=head_dim**-0.5,
        head_dim=head_dim,
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        qk_dim=qk_dim,
        v_dim=v_dim,
        conv_dim=conv_dim,
    )
    return output


def qwen35_gdn_decode_tile(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    recurrent_state_indices: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """Run the Qwen3.5-9B non-speculative one-token decode pipeline."""
    batch = projected_qkvz.shape[0] if projected_qkvz.ndim == 2 else 0
    if batch < 1 or projected_qkvz.shape != (batch, QWEN35_QKVZ_DIM):
        raise ValueError(f"expected projected_qkvz [B, {QWEN35_QKVZ_DIM}] with B >= 1, got {projected_qkvz.shape}")
    return gdn_decode_tile(
        projected_qkvz=projected_qkvz,
        projected_ba=projected_ba,
        conv_weight=conv_weight,
        conv_state=conv_state,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        a_log=a_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        recurrent_state_indices=recurrent_state_indices,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        num_qk_heads=QWEN35_NUM_QK_HEADS,
        num_v_heads=QWEN35_NUM_V_HEADS,
        head_dim=QWEN35_HEAD_DIM,
    )
