# adapted from vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# mypy: ignore-errors

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends import utils as attn_backend_utils  # type: ignore

PAD_SLOT_ID = attn_backend_utils.PAD_SLOT_ID
NULL_BLOCK_ID = getattr(attn_backend_utils, "NULL_BLOCK_ID", PAD_SLOT_ID)


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def _uses_apc_prefill_path(
    cache_indices: torch.Tensor | None,
    block_idx_first_scheduled_token: torch.Tensor | None,
    block_idx_last_scheduled_token: torch.Tensor | None,
    initial_state_idx: torch.Tensor | None,
    num_computed_tokens: torch.Tensor | None,
) -> bool:
    return (
        (cache_indices is not None and cache_indices.dim() == 2)
        or block_idx_first_scheduled_token is not None
        or block_idx_last_scheduled_token is not None
        or initial_state_idx is not None
        or num_computed_tokens is not None
    )


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    null_block_id: int = NULL_BLOCK_ID,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align: int = 0,
    metadata: Any | None = None,
    validate_data: bool = False,
):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch) int32 for the standard path, or
        (batch, max_blocks) int32 block table for APC all-mode prefill
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    null_block_id: int
        Upstream-compatible sentinel for empty APC block-table entries. The
        NPU path normalizes it to pad_slot_id before dispatch.
    block_idx_first_scheduled_token / block_idx_last_scheduled_token /
    initial_state_idx / num_computed_tokens:
        APC all-mode prefill metadata. When provided, causal_conv1d_fn routes
        to the NPU APC prefill implementation while keeping this public API
        aligned with upstream.
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim, seqlen)
    """
    forward_context = get_forward_context()
    num_decodes = 0
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is not None and isinstance(attn_metadata, dict):
        attn_metadata = next(iter(attn_metadata.values()), None)
    if attn_metadata is not None:
        num_decodes = attn_metadata.num_decodes

    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    del metadata, validate_data

    if _uses_apc_prefill_path(
        cache_indices,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
    ):
        if query_start_loc is None:
            raise RuntimeError(
                "APC causal_conv1d_fn requires query_start_loc for varlen inputs."
            )
        if cache_indices is None or cache_indices.dim() != 2:
            raise RuntimeError(
                "APC causal_conv1d_fn requires 2D cache_indices block tables."
            )
        if has_initial_state is None:
            raise RuntimeError(
                "APC causal_conv1d_fn requires has_initial_state."
            )
        if conv_states is None:
            raise RuntimeError("APC causal_conv1d_fn requires conv_states.")
        if block_idx_first_scheduled_token is None:
            raise RuntimeError(
                "APC causal_conv1d_fn requires block_idx_first_scheduled_token."
            )
        if block_idx_last_scheduled_token is None:
            raise RuntimeError(
                "APC causal_conv1d_fn requires block_idx_last_scheduled_token."
            )
        if initial_state_idx is None:
            raise RuntimeError("APC causal_conv1d_fn requires initial_state_idx.")
        if num_computed_tokens is None:
            raise RuntimeError(
                "APC causal_conv1d_fn requires num_computed_tokens."
            )
        if null_block_id != pad_slot_id:
            cache_indices = torch.where(
                cache_indices == null_block_id,
                torch.full_like(cache_indices, pad_slot_id),
                cache_indices,
            )
        return _causal_conv1d_fwd_npu(
            x=x,
            weight=weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
            pad_slot_id=pad_slot_id,
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            initial_state_idx=initial_state_idx,
            num_computed_tokens=num_computed_tokens,
            block_size_to_align=block_size_to_align,
        )

    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    out_ref = []
    out_ref_b = []
    seqlens = query_start_loc[1:] - query_start_loc[:-1]
    seqlens = seqlens.tolist()
    splits = torch.split(x, seqlens, dim=-1)
    width = weight.shape[1]
    last_width_prefill_x = extract_last_width(x, query_start_loc[num_decodes:], conv_states.shape[-1])

    if get_pcp_group().world_size > 1:
        all_last_width_prefill_x = get_pcp_group().all_gather(last_width_prefill_x.unsqueeze(0).contiguous(), 0)
        pcp_rank = get_pcp_group().rank_in_group
        if pcp_rank > 0:
            conv_states[cache_indices[num_decodes:]] = all_last_width_prefill_x[pcp_rank - 1, ...]

    for i in range(len(seqlens)):
        x_s = splits[i]
        if cache_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=conv_states[cache_indices[i]][..., : (width - 1)].unsqueeze(0),
                initial_states=conv_states[cache_indices[i]][..., : (width - 1)],
            )
        )

    if get_pcp_group().world_size > 1:
        conv_states[cache_indices[num_decodes:]] = all_last_width_prefill_x[-1, ...]
    out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=-1))
    out_ref_tensor = torch.cat(out_ref, dim=0)
    return out_ref_tensor


def extract_last_width(x, start_loc, width):
    end_loc = start_loc[1:]
    offsets = torch.arange(width, device=x.device)
    indices = end_loc.unsqueeze(1) - width + offsets.unsqueeze(0)  # (num_seqs, width)

    return x[:, indices].permute(1, 0, 2)


@triton.jit(
    do_not_specialize=[
        "batch",
        "state_len",
        "num_cache_lines",
        "stride_x_seq",
        "stride_x_token",
        "stride_conv_state_seq",
        "stride_state_indices",
        "stride_o_seq",
        "stride_o_token",
    ]
)
def _causal_conv1d_update_kernel_npu_tiled(
    # Pointers
    x_ptr,  # (batch, dim, seqlen) OR (num_tokens, dim) for varlen
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,  # (num_cache_lines, dim, state_len)
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,  # (batch + 1)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # same shape as x_ptr
    batch: tl.int32,
    dim: tl.constexpr,
    seqlen: tl.constexpr,  # max seqlen for varlen, or exact seqlen
    state_len,  # effective state_len computed in wrapper
    num_cache_lines,
    # Strides
    stride_x_seq,
    stride_x_dim: tl.constexpr,
    stride_x_token,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices,
    stride_o_seq,
    stride_o_dim: tl.constexpr,
    stride_o_token,
    # others
    pad_slot_id: tl.constexpr,
    # Meta
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,  # <= 6
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    # tiling
    BLOCK_N: tl.constexpr,  # channel tile (C_TILE)
    B_TILE: tl.constexpr,  # batch tile
    T_CHUNK: tl.constexpr,  # token chunk for state update
):
    # program ids
    pid_b = tl.program_id(0)  # batch-tile id
    pid_c = tl.program_id(1)  # channel-tile id

    # channel indices for this program
    idx_feats = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_w = idx_feats < dim

    # preload weights once per program (shared by B_TILE sequences)
    w_base = w_ptr + idx_feats * stride_w_dim
    # define to avoid "undefined" in branches
    w_col0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col4 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col5 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if KERNEL_WIDTH >= 1:
        w_col0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 2:
        w_col1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 5:
        w_col4 = tl.load(w_base + 4 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 6:
        w_col5 = tl.load(w_base + 5 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)

    # bias vector once per program
    if HAS_BIAS:
        acc_bias = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        acc_bias = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # token index vector for chunked copy
    tok_vec = tl.arange(0, T_CHUNK)  # [T_CHUNK]

    # process B_TILE sequences inside the same program instance
    for bi in tl.static_range(0, B_TILE):
        b = pid_b * B_TILE + bi  # scalar tl.int32
        lane_active = b < batch  # scalar predicate

        # -------------------------
        # APC mapping (optional)
        # -------------------------
        if IS_APC_ENABLED:
            conv_state_init = tl.load(initial_state_idx + b, mask=lane_active, other=0).to(tl.int32)
            current_last_index = tl.load(block_idx_last_scheduled_token + b, mask=lane_active, other=0).to(tl.int32)
        else:
            conv_state_init = tl.full((), 0, tl.int32)
            current_last_index = tl.full((), 0, tl.int32)

        # input cache line
        conv_states_input_coord = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + conv_state_init, mask=lane_active, other=0
        ).to(tl.int64)

        if USE_PAD_SLOT:
            lane_active = lane_active & (conv_states_input_coord != pad_slot_id)

        # -------------------------
        # varlen (optional): revise seqlen_run and state_len_run like original kernel does
        # -------------------------
        if IS_VARLEN:
            qs = tl.load(query_start_loc_ptr + b, mask=lane_active, other=0).to(tl.int64)
            qe = tl.load(query_start_loc_ptr + (b + 1), mask=lane_active, other=0).to(tl.int64)
            seqlen_run = (qe - qs).to(tl.int32)
            # revise effective state_len for shorter sequences (same formula as original)
            state_len_run = (state_len - (seqlen - seqlen_run)).to(tl.int32)
            x_offset = (qs * stride_x_token).to(tl.int64)
            o_offset = (qs * stride_o_token).to(tl.int64)
        else:
            seqlen_run = tl.full((), seqlen, tl.int32)
            state_len_run = tl.full((), state_len, tl.int32)
            x_offset = (b * stride_x_seq).to(tl.int64)
            o_offset = (b * stride_o_seq).to(tl.int64)

        # empty sequence -> skip (avoid early return because other lanes in tile)
        lane_active = lane_active & (seqlen_run > 0)

        # -------------------------
        # spec decoding offset (optional)
        # -------------------------
        if IS_SPEC_DECODING:
            conv_state_token_offset = tl.load(num_accepted_tokens_ptr + b, mask=lane_active, other=1).to(tl.int64) - 1
            shift = tl.full((), 1, tl.int32)  # sliding by 1 in spec mode
        else:
            conv_state_token_offset = tl.full((), 0, tl.int64)
            shift = seqlen_run  # normal mode shift by seqlen

        # -------------------------
        # STEP 1: read initial history cols BEFORE state update (out==x safe)
        # -------------------------
        conv_states_base = (
            conv_state_ptr + conv_states_input_coord * stride_conv_state_seq + idx_feats * stride_conv_state_dim
        )
        prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok

        # define history vectors as zeros then load conditionally
        col0 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col1 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col2 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col3 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col4 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        if KERNEL_WIDTH >= 2:
            col0 = tl.load(prior_tokens + 0 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0).to(
                tl.float16
            )
        if KERNEL_WIDTH >= 3:
            col1 = tl.load(prior_tokens + 1 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0).to(
                tl.float16
            )
        if KERNEL_WIDTH >= 4:
            col2 = tl.load(prior_tokens + 2 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0).to(
                tl.float16
            )
        if KERNEL_WIDTH >= 5:
            col3 = tl.load(prior_tokens + 3 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0).to(
                tl.float16
            )
        if KERNEL_WIDTH >= 6:
            col4 = tl.load(prior_tokens + 4 * stride_conv_state_tok, mask=lane_active & mask_w, other=0.0).to(
                tl.float16
            )

        # -------------------------
        # STEP 2: chunked state update (replaces original NP2_STATELEN x BLOCK_N big block)
        # Semantics: conv_state <- concat(old_state, x)[-state_len_run:].
        # - If seqlen_run >= state_len_run: dst[:] = x[seqlen_run - state_len_run : seqlen_run]
        # - Else: keep = state_len_run - seqlen_run,
        #         dst[0:keep] = src[shift : shift+keep], dst[keep:keep+seqlen_run] = x[0:seqlen_run]
        # -------------------------
        # output cache line
        conv_states_offset = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + current_last_index, mask=lane_active, other=0
        ).to(tl.int64)

        use_shift = seqlen_run < state_len_run
        use_tail = seqlen_run >= state_len_run

        zero_i32 = tl.full((), 0, tl.int32)
        keep_shift = tl.where(use_shift, (state_len_run - seqlen_run), zero_i32).to(tl.int32)
        tail_start = tl.where(use_tail, (seqlen_run - state_len_run), zero_i32).to(tl.int32)

        # base pointers
        state_src_base = (
            conv_state_ptr
            + conv_states_input_coord * stride_conv_state_seq
            + conv_state_token_offset * stride_conv_state_tok
            + idx_feats * stride_conv_state_dim
        )
        state_dst_base = conv_state_ptr + conv_states_offset * stride_conv_state_seq + idx_feats * stride_conv_state_dim

        x_base = x_ptr + x_offset + idx_feats * stride_x_dim

        # A) shift old state into dst[0:keep_shift)  (only when seqlen_run < state_len_run)
        for t0 in tl.static_range(0, NP2_STATELEN, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            src_tok = (dst_tok + shift).to(tl.int32)  # [T_CHUNK]
            m_tok = use_shift & (dst_tok < keep_shift) & (src_tok < state_len_run) & (dst_tok < state_len_run)
            m = (
                (lane_active & m_tok)[:, None]
                & mask_w[None, :]
                & (conv_states_input_coord < num_cache_lines)
                & (conv_states_offset < num_cache_lines)
            )

            src_ptrs = state_src_base[None, :] + src_tok[:, None] * stride_conv_state_tok
            dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            vals = tl.load(src_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, vals, mask=m)

        # B) append x into dst[keep_shift : keep_shift+seqlen_run) (only when seqlen_run < state_len_run)
        for t0 in tl.static_range(0, seqlen, T_CHUNK):
            x_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            dst_tok = (keep_shift + x_tok).to(tl.int32)  # [T_CHUNK]
            m_tok = use_shift & (x_tok < seqlen_run) & (dst_tok < state_len_run)
            m = (lane_active & m_tok)[:, None] & mask_w[None, :] & (conv_states_offset < num_cache_lines)

            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            x_vals = tl.load(x_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, x_vals, mask=m)

        # C) if seqlen_run >= state_len_run, overwrite dst with the tail of x
        for t0 in tl.static_range(0, NP2_STATELEN, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            x_tok = (tail_start + dst_tok).to(tl.int32)  # [T_CHUNK]
            m_tok = use_tail & (dst_tok < state_len_run) & (x_tok < seqlen_run)
            m = (lane_active & m_tok)[:, None] & mask_w[None, :] & (conv_states_offset < num_cache_lines)

            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            x_vals = tl.load(x_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, x_vals, mask=m)

        # -------------------------
        # STEP 3/4/5: causal conv1d (+ optional SiLU) and store output
        # This is original STEP3~5, but per-lane and without debug_barrier.
        # -------------------------
        x_base_1d = x_base
        o_base_1d = o_ptr + o_offset + idx_feats * stride_o_dim

        # accumulator preload (bias)
        acc_preload = acc_bias

        # compute each token; keep tl.range so varlen can use seqlen_run as runtime trip count (like original)
        for idx_token in tl.range(seqlen_run):
            acc = acc_preload

            # same selection logic as original (unrolled by KERNEL_WIDTH)
            matrix_w = w_col0
            matrix_x = col0
            for j in tl.static_range(KERNEL_WIDTH):
                if KERNEL_WIDTH == 1:
                    # only x[t] * w0
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(tl.float16)
                    matrix_w = w_col0
                elif KERNEL_WIDTH == 2:
                    if j == 1:
                        matrix_w = w_col1
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(tl.float16)
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(tl.float16)
                elif KERNEL_WIDTH == 4:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(tl.float16)
                elif KERNEL_WIDTH == 5:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = col3
                    elif j == 4:
                        matrix_w = w_col4
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(tl.float16)
                elif KERNEL_WIDTH == 6:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        matrix_x = col2
                    elif j == 3:
                        matrix_w = w_col3
                        matrix_x = col3
                    elif j == 4:
                        matrix_w = w_col4
                        matrix_x = col4
                    elif j == 5:
                        matrix_w = w_col5
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(tl.float16)

                acc += matrix_x.to(tl.float32) * matrix_w  # [BLOCK_N]

            # roll history window
            if KERNEL_WIDTH == 2:
                col0 = matrix_x
            elif KERNEL_WIDTH == 3:
                col0 = col1
                col1 = matrix_x
            elif KERNEL_WIDTH == 4:
                col0 = col1
                col1 = col2
                col2 = matrix_x
            elif KERNEL_WIDTH == 5:
                col0 = col1
                col1 = col2
                col2 = col3
                col3 = matrix_x
            elif KERNEL_WIDTH == 6:
                col0 = col1
                col1 = col2
                col2 = col3
                col3 = col4
                col4 = matrix_x

            if SILU_ACTIVATION:
                acc = acc / (1.0 + tl.exp(-acc))

            # store output
            o_ptrs = o_base_1d + idx_token * stride_o_token
            tl.store(o_ptrs, acc, mask=lane_active & mask_w)


def causal_conv1d_update_npu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
):
    """
    x: Input tensor which can take the following shapes:

    - `[batch, dim]` - single token prediction
    - `[batch, dim, seqlen]` - single or multiple tokens prediction
    - `[num_tokens, dim]` - continuous batching, where num_tokens is
        the total tokens of all sequences in that batch

    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into conv_state_indices, where the last cache block to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into conv_state_indices, where the cache block containing the initial state is located.
    num_accepted_tokens: (batch,), dtype int32
        If not None, it indicates the number of accepted tokens for each
        sequence in the batch.
        This is used in speculative decoding, where the conv_state is updated
        in a sliding window manner.
    query_start_loc: (batch + 1,) int32
        If not None, the inputs is given in a varlen fashion and this indicates
        the starting index of each sequence in the batch.
    max_query_len: int
        If query_start_loc is not None, this indicates the maximum query
        length in the batch.
    pad_slot_id: int
            if conv_state_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: conv_state_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen) or (num_tokens, dim), same shape as `x`
    """
    weight = weight.transpose(0, 1).contiguous()
    conv_state = conv_state.transpose(1, 2).contiguous()
    if validate_data:
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(1)

    if query_start_loc is None:
        batch, seqlen, dim = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len

    width, _ = weight.shape
    num_cache_lines, state_len_total, _ = conv_state.size()

    # overwrite-on-x strategy same as original
    out = x

    stride_w_width, stride_w_dim = weight.stride()
    if query_start_loc is None:
        stride_x_seq, stride_x_token, stride_x_dim = x.stride()
        stride_o_seq, stride_o_token, stride_o_dim = out.stride()
    else:
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_istate_seq, stride_istate_token, stride_istate_dim = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0) if conv_state_indices is not None else 0

    # effective state_len exactly as original
    if num_accepted_tokens is not None:
        eff_state_len = width - 1 + (seqlen - 1)
    else:
        eff_state_len = width - 1
    np2_statelen = triton.next_power_of_2(eff_state_len)

    # -------- tiling heuristic--------
    # keep program count around ~[80..160]
    # vector core 40
    # TODO: use driver to get the vector core num
    CORE_HINT = 40
    # channel tile: 512 when dim large (reduce tasks), else 256
    block_n = 512 if dim >= 512 else 256
    g = triton.cdiv(dim, block_n)
    target = 2 * CORE_HINT  # ~80
    b_tile_raw = max(1, (batch * g + target - 1) // target)
    # clamp to small set
    if b_tile_raw <= 1:
        b_tile = 1
    elif b_tile_raw <= 2:
        b_tile = 2
    elif b_tile_raw <= 4:
        b_tile = 4
    else:
        b_tile = 8

    # token chunk based on block_n (32KB UB idea); conservative
    t_chunk = 1 if block_n == 512 else 48

    def grid(META):
        return (
            triton.cdiv(batch, META["B_TILE"]),
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_update_kernel_npu_tiled[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        block_idx_last_scheduled_token,
        initial_state_idx,
        out,
        batch,
        dim,
        seqlen,
        eff_state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_VARLEN=query_start_loc is not None,
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=block_n,
        B_TILE=b_tile,
        T_CHUNK=t_chunk,
    )

    if unsqueeze:
        out = out.squeeze(1)
    return out.to(original_x_dtype)


# ============================================================================
# All-mode causal conv1d forward kernel (ported from upstream for NPU)
# Upstream source: vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# Only KERNEL_WIDTH=4 (state_len=3) is supported.
# NPU adaptations: no cache_modifier, row-major x layout (num_tokens, dim).
# ============================================================================


@triton.jit(
    do_not_specialize=[
        "seqlen",
        "num_cache_lines",
        "stride_x_token",
        "stride_istate_seq",
        "stride_istate_token",
        "stride_cache_indices",
        "stride_o_token",
    ]
)
def _causal_conv1d_fwd_kernel_npu(
    # Pointers
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_states_ptr,
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    block_idx_first_scheduled_token,
    block_idx_last_scheduled_token,
    initial_state_idx,
    num_computed_tokens,
    o_ptr,
    # Dimensions
    dim: tl.constexpr,
    seqlen,
    num_cache_lines,
    # Strides
    stride_x_dim: tl.constexpr,
    stride_x_token,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_istate_seq,
    stride_istate_dim: tl.constexpr,
    stride_istate_token,
    stride_cache_indices,
    stride_o_dim: tl.constexpr,
    stride_o_token,
    stride_block_m: tl.constexpr,
    # Others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """NPU Triton port of upstream _causal_conv1d_fwd_kernel (KERNEL_WIDTH=4).

    Each program processes one BLOCK_M-token chunk of one sequence along a
    BLOCK_N slice of the feature dimension.
    Grid: (num_programs, cdiv(dim, BLOCK_N))
    """
    # Aliases matching upstream naming
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = 3  # KERNEL_WIDTH(=4) - 1

    # --- Program mapping ---
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index

    B_size = stride_block_m * BLOCK_M  # = block_size_to_align

    # --- APC bookkeeping ---
    if IS_APC_ENABLED:
        current_first_index = tl.load(
            block_idx_first_scheduled_token + idx_seq)
        current_last_index = tl.load(
            block_idx_last_scheduled_token + idx_seq)
        sequence_completed_index = tl.load(num_computed_tokens + idx_seq)

        sequence_completed_offset_token = sequence_completed_index % B_size
        seq_completed_offset = B_size - sequence_completed_offset_token
        seq_end_offset = (seqlen - seq_completed_offset) % B_size
        last_full_block_token_index = sequence_end_index - seq_end_offset
        if seq_end_offset == 0:
            last_full_block_token_index = (
                last_full_block_token_index - B_size)

        n_block_to_fill = current_last_index - current_first_index
        conv_state_init_index = tl.load(initial_state_idx + idx_seq)
    else:
        n_block_to_fill = 0
        current_last_index = 0
        conv_state_init_index = 0
        current_first_index = 0
        last_full_block_token_index = 0

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    # --- Base pointers ---
    x_base = (
        x_ptr
        + sequence_start_index * stride_x_token
        + idx_feats * stride_x_dim
    )

    # SOURCE pool slot
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr
        + idx_seq * stride_cache_indices
        + conv_state_init_index
    ).to(tl.int64)

    if USE_PAD_SLOT:
        if conv_states_input_coord == pad_slot_id:
            return

    conv_states_base = (
        conv_states_ptr
        + conv_states_input_coord * stride_conv_state_seq
        + idx_feats * stride_conv_state_dim
    )

    w_base = w_ptr + idx_feats * stride_w_dim

    # ================================================================
    # Section 1: chunk_offset == 0 — read initial state + write final
    # ================================================================
    if chunk_offset == 0:
        load_init_state = tl.load(
            has_initial_states_ptr + idx_seq).to(tl.int1)

        if load_init_state:
            # Read 3 history values from SOURCE pool slot
            prior_tokens = (
                conv_states_base
                + (state_len - 1) * stride_conv_state_tok)
            mask_w = idx_feats < dim
            col2 = tl.load(prior_tokens, mask_w, 0.0)
            col1 = tl.load(
                prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
            col0 = tl.load(
                prior_tokens - 2 * stride_conv_state_tok, mask_w, 0.0)
        else:
            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # Write final conv_state to DEST pool slot
        # NPU workaround: use 1D row loop instead of 2D load→store
        # (2D data flow crashes triton-adapter-opt MLIR lowering)
        if state_len <= seqlen:
            # Common case: copy last state_len tokens from x to DEST
            conv_states_output_coord = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_cache_indices
                + current_last_index
            ).to(tl.int64)
            mask_f_wr = idx_feats < dim
            for row in range(state_len):
                src_tok = (seqlen - state_len) + row
                x_row_ptr = (
                    x_ptr
                    + (sequence_start_index + src_tok) * stride_x_token
                    + idx_feats * stride_x_dim
                )
                row_data = tl.load(x_row_ptr, mask_f_wr, 0.0)
                cs_row_ptr = (
                    conv_states_ptr
                    + conv_states_output_coord * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                    + row * stride_conv_state_tok
                )
                tl.store(cs_row_ptr, row_data, mask_f_wr)

        else:
            # Rare case: seqlen < state_len (fewer than 3 new tokens)
            if load_init_state:
                # Mix old conv_state with new x tokens (1D row loop)
                VAL = state_len - seqlen
                conv_states_output_coord_rare = tl.load(
                    conv_state_indices_ptr
                    + idx_seq * stride_cache_indices
                    + current_last_index
                ).to(tl.int64)
                mask_f_rare = idx_feats < dim
                pool_valid = conv_states_input_coord < num_cache_lines
                for row in range(state_len):
                    # Read from pool (shifted old state) or x (new tokens)
                    pool_tok = row + seqlen
                    pool_src = (
                        conv_states_ptr
                        + conv_states_input_coord * stride_conv_state_seq
                        + idx_feats * stride_conv_state_dim
                        + pool_tok * stride_conv_state_tok
                    )
                    pool_data = tl.load(
                        pool_src,
                        mask_f_rare & (pool_tok < state_len) & pool_valid,
                        0.0)
                    x_tok = row - VAL
                    x_src = x_base + x_tok * stride_x_token
                    x_data = tl.load(
                        x_src,
                        mask_f_rare & (x_tok >= 0) & (x_tok < seqlen),
                        0.0)
                    # Exactly one of pool_data/x_data is non-zero
                    row_data = pool_data + x_data
                    cs_dst = (
                        conv_states_ptr
                        + conv_states_output_coord_rare * stride_conv_state_seq
                        + idx_feats * stride_conv_state_dim
                        + row * stride_conv_state_tok
                    )
                    tl.store(cs_dst, row_data, mask_f_rare)
            else:
                # No initial state, seqlen < state_len: zero-pad + x (1D)
                VAL = state_len - seqlen
                conv_states_output_coord_rare2 = tl.load(
                    conv_state_indices_ptr
                    + idx_seq * stride_cache_indices
                    + current_last_index
                ).to(tl.int64)
                mask_f_rare2 = idx_feats < dim
                for row in range(state_len):
                    x_tok = row - VAL
                    x_src = x_base + x_tok * stride_x_token
                    # Valid only if x_tok in [0, seqlen); else 0.0
                    row_data = tl.load(
                        x_src,
                        mask_f_rare2 & (x_tok >= 0) & (x_tok < seqlen),
                        0.0)
                    cs_dst = (
                        conv_states_ptr
                        + conv_states_output_coord_rare2
                        * stride_conv_state_seq
                        + idx_feats * stride_conv_state_dim
                        + row * stride_conv_state_tok
                    )
                    tl.store(cs_dst, row_data, mask_f_rare2)

    else:
        # ================================================================
        # Section 2: chunk_offset > 0 — read prior tokens from x
        # ================================================================
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        # Read 3 prior tokens (no cache_modifier on NPU)
        col2 = tl.load(prior_tokens, mask_w, 0.0)
        col1 = tl.load(
            prior_tokens - 1 * stride_x_token, mask_w, 0.0)
        col0 = tl.load(
            prior_tokens - 2 * stride_x_token, mask_w, 0.0)

        # APC: write intermediate block boundary conv_state (1D row loop)
        if (chunk_offset - 1) < n_block_to_fill:
            base_token = (
                last_full_block_token_index
                - (n_block_to_fill - chunk_offset) * B_size
                - state_len
            )
            conv_states_output_coord = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_cache_indices
                + current_first_index
                + (chunk_offset - 1)
            ).to(tl.int64)
            mask_f_inter = idx_feats < dim
            for row in range(state_len):
                src_tok = base_token + row
                x_row_ptr = (
                    x_ptr
                    + src_tok * stride_x_token
                    + idx_feats * stride_x_dim
                )
                row_data = tl.load(
                    x_row_ptr, mask_f_inter & (src_tok >= 0), 0.0)
                cs_row_ptr = (
                    conv_states_ptr
                    + conv_states_output_coord * stride_conv_state_seq
                    + idx_feats * stride_conv_state_dim
                    + row * stride_conv_state_tok
                )
                tl.store(cs_row_ptr, row_data, mask_f_inter)

    # ================================================================
    # Section 3: Compute conv1d output (width=4 unrolled)
    # ================================================================
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(
            bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token

    # Preload weights (width=4)
    mask_w = idx_feats < dim
    w_col0 = tl.load(w_base + 0 * stride_w_width, mask_w, other=0.0)
    w_col1 = tl.load(w_base + 1 * stride_w_width, mask_w, other=0.0)
    w_col2 = tl.load(w_base + 2 * stride_w_width, mask_w, other=0.0)
    w_col3 = tl.load(w_base + 3 * stride_w_width, mask_w, other=0.0)

    mask_x_1d = idx_feats < dim

    for idx_token in range(segment_len):
        acc = acc_preload
        # Width=4 conv: acc = bias + col0*w0 + col1*w1 + col2*w2 + x*w3
        acc += col0 * w_col0
        acc += col1 * w_col1
        acc += col2 * w_col2
        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
        acc += matrix_x * w_col3

        # Roll sliding window
        col0 = col1
        col1 = col2
        col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))

        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        o_ptrs = (
            o_ptr
            + (sequence_start_index + token_offset + idx_token)
            * stride_o_token
            + idx_feats * stride_o_dim
        )
        tl.store(o_ptrs, acc, mask=mask_1d)


def compute_conv1d_grid_npu(
    query_start_loc: torch.Tensor,
    block_m: int,
    pad_slot_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Precompute grid scheduling tensors for the APC prefill conv1d path.

    Returns:
        batch_ptr: (num_programs,) seq index per program
        token_chunk_offset_ptr: (num_programs,) chunk offset per program
        num_programs: total number of programs
    """
    seqlens = query_start_loc.diff().cpu().numpy()
    nums = -(-seqlens // block_m)  # cdiv
    total = int(nums.sum())

    mlist = np.repeat(np.arange(len(nums)), nums)
    offsetlist: list[int] = []
    for num in nums:
        offsetlist.extend(range(int(num)))

    batch_ptr = torch.tensor(mlist, dtype=torch.int32, device=device)
    token_chunk_offset_ptr = torch.tensor(
        offsetlist, dtype=torch.int32, device=device)
    return batch_ptr, token_chunk_offset_ptr, total


def _causal_conv1d_fwd_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align: int = 0,
) -> torch.Tensor:
    """NPU causal conv1d forward for all-mode prefix caching.

    Ported from upstream _causal_conv1d_fwd_kernel (KERNEL_WIDTH=4 only).
    This is intentionally kept behind causal_conv1d_fn so callers can stay on
    the existing public conv1d interface.

    Args:
        x: (num_tokens, dim) row-major input or (dim, num_tokens) input
        weight: (dim, width=4) conv weights, or the transposed (width, dim)
        bias: (dim,) optional bias
        conv_states: (N, dim, state_len=3) pool view (transposed from gdn.py)
        query_start_loc: (batch+1,) cumulative sequence lengths
        cache_indices: (batch, max_blocks) 2D block table with pool slot IDs
        has_initial_state: (batch,) bool per seq
        activation: "silu" or None
        pad_slot_id: sentinel value for padding
        block_idx_first_scheduled_token: (batch,) first block to fill
        block_idx_last_scheduled_token: (batch,) DEST block index
        initial_state_idx: (batch,) SOURCE block pointer into cache_indices
        num_computed_tokens: (batch,) tokens already computed per seq
        block_size_to_align: mamba block size (divisible by BLOCK_M=8)

    Returns:
        (num_tokens, dim) output tensor
    """
    if x.dim() != 2:
        raise RuntimeError(
            f"APC causal_conv1d_fn expects 2D input, got x.dim()={x.dim()}"
        )

    input_is_dim_major = False
    if weight.shape[0] == x.shape[1]:
        x_row_major = x
        weight_row_major = weight
    elif weight.shape[1] == x.shape[1]:
        x_row_major = x
        weight_row_major = weight.transpose(0, 1)
    elif weight.shape[0] == x.shape[0]:
        input_is_dim_major = True
        x_row_major = x.transpose(0, 1)
        weight_row_major = weight
    elif weight.shape[1] == x.shape[0]:
        input_is_dim_major = True
        x_row_major = x.transpose(0, 1)
        weight_row_major = weight.transpose(0, 1)
    else:
        raise RuntimeError(
            "APC causal_conv1d_fn could not infer input layout: "
            f"x.shape={tuple(x.shape)}, weight.shape={tuple(weight.shape)}"
        )

    if x_row_major.stride(-1) != 1:
        x_row_major = x_row_major.contiguous()
    weight_row_major = weight_row_major.contiguous()
    bias = bias.contiguous() if bias is not None else None

    dim_val = x_row_major.shape[1]
    if conv_states.shape[-2] != dim_val and conv_states.shape[-1] == dim_val:
        conv_states = conv_states.transpose(-1, -2)
    if conv_states.shape[-2] != dim_val:
        raise RuntimeError(
            "APC causal_conv1d_fn: conv_states dim mismatch, "
            f"expected dim={dim_val}, conv_states.shape={tuple(conv_states.shape)}"
        )

    original_dtype = x_row_major.dtype
    x_row_major = x_row_major.to(conv_states.dtype)
    out = torch.empty_like(x_row_major)

    num_tokens, _ = x_row_major.shape
    _, width = weight_row_major.shape
    assert width == 4, f"Only KERNEL_WIDTH=4 supported, got {width}"
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    BLOCK_M = 8
    BLOCK_N = 256

    # Strides for (num_tokens, dim) row-major layout
    stride_x_dim = x_row_major.stride(1)      # 1 (feature contiguous)
    stride_x_token = x_row_major.stride(0)    # dim
    stride_w_dim = weight_row_major.stride(0)
    stride_w_width = weight_row_major.stride(1)
    stride_o_dim = out.stride(1)
    stride_o_token = out.stride(0)

    # Conv state strides: (N, dim, state_len) dim-contiguous view
    num_cache_lines = conv_states.size(0)
    stride_istate_seq = conv_states.stride(0)
    stride_istate_dim = conv_states.stride(1)
    stride_istate_token = conv_states.stride(2)
    assert stride_istate_dim == 1, (
        f"conv_states must be dim-contiguous, got stride(1)={stride_istate_dim}"
    )

    stride_cache_indices = (
        cache_indices.stride(0) if cache_indices is not None else 0)

    if block_size_to_align is not None and block_size_to_align > 0:
        assert (block_size_to_align % BLOCK_M) == 0, (
            f"block_size ({block_size_to_align}) not divisible by BLOCK_M"
            f" ({BLOCK_M})"
        )
    else:
        block_size_to_align = BLOCK_M

    # Grid scheduling
    batch_ptr, token_chunk_offset_ptr, num_programs = compute_conv1d_grid_npu(
        query_start_loc,
        BLOCK_M,
        pad_slot_id,
        x_row_major.device,
    )

    grid = (num_programs, triton.cdiv(dim_val, BLOCK_N))

    _causal_conv1d_fwd_kernel_npu[grid](
        # Pointers
        x_row_major, weight_row_major, bias, conv_states, cache_indices,
        has_initial_state, query_start_loc,
        batch_ptr, token_chunk_offset_ptr,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx, num_computed_tokens,
        out,
        # Dimensions
        dim_val, num_tokens, num_cache_lines,
        # Strides
        stride_x_dim, stride_x_token,
        stride_w_dim, stride_w_width,
        stride_istate_seq, stride_istate_dim, stride_istate_token,
        stride_cache_indices,
        stride_o_dim, stride_o_token,
        block_size_to_align // BLOCK_M,
        # Others
        pad_slot_id,
        # Meta-parameters
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    out = out.to(original_dtype)
    if input_is_dim_major:
        return out.transpose(0, 1).contiguous()
    return out
