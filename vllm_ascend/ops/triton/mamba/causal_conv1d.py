# adapted from vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# mypy: ignore-errors

from typing import Any

import torch
import torch.nn.functional as F
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # type: ignore


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


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    conv_states: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    metadata: Any | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
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
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
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
        cache_index = int(cache_indices[i].item())
        if cache_index == PAD_SLOT_ID:
            continue
        initial_state = (
            conv_states[cache_index][..., : (width - 1)]
            if has_initial_state is None or bool(has_initial_state[i].item())
            else None
        )
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=conv_states[cache_index][..., : (width - 1)].unsqueeze(0),
                initial_states=initial_state,
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


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "batch",
        "num_cache_lines",
        "stride_q_token",
        "stride_k_token",
        "stride_v_token",
        "stride_state_seq",
        "stride_state_indices",
    ]
)
def _causal_conv1d_prefill_qkv_pack4_kernel_npu(
    q_ptr,
    k_ptr,
    v_ptr,
    wq_ptr,
    wk_ptr,
    wv_ptr,
    bq_ptr,
    bk_ptr,
    bv_ptr,
    state_q_ptr,
    state_k_ptr,
    state_v_ptr,
    state_indices_ptr,
    has_initial_state_ptr,
    query_start_loc_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    num_tokens: tl.int32,
    batch: tl.int32,
    dim: tl.constexpr,
    num_cache_lines,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_x_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_state_seq,
    stride_state_tok: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_indices,
    pad_slot_id: tl.constexpr,
    HAS_BIAS_Q: tl.constexpr,
    HAS_BIAS_K: tl.constexpr,
    HAS_BIAS_V: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    Q_TO_STATE_RTNE: tl.constexpr,
    K_TO_STATE_RTNE: tl.constexpr,
    V_TO_STATE_RTNE: tl.constexpr,
    STATE_TO_Q_RTNE: tl.constexpr,
    STATE_TO_K_RTNE: tl.constexpr,
    STATE_TO_V_RTNE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)

    idx_tokens = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_feats = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_w = idx_feats < dim

    in_batch = pid_b < batch
    qs = tl.load(query_start_loc_ptr + pid_b, mask=in_batch, other=0).to(tl.int64)
    qe = tl.load(query_start_loc_ptr + pid_b + 1, mask=in_batch, other=0).to(tl.int64)
    seqlen = (qe - qs).to(tl.int32)
    token_abs = qs + idx_tokens
    token_mask = in_batch & (idx_tokens < seqlen) & (token_abs < num_tokens)

    state_idx = tl.load(
        state_indices_ptr + pid_b * stride_state_indices,
        mask=in_batch,
        other=0,
    ).to(tl.int64)
    valid_state = in_batch & (seqlen > 0) & (state_idx >= 0) & (state_idx < num_cache_lines)
    if USE_PAD_SLOT:
        valid_state = valid_state & (state_idx != pad_slot_id)
    # Keep masked cache pointers in range for Ascend MTE address validation.
    safe_state_idx = tl.where(valid_state, state_idx, 0)
    has_state = tl.load(has_initial_state_ptr + pid_b, mask=valid_state, other=0).to(tl.int1)

    wq_base = wq_ptr + idx_feats * stride_w_dim
    wk_base = wk_ptr + idx_feats * stride_w_dim
    wv_base = wv_ptr + idx_feats * stride_w_dim
    wq0 = tl.load(wq_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wq1 = tl.load(wq_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wq2 = tl.load(wq_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wq3 = tl.load(wq_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wk0 = tl.load(wk_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wk1 = tl.load(wk_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wk2 = tl.load(wk_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wk3 = tl.load(wk_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wv0 = tl.load(wv_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wv1 = tl.load(wv_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wv2 = tl.load(wv_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wv3 = tl.load(wv_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)

    if HAS_BIAS_Q:
        bias_q = tl.load(bq_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        bias_q = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS_K:
        bias_k = tl.load(bk_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        bias_k = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS_V:
        bias_v = tl.load(bv_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        bias_v = tl.zeros((BLOCK_N,), dtype=tl.float32)

    state_q_base = state_q_ptr + safe_state_idx * stride_state_seq + idx_feats * stride_state_dim
    state_k_base = state_k_ptr + safe_state_idx * stride_state_seq + idx_feats * stride_state_dim
    state_v_base = state_v_ptr + safe_state_idx * stride_state_seq + idx_feats * stride_state_dim
    token_mask_2d = token_mask[:, None] & mask_w[None, :]

    pos0 = idx_tokens - 3
    xmask0 = token_mask_2d & (pos0[:, None] >= 0)
    smask0 = token_mask_2d & (pos0[:, None] < 0) & valid_state & has_state
    stok0 = 3 + pos0
    safe_pos0 = tl.maximum(pos0, 0)
    q0_x = tl.load(
        q_ptr + (qs + safe_pos0)[:, None] * stride_q_token + idx_feats[None, :] * stride_x_dim,
        mask=xmask0,
        other=0.0,
    )
    k0_x = tl.load(
        k_ptr + (qs + safe_pos0)[:, None] * stride_k_token + idx_feats[None, :] * stride_x_dim,
        mask=xmask0,
        other=0.0,
    )
    v0_x = tl.load(
        v_ptr + (qs + safe_pos0)[:, None] * stride_v_token + idx_feats[None, :] * stride_x_dim,
        mask=xmask0,
        other=0.0,
    )
    q0_state_raw = tl.load(
        state_q_base[None, :] + stok0[:, None] * stride_state_tok,
        mask=smask0,
        other=0.0,
    )
    k0_state_raw = tl.load(
        state_k_base[None, :] + stok0[:, None] * stride_state_tok,
        mask=smask0,
        other=0.0,
    )
    v0_state_raw = tl.load(
        state_v_base[None, :] + stok0[:, None] * stride_state_tok,
        mask=smask0,
        other=0.0,
    )
    if STATE_TO_Q_RTNE:
        q0_state = q0_state_raw.to(q_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    else:
        q0_state = q0_state_raw.to(q_ptr.dtype.element_ty)
    if STATE_TO_K_RTNE:
        k0_state = k0_state_raw.to(k_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    else:
        k0_state = k0_state_raw.to(k_ptr.dtype.element_ty)
    if STATE_TO_V_RTNE:
        v0_state = v0_state_raw.to(v_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    else:
        v0_state = v0_state_raw.to(v_ptr.dtype.element_ty)
    q0 = q0_x + q0_state
    k0 = k0_x + k0_state
    v0 = v0_x + v0_state

    pos1 = idx_tokens - 2
    xmask1 = token_mask_2d & (pos1[:, None] >= 0)
    smask1 = token_mask_2d & (pos1[:, None] < 0) & valid_state & has_state
    stok1 = 3 + pos1
    safe_pos1 = tl.maximum(pos1, 0)
    q1_x = tl.load(
        q_ptr + (qs + safe_pos1)[:, None] * stride_q_token + idx_feats[None, :] * stride_x_dim,
        mask=xmask1,
        other=0.0,
    )
    k1_x = tl.load(
        k_ptr + (qs + safe_pos1)[:, None] * stride_k_token + idx_feats[None, :] * stride_x_dim,
        mask=xmask1,
        other=0.0,
    )
    v1_x = tl.load(
        v_ptr + (qs + safe_pos1)[:, None] * stride_v_token + idx_feats[None, :] * stride_x_dim,
        mask=xmask1,
        other=0.0,
    )
    q1_state_raw = tl.load(
        state_q_base[None, :] + stok1[:, None] * stride_state_tok,
        mask=smask1,
        other=0.0,
    )
    k1_state_raw = tl.load(
        state_k_base[None, :] + stok1[:, None] * stride_state_tok,
        mask=smask1,
        other=0.0,
    )
    v1_state_raw = tl.load(
        state_v_base[None, :] + stok1[:, None] * stride_state_tok,
        mask=smask1,
        other=0.0,
    )
    if STATE_TO_Q_RTNE:
        q1_state = q1_state_raw.to(q_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    else:
        q1_state = q1_state_raw.to(q_ptr.dtype.element_ty)
    if STATE_TO_K_RTNE:
        k1_state = k1_state_raw.to(k_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    else:
        k1_state = k1_state_raw.to(k_ptr.dtype.element_ty)
    if STATE_TO_V_RTNE:
        v1_state = v1_state_raw.to(v_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    else:
        v1_state = v1_state_raw.to(v_ptr.dtype.element_ty)
    q1 = q1_x + q1_state
    k1 = k1_x + k1_state
    v1 = v1_x + v1_state

    pos2 = idx_tokens - 1
    xmask2 = token_mask_2d & (pos2[:, None] >= 0)
    smask2 = token_mask_2d & (pos2[:, None] < 0) & valid_state & has_state
    stok2 = 3 + pos2
    safe_pos2 = tl.maximum(pos2, 0)
    q2_x = tl.load(
        q_ptr + (qs + safe_pos2)[:, None] * stride_q_token + idx_feats[None, :] * stride_x_dim,
        mask=xmask2,
        other=0.0,
    )
    k2_x = tl.load(
        k_ptr + (qs + safe_pos2)[:, None] * stride_k_token + idx_feats[None, :] * stride_x_dim,
        mask=xmask2,
        other=0.0,
    )
    v2_x = tl.load(
        v_ptr + (qs + safe_pos2)[:, None] * stride_v_token + idx_feats[None, :] * stride_x_dim,
        mask=xmask2,
        other=0.0,
    )
    q2_state_raw = tl.load(
        state_q_base[None, :] + stok2[:, None] * stride_state_tok,
        mask=smask2,
        other=0.0,
    )
    k2_state_raw = tl.load(
        state_k_base[None, :] + stok2[:, None] * stride_state_tok,
        mask=smask2,
        other=0.0,
    )
    v2_state_raw = tl.load(
        state_v_base[None, :] + stok2[:, None] * stride_state_tok,
        mask=smask2,
        other=0.0,
    )
    if STATE_TO_Q_RTNE:
        q2_state = q2_state_raw.to(q_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    else:
        q2_state = q2_state_raw.to(q_ptr.dtype.element_ty)
    if STATE_TO_K_RTNE:
        k2_state = k2_state_raw.to(k_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    else:
        k2_state = k2_state_raw.to(k_ptr.dtype.element_ty)
    if STATE_TO_V_RTNE:
        v2_state = v2_state_raw.to(v_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    else:
        v2_state = v2_state_raw.to(v_ptr.dtype.element_ty)
    q2 = q2_x + q2_state
    k2 = k2_x + k2_state
    v2 = v2_x + v2_state

    pos3 = idx_tokens
    q3 = tl.load(
        q_ptr + (qs + pos3)[:, None] * stride_q_token + idx_feats[None, :] * stride_x_dim,
        mask=token_mask_2d,
        other=0.0,
    )
    k3 = tl.load(
        k_ptr + (qs + pos3)[:, None] * stride_k_token + idx_feats[None, :] * stride_x_dim,
        mask=token_mask_2d,
        other=0.0,
    )
    v3 = tl.load(
        v_ptr + (qs + pos3)[:, None] * stride_v_token + idx_feats[None, :] * stride_x_dim,
        mask=token_mask_2d,
        other=0.0,
    )

    acc_q = bias_q[None, :]
    acc_q += q0.to(tl.float32) * wq0[None, :]
    acc_q += q1.to(tl.float32) * wq1[None, :]
    acc_q += q2.to(tl.float32) * wq2[None, :]
    acc_q += q3.to(tl.float32) * wq3[None, :]
    acc_k = bias_k[None, :]
    acc_k += k0.to(tl.float32) * wk0[None, :]
    acc_k += k1.to(tl.float32) * wk1[None, :]
    acc_k += k2.to(tl.float32) * wk2[None, :]
    acc_k += k3.to(tl.float32) * wk3[None, :]
    acc_v = bias_v[None, :]
    acc_v += v0.to(tl.float32) * wv0[None, :]
    acc_v += v1.to(tl.float32) * wv1[None, :]
    acc_v += v2.to(tl.float32) * wv2[None, :]
    acc_v += v3.to(tl.float32) * wv3[None, :]
    if SILU_ACTIVATION:
        acc_q = acc_q / (1.0 + tl.exp(-acc_q))
        acc_k = acc_k / (1.0 + tl.exp(-acc_k))
        acc_v = acc_v / (1.0 + tl.exp(-acc_v))

    out_offsets = token_abs[:, None] * dim + idx_feats[None, :]
    q_out = tl.where(valid_state, acc_q, q3.to(tl.float32))
    k_out = tl.where(valid_state, acc_k, k3.to(tl.float32))
    v_out = tl.where(valid_state, acc_v, v3.to(tl.float32))
    tl.store(q_out_ptr + out_offsets, q_out, mask=token_mask_2d)
    tl.store(k_out_ptr + out_offsets, k_out, mask=token_mask_2d)
    tl.store(v_out_ptr + out_offsets, v_out, mask=token_mask_2d)

    if pid_m == 0:
        state_slots = tl.arange(0, 4)
        state_mask = (state_slots < 3)[:, None] & mask_w[None, :] & valid_state
        final_pos = seqlen - 3 + state_slots
        state_from_x_mask = state_mask & (final_pos[:, None] >= 0)
        state_from_old_mask = state_mask & (final_pos[:, None] < 0) & has_state
        old_slot = 3 + final_pos

        q_final_x = tl.load(
            q_ptr + (qs + final_pos)[:, None] * stride_q_token + idx_feats[None, :] * stride_x_dim,
            mask=state_from_x_mask,
            other=0.0,
        )
        k_final_x = tl.load(
            k_ptr + (qs + final_pos)[:, None] * stride_k_token + idx_feats[None, :] * stride_x_dim,
            mask=state_from_x_mask,
            other=0.0,
        )
        v_final_x = tl.load(
            v_ptr + (qs + final_pos)[:, None] * stride_v_token + idx_feats[None, :] * stride_x_dim,
            mask=state_from_x_mask,
            other=0.0,
        )
        q_final_state_raw = tl.load(
            state_q_base[None, :] + old_slot[:, None] * stride_state_tok,
            mask=state_from_old_mask,
            other=0.0,
        )
        k_final_state_raw = tl.load(
            state_k_base[None, :] + old_slot[:, None] * stride_state_tok,
            mask=state_from_old_mask,
            other=0.0,
        )
        v_final_state_raw = tl.load(
            state_v_base[None, :] + old_slot[:, None] * stride_state_tok,
            mask=state_from_old_mask,
            other=0.0,
        )
        if STATE_TO_Q_RTNE:
            q_final_old = q_final_state_raw.to(q_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        else:
            q_final_old = q_final_state_raw.to(q_ptr.dtype.element_ty)
        if STATE_TO_K_RTNE:
            k_final_old = k_final_state_raw.to(k_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        else:
            k_final_old = k_final_state_raw.to(k_ptr.dtype.element_ty)
        if STATE_TO_V_RTNE:
            v_final_old = v_final_state_raw.to(v_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        else:
            v_final_old = v_final_state_raw.to(v_ptr.dtype.element_ty)

        q_final = q_final_x + q_final_old
        k_final = k_final_x + k_final_old
        v_final = v_final_x + v_final_old
        if Q_TO_STATE_RTNE:
            q_final_state = q_final.to(state_q_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        else:
            q_final_state = q_final.to(state_q_ptr.dtype.element_ty)
        if K_TO_STATE_RTNE:
            k_final_state = k_final.to(state_k_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        else:
            k_final_state = k_final.to(state_k_ptr.dtype.element_ty)
        if V_TO_STATE_RTNE:
            v_final_state = v_final.to(state_v_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        else:
            v_final_state = v_final.to(state_v_ptr.dtype.element_ty)
        tl.store(
            state_q_base[None, :] + state_slots[:, None] * stride_state_tok,
            q_final_state,
            mask=state_mask,
        )
        tl.store(
            state_k_base[None, :] + state_slots[:, None] * stride_state_tok,
            k_final_state,
            mask=state_mask,
        )
        tl.store(
            state_v_base[None, :] + state_slots[:, None] * stride_state_tok,
            v_final_state,
            mask=state_mask,
        )


def causal_conv1d_prefill_qkv_pack_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    conv_state_q: torch.Tensor,
    conv_state_k: torch.Tensor,
    conv_state_v: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    weight_v: torch.Tensor,
    bias_q: torch.Tensor | None = None,
    bias_k: torch.Tensor | None = None,
    bias_v: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    num_heads: int = 1,
    head_dim: int | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
    weight_is_transposed: bool = False,
    validate_data: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused q/k/v varlen prefill causal conv for width=4 raw state layout.

    FP16/BF16 inputs may use FP32 conv states. State values are converted to
    the corresponding input dtype before convolution and converted back when
    the active cache line is updated, matching the legacy prefill data flow.
    """
    if conv_state_indices is None:
        raise ValueError("conv_state_indices is required for fused qkv prefill conv")
    if has_initial_state is None:
        raise ValueError("has_initial_state is required for fused qkv prefill conv")
    if query_start_loc is None:
        raise ValueError("query_start_loc is required for fused qkv prefill conv")
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    if not weight_is_transposed:
        weight_q = weight_q.transpose(0, 1).contiguous()
        weight_k = weight_k.transpose(0, 1).contiguous()
        weight_v = weight_v.transpose(0, 1).contiguous()

    if q.dim() != 2 or k.dim() != 2 or v.dim() != 2:
        raise ValueError("fused qkv prefill conv expects 2D q/k/v inputs")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v inputs must have identical shapes")
    num_tokens, dim = q.shape
    if head_dim is None:
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        head_dim = dim // num_heads
    if num_heads * head_dim != dim:
        raise ValueError("num_heads * head_dim must match q/k/v dim")
    if max_seqlen is None or max_seqlen <= 0:
        raise ValueError("max_seqlen must be a positive integer")
    if weight_q.shape != weight_k.shape or weight_q.shape != weight_v.shape:
        raise ValueError("q/k/v weights must have identical shapes")
    if weight_q.shape != (4, dim):
        raise NotImplementedError("fused qkv prefill conv currently supports width=4 only")
    if conv_state_q.shape != conv_state_k.shape or conv_state_q.shape != conv_state_v.shape:
        raise ValueError("q/k/v conv states must have identical shapes")
    if conv_state_q.dtype != conv_state_k.dtype or conv_state_q.dtype != conv_state_v.dtype:
        raise ValueError("q/k/v conv states must have identical dtypes")
    state_dtype = conv_state_q.dtype
    input_dtypes = (q.dtype, k.dtype, v.dtype)
    supports_state_dtype = all(
        input_dtype == state_dtype or (state_dtype == torch.float32 and input_dtype in (torch.float16, torch.bfloat16))
        for input_dtype in input_dtypes
    )
    if not supports_state_dtype:
        raise ValueError(
            "fused qkv prefill conv supports matching input/state dtypes or FP16/BF16 inputs with FP32 conv states"
        )
    if conv_state_q.dim() != 3 or conv_state_q.shape[1] < 3 or conv_state_q.shape[2] != dim:
        raise ValueError("conv_state layout must be (num_cache_lines, >=3, dim)")
    if conv_state_q.stride() != conv_state_k.stride() or conv_state_q.stride() != conv_state_v.stride():
        raise ValueError("q/k/v conv states must have identical strides")
    if weight_q.dtype != q.dtype:
        weight_q = weight_q.to(dtype=q.dtype)
    if weight_k.dtype != k.dtype:
        weight_k = weight_k.to(dtype=k.dtype)
    if weight_v.dtype != v.dtype:
        weight_v = weight_v.to(dtype=v.dtype)
    if bias_q is not None and bias_q.dtype != q.dtype:
        bias_q = bias_q.to(dtype=q.dtype)
    if bias_k is not None and bias_k.dtype != k.dtype:
        bias_k = bias_k.to(dtype=k.dtype)
    if bias_v is not None and bias_v.dtype != v.dtype:
        bias_v = bias_v.to(dtype=v.dtype)
    if weight_q.stride() != weight_k.stride() or weight_q.stride() != weight_v.stride():
        weight_q = weight_q.contiguous()
        weight_k = weight_k.contiguous()
        weight_v = weight_v.contiguous()
    if validate_data:
        assert q.stride(1) == 1 and k.stride(1) == 1 and v.stride(1) == 1
        assert query_start_loc.dim() == 1
        assert conv_state_indices.dim() == 1
        assert has_initial_state.dim() == 1

    def _uses_rtne(src_dtype: torch.dtype, dst_dtype: torch.dtype) -> bool:
        return src_dtype == torch.float32 and dst_dtype in (torch.float16, torch.bfloat16)

    q_to_state_rtne = _uses_rtne(q.dtype, state_dtype)
    k_to_state_rtne = _uses_rtne(k.dtype, state_dtype)
    v_to_state_rtne = _uses_rtne(v.dtype, state_dtype)
    state_to_q_rtne = _uses_rtne(state_dtype, q.dtype)
    state_to_k_rtne = _uses_rtne(state_dtype, k.dtype)
    state_to_v_rtne = _uses_rtne(state_dtype, v.dtype)

    batch = query_start_loc.size(0) - 1
    if batch <= 0:
        raise ValueError("query_start_loc must describe at least one sequence")

    q_out = torch.empty((1, num_tokens, num_heads, head_dim), dtype=q.dtype, device=q.device)
    k_out = torch.empty((1, num_tokens, num_heads, head_dim), dtype=k.dtype, device=k.device)
    v_out = torch.empty((1, num_tokens, num_heads, head_dim), dtype=v.dtype, device=v.device)

    num_cache_lines = conv_state_q.size(0)
    stride_state_seq, stride_state_tok, stride_state_dim = conv_state_q.stride()
    stride_w_width, stride_w_dim = weight_q.stride()
    stride_state_indices = conv_state_indices.stride(0)

    block_m = 8
    block_n = 256

    def grid(META):
        return (
            batch,
            triton.cdiv(max_seqlen, META["BLOCK_M"]),
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_prefill_qkv_pack4_kernel_npu[grid](
        q,
        k,
        v,
        weight_q,
        weight_k,
        weight_v,
        bias_q,
        bias_k,
        bias_v,
        conv_state_q,
        conv_state_k,
        conv_state_v,
        conv_state_indices,
        has_initial_state,
        query_start_loc,
        q_out,
        k_out,
        v_out,
        num_tokens,
        batch,
        dim,
        num_cache_lines,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        q.stride(1),
        stride_w_width,
        stride_w_dim,
        stride_state_seq,
        stride_state_tok,
        stride_state_dim,
        stride_state_indices,
        pad_slot_id,
        HAS_BIAS_Q=bias_q is not None,
        HAS_BIAS_K=bias_k is not None,
        HAS_BIAS_V=bias_v is not None,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        USE_PAD_SLOT=pad_slot_id is not None,
        Q_TO_STATE_RTNE=q_to_state_rtne,
        K_TO_STATE_RTNE=k_to_state_rtne,
        V_TO_STATE_RTNE=v_to_state_rtne,
        STATE_TO_Q_RTNE=state_to_q_rtne,
        STATE_TO_K_RTNE=state_to_k_rtne,
        STATE_TO_V_RTNE=state_to_v_rtne,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return q_out, k_out, v_out


@triton.jit(
    do_not_specialize=[
        "batch",
        "num_cache_lines",
        "stride_q_token",
        "stride_k_token",
        "stride_v_token",
        "stride_state_seq",
        "stride_state_indices",
    ]
)
def _causal_conv1d_update_qkv_pack4_kernel_npu(
    q_ptr,
    k_ptr,
    v_ptr,
    wq_ptr,
    wk_ptr,
    wv_ptr,
    bq_ptr,
    bk_ptr,
    bv_ptr,
    state_q_ptr,
    state_k_ptr,
    state_v_ptr,
    state_indices_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    batch: tl.int32,
    dim: tl.constexpr,
    num_cache_lines,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    stride_x_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_state_seq,
    stride_state_tok: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_indices,
    pad_slot_id: tl.constexpr,
    HAS_BIAS_Q: tl.constexpr,
    HAS_BIAS_K: tl.constexpr,
    HAS_BIAS_V: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    Q_TO_STATE_RTNE: tl.constexpr,
    K_TO_STATE_RTNE: tl.constexpr,
    V_TO_STATE_RTNE: tl.constexpr,
    STATE_TO_Q_RTNE: tl.constexpr,
    STATE_TO_K_RTNE: tl.constexpr,
    STATE_TO_V_RTNE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    B_TILE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    idx_feats = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_w = idx_feats < dim

    wq_base = wq_ptr + idx_feats * stride_w_dim
    wk_base = wk_ptr + idx_feats * stride_w_dim
    wv_base = wv_ptr + idx_feats * stride_w_dim
    wq0 = tl.load(wq_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wq1 = tl.load(wq_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wq2 = tl.load(wq_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wq3 = tl.load(wq_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wk0 = tl.load(wk_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wk1 = tl.load(wk_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wk2 = tl.load(wk_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wk3 = tl.load(wk_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wv0 = tl.load(wv_base + 0 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wv1 = tl.load(wv_base + 1 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wv2 = tl.load(wv_base + 2 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)
    wv3 = tl.load(wv_base + 3 * stride_w_width, mask=mask_w, other=0.0).to(tl.float32)

    if HAS_BIAS_Q:
        bias_q = tl.load(bq_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        bias_q = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS_K:
        bias_k = tl.load(bk_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        bias_k = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS_V:
        bias_v = tl.load(bv_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        bias_v = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for bi in tl.static_range(0, B_TILE):
        b = pid_b * B_TILE + bi
        in_batch = b < batch
        # Keep the final tile's masked batch lanes on valid addresses.
        safe_b = tl.where(in_batch, b, 0)
        state_idx = tl.load(
            state_indices_ptr + safe_b * stride_state_indices,
            mask=in_batch,
            other=0,
        ).to(tl.int64)
        valid_state = in_batch & (state_idx >= 0) & (state_idx < num_cache_lines)
        if USE_PAD_SLOT:
            valid_state = valid_state & (state_idx != pad_slot_id)
        safe_state_idx = tl.where(valid_state, state_idx, 0)

        q_base = q_ptr + safe_b * stride_q_token + idx_feats * stride_x_dim
        k_base = k_ptr + safe_b * stride_k_token + idx_feats * stride_x_dim
        v_base = v_ptr + safe_b * stride_v_token + idx_feats * stride_x_dim
        xq_raw = tl.load(q_base, mask=in_batch & mask_w, other=0.0)
        xk_raw = tl.load(k_base, mask=in_batch & mask_w, other=0.0)
        xv_raw = tl.load(v_base, mask=in_batch & mask_w, other=0.0)
        if Q_TO_STATE_RTNE:
            xq_state = xq_raw.to(state_q_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        else:
            xq_state = xq_raw.to(state_q_ptr.dtype.element_ty)
        if K_TO_STATE_RTNE:
            xk_state = xk_raw.to(state_k_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        else:
            xk_state = xk_raw.to(state_k_ptr.dtype.element_ty)
        if V_TO_STATE_RTNE:
            xv_state = xv_raw.to(state_v_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        else:
            xv_state = xv_raw.to(state_v_ptr.dtype.element_ty)
        xq = xq_state.to(tl.float16)
        xk = xk_state.to(tl.float16)
        xv = xv_state.to(tl.float16)

        state_q_base = state_q_ptr + safe_state_idx * stride_state_seq + idx_feats * stride_state_dim
        state_k_base = state_k_ptr + safe_state_idx * stride_state_seq + idx_feats * stride_state_dim
        state_v_base = state_v_ptr + safe_state_idx * stride_state_seq + idx_feats * stride_state_dim
        q0_raw = tl.load(state_q_base + 0 * stride_state_tok, mask=valid_state & mask_w, other=0.0)
        q1_raw = tl.load(state_q_base + 1 * stride_state_tok, mask=valid_state & mask_w, other=0.0)
        q2_raw = tl.load(state_q_base + 2 * stride_state_tok, mask=valid_state & mask_w, other=0.0)
        k0_raw = tl.load(state_k_base + 0 * stride_state_tok, mask=valid_state & mask_w, other=0.0)
        k1_raw = tl.load(state_k_base + 1 * stride_state_tok, mask=valid_state & mask_w, other=0.0)
        k2_raw = tl.load(state_k_base + 2 * stride_state_tok, mask=valid_state & mask_w, other=0.0)
        v0_raw = tl.load(state_v_base + 0 * stride_state_tok, mask=valid_state & mask_w, other=0.0)
        v1_raw = tl.load(state_v_base + 1 * stride_state_tok, mask=valid_state & mask_w, other=0.0)
        v2_raw = tl.load(state_v_base + 2 * stride_state_tok, mask=valid_state & mask_w, other=0.0)
        q0 = q0_raw.to(tl.float16)
        q1 = q1_raw.to(tl.float16)
        q2 = q2_raw.to(tl.float16)
        k0 = k0_raw.to(tl.float16)
        k1 = k1_raw.to(tl.float16)
        k2 = k2_raw.to(tl.float16)
        v0 = v0_raw.to(tl.float16)
        v1 = v1_raw.to(tl.float16)
        v2 = v2_raw.to(tl.float16)

        acc_q = bias_q
        acc_q += q0.to(tl.float32) * wq0
        acc_q += q1.to(tl.float32) * wq1
        acc_q += q2.to(tl.float32) * wq2
        acc_q += xq.to(tl.float32) * wq3
        acc_k = bias_k
        acc_k += k0.to(tl.float32) * wk0
        acc_k += k1.to(tl.float32) * wk1
        acc_k += k2.to(tl.float32) * wk2
        acc_k += xk.to(tl.float32) * wk3
        acc_v = bias_v
        acc_v += v0.to(tl.float32) * wv0
        acc_v += v1.to(tl.float32) * wv1
        acc_v += v2.to(tl.float32) * wv2
        acc_v += xv.to(tl.float32) * wv3
        if SILU_ACTIVATION:
            acc_q = acc_q / (1.0 + tl.exp(-acc_q))
            acc_k = acc_k / (1.0 + tl.exp(-acc_k))
            acc_v = acc_v / (1.0 + tl.exp(-acc_v))

        out_base = safe_b * dim + idx_feats
        q_state_out = acc_q.to(state_q_ptr.dtype.element_ty)
        k_state_out = acc_k.to(state_k_ptr.dtype.element_ty)
        v_state_out = acc_v.to(state_v_ptr.dtype.element_ty)
        if STATE_TO_Q_RTNE:
            q_out = q_state_out.to(q_out_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
            q_passthrough = xq_state.to(q_out_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        else:
            q_out = q_state_out.to(q_out_ptr.dtype.element_ty)
            q_passthrough = xq_state.to(q_out_ptr.dtype.element_ty)
        if STATE_TO_K_RTNE:
            k_out = k_state_out.to(k_out_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
            k_passthrough = xk_state.to(k_out_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        else:
            k_out = k_state_out.to(k_out_ptr.dtype.element_ty)
            k_passthrough = xk_state.to(k_out_ptr.dtype.element_ty)
        if STATE_TO_V_RTNE:
            v_out = v_state_out.to(v_out_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
            v_passthrough = xv_state.to(v_out_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        else:
            v_out = v_state_out.to(v_out_ptr.dtype.element_ty)
            v_passthrough = xv_state.to(v_out_ptr.dtype.element_ty)
        tl.store(q_out_ptr + out_base, tl.where(valid_state, q_out, q_passthrough), mask=in_batch & mask_w)
        tl.store(k_out_ptr + out_base, tl.where(valid_state, k_out, k_passthrough), mask=in_batch & mask_w)
        tl.store(v_out_ptr + out_base, tl.where(valid_state, v_out, v_passthrough), mask=in_batch & mask_w)

        tl.store(state_q_base + 0 * stride_state_tok, q1_raw, mask=valid_state & mask_w)
        tl.store(state_q_base + 1 * stride_state_tok, q2_raw, mask=valid_state & mask_w)
        tl.store(state_q_base + 2 * stride_state_tok, xq_state, mask=valid_state & mask_w)
        tl.store(state_k_base + 0 * stride_state_tok, k1_raw, mask=valid_state & mask_w)
        tl.store(state_k_base + 1 * stride_state_tok, k2_raw, mask=valid_state & mask_w)
        tl.store(state_k_base + 2 * stride_state_tok, xk_state, mask=valid_state & mask_w)
        tl.store(state_v_base + 0 * stride_state_tok, v1_raw, mask=valid_state & mask_w)
        tl.store(state_v_base + 1 * stride_state_tok, v2_raw, mask=valid_state & mask_w)
        tl.store(state_v_base + 2 * stride_state_tok, xv_state, mask=valid_state & mask_w)


def causal_conv1d_update_qkv_pack_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    conv_state_q: torch.Tensor,
    conv_state_k: torch.Tensor,
    conv_state_v: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    weight_v: torch.Tensor,
    bias_q: torch.Tensor | None = None,
    bias_k: torch.Tensor | None = None,
    bias_v: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_heads: int = 1,
    head_dim: int | None = None,
    batch_size: int | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
    weight_is_transposed: bool = False,
    validate_data: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused q/k/v decode update matching causal_conv1d_update_npu semantics.

    The conv_state tensors use raw cache layout (num_cache_lines, state_len, dim).
    Input rows may include padded tail rows; batch_size limits the rows processed.
    State shifts keep the original cache dtype. q/k/v casts are folded into
    Triton while preserving the old input-state-output dtype flow.
    """
    if conv_state_indices is None:
        raise ValueError("conv_state_indices is required for fused qkv conv update")
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    if not weight_is_transposed:
        weight_q = weight_q.transpose(0, 1).contiguous()
        weight_k = weight_k.transpose(0, 1).contiguous()
        weight_v = weight_v.transpose(0, 1).contiguous()

    if q.dim() != 2 or k.dim() != 2 or v.dim() != 2:
        raise ValueError("fused qkv conv update expects 2D q/k/v inputs")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v inputs must have identical shapes")
    input_batch, dim = q.shape
    if batch_size is None:
        batch = input_batch
    else:
        batch = int(batch_size)
        if batch < 0 or batch > input_batch:
            raise ValueError("batch_size must be in [0, q.shape[0]]")
    if head_dim is None:
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        head_dim = dim // num_heads
    if num_heads * head_dim != dim:
        raise ValueError("num_heads * head_dim must match q/k/v dim")
    if weight_q.shape != weight_k.shape or weight_q.shape != weight_v.shape:
        raise ValueError("q/k/v weights must have identical shapes")
    if weight_q.shape != (4, dim):
        raise NotImplementedError("fused qkv conv update currently supports width=4 only")
    if conv_state_q.shape != conv_state_k.shape or conv_state_q.shape != conv_state_v.shape:
        raise ValueError("q/k/v conv states must have identical shapes")
    if conv_state_q.dtype != conv_state_k.dtype or conv_state_q.dtype != conv_state_v.dtype:
        raise ValueError("q/k/v conv states must have identical dtypes")
    if conv_state_q.dim() != 3 or conv_state_q.shape[1] < 3 or conv_state_q.shape[2] != dim:
        raise ValueError("conv_state layout must be (num_cache_lines, >=3, dim)")
    if conv_state_q.stride() != conv_state_k.stride() or conv_state_q.stride() != conv_state_v.stride():
        raise ValueError("q/k/v conv states must have identical strides")
    if weight_q.stride() != weight_k.stride() or weight_q.stride() != weight_v.stride():
        weight_q = weight_q.contiguous()
        weight_k = weight_k.contiguous()
        weight_v = weight_v.contiguous()
    if validate_data:
        assert q.stride(1) == 1 and k.stride(1) == 1 and v.stride(1) == 1

    original_q_dtype = q.dtype
    original_k_dtype = k.dtype
    original_v_dtype = v.dtype
    state_dtype = conv_state_q.dtype

    def _uses_rtne(src_dtype: torch.dtype, dst_dtype: torch.dtype) -> bool:
        return src_dtype == torch.float32 and dst_dtype in (torch.float16, torch.bfloat16)

    q_to_state_rtne = _uses_rtne(original_q_dtype, state_dtype)
    k_to_state_rtne = _uses_rtne(original_k_dtype, state_dtype)
    v_to_state_rtne = _uses_rtne(original_v_dtype, state_dtype)

    state_to_q_rtne = _uses_rtne(state_dtype, original_q_dtype)
    state_to_k_rtne = _uses_rtne(state_dtype, original_k_dtype)
    state_to_v_rtne = _uses_rtne(state_dtype, original_v_dtype)

    q_out = torch.empty((1, batch, num_heads, head_dim), dtype=original_q_dtype, device=q.device)
    k_out = torch.empty((1, batch, num_heads, head_dim), dtype=original_k_dtype, device=k.device)
    v_out = torch.empty((1, batch, num_heads, head_dim), dtype=original_v_dtype, device=v.device)

    num_cache_lines = conv_state_q.size(0)
    stride_state_seq, stride_state_tok, stride_state_dim = conv_state_q.stride()
    stride_w_width, stride_w_dim = weight_q.stride()
    stride_state_indices = conv_state_indices.stride(0)

    block_n = 512 if dim >= 512 else 256
    core_hint = 40
    g = triton.cdiv(dim, block_n)
    target = 2 * core_hint
    b_tile_raw = max(1, (batch * g + target - 1) // target)
    if b_tile_raw <= 1:
        b_tile = 1
    elif b_tile_raw <= 2:
        b_tile = 2
    elif b_tile_raw <= 4:
        b_tile = 4
    else:
        b_tile = 8

    def grid(META):
        return (
            triton.cdiv(batch, META["B_TILE"]),
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_update_qkv_pack4_kernel_npu[grid](
        q,
        k,
        v,
        weight_q,
        weight_k,
        weight_v,
        bias_q,
        bias_k,
        bias_v,
        conv_state_q,
        conv_state_k,
        conv_state_v,
        conv_state_indices,
        q_out,
        k_out,
        v_out,
        batch,
        dim,
        num_cache_lines,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        q.stride(1),
        stride_w_width,
        stride_w_dim,
        stride_state_seq,
        stride_state_tok,
        stride_state_dim,
        stride_state_indices,
        pad_slot_id,
        HAS_BIAS_Q=bias_q is not None,
        HAS_BIAS_K=bias_k is not None,
        HAS_BIAS_V=bias_v is not None,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        USE_PAD_SLOT=pad_slot_id is not None,
        Q_TO_STATE_RTNE=q_to_state_rtne,
        K_TO_STATE_RTNE=k_to_state_rtne,
        V_TO_STATE_RTNE=v_to_state_rtne,
        STATE_TO_Q_RTNE=state_to_q_rtne,
        STATE_TO_K_RTNE=state_to_k_rtne,
        STATE_TO_V_RTNE=state_to_v_rtne,
        BLOCK_N=block_n,
        B_TILE=b_tile,
    )
    return q_out, k_out, v_out


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
    weight_is_transposed: bool = False,
):
    """
    x: Input tensor which can take the following shapes:

    - `[batch, dim]` - single token prediction
    - `[batch, dim, seqlen]` - single or multiple tokens prediction
    - `[num_tokens, dim]` - continuous batching, where num_tokens is
        the total tokens of all sequences in that batch

    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width), or (width, dim) if weight_is_transposed is True
    weight_is_transposed: whether weight is already stored as (width, dim)
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
    if not weight_is_transposed:
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
