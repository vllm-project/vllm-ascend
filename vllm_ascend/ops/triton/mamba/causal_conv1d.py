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


def _normalize_apc_input_layout(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
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
    return x_row_major, weight_row_major.contiguous(), input_is_dim_major


def _extract_apc_state_from_history(
    history: torch.Tensor,
    num_processed_tokens: int,
    state_len: int,
) -> torch.Tensor:
    return history.narrow(-1, num_processed_tokens, state_len).contiguous()


def _write_apc_conv_state(
    conv_states: torch.Tensor,
    slot_id: int,
    state: torch.Tensor,
    state_len: int,
    pad_slot_id: int,
) -> None:
    if slot_id == pad_slot_id:
        return
    conv_states[slot_id][..., :state_len].copy_(state.to(conv_states.dtype))


def _write_apc_intermediate_states(
    conv_states: torch.Tensor,
    cache_row: torch.Tensor,
    first_block_idx: int,
    last_block_idx: int,
    num_computed_tokens: int,
    block_size_to_align: int,
    history: torch.Tensor,
    state_len: int,
    pad_slot_id: int,
    query_len: int,
) -> None:
    for block_idx in range(first_block_idx, last_block_idx):
        boundary_processed_tokens = (block_idx + 1) * block_size_to_align - num_computed_tokens
        if boundary_processed_tokens <= 0 or boundary_processed_tokens > query_len:
            continue
        slot_id = int(cache_row[block_idx].item())
        state = _extract_apc_state_from_history(
            history,
            boundary_processed_tokens,
            state_len,
        )
        _write_apc_conv_state(conv_states, slot_id, state, state_len, pad_slot_id)


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
    """All-mode causal conv1d forward behind the shared public API.

    This keeps the APC behavior inside causal_conv1d_fn, but implements the
    state routing with the existing causal_conv1d_ref instead of a dedicated
    forward Triton kernel. That keeps the public surface close to upstream
    while preserving the all-mode semantics required by the NPU path.

    Args:
        x: (num_tokens, dim) row-major input or (dim, num_tokens) input
        weight: (dim, width) conv weights, or the transposed (width, dim)
        bias: (dim,) optional bias
        conv_states: (N, dim, state_len) pool view (transposed from gdn.py)
        query_start_loc: (batch+1,) cumulative sequence lengths
        cache_indices: (batch, max_blocks) 2D block table with pool slot IDs
        has_initial_state: (batch,) bool per seq
        activation: "silu" or None
        pad_slot_id: sentinel value for padding
        block_idx_first_scheduled_token: (batch,) first block to fill
        block_idx_last_scheduled_token: (batch,) DEST block index
        initial_state_idx: (batch,) SOURCE block pointer into cache_indices
        num_computed_tokens: (batch,) tokens already computed per seq
        block_size_to_align: mamba block size

    Returns:
        (num_tokens, dim) output tensor
    """
    x_row_major, weight_row_major, input_is_dim_major = _normalize_apc_input_layout(
        x,
        weight,
    )
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

    _, width = weight_row_major.shape
    state_len = width - 1
    if block_size_to_align <= 0:
        raise RuntimeError("APC causal_conv1d_fn requires block_size_to_align > 0.")

    if any(
        tensor is None
        for tensor in (
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
            num_computed_tokens,
        )
    ):
        raise RuntimeError("APC causal_conv1d_fn requires complete block metadata.")

    history_zeros = torch.zeros(
        dim_val,
        state_len,
        dtype=conv_states.dtype,
        device=conv_states.device,
    )
    batch = int(query_start_loc.numel() - 1)

    for seq_idx in range(batch):
        start = int(query_start_loc[seq_idx].item())
        end = int(query_start_loc[seq_idx + 1].item())
        if end <= start:
            continue

        cache_row = cache_indices[seq_idx]
        query_tokens = x_row_major[start:end].transpose(0, 1).unsqueeze(0)
        query_len = query_tokens.shape[-1]

        init_state = None
        if bool(has_initial_state[seq_idx].item()):
            init_ptr = int(initial_state_idx[seq_idx].item())
            init_slot = int(cache_row[init_ptr].item())
            if init_slot != pad_slot_id:
                init_state = conv_states[init_slot][..., :state_len].unsqueeze(0)

        out_seq, final_state = causal_conv1d_ref(
            query_tokens,
            weight_row_major,
            bias,
            initial_states=init_state,
            return_final_states=True,
            activation=activation,
        )
        out[start:end] = out_seq.squeeze(0).transpose(0, 1)

        history = torch.cat(
            [
                history_zeros if init_state is None else init_state.squeeze(0),
                query_tokens.squeeze(0),
            ],
            dim=-1,
        )

        first_block_idx = int(block_idx_first_scheduled_token[seq_idx].item())
        last_block_idx = int(block_idx_last_scheduled_token[seq_idx].item())
        computed_tokens = int(num_computed_tokens[seq_idx].item())

        _write_apc_intermediate_states(
            conv_states=conv_states,
            cache_row=cache_row,
            first_block_idx=first_block_idx,
            last_block_idx=last_block_idx,
            num_computed_tokens=computed_tokens,
            block_size_to_align=block_size_to_align,
            history=history,
            state_len=state_len,
            pad_slot_id=pad_slot_id,
            query_len=query_len,
        )

        final_slot = int(cache_row[last_block_idx].item())
        _write_apc_conv_state(
            conv_states,
            final_slot,
            final_state.squeeze(0),
            state_len,
            pad_slot_id,
        )

    out = out.to(original_dtype)
    if input_is_dim_major:
        return out.transpose(0, 1).contiguous()
    return out
