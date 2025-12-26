# adapted from vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# mypy: ignore-errors

from typing import Any, Optional, Union

import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1


@triton.jit()
def _causal_conv1d_fwd_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_states_ptr,
    conv_state_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    o_ptr,  # (dim, seqlen)
    # Matrix dimensions
    dim: tl.constexpr,
    state_len: int,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_cache_indices: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # single-sequence id
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    # base of the sequence
    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim  # [BLOCK_N,]

    if IS_CONTINUOUS_BATCHING:
        # cache_idx
        conv_state_batch_coord = tl.load(conv_state_indices_ptr +
                                         idx_seq * stride_cache_indices).to(
                                             tl.int64)
    else:
        # cache_idx
        conv_state_batch_coord = idx_seq

    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = conv_states_ptr + (
        conv_state_batch_coord * stride_conv_state_seq) + (
            idx_feats * stride_conv_state_dim)  # [BLOCK_N,]
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    load_init_state = False
    if HAS_INITIAL_STATES:  # the new HAS_INITIAL_STATES
        load_init_state = tl.load(has_initial_states_ptr + idx_seq)

    mask_dim = idx_feats < dim

    # read prior-token data from `x`
    offset_x = token_offset - KERNEL_WIDTH + 1
    if KERNEL_WIDTH >= 2:
        x0_ptrs = x_base + offset_x * stride_x_token
        x0 = tl.load(x0_ptrs, mask_dim & (offset_x > 0))
    if KERNEL_WIDTH >= 3:
        x1_ptrs = x0_ptrs + 1 * stride_x_token
        x1 = tl.load(x1_ptrs, mask_dim & (offset_x + 1 > 0))
    if KERNEL_WIDTH >= 4:
        x2_ptrs = x1_ptrs + 1 * stride_x_token
        x2 = tl.load(x2_ptrs, mask_dim & (offset_x + 2 > 0))

    if load_init_state & (chunk_offset == 0):
        # load from conv_states
        offset_conv_state = state_len - KERNEL_WIDTH + 1
        if KERNEL_WIDTH >= 2:
            x0_ptrs = conv_states_base + offset_conv_state * stride_conv_state_tok
            x0 = tl.load(x0_ptrs, mask_dim, 0.0)
        if KERNEL_WIDTH >= 3:
            x1_ptrs = x0_ptrs + 1 * stride_conv_state_tok
            x1 = tl.load(x1_ptrs, mask_dim)
        if KERNEL_WIDTH >= 4:
            x2_ptrs = x1_ptrs + 1 * stride_conv_state_tok
            x2 = tl.load(x2_ptrs, mask_dim)

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias,
                              other=0.0).to(tl.float32)  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N, ), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_dim = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w0 = tl.load(w_ptrs, mask_dim, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w1 = tl.load(w_ptrs, mask_dim, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w2 = tl.load(w_ptrs, mask_dim, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w3 = tl.load(w_ptrs, mask_dim, other=0.0)

    for idx_token in tl.static_range(BLOCK_M):
        acc = acc_preload
        mask_1d = (idx_token
                   < segment_len) & mask_dim  # token-index  # feature-index
        x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
        x = tl.load(x_ptrs_1d, mask=mask_1d)

        if KERNEL_WIDTH == 2:
            acc += x0 * w0 + x * w1
            x0 = x
        elif KERNEL_WIDTH == 3:
            acc += x0 * w0 + x1 * w1 + x * w2
            x0 = x1
            x1 = x
        elif KERNEL_WIDTH == 4:
            acc += x0 * w0 + x1 * w1 + x2 * w2 + x * w3
            x0 = x1
            x1 = x2
            x2 = x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))

        o_ptrs = o_ptr + (sequence_start_index + token_offset + idx_token
                          ) * stride_o_token + (idx_feats * stride_o_dim)
        tl.store(o_ptrs, acc, mask=mask_1d)

    # update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        if state_len <= seqlen:  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN)  # [BLOCK_M]
            x_ptrs = x_ptr + (
                (sequence_start_index + idx_tokens_last) *
                stride_x_token)[:, None] + (
                    idx_feats * stride_x_dim)[None, :]  # [BLOCK_M,BLOCK_N,]
            mask_x = ((idx_tokens_last >= 0)[:, None] &
                      (idx_tokens_last < seqlen)[:, None] &
                      (idx_feats < dim)[None, :]
                      )  # token-index  # token-index  # feature-index
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            conv_states_ptrs_target = conv_states_base[None, :] + (
                idx_tokens_conv * stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats
                                                             < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
        elif load_init_state:
            # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

            conv_states_ptrs_source = (
                conv_states_ptr +
                (conv_state_batch_coord * stride_conv_state_seq) +
                (idx_feats * stride_conv_state_dim)[None, :] +
                ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
            )  # [BLOCK_M, BLOCK_N]
            mask = ((conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :])
            conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

            VAL = state_len - seqlen

            x_ptrs = x_base[None, :] + (
                (idx_tokens_conv - VAL) *
                stride_x_token)[:, None]  # [BLOCK_M, BLOCK_N]

            mask_x = ((idx_tokens_conv - VAL >= 0)[:, None] &
                      (idx_tokens_conv - VAL < seqlen)[:, None] &
                      (idx_feats < dim)[None, :]
                      )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            tl.debug_barrier()
            new_conv_state = tl.where(
                mask, conv_state, loaded_x
            )  # BUG in 'tl.where'  which requires a barrier before this
            conv_states_ptrs_target = conv_states_base + (
                idx_tokens_conv *
                stride_conv_state_tok)[:, None]  # [BLOCK_M, BLOCK_N]
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats
                                                             < dim)[None, :]
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
        else:
            # update conv_state by shifting left, BUT
            # set cols prior to 'x' as zeros + cols from 'x'
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

            VAL = state_len - seqlen

            x_ptrs = x_base[None, :] + (
                (idx_tokens_conv - VAL) *
                stride_x_token)[:, None]  # [BLOCK_M, BLOCK_N]

            mask_x = ((idx_tokens_conv - VAL >= 0)[:, None] &
                      (idx_tokens_conv - VAL < seqlen)[:, None] &
                      (idx_feats < dim)[None, :]
                      )  # token-index  # token-index  # feature-index
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

            conv_states_ptrs_target = conv_states_base + (
                idx_tokens_conv *
                stride_conv_state_tok)[:, None]  # [BLOCK_M, BLOCK_N]
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats
                                                             < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, new_conv_state, mask)


def causal_conv1d_fn(x: torch.Tensor,
                     weight: torch.Tensor,
                     bias: Union[torch.Tensor, None],
                     conv_states: torch.Tensor,
                     query_start_loc: torch.Tensor,
                     cache_indices: Optional[torch.Tensor] = None,
                     has_initial_state: Optional[torch.Tensor] = None,
                     activation: Optional[str] = "silu",
                     pad_slot_id: int = PAD_SLOT_ID,
                     metadata: Optional[Any] = None,
                     validate_data=False):
    """support varlen + continuous batching when x is 2D tensor
    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence
        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3
    out: same shape as `x`
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    # Store original dtype to cast back at the end
    out = torch.empty_strided(x.size(),
                              x.stride(),
                              dtype=x.dtype,
                              device=x.device)

    dim, _ = x.shape
    _, width = weight.shape

    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    padded_batch = query_start_loc.size(0) - 1
    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    stride_o_dim = out.stride(0)
    stride_o_token = out.stride(1)

    num_cache_lines = 0
    if conv_states is not None:
        # extensions to support vLLM:
        # 1. conv_states is used to replaced initial_states
        # 2. conv_states serve as a cache with num cache lines can be larger than batch size
        # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx]
        # 4. computation can be skipped if cache_indices[idx] == pad_slot_id
        num_cache_lines = conv_states.size(0)
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)

    stride_cache_indices = cache_indices.stride(
        0) if cache_indices is not None else 0

    if validate_data:
        is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
        assert x.dim() == 2
        assert width in [2, 3, 4]
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if conv_states is not None:
            assert (num_cache_lines == conv_states.shape[0]
                    and dim == conv_states.shape[1]
                    and conv_states.shape[2] >= width - 1)
            assert stride_istate_dim == 1
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch, )
            assert conv_states is not None, "ERROR: `has_initial_state` is used, which needs also `conv_states`"
        assert weight.stride(1) == 1
        assert (dim, width) == weight.shape
        assert is_channel_last, "Need to run in channel-last layout"

    BLOCK_M = 64
    seqlens = query_start_loc.diff()
    seq_blocks = -(-seqlens // BLOCK_M)
    total_seq_blocks = seq_blocks.sum().item()
    # tracking which seq-idx the Triton program is handling
    batch_ptr = torch.repeat_interleave(
        torch.arange(len(seq_blocks), device=x.device),
        seq_blocks).to(torch.int32)

    # tracking BLOCK_M-based index in the sequence the Triton program is handling
    max_blocks = seq_blocks.max().item() if len(seq_blocks) > 0 else 0
    arange = torch.arange(max_blocks, device=x.device)
    mask = arange.unsqueeze(0) < seq_blocks.unsqueeze(1)
    token_chunk_offset_ptr = arange.repeat(len(seq_blocks),
                                           1)[mask].to(torch.int32)

    BLOCK_N = 256
    grid = (total_seq_blocks, triton.cdiv(dim, BLOCK_N))

    with torch.npu.device(x.device.index):
        _causal_conv1d_fwd_kernel[grid](
            # Pointers to matrices
            x,
            weight,
            bias,
            conv_states,
            cache_indices,
            has_initial_state,
            query_start_loc,
            batch_ptr,
            token_chunk_offset_ptr,
            out,
            # Matrix dimensions
            dim,
            state_len,
            num_cache_lines,
            # stride
            stride_x_dim,
            stride_x_token,
            stride_w_dim,
            stride_w_width,
            stride_istate_seq,
            stride_istate_dim,
            stride_istate_token,
            stride_cache_indices,
            stride_o_dim,
            stride_o_token,
            # others
            pad_slot_id,
            # META
            HAS_BIAS=bias is not None,
            KERNEL_WIDTH=width,
            SILU_ACTIVATION=activation in ["silu", "swish"],
            HAS_INITIAL_STATES=has_initial_state is not None,
            IS_CONTINUOUS_BATCHING=cache_indices is not None,
            USE_PAD_SLOT=pad_slot_id is not None,
            NP2_STATELEN=np2_statelen,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N)

    return out


@triton.jit
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
        state_len: tl.constexpr,  # effective state_len computed in wrapper
        num_cache_lines: tl.constexpr,

        # Strides
        stride_x_seq: tl.constexpr,
        stride_x_dim: tl.constexpr,
        stride_x_token: tl.constexpr,
        stride_w_dim: tl.constexpr,
        stride_w_width: tl.constexpr,
        stride_conv_state_seq: tl.constexpr,
        stride_conv_state_dim: tl.constexpr,
        stride_conv_state_tok: tl.constexpr,
        stride_state_indices: tl.constexpr,
        stride_o_seq: tl.constexpr,
        stride_o_dim: tl.constexpr,
        stride_o_token: tl.constexpr,

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
    w_col0 = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    w_col1 = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    w_col2 = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    w_col3 = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    w_col4 = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    w_col5 = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    if KERNEL_WIDTH >= 1:
        w_col0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w,
                         other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 2:
        w_col1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w,
                         other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + 2 * stride_w_width, mask=mask_w,
                         other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + 3 * stride_w_width, mask=mask_w,
                         other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 5:
        w_col4 = tl.load(w_base + 4 * stride_w_width, mask=mask_w,
                         other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 6:
        w_col5 = tl.load(w_base + 5 * stride_w_width, mask=mask_w,
                         other=0.0).to(tl.float32)

    # bias vector once per program
    if HAS_BIAS:
        acc_bias = tl.load(bias_ptr + idx_feats, mask=mask_w,
                           other=0.0).to(tl.float32)
    else:
        acc_bias = tl.zeros((BLOCK_N, ), dtype=tl.float32)

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
            conv_state_init = tl.load(initial_state_idx + b,
                                      mask=lane_active,
                                      other=0).to(tl.int32)
            current_last_index = tl.load(block_idx_last_scheduled_token + b,
                                         mask=lane_active,
                                         other=0).to(tl.int32)
        else:
            conv_state_init = tl.full((), 0, tl.int32)
            current_last_index = tl.full((), 0, tl.int32)

        # input cache line
        conv_states_input_coord = tl.load(conv_state_indices_ptr +
                                          b * stride_state_indices +
                                          conv_state_init,
                                          mask=lane_active,
                                          other=0).to(tl.int64)

        if USE_PAD_SLOT:
            lane_active = lane_active & (conv_states_input_coord
                                         != pad_slot_id)

        # -------------------------
        # varlen (optional): revise seqlen_run and state_len_run like original kernel does
        # -------------------------
        if IS_VARLEN:
            qs = tl.load(query_start_loc_ptr + b, mask=lane_active,
                         other=0).to(tl.int64)
            qe = tl.load(query_start_loc_ptr + (b + 1),
                         mask=lane_active,
                         other=0).to(tl.int64)
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
            conv_state_token_offset = (
                tl.load(num_accepted_tokens_ptr + b, mask=lane_active,
                        other=1).to(tl.int64) - 1)
            shift = tl.full((), 1, tl.int32)  # sliding by 1 in spec mode
        else:
            conv_state_token_offset = tl.full((), 0, tl.int64)
            shift = seqlen_run  # normal mode shift by seqlen

        # -------------------------
        # STEP 1: read initial history cols BEFORE state update (out==x safe)
        # -------------------------
        conv_states_base = (conv_state_ptr +
                            conv_states_input_coord * stride_conv_state_seq +
                            idx_feats * stride_conv_state_dim)
        prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok

        # define history vectors as zeros then load conditionally
        col0 = tl.zeros((BLOCK_N, ), dtype=tl.float16)
        col1 = tl.zeros((BLOCK_N, ), dtype=tl.float16)
        col2 = tl.zeros((BLOCK_N, ), dtype=tl.float16)
        col3 = tl.zeros((BLOCK_N, ), dtype=tl.float16)
        col4 = tl.zeros((BLOCK_N, ), dtype=tl.float16)
        if KERNEL_WIDTH >= 2:
            col0 = tl.load(prior_tokens + 0 * stride_conv_state_tok,
                           mask=lane_active & mask_w,
                           other=0.0).to(tl.float16)
        if KERNEL_WIDTH >= 3:
            col1 = tl.load(prior_tokens + 1 * stride_conv_state_tok,
                           mask=lane_active & mask_w,
                           other=0.0).to(tl.float16)
        if KERNEL_WIDTH >= 4:
            col2 = tl.load(prior_tokens + 2 * stride_conv_state_tok,
                           mask=lane_active & mask_w,
                           other=0.0).to(tl.float16)
        if KERNEL_WIDTH >= 5:
            col3 = tl.load(prior_tokens + 3 * stride_conv_state_tok,
                           mask=lane_active & mask_w,
                           other=0.0).to(tl.float16)
        if KERNEL_WIDTH >= 6:
            col4 = tl.load(prior_tokens + 4 * stride_conv_state_tok,
                           mask=lane_active & mask_w,
                           other=0.0).to(tl.float16)

        # -------------------------
        # STEP 2: chunked state update (replaces original NP2_STATELEN x BLOCK_N big block)
        # Semantics: conv_state <- concat(old_state, x)[-state_len_run:].
        # - If seqlen_run >= state_len_run: dst[:] = x[seqlen_run - state_len_run : seqlen_run]
        # - Else: keep = state_len_run - seqlen_run,
        #         dst[0:keep] = src[shift : shift+keep], dst[keep:keep+seqlen_run] = x[0:seqlen_run]
        # -------------------------
        # output cache line
        conv_states_offset = tl.load(conv_state_indices_ptr +
                                     b * stride_state_indices +
                                     current_last_index,
                                     mask=lane_active,
                                     other=0).to(tl.int64)

        use_shift = (seqlen_run < state_len_run)
        use_tail = (seqlen_run >= state_len_run)

        zero_i32 = tl.full((), 0, tl.int32)
        keep_shift = tl.where(use_shift, (state_len_run - seqlen_run),
                              zero_i32).to(tl.int32)
        tail_start = tl.where(use_tail, (seqlen_run - state_len_run),
                              zero_i32).to(tl.int32)

        # base pointers
        state_src_base = (conv_state_ptr +
                          conv_states_input_coord * stride_conv_state_seq +
                          conv_state_token_offset * stride_conv_state_tok +
                          idx_feats * stride_conv_state_dim)
        state_dst_base = (conv_state_ptr +
                          conv_states_offset * stride_conv_state_seq +
                          idx_feats * stride_conv_state_dim)

        x_base = x_ptr + x_offset + idx_feats * stride_x_dim

        # A) shift old state into dst[0:keep_shift)  (only when seqlen_run < state_len_run)
        for t0 in tl.static_range(0, NP2_STATELEN, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            src_tok = (dst_tok + shift).to(tl.int32)  # [T_CHUNK]
            m_tok = use_shift & (dst_tok < keep_shift) & (
                src_tok < state_len_run) & (dst_tok < state_len_run)
            m = (lane_active & m_tok)[:, None] & mask_w[None, :] & (
                conv_states_input_coord
                < num_cache_lines) & (conv_states_offset < num_cache_lines)

            src_ptrs = state_src_base[
                None, :] + src_tok[:, None] * stride_conv_state_tok
            dst_ptrs = state_dst_base[
                None, :] + dst_tok[:, None] * stride_conv_state_tok
            vals = tl.load(src_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, vals, mask=m)

        # B) append x into dst[keep_shift : keep_shift+seqlen_run) (only when seqlen_run < state_len_run)
        for t0 in tl.static_range(0, seqlen, T_CHUNK):
            x_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            dst_tok = (keep_shift + x_tok).to(tl.int32)  # [T_CHUNK]
            m_tok = use_shift & (x_tok < seqlen_run) & (dst_tok
                                                        < state_len_run)
            m = (lane_active & m_tok)[:, None] & mask_w[None, :] & (
                conv_states_offset < num_cache_lines)

            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = state_dst_base[
                None, :] + dst_tok[:, None] * stride_conv_state_tok
            x_vals = tl.load(x_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, x_vals, mask=m)

        # C) if seqlen_run >= state_len_run, overwrite dst with the tail of x
        for t0 in tl.static_range(0, NP2_STATELEN, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)  # [T_CHUNK]
            x_tok = (tail_start + dst_tok).to(tl.int32)  # [T_CHUNK]
            m_tok = use_tail & (dst_tok < state_len_run) & (x_tok < seqlen_run)
            m = (lane_active & m_tok)[:, None] & mask_w[None, :] & (
                conv_states_offset < num_cache_lines)

            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = state_dst_base[
                None, :] + dst_tok[:, None] * stride_conv_state_tok
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
                    matrix_x = tl.load(x_ptrs_1d,
                                       mask=lane_active & mask_w,
                                       other=0.0).to(tl.float16)
                    matrix_w = w_col0
                elif KERNEL_WIDTH == 2:
                    if j == 1:
                        matrix_w = w_col1
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d,
                                           mask=lane_active & mask_w,
                                           other=0.0).to(tl.float16)
                elif KERNEL_WIDTH == 3:
                    if j == 1:
                        matrix_w = w_col1
                        matrix_x = col1
                    elif j == 2:
                        matrix_w = w_col2
                        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                        matrix_x = tl.load(x_ptrs_1d,
                                           mask=lane_active & mask_w,
                                           other=0.0).to(tl.float16)
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
                        matrix_x = tl.load(x_ptrs_1d,
                                           mask=lane_active & mask_w,
                                           other=0.0).to(tl.float16)
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
                        matrix_x = tl.load(x_ptrs_1d,
                                           mask=lane_active & mask_w,
                                           other=0.0).to(tl.float16)
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
                        matrix_x = tl.load(x_ptrs_1d,
                                           mask=lane_active & mask_w,
                                           other=0.0).to(tl.float16)

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
        x = x.unsqueeze(-1)

    if query_start_loc is None:
        batch, dim, seqlen = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len

    _, width = weight.shape
    num_cache_lines, _, state_len_total = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert conv_state.stride(-2) == 1
        assert state_len_total >= width - 1
        assert num_cache_lines >= batch
        assert weight.stride(1) == 1

    # overwrite-on-x strategy same as original
    out = x

    stride_w_dim, stride_w_width = weight.stride()
    if query_start_loc is None:
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride(
    )
    stride_state_indices = conv_state_indices.stride(
        0) if conv_state_indices is not None else 0

    # effective state_len exactly as original
    if num_accepted_tokens is not None:
        eff_state_len = width - 1 + (seqlen - 1)
    else:
        eff_state_len = width - 1
    np2_statelen = triton.next_power_of_2(eff_state_len)

    # -------- tiling heuristic--------
    #keep program count around ~[80..160]
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
    t_chunk = 20 if block_n == 512 else 48

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
        out = out.squeeze(-1)
    return out.to(original_x_dtype)
