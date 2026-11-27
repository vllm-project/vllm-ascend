# SPDX-License-Identifier: Apache-2.0
"""Fused decode Engram lookback/hash preparation; no tensor upload in wrapper.

Contract: GPU query boundaries start at zero, are nondecreasing, and each
request has at most max_query_len tokens. query_start_loc[num_reqs] <= output T.
Only launch on already valid request-state indices. CPU offloaded table lookup
remains the caller's responsibility.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _fused_engram_hash_kernel(
    ids,
    starts,
    mapping,
    computed,
    all_ids,
    token_map,
    multipliers,
    primes,
    offsets,
    output,
    rope_positions,
    rope_prefill,
    rope_delta,
    prefill_lens,
    packed_output,
    ALL_STRIDE: tl.constexpr,
    OUT_LAYER_STRIDE: tl.constexpr,
    OUT_TOKEN_STRIDE: tl.constexpr,
    OUT_HEAD_STRIDE: tl.constexpr,
    NREQ: tl.constexpr,
    NTOK: tl.constexpr,
    VOCAB: tl.constexpr,
    DEPTH: tl.constexpr,
    NHEAD: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    MAX_QUERY: tl.constexpr,
    PAD_ID: tl.constexpr,
    PAD_TOKEN: tl.constexpr,
    VISUAL_IDS: tl.constexpr,
    BLOCK: tl.constexpr,
    PAD_BLOCK: tl.constexpr,
    WITH_ROPE: tl.constexpr,
    ROPE_DIMS: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    PREFILL_STRIDE: tl.constexpr,
    WITH_PACKED_OUTPUT: tl.constexpr,
):
    layer = tl.program_id(0)
    req = tl.program_id(1)
    if req == NREQ:
        # Disjoint stores from request programs; covers even an all-empty batch.
        end = tl.load(starts + NREQ)
        pad_lane = tl.arange(0, PAD_BLOCK)
        token = pad_lane // TOTAL_HEADS
        pad_head = pad_lane % TOTAL_HEADS
        tl.store(
            output + layer * OUT_LAYER_STRIDE + token * OUT_TOKEN_STRIDE + pad_head * OUT_HEAD_STRIDE,
            -1,
            (token >= end) & (token < NTOK),
        )
        if WITH_PACKED_OUTPUT:
            tl.store(
                packed_output + (layer * NTOK + token) * TOTAL_HEADS + pad_head, -1, (token >= end) & (token < NTOK)
            )
    else:
        first = tl.load(starts + req)
        stop = tl.load(starts + req + 1)
        state = tl.load(mapping + req)
        num_computed = tl.load(computed + state)
        lane = tl.arange(0, BLOCK)
        local = lane // TOTAL_HEADS
        head = lane % TOTAL_HEADS
        position = first + local
        live = (local < MAX_QUERY) & (position < stop) & (position < NTOK)
        if WITH_ROPE:
            if layer == 0:
                # Exactly one head lane owns each output position. This keeps
                # RoPE outside the Target graph and avoids inter-program writes.
                rope_live = live & (head == 0)
                prefill_len = tl.load(prefill_lens + state)
                delta = tl.load(rope_delta + state)
                orig_pos = num_computed + local
                for dim in tl.static_range(ROPE_DIMS):
                    if num_computed < prefill_len:
                        pos = tl.load(
                            rope_prefill + (state * ROPE_DIMS + dim) * PREFILL_STRIDE + tl.maximum(orig_pos, 0),
                            rope_live,
                            other=0,
                        )
                    else:
                        pos = orig_pos + delta
                    tl.store(rope_positions + dim * ROPE_STRIDE + position, pos, rope_live)
        current = tl.load(ids + tl.maximum(position, 0), live, other=-1).to(tl.int64)
        invalid = (current < 0) | (current >= VOCAB) | (current == PAD_TOKEN)
        for visual in tl.static_range(len(VISUAL_IDS)):
            invalid = invalid | (current == VISUAL_IDS[visual])
        rolling = tl.full((BLOCK,), 0, tl.int64)
        blocked = tl.full((BLOCK,), False, tl.int1)
        for shift in tl.static_range(DEPTH + 1):
            preceding = local - shift
            in_current = preceding >= 0
            input_pos = tl.maximum(position - shift, 0)
            history_pos = num_computed + preceding
            safe_history_pos = tl.maximum(history_pos, 0)
            current_source = tl.load(ids + input_pos, live & in_current, other=-1)
            history_source = tl.load(
                all_ids + state * ALL_STRIDE + safe_history_pos,
                live & ~in_current & (history_pos >= 0),
                other=-1,
            )
            source = tl.where(in_current, current_source, history_source).to(tl.int64)
            blocked = blocked | (source < 0) | (source >= VOCAB)
            safe_source = tl.minimum(tl.maximum(source, 0), VOCAB - 1)
            mapped = tl.load(token_map + safe_source, live, other=0).to(tl.int64)
            mapped = tl.where(blocked, PAD_ID, mapped).to(tl.int64)
            factor = tl.load(multipliers + layer * (DEPTH + 1) + shift)
            rolling = rolling ^ (mapped * factor)
            if shift > 0:
                order_head = (shift - 1) * NHEAD + head % NHEAD
                prime = tl.load(primes + layer * TOTAL_HEADS + order_head)
                offset = tl.load(offsets + layer * TOTAL_HEADS + order_head)
                # Triton integer remainder is truncating. PyTorch '%' with a
                # positive prime is floor remainder, also for signed overflow.
                rem = rolling % prime
                rem = tl.where(rem < 0, rem + prime, rem)
                value = tl.where(invalid, -1, rem + offset).to(tl.int32)
                write = live & (head // NHEAD == shift - 1)
                tl.store(
                    output + layer * OUT_LAYER_STRIDE + position * OUT_TOKEN_STRIDE + head * OUT_HEAD_STRIDE,
                    value,
                    write,
                )
                if WITH_PACKED_OUTPUT:
                    tl.store(packed_output + (layer * NTOK + position) * TOTAL_HEADS + head, value, write)


def fused_engram_hashes(
    input_ids,
    query_start_loc,
    idx_mapping,
    num_computed_tokens,
    all_token_ids,
    token_map,
    multipliers,
    primes,
    offsets,
    pad_id,
    pad_token_id,
    visual_ids,
    output,
    *,
    num_reqs=None,
    max_query_len=6,
    rope_state=None,
    prefill_lens=None,
    packed_output=None,
):
    """Write integer-exact hashes including -1 padding in one kernel launch.

    All tensor inputs are NPU tensors. max_query_len is a host-side upper bound,
    not a device reduction; caller must ensure it covers the batch's query size.
    Output may be a noncontiguous slice of a larger capacity allocation.
    """
    nreq = idx_mapping.numel() if num_reqs is None else num_reqs
    layers, ntok, total_heads = output.shape
    depth = multipliers.shape[1] - 1
    if depth < 1 or primes.shape != (layers, depth, total_heads // depth):
        raise ValueError("Invalid Engram prime/head shape")
    if total_heads % depth or output.dtype != torch.int32:
        raise ValueError("Expected flattened int32 Engram heads")
    if query_start_loc.numel() < nreq + 1 or idx_mapping.numel() < nreq:
        raise ValueError("Insufficient request metadata")
    if max_query_len < 1 or token_map.numel() < 1:
        raise ValueError("Empty token map or invalid query bound")
    if not all(
        t.is_contiguous()
        for t in (input_ids, query_start_loc, idx_mapping, num_computed_tokens, token_map, multipliers, primes, offsets)
    ):
        raise ValueError("Engram metadata must be contiguous")
    if all_token_ids.stride(1) != 1:
        raise ValueError("Token histories need contiguous rows")
    if packed_output is not None and (
        packed_output.shape != output.shape
        or packed_output.dtype != output.dtype
        or packed_output.device != output.device
        or not packed_output.is_contiguous()
    ):
        raise ValueError("Packed hash destination must match shape/dtype/device and be contiguous")
    if layers == 0 or ntok == 0:
        return output
    if rope_state is not None:
        if prefill_lens is None or not prefill_lens.is_contiguous():
            raise ValueError("RoPE fusion needs contiguous device prefill lengths")
        positions = rope_state.positions
        prefill = rope_state.prefill_positions.gpu
        delta = rope_state.prefill_delta.gpu
        if positions.stride(1) != 1 or not prefill.is_contiguous() or not delta.is_contiguous():
            raise ValueError("Unsupported RoPE buffer strides")
        dims, pos_stride, prefill_stride = rope_state.num_dims, positions.stride(0), rope_state.max_model_len
    else:
        positions = prefill = delta = prefill_lens = input_ids
        dims, pos_stride, prefill_stride = 0, 0, 0
    _fused_engram_hash_kernel[(layers, nreq + 1)](
        input_ids,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
        token_map,
        multipliers,
        primes,
        offsets,
        output,
        positions,
        prefill,
        delta,
        prefill_lens,
        output if packed_output is None else packed_output,
        all_token_ids.stride(0),
        *output.stride(),
        nreq,
        ntok,
        token_map.numel(),
        depth,
        total_heads // depth,
        total_heads,
        max_query_len,
        pad_id,
        pad_token_id,
        tuple(visual_ids),
        triton.next_power_of_2(max_query_len * total_heads),
        triton.next_power_of_2(ntok * total_heads),
        rope_state is not None,
        dims,
        pos_stride,
        prefill_stride,
        packed_output is not None,
    )
    return output
