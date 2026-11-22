# SPDX-License-Identifier: Apache-2.0
"""Fused DFlash2 scoring and budgeted best-first search."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import _resolve_triton_ascend_op

index_select_simd = _resolve_triton_ascend_op("index_select_simd")


@triton.jit
def _pack_dflash2_score_operands_kernel(
    ids,
    hidden,
    root,
    pred,
    succ,
    left,
    right,
    PRED_VOCAB: tl.constexpr,
    SUCC_VOCAB: tl.constexpr,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
):
    panel = tl.program_id(0)
    batch_depth = panel // (C // ROWS)
    start = panel * ROWS
    row = tl.arange(0, ROWS)
    rank = tl.arange(0, R)
    child = tl.load(ids + start + row).to(tl.int32)
    if batch_depth % D == 0:
        parent = tl.full((ROWS,), tl.load(root + batch_depth // D).to(tl.int32), tl.int32)
    else:
        parent = tl.load(ids + start - C + row).to(tl.int32)
    pred_panel = index_select_simd(pred, 0, parent, [PRED_VOCAB, R], [-1, 0], [-1, R])
    succ_panel = index_select_simd(succ, 0, child, [SUCC_VOCAB, R], [-1, 0], [-1, R])
    state = tl.load(hidden + batch_depth * R + rank).to(tl.float32)
    weighted = (pred_panel.to(tl.float32) * state[None, :]).to(tl.bfloat16)
    offsets = (start + row[:, None]) * R + rank[None, :]
    tl.store(left + offsets, weighted)
    tl.store(right + offsets, succ_panel)


def _compute_dflash2_transition_bias(ids, hidden, root, pred, succ):
    requests, depth, candidates = ids.shape
    rank = hidden.shape[-1]
    rows = 16
    left = pred.new_empty((requests * depth, candidates, rank))
    right = succ.new_empty(left.shape)
    _pack_dflash2_score_operands_kernel[(requests * depth * candidates // rows,)](
        ids,
        hidden,
        root,
        pred,
        succ,
        left,
        right,
        pred.shape[0],
        succ.shape[0],
        rows,
        depth,
        candidates,
        rank,
        enable_auto_bind_sub_block=False,
    )
    return torch.bmm(left, right.transpose(1, 2)).view(requests, depth, candidates, candidates)


@triton.jit
def _add_edge(prefix, edge):
    score = prefix + edge
    low = -3.4028234663852886e38
    return tl.where(score == score, tl.maximum(score, low), low)


@triton.jit(do_not_specialize=["budget", "token_stride"])
def _build_tree_best_first_kernel(
    ids,
    values,
    slot_ids,
    out_tokens,
    out_depths,
    out_parents,
    out_num_nodes_ptr,
    out_visibility_ptr,
    out_first,
    out_sibling,
    budget,
    token_stride,
    DRAFT_DEPTH: tl.constexpr,
    NUM_CANDIDATES_PER_DEPTH: tl.constexpr,
    NUM_SORTED_CHILDREN: tl.constexpr,
    LANES: tl.constexpr,
    ID_BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    ids += req * DRAFT_DEPTH * NUM_CANDIDATES_PER_DEPTH
    values += req * DRAFT_DEPTH * NUM_CANDIDATES_PER_DEPTH * NUM_SORTED_CHILDREN
    slot_ids += req * DRAFT_DEPTH * NUM_CANDIDATES_PER_DEPTH * NUM_SORTED_CHILDREN
    lane = tl.arange(0, LANES)
    word = tl.arange(0, LANES // 32)
    pending = tl.where(lane == 0, _add_edge(0.0, tl.load(values)), -float("inf"))
    prefixes = tl.full((LANES,), 0.0, tl.float32)
    cursors = tl.full((LANES,), 0, tl.int32)
    first = tl.full((LANES,), -1, tl.int32)
    descriptor = tl.full((LANES,), 0, tl.int32)
    masks = tl.full((LANES // 32, LANES), 0, tl.uint32)
    # Child row probabilities are sorted. Adding the same prefix preserves
    # this order. Rounded ties are resolved in favor of the earlier child.
    for node in tl.range(1, budget + 1):
        best = tl.max(pending, 0)
        parent = tl.min(tl.where(pending == best, lane, LANES), 0)
        own = lane == parent
        cursor = tl.sum(tl.where(own, cursors, 0), 0)
        prefix = tl.sum(tl.where(own, prefixes, 0.0), 0)
        slot = tl.load(slot_ids + cursor).to(tl.int32)
        depth = cursor // (NUM_CANDIDATES_PER_DEPTH * NUM_SORTED_CHILDREN)
        row = ((depth + 1) * NUM_CANDIDATES_PER_DEPTH + slot) * NUM_SORTED_CHILDREN
        next_score = _add_edge(
            prefix,
            tl.load(values + tl.minimum(cursor + 1, DRAFT_DEPTH * NUM_CANDIDATES_PER_DEPTH * NUM_SORTED_CHILDREN - 1)),
        )
        child_score = _add_edge(
            best,
            tl.load(values + tl.minimum(row, DRAFT_DEPTH * NUM_CANDIDATES_PER_DEPTH * NUM_SORTED_CHILDREN - 1)),
        )
        next_score = tl.where(cursor % NUM_SORTED_CHILDREN < NUM_SORTED_CHILDREN - 1, next_score, -float("inf"))
        child_score = tl.where(depth + 1 < DRAFT_DEPTH, child_score, -float("inf"))
        pending = tl.where(lane == node, child_score, tl.where(own, next_score, pending))
        cursors = tl.where(lane == node, row, tl.where(own, cursor + 1, cursors))
        prefixes = tl.where(lane == node, best, prefixes)
        previous = tl.sum(tl.where(own, first, 0), 0)
        first = tl.where(own, node, first)
        # Parent IDs use the lane capacity as their radix. The final node
        # never becomes a parent, even when its ID equals LANES.
        record = depth * NUM_CANDIDATES_PER_DEPTH + slot + ID_BLOCK * (parent + LANES * (previous + 1))
        descriptor = tl.where(lane == node - 1, record, descriptor)
        inherited = tl.sum(tl.where(lane[None, :] + 1 == parent, masks, 0), 1).to(tl.uint32)
        bit = tl.full((), 1, tl.uint32) << ((node - 1) % 32)
        visible = inherited | tl.where(word == (node - 1) // 32, bit, 0)
        masks = tl.where(lane[None, :] == node - 1, visible[:, None], masks)

    token_index = descriptor % ID_BLOCK
    all_tokens = tl.arange(0, ID_BLOCK)
    ids_ub = tl.load(ids + all_tokens, all_tokens < DRAFT_DEPTH * NUM_CANDIDATES_PER_DEPTH, other=0).to(tl.int32)
    valid = lane < budget
    tl.store(out_tokens + req * token_stride + lane, tl.gather(ids_ub, token_index, 0), valid)
    tl.store(out_depths + req * budget + lane, token_index // NUM_CANDIDATES_PER_DEPTH + 1, valid)
    tl.store(out_parents + req * budget + lane, descriptor // ID_BLOCK % LANES, valid)
    tl.store(out_num_nodes_ptr + req, budget)
    out_first += req * (budget + 1)
    out_sibling += req * (budget + 1)
    tl.store(out_first + lane, first, valid)
    tl.store(out_first + budget, -1)
    sibling = descriptor // (ID_BLOCK * LANES) - 1
    tl.store(out_sibling + lane, tl.where(lane == 0, -1, tl.gather(sibling, tl.maximum(lane - 1, 0), 0)), valid)
    tl.store(out_sibling + budget, tl.sum(tl.where(lane == budget - 1, sibling, 0), 0))
    bits = tl.arange(0, 32)
    for w in tl.static_range(LANES // 32):
        columns = w * 32 + bits
        mask_word = tl.sum(tl.where(word[:, None] == w, masks, 0), 0).to(tl.uint32)
        visibility = (mask_word[:, None] >> bits[None, :]) & 1
        tl.store(
            out_visibility_ptr + (req * budget + lane[:, None]) * budget + columns[None, :],
            visibility,
            valid[:, None] & (columns[None, :] < budget),
        )


def build_tree_from_sorted_child_scores(
    candidate_token_ids, sorted_child_log_probs, sorted_child_candidate_slots, layout
):
    requests, depth, candidates = candidate_token_ids.shape
    budget = layout.tokens.shape[1]
    _build_tree_best_first_kernel[(requests,)](
        candidate_token_ids,
        sorted_child_log_probs,
        sorted_child_candidate_slots,
        layout.tokens,
        layout.depths,
        layout.parents,
        layout.num_nodes,
        layout.visibility.view(torch.int8),
        layout.first_child,
        layout.next_sibling,
        budget,
        layout.tokens.stride(0),
        depth,
        candidates,
        sorted_child_log_probs.shape[-1],
        max(32, triton.next_power_of_2(budget)),
        triton.next_power_of_2(depth * candidates),
        enable_auto_bind_sub_block=False,
    )
    return layout


def build_dflash2_tree(
    candidate_ids, unary_logits, projected_hidden, root_token_ids, predecessor_codebook, successor_codebook, layout
):
    """Fill each request's configured budget from its normalized transition rows."""
    corrections = _compute_dflash2_transition_bias(
        candidate_ids, projected_hidden, root_token_ids, predecessor_codebook, successor_codebook
    )
    scores = corrections.float() + unary_logits.float().unsqueeze(2)
    # At most B siblings per row can be selected. Sorting a whole lane bucket
    # shares the compiled kernel between budgets, including non-power-of-two B.
    width = min(candidate_ids.shape[-1], max(32, triton.next_power_of_2(layout.tokens.shape[1])))
    values, slots = torch.topk(torch.log_softmax(scores, dim=-1), width, dim=-1, sorted=True)
    return build_tree_from_sorted_child_scores(candidate_ids, values, slots, layout)
