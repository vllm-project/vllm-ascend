# SPDX-License-Identifier: Apache-2.0
# Best-first ordering adapted from SGLang's merged UNO tree builder,
# commit 2bb25dc18bc27321fb116bbefcaddaf13c1c1754 (Apache-2.0).
# Register-resident search state and ancestor bitsets are Ascend glue.
"""Fuse fixed-budget UNO tree selection and ancestor-mask construction."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _build_tree(
    roots,
    candidate_tokens,
    candidate_log_probs,
    tokens,
    parents,
    depths_out,
    allowed,
    DEPTHS: tl.constexpr,
    K: tl.constexpr,
    NODES: tl.constexpr,
    BLOCK_NODES: tl.constexpr,
    BLOCK_CANDIDATES: tl.constexpr,
):
    batch = tl.program_id(0)
    nodes = tl.arange(0, BLOCK_NODES)
    slots = tl.arange(0, BLOCK_CANDIDATES)
    candidate_parents = slots // K
    ranks = slots % K
    in_bounds = slots < NODES * K
    safe_parents = tl.minimum(candidate_parents, BLOCK_NODES - 1)
    # Triton-Ascend gather accepts floating input but not int32. Depths are
    # bounded by 15 and therefore represented exactly in FP32 registers.
    depths = tl.zeros((BLOCK_NODES,), tl.float32)
    masses = tl.zeros((BLOCK_NODES,), tl.float32)
    ancestry = tl.where(nodes == 0, 1, 0).to(tl.uint32)
    used = tl.zeros((BLOCK_CANDIDATES,), tl.int1)
    tl.store(tokens + batch * NODES, tl.load(roots + batch))
    tl.store(parents + batch * NODES, -1)
    for node in range(1, NODES):
        parent_depth = tl.gather(depths, safe_parents, 0).to(tl.int32)
        parent_mass = tl.gather(masses, safe_parents, 0)
        valid = in_bounds & (candidate_parents < node) & (parent_depth < DEPTHS) & ~used
        offset = batch * DEPTHS * K + tl.minimum(parent_depth, DEPTHS - 1) * K + ranks
        candidate_id = tl.load(candidate_tokens + offset, mask=valid, other=0)
        score = parent_mass + tl.load(candidate_log_probs + offset, mask=valid, other=-float("inf"))
        best = tl.max(tl.where(valid, score, -float("inf")), 0)
        winner = valid & (score == best)
        best_depth = tl.min(tl.where(winner, parent_depth, 1 << 30), 0)
        winner &= parent_depth == best_depth
        best_rank = tl.min(tl.where(winner, ranks, 1 << 30), 0)
        winner &= ranks == best_rank
        best_token = tl.min(tl.where(winner, candidate_id, 1 << 30), 0)
        winner &= candidate_id == best_token
        best_parent = tl.min(tl.where(winner, candidate_parents, 1 << 30), 0)
        winner &= candidate_parents == best_parent
        selected = tl.min(tl.where(winner, slots, 1 << 30), 0)
        used |= slots == selected
        parent_ancestry = tl.sum(tl.where(nodes == best_parent, ancestry, 0), 0)
        new_ancestry = parent_ancestry | (tl.full((), 1, tl.uint32) << node)
        ancestry = tl.where(nodes == node, new_ancestry, ancestry)
        depths = tl.where(nodes == node, (best_depth + 1).to(tl.float32), depths)
        masses = tl.where(nodes == node, best, masses)
        tl.store(tokens + batch * NODES + node, best_token)
        tl.store(parents + batch * NODES + node, best_parent)
    tl.store(depths_out + batch * NODES + nodes, depths.to(tl.int32), mask=nodes < NODES)
    matrix = ((ancestry[:, None] >> nodes[None, :]) & 1) != 0
    tl.store(
        allowed + batch * NODES * NODES + nodes[:, None] * NODES + nodes[None, :],
        matrix,
        mask=(nodes[:, None] < NODES) & (nodes[None, :] < NODES),
    )


def build_tree_from_candidates(roots, candidate_tokens, candidate_log_probs, nodes):
    if not 1 <= nodes <= 32:
        raise ValueError("The fused UNO tree uses 32-bit ancestor masks and supports at most 32 nodes.")
    batch, depth, top_k = candidate_tokens.shape
    tokens = torch.empty((batch, nodes), device=roots.device, dtype=torch.int64)
    parents = torch.empty_like(tokens)
    depths = torch.empty_like(tokens)
    allowed = torch.empty((batch, nodes, nodes), device=roots.device, dtype=torch.bool)
    _build_tree[(batch,)](
        roots,
        candidate_tokens.contiguous(),
        candidate_log_probs.contiguous(),
        tokens,
        parents,
        depths,
        allowed,
        DEPTHS=depth,
        K=top_k,
        NODES=nodes,
        BLOCK_NODES=triton.next_power_of_2(nodes),
        BLOCK_CANDIDATES=triton.next_power_of_2(nodes * top_k),
    )
    return tokens, parents, depths, allowed


@triton.jit
def _traverse_tree(
    Tokens, Parents, Samples, Output, Rows, NODES: tl.constexpr, OUTPUT: tl.constexpr, BLOCK: tl.constexpr
):
    batch = tl.program_id(0)
    indices = tl.arange(0, BLOCK)
    tokens = tl.load(Tokens + batch * NODES + indices, indices < NODES, other=-1)
    parents = tl.load(Parents + batch * NODES + indices, indices < NODES, other=-2)
    current = tl.full((), 0, tl.int32)
    active = tl.full((), True, tl.int1)
    tl.store(Output + batch * OUTPUT, tl.load(Tokens + batch * NODES))
    tl.store(Rows + batch * OUTPUT, 0)
    for step in range(1, OUTPUT):
        sampled = tl.load(Samples + batch * NODES + current)
        tl.store(Output + batch * OUTPUT + step, tl.where(active, sampled, -1))
        matching = (indices < NODES) & (parents == current) & (tokens == sampled)
        child = tl.min(tl.where(matching, indices, NODES), 0)
        active &= child < NODES
        current = tl.minimum(child, NODES - 1)
        tl.store(Rows + batch * OUTPUT + step, tl.where(active, child, -1))


def traverse_tree(tokens, parents, target_samples, output_width):
    batch, nodes = tokens.shape
    output = torch.empty((batch, output_width), device=tokens.device, dtype=torch.int64)
    rows = torch.empty_like(output)
    _traverse_tree[(batch,)](
        tokens.contiguous(),
        parents.contiguous(),
        target_samples.contiguous(),
        output,
        rows,
        NODES=nodes,
        OUTPUT=output_width,
        BLOCK=triton.next_power_of_2(nodes),
    )
    return output, rows


@triton.jit
def _compact_tree_kv(
    CachePointers,
    Slots,
    AcceptedRows,
    ROWS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    cache = tl.program_id(0)
    chunk = tl.program_id(1)
    # Every program owns a disjoint byte range in one cache. Load all source
    # rows before any destination write, preserving cross-row aliases.
    pointer = tl.load(CachePointers + cache).to(tl.pointer_type(tl.uint8))
    rows = tl.arange(0, BLOCK_ROWS)
    offsets = chunk * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    accepted = tl.load(AcceptedRows + rows, rows < ROWS, other=-1)
    sources = tl.where(accepted >= 0, accepted + 1, rows + 1)
    source_slots = tl.load(Slots + sources, rows < ROWS, other=0).to(tl.int64)
    destination_slots = tl.load(Slots + rows + 1, rows < ROWS, other=0).to(tl.int64)
    mask = (rows[:, None] < ROWS) & (offsets[None, :] < ROW_BYTES)
    values = tl.load(pointer + source_slots[:, None] * ROW_BYTES + offsets[None, :], mask=mask, other=0)
    tl.store(pointer + destination_slots[:, None] * ROW_BYTES + offsets[None, :], values, mask=mask)


def compact_tree_kv(cache_pointers, row_bytes, physical_slots, accepted_rows):
    rows = accepted_rows.numel()
    _compact_tree_kv[(cache_pointers.numel(), triton.cdiv(row_bytes, 256))](
        cache_pointers,
        physical_slots,
        accepted_rows,
        ROWS=rows,
        ROW_BYTES=row_bytes,
        BLOCK_ROWS=triton.next_power_of_2(rows),
        BLOCK_BYTES=256,
    )
