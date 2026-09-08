# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Fixed-budget UNO proposal trees and target-sampled path traversal.

Proposal ordering follows SGLang's merged UNO implementation at
2bb25dc18bc27321fb116bbefcaddaf13c1c1754. These tensor operations are an
Ascend implementation; they do not depend on EAGLE's CUDA lineage ABI.
"""

from dataclasses import dataclass

import torch


@dataclass
class UnoTreeProposal:
    tokens: torch.Tensor
    parents: torch.Tensor
    depths: torch.Tensor
    allowed_attention: torch.Tensor


def build_uno_tree(
    root_tokens: torch.Tensor,
    draft_logits: torch.Tensor,
    *,
    max_nodes: int,
    candidate_top_k: int,
    temperature: torch.Tensor,
    use_fused_kernel: bool = True,
) -> UnoTreeProposal:
    """Select a prefix-closed tree using full-vocabulary proposal log mass.

    ``draft_logits`` contains the F-1 noised rows, excluding the clean root.
    Ties prefer shallower depth, lower candidate rank, token ID, then parent.
    No device value is read by the host while constructing the tree.
    """
    if draft_logits.ndim != 3 or root_tokens.shape != draft_logits.shape[:1]:
        raise ValueError("UNO tree expects roots [batch] and logits [batch, depth, vocabulary].")
    batch, depth_limit, vocab = draft_logits.shape
    if not 1 <= candidate_top_k <= vocab:
        raise ValueError("UNO tree candidate_top_k must be within the vocabulary.")
    if not depth_limit + 1 <= max_nodes <= 128:
        raise ValueError("UNO tree node budget must include the draft width and be <= 128.")
    capacity, level = 1, 1
    for _ in range(depth_limit):
        level *= candidate_top_k
        capacity += level
        if capacity >= max_nodes:
            break
    if capacity < max_nodes:
        raise ValueError("UNO candidate set cannot fill the requested tree budget.")
    if temperature.numel() not in (1, batch):
        raise ValueError("UNO tree temperature must be scalar or one value per request.")
    device = draft_logits.device
    if root_tokens.device != device or temperature.device != device:
        raise ValueError("UNO roots, logits and temperature must share a device.")

    scaled = draft_logits.float() / torch.where(temperature > 0, temperature, 1).reshape(-1, 1, 1)
    values, ids = torch.topk(scaled, candidate_top_k, dim=-1, sorted=True)
    log_probs = values - torch.logsumexp(scaled, dim=-1, keepdim=True)
    log_probs = torch.nan_to_num(log_probs, nan=-torch.inf, neginf=-torch.inf).clamp_max_(0)
    if use_fused_kernel and device.type == "npu" and max_nodes <= 32:
        # Keep the tensor implementation as the independent reference and for
        # node budgets wider than the fused kernel's 32-bit ancestor bitset.
        from vllm_ascend.ops.triton.uno_tree import build_tree_from_candidates

        return UnoTreeProposal(*build_tree_from_candidates(root_tokens, ids, log_probs, max_nodes))
    tokens = torch.zeros((batch, max_nodes), dtype=torch.int64, device=device)
    parents = torch.full_like(tokens, -1)
    depths = torch.zeros_like(tokens)
    masses = torch.zeros((batch, max_nodes), dtype=torch.float32, device=device)
    allowed = torch.zeros((batch, max_nodes, max_nodes), dtype=torch.bool, device=device)
    tokens[:, 0] = root_tokens
    allowed[:, 0, 0] = True
    used = torch.zeros((batch, max_nodes * candidate_top_k), dtype=torch.bool, device=device)
    slots = torch.arange(max_nodes * candidate_top_k, device=device)
    candidate_parents = slots // candidate_top_k
    ranks = slots % candidate_top_k
    batches = torch.arange(batch, device=device)[:, None]
    sentinel = torch.iinfo(torch.int64).max

    for node in range(1, max_nodes):
        parent_depth = depths[:, candidate_parents]
        valid = (candidate_parents < node) & (parent_depth < depth_limit) & ~used
        safe_depth = parent_depth.clamp_max(depth_limit - 1)
        candidate_tokens = ids[batches, safe_depth, ranks]
        candidate_mass = masses[:, candidate_parents] + log_probs[batches, safe_depth, ranks]
        best_mass = candidate_mass.masked_fill(~valid, -torch.inf).amax(dim=1, keepdim=True)
        winner = valid & (candidate_mass == best_mass)
        for tie_value in (parent_depth, ranks[None], candidate_tokens, candidate_parents[None]):
            best = torch.where(winner, tie_value, sentinel).amin(dim=1, keepdim=True)
            winner = winner & (tie_value == best)
        selected = winner.to(torch.int32).argmax(dim=1)
        parent = candidate_parents[selected]
        row = batches[:, 0]
        tokens[:, node] = candidate_tokens[row, selected]
        parents[:, node] = parent
        depths[:, node] = depths[row, parent] + 1
        masses[:, node] = candidate_mass[row, selected]
        allowed[:, node, :] = allowed[row, parent, :]
        allowed[:, node, node] = True
        used[row, selected] = True
    return UnoTreeProposal(tokens, parents, depths, allowed)


def traverse_uno_tree(
    proposal: UnoTreeProposal,
    target_samples: torch.Tensor,
    max_output_tokens: int,
    *,
    use_fused_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Follow target samples through matching children, then emit a bonus.

    The clean root is already sampled from the base model. Each tree node's
    target sample is independent of proposal ranking, so a missing child ends
    the path without changing the target distribution. Returned KV row indices
    include accepted input nodes and exclude the final uncached bonus token.
    """
    if target_samples.shape != proposal.tokens.shape:
        raise ValueError("UNO target samples must match the tree node shape.")
    batch, nodes = proposal.tokens.shape
    if max_output_tokens < 2:
        raise ValueError("UNO tree output must have room for root and bonus.")
    device = proposal.tokens.device
    if use_fused_kernel and device.type == "npu" and nodes <= 32:
        from vllm_ascend.ops.triton.uno_tree import traverse_tree

        return traverse_tree(proposal.tokens, proposal.parents, target_samples, max_output_tokens)
    output = torch.full((batch, max_output_tokens), -1, dtype=torch.int64, device=device)
    kv_rows = torch.full_like(output, -1)
    output[:, 0] = proposal.tokens[:, 0]
    kv_rows[:, 0] = 0
    current = torch.zeros(batch, dtype=torch.int64, device=device)
    active = torch.ones(batch, dtype=torch.bool, device=device)
    batch_rows = torch.arange(batch, device=device)
    node_ids = torch.arange(nodes, device=device)[None]
    for step in range(1, max_output_tokens):
        sampled = target_samples[batch_rows, current]
        output[:, step] = torch.where(active, sampled, -1)
        matching = (proposal.parents == current[:, None]) & (proposal.tokens == sampled[:, None])
        child = torch.where(matching, node_ids, nodes).amin(dim=1)
        active = active & (child < nodes)
        current = child.clamp_max(nodes - 1)
        kv_rows[:, step] = torch.where(active, child, -1)
    return output, kv_rows


def fill_uno_tree_mask(mask, key_positions, frontier, allowed):
    """Mask physical tree rows while retaining every committed prefix key."""
    relative = key_positions - frontier
    width = allowed.shape[-1]
    in_tree = (relative >= 0) & (relative < width)
    tree_allowed = allowed[:, relative.clamp(0, width - 1)] & in_tree
    mask.copy_(((relative >= 0) & ~tree_allowed)[None, None])


def compact_uno_tree_kv(cache, physical_slots, accepted_rows):
    """Gather before writing, including when an accepted branch aliases a destination.

    ``physical_slots[0]`` holds the seed; tree row zero starts at slot one.
    Padded destinations copy themselves, keeping all indices static in shape.
    """
    count = accepted_rows.numel()
    destinations = physical_slots[1 : count + 1].long()
    source_rows = torch.where(
        accepted_rows >= 0,
        accepted_rows + 1,
        torch.arange(1, count + 1, device=accepted_rows.device),
    )
    sources = physical_slots[source_rows].long()
    # The two leading dimensions are page and offset within page.
    flat = cache.view(-1, *cache.shape[2:])
    values = flat.index_select(0, sources)
    flat.index_copy_(0, destinations, values)


class UnoTreeKVCompactor:
    """Reuse a device pointer table for all homogeneous GQA key/value caches."""

    def __init__(self, caches):
        self.caches = tuple({cache.data_ptr(): cache for cache in caches}.values())
        if not self.caches:
            raise ValueError("UNO tree requires populated KV caches.")
        first = self.caches[0]
        shape, dtype, device = first.shape, first.dtype, first.device
        if any(
            cache.shape != shape or cache.dtype != dtype or cache.device != device or not cache.is_contiguous()
            for cache in self.caches
        ):
            raise ValueError("UNO tree fused compaction requires homogeneous contiguous KV caches.")
        self.addresses = tuple(cache.data_ptr() for cache in self.caches)
        self.pointers = torch.tensor(self.addresses, dtype=torch.int64, device=device)
        self.row_bytes = first.numel() // (shape[0] * shape[1]) * first.element_size()

    def compact(self, physical_slots, accepted_rows):
        if self.pointers.device.type == "npu":
            from vllm_ascend.ops.triton.uno_tree import compact_tree_kv

            compact_tree_kv(self.pointers, self.row_bytes, physical_slots, accepted_rows)
        else:
            for cache in self.caches:
                compact_uno_tree_kv(cache, physical_slots, accepted_rows)
