# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""EPLB tensor operations without MoE execution or registration dependencies."""

import torch

EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS = 1024


def make_allgather_mask(
    num_tokens: int,
    num_unpadded_tokens: torch.Tensor,
    record_enabled: torch.Tensor,
    *,
    is_padding: torch.Tensor | None = None,
    tp_size: int = 1,
    tp_rank: int = 0,
) -> torch.Tensor:
    """Describe valid source rows before the same gather/unpad as MoE inputs."""
    start = tp_rank * num_tokens
    if is_padding is None:
        rows = torch.arange(start, start + num_tokens, device=num_unpadded_tokens.device)
        valid = rows < num_unpadded_tokens
    elif is_padding.numel() == num_tokens:
        valid = ~is_padding
    else:
        padding = max(0, num_tokens * tp_size - is_padding.numel())
        padded = torch.nn.functional.pad(is_padding[: num_tokens * tp_size], (0, padding), value=True)
        valid = ~padded[start : start + num_tokens]
    return valid & record_enabled.to(torch.bool)


def build_expert_replica_routing_table(
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    ep_rank: int,
) -> torch.Tensor:
    """Build a rank-aware table for routing logical experts to replicas."""
    if logical_to_physical_map.ndim != 2:
        raise ValueError("logical_to_physical_map must be a 2D tensor.")
    if logical_replica_count.ndim != 1:
        raise ValueError("logical_replica_count must be a 1D tensor.")
    if logical_to_physical_map.shape[0] != logical_replica_count.shape[0]:
        raise ValueError("Logical expert dimensions must match.")

    num_logical_experts = logical_replica_count.shape[0]
    device = logical_to_physical_map.device
    table_rows = torch.arange(
        EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS,
        dtype=torch.int64,
        device=device,
    )[:, None]
    logical_expert_ids = torch.arange(num_logical_experts, dtype=torch.int64, device=device)[None, :]
    replica_count = logical_replica_count.to(torch.int64).clamp_min(1)[None, :]
    replica_indices = (table_rows + ep_rank + logical_expert_ids) % replica_count
    routing_table = logical_to_physical_map.gather(1, replica_indices.T).T
    return routing_table.to(torch.int32).contiguous()


def count_allgather_tokens(
    topk_ids: torch.Tensor,
    token_mask: torch.Tensor,
    local_expert_start: int,
    local_expert_count: int,
) -> torch.Tensor:
    """Count eligible routes owned locally without changing the GMM counts."""
    local_ids = topk_ids.to(torch.int64) - local_expert_start
    eligible = token_mask[:, None] & (local_ids >= 0) & (local_ids < local_expert_count)
    indices = local_ids.clamp(0, local_expert_count - 1).reshape(-1)
    counts = torch.zeros(local_expert_count, dtype=torch.int64, device=topk_ids.device)
    counts.scatter_add_(0, indices, eligible.to(torch.int64).reshape(-1))
    return counts
