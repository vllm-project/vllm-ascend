# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.utils.torch_utils import direct_register_custom_op

EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS = 1024


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


def map_to_physical(
    topk_ids: torch.Tensor,
    expert_replica_routing_table: torch.Tensor,
) -> torch.Tensor:
    """Map logical expert IDs to global physical IDs through a periodic table."""
    if topk_ids.numel() == 0:
        return topk_ids
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must be a 2D tensor.")

    logical_ids = topk_ids.to(torch.int64) if topk_ids.device.type == "cpu" else topk_ids
    num_rows, topk = topk_ids.shape
    num_full_blocks, tail_rows = divmod(
        num_rows,
        EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS,
    )
    mapped_blocks = []

    if num_full_blocks:
        full_rows = num_full_blocks * EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS
        routing_table_blocks = expert_replica_routing_table.view(
            1,
            EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS,
            expert_replica_routing_table.shape[1],
        ).expand(num_full_blocks, -1, -1)
        logical_id_blocks = logical_ids[:full_rows].view(
            num_full_blocks,
            EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS,
            topk,
        )
        mapped_blocks.append(
            torch.gather(routing_table_blocks, 2, logical_id_blocks).reshape(
                full_rows,
                topk,
            )
        )

    if tail_rows:
        mapped_blocks.append(
            torch.gather(
                expert_replica_routing_table[:tail_rows],
                1,
                logical_ids[num_full_blocks * EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS :],
            )
        )

    physical_ids = mapped_blocks[0] if len(mapped_blocks) == 1 else torch.cat(mapped_blocks)
    return physical_ids if physical_ids.dtype == topk_ids.dtype else physical_ids.to(topk_ids.dtype)


def record_local_expert_load(
    expert_tokens: torch.Tensor,
    group_list_type: int,
    expert_load_view: torch.Tensor,
    ep_rank: int,
    ep_size: int,
) -> None:
    """Accumulate this rank's local physical-expert load into global slots."""
    if expert_load_view.numel() % ep_size != 0:
        raise ValueError("Physical experts must be evenly distributed across EP ranks.")

    num_local_physical_experts = expert_load_view.numel() // ep_size
    if expert_tokens.numel() < num_local_physical_experts:
        raise ValueError("expert_tokens has fewer entries than the number of local physical experts.")

    local_load = expert_tokens[:num_local_physical_experts]
    if group_list_type != 1:
        local_load = torch.cat((local_load[:1], local_load[1:] - local_load[:-1]))

    local_load_view = expert_load_view.narrow(
        0,
        ep_rank * num_local_physical_experts,
        num_local_physical_experts,
    )
    local_load_view.add_(local_load)


def _map_to_physical_fake(
    topk_ids: torch.Tensor,
    expert_replica_routing_table: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(topk_ids)


direct_register_custom_op(
    op_name="ascend_eplb_map_to_physical",
    op_func=map_to_physical,
    fake_impl=_map_to_physical_fake,
    dispatch_key="PrivateUse1",
)
