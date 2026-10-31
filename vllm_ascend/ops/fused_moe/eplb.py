# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.device.hardware_profile import MoECommPolicy, get_current_hardware_profile
from vllm_ascend.distributed.eplb.tensor_ops import (
    EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS,
)
from vllm_ascend.distributed.eplb.tensor_ops import (
    build_expert_replica_routing_table as build_expert_replica_routing_table,
)
from vllm_ascend.distributed.eplb.tensor_ops import (
    count_allgather_tokens as count_allgather_tokens,
)


def use_allgather_eplb(enabled: bool) -> bool:
    # The forward context imports the MoE implementations during setup.
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType

    return (
        enabled
        and get_current_hardware_profile().moe_comm_policy is MoECommPolicy.CAPACITY_AND_WORLD_SIZE
        and _EXTRA_CTX.moe_comm_type is MoECommType.ALLGATHER
    )


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


def map_to_physical_for_ascend(
    topk_ids: torch.Tensor,
    expert_replica_routing_table: torch.Tensor,
) -> torch.Tensor:
    """Map logical IDs through the graph-stable Ascend routing table."""
    if topk_ids.device.type != "cpu":
        from vllm_ascend.ops.triton.eplb import map_to_physical_triton

        return map_to_physical_triton(topk_ids, expert_replica_routing_table)
    return map_to_physical(topk_ids, expert_replica_routing_table)


def record_expert_tokens(
    expert_tokens: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_enabled: torch.Tensor,
    group_list_type: int,
    local_expert_start: int,
) -> None:
    """Accumulate local physical-expert counts returned by the MoE operator."""
    if expert_tokens.ndim != 1 or group_list_type not in (0, 1):
        raise ValueError("expert_tokens must be 1D and group_list_type must be 0 or 1")
    if local_expert_start < 0 or local_expert_start + expert_tokens.numel() > expert_load_view.numel():
        raise ValueError("Local expert counts are outside the physical expert load view")
    if expert_tokens.device.type != "cpu":
        from vllm_ascend.ops.triton.eplb import record_expert_tokens_triton

        record_expert_tokens_triton(
            expert_tokens, expert_load_view, record_enabled, group_list_type, local_expert_start
        )
        return

    local_load = (
        expert_tokens
        if group_list_type == 1
        else torch.cat([expert_tokens[:1], expert_tokens[1:] - expert_tokens[:-1]])
    )
    expert_load_view[local_expert_start : local_expert_start + expert_tokens.numel()].add_(
        local_load.to(expert_load_view.dtype) * record_enabled.to(expert_load_view.dtype)
    )


def _map_to_physical_fake(
    topk_ids: torch.Tensor,
    expert_replica_routing_table: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(topk_ids)


def _record_expert_tokens_fake(
    expert_tokens: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_enabled: torch.Tensor,
    group_list_type: int,
    local_expert_start: int,
) -> None:
    return None


direct_register_custom_op(
    op_name="ascend_eplb_map_to_physical",
    op_func=map_to_physical_for_ascend,
    mutates_args=[],
    fake_impl=_map_to_physical_fake,
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="ascend_eplb_record_expert_tokens",
    op_func=record_expert_tokens,
    mutates_args=["expert_load_view"],
    fake_impl=_record_expert_tokens_fake,
    dispatch_key="PrivateUse1",
)
