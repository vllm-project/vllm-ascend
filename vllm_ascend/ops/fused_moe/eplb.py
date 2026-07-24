# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.utils.torch_utils import direct_register_custom_op

_KNUTH_MULTIPLIER = 2654435769


def map_and_record(
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_enabled: torch.Tensor,
    num_unpadded_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map logical expert IDs to physical slots and accumulate expert load."""
    if topk_ids.numel() == 0:
        return topk_ids

    logical_ids = topk_ids.to(torch.int64)
    valid_logical = (logical_ids >= 0) & (logical_ids < logical_replica_count.shape[0])
    safe_logical_ids = torch.where(valid_logical, logical_ids, 0)
    replica_count = logical_replica_count[safe_logical_ids].to(torch.int64).clamp_min(1)

    token_indices = torch.arange(topk_ids.shape[0], dtype=torch.int64, device=topk_ids.device)
    hashed_token_indices = torch.bitwise_and(token_indices * _KNUTH_MULTIPLIER, 0xFFFFFFFF)
    replica_indices = hashed_token_indices[:, None] % replica_count
    physical_ids = logical_to_physical_map[safe_logical_ids, replica_indices].to(topk_ids.dtype)
    physical_ids = torch.where(valid_logical, physical_ids, -1)

    valid_tokens = torch.ones(topk_ids.shape[0], dtype=torch.bool, device=topk_ids.device)
    if num_unpadded_tokens is not None:
        valid_tokens &= token_indices < num_unpadded_tokens

    valid_physical = (physical_ids >= 0) & (physical_ids < expert_load_view.shape[0])
    should_record = valid_tokens[:, None] & valid_physical & record_enabled
    safe_physical_ids = torch.where(valid_physical, physical_ids, 0).to(torch.int64)
    increments = should_record.to(expert_load_view.dtype)
    expert_load_view.scatter_add_(0, safe_physical_ids.flatten(), increments.flatten())
    return physical_ids


def _map_and_record_fake(
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_enabled: torch.Tensor,
    num_unpadded_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(topk_ids)


direct_register_custom_op(
    op_name="ascend_eplb_map_and_record",
    op_func=map_and_record,
    fake_impl=_map_and_record_fake,
    mutates_args=["expert_load_view"],
    dispatch_key="PrivateUse1",
)
