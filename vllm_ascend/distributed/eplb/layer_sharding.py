# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Static layer sharding for host-side EPLB planning."""

import torch


def assigned_layer_ids(num_layers: int, group_rank: int, group_size: int) -> range:
    """Return stage-local layers owned by one rank in round-robin order."""
    if any(type(value) is not int for value in (num_layers, group_rank, group_size)):
        raise ValueError("layer sharding controls must be integers")
    if num_layers < 0 or group_size < 1 or not 0 <= group_rank < group_size:
        raise ValueError("num_layers must be non-negative and group_rank must be within group_size")
    return range(group_rank, num_layers, group_size)


def all_gather_layer_shards(
    local_values: torch.Tensor,
    num_layers: int,
    cpu_group,
) -> torch.Tensor:
    """Gather statically owned layer values into stage-local layer order.

    ``local_values`` is ``[owned_layers, ...]`` in the order returned by
    :func:`assigned_layer_ids`. Every rank must pass the same ``num_layers``,
    trailing shape, and dtype. The supplied group must contain only the current
    PP stage's EPLB ranks.
    """
    if local_values.device.type != "cpu" or local_values.ndim == 0:
        raise ValueError("EPLB layer shards must be non-scalar CPU tensors")
    group_size = cpu_group.size()
    group_rank = cpu_group.rank()
    owned_layers = assigned_layer_ids(num_layers, group_rank, group_size)
    if local_values.shape[0] != len(owned_layers):
        raise ValueError("EPLB layer shard size does not match its static assignment")
    if num_layers == 0 or group_size == 1:
        return local_values.clone()

    shard_capacity = (num_layers + group_size - 1) // group_size
    padded = local_values.new_zeros((shard_capacity, *local_values.shape[1:]))
    padded[: len(owned_layers)].copy_(local_values)
    gathered = [torch.empty_like(padded) for _ in range(group_size)]
    torch.distributed.all_gather(gathered, padded, group=cpu_group)

    merged = local_values.new_empty((num_layers, *local_values.shape[1:]))
    for owner_rank, shard in enumerate(gathered):
        owner_layers = range(owner_rank, num_layers, group_size)
        merged[owner_rank::group_size].copy_(shard[: len(owner_layers)])
    return merged
