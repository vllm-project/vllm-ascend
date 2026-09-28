# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Node-local Engram DP (EDP) exchange, backported from vLLM f84b0c4bce.

One table is sharded over TP x EDP, where EDP is the node-local DP group that
``vllm.distributed.parallel_state`` builds: the global DP dimension may
span nodes, but a shard and the per-step exchange for it never do. A rank takes
its heads TP-major inside that group, gathers the n-gram ids of the node-local
replicas, and gathers their rows back.
"""

import torch
from vllm.distributed import get_dp_group, get_tensor_model_parallel_rank
from vllm.distributed.parallel_state import get_engram_dp_group, get_engram_dp_size
from vllm.forward_context import get_forward_context

# Upstream #56741 normalized the V4.1 model package name.
from vllm.models.deepseek_v41.common.engram import DEAD_ID, _engram_select_rows


def resolve_dp_shared_memory(requested: bool) -> bool:
    """A node with one DP replica uses ordinary TP shards, without sharing."""
    return requested and get_engram_dp_size() > 1


def engram_head_shard_rank() -> int:
    """This rank's slot among the hash-head shards of one engram table.

    TP-major, so the shards an EDP gather brings in are contiguous heads and
    the following TP gather completes the head order.
    """
    dp_group = get_engram_dp_group()
    dp_size = dp_group.world_size if dp_group is not None else 1
    dp_rank = dp_group.rank_in_group if dp_group is not None else 0
    return get_tensor_model_parallel_rank() * dp_size + dp_rank


def engram_gathered_num_tokens() -> int:
    """Per-replica token slot for the node-local Engram DP group."""
    dp_metadata = get_forward_context().dp_metadata
    if dp_metadata is None:
        raise RuntimeError("a DP-shared engram table needs DP token metadata")
    group = get_engram_dp_group()
    assert group is not None
    # Engram groups are contiguous slices of the full DP group.
    start = get_dp_group().rank_in_group - group.rank_in_group
    return int(dp_metadata.num_tokens_across_dp_cpu[start : start + group.world_size].max())


def gather_engram_hashes(hash_ids: torch.Tensor, *, dp_shared_memory: bool = False) -> torch.Tensor:
    """Collect the n-gram ids of every EDP replica sharing one table.

    Replicas are padded to a common token slot, so the gathered shape is
    static under CUDA graph capture (where DP already pads alike). Sharing
    replaces the lookup collectives and keeps every replica on its own ids.
    """
    dp_group = get_engram_dp_group()
    if dp_group is None or dp_shared_memory:
        return hash_ids
    slot = engram_gathered_num_tokens()
    if hash_ids.shape[0] > slot:
        raise ValueError("Engram token count exceeds the DP token slot")
    if hash_ids.shape[0] < slot:
        pad = hash_ids.new_full((slot - hash_ids.shape[0], *hash_ids.shape[1:]), DEAD_ID)
        hash_ids = torch.cat((hash_ids, pad))
    return dp_group.all_gather(hash_ids, dim=0)


def _gather_engram_rows(staged: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Exchange EDP tokens for heads, retaining only this replica's tokens."""
    dp_group = get_engram_dp_group()
    assert dp_group is not None
    slot, remainder = divmod(staged.shape[0], dp_group.world_size)
    assert remainder == 0 and 0 <= num_tokens <= slot
    gathered = dp_group.all_gather(staged, dim=0)
    local_heads, dim = staged.shape[1:]
    rows = staged.new_empty((num_tokens, dp_group.world_size * local_heads, dim))
    _engram_select_rows(
        gathered,
        rows,
        staged.shape[0],
        dp_group.rank_in_group * slot,
        local_heads * dim,
    )
    return rows
