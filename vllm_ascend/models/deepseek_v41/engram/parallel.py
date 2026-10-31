# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Node-local Engram exchange with A5 recompute-decode fixed token slots.

Reuse mainline EDP ownership and row exchange. The A5 scheduler can skip DP
metadata synchronization, so hash gathering must retain its fixed-slot path.
"""

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_engram_dp_group, get_engram_dp_size
from vllm.forward_context import get_forward_context

# Upstream #56741 normalized the V4.1 model package name.
from vllm.models.deepseek_v41.common.engram import DEAD_ID
from vllm.models.deepseek_v41.nvidia.engram import (
    engram_head_shard_rank as engram_head_shard_rank,
)

from vllm_ascend.utils import get_potential_max_tokens, is_pd_decode_recompute_scheduler_enabled


def resolve_dp_shared_memory(requested: bool) -> bool:
    """Share a TP shard when local DP or PCP peers map the same host table."""
    return requested and get_engram_dp_size() > 1


def engram_gathered_num_tokens() -> int:
    """Per-rank token slot for the node-local Engram DP x PCP group."""
    context = get_forward_context()
    if (
        not getattr(context, "in_profile_run", False)
        and not getattr(context, "engram_uniform_dp_warmup", False)
        and is_pd_decode_recompute_scheduler_enabled()
    ):
        # Recompute decode can skip the DP metadata all-reduce. Its token
        # vector then contains only local counts, including on idle ranks or
        # ranks using different graph buckets. Size both Engram exchanges
        # from the shared configuration, never from that local vector.
        config = get_current_vllm_config()
        scheduler = config.scheduler_config
        query_len = 1 + config.speculative_config.num_speculative_tokens if config.speculative_config else 1
        return max(
            get_potential_max_tokens(),
            min(scheduler.max_num_batched_tokens, scheduler.max_num_seqs * query_len),
        )
    dp_metadata = context.dp_metadata
    if dp_metadata is None:
        raise RuntimeError("a DP-shared engram table needs DP token metadata")
    group = get_engram_dp_group()
    assert group is not None
    # Engram groups are contiguous slices of the full DP group.
    start = get_dp_group().rank_in_group - group.rank_in_group
    return int(dp_metadata.num_tokens_across_dp_cpu[start : start + group.world_size].max())


def gather_engram_hashes(hash_ids: torch.Tensor, *, dp_shared_memory: bool = False) -> torch.Tensor:
    """Collect the n-gram ids of every DP x PCP rank using one split table.

    Replicas are padded to a common token slot, including when recompute
    decode skips DP metadata synchronization and local graph sizes differ.
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


def exchange_engram_rows(staged: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Send each padded DP x PCP token block directly to its owning rank.

    Input is [EDP * slot, local_heads, dim], in destination-rank order.
    Each destination receives only its token block from every head owner,
    instead of materializing every destination's block as all-gather does.
    All ranks keep the same slot, including idle ranks with zero valid tokens.
    """
    group = get_engram_dp_group()
    assert group is not None
    dp_size = group.world_size
    slot, remainder = divmod(staged.shape[0], dp_size)
    assert remainder == 0 and 0 <= num_tokens <= slot
    if dp_size == 1:
        return staged[:num_tokens]
    local_heads, dim = staged.shape[1:]
    recv = torch.empty_like(staged)
    dist.all_to_all_single(recv, staged.contiguous(), group=group.device_group)
    # Received chunks are source-rank major; heads are contiguous across EDP.
    rows = recv.view(dp_size, slot, local_heads, dim).permute(1, 0, 2, 3)
    return rows[:num_tokens].reshape(num_tokens, dp_size * local_heads, dim)
