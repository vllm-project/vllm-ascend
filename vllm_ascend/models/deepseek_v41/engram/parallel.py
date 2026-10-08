# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Node-local Engram storage sharing and MRV2 pre-forward hash exchange."""

import torch
import torch.distributed as dist
from vllm.distributed import get_engram_dp_group, get_engram_dp_size
from vllm.distributed.parallel_state import get_world_group, in_the_same_node_as
from vllm.models.deepseek_v41.common.engram import DEAD_ID
from vllm.models.deepseek_v41.nvidia.engram import gather_engram_hashes as upstream_gather_engram_hashes


def resolve_dp_shared_memory(requested: bool, pcp_size: int = 1) -> bool:
    """Share across node-local EDP or PCP peers; retain ordinary singleton TP."""
    return requested and (get_engram_dp_size() > 1 or pcp_size > 1)


class EngramStorageGroup:
    """Model-owned CPU communicator for read-only Engram table storage."""

    def __init__(self, cpu_group, world_size: int, rank_in_group: int, *, owns_cpu_group: bool = False):
        self.cpu_group = cpu_group
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self._owns_cpu_group = owns_cpu_group
        self._closed = False

    def close(self) -> None:
        """Destroy a dedicated communicator once; leave borrowed DP groups intact."""
        if self._closed:
            return
        if self._owns_cpu_group:
            dist.destroy_process_group(self.cpu_group)
        self._closed = True


def create_engram_storage_group(parallel_config) -> EngramStorageGroup | None:
    """Share each TP shard across same-node DP and PCP ranks.

    All world ranks must call this once in the same model-construction order.
    The rank layout is ExternalDP x DP x PP x PCP x TP, matching vLLM's
    parallel state. External DP replicas and TP shards remain separate.
    PCP=1 borrows the existing DP CPU group; an unshared singleton returns
    None. Query collectives continue to use ``get_engram_dp_group``.
    """
    if parallel_config.pipeline_parallel_size != 1 or parallel_config.decode_context_parallel_size != 1:
        raise ValueError("Engram storage sharing requires PP=DCP=1")
    if getattr(parallel_config, "nnodes", 1) != 1:
        raise ValueError("Engram storage sharing requires a single-node deployment")
    if getattr(parallel_config, "enable_elastic_ep", False):
        raise ValueError("Engram storage sharing does not support elastic EP")

    dp_size = parallel_config.data_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    tp_size = parallel_config.tensor_parallel_size
    if min(dp_size, pcp_size, tp_size) < 1:
        raise ValueError("Engram storage sharing requires positive DP, PCP and TP sizes")
    if pcp_size == 1:
        dp_group = get_engram_dp_group()
        if dp_group is None or dp_group.world_size <= 1:
            return None
        return EngramStorageGroup(dp_group.cpu_group, dp_group.world_size, dp_group.rank_in_group)

    world = get_world_group()
    replica_size = dp_size * pcp_size * tp_size
    external_dp_size, remainder = divmod(world.world_size, replica_size)
    if remainder:
        raise ValueError("Engram storage sharing world size must be divisible by DP * PCP * TP")

    storage_group = None
    try:
        # new_group is a world collective, including for non-members. Keep its
        # order deterministic and finish every group before validating members.
        for external_rank in range(external_dp_size):
            rank_offset = external_rank * replica_size
            for tp_rank in range(tp_size):
                ranks = [
                    rank_offset + dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
                    for dp_rank in range(dp_size)
                    for pcp_rank in range(pcp_size)
                ]
                cpu_group = dist.new_group(ranks, backend="gloo")
                if world.rank in ranks:
                    storage_group = EngramStorageGroup(
                        cpu_group, len(ranks), ranks.index(world.rank), owns_cpu_group=True
                    )
        assert storage_group is not None

        # Shared-memory reachability is stronger than hostnames or nnodes=1.
        # Fence failures across the world before any TP shard allocates a table.
        same_node_error = None
        try:
            same_node = in_the_same_node_as(storage_group.cpu_group)
            members_share_memory = len(same_node) == storage_group.world_size and all(same_node)
        except Exception as error:
            members_share_memory = False
            same_node_error = error
        valid = torch.tensor([int(members_share_memory)], dtype=torch.int32, device="cpu")
        dist.all_reduce(valid, op=dist.ReduceOp.MIN, group=world.cpu_group)
        if not int(valid[0]):
            raise ValueError(
                "Engram storage sharing requires all DP * PCP members to share node IPC"
            ) from same_node_error
        return storage_group
    except BaseException:
        if storage_group is not None:
            storage_group.close()
        raise


def negotiate_engram_token_slot(num_tokens: int, dp_group) -> int:
    """Agree on an eager token slot before MRV2 installs ForwardContext.

    Counts describe the actual PCP-local rows, not the global prompt or
    graph padding. All replicas, including idle ones, join this CPU MAX.
    Keep one DEAD_ID transport row when the entire DP group is empty.
    """
    count = torch.tensor([num_tokens], dtype=torch.int64, device="cpu")
    dist.all_reduce(count, op=dist.ReduceOp.MAX, group=dp_group.cpu_group)
    return max(1, int(count[0]))


def gather_engram_hashes(
    hash_ids: torch.Tensor, *, dp_shared_memory: bool = False, pre_forward: bool = False
) -> torch.Tensor:
    """Collect the n-gram ids of every DP replica sharing one table.

    Replicas are padded to a common token slot, so the gathered shape is
    static under CUDA graph capture (where DP already pads alike).
    """
    if not pre_forward:
        return upstream_gather_engram_hashes(hash_ids, dp_shared_memory=dp_shared_memory)
    dp_group = get_engram_dp_group()
    if dp_group is None or dp_shared_memory:
        return hash_ids
    slot = negotiate_engram_token_slot(hash_ids.shape[0], dp_group)
    if hash_ids.shape[0] > slot:
        raise ValueError("Engram token count exceeds the DP token slot")
    if hash_ids.shape[0] < slot:
        pad = hash_ids.new_full((slot - hash_ids.shape[0], *hash_ids.shape[1:]), DEAD_ID)
        hash_ids = torch.cat((hash_ids, pad))
    return dp_group.all_gather(hash_ids, dim=0)
