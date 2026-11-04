# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM KDA head ownership shared by computation and cache transfers."""

from vllm.config import ParallelConfig

from vllm_ascend.core.kv_cache_interface import MambaStateSharding


def get_kda_parallel_size(parallel_config: ParallelConfig) -> int:
    """Use PCP head sharding only when tensor parallelism is disabled."""
    # TODO: Support joint TP * PCP head partitioning for projections and state.
    if parallel_config.tensor_parallel_size == 1:
        return parallel_config.prefill_context_parallel_size
    return parallel_config.tensor_parallel_size


def get_kda_state_sharding(parallel_config: ParallelConfig, tp_rank: int, pcp_rank: int) -> MambaStateSharding:
    shard_count = get_kda_parallel_size(parallel_config)
    if parallel_config.tensor_parallel_size == 1:
        return MambaStateSharding(shard_rank=pcp_rank, shard_count=shard_count)
    return MambaStateSharding(
        shard_rank=tp_rank,
        shard_count=shard_count,
        replica_rank=pcp_rank,
        replica_count=parallel_config.prefill_context_parallel_size,
    )
