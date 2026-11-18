# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.parallel import ParallelConfig


def _use_sequence_parallel_moe(self: ParallelConfig) -> bool:
    """Enable MoE sequence parallelism for TP/EP topologies, including DP=1."""
    return (
        self.all2all_backend
        in (
            "allgather_reducescatter",
            "deepep_high_throughput",
            "deepep_low_latency",
            "deepep_v2",
            "flashinfer_nvlink_one_sided",
            "mori_high_throughput",
            "mori_low_latency",
            "nixl_ep",
        )
        and self.enable_expert_parallel
        and self.tensor_parallel_size > 1
    )


# Upstream additionally requires data_parallel_size > 1. On Ascend, FlashComm
# supports the TP/EP, DP=1 topology and still needs SP's rank-local token layout.
ParallelConfig.use_sequence_parallel_moe = property(_use_sequence_parallel_moe)
