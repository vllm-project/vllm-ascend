# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic.dataclasses import rebuild_dataclass
from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config import parallel as parallel_config
from vllm.config.parallel import ParallelConfig, logger

from vllm_ascend.utils import vllm_version_is


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


# v0.29.0 (98dff2a81d747d1dba01a47f939f48c3526d4206) validator,
# with only the platform-independent PCP+DP rejection removed for Ascend.
# Upstream #54523 (7c2f1ff4958eaf0818405e9192c71608fe4a16b1)
# moved this restriction to CUDA/ROCm. Remove this patch with v0.29 support.
def _validate_parallel_config(self: ParallelConfig) -> ParallelConfig:
    if self._api_process_rank >= self._api_process_count:
        raise ValueError(
            "Invalid value of `_api_process_rank`. "
            f"Expected to be `-1` or `[0, {self._api_process_count})`, "
            f"but found: {self._api_process_rank}"
        )

    if self.enable_fault_tolerance and self._api_process_count > 1:
        raise ValueError(
            "Fault tolerance requires a single API server process "
            f"(--api-server-count=1), but got {self._api_process_count}. "
            "The FT system assumes one AsyncMPClient manages all engines."
        )

    if self.all2all_backend in ["pplx", "naive"]:
        logger.warning(
            "The '%s' all2all backend has been removed. Falling back to 'allgather_reducescatter'.",
            self.all2all_backend,
        )
        self.all2all_backend = "allgather_reducescatter"

    if self.data_parallel_size_local > self.data_parallel_size:
        raise ValueError(
            f"data_parallel_size_local ({self.data_parallel_size_local}) "
            f"must be <= data_parallel_size ({self.data_parallel_size})"
        )

    if self.data_parallel_size <= 1 and self.data_parallel_external_lb:
        raise ValueError("data_parallel_external_lb can only be set when data_parallel_size > 1")

    if not self.numa_bind and (self.numa_bind_nodes is not None or self.numa_bind_cpus is not None):
        raise ValueError("numa_bind_nodes and numa_bind_cpus require numa_bind=True.")

    if self.enable_eplb:
        if not parallel_config.current_platform.is_cuda_alike():
            raise ValueError("Expert parallelism load balancing is only supported on CUDA devices or ROCm devices now.")
        if not self.enable_expert_parallel:
            raise ValueError("enable_expert_parallel must be True to use EPLB.")
        # The EP group spans the TP x PCP x DP ranks. EPLB therefore needs
        # TP, PCP, or DP > 1.
        if self.tensor_parallel_size * self.prefill_context_parallel_size * self.data_parallel_size <= 1:
            raise ValueError(
                "EPLB requires tensor, prefill-context, or data parallelism, "
                f"but got TP={self.tensor_parallel_size}, "
                f"PCP={self.prefill_context_parallel_size}, "
                f"DP={self.data_parallel_size}."
            )
    else:
        if self.eplb_config.num_redundant_experts != 0:
            raise ValueError(
                "num_redundant_experts is set to "
                f"{self.eplb_config.num_redundant_experts} but EPLB is not "
                "enabled. Either enable EPLB or unset "
                "num_redundant_experts."
            )

    tp = self.tensor_parallel_size
    pcp = self.prefill_context_parallel_size
    dcp = self.decode_context_parallel_size
    if pcp == 1:
        # DCP reuses the TP ranks when PCP is disabled.
        if tp % dcp != 0:
            raise ValueError(f"tp_size={tp} must be divisible by dcp_size={dcp}.")
    elif dcp not in (1, pcp, tp * pcp):
        raise ValueError(
            "When PCP is enabled, DCP must be disabled, span the PCP "
            "axis, or span the full TP x PCP axis. "
            f"Got TP={tp}, PCP={pcp}, DCP={dcp}; valid DCP sizes are "
            f"{sorted({1, pcp, tp * pcp})}."
        )

    return self


if vllm_version_is("0.29.0"):
    ParallelConfig._validate_parallel_config = _validate_parallel_config
    ParallelConfig.__pydantic_decorators__.model_validators[
        "_validate_parallel_config"
    ].func = _validate_parallel_config
    rebuild_dataclass(ParallelConfig, force=True)
    # SpeculativeConfig retains two ParallelConfig schema references even under
    # SkipValidation. Rebuild it before VllmConfig to avoid reusing the old
    # shared schema for VllmConfig.parallel_config.
    rebuild_dataclass(SpeculativeConfig, force=True)
    rebuild_dataclass(VllmConfig, force=True)
