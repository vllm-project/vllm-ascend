#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""Backend-agnostic derivation and validation of the TP x PCP x DCP grid.

This module is pure and standard-library-only: it never calls the distributed
runtime, creates process groups, or reads environment variables. It describes
the TP x PCP rank grid inside a single DP replica and PP stage; the DCP group
reuses ranks from the TP axis, the PCP axis, or the whole grid, as documented
in ``docs/source/developer_guide/Design_Documents/context_parallel.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class DCPGroupAxis(Enum):
    """Rank axis that hosts the groups of the decode context-parallel size."""

    DISABLED = "disabled"  # dcp_size == 1
    TP = "tp"  # pcp_size == 1: the DCP group lives on the TP axis
    PCP = "pcp"  # dcp_size == pcp_size > 1: the DCP group lives on the PCP axis
    TP_PCP = "tp_pcp"  # dcp_size == tp_size * pcp_size: the DCP group spans the whole grid


@dataclass(frozen=True)
class CPTopology:
    """Immutable description of the TP x PCP x DCP grid for one DP replica."""

    tp_size: int
    pcp_size: int
    dcp_size: int
    interleave_size: int = 1
    pipeline_parallel_size: int = 1

    @property
    def is_pcp_enabled(self) -> bool:
        """Whether prefill context parallelism is enabled."""
        return self.pcp_size > 1

    @property
    def is_dcp_enabled(self) -> bool:
        """Whether decode context parallelism is enabled."""
        return self.dcp_size > 1

    @property
    def dcp_axis(self) -> DCPGroupAxis:
        """Axis that the DCP group spans.

        Upstream ``ParallelConfig`` already rejects unsupported DCP sizes, so
        this ValueError is a defensive derived-API check and must not be wired
        into platform validation.
        """
        if self.dcp_size <= 1:
            return DCPGroupAxis.DISABLED
        if self.pcp_size <= 1:
            return DCPGroupAxis.TP
        if self.dcp_size == self.pcp_size:
            return DCPGroupAxis.PCP
        if self.dcp_size == self.tp_size * self.pcp_size:
            return DCPGroupAxis.TP_PCP
        raise ValueError(
            f"Unsupported DCP size for the TP x PCP grid: TP={self.tp_size}, PCP={self.pcp_size}, DCP={self.dcp_size}."
        )

    @property
    def dcp_reuses_tp_ranks(self) -> bool:
        """Whether the DCP group is made of ranks that TP groups already use."""
        return self.dcp_axis in (DCPGroupAxis.TP, DCPGroupAxis.TP_PCP)

    def pcp_face_ranks(self) -> tuple[int, ...]:
        """Return one rank per PCP coordinate (the TP coordinate is fixed at 0)."""
        return tuple(range(0, self.tp_size * self.pcp_size, self.tp_size))

    def dcp_logical_rank_order(self) -> tuple[int, ...]:
        """Return the logical DCP rank order: the PCP coordinate varies before TP."""
        return tuple(
            pcp_coord * self.tp_size + tp_coord
            for tp_coord in range(self.tp_size)
            for pcp_coord in range(self.pcp_size)
        )

    def to_parallel_config_fields(self) -> dict[str, int]:
        """Return the ``ParallelConfig`` fields that define this topology."""
        return {
            "tensor_parallel_size": self.tp_size,
            "prefill_context_parallel_size": self.pcp_size,
            "decode_context_parallel_size": self.dcp_size,
            "cp_kv_cache_interleave_size": self.interleave_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
        }


def resolve_cp_topology(vllm_config: VllmConfig) -> CPTopology:
    """Read the CP topology from ``vllm_config`` without touching the runtime.

    ``tensor_parallel_size``, ``cp_kv_cache_interleave_size`` and
    ``pipeline_parallel_size`` fall back to 1 so partial configs used by tests
    and draft helpers resolve; the PCP and DCP sizes are required fields.
    """
    parallel_config = vllm_config.parallel_config
    return CPTopology(
        tp_size=getattr(parallel_config, "tensor_parallel_size", 1),
        pcp_size=parallel_config.prefill_context_parallel_size,
        dcp_size=parallel_config.decode_context_parallel_size,
        interleave_size=getattr(parallel_config, "cp_kv_cache_interleave_size", 1),
        pipeline_parallel_size=getattr(parallel_config, "pipeline_parallel_size", 1),
    )


def validate_common_topology(
    topology: CPTopology,
    model_config=None,
    cache_config=None,
    *,
    vllm_config: VllmConfig | None = None,
) -> None:
    """Validate backend-agnostic CP constraints, preserving the historical messages."""
    if vllm_config is not None and topology.pcp_size > 1 and not getattr(vllm_config, "use_v2_model_runner", False):
        raise ValueError(
            "PCP (Prefill Context Parallelism) is not supported by vLLM Ascend. "
            "Please set --prefill-context-parallel-size to 1. "
            f"Got prefill_context_parallel_size={topology.pcp_size}."
        )

    if (
        model_config is not None
        and not model_config.use_mla
        and topology.pcp_size > 1
        and topology.dcp_size > 1
        and topology.pcp_size != topology.dcp_size
    ):
        raise ValueError(
            "GQA/MQA PCP+DCP requires prefill_context_parallel_size "
            "to equal decode_context_parallel_size; "
            f"got PCP={topology.pcp_size}, DCP={topology.dcp_size}."
        )

    if cache_config is not None and topology.dcp_size > 1:
        block_size = getattr(cache_config, "block_size", None)
        if isinstance(block_size, int):
            interleave_size = topology.interleave_size
            if interleave_size <= 0 or block_size % interleave_size != 0:
                raise ValueError(
                    f"cp_kv_cache_interleave_size({interleave_size}) must divide the cache block_size({block_size}) "
                    "when decode context parallel is enabled."
                )
