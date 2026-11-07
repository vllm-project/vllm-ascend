# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bound wire-key construction shared by writers and scheduler lookups."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vllm_ascend.ascend_config import KVPPConfig


@dataclass(frozen=True)
class LayerwiseTopology:
    """Resolved key-space topology for layerwise pooling.

    KVPP partitions the persistent target caches by layer across the TP x PCP
    replica domain, so an owner coordinate (``kvpp_rank = pcp * tp + tp``)
    already encodes the PCP axis. Folding PCP into that coordinate keeps one
    writer coordinate per object, while the ``kvpp-tp{T}-pcp{P}`` namespace
    isolates incompatible topologies. Non-KVPP modes keep the historical PCP
    suffix.
    """

    use_kvpp: bool
    kvpp_size: int
    key_pcp_size: int


def _positive_int(value: Any, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else default


def resolve_layerwise_topology(vllm_config: Any) -> LayerwiseTopology:
    """Return the single source of truth for layerwise key topology.

    The scheduler and every worker must derive identical values so that the
    hit-check key set and the writer key set stay byte-identical.
    """
    parallel = vllm_config.parallel_config
    try:
        use_kvpp = KVPPConfig.from_vllm_config(vllm_config).size > 1
    except Exception:
        # Unit-test mocks and partially populated configs may not expose the
        # full configuration API used by KVPPConfig.
        use_kvpp = bool((getattr(vllm_config, "additional_config", None) or {}).get("enable_kvpp", False))
    tp_size = _positive_int(getattr(parallel, "tensor_parallel_size", 1), 1)
    pcp_size = _positive_int(getattr(parallel, "prefill_context_parallel_size", 1), 1)
    if not use_kvpp:
        return LayerwiseTopology(use_kvpp=False, kvpp_size=1, key_pcp_size=pcp_size)
    return LayerwiseTopology(use_kvpp=True, kvpp_size=tp_size * pcp_size, key_pcp_size=1)


@dataclass(frozen=True)
class LayerwiseKeyBuilder:
    # The backend binds model/layout identity; callers supply only coordinates.
    make_base_key: Callable[[int, str, int, int], str]
    pp_size: int
    # Key-space PCP dimension. KVPP folds PCP into the writer coordinate, so
    # ``pcp_size`` is 1 there even when the engine runs with PCP > 1.
    pcp_size: int = 1
    # True when the bound key space uses the KVPP writer coordinate.
    kvpp: bool = False

    def __post_init__(self) -> None:
        if self.kvpp and self.pcp_size != 1:
            raise ValueError(
                f"KVPP layerwise keys fold PCP into kvpp_rank; expected pcp_size=1, got pcp_size={self.pcp_size}."
            )

    def make_full_key(self, group_id: int, block_hash: str, rank: int, stage: int, pcp_rank: int = 0) -> str:
        key = self.make_base_key(group_id, block_hash, rank, stage)
        if self.pcp_size > 1:
            # Each PCP worker owns its own PUT/GET sessions. Including the
            # world size also prevents reuse across incompatible topologies.
            return f"{key}@pcp_size:{self.pcp_size}@pcp_rank:{pcp_rank}"
        return key

    def make_hit_check_keys(self, group_id: int, block_hash: str, num_ranks: int) -> list[str]:
        """Require every saving head on every PP stage and PCP rank.

        In KVPP mode ``num_ranks`` is the owner count (``tp * pcp``) and the PCP
        dimension is already folded into the writer coordinate, so the loop
        collapses to one PCP value.
        """
        return [
            self.make_full_key(group_id, block_hash, rank, stage, pcp_rank)
            for stage in range(self.pp_size)
            for pcp_rank in range(self.pcp_size)
            for rank in range(num_ranks)
        ]
