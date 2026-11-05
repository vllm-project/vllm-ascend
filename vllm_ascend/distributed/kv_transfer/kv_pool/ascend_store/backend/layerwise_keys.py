# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bound wire-key construction shared by writers and scheduler lookups."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_ascend.core.kv_cache_interface import MambaStateSharding


@dataclass(frozen=True)
class LayerwiseKeyBuilder:
    # The backend binds model/layout identity; callers supply only coordinates.
    make_base_key: Callable[[int, str, int, int], str]
    pp_size: int
    pcp_size: int = 1

    def make_full_key(
        self,
        group_id: int,
        block_hash: str,
        rank: int,
        stage: int,
        pcp_rank: int = 0,
        *,
        sharding: MambaStateSharding | None = None,
    ) -> str:
        """Build a key for an already-resolved head or state partition rank."""
        key = self.make_base_key(group_id, block_hash, rank, stage)
        if sharding is None and self.pcp_size > 1:
            # Each PCP worker owns its own PUT/GET sessions. Including the
            # world size also prevents reuse across incompatible topologies.
            return f"{key}@pcp_size:{self.pcp_size}@pcp_rank:{pcp_rank}"
        return key

    def make_hit_check_keys(
        self,
        group_id: int,
        block_hash: str,
        num_ranks: int,
        *,
        sharding: MambaStateSharding | None = None,
    ) -> list[str]:
        """Enumerate required keys, using num_ranks only without a declaration."""
        rank_count = sharding.shard_count if sharding is not None else num_ranks
        pcp_count = 1 if sharding is not None else self.pcp_size
        return [
            self.make_full_key(group_id, block_hash, rank, stage, pcp_rank, sharding=sharding)
            for stage in range(self.pp_size)
            for pcp_rank in range(pcp_count)
            for rank in range(rank_count)
        ]
