# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bound wire-key construction shared by writers and scheduler lookups."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerwiseKeyBuilder:
    # The backend binds model/layout identity; callers supply only coordinates.
    make_base_key: Callable[[int, str, int, int], str]
    pp_size: int
    pcp_size: int = 1
    dcp_size: int = 1

    def make_full_key(
        self,
        group_id: int,
        block_hash: str,
        rank: int,
        stage: int,
        pcp_rank: int = 0,
        dcp_rank: int = 0,
    ) -> str:
        key = self.make_base_key(group_id, block_hash, rank, stage)
        if self.pcp_size > 1:
            # Each PCP worker owns its own PUT/GET sessions. Including the
            # world size also prevents reuse across incompatible topologies.
            key = f"{key}@pcp_size:{self.pcp_size}@pcp_rank:{pcp_rank}"
        if self.dcp_size > 1:
            # Each DCP rank stores a distinct context shard of the same
            # logical block, so it needs its own objects and sessions.
            key = f"{key}@dcp_size:{self.dcp_size}@dcp_rank:{dcp_rank}"
        return key

    def make_hit_check_keys(self, group_id: int, block_hash: str, num_ranks: int) -> list[str]:
        """Require every saving head on every PP stage, PCP rank and DCP rank."""
        return [
            self.make_full_key(group_id, block_hash, rank, stage, pcp_rank, dcp_rank)
            for stage in range(self.pp_size)
            for pcp_rank in range(self.pcp_size)
            for dcp_rank in range(self.dcp_size)
            for rank in range(num_ranks)
        ]
