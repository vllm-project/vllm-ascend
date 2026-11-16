# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bound wire-key construction shared by writers and scheduler lookups."""

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LayerwiseKeyBuilder:
    # The backend binds model/layout identity; callers supply only coordinates.
    make_base_key: Callable[[int, str, int, int], str]
    pp_size: int
    pcp_size: int = 1
    pcp_sharded_group_ids: set[int] = field(default_factory=set)
    dcp_size: int = 1

    def make_full_key(
        self,
        group_id: int,
        block_hash: str,
        rank: int,
        stage: int,
        pcp_rank: int = 0,
        dcp_rank: int | None = None,
    ) -> str:
        key = self.make_base_key(group_id, block_hash, rank, stage)
        if self.pcp_size > 1 and group_id in self.pcp_sharded_group_ids:
            # State head partitions depend on the PCP topology.
            key = f"{key}@pcp_size:{self.pcp_size}@pcp_rank:{pcp_rank}"
        if self.dcp_size > 1:
            if dcp_rank is None:
                raise ValueError("dcp_rank is required when dcp_size > 1")
            key = f"{key}@dcp_size:{self.dcp_size}@dcp_rank:{dcp_rank}"
        return key

    def make_hit_check_keys(self, group_id: int, block_hash: str, num_ranks: int) -> list[str]:
        """Require every distinct shard on every PP stage."""
        pcp_size = self.pcp_size if group_id in self.pcp_sharded_group_ids else 1
        return [
            self.make_full_key(group_id, block_hash, rank, stage, pcp_rank, dcp_rank)
            for stage in range(self.pp_size)
            for pcp_rank in range(pcp_size)
            for dcp_rank in range(self.dcp_size)
            for rank in range(num_ranks)
        ]
