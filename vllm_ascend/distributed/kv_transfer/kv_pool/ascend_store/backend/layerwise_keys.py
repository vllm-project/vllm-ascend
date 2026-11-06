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

    def make_full_key(
        self,
        group_id: int,
        block_hash: str,
        rank: int,
        stage: int,
        pcp_rank: int = 0,
        use_joint_tp_pcp_rank: bool = False,
    ) -> str:
        """Build a partition key; joint TP/PCP ranks need no PCP suffix."""
        key = self.make_base_key(group_id, block_hash, rank, stage)
        if not use_joint_tp_pcp_rank and self.pcp_size > 1:
            # Each PCP worker owns its own PUT/GET sessions. Including the
            # world size also prevents reuse across incompatible topologies.
            return f"{key}@pcp_size:{self.pcp_size}@pcp_rank:{pcp_rank}"
        return key

    def make_hit_check_keys(
        self,
        group_id: int,
        block_hash: str,
        num_ranks: int,
        use_joint_tp_pcp_rank: bool = False,
    ) -> list[str]:
        """Require every saving head on every PP stage and PCP rank.

        When use_joint_tp_pcp_rank is True, rank already includes PCP,
        so pcp_rank is not expanded separately.
        """
        pcp_count = 1 if use_joint_tp_pcp_rank else self.pcp_size
        return [
            self.make_full_key(group_id, block_hash, rank, stage, pcp_rank, use_joint_tp_pcp_rank)
            for stage in range(self.pp_size)
            for pcp_rank in range(pcp_count)
            for rank in range(num_ranks)
        ]
