# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bound wire-key construction shared by writers and scheduler lookups."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerwiseKeyBuilder:
    # The backend binds model/layout identity; callers supply only coordinates.
    make_full_key: Callable[[int, str, int, int], str]
    pp_size: int

    def make_hit_check_keys(self, group_id: int, block_hash: str, num_ranks: int) -> list[str]:
        """Require every saving head on every stage, in stage-major order."""
        return [
            self.make_full_key(group_id, block_hash, rank, stage)
            for stage in range(self.pp_size)
            for rank in range(num_ranks)
        ]
