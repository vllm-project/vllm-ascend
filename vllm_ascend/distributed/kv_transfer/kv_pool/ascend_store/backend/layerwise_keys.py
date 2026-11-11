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
    make_full_key: Callable[[int, str, int, int], str]
    pp_size: int

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
        return [
            self.make_full_key(group_id, block_hash, rank, stage)
            for stage in range(self.pp_size)
            for rank in range(rank_count)
        ]
