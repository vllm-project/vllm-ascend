# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bound wire-key construction shared by writers and scheduler lookups."""

from collections.abc import Callable
from dataclasses import dataclass, field

_BLOCK_HASH_PLACEHOLDER = "__vllm_ascend_block_hash__"


@dataclass(frozen=True)
class LayerwiseKeyBuilder:
    # The backend binds model/layout identity; callers supply only coordinates.
    make_base_key: Callable[[int, str, int, int], str]
    pp_size: int
    pcp_size: int = 1
    pcp_sharded_group_ids: set[int] = field(default_factory=set)
    _hit_check_key_templates: dict[tuple[int, int], tuple[str, ...] | None] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def make_full_key(self, group_id: int, block_hash: str, rank: int, stage: int, pcp_rank: int = 0) -> str:
        key = self.make_base_key(group_id, block_hash, rank, stage)
        if self.pcp_size > 1 and group_id in self.pcp_sharded_group_ids:
            # State head partitions depend on the PCP topology.
            return f"{key}@pcp_size:{self.pcp_size}@pcp_rank:{pcp_rank}"
        return key

    def make_hit_check_keys(self, group_id: int, block_hash: str, num_ranks: int) -> list[str]:
        """Require every distinct shard on every PP stage."""
        pcp_size = self.pcp_size if group_id in self.pcp_sharded_group_ids else 1
        cache_key = (group_id, num_ranks)
        if cache_key not in self._hit_check_key_templates:
            templates = tuple(
                self.make_full_key(group_id, _BLOCK_HASH_PLACEHOLDER, rank, stage, pcp_rank)
                for stage in range(self.pp_size)
                for pcp_rank in range(pcp_size)
                for rank in range(num_ranks)
            )
            self._hit_check_key_templates[cache_key] = (
                templates if all(template.count(_BLOCK_HASH_PLACEHOLDER) == 1 for template in templates) else None
            )
        cached_templates = self._hit_check_key_templates[cache_key]
        if cached_templates is None:
            return [
                self.make_full_key(group_id, block_hash, rank, stage, pcp_rank)
                for stage in range(self.pp_size)
                for pcp_rank in range(pcp_size)
                for rank in range(num_ranks)
            ]
        return [template.replace(_BLOCK_HASH_PLACEHOLDER, block_hash) for template in cached_templates]
