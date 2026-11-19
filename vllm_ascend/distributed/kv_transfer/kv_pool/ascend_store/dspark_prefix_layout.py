# SPDX-License-Identifier: Apache-2.0
"""DSpark prefix ownership, available to the planner without store imports."""

from collections.abc import Sequence

from vllm_ascend.core.kv_cache_layers import CacheGroup


def draft_only_group_ids(groups: Sequence[CacheGroup], draft_names: set[str] | None = None) -> tuple[int, ...]:
    """Find groups whose prefix objects live entirely in the draft companion.

    K3.1 mixes target MLA/indexer and draft layers, so its groups still need
    ordinary target hit keys. Other layouts can have separate draft groups,
    for which no ordinary target object is saved. ``is_eagle_group`` marks
    groups containing any draft layer and cannot distinguish these cases.

    Run before scheduler conversion selects one representative layer spec.
    Inspect global specs even when this PP rank has no local layer names.
    """
    if draft_names is not None:
        return tuple(
            i for i, group in enumerate(groups) if group.layer_names and draft_names.issuperset(group.layer_names)
        )
    from vllm.v1.kv_cache_interface import iter_layer_specs

    result = []
    for group_id, group in enumerate(groups):
        specs = iter_layer_specs(group.kv_cache_spec)
        if specs and all(spec.is_draft for spec in specs):
            result.append(group_id)
    return tuple(result)
