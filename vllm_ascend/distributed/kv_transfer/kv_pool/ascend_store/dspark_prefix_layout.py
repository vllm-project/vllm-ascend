# SPDX-License-Identifier: Apache-2.0
"""DSpark prefix ownership, available to the planner without store imports."""

from collections.abc import Collection, Sequence

from vllm_ascend.core.kv_cache_interface import KVCacheGroup


def draft_only_group_ids(
    groups: Sequence[KVCacheGroup],
    draft_layer_names: Collection[str] | None = None,
) -> tuple[int, ...]:
    """Find groups whose prefix objects live entirely in the draft companion.

    K3.1 mixes target MLA/indexer and draft layers, so its groups still need
    ordinary target hit keys. Other layouts can have separate draft groups,
    for which no ordinary target object is saved. ``is_eagle_group`` marks
    groups containing any draft layer and cannot distinguish these cases.

    Run before scheduler conversion selects one representative layer spec.
    Inspect global specs even when this PP rank has no local layer names.
    """
    result = []
    if draft_layer_names is not None:
        draft_set = set(draft_layer_names)
        for group_id, group in enumerate(groups):
            if group.layer_names and all(name in draft_set for name in group.layer_names):
                result.append(group_id)
        return tuple(result)
    for group_id, group in enumerate(groups):
        specs = group.distinct_specs
        if specs and all(getattr(layer_spec, "is_draft", False) for layer_spec in specs):
            result.append(group_id)
    return tuple(result)
