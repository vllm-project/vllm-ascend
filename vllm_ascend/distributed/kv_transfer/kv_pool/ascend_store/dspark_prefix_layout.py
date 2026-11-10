# SPDX-License-Identifier: Apache-2.0
"""DSpark prefix ownership, available to the planner without store imports."""

from collections.abc import Sequence

from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs


def draft_only_group_ids(groups: Sequence[KVCacheGroupSpec]) -> tuple[int, ...]:
    """Find groups whose prefix objects live entirely in the draft companion.

    K3.1 mixes target MLA/indexer and draft layers, so its groups still need
    ordinary target hit keys. Other layouts can have separate draft groups,
    for which no ordinary target object is saved. ``is_eagle_group`` marks
    groups containing any draft layer and cannot distinguish these cases.

    Run before scheduler conversion selects one representative layer spec.
    Inspect global specs even when this PP rank has no local layer names.
    """
    result = []
    for group_id, group in enumerate(groups):
        spec = group.kv_cache_spec
        specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        if specs and all(layer_spec.is_draft for layer_spec in specs):
            result.append(group_id)
    return tuple(result)
