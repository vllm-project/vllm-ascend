from __future__ import annotations

from typing import Any

import torch
from vllm.logger import logger


def get_group_pp_ranks(
    owners: tuple[tuple[int, ...], ...] | None,
    group_id: int,
    pp_size: int,
) -> list[int]:
    """Return PP ranks owning a cache group, or all ranks as fallback."""
    if owners is None or group_id < 0 or group_id >= len(owners):
        return list(range(pp_size))
    return list(owners[group_id])


def collect_pp_group_owners(
    kv_cache_config: Any,
    pp_size: int,
    num_kv_cache_groups: int,
    cpu_group: Any = None,
) -> tuple[tuple[int, ...], ...] | None:
    """Collect cache-group ownership from every pipeline stage.

    ``kv_cache_config`` is worker-local after vLLM projects the model onto
    pipeline stages. The rank-0 lookup server therefore cannot infer global
    ownership from its local ``layer_names``. Gather the per-stage presence
    mask while every worker is initializing and return a global manifest.

    ``None`` deliberately means that ownership could not be established. The
    lookup path treats that as a conservative all-PP-ranks fallback.
    """
    if pp_size <= 1:
        return tuple((0,) for _ in range(num_kv_cache_groups))

    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if groups is None or cpu_group is None:
        return None

    local_presence = tuple(bool(group.layer_names) for group in groups)
    gathered: list[tuple[bool, ...] | None] = [None] * pp_size
    try:
        torch.distributed.all_gather_object(gathered, local_presence, group=cpu_group)
    except (RuntimeError, ValueError) as error:
        logger.warning("Failed to collect PP cache-group ownership; lookup will check all PP ranks: %s", error)
        return None

    if any(presence is None or len(presence) != num_kv_cache_groups for presence in gathered):
        logger.warning("Inconsistent PP cache-group manifests; lookup will check all PP ranks")
        return None

    owners = tuple(
        tuple(pp_rank for pp_rank, presence in enumerate(gathered) if presence is not None and presence[group_id])
        for group_id in range(num_kv_cache_groups)
    )
    if any(not group_owners for group_owners in owners):
        logger.warning("PP cache-group manifest contains an unowned group; lookup will check all PP ranks")
        return None
    return owners
