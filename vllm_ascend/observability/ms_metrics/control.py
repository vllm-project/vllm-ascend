# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply an MS metric level in one process, and fan it out from the API."""

from __future__ import annotations

from typing import Any

from vllm.logger import logger

from vllm_ascend.observability.ms_metrics.level import canonical_metric_level, set_metric_level


def apply_process_metrics_level(level: Any) -> str:
    """Set this process's metric level. Returns the canonical name."""
    canonical = canonical_metric_level(level)
    set_metric_level(canonical)
    return canonical


def apply_engine_metrics_level(engine: Any, level: Any) -> str:
    """Set this EngineCore's level, then broadcast the same call to its workers."""
    canonical = apply_process_metrics_level(level)
    engine.collective_rpc("set_metrics_level", args=(canonical,))
    return canonical


def publish_metrics_level(level: Any) -> str:
    """Set the API process level and write ``metrics.level`` when the JSON exists.

    A later hot-reload reads that file. Writing it here keeps the file and the
    integer on the same value. Startup still rebuilds the file from defaults
    and ``additional_config.runtime_config``.
    """
    canonical = apply_process_metrics_level(level)
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        runtime_cfg = get_ascend_config().runtime_config
    except Exception:
        logger.warning("[ms_metrics] metric level set in this process only; runtime config is not available")
        return canonical
    if runtime_cfg is None:
        return canonical
    try:
        runtime_cfg.persist_metrics_level(canonical)
    except Exception:
        logger.warning("[ms_metrics] failed to persist metrics.level; in-process level is still %s", canonical)
    return canonical


async def broadcast_metrics_level(engine_client: Any, level: str) -> list[Any]:
    """Send ``set_metrics_level`` to every EngineCore this API server owns.

    Each EngineCore applies the level locally and collective-RPCs its workers.
    ``call_utility_all_async`` covers internal data parallel. A single-engine
    client uses ``call_utility_async``.
    """
    core = getattr(engine_client, "engine_core", None)
    if core is not None and hasattr(core, "call_utility_all_async"):
        return list(await core.call_utility_all_async("set_metrics_level", level))
    if core is not None and hasattr(core, "call_utility_async"):
        return [await core.call_utility_async("set_metrics_level", level)]
    return list(await engine_client.collective_rpc("set_metrics_level", args=(level,)))


async def apply_metrics_level_command(engine_client: Any, level: Any) -> str:
    """API entry: publish locally, then fan the canonical name out to engines."""
    canonical = publish_metrics_level(level)
    await broadcast_metrics_level(engine_client, canonical)
    return canonical
