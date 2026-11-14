# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply an MS metric level in one process, and fan it out from the API."""

from __future__ import annotations

from typing import Any

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
    """Set the API process level.

    The level is applied in memory only; ``runtime_config.json`` is not
    written back. The file and the endpoint are two separate channels and
    must not be mixed: see the note in ``ms_service_metric.md`` — any later
    file edit (of any key) triggers a digest change and re-applies the
    file's ``metrics.level``, silently resetting an endpoint-set level.
    """
    return apply_process_metrics_level(level)


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
