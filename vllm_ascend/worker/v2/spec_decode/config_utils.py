# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

from vllm.config import VllmConfig

from vllm_ascend.ascend_config import validate_additional_config_bool


def _draft_additional_config(vllm_config: VllmConfig) -> dict[str, Any] | None:
    """Return a CPP-disabled copy when a PP target creates a PP=1 draft."""
    if vllm_config.parallel_config.pipeline_parallel_size <= 1:
        return None

    additional_config = vllm_config.additional_config
    if not isinstance(additional_config, dict):
        return None

    scheduler_config = additional_config.get("scheduler_config")
    if scheduler_config is None:
        scheduler_config = {}
    elif not isinstance(scheduler_config, dict):
        return None

    # Match SchedulerConfig.from_additional_config: the nested form wins over
    # the deprecated top-level form when both are present.
    if "profiling_chunk_config" in scheduler_config:
        profiling_chunk_config = scheduler_config["profiling_chunk_config"]
        config_path = "additional_config.scheduler_config.profiling_chunk_config"
        nested = True
    elif "profiling_chunk_config" in additional_config:
        profiling_chunk_config = additional_config["profiling_chunk_config"]
        config_path = "additional_config.profiling_chunk_config"
        nested = False
    else:
        return None

    if not isinstance(profiling_chunk_config, dict):
        return None

    enabled = validate_additional_config_bool(
        profiling_chunk_config.get("enabled", False),
        f"{config_path}.enabled",
    )
    if not enabled:
        return None

    draft_additional_config = deepcopy(additional_config)
    if nested:
        draft_additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] = False
    else:
        draft_additional_config["profiling_chunk_config"]["enabled"] = False
    return draft_additional_config


@contextmanager
def disable_profiling_chunk_for_draft(vllm_config: VllmConfig) -> Iterator[None]:
    """Temporarily expose CPP-disabled input while constructing a PP=1 draft.

    The draft still goes through the normal ``VllmConfig.replace`` validation.
    Rebinding the target's ``additional_config`` only for that call lets the
    resulting draft retain the copied input while the target is restored.
    """
    draft_additional_config = _draft_additional_config(vllm_config)
    if draft_additional_config is None:
        yield
        return

    target_additional_config = vllm_config.additional_config
    vllm_config.additional_config = draft_additional_config
    try:
        yield
    finally:
        vllm_config.additional_config = target_additional_config
