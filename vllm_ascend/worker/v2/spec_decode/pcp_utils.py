# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace as dataclass_replace
from typing import Any

from vllm.config import VllmConfig, replace
from vllm.logger import logger

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

    draft_profiling_chunk_config = profiling_chunk_config.copy()
    draft_profiling_chunk_config["enabled"] = False

    draft_additional_config = additional_config.copy()
    if nested:
        draft_scheduler_config = scheduler_config.copy()
        draft_scheduler_config["profiling_chunk_config"] = draft_profiling_chunk_config
        draft_additional_config["scheduler_config"] = draft_scheduler_config
    else:
        draft_additional_config["profiling_chunk_config"] = draft_profiling_chunk_config
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


def build_dspark_draft_config(
    vllm_config: VllmConfig,
) -> VllmConfig:
    """Add the Ascend settings that follow the PCP=1 DSpark draft topology.

    Upstream ``DFlashSpeculator.__init__`` copies the config and forces PCP=1
    (keeping DCP only for MLA drafts). This copies the same way, without
    re-validating the target config, and only adjusts the Ascend features
    tied to that topology: static EPLB when PCP was the only parallel
    dimension, and Engram DP sharing, which belongs to the target.
    """
    target_parallel_config = vllm_config.parallel_config
    if target_parallel_config.prefill_context_parallel_size > 1:
        enable_eplb = target_parallel_config.enable_eplb
        eplb_config = target_parallel_config.eplb_config
        if (
            enable_eplb
            and target_parallel_config.tensor_parallel_size == 1
            and target_parallel_config.data_parallel_size == 1
        ):
            # Replicated draft attention uses PCP=1. Keep EP, but use a static
            # expert layout when PCP is the target's only parallel dimension.
            enable_eplb = False
            # Copy declared fields without transient communicator-selection state.
            eplb_config = dataclass_replace(eplb_config, num_redundant_experts=0)
            logger.warning_once("EPLB is disabled for the replicated PCP draft model; target EPLB remains enabled.")

        engram_config = vllm_config.engram_config
        if (
            engram_config is not None
            and engram_config.dp_shared_memory
            and target_parallel_config.data_parallel_size == 1
        ):
            # The PCP=1 draft has no Engram tables; sharing belongs to the target.
            engram_config = replace(engram_config, dp_shared_memory=False)
        vllm_config = copy.copy(vllm_config)
        vllm_config.parallel_config = replace(
            target_parallel_config,
            enable_eplb=enable_eplb,
            eplb_config=eplb_config,
        )
        vllm_config.engram_config = engram_config
    return vllm_config
