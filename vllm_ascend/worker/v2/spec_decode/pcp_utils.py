# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import copy
from dataclasses import replace as dataclass_replace

from vllm.config import VllmConfig, replace
from vllm.logger import logger


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
