# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace as dataclass_replace
from typing import TYPE_CHECKING, Protocol

from vllm.config import VllmConfig, replace
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.model_states.default import AscendModelState
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager


class ReplicatedPCPDraftSpeculator(Protocol):
    """State required to suspend target PCP for replicated draft execution."""

    replicated_pcp: bool
    model_state: "AscendModelState"
    pcp_manager: "AscendPCPManager | None"


@contextmanager
def disable_target_pcp_for_replicated_draft(
    speculator: ReplicatedPCPDraftSpeculator,
) -> Generator[None, None, None]:
    """Keep replicated PCP=1 draft out of target PCP partitioning."""
    target_pcp_manager = speculator.pcp_manager
    if not speculator.replicated_pcp or target_pcp_manager is None:
        yield
        return

    model_state = speculator.model_state
    # Target and draft share model_state, so validate the target manager
    # before temporarily detaching it for replicated PCP=1 execution.
    if model_state.pcp_manager is not target_pcp_manager:
        raise RuntimeError("Replicated draft execution requires model_state to use the target PCP manager.")

    model_state.pcp_manager = None
    try:
        yield
    finally:
        model_state.pcp_manager = target_pcp_manager


def prepare_replicated_pcp_config(
    vllm_config: VllmConfig,
) -> tuple[VllmConfig, bool]:
    """Return the draft execution config and whether target PCP is replicated."""
    target_parallel_config = vllm_config.parallel_config
    replicated_pcp = target_parallel_config.prefill_context_parallel_size > 1
    if replicated_pcp:
        enable_eplb = target_parallel_config.enable_eplb
        eplb_config = target_parallel_config.eplb_config
        if (
            enable_eplb
            and target_parallel_config.tensor_parallel_size == 1
            and target_parallel_config.data_parallel_size == 1
        ):
            # Replicated draft attention uses PCP=1. Keep EP, but use a static
            # expert layout when PCP is the target's only parallel dimension.
            # TODO: Refactor this policy when MTP supports prefill sharding across
            # PCP ranks; disabling draft EPLB assumes replicated PCP=1 execution.
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
        # TODO: Separate draft execution settings from the worker topology.
        # Temporarily disable DCP during reconstruction to avoid validating the
        # target model with PCP=1; restoring DCP below does not rerun DCP checks
        # or recompute DCP-dependent settings.
        vllm_config = replace(
            vllm_config,
            parallel_config=replace(
                target_parallel_config,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                enable_eplb=enable_eplb,
                eplb_config=eplb_config,
            ),
            engram_config=engram_config,
        )
        vllm_config.parallel_config.decode_context_parallel_size = target_parallel_config.decode_context_parallel_size
    return vllm_config, replicated_pcp
