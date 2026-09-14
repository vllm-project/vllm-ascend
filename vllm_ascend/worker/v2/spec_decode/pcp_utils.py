# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

from vllm.config import VllmConfig, replace

from vllm_ascend.utils import vllm_version_is

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
    if not vllm_version_is("0.28.0"):
        # vLLM #56107 also partitions through the speculator's own manager.
        speculator.pcp_manager = None
    try:
        yield
    finally:
        model_state.pcp_manager = target_pcp_manager
        if not vllm_version_is("0.28.0"):
            speculator.pcp_manager = target_pcp_manager


def prepare_replicated_pcp_config(
    vllm_config: VllmConfig,
) -> tuple[VllmConfig, bool]:
    """Return the draft execution config and whether target PCP is replicated."""
    target_parallel_config = vllm_config.parallel_config
    replicated_pcp = target_parallel_config.prefill_context_parallel_size > 1
    if replicated_pcp:
        vllm_config = replace(
            vllm_config,
            parallel_config=replace(
                target_parallel_config,
                prefill_context_parallel_size=1,
            ),
        )
    return vllm_config, replicated_pcp
