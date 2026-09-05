# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.model_states.default import AscendModelState


class ReplicatedPCPDraftSpeculator(Protocol):
    """State required to suspend target PCP for replicated draft execution."""

    replicated_pcp: bool
    model_state: "AscendModelState"
    pcp_manager: AscendPCPManager | None


@contextmanager
def disable_target_pcp_for_replicated_draft(
    speculator: ReplicatedPCPDraftSpeculator,
) -> Iterator[None]:
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
