# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor

from vllm_ascend.utils import is_310p


class DFlashFullExecutionSignature(Enum):
    """Execution identities kept separate by the private FULL graph store."""

    PREFILL = auto()
    CHUNKED_PREFILL = auto()
    MIXED = auto()
    DECODE = auto()
    SPEC_DECODE = auto()
    MIXED_WITH_SPEC = auto()


@dataclass(frozen=True)
class DFlashFullDecision:
    """An immutable observation of the parent dispatch result."""

    signature: DFlashFullExecutionSignature
    component: Literal["target", "draft"]
    rank: int
    parent_mode: CUDAGraphMode
    descriptor: BatchDescriptor
    within_capture_range: bool

    @property
    def graph_eligible(self) -> bool:
        return self.parent_mode is CUDAGraphMode.FULL and self.within_capture_range


class DFlashFullDispatchError(RuntimeError):
    """Raised when native FULL unexpectedly loses an eligible invocation."""


def is_310p_dflash_full(
    vllm_config: VllmConfig,
    *,
    runtime_mode: CUDAGraphMode | None = None,
) -> bool:
    """Return whether the isolated 310P DFlash FULL scope is active.

    ``runtime_mode`` is omitted only while constructing the engine-owned
    controller or querying attention capability. Runtime graph routing must
    pass the parent-selected mode explicitly.
    """
    speculative_config = vllm_config.speculative_config
    configured = (
        is_310p()
        and speculative_config is not None
        and speculative_config.method == "dflash"
        and vllm_config.compilation_config.cudagraph_mode is CUDAGraphMode.FULL
    )
    return configured and (runtime_mode is None or runtime_mode is CUDAGraphMode.FULL)


def _execution_signature(*, state_name: str | None, all_decode: bool) -> DFlashFullExecutionSignature:
    if state_name == "PrefillNoCache":
        return DFlashFullExecutionSignature.PREFILL
    if state_name == "ChunkedPrefill":
        return DFlashFullExecutionSignature.CHUNKED_PREFILL
    if state_name == "PrefillCacheHit":
        return DFlashFullExecutionSignature.MIXED
    if state_name == "SpecDecoding":
        return DFlashFullExecutionSignature.SPEC_DECODE if all_decode else DFlashFullExecutionSignature.MIXED_WITH_SPEC
    if state_name == "DecodeOnly" and all_decode:
        return DFlashFullExecutionSignature.DECODE
    return DFlashFullExecutionSignature.MIXED


def classify_dflash_full_execution(
    *,
    attn_state: Any,
    all_decode: bool,
    component: Literal["target", "draft"],
    rank: int,
    parent_mode: CUDAGraphMode,
    descriptor: BatchDescriptor,
    max_capture_tokens: int,
) -> DFlashFullDecision:
    """Classify parent-owned dispatch state without rebuilding its descriptor."""
    if component not in ("target", "draft"):
        raise ValueError(f"invalid DFlash FULL component: {component}")
    if rank < 0:
        raise ValueError(f"invalid DFlash FULL rank: {rank}")

    num_tokens = descriptor.num_tokens
    within_capture_range = 0 < num_tokens <= max_capture_tokens
    return DFlashFullDecision(
        signature=_execution_signature(
            state_name=getattr(attn_state, "name", None),
            all_decode=all_decode,
        ),
        component=component,
        rank=rank,
        parent_mode=parent_mode,
        descriptor=descriptor,
        within_capture_range=within_capture_range,
    )


def validate_dflash_full_dispatch(
    *,
    decision: DFlashFullDecision,
    upstream_full_allowed: bool,
    strict: bool,
) -> str | None:
    """Validate the final parent decision and return only closed fallbacks."""
    if decision.parent_mode is CUDAGraphMode.FULL:
        if not upstream_full_allowed or not decision.within_capture_range:
            raise DFlashFullDispatchError(_dispatch_error_context(decision, "invalid_parent_full_selection"))
        return None

    if not upstream_full_allowed:
        return "upstream_full_not_allowed"
    if not decision.within_capture_range:
        return "outside_capture_range"

    if strict:
        raise DFlashFullDispatchError(_dispatch_error_context(decision, "unexpected_in_range_fallback"))
    return "unexpected_in_range_fallback"


def _dispatch_error_context(decision: DFlashFullDecision, reason: str) -> str:
    return (
        f"DFlash FULL dispatch failure: reason={reason} requested=FULL "
        f"resolved={decision.parent_mode.name} component={decision.component} "
        f"rank={decision.rank} signature={decision.signature.name} "
        f"descriptor={decision.descriptor}"
    )


class DFlashFullController:
    """Small engine-owned policy observer with no process-global state."""

    def __init__(self, *, strict: bool) -> None:
        self.strict = strict
        self.latest_decision: DFlashFullDecision | None = None
        self.decision_count = 0
        self.closed_reason_counts: dict[str, int] = {}

    def classify(self, **kwargs: Any) -> DFlashFullDecision:
        decision = classify_dflash_full_execution(**kwargs)
        self.latest_decision = decision
        self.decision_count += 1
        return decision

    def validate(
        self,
        decision: DFlashFullDecision,
        *,
        upstream_full_allowed: bool,
    ) -> str | None:
        reason = validate_dflash_full_dispatch(
            decision=decision,
            upstream_full_allowed=upstream_full_allowed,
            strict=self.strict,
        )
        if reason is not None:
            self.closed_reason_counts[reason] = self.closed_reason_counts.get(reason, 0) + 1
        return reason
