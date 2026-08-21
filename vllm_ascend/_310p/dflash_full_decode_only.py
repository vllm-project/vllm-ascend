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

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor

from vllm_ascend.utils import is_310p


class DFlashFullDecodeState(Enum):
    """Closed states for the scoped 310P DFlash FULL_DECODE_ONLY policy."""

    EXPECTED_NONE_PREFILL = auto()
    EXPECTED_NONE_CHUNKED_PREFILL = auto()
    EXPECTED_NONE_PREFIX_TRANSITION = auto()
    EXPECTED_NONE_MIXED = auto()
    FULL_ELIGIBLE_UNIFORM_DECODE = auto()
    UNSUPPORTED_UNIFORM_DESCRIPTOR = auto()
    MODE_MISMATCH = auto()
    SAFETY_FAILURE = auto()


@dataclass(frozen=True)
class DFlashFullDecodeDecision:
    state: DFlashFullDecodeState
    expected_runtime_mode: CUDAGraphMode
    reason: str

    @property
    def graph_eligible(self) -> bool:
        return self.state is DFlashFullDecodeState.FULL_ELIGIBLE_UNIFORM_DECODE


class DFlashFullDecodeDispatchError(RuntimeError):
    """Raised when the exact-scope FULL decode contract cannot be honored."""


def is_310p_dflash_full_decode_only(vllm_config: VllmConfig) -> bool:
    """Return whether the exact 310P DFlash FULL_DECODE_ONLY scope is active."""
    speculative_config = vllm_config.speculative_config
    return (
        is_310p()
        and speculative_config is not None
        and speculative_config.method == "dflash"
        and vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    )


def should_skip_compiled_for_dflash_fdo_none(
    vllm_config: VllmConfig,
    *,
    runtime_mode: CUDAGraphMode,
    in_profile_run: bool,
) -> bool:
    """Keep expected NONE work out of the FULL-profile compiled callable.

    FULL_DECODE_ONLY profiles one dynamic callable so uniform decode can be
    captured as a whole graph. Reusing that callable for a later mixed
    prefill/decode batch also reuses its FULL-profile W8A8 FX topology. On
    310P that topology is not safe for the replacement-prefill token shape.

    Runtime NONE is the native execution mode for prefill and mixed batches;
    it is not a fallback from graph-eligible uniform decode. Profile runs must
    still enter the compiler so subsequent FULL capture has a compiled target.
    """
    if in_profile_run or runtime_mode is not CUDAGraphMode.NONE:
        return False
    if not hasattr(vllm_config, "speculative_config") or not hasattr(
        vllm_config,
        "compilation_config",
    ):
        return False
    return is_310p_dflash_full_decode_only(vllm_config)


def classify_dflash_full_decode_batch(
    *,
    attn_state: Any,
    num_tokens: int,
    num_reqs: int,
    max_num_scheduled_tokens: int,
    uniform_decode_query_len: int,
    all_decode: bool,
    forced_uniform_capture: bool = False,
) -> DFlashFullDecodeDecision:
    """Classify a batch without changing dispatcher-owned graph selection."""
    state_name = getattr(attn_state, "name", None)
    structurally_uniform = (
        num_reqs > 0
        and uniform_decode_query_len > 0
        and max_num_scheduled_tokens == uniform_decode_query_len
        and num_tokens == num_reqs * uniform_decode_query_len
    )
    if forced_uniform_capture and structurally_uniform:
        return DFlashFullDecodeDecision(
            DFlashFullDecodeState.FULL_ELIGIBLE_UNIFORM_DECODE,
            CUDAGraphMode.FULL,
            "startup_uniform_decode_capture",
        )
    if state_name == "PrefillNoCache":
        return DFlashFullDecodeDecision(
            DFlashFullDecodeState.EXPECTED_NONE_PREFILL,
            CUDAGraphMode.NONE,
            "prefill",
        )
    if state_name == "ChunkedPrefill":
        return DFlashFullDecodeDecision(
            DFlashFullDecodeState.EXPECTED_NONE_CHUNKED_PREFILL,
            CUDAGraphMode.NONE,
            "chunked_prefill",
        )
    if state_name == "PrefillCacheHit":
        return DFlashFullDecodeDecision(
            DFlashFullDecodeState.EXPECTED_NONE_PREFIX_TRANSITION,
            CUDAGraphMode.NONE,
            "prefix_cache_transition",
        )
    if state_name == "SpecDecoding" and all_decode and structurally_uniform:
        return DFlashFullDecodeDecision(
            DFlashFullDecodeState.FULL_ELIGIBLE_UNIFORM_DECODE,
            CUDAGraphMode.FULL,
            "uniform_dflash_decode",
        )
    return DFlashFullDecodeDecision(
        DFlashFullDecodeState.EXPECTED_NONE_MIXED,
        CUDAGraphMode.NONE,
        "mixed_or_nonuniform_decode",
    )


def resolve_dflash_full_decode_descriptor(
    *,
    num_tokens: int,
    num_reqs: int,
    uniform_decode_query_len: int,
    capture_sizes: Sequence[int],
) -> int:
    """Resolve the same padded token descriptor the upstream dispatcher uses."""
    if uniform_decode_query_len <= 0 or num_reqs <= 0:
        raise DFlashFullDecodeDispatchError("uniform decode query length and request count must be positive")
    expected_tokens = num_reqs * uniform_decode_query_len
    if num_tokens != expected_tokens:
        raise DFlashFullDecodeDispatchError(
            "uniform decode token count mismatch: "
            f"tokens={num_tokens}, requests={num_reqs}, "
            f"query_len={uniform_decode_query_len}, expected={expected_tokens}"
        )
    normalized_sizes = sorted({int(size) for size in capture_sizes})
    candidate = next(
        (size for size in normalized_sizes if size >= num_tokens),
        None,
    )
    if candidate is None:
        raise DFlashFullDecodeDispatchError(
            "no configured FULL descriptor can hold uniform decode: "
            f"tokens={num_tokens}, capture_sizes={list(capture_sizes)}"
        )
    if candidate % uniform_decode_query_len != 0:
        raise DFlashFullDecodeDispatchError(
            "configured FULL descriptor is not divisible by uniform query length: "
            f"descriptor={candidate}, query_len={uniform_decode_query_len}, "
            f"capture_sizes={list(capture_sizes)}"
        )
    return candidate


def select_dflash_full_decode_slot_mapping(
    *,
    vllm_config: VllmConfig,
    attn_state: Any,
    slot_mapping: torch.Tensor,
    num_actual_tokens: int,
    num_input_tokens: int,
) -> torch.Tensor:
    """Keep the target slot-mapping view stable for padded FULL replay.

    A FULL descriptor may outlive one of the requests that selected it.  For
    example, three 16-token DFlash requests still replay the 64-token graph
    captured for four requests.  The padded tail is already filled with the
    KV-cache padding slot by the model runner, so the graph must retain the
    descriptor-sized view rather than shrinking it back to the 48 logical
    tokens in the attention metadata builder.
    """
    use_descriptor_view = (
        is_310p_dflash_full_decode_only(vllm_config) and getattr(attn_state, "name", None) == "SpecDecoding"
    )
    view_extent = num_input_tokens if use_descriptor_view else num_actual_tokens
    if view_extent < 0 or view_extent > slot_mapping.shape[0]:
        raise DFlashFullDecodeDispatchError(
            "slot_mapping view extent is outside the persistent buffer: "
            f"extent={view_extent}, buffer={slot_mapping.shape[0]}, "
            f"actual={num_actual_tokens}, input={num_input_tokens}"
        )
    return slot_mapping[:view_extent]


def validate_dflash_full_decode_dispatch(
    *,
    decision: DFlashFullDecodeDecision,
    runtime_mode: CUDAGraphMode,
    batch_descriptor: BatchDescriptor,
    expected_descriptor: int | None,
    strict: bool,
) -> str | None:
    """Validate dispatcher output and classify availability-only fallback."""
    if decision.graph_eligible:
        if runtime_mode is not CUDAGraphMode.FULL:
            reason = "eligible_uniform_decode_selected_none"
            if strict:
                raise DFlashFullDecodeDispatchError(
                    f"eligible uniform decode selected NONE: state={decision.state.name}, descriptor={batch_descriptor}"
                )
            return reason
        if expected_descriptor is None:
            raise DFlashFullDecodeDispatchError("eligible uniform decode has no validated FULL descriptor")
        if batch_descriptor.num_tokens != expected_descriptor:
            raise DFlashFullDecodeDispatchError(
                "FULL descriptor mismatch: "
                f"expected={expected_descriptor}, actual={batch_descriptor.num_tokens}, "
                f"descriptor={batch_descriptor}"
            )
        if not batch_descriptor.uniform:
            raise DFlashFullDecodeDispatchError(f"FULL descriptor is not uniform: {batch_descriptor}")
        return None

    if runtime_mode is not CUDAGraphMode.NONE:
        raise DFlashFullDecodeDispatchError(
            "expected NONE selected FULL: "
            f"state={decision.state.name}, runtime_mode={runtime_mode.name}, "
            f"descriptor={batch_descriptor}"
        )
    return None
