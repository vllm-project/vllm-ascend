# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""310P DFlash FULL_AND_PIECEWISE capability and route semantics.

This module deliberately keeps configured capabilities separate from the
runtime mode selected by vLLM's existing cudagraph dispatcher.  It contains no
descriptor selection or fallback implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import numpy as np
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor

from vllm_ascend.ascend_config import (
    DFLASH_FULL_AND_PIECEWISE_CAPTURE_CONFIG,
    DFlashFullAndPiecewiseCaptureConfig,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import is_310p


@dataclass(frozen=True)
class DFlashGraphCapabilities:
    supports_piecewise: bool
    supports_full: bool
    hybrid: bool

    @property
    def any(self) -> bool:
        return self.supports_piecewise or self.supports_full


class DFlashRuntimePhase(str, Enum):
    PREFILL = "prefill"
    CHUNKED_PREFILL = "chunked_prefill"
    MIXED = "mixed"
    UNIFORM_SPEC = "uniform_spec"


@dataclass(frozen=True)
class DFlashHybridRouteDecision:
    phase: DFlashRuntimePhase
    candidate_mode: CUDAGraphMode
    active_num_reqs: int
    real_num_tokens: int
    num_speculative_tokens: int
    verification_width: int
    required_tokens: int
    contract_reason: str


@dataclass(frozen=True)
class DFlashHybridRouteObservation:
    configured_mode: CUDAGraphMode
    effective_mode: CUDAGraphMode
    runtime_phase: DFlashRuntimePhase
    active_num_reqs: int
    real_num_tokens: int
    num_speculative_tokens: int
    verification_width: int
    required_tokens: int
    candidate_mode: CUDAGraphMode
    selected_mode: CUDAGraphMode
    descriptor: BatchDescriptor
    physical_request_capacity: int
    physical_token_capacity: int
    padding_request_count: int
    padding_token_count: int
    padding_ratio: float
    fallback_reason: str | None
    contract_mismatch_reason: str | None


@dataclass(frozen=True)
class DFlashDraftForwardContract:
    """Logical Draft-forward workload bound to a physical graph capacity."""

    logical_num_reqs: int
    logical_num_tokens: int
    query_lens: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    physical_request_capacity: int
    physical_token_capacity: int
    padding_request_count: int
    padding_token_count: int
    graph_eligible: bool
    fallback_reason: str | None


def build_dflash_draft_forward_contract(
    *,
    logical_query_lens: Sequence[int],
    physical_token_capacity: int,
    physical_request_capacity: int,
) -> DFlashDraftForwardContract:
    """Build a Draft-owned logical view without consulting Target topology."""
    query_lens = tuple(int(length) for length in logical_query_lens)
    if not query_lens:
        raise ValueError("Draft forward requires at least one logical request")
    if any(length <= 0 for length in query_lens):
        raise ValueError("Draft query lengths must be positive")
    if physical_token_capacity <= 0:
        raise ValueError("Draft physical token capacity must be positive")
    if physical_request_capacity <= 0:
        raise ValueError("Draft physical request capacity must be positive")

    query_start_loc = [0]
    for length in query_lens:
        query_start_loc.append(query_start_loc[-1] + length)

    logical_num_reqs = len(query_lens)
    logical_num_tokens = query_start_loc[-1]
    fallback_reason = None
    if logical_num_reqs > physical_request_capacity:
        fallback_reason = "draft_request_capacity_exceeded"
    elif logical_num_tokens > physical_token_capacity:
        fallback_reason = "draft_token_capacity_exceeded"

    return DFlashDraftForwardContract(
        logical_num_reqs=logical_num_reqs,
        logical_num_tokens=logical_num_tokens,
        query_lens=query_lens,
        query_start_loc=tuple(query_start_loc),
        physical_request_capacity=physical_request_capacity,
        physical_token_capacity=physical_token_capacity,
        padding_request_count=max(
            physical_request_capacity - logical_num_reqs,
            0,
        ),
        padding_token_count=max(
            physical_token_capacity - logical_num_tokens,
            0,
        ),
        graph_eligible=fallback_reason is None,
        fallback_reason=fallback_reason,
    )


def _is_310p_dflash(vllm_config: VllmConfig) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    return is_310p() and speculative_config is not None and speculative_config.method == "dflash"


def is_310p_dflash_full_and_piecewise(vllm_config: VllmConfig) -> bool:
    """Return whether the exact 310P DFlash hybrid configuration is active."""
    return (
        _is_310p_dflash(vllm_config)
        and getattr(
            getattr(vllm_config, "compilation_config", None),
            "cudagraph_mode",
            None,
        )
        == CUDAGraphMode.FULL_AND_PIECEWISE
        and get_dflash_full_and_piecewise_capture_config(vllm_config)
        is not None
    )


def get_dflash_full_and_piecewise_capture_config(
    vllm_config: VllmConfig,
) -> DFlashFullAndPiecewiseCaptureConfig | None:
    """Read the explicit production portfolio without changing defaults."""
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    ascend_compilation = additional_config.get(
        "ascend_compilation_config",
        {},
    )
    if not isinstance(ascend_compilation, dict):
        return None
    raw = ascend_compilation.get(
        DFLASH_FULL_AND_PIECEWISE_CAPTURE_CONFIG
    )
    return DFlashFullAndPiecewiseCaptureConfig.from_raw(raw)


def apply_dflash_full_and_piecewise_capture_config(
    vllm_config: VllmConfig,
) -> bool:
    """Install the explicit descriptor union for the exact production scope.

    Ownership is deliberately not encoded in vLLM's shared size list. It is
    installed later by ``initialize_dflash_full_and_piecewise_cudagraph_keys``.
    """
    portfolio = get_dflash_full_and_piecewise_capture_config(vllm_config)
    if portfolio is None or not is_310p_dflash_full_and_piecewise(vllm_config):
        return False

    speculative_config = vllm_config.speculative_config
    verification_width = int(speculative_config.num_speculative_tokens) + 1
    full_size = portfolio.full_capture_size
    piecewise_size = portfolio.piecewise_capture_size
    if full_size % verification_width != 0:
        raise ValueError(
            "full_capture_size must be divisible by the DFlash verification "
            f"width ({verification_width}), got {full_size}"
        )

    scheduler_config = vllm_config.scheduler_config
    logical_upper_bound = (
        int(scheduler_config.max_num_seqs) * verification_width
    )
    if full_size > logical_upper_bound:
        raise ValueError(
            "full_capture_size exceeds the uniform SPEC_DECODE logical "
            f"deployment bound: size={full_size}, "
            f"bound={logical_upper_bound}"
        )

    max_num_batched_tokens = int(
        scheduler_config.max_num_batched_tokens
    )
    if max(piecewise_size, full_size) > max_num_batched_tokens:
        raise ValueError(
            "DFlash FULL_AND_PIECEWISE capture capacity exceeds "
            "max_num_batched_tokens: "
            f"piecewise={piecewise_size}, full={full_size}, "
            f"max_num_batched_tokens={max_num_batched_tokens}"
        )

    compilation_config = vllm_config.compilation_config
    capture_sizes = sorted({piecewise_size, full_size})
    compilation_config.cudagraph_capture_sizes = capture_sizes
    compilation_config.max_cudagraph_capture_size = capture_sizes[-1]
    return True


def initialize_dflash_full_and_piecewise_cudagraph_keys(
    dispatcher: Any,
    cudagraph_mode: CUDAGraphMode,
    uniform_decode_query_len: int = 1,
) -> bool:
    """Create exact mode-owned keys for the explicit production portfolio.

    This is a capture-planner adapter only. Runtime dispatch continues through
    vLLM's unmodified ``CudagraphDispatcher.dispatch`` implementation.
    """
    vllm_config = dispatcher.vllm_config
    portfolio = get_dflash_full_and_piecewise_capture_config(vllm_config)
    if portfolio is None or not is_310p_dflash_full_and_piecewise(vllm_config):
        return False
    if cudagraph_mode not in (
        CUDAGraphMode.FULL_AND_PIECEWISE,
        CUDAGraphMode.PIECEWISE,
    ):
        return False

    # vLLM 0.24 normalizes its shared list for uniform speculative decode.
    # Restore the explicit union before building the mode-specific inventory;
    # FULL divisibility was already validated at platform configuration time.
    compilation_config = dispatcher.compilation_config
    capture_sizes = sorted(
        {
            portfolio.piecewise_capture_size,
            portfolio.full_capture_size,
        }
    )
    compilation_config.cudagraph_capture_sizes = capture_sizes
    compilation_config.max_cudagraph_capture_size = capture_sizes[-1]

    dispatcher.cudagraph_mode = cudagraph_mode
    dispatcher.cudagraph_keys = {
        CUDAGraphMode.PIECEWISE: set(),
        CUDAGraphMode.FULL: set(),
    }
    dispatcher._compute_bs_to_padded_graph_size()

    lora_cases = dispatcher._get_lora_cases()
    dispatcher.captured_lora_counts = [
        lora_count for lora_count in lora_cases if lora_count
    ]
    for num_active_loras in lora_cases:
        piecewise_descriptor = dispatcher._create_padded_batch_descriptor(
            portfolio.piecewise_capture_size,
            False,
            num_active_loras > 0,
            num_active_loras,
        )
        dispatcher.add_cudagraph_key(
            CUDAGraphMode.PIECEWISE,
            replace(
                piecewise_descriptor,
                num_reqs=None,
                uniform=False,
            ),
        )

    if cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE:
        if uniform_decode_query_len <= 1:
            raise ValueError(
                "DFlash FULL_AND_PIECEWISE requires speculative verification "
                "width greater than one"
            )
        for num_active_loras in lora_cases:
            dispatcher.add_cudagraph_key(
                CUDAGraphMode.FULL,
                dispatcher._create_padded_batch_descriptor(
                    portfolio.full_capture_size,
                    True,
                    num_active_loras > 0,
                    num_active_loras,
                ),
            )

    dispatcher.keys_initialized = True
    return True


def get_310p_dflash_graph_capabilities(
    vllm_config: VllmConfig,
) -> DFlashGraphCapabilities:
    """Return configured graph capabilities without selecting a runtime mode."""
    if not _is_310p_dflash(vllm_config):
        return DFlashGraphCapabilities(False, False, False)

    compilation_config = getattr(vllm_config, "compilation_config", None)
    configured_mode = getattr(
        compilation_config,
        "cudagraph_mode",
        CUDAGraphMode.NONE,
    )
    if (
        configured_mode == CUDAGraphMode.FULL_AND_PIECEWISE
        and not is_310p_dflash_full_and_piecewise(vllm_config)
    ):
        return DFlashGraphCapabilities(False, False, False)
    return DFlashGraphCapabilities(
        supports_piecewise=configured_mode.has_mode(CUDAGraphMode.PIECEWISE),
        supports_full=configured_mode.has_mode(CUDAGraphMode.FULL),
        hybrid=configured_mode == CUDAGraphMode.FULL_AND_PIECEWISE,
    )


def is_310p_dflash_effective_piecewise(
    vllm_config: VllmConfig,
    runtime_mode: CUDAGraphMode,
) -> bool:
    capabilities = get_310p_dflash_graph_capabilities(vllm_config)
    return capabilities.supports_piecewise and runtime_mode == CUDAGraphMode.PIECEWISE


def is_310p_dflash_effective_full(
    vllm_config: VllmConfig,
    runtime_mode: CUDAGraphMode,
) -> bool:
    capabilities = get_310p_dflash_graph_capabilities(vllm_config)
    return capabilities.supports_full and runtime_mode == CUDAGraphMode.FULL


def classify_dflash_hybrid_route(
    *,
    attn_state: AscendAttentionState | None,
    num_reqs: int,
    num_tokens: int,
    num_scheduled_tokens: Sequence[int] | np.ndarray,
    all_decode: bool,
    num_speculative_tokens: int,
    forced_uniform_capture: bool = False,
) -> DFlashHybridRouteDecision:
    """Classify the hybrid candidate while leaving selection to the dispatcher."""
    if num_reqs <= 0:
        raise ValueError("active request count must be positive")
    if num_speculative_tokens < 0:
        raise ValueError("num_speculative_tokens must be non-negative")

    scheduled = np.asarray(num_scheduled_tokens, dtype=np.int64).reshape(-1)
    if scheduled.size != num_reqs:
        raise ValueError(
            "num_scheduled_tokens must contain one value per active request: "
            f"expected={num_reqs}, actual={scheduled.size}"
        )

    verification_width = num_speculative_tokens + 1
    required_tokens = num_reqs * verification_width
    uniform_spec = forced_uniform_capture or (
        attn_state == AscendAttentionState.SpecDecoding
        and all_decode
        and num_tokens == required_tokens
        and bool(np.all(scheduled == verification_width))
    )
    if uniform_spec:
        return DFlashHybridRouteDecision(
            phase=DFlashRuntimePhase.UNIFORM_SPEC,
            candidate_mode=CUDAGraphMode.FULL,
            active_num_reqs=num_reqs,
            real_num_tokens=num_tokens,
            num_speculative_tokens=num_speculative_tokens,
            verification_width=verification_width,
            required_tokens=required_tokens,
            contract_reason=(
                "forced_uniform_capture_contract" if forced_uniform_capture else "uniform_spec_contract_satisfied"
            ),
        )

    if attn_state == AscendAttentionState.PrefillNoCache:
        phase = DFlashRuntimePhase.PREFILL
        reason = "prefill_uses_piecewise"
    elif attn_state in (
        AscendAttentionState.ChunkedPrefill,
        AscendAttentionState.PrefillCacheHit,
    ):
        has_spec_width = bool(np.any(scheduled == verification_width))
        has_other_width = bool(np.any(scheduled != verification_width))
        mixed = num_reqs > 1 and (not all_decode or (has_spec_width and has_other_width))
        phase = DFlashRuntimePhase.MIXED if mixed else DFlashRuntimePhase.CHUNKED_PREFILL
        reason = "mixed_batch_uses_piecewise" if mixed else "chunked_prefill_uses_piecewise"
    else:
        phase = DFlashRuntimePhase.MIXED
        reason = "nonuniform_spec_uses_piecewise"

    return DFlashHybridRouteDecision(
        phase=phase,
        candidate_mode=CUDAGraphMode.PIECEWISE,
        active_num_reqs=num_reqs,
        real_num_tokens=num_tokens,
        num_speculative_tokens=num_speculative_tokens,
        verification_width=verification_width,
        required_tokens=required_tokens,
        contract_reason=reason,
    )


def build_dflash_hybrid_route_observation(
    *,
    configured_mode: CUDAGraphMode,
    effective_mode: CUDAGraphMode,
    decision: DFlashHybridRouteDecision,
    descriptor: BatchDescriptor,
    max_num_reqs: int,
    max_capture_tokens: int | None,
) -> DFlashHybridRouteObservation:
    """Build debug telemetry from runtime values and the selected descriptor."""
    physical_token_capacity = int(descriptor.num_tokens)
    descriptor_num_reqs = descriptor.num_reqs
    physical_request_capacity = int(descriptor_num_reqs) if descriptor_num_reqs is not None else int(max_num_reqs)
    padding_request_count = max(
        physical_request_capacity - decision.active_num_reqs,
        0,
    )
    padding_token_count = max(
        physical_token_capacity - decision.real_num_tokens,
        0,
    )
    padding_ratio = padding_token_count / physical_token_capacity if physical_token_capacity > 0 else 0.0

    fallback_reason: str | None = None
    contract_mismatch_reason: str | None = None
    if effective_mode == CUDAGraphMode.FULL:
        if decision.active_num_reqs > physical_request_capacity:
            contract_mismatch_reason = "active_request_capacity_exceeded"
        elif decision.required_tokens > physical_token_capacity:
            contract_mismatch_reason = "required_token_capacity_exceeded"

    if effective_mode != decision.candidate_mode:
        if (
            effective_mode == CUDAGraphMode.NONE
            and max_capture_tokens is not None
            and decision.real_num_tokens > max_capture_tokens
        ):
            fallback_reason = "token_capacity_exceeded"
        elif decision.candidate_mode == CUDAGraphMode.FULL and effective_mode == CUDAGraphMode.PIECEWISE:
            fallback_reason = "full_descriptor_unavailable"
        elif decision.candidate_mode == CUDAGraphMode.FULL and effective_mode == CUDAGraphMode.NONE:
            fallback_reason = "no_matching_full_or_piecewise_descriptor"
        elif decision.candidate_mode == CUDAGraphMode.PIECEWISE and effective_mode == CUDAGraphMode.NONE:
            fallback_reason = "no_matching_piecewise_descriptor"
        else:
            fallback_reason = "unexpected_dispatch_result"
            contract_mismatch_reason = f"candidate={decision.candidate_mode.name},effective={effective_mode.name}"

    return DFlashHybridRouteObservation(
        configured_mode=configured_mode,
        effective_mode=effective_mode,
        runtime_phase=decision.phase,
        active_num_reqs=decision.active_num_reqs,
        real_num_tokens=decision.real_num_tokens,
        num_speculative_tokens=decision.num_speculative_tokens,
        verification_width=decision.verification_width,
        required_tokens=decision.required_tokens,
        candidate_mode=decision.candidate_mode,
        selected_mode=effective_mode,
        descriptor=descriptor,
        physical_request_capacity=physical_request_capacity,
        physical_token_capacity=physical_token_capacity,
        padding_request_count=padding_request_count,
        padding_token_count=padding_token_count,
        padding_ratio=padding_ratio,
        fallback_reason=fallback_reason,
        contract_mismatch_reason=contract_mismatch_reason,
    )
