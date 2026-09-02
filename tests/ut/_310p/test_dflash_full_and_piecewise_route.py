# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

import vllm_ascend._310p.model_runner_310p as model_runner_310p
from vllm_ascend._310p.dflash_full_and_piecewise import (
    DFlashRuntimePhase,
    build_dflash_draft_forward_contract,
    build_dflash_hybrid_route_observation,
    classify_dflash_hybrid_route,
    get_310p_dflash_graph_capabilities,
    is_310p_dflash_effective_full,
    is_310p_dflash_effective_piecewise,
    is_310p_dflash_full_and_piecewise,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionState


def _config(mode: CUDAGraphMode, method: str | None = "dflash"):
    return SimpleNamespace(
        speculative_config=(SimpleNamespace(method=method, num_speculative_tokens=15) if method is not None else None),
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        additional_config={
            "ascend_compilation_config": {
                "dflash_full_and_piecewise_capture_config": {
                    "piecewise_capture_size": 64,
                    "full_capture_size": 160,
                },
            },
        },
    )


def test_hybrid_and_fdo_share_safe_int32_position_staging_scope():
    config = object()

    with (
        patch.object(
            model_runner_310p,
            "is_310p_dflash_full_and_piecewise",
            return_value=True,
        ),
        patch.object(
            model_runner_310p,
            "is_310p_dflash_full_decode_only",
            return_value=False,
        ),
    ):
        assert model_runner_310p._uses_dflash_graph_int32_position_staging_310(config)

    with (
        patch.object(
            model_runner_310p,
            "is_310p_dflash_full_and_piecewise",
            return_value=False,
        ),
        patch.object(
            model_runner_310p,
            "is_310p_dflash_full_decode_only",
            return_value=True,
        ),
    ):
        assert model_runner_310p._uses_dflash_graph_int32_position_staging_310(config)

    with (
        patch.object(
            model_runner_310p,
            "is_310p_dflash_full_and_piecewise",
            return_value=False,
        ),
        patch.object(
            model_runner_310p,
            "is_310p_dflash_full_decode_only",
            return_value=False,
        ),
    ):
        assert not model_runner_310p._uses_dflash_graph_int32_position_staging_310(config)


def test_hybrid_and_fdo_share_alignment_safe_rejection_scope():
    config = object()

    with (
        patch.object(
            model_runner_310p,
            "is_310p_dflash_full_and_piecewise",
            return_value=True,
        ),
        patch.object(
            model_runner_310p,
            "is_310p_dflash_full_decode_only",
            return_value=False,
        ),
    ):
        assert model_runner_310p._uses_dflash_graph_alignment_safe_rejection_310(config)

    with (
        patch.object(
            model_runner_310p,
            "is_310p_dflash_full_and_piecewise",
            return_value=False,
        ),
        patch.object(
            model_runner_310p,
            "is_310p_dflash_full_decode_only",
            return_value=True,
        ),
    ):
        assert model_runner_310p._uses_dflash_graph_alignment_safe_rejection_310(config)

    with (
        patch.object(
            model_runner_310p,
            "is_310p_dflash_full_and_piecewise",
            return_value=False,
        ),
        patch.object(
            model_runner_310p,
            "is_310p_dflash_full_decode_only",
            return_value=False,
        ),
    ):
        assert not model_runner_310p._uses_dflash_graph_alignment_safe_rejection_310(config)


@pytest.mark.parametrize(
    ("platform_310p", "method", "mode", "expected"),
    [
        (True, "dflash", CUDAGraphMode.FULL_AND_PIECEWISE, True),
        (False, "dflash", CUDAGraphMode.FULL_AND_PIECEWISE, False),
        (True, "mtp", CUDAGraphMode.FULL_AND_PIECEWISE, False),
        (True, None, CUDAGraphMode.FULL_AND_PIECEWISE, False),
        (True, "dflash", CUDAGraphMode.PIECEWISE, False),
        (True, "dflash", CUDAGraphMode.FULL, False),
        (True, "dflash", CUDAGraphMode.FULL_DECODE_ONLY, False),
        (True, "dflash", CUDAGraphMode.NONE, False),
    ],
)
def test_hybrid_scope_is_exact(platform_310p, method, mode, expected):
    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=platform_310p,
    ):
        assert is_310p_dflash_full_and_piecewise(_config(mode, method)) is expected


@pytest.mark.parametrize(
    ("mode", "piecewise", "full", "hybrid"),
    [
        (CUDAGraphMode.NONE, False, False, False),
        (CUDAGraphMode.PIECEWISE, True, False, False),
        (CUDAGraphMode.FULL, False, True, False),
        (CUDAGraphMode.FULL_DECODE_ONLY, False, True, False),
        (CUDAGraphMode.FULL_AND_PIECEWISE, True, True, True),
    ],
)
def test_capabilities_follow_upstream_mode_semantics(mode, piecewise, full, hybrid):
    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        capabilities = get_310p_dflash_graph_capabilities(_config(mode))

    assert capabilities.supports_piecewise is piecewise
    assert capabilities.supports_full is full
    assert capabilities.hybrid is hybrid


def test_non_target_scope_has_no_graph_capabilities():
    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        assert not get_310p_dflash_graph_capabilities(_config(CUDAGraphMode.PIECEWISE, "mtp")).any
        assert not get_310p_dflash_graph_capabilities(_config(CUDAGraphMode.PIECEWISE, None)).any


def test_effective_runtime_mode_is_not_inferred_from_hybrid_configuration():
    config = _config(CUDAGraphMode.FULL_AND_PIECEWISE)
    with patch(
        "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
        return_value=True,
    ):
        assert is_310p_dflash_effective_piecewise(config, CUDAGraphMode.PIECEWISE)
        assert not is_310p_dflash_effective_piecewise(config, CUDAGraphMode.FULL)
        assert is_310p_dflash_effective_full(config, CUDAGraphMode.FULL)
        assert not is_310p_dflash_effective_full(config, CUDAGraphMode.PIECEWISE)
        assert not is_310p_dflash_effective_full(config, CUDAGraphMode.NONE)


@pytest.mark.parametrize(
    ("attn_state", "all_decode", "scheduled", "expected_phase", "candidate"),
    [
        (
            AscendAttentionState.PrefillNoCache,
            False,
            [82],
            DFlashRuntimePhase.PREFILL,
            CUDAGraphMode.PIECEWISE,
        ),
        (
            AscendAttentionState.ChunkedPrefill,
            True,
            [64],
            DFlashRuntimePhase.CHUNKED_PREFILL,
            CUDAGraphMode.PIECEWISE,
        ),
        (
            AscendAttentionState.ChunkedPrefill,
            False,
            [16, 40],
            DFlashRuntimePhase.MIXED,
            CUDAGraphMode.PIECEWISE,
        ),
        (
            AscendAttentionState.SpecDecoding,
            True,
            [16, 16, 16],
            DFlashRuntimePhase.UNIFORM_SPEC,
            CUDAGraphMode.FULL,
        ),
    ],
)
def test_route_phase_candidates(attn_state, all_decode, scheduled, expected_phase, candidate):
    decision = classify_dflash_hybrid_route(
        attn_state=attn_state,
        num_reqs=len(scheduled),
        num_tokens=sum(scheduled),
        num_scheduled_tokens=np.asarray(scheduled, dtype=np.int32),
        all_decode=all_decode,
        num_speculative_tokens=15,
    )

    assert decision.phase is expected_phase
    assert decision.candidate_mode is candidate


@pytest.mark.parametrize("active_reqs", [1, 2, 4, 7, 9, 10])
@pytest.mark.parametrize("k", [1, 8, 15])
def test_uniform_spec_width_and_required_tokens_are_runtime_derived(active_reqs, k):
    width = k + 1
    decision = classify_dflash_hybrid_route(
        attn_state=AscendAttentionState.SpecDecoding,
        num_reqs=active_reqs,
        num_tokens=active_reqs * width,
        num_scheduled_tokens=np.full(active_reqs, width, dtype=np.int32),
        all_decode=True,
        num_speculative_tokens=k,
    )

    assert decision.phase is DFlashRuntimePhase.UNIFORM_SPEC
    assert decision.verification_width == width
    assert decision.required_tokens == active_reqs * width
    assert decision.candidate_mode is CUDAGraphMode.FULL


@pytest.mark.parametrize(
    ("descriptor", "expected_reason"),
    [
        (
            BatchDescriptor(num_tokens=62, num_reqs=7, uniform=True),
            "required_token_capacity_exceeded",
        ),
        (
            BatchDescriptor(num_tokens=63, num_reqs=6, uniform=True),
            "active_request_capacity_exceeded",
        ),
    ],
)
def test_selected_full_descriptor_reports_unsafe_capacity_contract(
    descriptor,
    expected_reason,
):
    decision = classify_dflash_hybrid_route(
        attn_state=AscendAttentionState.SpecDecoding,
        num_reqs=7,
        num_tokens=63,
        num_scheduled_tokens=np.full(7, 9, dtype=np.int32),
        all_decode=True,
        num_speculative_tokens=8,
    )

    observation = build_dflash_hybrid_route_observation(
        configured_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        effective_mode=CUDAGraphMode.FULL,
        decision=decision,
        descriptor=descriptor,
        max_num_reqs=20,
        max_capture_tokens=400,
    )

    assert observation.contract_mismatch_reason == expected_reason


def test_nonuniform_spec_is_piecewise_candidate_not_fake_full():
    decision = classify_dflash_hybrid_route(
        attn_state=AscendAttentionState.SpecDecoding,
        num_reqs=3,
        num_tokens=34,
        num_scheduled_tokens=np.asarray([16, 16, 2], dtype=np.int32),
        all_decode=True,
        num_speculative_tokens=15,
    )

    assert decision.phase is DFlashRuntimePhase.MIXED
    assert decision.candidate_mode is CUDAGraphMode.PIECEWISE
    assert decision.contract_reason == "nonuniform_spec_uses_piecewise"


def test_forced_uniform_dummy_capture_is_observed_as_full_candidate():
    decision = classify_dflash_hybrid_route(
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_reqs=10,
        num_tokens=160,
        num_scheduled_tokens=np.full(10, 16, dtype=np.int32),
        all_decode=False,
        num_speculative_tokens=15,
        forced_uniform_capture=True,
    )

    assert decision.phase is DFlashRuntimePhase.UNIFORM_SPEC
    assert decision.candidate_mode is CUDAGraphMode.FULL
    assert decision.contract_reason == "forced_uniform_capture_contract"


def test_route_observation_reports_descriptor_capacity_and_padding_without_constants():
    decision = classify_dflash_hybrid_route(
        attn_state=AscendAttentionState.SpecDecoding,
        num_reqs=7,
        num_tokens=63,
        num_scheduled_tokens=np.full(7, 9, dtype=np.int32),
        all_decode=True,
        num_speculative_tokens=8,
    )
    observation = build_dflash_hybrid_route_observation(
        configured_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        effective_mode=CUDAGraphMode.FULL,
        decision=decision,
        descriptor=BatchDescriptor(num_tokens=72, num_reqs=8, uniform=True),
        max_num_reqs=20,
        max_capture_tokens=400,
    )

    assert observation.active_num_reqs == 7
    assert observation.real_num_tokens == 63
    assert observation.physical_request_capacity == 8
    assert observation.physical_token_capacity == 72
    assert observation.padding_request_count == 1
    assert observation.padding_token_count == 9
    assert observation.padding_ratio == pytest.approx(9 / 72)
    assert observation.fallback_reason is None
    assert observation.contract_mismatch_reason is None


def test_route_observation_preserves_existing_full_to_piecewise_fallback():
    decision = classify_dflash_hybrid_route(
        attn_state=AscendAttentionState.SpecDecoding,
        num_reqs=10,
        num_tokens=160,
        num_scheduled_tokens=np.full(10, 16, dtype=np.int32),
        all_decode=True,
        num_speculative_tokens=15,
    )
    observation = build_dflash_hybrid_route_observation(
        configured_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        effective_mode=CUDAGraphMode.PIECEWISE,
        decision=decision,
        descriptor=BatchDescriptor(num_tokens=192),
        max_num_reqs=20,
        max_capture_tokens=400,
    )

    assert observation.selected_mode is CUDAGraphMode.PIECEWISE
    assert observation.fallback_reason == "full_descriptor_unavailable"
    assert observation.physical_request_capacity == 20
    assert observation.padding_token_count == 32


def test_route_observation_identifies_capture_capacity_fallback():
    decision = classify_dflash_hybrid_route(
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_reqs=2,
        num_tokens=420,
        num_scheduled_tokens=np.asarray([210, 210], dtype=np.int32),
        all_decode=True,
        num_speculative_tokens=15,
    )
    observation = build_dflash_hybrid_route_observation(
        configured_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        effective_mode=CUDAGraphMode.NONE,
        decision=decision,
        descriptor=BatchDescriptor(num_tokens=420),
        max_num_reqs=20,
        max_capture_tokens=400,
    )

    assert observation.fallback_reason == "token_capacity_exceeded"
    assert observation.contract_mismatch_reason is None


@pytest.mark.parametrize("active_reqs", [1, 2, 4, 7, 9, 10])
@pytest.mark.parametrize("k", [1, 8, 15])
def test_draft_forward_contract_is_derived_from_draft_query_lens(
    active_reqs,
    k,
):
    query_width = k + 1
    logical_tokens = active_reqs * query_width
    contract = build_dflash_draft_forward_contract(
        logical_query_lens=[query_width] * active_reqs,
        physical_token_capacity=max(logical_tokens, 160),
        physical_request_capacity=16,
    )

    assert contract.logical_num_reqs == active_reqs
    assert contract.logical_num_tokens == logical_tokens
    assert contract.query_lens == (query_width,) * active_reqs
    assert contract.query_start_loc == tuple(
        request_index * query_width
        for request_index in range(active_reqs + 1)
    )
    assert contract.graph_eligible
    assert contract.fallback_reason is None


def test_draft_forward_contract_does_not_assume_target_descriptor_topology():
    # The parent Target descriptor for this diagnostic case is 160/10.  The
    # Draft forward owns an independent 32-token/4-request physical buffer and
    # a non-Target logical topology.  Only the Draft values are passed here.
    contract = build_dflash_draft_forward_contract(
        logical_query_lens=[3, 5],
        physical_token_capacity=32,
        physical_request_capacity=4,
    )

    assert contract.logical_num_reqs == 2
    assert contract.logical_num_tokens == 8
    assert contract.query_lens == (3, 5)
    assert contract.query_start_loc == (0, 3, 8)
    assert contract.padding_request_count == 2
    assert contract.padding_token_count == 24
    assert contract.graph_eligible


@pytest.mark.parametrize(
    ("query_lens", "token_capacity", "request_capacity", "reason"),
    [
        ([9] * 4, 35, 4, "draft_token_capacity_exceeded"),
        ([2] * 5, 16, 4, "draft_request_capacity_exceeded"),
    ],
)
def test_draft_forward_contract_rejects_unsafe_replay_without_truncating(
    query_lens,
    token_capacity,
    request_capacity,
    reason,
):
    contract = build_dflash_draft_forward_contract(
        logical_query_lens=query_lens,
        physical_token_capacity=token_capacity,
        physical_request_capacity=request_capacity,
    )

    assert not contract.graph_eligible
    assert contract.fallback_reason == reason
    assert contract.logical_num_reqs == len(query_lens)
    assert contract.logical_num_tokens == sum(query_lens)
