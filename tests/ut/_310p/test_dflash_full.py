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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_ascend._310p.dflash_full import (
    DFlashFullController,
    DFlashFullDispatchError,
    DFlashFullExecutionSignature,
    classify_dflash_full_execution,
    is_310p_dflash_full,
    validate_dflash_full_dispatch,
)


def _config(method: str | None, mode: CUDAGraphMode) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=(SimpleNamespace(method=method) if method is not None else None),
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
    )


@pytest.mark.parametrize(
    ("is_310p_platform", "method", "configured_mode", "runtime_mode", "expected"),
    [
        (True, "dflash", CUDAGraphMode.FULL, CUDAGraphMode.FULL, True),
        (False, "dflash", CUDAGraphMode.FULL, CUDAGraphMode.FULL, False),
        (True, "mtp", CUDAGraphMode.FULL, CUDAGraphMode.FULL, False),
        (True, None, CUDAGraphMode.FULL, CUDAGraphMode.FULL, False),
        (True, "dflash", CUDAGraphMode.NONE, CUDAGraphMode.FULL, False),
        (True, "dflash", CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL, False),
        (True, "dflash", CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL, False),
        (True, "dflash", CUDAGraphMode.FULL, CUDAGraphMode.NONE, False),
        (True, "dflash", CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE, False),
    ],
)
def test_exact_full_activation(
    is_310p_platform: bool,
    method: str | None,
    configured_mode: CUDAGraphMode,
    runtime_mode: CUDAGraphMode,
    expected: bool,
) -> None:
    config = _config(method, configured_mode)
    with patch("vllm_ascend._310p.dflash_full.is_310p", return_value=is_310p_platform):
        assert is_310p_dflash_full(config, runtime_mode=runtime_mode) is expected


def test_configured_predicate_can_be_checked_before_runtime_dispatch() -> None:
    config = _config("dflash", CUDAGraphMode.FULL)
    with patch("vllm_ascend._310p.dflash_full.is_310p", return_value=True):
        assert is_310p_dflash_full(config) is True


@pytest.mark.parametrize(
    ("state", "all_decode", "component", "expected"),
    [
        ("PrefillNoCache", False, "target", DFlashFullExecutionSignature.PREFILL),
        (
            "ChunkedPrefill",
            False,
            "target",
            DFlashFullExecutionSignature.CHUNKED_PREFILL,
        ),
        ("PrefillCacheHit", False, "target", DFlashFullExecutionSignature.MIXED),
        ("DecodeOnly", True, "target", DFlashFullExecutionSignature.DECODE),
        (
            "SpecDecoding",
            True,
            "draft",
            DFlashFullExecutionSignature.SPEC_DECODE,
        ),
        (
            "SpecDecoding",
            False,
            "target",
            DFlashFullExecutionSignature.MIXED_WITH_SPEC,
        ),
    ],
)
def test_classifies_all_execution_signatures_without_mutating_descriptor(
    state: str,
    all_decode: bool,
    component: str,
    expected: DFlashFullExecutionSignature,
) -> None:
    descriptor = BatchDescriptor(num_tokens=16, num_reqs=1, uniform=True)
    before = repr(descriptor)

    decision = classify_dflash_full_execution(
        attn_state=SimpleNamespace(name=state),
        all_decode=all_decode,
        component=component,
        rank=0,
        parent_mode=CUDAGraphMode.FULL,
        descriptor=descriptor,
        max_capture_tokens=160,
    )

    assert decision.signature is expected
    assert decision.descriptor is descriptor
    assert repr(descriptor) == before
    assert decision.graph_eligible is True


def test_nonuniform_decode_is_mixed_without_reconstructing_descriptor() -> None:
    descriptor = BatchDescriptor(num_tokens=31, num_reqs=2, uniform=False)
    decision = classify_dflash_full_execution(
        attn_state=SimpleNamespace(name="DecodeOnly"),
        all_decode=False,
        component="target",
        rank=1,
        parent_mode=CUDAGraphMode.FULL,
        descriptor=descriptor,
        max_capture_tokens=160,
    )
    assert decision.signature is DFlashFullExecutionSignature.MIXED
    assert decision.descriptor is descriptor


def test_parent_rejected_context_is_a_closed_non_full_reason() -> None:
    descriptor = BatchDescriptor(num_tokens=16, num_reqs=1, uniform=True)
    decision = classify_dflash_full_execution(
        attn_state=SimpleNamespace(name="DecodeOnly"),
        all_decode=True,
        component="target",
        rank=0,
        parent_mode=CUDAGraphMode.NONE,
        descriptor=descriptor,
        max_capture_tokens=160,
    )
    reason = validate_dflash_full_dispatch(
        decision=decision,
        upstream_full_allowed=False,
        strict=True,
    )
    assert reason == "upstream_full_not_allowed"


def test_out_of_range_non_full_is_closed_without_claiming_replay() -> None:
    descriptor = BatchDescriptor(num_tokens=176, num_reqs=11, uniform=True)
    decision = classify_dflash_full_execution(
        attn_state=SimpleNamespace(name="DecodeOnly"),
        all_decode=True,
        component="draft",
        rank=1,
        parent_mode=CUDAGraphMode.NONE,
        descriptor=descriptor,
        max_capture_tokens=160,
    )
    reason = validate_dflash_full_dispatch(
        decision=decision,
        upstream_full_allowed=True,
        strict=True,
    )
    assert reason == "outside_capture_range"


@pytest.mark.parametrize("resolved", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_unexpected_in_range_fallback_fails_closed_with_exact_context(
    resolved: CUDAGraphMode,
) -> None:
    descriptor = BatchDescriptor(num_tokens=16, num_reqs=1, uniform=True)
    decision = classify_dflash_full_execution(
        attn_state=SimpleNamespace(name="SpecDecoding"),
        all_decode=True,
        component="draft",
        rank=2,
        parent_mode=resolved,
        descriptor=descriptor,
        max_capture_tokens=160,
    )
    with pytest.raises(
        DFlashFullDispatchError,
        match=("requested=FULL.*resolved=.*component=draft.*rank=2.*signature=SPEC_DECODE.*descriptor="),
    ):
        validate_dflash_full_dispatch(
            decision=decision,
            upstream_full_allowed=True,
            strict=True,
        )


def test_non_strict_unexpected_fallback_is_recorded_not_accepted() -> None:
    decision = classify_dflash_full_execution(
        attn_state=SimpleNamespace(name="DecodeOnly"),
        all_decode=True,
        component="target",
        rank=0,
        parent_mode=CUDAGraphMode.NONE,
        descriptor=BatchDescriptor(num_tokens=16, num_reqs=1, uniform=True),
        max_capture_tokens=160,
    )
    assert (
        validate_dflash_full_dispatch(
            decision=decision,
            upstream_full_allowed=True,
            strict=False,
        )
        == "unexpected_in_range_fallback"
    )


def test_parent_full_selection_validates_without_fallback() -> None:
    decision = classify_dflash_full_execution(
        attn_state=SimpleNamespace(name="PrefillNoCache"),
        all_decode=False,
        component="target",
        rank=0,
        parent_mode=CUDAGraphMode.FULL,
        descriptor=BatchDescriptor(num_tokens=102, uniform=False),
        max_capture_tokens=160,
    )
    assert (
        validate_dflash_full_dispatch(
            decision=decision,
            upstream_full_allowed=True,
            strict=True,
        )
        is None
    )


def test_controllers_keep_latest_decision_and_closed_reasons_per_instance() -> None:
    first = DFlashFullController(strict=True)
    second = DFlashFullController(strict=True)
    descriptor = BatchDescriptor(num_tokens=16, num_reqs=1, uniform=True)

    decision = first.classify(
        attn_state=SimpleNamespace(name="DecodeOnly"),
        all_decode=True,
        component="target",
        rank=0,
        parent_mode=CUDAGraphMode.NONE,
        descriptor=descriptor,
        max_capture_tokens=160,
    )
    assert first.validate(decision, upstream_full_allowed=False) == "upstream_full_not_allowed"

    assert first.latest_decision is decision
    assert first.decision_count == 1
    assert first.closed_reason_counts == {"upstream_full_not_allowed": 1}
    assert second.latest_decision is None
    assert second.decision_count == 0
    assert second.closed_reason_counts == {}
