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

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

import vllm_ascend._310p.model_runner_310p as model_runner_310p
from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class _RejectInt64Add(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add_.Tensor,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.sub_.Tensor,
        ) and any(isinstance(arg, torch.Tensor) and arg.dtype == torch.int64 for arg in args):
            raise AssertionError("310P FULL metadata must not launch int64 Add")
        return func(*args, **kwargs)


def test_async_positions_use_persistent_int32_staging_before_int64_copy():
    copy_positions = getattr(
        model_runner_310p,
        "_copy_positions_via_int32_staging_310",
        None,
    )
    assert callable(copy_positions), "310P async position staging is missing"

    num_computed_tokens = torch.tensor([91, 102], dtype=torch.int32)
    req_indices = torch.tensor([0] * 16 + [1] * 16, dtype=torch.long)
    query_pos = torch.tensor(list(range(16)) * 2, dtype=torch.int64)
    positions = torch.empty(32, dtype=torch.int64)
    base_i32 = torch.empty(32, dtype=torch.int32)
    query_i32 = torch.empty(32, dtype=torch.int32)
    positions_i32 = torch.empty(32, dtype=torch.int32)

    with _RejectInt64Add():
        copy_positions(
            positions,
            base_i32,
            query_i32,
            positions_i32,
            num_computed_tokens,
            req_indices,
            query_pos,
        )

    assert positions.tolist() == list(range(91, 107)) + list(range(102, 118))


def test_async_rope_drift_uses_persistent_int32_staging():
    apply_drift = getattr(
        model_runner_310p,
        "_apply_position_drift_via_int32_staging_310",
        None,
    )
    assert callable(apply_drift), "310P async RoPE drift staging is missing"

    target = torch.tensor(
        [
            list(range(75, 91)) + list(range(102, 118)),
            list(range(175, 191)) + list(range(202, 218)),
            list(range(275, 291)) + list(range(302, 318)),
        ],
        dtype=torch.int64,
    )
    num_computed_tokens = torch.tensor([83, 110], dtype=torch.int32)
    cpu_values = torch.tensor([91, 102], dtype=torch.int32)
    req_indices = torch.tensor([0] * 16 + [1] * 16, dtype=torch.long)
    base_i32 = torch.empty(32, dtype=torch.int32)
    cpu_i32 = torch.empty(32, dtype=torch.int32)
    drift_i32 = torch.empty(32, dtype=torch.int32)
    rope_i32 = torch.empty((3, 32), dtype=torch.int32)
    result_i32 = torch.empty((3, 32), dtype=torch.int32)

    with _RejectInt64Add():
        apply_drift(
            target,
            rope_i32,
            result_i32,
            base_i32,
            cpu_i32,
            drift_i32,
            num_computed_tokens,
            cpu_values,
            req_indices,
        )

    expected_drift = torch.tensor([-8] * 16 + [8] * 16)
    expected = (
        torch.tensor(
            [
                list(range(75, 91)) + list(range(102, 118)),
                list(range(175, 191)) + list(range(202, 218)),
                list(range(275, 291)) + list(range(302, 318)),
            ],
            dtype=torch.int64,
        )
        + expected_drift
    )
    torch.testing.assert_close(target, expected)


def _load_policy_module():
    try:
        return importlib.import_module("vllm_ascend._310p.dflash_full_decode_only")
    except ModuleNotFoundError:
        pytest.fail("310P DFlash FULL_DECODE_ONLY policy module is missing")


def _config(method: str | None, mode: CUDAGraphMode):
    return SimpleNamespace(
        speculative_config=(SimpleNamespace(method=method, num_speculative_tokens=15) if method is not None else None),
        compilation_config=SimpleNamespace(
            cudagraph_mode=mode,
            cudagraph_capture_sizes=[64, 32, 16],
        ),
    )


@pytest.mark.parametrize(
    ("is_310p_platform", "method", "mode", "expected"),
    [
        (True, "dflash", CUDAGraphMode.FULL_DECODE_ONLY, True),
        (False, "dflash", CUDAGraphMode.FULL_DECODE_ONLY, False),
        (True, "mtp", CUDAGraphMode.FULL_DECODE_ONLY, False),
        (True, None, CUDAGraphMode.FULL_DECODE_ONLY, False),
        (True, "dflash", CUDAGraphMode.NONE, False),
        (True, "dflash", CUDAGraphMode.PIECEWISE, False),
        (True, "dflash", CUDAGraphMode.FULL, False),
        (True, "dflash", CUDAGraphMode.FULL_AND_PIECEWISE, False),
    ],
)
def test_full_decode_only_scope_requires_every_condition(
    is_310p_platform: bool,
    method: str | None,
    mode: CUDAGraphMode,
    expected: bool,
) -> None:
    policy = _load_policy_module()
    config = _config(method, mode)

    with patch.object(policy, "is_310p", return_value=is_310p_platform):
        active = policy.is_310p_dflash_full_decode_only(config)

    assert active is expected
    assert config.compilation_config.cudagraph_mode is mode
    assert config.compilation_config.cudagraph_capture_sizes == [64, 32, 16]


@pytest.mark.parametrize(
    (
        "is_310p_platform",
        "method",
        "configured_mode",
        "runtime_mode",
        "in_profile_run",
        "expected",
    ),
    [
        (True, "dflash", CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.NONE, False, True),
        (True, "dflash", CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.NONE, True, False),
        (True, "dflash", CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL, False, False),
        (False, "dflash", CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.NONE, False, False),
        (True, "mtp", CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.NONE, False, False),
        (True, "dflash", CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE, False, False),
        (True, "dflash", CUDAGraphMode.NONE, CUDAGraphMode.NONE, False, False),
    ],
)
def test_only_runtime_none_in_exact_fdo_scope_uses_uncompiled_execution(
    is_310p_platform: bool,
    method: str,
    configured_mode: CUDAGraphMode,
    runtime_mode: CUDAGraphMode,
    in_profile_run: bool,
    expected: bool,
) -> None:
    policy = _load_policy_module()
    config = _config(method, configured_mode)

    with patch.object(policy, "is_310p", return_value=is_310p_platform):
        actual = policy.should_skip_compiled_for_dflash_fdo_none(
            config,
            runtime_mode=runtime_mode,
            in_profile_run=in_profile_run,
        )

    assert actual is expected


@pytest.mark.parametrize(
    (
        "attn_state",
        "num_tokens",
        "num_reqs",
        "max_query_len",
        "all_decode",
        "forced_capture",
        "expected_state",
        "expected_mode",
    ),
    [
        (
            AscendAttentionState.PrefillNoCache,
            64,
            1,
            64,
            False,
            False,
            "EXPECTED_NONE_PREFILL",
            CUDAGraphMode.NONE,
        ),
        (
            AscendAttentionState.ChunkedPrefill,
            64,
            1,
            64,
            False,
            False,
            "EXPECTED_NONE_CHUNKED_PREFILL",
            CUDAGraphMode.NONE,
        ),
        (
            AscendAttentionState.PrefillCacheHit,
            16,
            1,
            16,
            True,
            False,
            "EXPECTED_NONE_PREFIX_TRANSITION",
            CUDAGraphMode.NONE,
        ),
        (
            AscendAttentionState.DecodeOnly,
            1,
            1,
            1,
            True,
            False,
            "EXPECTED_NONE_MIXED",
            CUDAGraphMode.NONE,
        ),
        (
            AscendAttentionState.SpecDecoding,
            16,
            1,
            16,
            True,
            False,
            "FULL_ELIGIBLE_UNIFORM_DECODE",
            CUDAGraphMode.FULL,
        ),
        (
            AscendAttentionState.SpecDecoding,
            32,
            2,
            15,
            True,
            False,
            "EXPECTED_NONE_MIXED",
            CUDAGraphMode.NONE,
        ),
        (
            AscendAttentionState.SpecDecoding,
            31,
            2,
            16,
            True,
            False,
            "EXPECTED_NONE_MIXED",
            CUDAGraphMode.NONE,
        ),
        (
            AscendAttentionState.SpecDecoding,
            16,
            1,
            16,
            False,
            False,
            "EXPECTED_NONE_MIXED",
            CUDAGraphMode.NONE,
        ),
        (
            None,
            64,
            4,
            16,
            False,
            True,
            "FULL_ELIGIBLE_UNIFORM_DECODE",
            CUDAGraphMode.FULL,
        ),
    ],
)
def test_full_decode_only_batch_classifier_is_closed(
    attn_state,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    all_decode: bool,
    forced_capture: bool,
    expected_state: str,
    expected_mode: CUDAGraphMode,
) -> None:
    policy = _load_policy_module()

    decision = policy.classify_dflash_full_decode_batch(
        attn_state=attn_state,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        max_num_scheduled_tokens=max_query_len,
        uniform_decode_query_len=16,
        all_decode=all_decode,
        forced_uniform_capture=forced_capture,
    )

    assert decision.state.name == expected_state
    assert decision.expected_runtime_mode is expected_mode


@pytest.mark.parametrize(
    ("num_tokens", "num_reqs", "expected_descriptor"),
    [
        (16, 1, 16),
        (32, 2, 32),
        (48, 3, 64),
        (64, 4, 64),
    ],
)
def test_full_decode_descriptor_maps_k15_tokens_not_request_count(
    num_tokens: int,
    num_reqs: int,
    expected_descriptor: int,
) -> None:
    policy = _load_policy_module()
    capture_sizes = [64, 32, 16]

    descriptor = policy.resolve_dflash_full_decode_descriptor(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        uniform_decode_query_len=16,
        capture_sizes=capture_sizes,
    )

    assert descriptor == expected_descriptor
    assert capture_sizes == [64, 32, 16]


def test_full_decode_slot_mapping_keeps_descriptor_view_after_batch_shrinks() -> None:
    policy = _load_policy_module()
    slot_mapping = torch.arange(128, dtype=torch.int32)
    config = _config("dflash", CUDAGraphMode.FULL_DECODE_ONLY)

    with patch.object(policy, "is_310p", return_value=True):
        selected = policy.select_dflash_full_decode_slot_mapping(
            vllm_config=config,
            attn_state=AscendAttentionState.SpecDecoding,
            slot_mapping=slot_mapping,
            num_actual_tokens=48,
            num_input_tokens=64,
        )

    assert selected.data_ptr() == slot_mapping.data_ptr()
    assert selected.shape == (64,)


def test_full_decode_slot_mapping_does_not_change_eager_or_other_scopes() -> None:
    policy = _load_policy_module()
    slot_mapping = torch.arange(128, dtype=torch.int32)

    for config, state, is_310p_platform in (
        (
            _config("dflash", CUDAGraphMode.FULL_DECODE_ONLY),
            AscendAttentionState.ChunkedPrefill,
            True,
        ),
        (
            _config("dflash", CUDAGraphMode.PIECEWISE),
            AscendAttentionState.SpecDecoding,
            True,
        ),
        (
            _config("dflash", CUDAGraphMode.FULL_DECODE_ONLY),
            AscendAttentionState.SpecDecoding,
            False,
        ),
    ):
        with patch.object(policy, "is_310p", return_value=is_310p_platform):
            selected = policy.select_dflash_full_decode_slot_mapping(
                vllm_config=config,
                attn_state=state,
                slot_mapping=slot_mapping,
                num_actual_tokens=48,
                num_input_tokens=64,
            )
        assert selected.shape == (48,)


@pytest.mark.parametrize(
    ("num_tokens", "num_reqs", "query_len", "capture_sizes"),
    [
        (15, 1, 16, [16, 32, 64]),
        (16, 1, 16, [20, 32, 64]),
        (64, 4, 16, [16, 32]),
        (16, 1, 0, [16, 32, 64]),
    ],
)
def test_full_decode_descriptor_rejects_unsafe_mapping(
    num_tokens: int,
    num_reqs: int,
    query_len: int,
    capture_sizes: list[int],
) -> None:
    policy = _load_policy_module()

    with pytest.raises(policy.DFlashFullDecodeDispatchError):
        policy.resolve_dflash_full_decode_descriptor(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            uniform_decode_query_len=query_len,
            capture_sizes=capture_sizes,
        )


def test_debug_validation_rejects_eligible_decode_eager_fallback() -> None:
    policy = _load_policy_module()
    decision = policy.classify_dflash_full_decode_batch(
        attn_state=AscendAttentionState.SpecDecoding,
        num_tokens=16,
        num_reqs=1,
        max_num_scheduled_tokens=16,
        uniform_decode_query_len=16,
        all_decode=True,
    )

    with pytest.raises(
        policy.DFlashFullDecodeDispatchError,
        match="eligible uniform decode selected NONE",
    ):
        policy.validate_dflash_full_decode_dispatch(
            decision=decision,
            runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=16),
            expected_descriptor=16,
            strict=True,
        )

    assert (
        policy.validate_dflash_full_decode_dispatch(
            decision=decision,
            runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=16),
            expected_descriptor=16,
            strict=False,
        )
        == "eligible_uniform_decode_selected_none"
    )


def test_validation_rejects_full_for_expected_none_even_in_info() -> None:
    policy = _load_policy_module()
    decision = policy.classify_dflash_full_decode_batch(
        attn_state=AscendAttentionState.PrefillNoCache,
        num_tokens=32,
        num_reqs=1,
        max_num_scheduled_tokens=32,
        uniform_decode_query_len=16,
        all_decode=False,
    )

    with pytest.raises(
        policy.DFlashFullDecodeDispatchError,
        match="expected NONE selected FULL",
    ):
        policy.validate_dflash_full_decode_dispatch(
            decision=decision,
            runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=BatchDescriptor(num_tokens=32),
            expected_descriptor=None,
            strict=False,
        )


@pytest.mark.parametrize(
    ("attn_state", "num_tokens", "num_reqs", "max_query_len", "parent_mode", "expected_force_eager"),
    [
        (
            AscendAttentionState.SpecDecoding,
            16,
            1,
            16,
            CUDAGraphMode.FULL,
            False,
        ),
        (
            AscendAttentionState.PrefillNoCache,
            32,
            1,
            32,
            CUDAGraphMode.NONE,
            True,
        ),
    ],
)
def test_runner_applies_full_decode_only_decision_without_mode_coercion(
    attn_state: AscendAttentionState,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    parent_mode: CUDAGraphMode,
    expected_force_eager: bool,
) -> None:
    policy = _load_policy_module()
    runner = object.__new__(NPUModelRunner310)
    runner.attn_state = attn_state
    runner._spec_dummy_capture = False
    runner.speculative_config = SimpleNamespace(
        method="dflash",
        num_speculative_tokens=15,
    )
    runner.uniform_decode_query_len = 16
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.ones(16, dtype=np.int32),
    )
    runner.vllm_config = _config("dflash", CUDAGraphMode.FULL_DECODE_ONLY)
    observed_force_eager: list[bool] = []

    def parent_determine(self, **kwargs):
        observed_force_eager.append(kwargs["force_eager"])
        descriptor = BatchDescriptor(
            num_tokens=num_tokens,
            num_reqs=num_reqs if parent_mode == CUDAGraphMode.FULL else None,
            uniform=parent_mode == CUDAGraphMode.FULL,
        )
        return parent_mode, descriptor, False, None, None

    with (
        patch.object(policy, "is_310p", return_value=True),
        patch.object(
            NPUModelRunner,
            "_determine_batch_execution_and_padding",
            new=parent_determine,
        ),
    ):
        result = runner._determine_batch_execution_and_padding(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=np.full((num_reqs,), max_query_len, dtype=np.int32),
            max_num_scheduled_tokens=max_query_len,
            use_cascade_attn=False,
        )

    assert result[0] is parent_mode
    assert observed_force_eager == [expected_force_eager]
    assert runner._dflash_full_decode_decision.expected_runtime_mode is parent_mode


def test_runner_forced_uniform_startup_capture_is_full_eligible() -> None:
    policy = _load_policy_module()
    runner = object.__new__(NPUModelRunner310)
    # Parent dummy capture can leave this stale pre-build state on 310P. The
    # explicit uniform capture contract must win for this startup-only call.
    runner.attn_state = AscendAttentionState.ChunkedPrefill
    runner._spec_dummy_capture = True
    runner._fdo_graph_capture_active = True
    runner.speculative_config = SimpleNamespace(
        method="dflash",
        num_speculative_tokens=15,
    )
    runner.uniform_decode_query_len = 16
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.zeros(16, dtype=np.int32),
    )
    runner.vllm_config = _config("dflash", CUDAGraphMode.FULL_DECODE_ONLY)

    def parent_determine(self, **kwargs):
        assert kwargs["force_eager"] is False
        assert kwargs["force_uniform_decode"] is True
        return (
            CUDAGraphMode.FULL,
            BatchDescriptor(num_tokens=64, num_reqs=4, uniform=True),
            False,
            None,
            None,
        )

    with (
        patch.object(policy, "is_310p", return_value=True),
        patch.object(
            NPUModelRunner,
            "_determine_batch_execution_and_padding",
            new=parent_determine,
        ),
    ):
        result = runner._determine_batch_execution_and_padding(
            num_tokens=64,
            num_reqs=4,
            num_scheduled_tokens_np=np.full((4,), 16, dtype=np.int32),
            max_num_scheduled_tokens=16,
            use_cascade_attn=False,
            force_uniform_decode=True,
        )

    assert result[0] is CUDAGraphMode.FULL
    assert runner._dflash_full_decode_decision.state.name == ("FULL_ELIGIBLE_UNIFORM_DECODE")


def test_runner_startup_warmup_remains_expected_none() -> None:
    policy = _load_policy_module()
    runner = object.__new__(NPUModelRunner310)
    runner.attn_state = AscendAttentionState.ChunkedPrefill
    runner._spec_dummy_capture = True
    runner._fdo_graph_capture_active = False
    runner.speculative_config = SimpleNamespace(
        method="dflash",
        num_speculative_tokens=15,
    )
    runner.uniform_decode_query_len = 16
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.zeros(16, dtype=np.int32),
    )
    runner.vllm_config = _config("dflash", CUDAGraphMode.FULL_DECODE_ONLY)

    def parent_determine(self, **kwargs):
        assert kwargs["force_eager"] is True
        return (
            CUDAGraphMode.NONE,
            BatchDescriptor(num_tokens=64),
            False,
            None,
            None,
        )

    with (
        patch.object(policy, "is_310p", return_value=True),
        patch.object(
            NPUModelRunner,
            "_determine_batch_execution_and_padding",
            new=parent_determine,
        ),
    ):
        result = runner._determine_batch_execution_and_padding(
            num_tokens=64,
            num_reqs=4,
            num_scheduled_tokens_np=np.full((4,), 16, dtype=np.int32),
            max_num_scheduled_tokens=16,
            use_cascade_attn=False,
            force_uniform_decode=True,
        )

    assert result[0] is CUDAGraphMode.NONE
    assert runner._dflash_full_decode_decision.state.name == ("EXPECTED_NONE_CHUNKED_PREFILL")
