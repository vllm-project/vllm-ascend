#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from types import SimpleNamespace

import pytest

from vllm_ascend.sequence_parallel import (
    DENSE_SP_MIN_TOKENS,
    MOE_SP_MIN_TOKENS,
    SequenceParallelActivationState,
    SequenceParallelCollective,
    SequenceParallelRuntimeState,
    plan_local_sequence_shard,
    plan_partial_reduction,
    plan_sequence_gather,
    resolve_sequence_parallel_policy,
)


def _make_config(
    *,
    enable_sp: bool = False,
    enforce_eager: bool = False,
    is_moe: bool = False,
    min_tokens: int | None = None,
    tp_size: int = 2,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=enforce_eager,
            is_moe=is_moe,
        ),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(
                enable_sp=enable_sp,
                sp_min_token_num=min_tokens,
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
    )


@pytest.mark.parametrize(
    ("legacy_enabled", "pass_enabled", "expected_enabled"),
    [
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_resolve_enable_sources(
    legacy_enabled: bool,
    pass_enabled: bool,
    expected_enabled: bool,
):
    policy = resolve_sequence_parallel_policy(
        _make_config(enable_sp=pass_enabled),
        legacy_flashcomm_enabled=legacy_enabled,
    )

    assert policy.enabled is expected_enabled
    assert policy.legacy_flashcomm_enabled is legacy_enabled
    assert policy.compile_pass_enabled is pass_enabled


def test_compile_pass_is_disabled_in_eager_mode():
    policy = resolve_sequence_parallel_policy(
        _make_config(enable_sp=True, enforce_eager=True),
        legacy_flashcomm_enabled=False,
    )

    assert not policy.configured
    assert not policy.enabled


def test_legacy_flashcomm_remains_available_in_eager_mode():
    policy = resolve_sequence_parallel_policy(
        _make_config(enable_sp=True, enforce_eager=True),
        legacy_flashcomm_enabled=True,
    )

    assert policy.configured
    assert policy.enabled
    assert not policy.compile_pass_enabled


def test_tp_one_disables_runtime_policy():
    policy = resolve_sequence_parallel_policy(
        _make_config(enable_sp=True, tp_size=1),
        legacy_flashcomm_enabled=False,
    )

    assert policy.configured
    assert not policy.enabled


@pytest.mark.parametrize(
    ("is_moe", "expected_min_tokens"),
    [
        (False, DENSE_SP_MIN_TOKENS),
        (True, MOE_SP_MIN_TOKENS),
    ],
)
def test_default_threshold_depends_on_model_type(
    is_moe: bool,
    expected_min_tokens: int,
):
    policy = resolve_sequence_parallel_policy(
        _make_config(enable_sp=True, is_moe=is_moe),
        legacy_flashcomm_enabled=False,
    )

    assert policy.min_tokens == expected_min_tokens


def test_user_threshold_takes_precedence():
    policy = resolve_sequence_parallel_policy(
        _make_config(enable_sp=True, is_moe=True, min_tokens=64),
        legacy_flashcomm_enabled=False,
    )

    assert policy.min_tokens == 64
    assert not policy.should_shard(63)
    assert policy.should_shard(64)


def test_backend_decisions_remain_source_specific():
    legacy_policy = resolve_sequence_parallel_policy(
        _make_config(enable_sp=False, min_tokens=64),
        legacy_flashcomm_enabled=True,
    )
    pass_policy = resolve_sequence_parallel_policy(
        _make_config(enable_sp=True, min_tokens=64),
        legacy_flashcomm_enabled=False,
    )

    assert legacy_policy.should_use_legacy_backend(64)
    assert not legacy_policy.should_use_compile_pass(64)
    assert pass_policy.should_use_compile_pass(64)
    assert not pass_policy.should_use_legacy_backend(64)


def test_zero_token_forward_is_not_sharded():
    policy = resolve_sequence_parallel_policy(
        _make_config(enable_sp=True, min_tokens=0),
        legacy_flashcomm_enabled=False,
    )

    assert not policy.should_shard(0)
    assert policy.should_shard(1)


@pytest.mark.parametrize("min_tokens", [-1, True, 1.5])
def test_invalid_threshold_is_rejected(min_tokens):
    config = _make_config(enable_sp=True, min_tokens=min_tokens)

    with pytest.raises((TypeError, ValueError)):
        resolve_sequence_parallel_policy(
            config,
            legacy_flashcomm_enabled=False,
        )


def test_negative_num_tokens_is_rejected():
    policy = resolve_sequence_parallel_policy(
        _make_config(enable_sp=True),
        legacy_flashcomm_enabled=False,
    )

    with pytest.raises(ValueError, match="num_tokens"):
        policy.should_shard(-1)


def test_runtime_state_calculates_local_geometry():
    state = SequenceParallelRuntimeState.create(
        active=True,
        world_size=4,
        num_tokens=10,
    )

    assert state.padded_num_tokens == 12
    assert state.local_num_tokens == 3
    assert state.pad_size == 2
    assert state.activation is SequenceParallelActivationState.FULL


def test_runtime_state_uses_dp_maximum_for_padding():
    state = SequenceParallelRuntimeState.create(
        active=True,
        world_size=4,
        num_tokens=7,
        max_num_tokens=10,
    )

    assert state.padded_num_tokens == 12
    assert state.local_num_tokens == 3
    assert state.pad_size == 5


def test_inactive_runtime_state_does_not_pad():
    state = SequenceParallelRuntimeState.create(
        active=False,
        world_size=4,
        num_tokens=10,
        max_num_tokens=15,
    )

    assert state.padded_num_tokens == 10
    assert state.local_num_tokens == 10
    assert state.pad_size == 0


def test_runtime_state_tracks_explicit_transitions():
    state = SequenceParallelRuntimeState.create(
        active=True,
        world_size=2,
        num_tokens=8,
    )

    partial = state.transition_to(SequenceParallelActivationState.TP_PARTIAL)
    sharded = partial.transition_to(SequenceParallelActivationState.SEQUENCE_SHARDED)
    full = sharded.transition_to(SequenceParallelActivationState.FULL)

    assert state.activation is SequenceParallelActivationState.FULL
    assert partial.activation is SequenceParallelActivationState.TP_PARTIAL
    assert sharded.activation is SequenceParallelActivationState.SEQUENCE_SHARDED
    assert full.activation is SequenceParallelActivationState.FULL


def test_runtime_state_rejects_implicit_partial_transition():
    state = SequenceParallelRuntimeState.create(
        active=True,
        world_size=2,
        num_tokens=8,
    ).transition_to(SequenceParallelActivationState.SEQUENCE_SHARDED)

    with pytest.raises(ValueError, match="sequence_sharded -> tp_partial"):
        state.transition_to(SequenceParallelActivationState.TP_PARTIAL)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"active": True, "world_size": 1, "num_tokens": 8},
        {"active": True, "world_size": 0, "num_tokens": 8},
        {"active": True, "world_size": 2, "num_tokens": -1},
        {
            "active": True,
            "world_size": 2,
            "num_tokens": 8,
            "max_num_tokens": 7,
        },
    ],
)
def test_runtime_state_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        SequenceParallelRuntimeState.create(**kwargs)


def test_inactive_runtime_state_rejects_sharding_transition():
    state = SequenceParallelRuntimeState.create(
        active=False,
        world_size=2,
        num_tokens=8,
    )

    with pytest.raises(ValueError, match="inactive"):
        state.transition_to(SequenceParallelActivationState.SEQUENCE_SHARDED)


def test_inactive_runtime_state_tracks_plain_tp_partial_sum():
    state = SequenceParallelRuntimeState.create(
        active=False,
        world_size=2,
        num_tokens=8,
    )

    partial = state.transition_to(SequenceParallelActivationState.TP_PARTIAL)
    full = partial.transition_to(SequenceParallelActivationState.FULL)

    assert partial.activation is SequenceParallelActivationState.TP_PARTIAL
    assert full.activation is SequenceParallelActivationState.FULL


def test_partial_reduction_selects_reduce_scatter_for_sp():
    partial = SequenceParallelRuntimeState.create(
        active=True,
        world_size=4,
        num_tokens=10,
    ).transition_to(SequenceParallelActivationState.TP_PARTIAL)

    plan = plan_partial_reduction(partial)

    assert plan.collective is SequenceParallelCollective.REDUCE_SCATTER
    assert plan.pad_size == 2
    assert plan.output_state.activation is SequenceParallelActivationState.SEQUENCE_SHARDED


def test_partial_reduction_selects_all_reduce_for_plain_tp():
    partial = SequenceParallelRuntimeState.create(
        active=False,
        world_size=4,
        num_tokens=10,
    ).transition_to(SequenceParallelActivationState.TP_PARTIAL)

    plan = plan_partial_reduction(partial)

    assert plan.collective is SequenceParallelCollective.ALL_REDUCE
    assert plan.pad_size == 0
    assert plan.output_state.activation is SequenceParallelActivationState.FULL


def test_sequence_gather_plans_unpadding():
    sharded = SequenceParallelRuntimeState.create(
        active=True,
        world_size=4,
        num_tokens=10,
    ).transition_to(SequenceParallelActivationState.SEQUENCE_SHARDED)

    plan = plan_sequence_gather(sharded)

    assert plan.collective is SequenceParallelCollective.ALL_GATHER
    assert plan.unpad_size == 2
    assert plan.output_state.activation is SequenceParallelActivationState.FULL


def test_local_sequence_shard_plans_padding():
    state = SequenceParallelRuntimeState.create(
        active=True,
        world_size=4,
        num_tokens=10,
    )

    plan = plan_local_sequence_shard(state)

    assert plan.collective is SequenceParallelCollective.LOCAL_SHARD
    assert plan.pad_size == 2
    assert plan.output_state.activation is SequenceParallelActivationState.SEQUENCE_SHARDED


@pytest.mark.parametrize(
    ("planner", "activation"),
    [
        (plan_partial_reduction, SequenceParallelActivationState.FULL),
        (plan_sequence_gather, SequenceParallelActivationState.FULL),
        (plan_local_sequence_shard, SequenceParallelActivationState.TP_PARTIAL),
    ],
)
def test_transition_planners_reject_wrong_input_state(planner, activation):
    state = SequenceParallelRuntimeState.create(
        active=True,
        world_size=2,
        num_tokens=8,
    ).transition_to(activation)

    with pytest.raises(ValueError):
        planner(state)
