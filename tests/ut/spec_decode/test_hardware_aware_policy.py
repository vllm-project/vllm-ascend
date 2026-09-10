# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration and CPU policy tests for hardware-aware decoding."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from vllm_ascend.dynamic_spec import (
    AdaptiveDraftKController,
    ProposalGate,
    resolve_method_params,
    v2_physical_k_enabled,
    validate_v1_dynamic_policy,
)


@pytest.mark.parametrize("method", ["dspark", "dflash", "eagle", "mtp"])
@pytest.mark.parametrize("dynamic_method", [None, "dspark", "dflash"])
@pytest.mark.parametrize("policy", ["confidence_budget", "hardware_aware"])
def test_shared_v1_guard_matches_child_constructor_paths(method, dynamic_method, policy):
    config = SimpleNamespace(method=dynamic_method, policy=policy)
    constructs_scheduler = (method == "dspark" and dynamic_method in ("dspark", "dflash")) or (
        method == "dflash" and dynamic_method == "dflash"
    )
    if constructs_scheduler and policy != "confidence_budget":
        with pytest.raises(ValueError, match="legacy V1 scheduler"):
            validate_v1_dynamic_policy(method, config)
    else:
        validate_v1_dynamic_policy(method, config)


def compact(**physical):
    return {"method": "dspark", "policy": "hardware_aware", "physical_k": physical}


def test_current_compact_config_matches_expanded_experiment():
    value = compact(min_k=3, capture_k=[3, 5])
    before = deepcopy(value)
    assert resolve_method_params(value) == {
        "adaptive_draft_k": True,
        "adaptive_draft_k_min": 3,
        "adaptive_draft_k_slack": 0,
        "adaptive_draft_k_percentile": 0.5,
        "v2_varlen_physical_k": True,
        "v2_varlen_capture_k": [3, 5],
        "hybrid_policy_enabled": True,
        "hybrid_min_batch_size": 8,
        "hybrid_acceptance_threshold": 0.6,
        "hybrid_low_steps": 4,
        "hybrid_high_steps": 2,
        "hybrid_probe_interval": 32,
    }
    assert value == before


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"adaptive_draft_k": True},
        {"v2_varlen_physical_k": False, "adaptive_draft_k_slack": 1},
        {"adaptive_draft_k_v2": True, "hybrid_policy_enabled": False},
    ],
)
def test_legacy_parameters_are_not_defaulted_or_mutated(params):
    assert resolve_method_params({"method_params": params}) == params
    assert resolve_method_params({"physical_k": None, "method_params": params}) == params


def test_new_defaults_and_explicit_disable():
    resolved = resolve_method_params(compact())
    assert resolved["adaptive_draft_k_min"] == 1
    assert resolved["adaptive_draft_k_slack"] == 0
    assert resolved["hybrid_policy_enabled"] is True
    assert "v2_varlen_capture_k" not in resolved
    assert v2_physical_k_enabled(compact()) is True
    assert v2_physical_k_enabled(compact(enabled=False)) is False
    assert resolve_method_params(compact(enabled=False))["adaptive_draft_k"] is False
    assert v2_physical_k_enabled({}) is False


def test_advanced_overrides():
    value = compact(slack=2, percentile=0.2, hybrid={"enabled": False, "probe_interval": 0})
    params = resolve_method_params(value)
    assert params["adaptive_draft_k_slack"] == 2
    assert params["adaptive_draft_k_percentile"] == 0.2
    assert params["hybrid_policy_enabled"] is False
    assert params["hybrid_probe_interval"] == 0


@pytest.mark.parametrize(
    "physical",
    [
        {"enabled": "false"},
        {"min_k": 0},
        {"min_k": True},
        {"min_k": 1.5},
        {"slack": -1},
        {"percentile": 1.1},
        {"percentile": float("nan")},
        {"capture_k": []},
        {"capture_k": [True]},
        {"capture_k": [0, 3]},
        {"capture_k": "3,5"},
        {"minimum_k": 3},
        {"hybrid": False},
        {"hybrid": {"low_steps": 0}},
        {"hybrid": {"probe_interval": -1}},
        {"hybrid": {"acceptance_threshold": float("inf")}},
        {"hybrid": {"enabled": "false"}},
        {"hybrid": {"unknown": 1}},
    ],
)
def test_new_config_rejects_invalid_values(physical):
    with pytest.raises(ValueError):
        resolve_method_params(compact(**physical))


@pytest.mark.parametrize(
    "key",
    [
        "adaptive_draft_k",
        "v2_varlen_physical_k",
        "adaptive_draft_k_v2",
        "adaptive_draft_k_min",
        "adaptive_draft_k_slack",
        "adaptive_draft_k_percentile",
        "v2_varlen_capture_k",
        "hybrid_policy_enabled",
        "hybrid_min_batch_size",
        "hybrid_acceptance_threshold",
        "hybrid_low_steps",
        "hybrid_high_steps",
        "hybrid_probe_interval",
    ],
)
def test_new_and_legacy_physical_settings_cannot_mix(key):
    value = compact()
    value["method_params"] = {key: 1}
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_method_params(value)


def test_unrelated_legacy_setting_is_preserved():
    value = compact()
    value["method_params"] = {"reuse_upstream_adaptive_verification": True}
    assert resolve_method_params(value)["reuse_upstream_adaptive_verification"] is True


@pytest.mark.parametrize(
    "policy,method", [("confidence_budget", "dspark"), ("hardware_aware", None), ("hardware_aware", "eagle")]
)
def test_compact_interface_requires_supported_policy_and_method(policy, method):
    with pytest.raises(ValueError, match="requires"):
        resolve_method_params({"physical_k": {}, "method": method, "policy": policy})


def test_dflash_supported():
    value = compact(min_k=2)
    value["method"] = "dflash"
    assert resolve_method_params(value)["adaptive_draft_k_min"] == 2


def test_adaptive_draft_k_hybrid_keeps_small_batch_at_full_width() -> None:
    controller = AdaptiveDraftKController(
        max_k=5,
        min_k=4,
        hybrid_enabled=True,
        hybrid_min_batch_size=8,
    )

    controller.observe([5] * 4, [[0]] * 4)

    assert controller.current_k == 5
    assert controller.last_reason == "small_batch_full_k"


def test_adaptive_draft_k_hybrid_uses_hysteresis_and_probe() -> None:
    controller = AdaptiveDraftKController(
        max_k=5,
        min_k=4,
        slack=0,
        hybrid_enabled=True,
        hybrid_min_batch_size=8,
        hybrid_acceptance_threshold=0.6,
        hybrid_low_steps=2,
        hybrid_probe_interval=3,
    )
    sampled = [[0]] * 8

    controller.observe([5] * 8, sampled)
    assert controller.current_k is None
    assert controller.last_reason == "low_acceptance_hysteresis"

    controller.observe([5] * 8, sampled)
    assert controller.current_k == 4
    assert controller.last_reason == "low_acceptance_dynamic_k"

    controller.observe([4] * 8, sampled)
    assert controller.current_k == 5
    assert controller.last_reason == "periodic_full_k_probe"


def test_proposal_gate_enters_latency_profile_after_low_load_streak() -> None:
    gate = ProposalGate(
        max_num_seqs=8,
        enter_ratio=0.5,
        enter_steps=2,
        exit_steps=1,
    )

    assert (
        gate.select_k(
            4,
            num_running=1,
            num_waiting=0,
            total_num_scheduled_tokens=1,
            num_scheduled_requests=1,
            prefill_scheduled=False,
        )
        == 0
    )
    assert (
        gate.select_k(
            4,
            num_running=1,
            num_waiting=0,
            total_num_scheduled_tokens=1,
            num_scheduled_requests=1,
            prefill_scheduled=False,
        )
        == 4
    )


def test_proposal_gate_exits_immediately_when_queue_builds() -> None:
    gate = ProposalGate(max_num_seqs=4, enter_steps=1, exit_steps=1)
    assert (
        gate.select_k(
            2,
            num_running=1,
            num_waiting=0,
            total_num_scheduled_tokens=1,
            num_scheduled_requests=1,
            prefill_scheduled=False,
        )
        == 2
    )
    assert (
        gate.select_k(
            2,
            num_running=2,
            num_waiting=1,
            total_num_scheduled_tokens=2,
            num_scheduled_requests=2,
            prefill_scheduled=False,
        )
        == 0
    )


def test_adaptive_draft_k_tracks_actual_accepted_width() -> None:
    controller = AdaptiveDraftKController(max_k=5, min_k=1, slack=1)

    # The first step keeps the configured width; the result feeds the next
    # scheduler step and removes one unused draft position.
    assert controller.cap(5) == 5
    controller.observe([5, 5], [[1, 2, 3, 4], [1, 2, 3]])
    assert controller.current_k == 4
    assert controller.last_accepted_lengths == [3, 2]
    assert controller.cap(5) == 4

    # A prefix that reaches the physical width allows gradual growth again.
    controller.observe([4, 4], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    assert controller.current_k == 5
    assert controller.cap(5) == 5


def test_adaptive_draft_k_preserves_gate_zero_and_minimum() -> None:
    controller = AdaptiveDraftKController(max_k=5, min_k=1, slack=1)
    assert controller.cap(5) == 5
    controller.update([0, 0])
    assert controller.current_k == 1
    assert controller.cap(0) == 0
    # A temporary batch-level gate must not permanently disable speculation.
    assert controller.cap(5) == 1


def test_removed_v2_legacy_manager_configuration_is_rejected() -> None:
    from vllm_ascend.ascend_config import DynamicSpecConfig

    with pytest.raises(ValueError, match="reuse_upstream_adaptive_verification"):
        DynamicSpecConfig(
            method="dspark",
            policy="hardware_aware",
            method_params={"reuse_upstream_adaptive_verification": False},
        )


def test_upstream_manager_configuration_remains_compatible() -> None:
    from vllm_ascend.ascend_config import DynamicSpecConfig

    config = DynamicSpecConfig(
        method="dspark",
        policy="hardware_aware",
        method_params={"reuse_upstream_adaptive_verification": True},
    )
    assert config.policy == "hardware_aware"


def test_dynamic_spec_config_compact_physical_k():
    from vllm_ascend.ascend_config import DynamicSpecConfig

    cfg = DynamicSpecConfig(method="dspark", policy="hardware_aware", physical_k={"min_k": 3, "capture_k": [3, 5]})
    assert cfg.physical_k["min_k"] == 3
    assert cfg.method_params == {}


def test_dynamic_spec_config_rejects_compact_typos_and_conflicts():
    from vllm_ascend.ascend_config import DynamicSpecConfig

    with pytest.raises(ValueError):
        DynamicSpecConfig(method="dspark", policy="hardware_aware", physical_k={"minimum_k": 3})
    with pytest.raises(ValueError):
        DynamicSpecConfig(
            method="dspark", policy="hardware_aware", physical_k={}, method_params={"adaptive_draft_k": True}
        )
