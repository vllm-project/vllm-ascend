# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

import pytest

from vllm_ascend.dynamic_spec_config import resolve_method_params, v2_physical_k_enabled


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
