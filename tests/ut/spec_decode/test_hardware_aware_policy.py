# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration and CPU policy tests for hardware-aware decoding."""

import pytest

from vllm_ascend.dynamic_spec import (
    AdaptiveDraftKController,
    resolve_physical_k,
    v2_physical_k_enabled,
)


def compact(**physical):
    return {"method": "dspark", "policy": "hardware_aware", "physical_k": physical}


def test_compact_config_defaults_and_overrides():
    params = resolve_physical_k(
        compact(
            min_k=3,
            capture_k=[5, 3, 5],
            hybrid={"enabled": False, "probe_interval": 0},
        )
    )
    assert params == {
        "enabled": True,
        "min_k": 3,
        "slack": 0,
        "percentile": 0.5,
        "capture_k": (3, 5),
        "hybrid_enabled": False,
        "hybrid_min_batch_size": 8,
        "hybrid_acceptance_threshold": 0.6,
        "hybrid_low_steps": 4,
        "hybrid_high_steps": 2,
        "hybrid_probe_interval": 0,
        "auto_tune_enabled": False,
        "auto_tune_warmup_steps": 64,
        "auto_tune_explore_ratio": 0.05,
        "auto_tune_update_interval": 32,
        "auto_tune_min_gain": 0.02,
        "auto_tune_ema_decay": 0.9,
    }
    assert v2_physical_k_enabled(compact())
    assert not v2_physical_k_enabled(compact(enabled=False))
    assert not v2_physical_k_enabled({})


@pytest.mark.parametrize(
    "physical",
    [
        {"enabled": "false"},
        {"min_k": 0},
        {"slack": -1},
        {"percentile": float("nan")},
        {"capture_k": []},
        {"capture_k": [True]},
        {"hybrid": False},
        {"hybrid": {"low_steps": 0}},
        {"hybrid": {"unknown": 1}},
        {"auto_tune": False},
        {"auto_tune": {"warmup_steps": 0}},
        {"auto_tune": {"unknown": 1}},
        {"unknown": 1},
    ],
)
def test_compact_config_rejects_invalid_values(physical):
    with pytest.raises(ValueError):
        resolve_physical_k(compact(**physical))


@pytest.mark.parametrize(
    "policy,method",
    [("confidence_budget", "dspark"), ("hardware_aware", None), ("hardware_aware", "eagle")],
)
def test_compact_interface_requires_supported_policy_and_method(policy, method):
    with pytest.raises(ValueError, match="requires"):
        resolve_physical_k({"physical_k": {}, "method": method, "policy": policy})


def test_dflash_is_supported():
    value = compact(min_k=2)
    value["method"] = "dflash"
    assert resolve_physical_k(value)["min_k"] == 2


def test_small_batch_keeps_full_width():
    controller = AdaptiveDraftKController(max_k=5, min_k=4, hybrid_min_batch_size=8)
    controller.observe([5] * 4, [[0]] * 4)
    assert controller.current_k == 5
    assert controller.last_reason == "small_batch_full_k"


def test_hybrid_uses_hysteresis_and_periodic_probe():
    controller = AdaptiveDraftKController(
        max_k=5,
        min_k=4,
        hybrid_min_batch_size=8,
        hybrid_acceptance_threshold=0.6,
        hybrid_low_steps=2,
        hybrid_probe_interval=3,
    )
    rejected = [[0]] * 8
    controller.observe([5] * 8, rejected)
    assert controller.current_k is None
    controller.observe([5] * 8, rejected)
    assert controller.current_k == 4
    controller.observe([4] * 8, rejected)
    assert controller.current_k == 5
    assert controller.last_reason == "periodic_full_k_probe"


def test_non_hybrid_tracks_accepted_width_and_recovers():
    controller = AdaptiveDraftKController(max_k=5, min_k=1, slack=1, hybrid_enabled=False)
    assert controller.cap(5) == 5
    controller.observe([5, 5], [[1, 2, 3, 4], [1, 2, 3]])
    assert controller.cap(5) == 4
    controller.observe([4, 4], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    assert controller.cap(5) == 5


def test_online_cost_model_explores_and_selects_best_k():
    controller = AdaptiveDraftKController(
        max_k=5,
        min_k=4,
        capture_k=(4, 5),
        auto_tune_enabled=True,
        auto_tune_warmup_steps=1,
        auto_tune_explore_ratio=0.0,
        auto_tune_update_interval=1,
        auto_tune_min_gain=0.0,
    )
    assert controller.cap(5) == 5
    # First observation measures K=5, then warmup exploration switches to K=4.
    controller.observe([5] * 8, [[0, 1]] * 8, elapsed_ms=10.0)
    assert controller.current_k == 4
    # K=4 produces more effective tokens for the same measured time.
    controller.observe([4] * 8, [[0, 1, 2, 3, 4]] * 8, elapsed_ms=10.0)
    assert controller.current_k == 4
    assert controller.last_reason == "auto_cost_model_best_k"


def test_dynamic_spec_config_validates_compact_interface():
    from vllm_ascend.ascend_config import DynamicSpecConfig

    config = DynamicSpecConfig(
        method="dspark",
        policy="hardware_aware",
        physical_k={"min_k": 3, "capture_k": [3, 5]},
    )
    assert config.physical_k["min_k"] == 3
    with pytest.raises(ValueError):
        DynamicSpecConfig(method="dspark", policy="hardware_aware", physical_k={"minimum_k": 3})
