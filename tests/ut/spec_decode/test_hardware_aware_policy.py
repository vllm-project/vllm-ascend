# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.dynamic_spec import (
    AdaptiveDraftKController,
    _create_controller,
    _update_controller,
    resolve_physical_k,
    v2_physical_k_enabled,
)


def compact(**physical):
    return {"method": "dspark", "policy": "hardware_aware", "physical_k": physical}


def test_compact_defaults_and_override():
    params = resolve_physical_k(
        compact(min_k=3, capture_k=[5, 3, 5], hybrid={"enabled": False})
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
        "hybrid_low_steps": 3,
        "hybrid_high_steps": 2,
        "hybrid_probe_interval": 32,
        "auto_tune_enabled": True,
    }
    assert v2_physical_k_enabled(compact())
    assert not v2_physical_k_enabled(compact(enabled=False))


@pytest.mark.parametrize(
    "physical",
    [
        {"enabled": "false"},
        {"min_k": 0},
        {"capture_k": []},
        {"hybrid": False},
        {"hybrid": {"unknown": 1}},
        {"auto_tune": False},
        {"auto_tune": {"warmup_steps": 1}},
        {"unknown": 1},
    ],
)
def test_compact_rejects_removed_or_invalid_options(physical):
    with pytest.raises(ValueError):
        resolve_physical_k(compact(**physical))


def test_controller_debounces_worker_downshift():
    controller = AdaptiveDraftKController(max_k=5, min_k=3, hybrid_min_batch_size=1)
    assert controller.cap(5, 8) == 5
    controller.recommend(8, 3)
    for _ in range(2):
        controller.observe([5] * 8, [[0]] * 8)
        assert controller.cap(5, 8) == 5
    controller.observe([5] * 8, [[0]] * 8)
    assert controller.cap(5, 8) == 3
    assert controller.current_k == 3


def test_recommendations_are_batch_bucket_specific():
    controller = AdaptiveDraftKController(
        max_k=5, min_k=3, hybrid_min_batch_size=1, hybrid_low_steps=1
    )
    controller.recommend(8, 3)
    controller.observe([5] * 8, [[0]] * 8)
    assert controller.cap(5, 8) == 3
    # A new bucket must persist for two steps before its independent state is
    # used; one transient batch-size change keeps the previous stable K.
    assert controller.cap(5, 16) == 3
    assert controller.cap(5, 16) == 5


def test_small_batch_keeps_full_k():
    controller = AdaptiveDraftKController(max_k=5, min_k=3, hybrid_min_batch_size=8)
    controller.recommend(4, 3)
    assert controller.cap(5, 4) == 5
    controller.observe([5] * 4, [[0]] * 4)
    assert controller.observation_count == 0


def test_periodic_probe_forces_full_k():
    controller = AdaptiveDraftKController(
        max_k=5,
        min_k=3,
        hybrid_min_batch_size=1,
        hybrid_probe_interval=2,
    )
    controller.recommend(8, 3)
    state = controller._state(8)
    state.observations = 2
    assert controller.cap(5, 8) == 5
    assert controller.last_reason == "periodic_full_k_probe"
    assert state.stable_k == 5


def test_missing_profile_uses_acceptance_fallback():
    controller = AdaptiveDraftKController(
        max_k=5,
        min_k=4,
        hybrid_min_batch_size=1,
        hybrid_low_steps=1,
    )
    controller.cap(5, 8)
    controller.observe([5] * 8, [[0]] * 8)
    assert controller.current_k == 4
    assert controller.last_reason == "combined_downshift"


def test_output_recommendation_and_acceptance_are_combined():
    controller = AdaptiveDraftKController(
        max_k=5, min_k=3, hybrid_min_batch_size=1, hybrid_low_steps=1
    )
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={str(i): [1] * 5 for i in range(8)}
    )
    model_output = SimpleNamespace(
        physical_k_recommendation=(8, 3, 1.25),
        req_ids=[str(i) for i in range(8)],
        sampled_token_ids=[[1, 2] for _ in range(8)],
    )
    _update_controller(controller, scheduler_output, model_output)
    assert controller.cap(5, 8) == 3


def test_acceptance_survival_is_tracked_by_position():
    controller = AdaptiveDraftKController(
        max_k=5,
        min_k=3,
        hybrid_min_batch_size=1,
        hybrid_low_steps=1,
        hybrid_acceptance_threshold=0.6,
    )
    controller.recommend(8, 5)
    sampled = [[0] * 5 for _ in range(5)] + [[0] * 4 for _ in range(3)]
    controller.observe([5] * 8, sampled)
    state = controller._state(8)
    assert state.survival[:5] == [1.0, 1.0, 1.0, 0.625, 0.0]
    assert state.empirical_k == 4
    assert controller.cap(5, 8) == 4


def test_empirical_acceptance_cannot_cross_profile_cost_floor():
    controller = AdaptiveDraftKController(
        max_k=5,
        min_k=3,
        hybrid_min_batch_size=1,
        hybrid_low_steps=1,
    )
    controller.recommend(64, 4, cost_floor_k=4)
    controller.observe([5] * 64, [[0]] * 64)
    state = controller._state(64)
    assert state.empirical_k == 3
    assert state.stable_k == 4
    assert controller.cap(5, 64) == 4


def test_rollout_batch_decay_uses_independent_state_after_two_steps():
    controller = AdaptiveDraftKController(
        max_k=5, min_k=3, hybrid_min_batch_size=1, hybrid_low_steps=1
    )
    controller.recommend(128, 3)
    controller.observe([5] * 128, [[0]] * 128)
    assert controller.cap(5, 128) == 3

    controller.recommend(64, 4)
    controller.observe([5] * 64, [[0] * 5] * 64)
    assert controller.cap(5, 64) == 3
    assert controller.cap(5, 64) == 4


def test_default_candidates_start_at_k3():
    assert resolve_physical_k(compact())["min_k"] == 3


def test_create_controller_and_opt_out():
    config = SimpleNamespace(
        use_v2_model_runner=True,
        additional_config={"dynamic_spec_config": compact(min_k=4)},
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    assert _create_controller(config).auto_tune_enabled
    config.additional_config["dynamic_spec_config"] = compact(
        min_k=4, auto_tune={"enabled": False}
    )
    assert not _create_controller(config).auto_tune_enabled


def test_dynamic_spec_config_validates_compact_interface():
    from vllm_ascend.ascend_config import DynamicSpecConfig

    config = DynamicSpecConfig(
        method="dspark",
        policy="hardware_aware",
        physical_k={"min_k": 3},
    )
    assert config.physical_k["min_k"] == 3
