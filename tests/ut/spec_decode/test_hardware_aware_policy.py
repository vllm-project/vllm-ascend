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
        "hybrid_low_steps": 4,
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


def test_controller_defaults_to_full_k_until_worker_recommends():
    controller = AdaptiveDraftKController(max_k=5, min_k=3)
    assert controller.cap(5, 8) == 5
    controller.recommend(8, 3)
    assert controller.cap(5, 8) == 3
    assert controller.last_reason == "av_profile_recommendation"


def test_recommendations_are_batch_bucket_specific():
    controller = AdaptiveDraftKController(max_k=5, min_k=3, hybrid_min_batch_size=1)
    controller.recommend(8, 3)
    assert controller.cap(5, 8) == 3
    assert controller.cap(5, 16) == 5


def test_small_batch_keeps_full_k():
    controller = AdaptiveDraftKController(max_k=5, min_k=3, hybrid_min_batch_size=8)
    controller.recommend(4, 3)
    assert controller.cap(5, 4) == 5
    controller.observe([5] * 4, [[0]] * 4, use_acceptance_fallback=False)
    assert controller.observation_count == 0


def test_periodic_probe_forces_full_k():
    controller = AdaptiveDraftKController(
        max_k=5,
        min_k=3,
        hybrid_min_batch_size=1,
        hybrid_probe_interval=2,
    )
    controller.recommend(8, 3)
    controller.observation_count = 2
    assert controller.cap(5, 8) == 5
    assert controller.last_reason == "periodic_full_k_probe"


def test_missing_profile_uses_acceptance_fallback():
    controller = AdaptiveDraftKController(
        max_k=5,
        min_k=4,
        hybrid_min_batch_size=1,
        hybrid_low_steps=1,
    )
    controller.cap(5, 8)
    controller.observe([5] * 8, [[0]] * 8, use_acceptance_fallback=True)
    assert controller.current_k == 4
    assert controller.last_reason == "low_acceptance_dynamic_k"


def test_output_recommendation_is_applied():
    controller = AdaptiveDraftKController(max_k=5, min_k=3, hybrid_min_batch_size=1)
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
