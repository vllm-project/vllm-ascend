# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.dynamic_spec import (
    AdaptiveDraftKController,
    _create_controller,
    _update_controller,
    resolve_physical_k,
)


def compact(**physical):
    return {"method": "dspark", "physical_k": physical}


def test_compact_defaults_and_override():
    assert resolve_physical_k(compact(min_k=4)) == {
        "min_k": 4,
        "auto_tune": True,
    }
    assert resolve_physical_k(compact()) is not None
    assert resolve_physical_k({"method": "dspark"}) is None


@pytest.mark.parametrize(
    "physical",
    [
        {"min_k": 0},
        {"capture_k": []},
        {"hybrid": False},
        {"auto_tune": {"warmup_steps": 1}},
        {"unknown": 1},
    ],
)
def test_compact_rejects_removed_or_invalid_options(physical):
    with pytest.raises(ValueError):
        resolve_physical_k(compact(**physical))


def test_controller_debounces_worker_downshift():
    controller = AdaptiveDraftKController(max_k=5, min_k=3)
    assert controller.cap(5, 16) == 5
    controller.recommend(16, 3)
    for _ in range(2):
        controller.observe([5] * 16, [[0]] * 16)
        assert controller.cap(5, 16) == 5
    controller.observe([5] * 16, [[0]] * 16)
    assert controller.cap(5, 16) == 3


def test_recommendations_are_batch_bucket_specific():
    controller = AdaptiveDraftKController(max_k=5, min_k=3)
    controller.recommend(16, 3)
    for _ in range(3):
        controller.observe([5] * 16, [[0]] * 16)
    assert controller.cap(5, 16) == 3
    # A new bucket must persist for two steps before its independent state is
    # used; one transient batch-size change keeps the previous stable K.
    assert controller.cap(5, 32) == 3
    assert controller.cap(5, 32) == 5


def test_small_batch_uses_its_own_profile_recommendation():
    controller = AdaptiveDraftKController(max_k=5, min_k=3)
    controller.recommend(8, 3)
    assert controller.cap(5, 8) == 5
    for _ in range(3):
        controller.observe([5] * 8, [[0]] * 8)
    assert controller._state(8).observations == 3
    assert controller.cap(5, 8) == 3


def test_profiled_auto_tune_observation_skips_sampled_token_iteration():
    class NoIterationSequence(list):
        def __iter__(self):
            raise AssertionError("profiled auto-tune must not reprocess accepted tokens")

    controller = AdaptiveDraftKController(max_k=5, min_k=3)
    controller.recommend(8, 4)
    controller.observe([5] * 8, NoIterationSequence([[1, 2]] * 8))
    assert controller._state(8).observations == 1


def test_periodic_probe_forces_full_k():
    controller = AdaptiveDraftKController(max_k=5, min_k=3)
    controller.recommend(16, 3)
    state = controller._state(16)
    state.observations = 32
    assert controller.cap(5, 16) == 5
    assert state.stable_k == 5


def test_missing_profile_uses_acceptance_fallback():
    controller = AdaptiveDraftKController(max_k=5, min_k=4)
    controller.cap(5, 16)
    for _ in range(3):
        controller.observe([5] * 16, [[0]] * 16)
    assert controller.cap(5, 16) == 4


def test_output_recommendation_drives_controller():
    controller = AdaptiveDraftKController(max_k=5, min_k=3)
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={str(i): [1] * 5 for i in range(16)}
    )
    model_output = SimpleNamespace(
        physical_k_recommendation=(16, 3),
        req_ids=[str(i) for i in range(16)],
        sampled_token_ids=[[1, 2] for _ in range(16)],
    )
    for _ in range(3):
        _update_controller(controller, scheduler_output, model_output)
    assert controller.cap(5, 16) == 3


def test_full_k_profile_is_a_runtime_noop():
    controller = AdaptiveDraftKController(max_k=5, min_k=3)
    controller.recommend(8, 5)

    def unexpected_bucket_settle(batch_size):
        raise AssertionError("a full-K profile must not enter dynamic bucket logic")

    controller._settled_bucket = unexpected_bucket_settle
    assert controller.cap(5, 8) == 5


def test_worker_recommendation_is_not_applied_to_a_different_batch_bucket():
    controller = AdaptiveDraftKController(max_k=5, min_k=3)
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={str(i): [1] * 5 for i in range(8)}
    )
    model_output = SimpleNamespace(
        physical_k_recommendation=(1, 3),
        req_ids=[str(i) for i in range(8)],
        sampled_token_ids=[[1, 2] for _ in range(8)],
    )
    _update_controller(controller, scheduler_output, model_output)
    assert controller._state(8).profile_k == 5
    assert controller._state(1).profile_k is None


def test_small_batch_updates_controller_observation():
    class CountingController(AdaptiveDraftKController):
        observed = False

        def observe(self, scheduled_widths, sampled_token_ids):
            self.observed = True
            super().observe(scheduled_widths, sampled_token_ids)

    controller = CountingController(max_k=5, min_k=3)
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={str(i): [1] * 5 for i in range(8)}
    )
    model_output = SimpleNamespace(
        physical_k_recommendation=None,
        req_ids=[str(i) for i in range(8)],
        sampled_token_ids=[[1, 2] for _ in range(8)],
    )
    _update_controller(controller, scheduler_output, model_output)
    assert controller.observed
    assert controller._state(8).observations == 1


def test_profile_recommendation_is_authoritative_over_acceptance():
    controller = AdaptiveDraftKController(max_k=5, min_k=3)
    controller.recommend(16, 5)
    sampled = [[0] * 5 for _ in range(10)] + [[0] * 4 for _ in range(6)]
    for _ in range(3):
        controller.observe([5] * 16, sampled)
    state = controller._state(16)
    assert state.empirical_k == 5
    assert controller.cap(5, 16) == 5


def test_rollout_batch_decay_uses_independent_state_after_two_steps():
    controller = AdaptiveDraftKController(max_k=5, min_k=3)
    controller.recommend(128, 3)
    for _ in range(3):
        controller.observe([5] * 128, [[0]] * 128)
    assert controller.cap(5, 128) == 3

    controller.recommend(64, 4)
    for _ in range(3):
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
    assert _create_controller(config).auto_tune
    config.additional_config["dynamic_spec_config"] = compact(min_k=4, auto_tune=False)
    assert not _create_controller(config).auto_tune


def test_dynamic_spec_config_validates_compact_interface():
    from vllm_ascend.ascend_config import DynamicSpecConfig

    config = DynamicSpecConfig(
        method="dspark",
        physical_k={"min_k": 3},
    )
    assert config.physical_k["min_k"] == 3
