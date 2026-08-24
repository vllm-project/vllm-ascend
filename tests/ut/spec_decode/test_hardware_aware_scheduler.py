# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm_ascend.spec_decode.dynamic.calibration import SequentialTemperatureScaler
from vllm_ascend.spec_decode.dynamic.cost_model import (
    HardwareCostModel,
    HardwareProfileCollector,
)
from vllm_ascend.spec_decode.dynamic.draft_k_controller import AdaptiveDraftKController
from vllm_ascend.spec_decode.dynamic.policy import HardwareAwarePrefixPolicy
from vllm_ascend.spec_decode.dynamic.proposal_gate import ProposalGate


class _FakeDSparkModel:
    def __init__(self) -> None:
        self.confidence_calls = 0

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.float().unsqueeze(-1)

    def confidence_logits(
        self,
        hidden_states: torch.Tensor,
        markov_embs: torch.Tensor,
    ) -> torch.Tensor:
        self.confidence_calls += 1
        return (hidden_states[..., :1] + markov_embs).squeeze(-1)


def _policy(
    latency_ms: dict[int, float],
    *,
    min_k: int = 0,
    max_batch_size: int = 4,
    max_draft_tokens: int = 3,
    decision_interval: int = 16,
    allocation_interval: int = 1,
) -> HardwareAwarePrefixPolicy:
    model = HardwareCostModel(latency_ms=latency_ms, fingerprint={})
    return HardwareAwarePrefixPolicy(
        cost_model=model,
        min_k=min_k,
        max_batch_size=max_batch_size,
        max_draft_tokens=max_draft_tokens,
        device=torch.device("cpu"),
        decision_interval=decision_interval,
        allocation_interval=allocation_interval,
    )


def test_hardware_policy_can_choose_zero_tokens() -> None:
    # The target is most efficient at the bonus-token-only shape.  The
    # profile key includes one bonus-token row per request, so a two-request
    # batch starts at verification width 2.
    policy = _policy({2: 1.0, 3: 10.0, 4: 10.0, 5: 10.0, 6: 10.0, 7: 10.0, 8: 10.0})
    survival = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])

    lengths = policy.allocate(survival)

    assert lengths.tolist() == [0, 0]


def test_hardware_policy_allocates_prefixes_globally() -> None:
    # The first request has the strongest second position, while the second
    # request has the strongest first position.  The selected counts must still
    # describe valid per-request prefixes.
    policy = _policy({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0})
    survival = torch.tensor([[0.9, 0.8, 0.1], [0.7, 0.2, 0.1]])

    lengths = policy.allocate(survival)

    assert all(0 <= length <= 3 for length in lengths.tolist())
    assert lengths.tolist() == [3, 3]


def test_hardware_policy_respects_minimum_total_budget() -> None:
    policy = _policy(
        {1: 1.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0, 6: 10.0},
        max_draft_tokens=3,
    )
    survival = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])

    lengths = policy.allocate(survival, min_total_tokens=4)

    assert int(lengths.sum().item()) >= 4


def test_hardware_policy_amortizes_request_prefix_mapping() -> None:
    policy = _policy(
        {2: 1.0, 3: 1.0, 4: 1.0, 5: 100.0, 6: 100.0, 7: 100.0},
        max_batch_size=2,
        decision_interval=64,
        allocation_interval=4,
    )

    first = policy.allocate(torch.tensor([[0.9, 0.1, 0.1], [0.8, 0.1, 0.1]]))
    held = policy.allocate(torch.tensor([[0.9, 0.9, 0.1], [0.1, 0.1, 0.1]]))

    # The hardware total stays fixed, but the request mapping is held until
    # the configured cadence instead of re-running top-k on every step.
    assert first.tolist() == [1, 1]
    assert held.tolist() == [1, 1]
    assert held.data_ptr() == first.data_ptr()

    policy.allocate(torch.tensor([[0.9, 0.9, 0.1], [0.1, 0.1, 0.1]]))
    refreshed = policy.allocate(
        torch.tensor([[0.9, 0.9, 0.1], [0.1, 0.1, 0.1]])
    )
    assert refreshed.tolist() == [2, 0]


def test_hardware_scheduler_can_hold_confidence_result() -> None:
    from vllm_ascend.spec_decode.utils import DynamicSpecScheduler

    scheduler = DynamicSpecScheduler(
        method="dspark",
        policy="hardware_aware",
        method_params={
            "profile": {
                "fingerprint": {"device": "Ascend", "graph_mode": "FULL_DECODE_ONLY"},
                "latency_ms": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 10.0},
                "confidence_temperatures": [1.0, 1.0, 1.0],
            },
            "confidence_update_interval": 2,
            "decision_interval": 64,
            "hardware_min_budget_ratio": 0.0,
        },
        max_batch_size=2,
        num_speculative_tokens=3,
        device=torch.device("cpu"),
    )
    model = _FakeDSparkModel()
    draft_ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    hidden = torch.zeros((6, 1))

    scheduler.update(
        model=model,
        last_hidden_states=hidden,
        draft_token_ids=draft_ids,
        num_reqs=2,
    )
    scheduler.update(
        model=model,
        last_hidden_states=hidden + 1,
        draft_token_ids=draft_ids,
        num_reqs=2,
    )

    assert model.confidence_calls == 1

    scheduler.update(
        model=model,
        last_hidden_states=hidden + 1,
        draft_token_ids=draft_ids,
        num_reqs=2,
    )
    assert model.confidence_calls == 2


def test_hybrid_policy_skips_confidence_for_small_batch() -> None:
    from vllm_ascend.spec_decode.utils import DynamicSpecScheduler

    scheduler = DynamicSpecScheduler(
        method="dspark",
        policy="hardware_aware",
        method_params={
            "profile": {
                "fingerprint": {"device": "Ascend", "graph_mode": "FULL_DECODE_ONLY"},
                "latency_ms": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0},
            },
            "hardware_min_budget_ratio": 0.0,
            "hybrid_policy_enabled": True,
            "hybrid_min_batch_size": 8,
        },
        max_batch_size=8,
        num_speculative_tokens=3,
        device=torch.device("cpu"),
    )
    model = _FakeDSparkModel()
    draft_ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    hidden = torch.zeros((6, 1))

    first = scheduler.update(
        model=model,
        last_hidden_states=hidden,
        draft_token_ids=draft_ids,
        num_reqs=2,
    )
    second = scheduler.update(
        model=model,
        last_hidden_states=hidden,
        draft_token_ids=draft_ids,
        num_reqs=2,
    )

    assert first.tolist() == [3, 3]
    assert second.tolist() == [3, 3]
    assert model.confidence_calls == 0
    assert scheduler.reused_last_result


def test_hybrid_policy_enters_dynamic_path_for_large_low_acceptance_batch() -> None:
    from vllm_ascend.spec_decode.utils import DynamicSpecScheduler

    latency_ms = {str(size): (1.0 if size <= 8 else 100.0) for size in range(1, 33)}
    scheduler = DynamicSpecScheduler(
        method="dspark",
        policy="hardware_aware",
        method_params={
            "profile": {
                "fingerprint": {"device": "Ascend", "graph_mode": "FULL_DECODE_ONLY"},
                "latency_ms": latency_ms,
            },
            "hardware_min_budget_ratio": 0.0,
            "hybrid_policy_enabled": True,
            "hybrid_min_batch_size": 8,
            "hybrid_acceptance_threshold": 0.99,
        },
        max_batch_size=8,
        num_speculative_tokens=3,
        device=torch.device("cpu"),
    )
    model = _FakeDSparkModel()
    draft_ids = torch.tensor([[1, 2, 3, 4]] * 8)
    hidden = torch.zeros((24, 1))

    lengths = scheduler.update(
        model=model,
        last_hidden_states=hidden,
        draft_token_ids=draft_ids,
        num_reqs=8,
    )

    assert model.confidence_calls == 1
    assert int(lengths.sum().item()) < 8 * 3


def test_hybrid_policy_uses_profile_goodput_when_acceptance_is_borderline() -> None:
    from vllm_ascend.spec_decode.utils import DynamicSpecScheduler

    # The full-prefix survival is just above the acceptance threshold, but
    # the profile makes the full-width graph much more expensive than the
    # shorter dynamic candidates.  The cost guard must keep the scheduler in
    # dynamic K instead of holding full width only because of acceptance.
    latency_ms = {
        **{str(size): 1.0 for size in range(1, 25)},
        **{str(size): 10.0 for size in range(25, 33)},
    }
    scheduler = DynamicSpecScheduler(
        method="dspark",
        policy="hardware_aware",
        method_params={
            "profile": {
                "fingerprint": {"device": "Ascend", "graph_mode": "FULL_DECODE_ONLY"},
                "latency_ms": latency_ms,
            },
            "hardware_min_budget_ratio": 0.0,
            "hybrid_policy_enabled": True,
            "hybrid_min_batch_size": 8,
            "hybrid_acceptance_threshold": 0.6,
            "confidence_update_interval": 16,
        },
        max_batch_size=8,
        num_speculative_tokens=3,
        device=torch.device("cpu"),
    )
    model = _FakeDSparkModel()
    draft_ids = torch.tensor([[1, 2, 3, 4]] * 8)
    hidden = torch.zeros((24, 1))

    first = scheduler.update(
        model=model,
        last_hidden_states=hidden,
        draft_token_ids=draft_ids,
        num_reqs=8,
    )
    second = scheduler.update(
        model=model,
        last_hidden_states=hidden,
        draft_token_ids=draft_ids,
        num_reqs=8,
    )

    assert float(scheduler._hybrid_last_acceptance) >= 0.6
    assert scheduler._hybrid_last_dynamic_goodput is not None
    assert scheduler._hybrid_last_full_width_goodput is not None
    assert (
        scheduler._hybrid_last_dynamic_goodput
        > scheduler._hybrid_last_full_width_goodput
    )
    assert int(first.sum().item()) < 8 * 3
    assert int(second.sum().item()) < 8 * 3
    assert model.confidence_calls == 1


def test_hybrid_policy_holds_large_high_acceptance_batch_until_probe() -> None:
    from vllm_ascend.spec_decode.utils import DynamicSpecScheduler

    scheduler = DynamicSpecScheduler(
        method="dspark",
        policy="hardware_aware",
        method_params={
            "profile": {
                "fingerprint": {"device": "Ascend", "graph_mode": "FULL_DECODE_ONLY"},
                "latency_ms": {str(size): 1.0 for size in range(1, 33)},
            },
            "hardware_min_budget_ratio": 0.0,
            "hybrid_policy_enabled": True,
            "hybrid_min_batch_size": 8,
            "hybrid_acceptance_threshold": 0.6,
            "hybrid_probe_interval": 4,
        },
        max_batch_size=8,
        num_speculative_tokens=3,
        device=torch.device("cpu"),
    )
    model = _FakeDSparkModel()
    draft_ids = torch.tensor([[1, 2, 3, 4]] * 8)
    hidden = torch.full((24, 1), 10.0)

    first = scheduler.update(
        model=model,
        last_hidden_states=hidden,
        draft_token_ids=draft_ids,
        num_reqs=8,
    )
    second = scheduler.update(
        model=model,
        last_hidden_states=hidden,
        draft_token_ids=draft_ids,
        num_reqs=8,
    )
    third = scheduler.update(
        model=model,
        last_hidden_states=hidden,
        draft_token_ids=draft_ids,
        num_reqs=8,
    )

    assert first.tolist() == [3] * 8
    assert second.tolist() == [3] * 8
    assert third.tolist() == [3] * 8
    assert model.confidence_calls == 1
    # The first held step refreshes the request-to-length host mapping after
    # the probe. Subsequent held steps can safely reuse that mapping.
    assert scheduler.reused_last_result


def test_hardware_policy_uses_nearest_profiled_shape() -> None:
    model = HardwareCostModel.from_dict(
        {
            "fingerprint": {"device": "Ascend"},
            "latency_ms": {"1": 1.0, "4": 2.0},
        },
        expected_fingerprint={"device": "Ascend"},
    )

    assert model.latency(2) == 2.0
    assert model.latency(10) == 2.0


def test_hardware_profile_fingerprint_mismatch() -> None:
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        HardwareCostModel.from_dict(
            {"fingerprint": {"device": "A3"}, "latency_ms": {"1": 1.0}},
            expected_fingerprint={"device": "A5"},
        )


def test_startup_profile_collector_uses_median_and_persists(tmp_path) -> None:
    collector = HardwareProfileCollector(
        batch_sizes=(1, 2),
        verify_token_sizes=(0, 1),
        warmup_runs=1,
        measure_runs=3,
    )
    calls = []

    def measure(batch_size: int, verify_k: int) -> float:
        calls.append((batch_size, verify_k))
        return float(batch_size * 10 + verify_k + len(calls) % 2)

    payload = collector.collect(measure, fingerprint={"device": "A3"})
    assert len(calls) == 2 * 2 * 4
    assert payload["profile_kind"] == "startup_dummy_step"
    assert payload["fingerprint"] == {"device": "A3"}
    assert set(payload["latency_ms"]) == {"1", "2", "4"}

    output_path = tmp_path / "startup_profile.json"
    HardwareProfileCollector.save(payload, output_path)
    loaded = HardwareCostModel.from_json(output_path)
    assert loaded.latency(1) == payload["latency_ms"]["1"]


def test_hardware_profile_collector_filters_shapes_by_capacity() -> None:
    collector = HardwareProfileCollector.from_params(
        max_batch_size=8,
        max_draft_tokens=3,
        max_token_capacity=16,
        params={
            "profile_batch_sizes": [1, 4, 8],
            "profile_verify_tokens": [0, 1, 3],
        },
    )

    assert collector.batch_sizes == (1, 4)
    assert collector.verify_token_sizes == (0, 1, 3)


def test_confidence_ema_follows_request_ids_when_batch_reorders() -> None:
    from vllm_ascend.spec_decode.utils import DynamicSpecScheduler

    scheduler = DynamicSpecScheduler(
        method="dflash",
        policy="confidence_budget",
        method_params={"ema_alpha": 0.5},
        max_batch_size=2,
        num_speculative_tokens=2,
        device=torch.device("cpu"),
    )

    first = scheduler._update_from_token_probs(
        torch.tensor([[0.9, 0.8], [0.2, 0.2]]),
        request_ids=["a", "b"],
    )
    assert first.shape == (2,)

    second_probs = torch.tensor([[0.6, 0.6], [0.8, 0.8]])
    scheduler._update_from_token_probs(
        second_probs,
        request_ids=["b", "a"],
    )
    assert torch.allclose(second_probs[0], torch.tensor([0.4, 0.4]))
    assert torch.allclose(second_probs[1], torch.tensor([0.85, 0.8]))


def test_auto_profile_scheduler_accepts_runtime_profile() -> None:
    from vllm_ascend.spec_decode.utils import DynamicSpecScheduler

    scheduler = DynamicSpecScheduler(
        method="dspark",
        policy="hardware_aware",
        method_params={
            "auto_profile": True,
            "hardware_min_budget_ratio": 0.0,
        },
        max_batch_size=2,
        num_speculative_tokens=2,
        device=torch.device("cpu"),
    )
    assert scheduler.policy_name == "hardware_aware"
    assert scheduler.hardware_policy is None

    scheduler.set_hardware_profile(
        {
            "fingerprint": {"device": "A3"},
            "latency_ms": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "6": 1.0},
        }
    )
    assert scheduler.hardware_policy is not None
    assert scheduler.cost_model is not None


def test_hardware_profile_accepts_offline_tuner_artifact() -> None:
    model = HardwareCostModel.from_dict(
        {"batch_stats": {}, "hardware_profile": {"latency_ms": {"1": 1.0}}}
    )

    assert model.latency(1) == 1.0


def test_sequential_temperature_scaler_calibrates_positions() -> None:
    scaler = SequentialTemperatureScaler.from_config([2.0, 0.5], 2)
    probabilities = torch.tensor([[0.8, 0.2]])

    calibrated = scaler.calibrate_probabilities(probabilities)

    assert calibrated.shape == probabilities.shape
    assert calibrated[0, 0] < probabilities[0, 0]
    assert calibrated[0, 1] < probabilities[0, 1]


def test_sequential_temperature_scaler_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="exactly 3"):
        SequentialTemperatureScaler.from_config([1.0, 1.0], 3)


def test_sequential_temperature_scaler_accepts_dynamic_prefix_width() -> None:
    scaler = SequentialTemperatureScaler.from_config([1.0, 1.2, 1.4], 3)
    logits = torch.tensor([[0.2, -0.4]])
    calibrated = scaler.calibrate_logits(logits)
    assert calibrated.shape == logits.shape


def test_hardware_policy_accepts_smaller_dynamic_width() -> None:
    policy = _policy({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}, max_draft_tokens=3)
    lengths = policy.allocate(torch.tensor([[0.9, 0.8]]))
    assert lengths.shape == (1,)
    assert lengths.item() <= 2


def test_proposal_gate_enters_latency_profile_after_low_load_streak() -> None:
    gate = ProposalGate(
        max_num_seqs=8,
        enter_ratio=0.5,
        enter_steps=2,
        exit_steps=1,
    )

    assert gate.select_k(
        4,
        num_running=1,
        num_waiting=0,
        total_num_scheduled_tokens=1,
        num_scheduled_requests=1,
        prefill_scheduled=False,
    ) == 0
    assert gate.select_k(
        4,
        num_running=1,
        num_waiting=0,
        total_num_scheduled_tokens=1,
        num_scheduled_requests=1,
        prefill_scheduled=False,
    ) == 4


def test_proposal_gate_exits_immediately_when_queue_builds() -> None:
    gate = ProposalGate(max_num_seqs=4, enter_steps=1, exit_steps=1)
    assert gate.select_k(
        2,
        num_running=1,
        num_waiting=0,
        total_num_scheduled_tokens=1,
        num_scheduled_requests=1,
        prefill_scheduled=False,
    ) == 2
    assert gate.select_k(
        2,
        num_running=2,
        num_waiting=1,
        total_num_scheduled_tokens=2,
        num_scheduled_requests=2,
        prefill_scheduled=False,
    ) == 0


def test_adaptive_draft_k_tracks_logical_verify_width() -> None:
    controller = AdaptiveDraftKController(max_k=5, min_k=1, slack=1)

    # The first step keeps the configured width; the result feeds the next
    # scheduler step and removes one unused draft position.
    assert controller.cap(5) == 5
    controller.update([3, 2])
    assert controller.current_k == 4
    assert controller.cap(5) == 4

    # A prefix that reaches the physical width allows gradual growth again.
    controller.update([4, 4])
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
