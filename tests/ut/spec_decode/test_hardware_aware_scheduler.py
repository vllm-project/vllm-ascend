# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm_ascend.spec_decode.dynamic.calibration import SequentialTemperatureScaler
from vllm_ascend.spec_decode.dynamic.cost_model import HardwareCostModel
from vllm_ascend.spec_decode.dynamic.policy import HardwareAwarePrefixPolicy
from vllm_ascend.spec_decode.dynamic.proposal_gate import ProposalGate


def _policy(
    latency_ms: dict[int, float],
    *,
    min_k: int = 0,
    max_batch_size: int = 4,
    max_draft_tokens: int = 3,
) -> HardwareAwarePrefixPolicy:
    model = HardwareCostModel(latency_ms=latency_ms, fingerprint={})
    return HardwareAwarePrefixPolicy(
        cost_model=model,
        min_k=min_k,
        max_batch_size=max_batch_size,
        max_draft_tokens=max_draft_tokens,
        device=torch.device("cpu"),
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
