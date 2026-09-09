# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Physical draft K and batch gating layered over upstream verification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ascend_config import DynamicSpecConfig
from vllm_ascend.spec_decode.dynamic.draft_k_controller import AdaptiveDraftKController
from vllm_ascend.spec_decode.dynamic.proposal_gate import ProposalGate
from vllm_ascend.spec_decode.utils import DynamicSpecScheduler
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.physical_k import physical_k_scope


@pytest.mark.parametrize(
    "dynamic_config",
    [
        {"method_params": {"v2_varlen_physical_k": True}},
        {"method": "dspark", "policy": "hardware_aware", "physical_k": {"min_k": 3, "capture_k": [3, 5]}},
    ],
)
def test_v2_physical_k_scope_updates_query_layout_atomically(dynamic_config) -> None:
    speculator = SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config={"dynamic_spec_config": dynamic_config}),
        num_speculative_steps=5,
        num_query_per_req=5,
        max_num_reqs=2,
        sample_from_anchor=True,
        sample_col=torch.arange(5).repeat(2),
        draft_token_confidence_probs=torch.zeros((2, 5)),
        _anchor_idx=torch.arange(2) * 5,
    )
    batch = SimpleNamespace(num_draft_tokens_per_req=torch.tensor([2, 2]))

    with physical_k_scope(speculator, batch) as active_k:
        assert active_k == 2
        assert speculator.num_speculative_steps == 2
        assert speculator.num_query_per_req == 2
        assert speculator.sample_col.tolist() == [0, 1, 0, 1]
        assert tuple(speculator.draft_token_confidence_probs.shape) == (2, 2)
        assert speculator._anchor_idx.tolist() == [0, 2]

    assert speculator.num_speculative_steps == 5
    assert speculator.num_query_per_req == 5
    assert speculator.sample_col.tolist() == [0, 1, 2, 3, 4] * 2
    assert tuple(speculator.draft_token_confidence_probs.shape) == (2, 5)
    assert speculator._anchor_idx.tolist() == [0, 5]


def test_v2_physical_k_scope_keeps_mixed_batch_on_safe_width() -> None:
    speculator = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={"dynamic_spec_config": {"method_params": {"v2_varlen_physical_k": True}}}
        ),
        num_speculative_steps=5,
        num_query_per_req=5,
        max_num_reqs=2,
        sample_from_anchor=True,
        sample_col=torch.arange(5).repeat(2),
        _anchor_idx=torch.arange(2) * 5,
    )
    batch = SimpleNamespace(num_draft_tokens_per_req=torch.tensor([2, 3]))

    with physical_k_scope(speculator, batch) as active_k:
        assert active_k == 5
        assert speculator.num_speculative_steps == 5
        assert speculator.num_query_per_req == 5


def test_v2_physical_k_scope_uses_next_draft_width_from_scheduler() -> None:
    speculator = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={"dynamic_spec_config": {"method_params": {"v2_varlen_physical_k": True}}}
        ),
        num_speculative_steps=5,
        num_query_per_req=5,
        max_num_reqs=2,
        sample_from_anchor=True,
        sample_col=torch.arange(5).repeat(2),
        _anchor_idx=torch.arange(2) * 5,
    )
    batch = SimpleNamespace(
        num_draft_tokens_per_req=torch.tensor([5, 5]),
        _vllm_ascend_physical_draft_k=2,
    )

    with physical_k_scope(speculator, batch) as active_k:
        assert active_k == 2
        assert speculator.num_speculative_steps == 2
        assert speculator.num_query_per_req == 2


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


def test_v2_physical_k_scope_reuses_preallocated_device_indices() -> None:
    speculator = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={"dynamic_spec_config": {"method_params": {"v2_varlen_physical_k": True}}}
        ),
        num_speculative_steps=5,
        num_query_per_req=5,
        max_num_reqs=2,
        sample_from_anchor=True,
        sample_col=torch.arange(5).repeat(2),
        _anchor_idx=torch.arange(2) * 5,
    )
    batch = SimpleNamespace(num_draft_tokens_per_req=torch.tensor([2, 2]))

    with physical_k_scope(speculator, batch):
        sample_col = speculator.sample_col
        anchor_idx = speculator._anchor_idx

    with physical_k_scope(speculator, batch):
        assert speculator.sample_col.data_ptr() == sample_col.data_ptr()
        assert speculator._anchor_idx.data_ptr() == anchor_idx.data_ptr()


def test_v2_dflash_physical_k_writes_only_active_prefix() -> None:
    """A smaller captured K must not resize the fixed draft-token buffer."""
    speculator = SimpleNamespace(
        num_speculative_steps=4,
        sample_indices=torch.arange(8, dtype=torch.int64),
        sample_pos=torch.zeros(8, dtype=torch.int64),
        sample_idx_mapping=torch.zeros(8, dtype=torch.int32),
        temperature=torch.zeros(2),
        seeds=torch.zeros(2, dtype=torch.int64),
        sample_col=torch.arange(4, dtype=torch.int32).repeat(2),
        draft_logits=None,
        draft_tokens=torch.full((2, 5), -1, dtype=torch.int64),
    )
    speculator._run_model = lambda *args: torch.zeros((8, 1))
    speculator.sample_draft = lambda *args: torch.arange(8, dtype=torch.int64)

    AscendDFlashSpeculator._generate_draft(
        speculator,
        num_reqs=2,
        num_tokens_padded=8,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
    )

    assert speculator.draft_tokens.tolist() == [
        [0, 1, 2, 3, -1],
        [4, 5, 6, 7, -1],
    ]


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
    with pytest.raises(ValueError, match="reuse_upstream_adaptive_verification"):
        DynamicSpecConfig(
            method="dspark",
            policy="hardware_aware",
            method_params={"reuse_upstream_adaptive_verification": False},
        )


def test_upstream_manager_configuration_remains_compatible() -> None:
    config = DynamicSpecConfig(
        method="dspark",
        policy="hardware_aware",
        method_params={"reuse_upstream_adaptive_verification": True},
    )
    assert config.policy == "hardware_aware"


def test_v1_scheduler_rejects_removed_hardware_policy() -> None:
    with pytest.raises(ValueError, match="legacy V1 scheduler"):
        DynamicSpecScheduler(
            method="dspark",
            policy="hardware_aware",
            method_params={},
            max_batch_size=2,
            num_speculative_tokens=5,
            device=torch.device("cpu"),
        )


def test_v1_confidence_budget_handles_smaller_draft_width() -> None:
    scheduler = DynamicSpecScheduler(
        method="dflash",
        method_params={},
        max_batch_size=2,
        num_speculative_tokens=5,
        device=torch.device("cpu"),
    )
    result = scheduler.update(logits=torch.zeros((4, 8)), num_reqs=2)
    assert result.tolist() == [2, 2]
