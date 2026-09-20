# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from vllm_ascend.worker.v2.spec_decode.physical_k_profile import (
    configure_physical_k_profiling,
    physical_k_profile_scope,
    profiling_physical_k,
)


class FakeManager:
    num_speculative_steps = 4

    def __init__(self):
        self.req_states = SimpleNamespace(
            req_id_to_index={str(i): i for i in range(8)},
            num_computed_tokens_np=np.full(8, 100),
            prefill_len=SimpleNamespace(np=np.zeros(8)),
        )
        self._stale_idx = 0
        self.num_bonus_tokens = 1
        self._max_total_logits = 128
        self._stale_confidences = [
            SimpleNamespace(np=np.full((8, 4), 0.9)),
        ]
        self.cost_tables = None

    def batches_to_profile(self, capture_sizes):
        del capture_sizes
        for _ in range(2):
            yield {"num_tokens": 8, "context_len": 1024}

    def set_initial_cost_curves(self, samples):
        assert all(sample.physical_k == 4 for sample in samples)
        self.cost_tables = (np.ones(16), np.ones(128))

    def get_num_tokens(self, num_tokens_per_req, draft_tokens):
        del draft_tokens
        return sum(num_tokens_per_req.values())


def config():
    return SimpleNamespace(
        additional_config={
            "dynamic_spec_config": {
                "method": "dspark",
                "physical_k": {"min_k": 3},
            }
        }
    )


def test_profile_scope_is_nested_and_restored():
    assert profiling_physical_k() is None
    with physical_k_profile_scope(3):
        assert profiling_physical_k() == 3
        with physical_k_profile_scope(4):
            assert profiling_physical_k() == 4
        assert profiling_physical_k() == 3
    assert profiling_physical_k() is None


def test_profile_grid_repeats_upstream_cases_for_every_k():
    manager = configure_physical_k_profiling(FakeManager(), config())
    cases = list(manager.batches_to_profile([8]))
    assert [case["profile_physical_k"] for case in cases] == [3, 3, 4, 4]


def test_lower_k_profile_replays_are_sparse_but_max_k_is_unchanged():
    manager = FakeManager()

    def batches_to_profile(self, capture_sizes):
        del self, capture_sizes
        for _ in range(3):
            yield {"num_tokens": 8, "context_len": 1024}

    manager.batches_to_profile = MethodType(batches_to_profile, manager)
    manager = configure_physical_k_profiling(manager, config())
    cases = list(manager.batches_to_profile([8]))
    assert [case["profile_physical_k"] for case in cases] == [3, 3, 4, 4, 4]


def test_profile_grid_extends_through_large_rl_batch_buckets():
    manager = FakeManager()
    manager.req_states.max_num_reqs = 64
    manager = configure_physical_k_profiling(manager, config())
    cases = list(manager.batches_to_profile([8]))
    by_k = {
        physical_k: {
            case["num_tokens"]
            for case in cases
            if case["profile_physical_k"] == physical_k
        }
        for physical_k in (3, 4)
    }
    assert by_k == {3: {8, 16, 32, 64}, 4: {8, 16, 32, 64}}


def test_worker_recommends_k_from_profile_cost_and_confidence():
    manager = configure_physical_k_profiling(FakeManager(), config())
    list(manager.batches_to_profile([8]))
    samples = [
        SimpleNamespace(num_reqs=8, drafter_ms=1.0, physical_k=3),
        SimpleNamespace(num_reqs=8, drafter_ms=1.2, physical_k=3),
        SimpleNamespace(num_reqs=8, drafter_ms=10.0, physical_k=4),
        SimpleNamespace(num_reqs=8, drafter_ms=10.2, physical_k=4),
    ]
    manager.set_initial_cost_curves(samples)
    per_req = {str(i): 5 for i in range(8)}
    drafts = {str(i): [1, 2, 3, 4] for i in range(8)}
    manager.get_num_tokens(per_req, drafts)
    batch_size, physical_k, cost_floor = manager._physical_k_recommendation
    assert batch_size == 8
    assert physical_k == 3
    assert cost_floor == 3


def test_narrowed_runtime_width_does_not_replace_full_width_recommendation():
    manager = configure_physical_k_profiling(FakeManager(), config())
    list(manager.batches_to_profile([8]))
    samples = [
        SimpleNamespace(num_reqs=8, drafter_ms=5.0, physical_k=3),
        SimpleNamespace(num_reqs=8, drafter_ms=5.0, physical_k=3),
        SimpleNamespace(num_reqs=8, drafter_ms=1.0, physical_k=4),
        SimpleNamespace(num_reqs=8, drafter_ms=1.0, physical_k=4),
    ]
    manager.set_initial_cost_curves(samples)
    per_req = {str(i): 4 for i in range(8)}
    drafts = {str(i): [1, 2, 3] for i in range(8)}
    manager.get_num_tokens(per_req, drafts)
    assert manager._physical_k_recommendation is None


def test_cost_floor_rejects_shorter_k_dominated_by_wider_graph():
    manager = configure_physical_k_profiling(FakeManager(), config())
    list(manager.batches_to_profile([8]))
    samples = [
        SimpleNamespace(num_reqs=8, drafter_ms=10.0, physical_k=3),
        SimpleNamespace(num_reqs=8, drafter_ms=10.0, physical_k=3),
        SimpleNamespace(num_reqs=8, drafter_ms=2.0, physical_k=4),
        SimpleNamespace(num_reqs=8, drafter_ms=2.0, physical_k=4),
    ]
    manager.set_initial_cost_curves(samples)
    per_req = {str(i): 5 for i in range(8)}
    drafts = {str(i): [1, 2, 3, 4] for i in range(8)}
    manager.get_num_tokens(per_req, drafts)
    assert manager._physical_k_recommendation[2] == 4


def test_eager_target_samples_do_not_price_draft_k():
    manager = configure_physical_k_profiling(FakeManager(), config())
    list(manager.batches_to_profile([8]))
    samples = [
        SimpleNamespace(num_reqs=8, drafter_ms=1.0, physical_k=3, full_cudagraph=True),
        SimpleNamespace(num_reqs=8, drafter_ms=100.0, physical_k=3, full_cudagraph=False),
        SimpleNamespace(num_reqs=8, drafter_ms=2.0, physical_k=4, full_cudagraph=True),
        SimpleNamespace(num_reqs=8, drafter_ms=200.0, physical_k=4, full_cudagraph=False),
    ]
    manager.set_initial_cost_curves(samples)
    assert manager._physical_k_draft_costs == {3: {8: 1.0}, 4: {8: 2.0}}


def test_profile_cost_is_not_extrapolated_to_larger_batch():
    manager = configure_physical_k_profiling(FakeManager(), config())
    list(manager.batches_to_profile([8]))
    samples = [
        SimpleNamespace(num_reqs=8, drafter_ms=1.0, physical_k=3),
        SimpleNamespace(num_reqs=8, drafter_ms=1.0, physical_k=3),
        SimpleNamespace(num_reqs=8, drafter_ms=2.0, physical_k=4),
        SimpleNamespace(num_reqs=8, drafter_ms=2.0, physical_k=4),
    ]
    manager.set_initial_cost_curves(samples)
    manager.req_states.req_id_to_index.update({str(i): i for i in range(8, 16)})
    manager.req_states.num_computed_tokens_np = np.full(16, 100)
    manager.req_states.prefill_len.np = np.zeros(16)
    manager._stale_confidences[0].np = np.full((16, 4), 0.9)
    per_req = {str(i): 5 for i in range(16)}
    drafts = {str(i): [1, 2, 3, 4] for i in range(16)}
    manager.get_num_tokens(per_req, drafts)
    assert manager._physical_k_recommendation is None


def test_nonzero_tp_rank_skips_runtime_scoring():
    manager = configure_physical_k_profiling(FakeManager(), config())
    list(manager.batches_to_profile([8]))
    samples = [
        SimpleNamespace(num_reqs=8, drafter_ms=1.0, physical_k=3),
        SimpleNamespace(num_reqs=8, drafter_ms=1.0, physical_k=3),
        SimpleNamespace(num_reqs=8, drafter_ms=2.0, physical_k=4),
        SimpleNamespace(num_reqs=8, drafter_ms=2.0, physical_k=4),
    ]
    manager.set_initial_cost_curves(samples)
    per_req = {str(i): 5 for i in range(8)}
    drafts = {str(i): [1, 2, 3, 4] for i in range(8)}
    with patch(
        "vllm_ascend.worker.v2.spec_decode.physical_k_profile._is_tp_rank_zero",
        return_value=False,
    ):
        manager.get_num_tokens(per_req, drafts)
    assert manager._physical_k_recommendation is None
