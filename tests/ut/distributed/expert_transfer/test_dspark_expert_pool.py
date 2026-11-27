# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend.distributed.expert_transfer import dspark_expert_pool as pool_module
from vllm_ascend.distributed.expert_transfer.dspark_expert_pool import (
    DSparkExpertPool,
    build_roles,
)
from vllm_ascend.distributed.expert_transfer.expert_offload_manager import ExpertOffloadManager


@pytest.mark.parametrize(
    ("boundaries", "drafts", "prefilling", "padded_rows", "expected"),
    [
        ([0, 6], [5], [False], 6, [1, 1, 2, 2, 2, 2]),
        ([0, 6, 12], [5, 5], [False, False], 12, [1, 1, 2, 2, 2, 2] * 2),
        ([0, 6], [5], [False], 12, [1, 1, 2, 2, 2, 2] + [0] * 6),
        ([0, 6, 12], [5, 5], [True, False], 12, [0] * 6 + [1, 1, 2, 2, 2, 2]),
        ([0, 2, 6], [1, 3], [False, False], 6, [1, 1, 1, 1, 2, 2]),
        ([0, 1], None, [False], 1, [0]),
        ([0, 1, 7], [0, 5], [False, False], 12, [1, 1, 1, 2, 2, 2, 2] + [0] * 5),
    ],
)
def test_roles_follow_real_requests(boundaries, drafts, prefilling, padded_rows, expected):
    roles = build_roles(
        query_start_loc=np.array(boundaries),
        draft_counts=drafts,
        is_prefilling=prefilling,
        padded_rows=padded_rows,
        protected_positions=2,
    )
    assert roles.tolist() == expected


@pytest.mark.parametrize(
    ("configured_drafts", "rows", "drafts", "prefilling", "structured", "expected"),
    [
        (5, [6], [5], [False], False, True),
        (5, [6, 6], [5, 5], [False, False], False, True),
        (5, [6, 6], [5, 5], [True, False], False, False),
        (5, [6, 6], [5, 5], [False, False], True, False),
        (5, [6, 6], [5, 1], [False, False], False, False),
        (5, [1, 1], None, [False, False], False, False),
        (5, [6, 6, 6], [5, 5, 5], [False] * 3, False, True),
        (3, [4, 4, 4], [3, 3, 3], [False] * 3, False, True),
        (7, [6, 6], [5, 5], [False, False], False, False),
    ],
)
def test_verify_graph_dispatch(configured_drafts, rows, drafts, prefilling, structured, expected):
    state = SimpleNamespace(
        num_scheduled_tokens=np.array(rows), num_draft_tokens_np=drafts, is_prefilling_np=prefilling
    )
    assert (
        ExpertOffloadManager._verify_graph_eligible(
            state, num_speculative_tokens=configured_drafts, structured=structured
        )
        is expected
    )


def make_batch(boundaries, drafts, padded_rows, *, structured=False):
    return SimpleNamespace(
        num_reqs=len(boundaries) - 1,
        num_tokens=boundaries[-1],
        num_tokens_after_padding=padded_rows,
        query_start_loc_np=np.array(boundaries),
        num_draft_tokens_per_req=drafts,
        is_prefilling_np=[False] * (len(boundaries) - 1),
        has_structured_output_reqs=structured,
        has_prefill=False,
    )


def test_same_graph_bucket_refreshes_padding_and_disables_on_ordinary_decode():
    pool = DSparkExpertPool(
        torch.device("cpu"),
        capacity=12,
        protected_positions=2,
        suffix_experts=0,
        num_speculative_tokens=5,
        num_device_experts=84,
    )
    pool.prepare_capture(make_batch([0, 6, 12], None, 12))
    address = pool.roles.data_ptr()
    assert pool.active and pool.roles.tolist() == [1, 1, 2, 2, 2, 2] * 2
    pool.prepare(make_batch([0, 6], [5], 12))
    assert pool.active and pool.roles.tolist() == [1, 1, 2, 2, 2, 2] + [0] * 6
    assert pool.roles.data_ptr() == address
    pool.prepare(make_batch([0, 1], None, 1))
    assert not pool.active


def test_structured_step_clears_active_routing():
    pool = DSparkExpertPool(
        torch.device("cpu"),
        capacity=6,
        protected_positions=2,
        suffix_experts=0,
        num_speculative_tokens=5,
        num_device_experts=84,
    )
    pool.prepare(make_batch([0, 6], [5], 6))
    assert pool.active
    pool.prepare(make_batch([0, 6], [5], 6, structured=True))
    assert not pool.active


def test_prefill_step_disables_pool_then_decode_restores_current_roles():
    pool = DSparkExpertPool(torch.device("cpu"), 12, 2, 2, num_speculative_tokens=5, num_device_experts=84)
    batch = make_batch([0, 1, 7], [0, 5], 12)
    pool.prepare(batch)
    assert pool.active
    address = pool.roles.data_ptr()
    batch.has_prefill = True
    batch.is_prefilling_np = [True, False]
    pool.prepare(batch)
    assert not pool.active and not pool.has_native_rows
    batch.has_prefill = False
    batch.is_prefilling_np = [False, False]
    pool.prepare(batch)
    assert pool.active and not pool.has_native_rows
    assert pool.roles.data_ptr() == address
    assert pool.roles.tolist() == [1, 1, 1, 2, 2, 2, 2] + [0] * 5


def test_mixed_step_preserves_native_outputs_only_outside_verification():
    pool = DSparkExpertPool(
        torch.device("cpu"),
        capacity=7,
        protected_positions=2,
        suffix_experts=1,
        num_speculative_tokens=5,
        num_device_experts=84,
    )
    # A real row before the eligible verification suffix keeps native routing.
    pool.prepare(make_batch([0, 7], [5], 7))
    native = (torch.full((7, 6), 0.25), torch.full((7, 6), 17, dtype=torch.int32))
    fused = (torch.full((7, 6), 0.75), torch.full((7, 6), 3, dtype=torch.int32))
    weights, ids = pool.merge_native(fused, native)
    torch.testing.assert_close(weights[:, 0], torch.tensor([0.25] + [0.75] * 6))
    assert ids[:, 0].tolist() == [17] + [3] * 6


def test_capture_uses_the_configured_verify_width_for_all_requests():
    pool = DSparkExpertPool(
        torch.device("cpu"),
        capacity=16,
        protected_positions=2,
        suffix_experts=0,
        num_speculative_tokens=3,
        num_device_experts=72,
    )
    pool.prepare_capture(make_batch([0, 4, 8, 12], None, 12))
    assert pool.active
    assert pool.roles[:12].tolist() == [1, 1, 2, 2] * 3
    pool.prepare(make_batch([0, 4, 8], [3, 3], 12))
    assert pool.roles[:12].tolist() == [1, 1, 2, 2] * 2 + [0] * 4


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_non_fp32_logits_are_rejected_before_kernel_dispatch(monkeypatch, dtype):
    route = Mock()
    monkeypatch.setattr(pool_module, "route_experts", route)
    pool = DSparkExpertPool(torch.device("cpu"), 6, 2, 2, num_speculative_tokens=5, num_device_experts=84)
    pool.num_rows = 6
    router = SimpleNamespace(num_logical_experts=256, e_score_correction_bias=None, routed_scaling_factor=1.5)

    with pytest.raises(ValueError, match="aligned FP32 logits"):
        pool.route(router, torch.zeros(6, 256, dtype=dtype))

    route.assert_not_called()
    assert pool.workspaces == {}


def test_fp32_logits_reach_kernel_unchanged_with_fp32_scratch(monkeypatch):
    weights = torch.full((6, 6), 0.25)
    ids = torch.zeros((6, 6), dtype=torch.int32)
    route = Mock(return_value=(weights, ids))
    monkeypatch.setattr(pool_module, "route_experts", route)
    pool = DSparkExpertPool(torch.device("cpu"), 6, 2, 2, num_speculative_tokens=5, num_device_experts=84)
    pool.num_rows = 6
    logits = torch.arange(6 * 256, dtype=torch.float32).reshape(6, 256)
    original = logits.clone()
    router = SimpleNamespace(num_logical_experts=256, e_score_correction_bias=None, routed_scaling_factor=1.5)

    actual_weights, actual_ids = pool.route(router, logits)

    assert actual_weights is weights and actual_ids is ids
    passed_logits, _, bias, raw_scores, votes, _, _ = route.call_args.args
    assert passed_logits is logits
    assert bias.dtype == raw_scores.dtype == torch.float32
    assert votes.dtype == torch.int32
    torch.testing.assert_close(logits, original)
