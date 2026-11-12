# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.dspark_expert_pool import (
    DSparkExpertPool,
    build_roles,
    has_short_drafts,
)


@pytest.mark.parametrize(
    ("boundaries", "drafts", "prefilling", "padded_rows", "expected"),
    [
        ([0, 6], [5], [False], 6, [1, 1, 2, 2, 2, 2]),
        ([0, 6, 12], [5, 5], [False, False], 12, [1, 1, 2, 2, 2, 2] * 2),
        ([0, 6], [5], [False], 12, [1, 1, 2, 2, 2, 2] + [0] * 6),
        ([0, 6, 12], [5, 5], [True, False], 12, [0] * 6 + [1, 1, 2, 2, 2, 2]),
        ([0, 2, 6], [1, 3], [False, False], 6, [1, 1, 1, 1, 2, 2]),
        ([0, 1], None, [False], 1, [0]),
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
    ("configured_drafts", "drafts", "expected"),
    [
        (5, [5], False),
        (5, [5, 5], False),
        (5, [5, 5, 5], False),
        (5, [5, 1], True),
        (5, [1], True),
        (5, None, False),
        (7, [5, 5], True),
    ],
)
def test_short_draft_steps_run_eager(configured_drafts, drafts, expected):
    state = SimpleNamespace(num_draft_tokens_np=drafts)
    assert has_short_drafts(state, num_speculative_tokens=configured_drafts) is expected


def make_batch(boundaries, drafts, padded_rows, *, structured=False):
    return SimpleNamespace(
        num_reqs=len(boundaries) - 1,
        num_tokens=boundaries[-1],
        num_tokens_after_padding=padded_rows,
        query_start_loc_np=np.array(boundaries),
        num_draft_tokens_per_req=drafts,
        is_prefilling_np=[False] * (len(boundaries) - 1),
        has_structured_output_reqs=structured,
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
    # Inactive steps still stage, so a replayed graph cannot read stale roles.
    assert pool.roles[:1].tolist() == [0]


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
    # Same width, inactive: the staged roles must be fully cleared so a
    # replayed graph routes every row with the native outputs.
    assert pool.roles.tolist() == [0] * 6


def test_mixed_step_preserves_native_outputs_only_outside_verification():
    pool = DSparkExpertPool(
        torch.device("cpu"),
        capacity=7,
        protected_positions=2,
        suffix_experts=1,
        num_speculative_tokens=5,
        num_device_experts=84,
    )
    pool.prepare(make_batch([0, 1, 7], [0, 5], 7))
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
