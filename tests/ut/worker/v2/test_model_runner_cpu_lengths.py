# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Unit tests for the CPU length mirrors used by MRv2 input preparation.

``_update_seq_lens_cpu`` writes two CPU buffers per step. Both are pinned CPU
memory and the previous implementation touched them element by element through
torch, which cost ~8 us per access. These tests pin the vectorized behavior to
the old loop semantics so the rewrite cannot change which rows are updated.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.model_runner import NPUModelRunner, count_draft_tokens_per_req


def _make_runner(
    max_num_reqs: int,
    req_id_to_index: dict[str, int],
    d2h_values: list[int] | None = None,
    skip_sync: bool = False,
):
    num_computed_tokens_cpu = torch.from_numpy(np.zeros(max_num_reqs, dtype=np.int32))
    seq_lens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)
    # The D2H staging buffer only holds valid data once the event sync
    # completes, so publish it from synchronize(). A gather that ran before
    # the wait would read zeros and fail the assertions.
    d2h_staging = torch.zeros(max_num_reqs, dtype=torch.int32)
    event = MagicMock()
    if d2h_values is not None:
        published = torch.tensor(d2h_values, dtype=torch.int32)
        event.synchronize.side_effect = lambda: d2h_staging.copy_(published)
    runner = SimpleNamespace(
        req_states=SimpleNamespace(
            req_id_to_index=req_id_to_index,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
        ),
        num_computed_tokens_cpu=d2h_staging,
        input_buffers=SimpleNamespace(
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_np=seq_lens_cpu.numpy(),
        ),
        num_computed_tokens_event=event,
        speculator=object(),
        sync_spec_pp_cpu_counts=False,
        _can_skip_spec_cpu_length_sync=lambda: skip_sync,
    )
    return runner


def _make_scheduler_output(
    num_scheduled_tokens: dict[str, int],
    cached_req_ids: list[str] | None = None,
):
    return SimpleNamespace(
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_cached_reqs=SimpleNamespace(req_ids=cached_req_ids or []),
    )


def _reference_update(runner, scheduler_output, req_ids, apply_cached):
    """The pre-rewrite loop implementation, kept as the oracle."""
    if apply_cached:
        # The old loop also ran after the D2H wait.
        runner.num_computed_tokens_event.synchronize()
        for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
            req_index = runner.req_states.req_id_to_index[req_id]
            runner.req_states.num_computed_tokens_cpu[req_index] = runner.num_computed_tokens_cpu[req_index]
    for i, req_id in enumerate(req_ids):
        req_index = runner.req_states.req_id_to_index[req_id]
        num_computed_tokens = runner.req_states.num_computed_tokens_cpu[req_index]
        runner.input_buffers.seq_lens_cpu[i] = num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]


def test_update_seq_lens_matches_loop_semantics():
    """Cached rows are published by the wait; every scheduled row gets a seq_lens."""
    mapping = {"a": 0, "b": 1, "c": 2, "d": 3}
    d2h = [10, 20, 30, 40, 0, 0]
    scheduler_output = _make_scheduler_output({"b": 3, "a": 3, "d": 1}, cached_req_ids=["a", "c"])
    req_ids = ["b", "a", "d"]

    runner = _make_runner(6, mapping, d2h)
    NPUModelRunner._update_seq_lens_cpu(runner, scheduler_output, req_ids)

    # Exactly one wait, and the cached rows only carry the snapshot because
    # it is published inside synchronize().
    runner.num_computed_tokens_event.synchronize.assert_called_once()
    # Cached requests refreshed from the D2H snapshot; b is not cached so its
    # mirror keeps the old value even though the D2H buffer carries one.
    assert runner.req_states.num_computed_tokens_cpu.tolist()[:4] == [10, 0, 30, 0]
    # seq_lens = mirror + scheduled tokens, in batch order.
    assert runner.input_buffers.seq_lens_cpu.tolist() == [3, 13, 1, 0, 0, 0]


def test_update_seq_lens_keeps_padded_tail():
    """Rows past len(req_ids) belong to padding and must stay untouched."""
    mapping = {"a": 0, "b": 1}
    runner = _make_runner(4, mapping, [5, 6, 0, 0])
    runner.input_buffers.seq_lens_cpu[2:] = 99
    scheduler_output = _make_scheduler_output({"a": 2, "b": 4}, cached_req_ids=["a", "b"])
    NPUModelRunner._update_seq_lens_cpu(runner, scheduler_output, ["a", "b"])

    assert runner.input_buffers.seq_lens_cpu.tolist() == [7, 10, 99, 99]


def test_update_seq_lens_skip_sync_keeps_mirror_rows():
    """Skipping the sync must not consume the stale D2H snapshot."""
    mapping = {"a": 0, "b": 1}
    runner = _make_runner(2, mapping, [100, 100], skip_sync=True)
    scheduler_output = _make_scheduler_output({"a": 1, "b": 1}, cached_req_ids=["a"])

    NPUModelRunner._update_seq_lens_cpu(runner, scheduler_output, ["a", "b"])

    assert runner.req_states.num_computed_tokens_cpu.tolist() == [0, 0]
    assert runner.input_buffers.seq_lens_cpu.tolist() == [1, 1]
    runner.num_computed_tokens_event.synchronize.assert_not_called()


def test_update_seq_lens_empty_cached_list():
    mapping = {"a": 0}
    runner = _make_runner(2, mapping, [7, 0])
    NPUModelRunner._update_seq_lens_cpu(runner, _make_scheduler_output({"a": 5}), ["a"])

    assert runner.input_buffers.seq_lens_cpu.tolist() == [5, 0]


@pytest.mark.parametrize("seed", range(10))
def test_update_seq_lens_randomized_parity(seed):
    """Vectorized update must equal the old loops for arbitrary batches."""
    rng = np.random.default_rng(seed)
    num_reqs = int(rng.integers(1, 9))
    mapping = {f"r{i}": i for i in range(num_reqs)}
    d2h = rng.integers(0, 512, size=num_reqs).tolist()
    cached = [f"r{i}" for i in range(num_reqs) if rng.random() < 0.5]
    req_ids = [f"r{i}" for i in rng.permutation(num_reqs)]
    num_scheduled = {req_id: int(rng.integers(1, 5)) for req_id in mapping}
    scheduler_output = _make_scheduler_output(num_scheduled, cached_req_ids=cached)

    vectorized = _make_runner(num_reqs, dict(mapping), d2h)
    NPUModelRunner._update_seq_lens_cpu(vectorized, scheduler_output, req_ids)

    reference = _make_runner(num_reqs, dict(mapping), d2h)
    _reference_update(reference, scheduler_output, req_ids, apply_cached=True)

    np.testing.assert_array_equal(
        vectorized.req_states.num_computed_tokens_cpu.numpy(),
        reference.req_states.num_computed_tokens_cpu.numpy(),
    )
    np.testing.assert_array_equal(
        vectorized.input_buffers.seq_lens_np,
        reference.input_buffers.seq_lens_cpu.numpy(),
    )


def test_count_draft_tokens_per_req_follows_batch_order():
    draft_tokens = {"a": [1, 2], "c": [3]}
    assert count_draft_tokens_per_req(draft_tokens, ["c", "b", "a"]).tolist() == [1, 0, 2]


@pytest.mark.parametrize("seed", range(5))
def test_num_valid_tokens_matches_legacy_comprehension(seed):
    """num_valid = scheduled - draft counts, as the removed comprehension did."""
    rng = np.random.default_rng(100 + seed)
    num_reqs = int(rng.integers(1, 8))
    req_ids = [f"r{i}" for i in range(num_reqs)]
    scheduled = rng.integers(1, 6, size=num_reqs)
    draft_tokens = {req_id: [0] * int(rng.integers(0, 3)) for req_id in req_ids}
    draft_tokens = {req_id: tokens for req_id, tokens in draft_tokens.items() if tokens}

    counts = count_draft_tokens_per_req(draft_tokens, req_ids)
    vectorized = (scheduled - counts).astype(np.int32, copy=False)
    legacy = np.array(
        [num_toks - len(draft_tokens.get(req_id, [])) for num_toks, req_id in zip(scheduled, req_ids)],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(vectorized, legacy)
