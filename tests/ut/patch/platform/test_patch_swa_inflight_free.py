#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for the PR #47728 runtime backport (``patch_swa_inflight_free``).

The backport frees sliding-window blocks on the *processed-token* basis under
async scheduling, so an in-flight step's attention window is not recycled and
overwritten before it finishes reading (a load-WAR that collapsed spec-decode
acceptance length). These tests pin the accounting arithmetic without touching
real vLLM Scheduler/Coordinator internals.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.request import Request

import vllm_ascend.patch.platform.patch_swa_inflight_free as swa

# The whole backport is inert once vLLM ships `Request.num_in_flight_tokens`
# (PR #47728); there is nothing to test in that case.
pytestmark = pytest.mark.skipif(
    hasattr(Request, "num_in_flight_tokens"),
    reason="PR #47728 backport is a no-op once vLLM ships Request.num_in_flight_tokens",
)


class _SWABackportTestBase:
    """Shared helpers for the SWA in-flight-free backport tests."""

    @staticmethod
    def _scheduler_output(num_scheduled_tokens: dict[str, int]) -> SimpleNamespace:
        return SimpleNamespace(num_scheduled_tokens=num_scheduled_tokens)

    @staticmethod
    def _reset_inflight() -> None:
        swa._inflight.clear()

    @staticmethod
    def _record_free_boundary(monkeypatch) -> list[int]:
        """Record the ``processed`` token count handed to the real free path."""
        freed: list[int] = []
        monkeypatch.setattr(
            swa,
            "_orig_remove_skipped_blocks",
            lambda self, request_id, total, num_prompt_tokens=None: freed.append(total),
        )
        return freed


class TestRemoveSkippedBlocksOnProcessedBasis(_SWABackportTestBase):
    """Guard: ``remove_skipped_blocks`` must free on the processed-token basis
    (``total - in_flight``), never the optimistic ``total`` — the root cause of
    the spec-decode acceptance-length regression under async scheduling."""

    def test_subtracts_in_flight_before_freeing(self, monkeypatch):
        """An in-flight step shifts the free boundary back by its token count."""
        self._reset_inflight()
        swa._inflight["req-0"] = 6  # one in-flight spec step (1 bonus + 5 spec)
        freed = self._record_free_boundary(monkeypatch)
        swa._remove_skipped_blocks(object(), "req-0", 100, num_prompt_tokens=2048)
        assert freed == [94]  # 100 - 6: free only the committed prefix

    def test_clamps_to_zero_when_in_flight_exceeds_total(self, monkeypatch):
        """Rejected spec tokens can roll ``total`` below ``in_flight``; the
        processed basis must clamp to 0, not go negative."""
        self._reset_inflight()
        swa._inflight["req-0"] = 11
        freed = self._record_free_boundary(monkeypatch)
        swa._remove_skipped_blocks(object(), "req-0", 5)
        assert freed == [0]

    def test_uses_committed_basis_when_no_in_flight(self, monkeypatch):
        """First schedule (no in-flight step) frees on the full total."""
        self._reset_inflight()
        freed = self._record_free_boundary(monkeypatch)
        swa._remove_skipped_blocks(object(), "req-0", 50)
        assert freed == [50]


class TestInFlightAccounting(_SWABackportTestBase):
    """Guard: ``_inflight`` is incremented in ``_update_after_schedule`` and
    decremented (popped at zero) in ``update_from_output``."""

    def test_increment_after_schedule(self, monkeypatch):
        self._reset_inflight()
        monkeypatch.setattr(swa, "_orig_update_after_schedule", lambda *a, **k: None)
        swa._update_after_schedule(object(), self._scheduler_output({"req-0": 6, "req-1": 1}))
        assert swa._inflight == {"req-0": 6, "req-1": 1}

    def test_accumulates_across_steps(self, monkeypatch):
        self._reset_inflight()
        monkeypatch.setattr(swa, "_orig_update_after_schedule", lambda *a, **k: None)
        swa._update_after_schedule(object(), self._scheduler_output({"req-0": 6}))
        swa._update_after_schedule(object(), self._scheduler_output({"req-0": 6}))
        assert swa._inflight == {"req-0": 12}

    def test_decrement_pops_at_zero(self, monkeypatch):
        self._reset_inflight()
        swa._inflight["req-0"] = 6
        monkeypatch.setattr(swa, "_orig_update_from_output", lambda *a, **k: None)
        swa._update_from_output(object(), self._scheduler_output({"req-0": 6}), None)
        assert "req-0" not in swa._inflight

    def test_decrement_keeps_positive_remainder(self, monkeypatch):
        """Two in-flight steps (12) settled by one (6) leaves 6, not popped."""
        self._reset_inflight()
        swa._inflight["req-0"] = 12
        monkeypatch.setattr(swa, "_orig_update_from_output", lambda *a, **k: None)
        swa._update_from_output(object(), self._scheduler_output({"req-0": 6}), None)
        assert swa._inflight == {"req-0": 6}


class TestSpecDecodeArithmetic(_SWABackportTestBase):
    """End-to-end guards for the in-flight accounting under spec decoding."""

    def test_async_in_flight_step_does_not_advance_free_boundary(self, monkeypatch):
        """Under async scheduling the scheduler runs a step ahead: when step B
        is scheduled while step A is still in flight, ``allocate_slots`` must
        free on A's committed basis (``total - in_flight``), not the optimistic
        ``total`` — otherwise A's SWA window blocks are freed, recycled, and
        overwritten before A finishes reading them (the load-WAR that collapsed
        spec-decode acceptance length)."""
        self._reset_inflight()
        monkeypatch.setattr(swa, "_orig_update_after_schedule", lambda *a, **k: None)
        monkeypatch.setattr(swa, "_orig_update_from_output", lambda *a, **k: None)
        freed = self._record_free_boundary(monkeypatch)
        rid = "req-0"
        committed = 0

        # Step A schedules 6 tokens (1 bonus + 5 spec); allocate_slots runs
        # before _update_after_schedule, so in_flight is still 0 here.
        swa._remove_skipped_blocks(object(), rid, committed)
        assert freed[-1] == 0
        swa._update_after_schedule(object(), self._scheduler_output({rid: 6}))
        total_opt = committed + 6  # vLLM counts A optimistically

        # Step B is scheduled while A is still in flight (async look-ahead).
        swa._remove_skipped_blocks(object(), rid, total_opt)
        # Free boundary lags by A's 6 in-flight tokens -> still committed (0),
        # NOT total_opt (6): A's window blocks must survive until A settles.
        assert freed[-1] == committed
        assert freed[-1] != total_opt

        # A settles (4 accepted, 2 rejected); patch drops A's in-flight step.
        swa._update_from_output(object(), self._scheduler_output({rid: 6}), None)
        committed += 4
        assert swa._inflight.get(rid, 0) == 0

    @pytest.mark.parametrize("accepted", [0, 1, 3, 4, 5])
    def test_processed_equals_committed_across_settled_steps(self, monkeypatch, accepted):
        """Across a schedule/settle sequence the processed position handed to
        ``remove_skipped_blocks`` always equals the committed position, and
        ``_inflight`` leaves no leak — the property PR #47728 restores, holding
        for any spec acceptance rate (0..num_spec)."""
        self._reset_inflight()
        monkeypatch.setattr(swa, "_orig_update_after_schedule", lambda *a, **k: None)
        monkeypatch.setattr(swa, "_orig_update_from_output", lambda *a, **k: None)
        freed = self._record_free_boundary(monkeypatch)
        rid = "req-0"
        committed = 0
        total_opt = 0
        scheduled = 6  # 1 bonus + 5 spec
        for _ in range(4):
            # allocate_slots() runs inside schedule() before the in-flight bump.
            swa._remove_skipped_blocks(object(), rid, total_opt)
            assert freed[-1] == committed  # processed == committed
            swa._update_after_schedule(object(), self._scheduler_output({rid: scheduled}))
            total_opt += scheduled  # optimistic
            swa._update_from_output(object(), self._scheduler_output({rid: scheduled}), None)
            committed += accepted  # vLLM commits the accepted tokens after rollback
            total_opt = committed  # optimistic collapses to committed after settle
        assert swa._inflight == {}  # no in-flight leak
