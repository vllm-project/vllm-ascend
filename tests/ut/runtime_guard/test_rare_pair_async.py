#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

# mypy: ignore-errors
"""UT: async rare_chars pair wait queue + per-step hold (no single-slot race)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.observability.runtime_guard.hooks import AscendAsyncOutput
from vllm_ascend.observability.runtime_guard.processor import RuntimeGuardProcessor
from vllm_ascend.observability.runtime_guard.rare_pair_wait import (
    get_rare_pair_wait_queue,
    reset_rare_pair_wait_queue_for_tests,
)
from vllm_ascend.observability.runtime_guard.state import RequestGuardStore


def test_rare_pair_jobs_block_reap_without_force():
    RequestGuardStore.reset_for_tests()
    try:
        store = RequestGuardStore.get()
        store.max_deferred_waves = 2
        store.get_or_create("r1")
        store.mark_finished(["r1"], wave=1)
        store.add_rare_pair_jobs(["r1"])
        assert store.list_reapable(current_wave=2) == []  # age < cap
        # Past cap → force clear rare_pair_jobs and reap.
        assert store.list_reapable(current_wave=100) == ["r1"]
    finally:
        RequestGuardStore.reset_for_tests()


def test_mark_finished_defers_rare_finish_when_async_pair_enabled():
    RequestGuardStore.reset_for_tests()
    try:
        p = object.__new__(RuntimeGuardProcessor)
        p.wave_tracker = SimpleNamespace(current_wave=lambda: 3)
        p.runtime_config = MagicMock()
        p.runtime_config.detector_get.return_value = True
        p.detectors = MagicMock()
        p.detectors.finish_rare_chars.return_value = []
        p.detectors.get.return_value = None
        p._enable_rare_chars_finish_on_reap()

        p.mark_finished(["r1"])
        p.detectors.finish_rare_chars.assert_not_called()

        store = RequestGuardStore.get()
        assert store.is_request_finished("r1")
        p._reap_finished_requests()
        p.detectors.finish_rare_chars.assert_called_once_with(["r1"])
    finally:
        RequestGuardStore.reset_for_tests()


def test_consume_rare_pairs_async_submits_and_releases_pin():
    RequestGuardStore.reset_for_tests()
    reset_rare_pair_wait_queue_for_tests()
    try:
        p = object.__new__(RuntimeGuardProcessor)
        p.runner = SimpleNamespace()
        p.detectors = MagicMock()
        p.detectors.get.return_value = None
        p.runtime_config = MagicMock()
        p.runtime_config.detector_get.return_value = True
        p.wave_tracker = SimpleNamespace(current_wave=lambda: 1, discard_many=lambda *_: None)
        fed: list[tuple] = []

        def _pack_and_feed(cpu, sampled, req_ids, n_req):
            fed.append((list(req_ids), n_req))

        p._rare_pair_pack_and_feed = _pack_and_feed  # type: ignore[method-assign]
        p._reap_finished_requests = lambda: None  # type: ignore[method-assign]
        materialized: list[object] = []

        def _materialize(hold, collect_event):
            if collect_event is not None:
                collect_event.synchronize()
            cpu = object()
            materialized.append(cpu)
            return cpu

        p._materialize_rare_pair_cpu = _materialize  # type: ignore[method-assign]

        store = RequestGuardStore.get()
        store.get_or_create("a")
        store.mark_finished(["a"], wave=0)

        collect_event = MagicMock()
        pending = (object(), collect_event)
        output = SimpleNamespace(req_ids=["a"], sampled_token_ids=None)
        p.consume_rare_pairs(output, pending=pending)

        q = get_rare_pair_wait_queue()
        q._q.join()
        collect_event.synchronize.assert_called_once()
        assert len(materialized) == 1
        assert fed == [(["a"], 1)]
        assert store.get_state("a") is not None
        assert store.get_state("a").rare_pair_jobs == 0
    finally:
        reset_rare_pair_wait_queue_for_tests()
        RequestGuardStore.reset_for_tests()


def test_per_step_rare_pair_survives_overlapping_sample():
    """Two steps' holds must not share a single processor slot."""
    RequestGuardStore.reset_for_tests()
    try:
        p = object.__new__(RuntimeGuardProcessor)
        p.runner = SimpleNamespace(main_stream=None, runtime_guard=None)
        hold1, ev1 = object(), MagicMock()
        hold2, ev2 = object(), MagicMock()
        p._step_rare_pair = (hold2, ev2)
        got = p.take_step_rare_pair()
        assert got == (hold2, ev2)
        assert p.take_step_rare_pair() is None

        # AscendAsyncOutput owns its pending independently of processor slot.
        consumed: list[object] = []

        def _consume(output, *, pending=None):
            consumed.append(pending)

        p.consume_rare_pairs = _consume  # type: ignore[method-assign]
        p.runner.runtime_guard = p
        inner = MagicMock()
        inner.get_output.return_value = SimpleNamespace(req_ids=["x"], sampled_token_ids=None)
        out_a = AscendAsyncOutput(inner, p.runner, rare_pair=(hold1, ev1))
        out_b = AscendAsyncOutput(inner, p.runner, rare_pair=(hold2, ev2))
        out_a.get_output()
        out_b.get_output()
        assert consumed == [(hold1, ev1), (hold2, ev2)]
    finally:
        RequestGuardStore.reset_for_tests()
