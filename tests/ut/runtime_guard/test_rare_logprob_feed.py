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
"""UT: async rare_chars logprob feed queue + per-step RareLogprobBatch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.observability.runtime_guard.hooks import AscendAsyncOutput
from vllm_ascend.observability.runtime_guard.processor import RuntimeGuardProcessor
from vllm_ascend.observability.runtime_guard.rare_logprob_feed import (
    RareLogprobBatch,
    get_rare_logprob_feed_queue,
    reset_rare_logprob_feed_queue_for_tests,
)
from vllm_ascend.observability.runtime_guard.state import RequestGuardStore


def test_rare_feed_jobs_block_reap_without_force():
    RequestGuardStore.reset_for_tests()
    try:
        store = RequestGuardStore.get()
        store.max_deferred_waves = 2
        store.get_or_create("r1")
        store.mark_finished(["r1"], wave=1)
        store.add_rare_feed_jobs(["r1"])
        assert store.list_reapable(current_wave=2) == []  # age < cap
        # Past cap → force clear rare_feed_jobs and reap.
        assert store.list_reapable(current_wave=100) == ["r1"]
    finally:
        RequestGuardStore.reset_for_tests()


def test_mark_finished_defers_rare_finish_when_rare_enabled():
    RequestGuardStore.reset_for_tests()
    try:
        p = object.__new__(RuntimeGuardProcessor)
        p.wave_tracker = SimpleNamespace(current_wave=lambda: 3, discard_many=lambda *_: None)
        p.runtime_config = MagicMock()
        p.runtime_config.detector_get.return_value = True
        p.detectors = MagicMock()
        p.detectors.finish_rare_chars.return_value = []
        p.detectors.rare_chars_enabled.return_value = True
        p.detectors.get.return_value = None

        p.mark_finished(["r1"])
        p.detectors.finish_rare_chars.assert_not_called()

        store = RequestGuardStore.get()
        assert store.is_request_finished("r1")
        p._reap_finished_requests()
        p.detectors.finish_rare_chars.assert_called_once_with(["r1"])
    finally:
        RequestGuardStore.reset_for_tests()


def test_feed_rare_logprobs_async_submits_and_releases_pin():
    RequestGuardStore.reset_for_tests()
    reset_rare_logprob_feed_queue_for_tests()
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

        p._pack_and_feed_rare = _pack_and_feed  # type: ignore[method-assign]
        p._reap_finished_requests = lambda: None  # type: ignore[method-assign]
        materialized: list[object] = []

        def _copy(batch):
            if batch.ready_event is not None:
                batch.ready_event.synchronize()
            cpu = object()
            materialized.append(cpu)
            return cpu

        p._copy_rare_batch_to_host = _copy  # type: ignore[method-assign]

        store = RequestGuardStore.get()
        store.get_or_create("a")
        store.mark_finished(["a"], wave=0)

        ready_event = MagicMock()
        batch = RareLogprobBatch(device_logprobs=object(), ready_event=ready_event)
        output = SimpleNamespace(req_ids=["a"], sampled_token_ids=None)
        p.feed_rare_logprobs(output, batch=batch)

        q = get_rare_logprob_feed_queue()
        q._q.join()
        ready_event.synchronize.assert_called_once()
        assert len(materialized) == 1
        assert fed == [(["a"], 1)]
        assert store.get_state("a") is not None
        assert store.get_state("a").rare_feed_jobs == 0
    finally:
        reset_rare_logprob_feed_queue_for_tests()
        RequestGuardStore.reset_for_tests()


def test_per_step_rare_batch_survives_overlapping_sample():
    """Two steps' batches must not share a single processor slot."""
    RequestGuardStore.reset_for_tests()
    try:
        p = object.__new__(RuntimeGuardProcessor)
        p.runner = SimpleNamespace(main_stream=None, runtime_guard=None)
        batch1 = RareLogprobBatch(device_logprobs=object(), ready_event=MagicMock())
        batch2 = RareLogprobBatch(device_logprobs=object(), ready_event=MagicMock())
        p._step_rare_batch = batch2
        got = p.take_step_rare_batch()
        assert got is batch2
        assert p.take_step_rare_batch() is None

        # AscendAsyncOutput owns its batch independently of processor slot.
        consumed: list[object] = []

        def _feed(output, *, batch=None):
            consumed.append(batch)

        p.feed_rare_logprobs = _feed  # type: ignore[method-assign]
        p.runner.runtime_guard = p
        inner = MagicMock()
        inner.get_output.return_value = SimpleNamespace(req_ids=["x"], sampled_token_ids=None)
        out_a = AscendAsyncOutput(inner, p.runner, rare_batch=batch1)
        out_b = AscendAsyncOutput(inner, p.runner, rare_batch=batch2)
        out_a.get_output()
        out_b.get_output()
        assert consumed == [batch1, batch2]
    finally:
        RequestGuardStore.reset_for_tests()
