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
"""UT: async rare_chars pair wait queue pins + deferred finish."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.observability.runtime_guard.processor import RuntimeGuardProcessor
from vllm_ascend.observability.runtime_guard.rare_pair_wait import (
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
        # Past the usual cpu/dump defer cap — still blocked while pinned.
        assert store.list_reapable(current_wave=100) == []
        store.finish_rare_pair_jobs(["r1"])
        assert store.list_reapable(current_wave=100) == ["r1"]
    finally:
        RequestGuardStore.reset_for_tests()


def test_mark_finished_defers_rare_finish_when_async_pair_enabled():
    RequestGuardStore.reset_for_tests()
    try:
        p = RuntimeGuardProcessor.__new__(RuntimeGuardProcessor)
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
        # No rare_pair pin → immediately reapable; finish runs at reap.
        p._reap_finished_requests()
        p.detectors.finish_rare_chars.assert_called_once_with(["r1"])
    finally:
        RequestGuardStore.reset_for_tests()


def test_consume_rare_pairs_async_submits_and_releases_pin(monkeypatch):
    RequestGuardStore.reset_for_tests()
    reset_rare_pair_wait_queue_for_tests()
    try:
        import vllm_ascend.envs as envs

        monkeypatch.setattr(envs, "VLLM_ASCEND_RARE_PAIR_SYNC", False)

        p = RuntimeGuardProcessor.__new__(RuntimeGuardProcessor)
        p.runner = SimpleNamespace(_rg_rare_pair_hold=object())
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

        store = RequestGuardStore.get()
        store.get_or_create("a")
        store.mark_finished(["a"], wave=0)

        event = MagicMock()
        cpu = object()
        p._pending_rare_pair = (cpu, event)
        output = SimpleNamespace(req_ids=["a"], sampled_token_ids=None)
        p.consume_rare_pairs(output)

        # Job runs on the waiter thread; drain by stopping the queue.
        from vllm_ascend.observability.runtime_guard.rare_pair_wait import (
            get_rare_pair_wait_queue,
        )

        q = get_rare_pair_wait_queue()
        q._q.join()
        event.synchronize.assert_called_once()
        assert fed == [(["a"], 1)]
        assert store.get_state("a") is not None
        assert store.get_state("a").rare_pair_jobs == 0
    finally:
        reset_rare_pair_wait_queue_for_tests()
        RequestGuardStore.reset_for_tests()
