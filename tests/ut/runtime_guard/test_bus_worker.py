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
"""Unit tests for DueBitsBusWorker + async merged-bus drain."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.observability.runtime_guard.bus_worker import (
    DueBitsBusWorker,
    MergedBusRequest,
)
from vllm_ascend.observability.runtime_guard.processor_bus import RuntimeGuardBusMixin

from ._helpers import attach_bus_worker


def test_bus_worker_runs_due_bits_off_main_thread():
    worker = DueBitsBusWorker(name="ut-bus")
    worker.start()
    try:
        gate = threading.Event()
        seen_thread: list[str] = []

        def _fake_sync(group, bits, *, wave_idx=None):
            seen_thread.append(threading.current_thread().name)
            gate.wait(2.0)
            return [False, False]

        group = SimpleNamespace(world_size=2, is_first_rank=True, rank_in_group=0)
        with (
            patch(
                "vllm_ascend.observability.runtime_config.dist.broadcast_due_bits_from_src",
                side_effect=_fake_sync,
            ),
            patch(
                "vllm_ascend.observability.runtime_config.dist.broadcast_when_due",
                return_value=None,
            ),
        ):
            worker.submit(
                MergedBusRequest(
                    sync_group=group,
                    config_due_local=False,
                    dump_due_local=False,
                    dump_jobs=[],
                    hot_reload_enabled=True,
                    is_first_rank=True,
                )
            )
            assert worker.poll_ready() is False
            gate.set()
            result = worker.wait_result(timeout=2.0)
        assert result is not None
        assert result.error is None
        assert result.config_due is False
        assert seen_thread == ["ut-bus"]
        assert worker.poll_ready() is True
    finally:
        worker.stop()


def test_bus_worker_rejects_second_submit_while_inflight():
    worker = DueBitsBusWorker(name="ut-bus-depth")
    worker.start()
    try:
        gate = threading.Event()

        def _fake_sync(group, bits, *, wave_idx=None):
            gate.wait(2.0)
            return list(bits)

        group = SimpleNamespace(world_size=2, is_first_rank=True)
        with (
            patch(
                "vllm_ascend.observability.runtime_config.dist.broadcast_due_bits_from_src",
                side_effect=_fake_sync,
            ),
            patch(
                "vllm_ascend.observability.runtime_config.dist.broadcast_when_due",
                return_value=None,
            ),
        ):
            req = MergedBusRequest(
                sync_group=group,
                config_due_local=True,
                dump_due_local=False,
                dump_jobs=[],
                hot_reload_enabled=False,
                is_first_rank=True,
            )
            worker.submit(req)
            with pytest.raises(RuntimeError, match="queue depth is 1"):
                worker.submit(req)
            gate.set()
            worker.wait_result(timeout=2.0)
    finally:
        worker.stop()


def test_async_merged_bus_submit_then_drain_applies_config():
    """Wave-head submit + end-of-wave drain mirrors sync semantics."""
    from vllm_ascend.observability.runtime_guard.bus_worker import DueBitsBusWorker

    cfg = MagicMock()
    cfg.hot_reload_enabled = True
    cfg.config_due_local.return_value = True
    cfg.build_config_sync_payload.return_value = ({"version": 1.0, "data": {}}, True)
    cfg.apply_config_sync_payload.return_value = True

    worker = DueBitsBusWorker(name="ut-bus-drain")
    worker.start()
    proc = SimpleNamespace(
        runner=SimpleNamespace(),
        runtime_config=cfg,
        _kv_dump_jobs=[],
        _deferred_kv_dump_jobs=[],
        _apply_config_cascade=MagicMock(),
        needs_sample_phase_hooks=lambda: False,
        _bus_worker=worker,
        _merged_bus_inflight=False,
        _pending_merged_bus_dump_jobs=[],
        _pending_merged_bus_can_dump=False,
        _pending_merged_bus_is_first=False,
        _pending_prev_needs_sample_hooks=None,
        _bus_wave_seq=0,
        _merged_bus_warn_ts=0.0,
        _refund_dropped_dump_arms=MagicMock(),
        _drop_pending_dump_jobs=MagicMock(),
    )
    # Bind mixin methods.
    proc._prepare_merged_bus_locals = RuntimeGuardBusMixin._prepare_merged_bus_locals.__get__(proc)
    proc._wave_head_merged_bus = RuntimeGuardBusMixin._wave_head_merged_bus.__get__(proc)
    proc._drain_merged_bus = RuntimeGuardBusMixin._drain_merged_bus.__get__(proc)
    proc._apply_merged_bus_result = RuntimeGuardBusMixin._apply_merged_bus_result.__get__(proc)

    group = MagicMock()
    group.world_size = 2
    group.is_first_rank = True
    group.rank_in_group = 0
    group.cpu_group = object()
    group.broadcast_object = MagicMock(side_effect=lambda obj, src=0: obj)

    try:
        with (
            patch(
                "vllm_ascend.observability.runtime_guard.processor_bus.should_dump_kv_on_rank",
                return_value=False,
            ),
            patch("torch.distributed.broadcast"),
            patch("torch.distributed.get_process_group_ranks", return_value=[7, 8]),
        ):
            assert proc._wave_head_merged_bus(group) is False
            assert proc._merged_bus_inflight is True
            # Allow the daemon a tick to finish AR+bcast.
            deadline = time.time() + 2.0
            while time.time() < deadline and not worker.poll_ready():
                time.sleep(0.01)
            changed = proc._drain_merged_bus(warn_if_pending=False)
        assert changed is True
        cfg.apply_config_sync_payload.assert_called_once()
        proc._apply_config_cascade.assert_called_once()
        assert proc._merged_bus_inflight is False
    finally:
        worker.stop()


def test_async_merged_bus_warns_when_drain_waits():
    import logging

    from vllm_ascend.observability.runtime_guard.bus_worker import DueBitsBusWorker

    from ._helpers import capture_logger_text

    cfg = MagicMock()
    cfg.hot_reload_enabled = False
    cfg.config_due_local.return_value = False

    worker = DueBitsBusWorker(name="ut-bus-warn")
    worker.start()
    proc = SimpleNamespace(
        runner=SimpleNamespace(),
        runtime_config=cfg,
        _kv_dump_jobs=[],
        _deferred_kv_dump_jobs=[],
        _apply_config_cascade=MagicMock(),
        needs_sample_phase_hooks=lambda: False,
        _bus_worker=worker,
        _merged_bus_inflight=False,
        _pending_merged_bus_dump_jobs=[],
        _pending_merged_bus_can_dump=False,
        _pending_merged_bus_is_first=False,
        _pending_prev_needs_sample_hooks=None,
        _bus_wave_seq=0,
        _merged_bus_warn_ts=0.0,
        _refund_dropped_dump_arms=MagicMock(),
        _drop_pending_dump_jobs=MagicMock(),
    )
    proc._prepare_merged_bus_locals = RuntimeGuardBusMixin._prepare_merged_bus_locals.__get__(proc)
    proc._wave_head_merged_bus = RuntimeGuardBusMixin._wave_head_merged_bus.__get__(proc)
    proc._drain_merged_bus = RuntimeGuardBusMixin._drain_merged_bus.__get__(proc)
    proc._apply_merged_bus_result = RuntimeGuardBusMixin._apply_merged_bus_result.__get__(proc)

    gate = threading.Event()
    group = MagicMock()
    group.world_size = 2
    group.is_first_rank = True
    group.rank_in_group = 0

    def _slow_sync(g, bits, *, wave_idx=None):
        gate.wait(2.0)
        return [False, False]

    try:
        with (
            patch(
                "vllm_ascend.observability.runtime_guard.processor_bus.should_dump_kv_on_rank",
                return_value=False,
            ),
            patch(
                "vllm_ascend.observability.runtime_config.dist.broadcast_due_bits_from_src",
                side_effect=_slow_sync,
            ),
            patch(
                "vllm_ascend.observability.runtime_config.dist.broadcast_when_due",
                return_value=None,
            ),
            capture_logger_text(
                "vllm_ascend.observability.runtime_guard.processor_bus",
                level=logging.WARNING,
            ) as buf,
        ):
            assert proc._wave_head_merged_bus(group) is False

            # Drain in a helper thread and hold the gate until the pending
            # warning has actually been captured. Releasing the gate on a
            # fixed sleep races CI scheduling: if the drain arrives after the
            # worker already finished (slipcover + xdist starve the thread
            # past the sleep window), no warning fires at all.
            drain_errors = []

            def _drain():
                try:
                    proc._drain_merged_bus(warn_if_pending=True)
                except Exception as exc:  # pragma: no cover - worker fault path
                    drain_errors.append(exc)

            drain_thread = threading.Thread(target=_drain, daemon=True)
            drain_thread.start()
            deadline = time.monotonic() + 5.0
            while "merged bus not finished before end-of-wave" not in buf.getvalue():
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
            gate.set()
            drain_thread.join(10.0)
            assert not drain_thread.is_alive(), "drain thread did not finish after the gate opened"
            assert not drain_errors, drain_errors
        assert "merged bus not finished before end-of-wave" in buf.getvalue()
    finally:
        gate.set()
        worker.stop()


def test_wave_head_submit_failure_refunds_handed_off_jobs():
    """Shutdown racing a wave head: submit refuses → arms refunded, no inflight."""
    job = {"req_id": "r1", "consume_quota": True, "arm_id": "arm-1"}
    refunded: list[list] = []
    cfg = MagicMock()
    cfg.hot_reload_enabled = True
    cfg.config_due_local.return_value = False

    proc = SimpleNamespace(
        runner=SimpleNamespace(),
        runtime_config=cfg,
        _kv_dump_jobs=[job],
        _deferred_kv_dump_jobs=[],
        _apply_config_cascade=MagicMock(),
        needs_sample_phase_hooks=lambda: False,
        _refund_dropped_dump_arms=lambda jobs: refunded.append(list(jobs)),
        _drop_pending_dump_jobs=MagicMock(),
    )
    worker = attach_bus_worker(proc, name="ut-bus-submit-fail")
    try:
        # Simulate shutdown racing this wave head: submit() now refuses.
        worker._stopping = True
        group = SimpleNamespace(world_size=2, is_first_rank=True, rank_in_group=0)
        with (
            patch(
                "vllm_ascend.observability.runtime_guard.processor_bus.should_dump_kv_on_rank",
                return_value=True,
            ),
            patch(
                "vllm_ascend.observability.runtime_config.dist.broadcast_due_bits_from_src",
                return_value=[False, False],
            ),
            patch(
                "vllm_ascend.observability.runtime_config.dist.broadcast_when_due",
                return_value=None,
            ),
            pytest.raises(RuntimeError, match="not started"),
        ):
            RuntimeGuardBusMixin._wave_head_merged_bus(proc, group)
        # Jobs were cleared for the bus; submit refused → refund their arms.
        assert refunded == [[job]]
        assert proc._kv_dump_jobs == []
        assert proc._pending_merged_bus_dump_jobs == []
        assert proc._pending_merged_bus_is_first is False
        assert proc._merged_bus_inflight is False
    finally:
        worker.stop()


def test_drain_timeout_refunds_pending_jobs_and_clears_state():
    """Bounded drain (teardown): TimeoutError → refund pending, reset, re-raise."""
    job = {"req_id": "r1", "consume_quota": True, "arm_id": "arm-1"}
    refunded: list[list] = []
    cfg = MagicMock()
    cfg.hot_reload_enabled = True
    cfg.config_due_local.return_value = False

    proc = SimpleNamespace(
        runner=SimpleNamespace(),
        runtime_config=cfg,
        _kv_dump_jobs=[job],
        _deferred_kv_dump_jobs=[],
        _apply_config_cascade=MagicMock(),
        needs_sample_phase_hooks=lambda: False,
        _refund_dropped_dump_arms=lambda jobs: refunded.append(list(jobs)),
        _drop_pending_dump_jobs=MagicMock(),
    )
    worker = attach_bus_worker(proc, name="ut-bus-drain-timeout")
    gate = threading.Event()
    group = SimpleNamespace(world_size=2, is_first_rank=True, rank_in_group=0)

    def _stuck_sync(g, bits, *, wave_idx=None):
        gate.wait(5.0)
        return [False, False]

    try:
        with (
            patch(
                "vllm_ascend.observability.runtime_guard.processor_bus.should_dump_kv_on_rank",
                return_value=True,
            ),
            patch(
                "vllm_ascend.observability.runtime_config.dist.broadcast_due_bits_from_src",
                side_effect=_stuck_sync,
            ),
            patch(
                "vllm_ascend.observability.runtime_config.dist.broadcast_when_due",
                return_value=None,
            ),
        ):
            assert RuntimeGuardBusMixin._wave_head_merged_bus(proc, group) is False
            assert proc._merged_bus_inflight is True
            assert proc._pending_merged_bus_dump_jobs == [job]
            with pytest.raises(TimeoutError):
                RuntimeGuardBusMixin._drain_merged_bus(proc, timeout=0.05)
        # Timeout path: pending handed-off jobs refunded, all state cleared.
        assert refunded == [[job]]
        assert proc._merged_bus_inflight is False
        assert proc._pending_merged_bus_dump_jobs == []
        assert proc._pending_merged_bus_is_first is False
    finally:
        gate.set()
        worker.stop()


def test_run_sample_phase_drains_bus_before_sample_fn():
    """Sample entry must drain TP Gloo bus before connector post_forward/SFA.

    Wave-head submit overlaps forward; draining at ``run_sample_phase`` entry
    keeps that overlap while preventing bus broadcast from crossing
    ``all_gather_object`` inside ``sample_fn`` / ``get_finished``.
    """
    from vllm_ascend.observability.runtime_guard.processor import (
        RuntimeGuardProcessor,
        SamplePhaseResult,
    )

    order: list[str] = []
    p = object.__new__(RuntimeGuardProcessor)
    p.runtime_config = MagicMock()
    p.runtime_config.is_fully_off.return_value = False
    p.detectors = MagicMock()
    p.wave_tracker = MagicMock()
    p.runner = MagicMock()
    p._last_input_batch = None
    p._merged_bus_inflight = True

    def _drain(*, warn_if_pending: bool = False, timeout: float | None = None):
        order.append("drain")
        p._merged_bus_inflight = False
        return True

    def sample_fn():
        order.append("sample")
        assert p._merged_bus_inflight is False
        return SamplePhaseResult(
            scheduler_output=None,
            input_batch=None,
            model_runner_output=None,
            sampler_output=SimpleNamespace(sampled_token_ids=[1]),
            valid_sampled_token_ids=[1],
            req_ids_output_copy=["r1"],
            finished_req_ids=None,
        )

    p._drain_merged_bus = _drain  # type: ignore[method-assign]
    p._drain_merged_bus_before_sample_tp_gloo = (  # type: ignore[method-assign]
        RuntimeGuardBusMixin._drain_merged_bus_before_sample_tp_gloo.__get__(p)
    )
    p.mark_finished = MagicMock()
    p.needs_sample_phase_hooks = lambda: False  # type: ignore[method-assign]
    p.end_of_wave = MagicMock()  # type: ignore[method-assign]

    p.run_sample_phase(sample_fn=sample_fn, speculative_config=None, use_async=False)
    assert order == ["drain", "sample"]
    p.end_of_wave.assert_called_once()


def test_tp0_drain_uses_pre_reload_hooks_for_soft_purge():
    """TP0 bus-worker reload must not make drain see already-False hooks.

    Leader ``build_config_sync_payload`` applies JSON on the bus thread before
    end-of-wave drain. Capture ``needs_sample_phase_hooks`` at submit so a
    True→False disable still soft-purges, then a later re-enable starts clean.
    """
    from vllm_ascend.observability.runtime_guard.bus_worker import DueBitsBusWorker
    from vllm_ascend.observability.runtime_guard.state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    try:
        store = RequestGuardStore.get()
        store.get_or_create("r1")
        store.set_prompt_token_ids("r1", [1, 2, 3])
        store.append_output_ids("r1", [9, 9])

        hooks_on = True

        def _hooks() -> bool:
            return hooks_on

        def _build_payload():
            # Simulate TP0 reload flipping gates before drain.
            nonlocal hooks_on
            hooks_on = False
            return ({"version": 2.0, "data": {"detector": {}}}, True)

        cfg = MagicMock()
        cfg.hot_reload_enabled = True
        cfg.config_due_local.return_value = True
        cfg.build_config_sync_payload.side_effect = _build_payload
        cfg.apply_config_sync_payload.return_value = True
        cfg.report_save_sensitive_info.return_value = False
        cfg.report_max_prompt_token_ids.return_value = 0
        cfg.report_max_output_token_ids.return_value = 0
        cfg.report_decode_token_ids.return_value = False
        cfg.report_max_per_req.return_value = 0
        cfg.apply_ascend_log_level = MagicMock()

        worker = DueBitsBusWorker(name="ut-bus-tp0-purge")
        worker.start()
        proc = SimpleNamespace(
            runner=SimpleNamespace(),
            runtime_config=cfg,
            detectors=MagicMock(),
            action_executor=MagicMock(),
            report_writer=MagicMock(),
            wave_tracker=MagicMock(),
            _kv_dump_jobs=[],
            _deferred_kv_dump_jobs=[],
            needs_sample_phase_hooks=_hooks,
            _bus_worker=worker,
            _merged_bus_inflight=False,
            _pending_merged_bus_dump_jobs=[],
            _pending_merged_bus_can_dump=False,
            _pending_merged_bus_is_first=False,
            _pending_prev_needs_sample_hooks=None,
            _bus_wave_seq=0,
            _merged_bus_warn_ts=0.0,
            _refund_dropped_dump_arms=MagicMock(),
            _drop_pending_dump_jobs=MagicMock(),
        )
        proc._apply_config_cascade = RuntimeGuardBusMixin._apply_config_cascade.__get__(proc)
        proc._purge_guard_state_after_detectors_off = (
            RuntimeGuardBusMixin._purge_guard_state_after_detectors_off.__get__(proc)
        )
        proc._prepare_merged_bus_locals = RuntimeGuardBusMixin._prepare_merged_bus_locals.__get__(proc)
        proc._wave_head_merged_bus = RuntimeGuardBusMixin._wave_head_merged_bus.__get__(proc)
        proc._drain_merged_bus = RuntimeGuardBusMixin._drain_merged_bus.__get__(proc)
        proc._apply_merged_bus_result = RuntimeGuardBusMixin._apply_merged_bus_result.__get__(proc)

        group = MagicMock()
        group.world_size = 2
        group.is_first_rank = True
        group.rank_in_group = 0
        group.cpu_group = object()
        group.broadcast_object = MagicMock(side_effect=lambda obj, src=0: obj)

        try:
            with (
                patch(
                    "vllm_ascend.observability.runtime_guard.processor_bus.should_dump_kv_on_rank",
                    return_value=False,
                ),
                patch("torch.distributed.broadcast"),
                patch("torch.distributed.get_process_group_ranks", return_value=[0, 1]),
            ):
                assert proc._wave_head_merged_bus(group) is False
                assert proc._pending_prev_needs_sample_hooks is True
                deadline = time.time() + 2.0
                while time.time() < deadline and not worker.poll_ready():
                    time.sleep(0.01)
                # Gates already False (TP0 reload), but snapshot keeps soft-purge.
                assert hooks_on is False
                assert proc._drain_merged_bus(warn_if_pending=False) is True

            st = store.get_state("r1")
            assert st is not None
            assert st.prompt_token_ids == [1, 2, 3]
            assert st.output_token_ids == []
            proc.detectors.disable_all.assert_called_once()

            # Re-enable: cumulative IO starts from empty suffix (mid-enable contract).
            hooks_on = True
            store.append_output_ids("r1", [42])
            st2 = store.get_state("r1")
            assert st2 is not None
            assert st2.prompt_token_ids == [1, 2, 3]
            assert st2.output_token_ids == [42]
        finally:
            worker.stop()
    finally:
        RequestGuardStore.reset_for_tests()
