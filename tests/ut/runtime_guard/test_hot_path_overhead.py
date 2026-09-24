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
"""Correctness UTs for hot-path gates / idle sync (no wall-clock bounds)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.runtime_config.config import RuntimeConfig
from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor, SamplePhaseResult


def _cfg(tmp_path: Path, *, reload: float) -> RuntimeConfig:
    path = tmp_path / "runtime_config.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    return RuntimeConfig(
        config_path=path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=reload,
        sync_mode="file",
    )


def _bind(cfg: RuntimeConfig, *, runner: MagicMock | None = None) -> RuntimeGuardProcessor:
    """Bind a Processor with Magicked deps (avoids repeating _init blocks)."""
    RuntimeGuardProcessor.reset_for_tests()
    if runner is None:
        runner = MagicMock()
        runner.tp_rank = 0
        runner.dp_rank = 0

    def _init(self, r):
        self.runner = r
        self.runtime_config = cfg
        self.manual_triggers = MagicMock()
        self.manual_triggers.consume_once.return_value = None
        self.action_executor = MagicMock()
        self.action_executor.apply_runtime_config = MagicMock()
        self.detectors = MagicMock()
        self.detectors.apply_runtime_config = MagicMock()
        self.report_writer = MagicMock()
        self.kv_reader = MagicMock()
        self.quota = MagicMock()
        self.wave_tracker = MagicMock()

    with patch.object(RuntimeGuardProcessor, "_init_from_runner", _init):
        return RuntimeGuardProcessor.bind(runner)


def test_runtime_config_sync_noop_when_reload_disabled(tmp_path: Path):
    cfg = _cfg(tmp_path, reload=0.0)
    assert cfg.hot_reload_enabled is False
    assert cfg.sync_runtime_config() is False


def test_needs_sample_phase_hooks_and_cumulative_io_flags(tmp_path: Path):
    cfg = _cfg(tmp_path, reload=0.0)
    assert cfg.needs_sample_phase_hooks() is False
    assert cfg.needs_cumulative_io() is False

    cfg._data["log"]["print_output_on_finish"] = True
    cfg._invalidate_hot_path_gates()
    assert cfg.needs_sample_phase_hooks() is True
    assert cfg.needs_cumulative_io() is True

    cfg._data["log"]["print_output_on_finish"] = False
    cfg._data["detector"]["token_repeat"]["enabled"] = True
    cfg._invalidate_hot_path_gates()
    assert cfg.needs_sample_phase_hooks() is True
    assert cfg.needs_cumulative_io() is True

    cfg._data["report"]["save_sensitive_info"] = True
    cfg._invalidate_hot_path_gates()
    assert cfg.needs_cumulative_io() is True


def test_hot_path_gates_cache_and_local_interval_skip(tmp_path: Path):
    """Gates stay cached until invalidate; file-mode sync skips while not due."""
    cfg = _cfg(tmp_path, reload=5.0)
    assert cfg.needs_sample_phase_hooks() is False
    g1 = cfg._hot_path_gates
    assert g1 is not None
    assert cfg.needs_sample_phase_hooks() is False
    assert cfg._hot_path_gates is g1

    cfg._last_reload_ts = time.time()
    cfg._initial_broadcast_done = True
    assert cfg.sync_runtime_config() is False

    cfg._data["detector"]["token_repeat"]["enabled"] = True
    assert cfg.needs_sample_phase_hooks() is False
    cfg._invalidate_hot_path_gates()
    assert cfg.needs_sample_phase_hooks() is True


def test_sync_for_step_skips_refresh_when_reload_off_idle(tmp_path: Path):
    """T1 idle shell: reload=0 + idle gates → advance only, no refresh."""
    cfg = _cfg(tmp_path, reload=0.0)
    assert cfg.hot_reload_enabled is False
    with (
        patch.object(RuntimeGuardProcessor, "refresh_config") as refresh,
        patch(
            "vllm_ascend.runtime_guard.processor.get_pp_group",
            side_effect=Exception("no pp"),
        ),
    ):
        proc = _bind(cfg)
        proc.sync_for_step(allow_arm=True, scheduler_output=None)
        proc.wave_tracker.advance.assert_called_once_with(allow_arm=True)
        refresh.assert_not_called()
        proc.manual_triggers.consume_once.assert_not_called()
    RuntimeGuardProcessor.reset_for_tests()


def test_sync_for_step_refreshes_when_reload_on_idle(tmp_path: Path):
    """T2 idle: hot-reload on → always refresh (broadcast lockstep)."""
    cfg = _cfg(tmp_path, reload=5.0)
    cfg._last_reload_ts = time.time()
    cfg._initial_broadcast_done = True
    with (
        patch.object(RuntimeGuardProcessor, "refresh_config") as refresh,
        patch(
            "vllm_ascend.runtime_guard.processor.get_pp_group",
            side_effect=Exception("no pp"),
        ),
    ):
        proc = _bind(cfg)
        proc.sync_for_step(allow_arm=True, scheduler_output=None)
        proc.wave_tracker.advance.assert_called_once_with(allow_arm=True)
        refresh.assert_called_once()
    RuntimeGuardProcessor.reset_for_tests()


def test_sync_for_step_marks_finished_on_empty_batch(tmp_path: Path):
    """Final-request-before-idle: empty batch still marks+reaps finished reqs.

    Async scheduling returns EMPTY_MODEL_RUNNER_OUTPUT for the trailing empty
    batch, so ``run_sample_phase`` (and its ``mark_finished`` hook) never fires.
    ``sync_for_step`` must mark ``scheduler_output.finished_req_ids`` itself so
    ``print_output_on_finish`` / reap still run for the last request.
    """
    cfg = _cfg(tmp_path, reload=0.0)
    cfg._data["log"]["print_output_on_finish"] = True
    cfg._invalidate_hot_path_gates()
    assert cfg.needs_sample_phase_hooks() is True

    calls: list[tuple] = []

    def _mark(*a, **k):
        calls.append(("mark", a, k))

    def _reap(*a, **k):
        calls.append(("reap", a, k))

    so_empty = SimpleNamespace(
        finished_req_ids={"r_final"},
        total_num_scheduled_tokens=0,
    )
    so_busy = SimpleNamespace(
        finished_req_ids={"r_prev"},
        total_num_scheduled_tokens=3,
    )

    with (
        patch.object(RuntimeGuardProcessor, "refresh_config"),
        patch.object(RuntimeGuardProcessor, "mark_finished", _mark),
        patch.object(RuntimeGuardProcessor, "_reap_finished_requests", _reap),
        patch(
            "vllm_ascend.runtime_guard.processor.get_pp_group",
            side_effect=Exception("no pp"),
        ),
    ):
        proc = _bind(cfg)
        proc.sync_for_step(allow_arm=True, scheduler_output=so_empty)
        mark_calls = [c for c in calls if c[0] == "mark"]
        assert mark_calls, "mark_finished must fire on the empty final batch"
        # bound method: _mark(self, finished_req_ids) → args[1] is the id set.
        assert mark_calls[0][1][1] == {"r_final"}
        calls.clear()
        proc.sync_for_step(allow_arm=True, scheduler_output=so_busy)
        assert not [c for c in calls if c[0] == "mark"]
    RuntimeGuardProcessor.reset_for_tests()


def test_refresh_config_skips_clear_wave_cache_when_idle(tmp_path: Path):
    cfg = _cfg(tmp_path, reload=0.0)
    with patch("vllm_ascend.runtime_guard.processor.RequestIoSnapshotManager") as io_mgr:
        io = MagicMock()
        io_mgr.get.return_value = io
        proc = _bind(cfg)
        proc.refresh_config()
        io.clear_wave_cache.assert_not_called()

        cfg._data["log"]["print_output_on_finish"] = True
        cfg._invalidate_hot_path_gates()
        proc.refresh_config()
        io.clear_wave_cache.assert_called_once()
    RuntimeGuardProcessor.reset_for_tests()


def test_run_sample_phase_idle_skips_hooks(tmp_path: Path):
    """A-path: detectors/print/block-meta off → sample_fn only."""
    cfg = _cfg(tmp_path, reload=0.0)
    proc = _bind(cfg)
    calls: list[str] = []

    def sample_fn():
        calls.append("sample")
        return SamplePhaseResult(
            scheduler_output=None,
            input_batch=None,
            model_runner_output=None,
            sampler_output=MagicMock(),
            valid_sampled_token_ids=[1],
            req_ids_output_copy=["r1"],
            invalid_req_indices=None,
            finished_req_ids=None,
        )

    proc.mark_finished = lambda *a, **k: calls.append("mark")  # type: ignore[method-assign]
    proc.record_sample_waves = lambda *a, **k: calls.append("waves")  # type: ignore[method-assign]
    proc.check_after_sample = lambda *a, **k: calls.append("after")  # type: ignore[method-assign]

    proc.run_sample_phase(
        sample_fn=sample_fn,
        speculative_config=None,
        need_accepted_tokens=False,
        use_async=False,
    )
    assert calls == ["sample"]
    RuntimeGuardProcessor.reset_for_tests()


def test_broadcast_all_reduce_every_call_even_when_not_due(tmp_path: Path):
    """B-path: always all_reduce; far from interval → due=0, no broadcast."""
    cfg = _cfg(tmp_path, reload=5.0)
    cfg._initial_broadcast_done = True
    cfg._last_reload_ts = time.time()

    group = MagicMock()
    group.world_size = 2
    group.cpu_group = object()
    group.is_first_rank = True
    group.rank_in_group = 0
    group.broadcast_object = MagicMock()
    with patch("torch.distributed.all_reduce") as ar:
        assert cfg._maybe_reload_broadcast(group) is False
        ar.assert_called_once()
        group.broadcast_object.assert_not_called()


def test_refresh_config_skips_detector_apply_when_unchanged(tmp_path: Path):
    cfg = _cfg(tmp_path, reload=0.05)
    with patch.object(cfg, "sync_runtime_config", return_value=False):
        proc = _bind(cfg)
        proc.refresh_config()
    proc.detectors.apply_runtime_config.assert_not_called()
    RuntimeGuardProcessor.reset_for_tests()
