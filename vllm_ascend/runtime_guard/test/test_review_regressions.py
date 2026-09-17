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

"""Regression UTs from the 2026-09-01 white-box review (task_spec/review_findings_20260901.md).

IDs map to review findings:
  V2  P0-5  dumps_report_json tolerates non-JSON scalars (report never lost)
  V3  P0-1  soft-fail: hook exceptions must never propagate into the engine
  V4  P0-2  shipped example template loads + validates as-is
  V5  P0-3  bootstrap invalid content falls back to defaults (no crash)
  V6  P1-C1 sync_mode frozen across hot-reload (DP collective safety)
  V8  P1-B2 wave stamps discarded when requests are reaped (no leak)
  V9  P1-B3 ActionQueue: heavy job dropped (never inline on hot path); stop works with full queue
  V10 P1-C5 unknown detector sub-key rejected on reload (typo protection)
  V12 P1-A1 logits_finite: unattributable row warns only (never misattributes / no null-req incident)
  V13 P0-2  JSONC comments + trailing commas parse
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import vllm_ascend.runtime_config.config as cfg_mod
from vllm_ascend.runtime_config.config import RuntimeConfig
from vllm_ascend.runtime_guard.action.queue import ActionQueue
from vllm_ascend.runtime_guard.runner_bridge import AscendAsyncOutput
from vllm_ascend.runtime_guard.detector.logits_finite import LogitsFiniteDetector
from vllm_ascend.runtime_guard.detector.manager import DetectorManager
from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor
from vllm_ascend.runtime_guard.report import dumps_report_json
from vllm_ascend.runtime_guard.wave_tracker import WaveTracker
from vllm_ascend.runtime_guard.test._helpers import bare_processor as _bare_processor


# ---------------------------------------------------------------- V2 (P0-5)


def test_v2_report_json_tolerates_non_json_scalars():
    out = dumps_report_json(
        {
            "incident_type": "token_repeat",
            "detail": {
                "np_int": np.int64(5),
                "torch_scalar": torch.tensor(7),
                "nan_float": float("nan"),
            },
        }
    )
    parsed = json.loads(out)
    assert parsed["incident_type"] == "token_repeat"
    assert "np_int" in parsed["detail"]
    assert "torch_scalar" in parsed["detail"]


# ---------------------------------------------------------------- V3 (P0-1)


def test_v3a_check_after_sample_soft_fail():
    p = _bare_processor()
    p.check_after_sample(sampled_token_ids=[1], req_ids=["r1"])


def test_v3b_check_before_sample_soft_fail():
    p = _bare_processor()
    p.check_before_sample(scheduler_output=None, logits=torch.randn(2, 8))


def test_v3c_async_output_boundary_soft_fail():
    out = SimpleNamespace(sampled_token_ids=[1], logprobs=None, req_ids=["r1"])
    inner = MagicMock()
    inner.get_output.return_value = out
    guard = MagicMock()
    guard.check_after_sample.side_effect = RuntimeError("boom")
    runner = SimpleNamespace(runtime_guard=guard)
    assert AscendAsyncOutput(inner, runner).get_output() is out


def test_v3d_run_sample_phase_hook_failure_does_not_block_sampling():
    p = _bare_processor()

    sampled: list[int] = []

    def sample_fn():
        sampled.append(1)
        return SimpleNamespace(
            scheduler_output=None,
            input_batch=None,
            finished_req_ids=None,
            req_ids_output_copy=["r1"],
            valid_sampled_token_ids=[1],
            sampler_output=SimpleNamespace(sampled_token_ids=[1]),
        )

    def boom(*args, **kwargs):
        raise RuntimeError("hook boom")

    p.mark_finished = boom
    p.record_sample_waves = boom
    p.check_after_spec = boom
    p.check_after_sample = boom
    p.should_check_after_spec = lambda: False

    result, routed = p.run_sample_phase(
        sample_fn=sample_fn,
        speculative_config=None,
        need_accepted_tokens=False,
        use_async=False,
    )
    assert sampled == [1]
    assert result.req_ids_output_copy == ["r1"]


# ---------------------------------------------------------------- V4 (P0-2)


def _template_path() -> Path:
    return Path(cfg_mod.__file__).parent / "templates" / "runtime_config.example.jsonc"


def test_v4_example_template_loads_and_validates(tmp_path: Path):
    from copy import deepcopy

    from vllm_ascend.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.runtime_config._merge import _deep_merge, _normalize_config_sections
    from vllm_ascend.runtime_config._validate import validate_runtime_config
    from vllm_ascend.runtime_config.jsonc_io import loads_jsonc

    raw = loads_jsonc(_template_path().read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    merged = _normalize_config_sections(_deep_merge(deepcopy(_DEFAULTS), raw))
    validate_runtime_config(merged)

    # Bootstrap ignores / overwrites any preexisting file with defaults.
    cfg_path = tmp_path / "runtime_config.json"
    shutil.copy(_template_path(), cfg_path)
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
    )
    assert cfg.detectors_enabled_in(cfg._data) is False
    assert cfg.dump_enabled() is False


def test_v13_jsonc_comments_and_trailing_commas(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=1,
        sync_mode="file",
    )
    cfg_path.write_text(
        """{
  // enable repeat detection
  "detector": {
    "token_repeat": { "enabled": true, "window": 8, }, /* trailing comma above */
  },
}""",
        encoding="utf-8",
    )
    os.utime(cfg_path, (time.time() + 10, time.time() + 10))
    assert cfg.reload(force=True) is True
    assert cfg.detector_get("token_repeat", "enabled") is True
    assert cfg.detector_get("token_repeat", "window") == 8


# ---------------------------------------------------------------- V5 (P0-3)


def test_v5_bootstrap_invalid_content_falls_back_to_defaults(tmp_path: Path):
    """Preexisting invalid JSON is ignored at bootstrap; defaults are written over it."""
    cfg_path = tmp_path / "runtime_config.json"
    cfg_path.write_text(
        json.dumps({"detector": {"fatal_error": {"enabled": True}}}),
        encoding="utf-8",
    )
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
    )
    assert cfg.detectors_enabled_in(cfg._data) is False
    on_disk = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "fatal_error" not in on_disk.get("detector", {})


# ---------------------------------------------------------------- V6 (P1-C1)


def test_v6_sync_mode_frozen_across_reload(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    cfg_path.write_text(json.dumps({"sync_mode": "broadcast"}), encoding="utf-8")
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
        sync_mode="broadcast",
    )
    cfg_path.write_text(json.dumps({"sync_mode": "file"}), encoding="utf-8")
    os.utime(cfg_path, (time.time() + 10, time.time() + 10))
    assert cfg.reload(force=True) is True
    assert cfg.sync_mode == "broadcast"


# ---------------------------------------------------------------- V8 (P1-B2)


def test_v8a_wave_tracker_discard():
    wt = WaveTracker()
    wt.advance(allow_arm=True)
    wt.record_sample_waves(["r1"])
    assert wt.take_sample_wave("r1") == 1
    wt.record_sample_waves(["r1"])
    wt.discard("r1")
    wt.discard("never-seen")  # no raise
    assert wt._sample_waves == {}


def test_v8a3_wave_tracker_fifo_under_async_lag():
    """W1-3: later record must not erase an untaken earlier stamp."""
    wt = WaveTracker()
    wt.advance(allow_arm=True)
    wt.record_sample_waves(["r1"])  # wave 1
    wt.advance(allow_arm=True)
    wt.record_sample_waves(["r1"])  # wave 2 (must queue, not overwrite)
    assert wt.take_sample_wave("r1") == 1
    assert wt.take_sample_wave("r1") == 2
    assert wt.take_sample_wave("r1") is None
    assert wt.pending("r1") is False


def test_v8a2_wave_tracker_lock_record_take_concurrent():
    wt = WaveTracker()
    wt.advance(allow_arm=True)
    errs: list[BaseException] = []

    def _record() -> None:
        try:
            for i in range(200):
                wt.record_sample_waves([f"r{i % 8}"])
        except BaseException as exc:  # noqa: BLE001 — surface to main thread
            errs.append(exc)

    def _take() -> None:
        try:
            for i in range(200):
                wt.take_sample_wave(f"r{i % 8}")
                wt.pending(f"r{i % 8}")
        except BaseException as exc:  # noqa: BLE001
            errs.append(exc)

    threads = [threading.Thread(target=_record), threading.Thread(target=_take)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)
    assert errs == []


def test_v8b_reap_discards_wave_stamps():
    p = _bare_processor()
    wt = WaveTracker()
    wt.advance(allow_arm=True)
    wt.record_sample_waves(["r1", "r2"])
    p.wave_tracker = wt
    p.runtime_config = MagicMock()
    p.runtime_config.log_print_output_on_finish.return_value = False
    store = MagicMock()
    store.list_reapable.return_value = ["r1", "r2"]
    with patch("vllm_ascend.runtime_guard.processor.RequestGuardStore") as store_cls:
        store_cls.get.return_value = store
        # Call the real method (the bare processor mocks the instance attr).
        RuntimeGuardProcessor._reap_finished_requests(p)
    assert wt._sample_waves == {}
    assert store.clear_many.called


def test_v8c_async_skips_sample_wave_stamp_on_non_tp0():
    """Async non-TP0 must not record stamps (no AscendAsync take on that rank)."""
    from vllm_ascend.runtime_guard.processor import SamplePhaseResult

    p = _bare_processor()
    wt = WaveTracker()
    wt.advance(allow_arm=True)
    p.wave_tracker = wt
    p.runner = SimpleNamespace(tp_rank=1)
    p.needs_sample_phase_hooks = lambda: True  # type: ignore[method-assign]
    p.should_check_after_spec = lambda: False  # type: ignore[method-assign]
    p.mark_finished = lambda *a, **k: None  # type: ignore[method-assign]

    def sample_fn():
        return SamplePhaseResult(
            scheduler_output=None,
            input_batch=None,
            model_runner_output=None,
            sampler_output=SimpleNamespace(sampled_token_ids=[1]),
            valid_sampled_token_ids=[1],
            req_ids_output_copy=["r1"],
            invalid_req_indices=None,
            finished_req_ids=None,
        )

    with patch(
        "vllm_ascend.runtime_guard.processor.runner_tp_rank",
        return_value=1,
    ):
        p.run_sample_phase(
            sample_fn=sample_fn,
            speculative_config=None,
            need_accepted_tokens=False,
            use_async=True,
        )
    assert wt._sample_waves == {}
    assert wt.pending("r1") is False

    with patch(
        "vllm_ascend.runtime_guard.processor.runner_tp_rank",
        return_value=0,
    ):
        p.run_sample_phase(
            sample_fn=sample_fn,
            speculative_config=None,
            need_accepted_tokens=False,
            use_async=True,
        )
    assert wt.take_sample_wave("r1") == 1


# ---------------------------------------------------------------- V9 (P1-B3)


def test_v9a_full_queue_drops_heavy_job():
    q = ActionQueue(maxsize=1, name="ut-heavy-drop")
    q.start()
    gate = threading.Event()
    try:
        q.submit(lambda: gate.wait(2.0))
        time.sleep(0.05)
        ran: list[int] = []
        q.submit(lambda: ran.append(1), heavy=True)
        assert ran == []
    finally:
        gate.set()
        q.stop()


def test_v9b_stop_with_full_queue_still_stops_worker():
    q = ActionQueue(maxsize=2, name="ut-stop-full")
    q.start()
    gate = threading.Event()
    q.submit(lambda: gate.wait(3.0))
    q.submit(lambda: None)
    q.submit(lambda: None)  # pending queue now full
    t = q._thread
    assert t is not None
    q.stop()  # queue full: sentinel must still be delivered
    gate.set()  # let the in-flight job finish
    t.join(timeout=3.0)
    assert not t.is_alive()


def test_v9c_submit_during_stop_is_dropped():
    q = ActionQueue(maxsize=4, name="ut-stop-submit")
    q.start()
    gate = threading.Event()
    entered = threading.Event()

    def _block() -> None:
        entered.set()
        gate.wait(timeout=2.0)

    q.submit(_block)
    assert entered.wait(timeout=1.0)
    stopper = threading.Thread(target=lambda: q.stop(timeout=2.0))
    stopper.start()
    time.sleep(0.05)
    assert q.submit(lambda: None, heavy=True) is False
    gate.set()
    stopper.join(timeout=2.0)
    assert not q.started


def test_v9d_dedupe_key_skips_second_submit_same_key(caplog):
    import logging

    q = ActionQueue(maxsize=8, name="ut-dedupe")
    q.start()
    gate = threading.Event()
    ran: list[int] = []
    try:
        with caplog.at_level(logging.INFO, logger="vllm_ascend.runtime_guard.action.queue"):
            assert q.submit(lambda: gate.wait(2.0), dedupe_key=("report", 1, "r1")) is True
            time.sleep(0.05)
            assert q.submit(lambda: ran.append(1), dedupe_key=("report", 1, "r1")) is False
        assert any("duplicate key" in r.message for r in caplog.records)
        assert q.submit(lambda: ran.append(2), dedupe_key=("report", 2, "r1")) is True
        assert ran == []
        gate.set()
        deadline = time.time() + 2.0
        while time.time() < deadline and ran != [2]:
            time.sleep(0.01)
        assert ran == [2]
    finally:
        gate.set()
        q.stop()


# ---------------------------------------------------------------- V10 (P1-C5)


def test_v10_unknown_detector_key_rejected_on_reload(tmp_path: Path):
    cfg_path = tmp_path / "runtime_config.json"
    cfg_path.write_text(json.dumps({"detector": {"token_repeat": {"enabled": False}}}), encoding="utf-8")
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
    )
    # typo key "windw" must fail the reload loudly instead of silently defaulting
    cfg_path.write_text(
        json.dumps({"detector": {"token_repeat": {"enabled": True, "windw": 8}}}),
        encoding="utf-8",
    )
    os.utime(cfg_path, (time.time() + 10, time.time() + 10))
    assert cfg.reload(force=True) is False
    assert cfg.detector_get("token_repeat", "enabled") is False


def test_v10b_unknown_top_level_key_rejected_on_reload(tmp_path: Path, caplog):
    """W2-1 / F-07: typo top-level keys (e.g. windw) must not be silently persisted."""
    import logging

    from vllm_ascend.runtime_config._validate import validate_runtime_config

    with pytest.raises(ValueError, match="unknown top-level key"):
        validate_runtime_config({"windw": 10, "dump": {}, "log": {}, "report": {}, "ascend_log": {}, "detector": {}, "actions": {}})

    cfg_path = tmp_path / "runtime_config.json"
    cfg_path.write_text(json.dumps({"detector": {"token_repeat": {"enabled": False}}}), encoding="utf-8")
    cfg = RuntimeConfig(
        config_path=cfg_path,
        report_dir=tmp_path / "report",
        ensure_file=True,
        reload_interval_seconds=0,
    )
    assert cfg.detector_get("token_repeat", "enabled") is False
    cfg_path.write_text(json.dumps({"windw": 10, "detector": {"token_repeat": {"enabled": True}}}), encoding="utf-8")
    os.utime(cfg_path, (time.time() + 10, time.time() + 10))
    with caplog.at_level(logging.ERROR, logger="vllm_ascend.runtime_config.config"):
        assert cfg.reload(force=True) is False
    assert any("unknown top-level key" in r.message for r in caplog.records)
    assert cfg.detector_get("token_repeat", "enabled") is False
    assert "windw" not in cfg._data


# ---------------------------------------------------------------- V12 (P1-A1)


def test_v12_logits_finite_unattributable_row_warns_not_misattributes():
    import io
    import logging

    section = {"enabled": True}
    rc = SimpleNamespace(
        detector_section=lambda name: section,
        detector_get=lambda sec, key, default=None: section.get(key, default),
    )
    input_batch = SimpleNamespace(req_ids=["a", "b"])
    runner = SimpleNamespace(input_batch=input_batch)  # no query_start_loc
    det = LogitsFiniteDetector(runtime_config=rc, runner=runner)

    logits = torch.randn(16, 8)
    logits[5, :] = float("nan")  # chunked-prefill row; indices without qsl → unresolved
    idx = torch.arange(16)

    # vllm_ascend parent has propagate=False + its own StreamHandler, so neither
    # caplog(root) nor leaf propagate=True reaches pytest. Capture on the leaf.
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.WARNING)
    lg = logging.getLogger("vllm_ascend.runtime_guard.detector.logits_finite")
    lg.addHandler(handler)
    try:
        det.check_all(logits=logits, logits_indices=idx, input_batch=input_batch)
    finally:
        lg.removeHandler(handler)
    assert det.check_deferred() == []  # no guessed req_id, no null-req incident
    assert "cannot attribute" in buf.getvalue()


def test_v12b_logits_finite_decode_rows_still_attributed():
    section = {"enabled": True}
    rc = SimpleNamespace(
        detector_section=lambda name: section,
        detector_get=lambda sec, key, default=None: section.get(key, default),
    )
    input_batch = SimpleNamespace(req_ids=["a", "b"])
    runner = SimpleNamespace(input_batch=input_batch)
    det = LogitsFiniteDetector(runtime_config=rc, runner=runner)

    logits = torch.randn(2, 8)
    logits[1, :] = float("nan")
    det.check_all(logits=logits, logits_indices=None, input_batch=input_batch)
    alerts = det.check_deferred()
    assert len(alerts) == 1
    assert alerts[0].req_id == "b"


def test_v12c_logits_finite_alerts_same_step_no_window():
    """Dirty logits alert on the same step's check_deferred (no multi-step OR)."""
    section = {"enabled": True}
    rc = SimpleNamespace(
        detector_section=lambda name: section,
        detector_get=lambda sec, key, default=None: section.get(key, default),
    )
    input_batch = SimpleNamespace(req_ids=["a"])
    runner = SimpleNamespace(input_batch=input_batch)
    det = LogitsFiniteDetector(runtime_config=rc, runner=runner)

    dirty = torch.randn(1, 8)
    dirty[0, 0] = float("nan")
    det.check_all(logits=dirty, logits_indices=None, input_batch=input_batch)
    alerts = det.check_deferred()
    assert len(alerts) == 1
    assert alerts[0].req_id == "a"
    assert alerts[0].detail.get("finite_kind") == "nan"
    assert "window_tokens" not in alerts[0].detail


def test_v12c2_logits_finite_hit_resolved_before_enqueue():
    """On hit, indices/kind are resolved before enqueue (inplace rewrite is harmless)."""
    section = {"enabled": True}
    rc = SimpleNamespace(
        detector_section=lambda name: section,
        detector_get=lambda sec, key, default=None: section.get(key, default),
    )
    input_batch = SimpleNamespace(req_ids=["a"])
    qsl = __import__("numpy").asarray([0, 10], dtype="int64")
    input_batch.query_start_loc_np = qsl
    runner = SimpleNamespace(input_batch=input_batch)
    det = LogitsFiniteDetector(runtime_config=rc, runner=runner)

    logits = torch.randn(2, 4)
    logits[1, :] = float("nan")
    idx = torch.tensor([0, 1], dtype=torch.long)
    det.check_all(logits=logits, logits_indices=idx, input_batch=input_batch)
    idx.fill_(-1)
    logits.fill_(0.0)
    alerts = det.check_deferred()
    assert len(alerts) == 1
    assert alerts[0].req_id == "a"
    assert alerts[0].detail.get("flat_token_index") == 1
    assert alerts[0].detail.get("finite_kind") == "nan"

def test_v12d_logits_finite_check_every_tokens_rejected():
    import copy

    from vllm_ascend.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.runtime_config._validate import validate_runtime_config

    data = copy.deepcopy(_DEFAULTS)
    data["detector"]["logits_finite"] = {
        "enabled": True,
        "check_every_tokens": 100,
    }
    with pytest.raises(ValueError, match="unknown key"):
        validate_runtime_config(data)


def test_v12e_logits_finite_skip_stopped_on_hot_path():
    """detection_stopped reqs must not enqueue; all-stopped skips .item() work."""
    section = {"enabled": True}
    rc = SimpleNamespace(
        detector_section=lambda name: section,
        detector_get=lambda sec, key, default=None: section.get(key, default),
    )
    input_batch = SimpleNamespace(req_ids=["a", "b"])
    runner = SimpleNamespace(input_batch=input_batch)
    det = LogitsFiniteDetector(runtime_config=rc, runner=runner)

    dirty = torch.randn(2, 8)
    dirty[:, 0] = float("nan")
    det.check_all(
        logits=dirty,
        logits_indices=None,
        input_batch=input_batch,
        skip_req_ids={"a"},
    )
    alerts = det.check_deferred()
    assert len(alerts) == 1
    assert alerts[0].req_id == "b"

    det.check_all(
        logits=dirty,
        logits_indices=None,
        input_batch=input_batch,
        skip_req_ids={"a", "b"},
    )
    assert det.check_deferred() == []


# ---------------------------------------------------------------- V14 (P1-B'4)


def _quota_rc(max_times: int, cooldown: float) -> SimpleNamespace:
    return SimpleNamespace(
        dump_max_times=lambda: max_times,
        dump_cooldown_seconds=lambda: cooldown,
    )


def test_v14_quota_try_consume_atomic_and_refund():
    from vllm_ascend.runtime_guard.quota import DumpQuota

    q = DumpQuota(_quota_rc(max_times=2, cooldown=0.0))
    assert q.try_consume() is True
    assert q.try_consume() is True
    assert q.try_consume() is False  # cap reached, atomically
    q.refund()
    assert q.try_consume() is True  # refund restored one unit


def test_v14b_quota_cooldown_block_must_not_burn():
    from vllm_ascend.runtime_guard.quota import DumpQuota

    q = DumpQuota(_quota_rc(max_times=5, cooldown=3600.0))
    assert q.try_consume() is True
    assert q.try_consume() is False  # inside cooldown
    assert q.total_count == 1  # blocked consume must not count
    q.refund()  # captured-nothing scenario
    assert q.total_count == 0
    # Refund must clear cooldown so the next arm is not stuck for 3600s.
    assert q.try_consume() is True
    assert q.total_count == 1


def test_v14c_refund_clears_cooldown_without_prior_count():
    """Defensive: refund with total_count==0 still drops a stale last_ts."""
    from vllm_ascend.runtime_guard.quota import DumpQuota

    q = DumpQuota(_quota_rc(max_times=5, cooldown=3600.0))
    assert q.try_consume() is True
    q._total_count = 0  # simulate torn state; cooldown stamp remains
    q.refund()
    assert q.try_consume() is True


# ---------------------------------------------------------------- V15 (P1-B'2)


def test_v15_report_writer_dedupes_same_pair(tmp_path: Path):
    from vllm_ascend.runtime_guard.report import ReportWriter
    from vllm_ascend.runtime_guard.request_state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    store = RequestGuardStore.get()
    store.get_or_create("r1")
    store.get_or_create("r2")
    w = ReportWriter(tmp_path / "report", max_per_req=1)
    kw = dict(incident_type="token_repeat", req_id="r1", detail={"x": 1}, dump_arm_wave=10)
    assert w.write(**kw) is not None
    assert store.get_state("r1").detection_stopped is True
    assert w.write(**kw) is None  # max_per_req=1
    assert w.write(incident_type="token_repeat", req_id="r2", detail={"x": 1}, dump_arm_wave=10) is not None


def test_v15b_report_writer_wave_backoff(tmp_path: Path):
    from vllm_ascend.runtime_guard.report import _SAME_PAIR_BASE_WAVES, ReportWriter
    from vllm_ascend.runtime_guard.request_state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    store = RequestGuardStore.get()
    store.get_or_create("r1")
    w = ReportWriter(tmp_path / "report", max_per_req=3)
    kw = dict(incident_type="token_repeat", req_id="r1", detail={"x": 1})
    assert w.write(**kw, dump_arm_wave=100) is not None
    assert store.get_state("r1").detection_stopped is False
    assert w.write(**kw, dump_arm_wave=100 + _SAME_PAIR_BASE_WAVES - 1) is None
    assert w.write(**kw, dump_arm_wave=100 + _SAME_PAIR_BASE_WAVES) is not None
    # Next gap doubles: 128 waves from the 2nd write's wave.
    second_wave = 100 + _SAME_PAIR_BASE_WAVES
    assert w.write(**kw, dump_arm_wave=second_wave + 2 * _SAME_PAIR_BASE_WAVES - 1) is None
    assert w.write(**kw, dump_arm_wave=second_wave + 2 * _SAME_PAIR_BASE_WAVES) is not None
    assert store.get_state("r1").detection_stopped is True


def test_v15c_stop_detect_after_reap_does_not_resurrect():
    """Report commit can race finish→reap; late stop must not allocate orphans."""
    from vllm_ascend.runtime_guard.request_state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    store = RequestGuardStore.get()
    store.get_or_create("r1")
    store.mark_finished(["r1"], wave=1)
    assert store.clear("r1") is not None
    assert store.get_state("r1") is None

    store.mark_detection_stopped("r1")
    assert store.get_state("r1") is None
    # Fresh id reuse must start with detection enabled.
    assert store.get_or_create("r1").detection_stopped is False
    RequestGuardStore.reset_for_tests()


# ------------------------------------------------- V16 (detector clear_finished)


def test_v16b_spec_acceptance_clear_finished_history():
    from vllm_ascend.runtime_guard.detector.spec_acceptance import SpecAcceptanceDetector

    section = {"enabled": True, "window": 2}
    rc = SimpleNamespace(
        detector_section=lambda name: section,
        detector_get=lambda sec, key, default=None: section.get(key, default),
    )
    det = SpecAcceptanceDetector(runtime_config=rc, runner=None)
    det._history["r1"].append((1, 2, [1, 2], [1]))
    det.clear_finished("r1")
    assert "r1" not in det._history


# ------------------------------------------------- V17 (spec short batch)


def test_v17_spec_acceptance_short_batch_no_index_error():
    from vllm_ascend.runtime_guard.detector.spec_acceptance import SpecAcceptanceDetector

    section = {
        "enabled": True,
        "window": 2,
        "low_threshold": 0.3,
        "len_low_threshold": 1.4,
        "high_threshold": 0.96,
        "len_high_threshold": 2.8,
    }
    rc = SimpleNamespace(
        detector_section=lambda name: section,
        detector_get=lambda sec, key, default=None: section.get(key, default),
    )
    runner = SimpleNamespace(
        tp_rank=1,
        speculative_config=SimpleNamespace(),
        input_batch=SimpleNamespace(req_ids=["a", "b"], num_draft_tokens_per_req=None),
        requests=None,
    )
    det = SpecAcceptanceDetector(runtime_config=rc, runner=runner)
    sampled = torch.tensor([[7, 8, 9, 10], [7, 8, 9, 10]])
    with patch(
        "vllm_ascend.runtime_guard.detector.spec_acceptance.get_pp_group",
        return_value=SimpleNamespace(is_last_rank=True),
    ):
        alerts = det.check_all(sampled, [1])  # accepted shorter than req_ids
    assert isinstance(alerts, list)


def test_v17b_spec_acceptance_v2_threads_req_ids_without_input_batch():
    """v2: runner.input_batch is None; req_ids must come from SamplePhaseResult."""
    from vllm_ascend.runtime_guard.detector.spec_acceptance import SpecAcceptanceDetector

    section = {
        "enabled": True,
        "window": 1,
        "low_threshold": 0.3,
        "len_low_threshold": 1.4,
        "high_threshold": 0.96,
        "len_high_threshold": 2.8,
    }
    rc = SimpleNamespace(
        detector_section=lambda name: section,
        detector_get=lambda sec, key, default=None: section.get(key, default),
    )
    runner = SimpleNamespace(
        tp_rank=0,
        speculative_config=SimpleNamespace(),
        input_batch=None,
        requests=None,
    )
    det = SpecAcceptanceDetector(runtime_config=rc, runner=runner)
    sampled = torch.tensor([[7, 8, 9, 10], [7, 8, 9, 10]])
    with patch(
        "vllm_ascend.runtime_guard.detector.spec_acceptance.get_pp_group",
        return_value=SimpleNamespace(is_last_rank=True),
    ):
        assert det.check_all(sampled, [0, 0]) == []
        alerts = det.check_all(sampled, [0, 0], req_ids=["r1", "r2"])
    assert "r1" in det._history and "r2" in det._history
    assert isinstance(alerts, list)


def test_v17c_check_after_spec_forwards_req_ids():
    """Processor/DetectorManager must pass SamplePhaseResult.req_ids_output_copy."""
    from vllm_ascend.runtime_guard.detector.manager import DetectorManager

    seen: dict[str, object] = {}

    class _FakeSpec:
        def check_all(self, sampled, accepted, skip_req_ids=None, req_ids=None):
            seen["req_ids"] = req_ids
            seen["skip"] = skip_req_ids
            return []

    runner = SimpleNamespace(tp_rank=0, input_batch=None)
    rc = SimpleNamespace(
        detector_section=lambda _name: {"enabled": True},
        detector_get=lambda _sec, _key, default=None: default,
    )
    mgr = DetectorManager(runtime_config=rc, runner=runner, detection_gate=lambda: True)
    mgr._spec_det = _FakeSpec()  # type: ignore[method-assign]
    mgr._gated = lambda _phase: False  # type: ignore[method-assign]
    mgr.check_after_spec(sampled_tokens=[[1]], accepted_token_nums=[0], req_ids=["cmpl-a"])
    assert seen["req_ids"] == ["cmpl-a"]


# ---------------------------------------------------------------- payload (B'6)


def test_v18_dump_payload_carries_tp_rank_and_heads(tmp_path: Path):
    from vllm_ascend.runtime_guard.kv_cache_reader import KvCacheReader

    cache = torch.randn(4, 8, 2, 16)  # [blocks, block_size, kv_heads, head_dim]
    runner = SimpleNamespace(kv_caches={"L0": cache}, tp_rank=3, dp_rank=0, dcp_rank=0, dcp_size=1)
    reader = KvCacheReader(runner)
    snaps = reader.snapshot_request_blocks(
        req_id="r1",
        block_ids=[0, 2],
        out_dir=tmp_path / "kv",
    )
    assert snaps[0].payload["tp_rank"] == 3
    assert snaps[0].payload["num_kv_heads"] == 2
    assert "tp3" in str(snaps[0].payload["rank_tag"])
    assert snaps[0].path.parent == tmp_path / "kv"


def test_v18c_list_kv_caches_use_global_layer_names(tmp_path: Path):
    """PP last stage: list caches must not be named layer_0.. locally."""
    from vllm_ascend.runtime_guard.kv_cache_reader import KvCacheReader

    cache0 = torch.randn(2, 4, 1, 8)
    cache1 = torch.randn(2, 4, 1, 8)
    groups = [
        SimpleNamespace(
            layer_names=["model.layers.14.self_attn", "model.layers.15.self_attn"]
        )
    ]
    runner = SimpleNamespace(
        kv_caches=[cache0, cache1],
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        model=SimpleNamespace(start_layer=14),
        tp_rank=0,
        dp_rank=0,
        dcp_rank=0,
        dcp_size=1,
    )
    reader = KvCacheReader(runner)
    snaps = reader.snapshot_request_blocks(
        req_id="r1",
        block_ids=[0],
        out_dir=tmp_path / "kv",
    )
    layers = [s.payload["layer"] for s in snaps]
    assert layers == ["model.layers.14.self_attn", "model.layers.15.self_attn"]
    assert all("layer_0" not in str(s.path) for s in snaps)


def test_v18d_list_kv_caches_fallback_start_layer(tmp_path: Path):
    from vllm_ascend.runtime_guard.kv_cache_reader import KvCacheReader

    cache = torch.randn(2, 4, 1, 8)
    runner = SimpleNamespace(
        kv_caches=[cache],
        model=SimpleNamespace(start_layer=14),
        tp_rank=0,
        dp_rank=0,
        dcp_rank=0,
        dcp_size=1,
    )
    snaps = KvCacheReader(runner).snapshot_request_blocks(
        req_id="r1",
        block_ids=[0],
        out_dir=tmp_path / "kv",
    )
    assert snaps[0].payload["layer"] == "layer_14"



@pytest.mark.parametrize("tp_rank", [0, 1])
def test_v18b_kv_drain_dumps_on_tp_ranks(tmp_path: Path, tp_rank: int):
    from vllm_ascend.runtime_guard.kv_cache_reader import KvCacheReader
    from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor
    from vllm_ascend.runtime_guard.rank_gate import dump_rank_tag

    cache = torch.randn(2, 4, 1, 8)
    runner = SimpleNamespace(
        kv_caches={"L0": cache},
        tp_rank=tp_rank,
        dp_rank=0,
        dcp_rank=0,
        dcp_size=1,
    )
    proc = SimpleNamespace(
        runner=runner,
        runtime_config=SimpleNamespace(dump_root=lambda: str(tmp_path / "kv_cache")),
        action_executor=SimpleNamespace(
            _kv_reader=KvCacheReader(runner),
            submit_heavy=None,
        ),
    )
    jobs = [
        {
            "req_id": "r1",
            "incident_type": "token_repeat",
        }
    ]
    with patch(
        "vllm_ascend.runtime_guard.processor_dump.block_ids_for_request",
        return_value=[0],
    ):
        RuntimeGuardProcessor._run_kv_dumps(proc, jobs)
    shard = tmp_path / "kv_cache" / "token_repeat" / "r1" / "wave_unknown" / dump_rank_tag(runner)
    assert shard.is_dir()
    assert list(shard.glob("*.pt"))


def test_v18e_file_mode_claim_skips_object_broadcast_when_empty():
    from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor

    proc = SimpleNamespace(
        runner=SimpleNamespace(),
        _kv_dump_jobs=[],
        _deferred_kv_dump_jobs=[],
    )
    tp = MagicMock()
    tp.world_size = 2
    tp.rank_in_group = 0
    tp.cpu_group = object()
    tp.broadcast_object = MagicMock()
    with (
        patch(
            "vllm_ascend.runtime_guard.processor_dump.should_dump_kv_on_rank",
            return_value=True,
        ),
        patch(
            "vllm_ascend.runtime_guard.processor_dump.get_tp_group",
            return_value=tp,
        ),
        patch("torch.distributed.all_reduce"),
    ):
        RuntimeGuardProcessor._claim_dump_jobs_to_deferred_via_tp(proc)
    tp.broadcast_object.assert_not_called()
    assert proc._deferred_kv_dump_jobs == []


def test_v18h_apply_config_payload_bumps_follower_reload_ts(tmp_path: Path):
    """Followers must advance _last_reload_ts even when data is None."""
    from vllm_ascend.runtime_config.config import RuntimeConfig

    path = tmp_path / "runtime_config.json"
    path.write_text("{}", encoding="utf-8")
    cfg = RuntimeConfig(
        config_path=path,
        reload_interval_seconds=5.0,
        sync_mode="broadcast",
        ensure_file=False,
    )
    cfg._initial_broadcast_done = True
    cfg._last_reload_ts = time.time() - 100.0
    before = cfg._last_reload_ts
    changed = cfg.apply_config_sync_payload(
        {"version": float(cfg._version), "data": None},
        is_leader=False,
        leader_changed=False,
    )
    assert changed is False
    assert cfg._last_reload_ts >= before
    assert cfg.config_due_local() is False


def test_v18f_merged_bus_idle_one_ar_zero_bcast():
    """Wave-head: neither lane due → one due AR, no broadcast_object."""
    from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor

    cfg = MagicMock()
    cfg.hot_reload_enabled = True
    cfg.config_due_local.return_value = False
    group = MagicMock()
    group.world_size = 2
    group.cpu_group = object()
    group.is_first_rank = True
    group.rank_in_group = 0
    group.broadcast_object = MagicMock()
    proc = SimpleNamespace(
        runner=SimpleNamespace(),
        runtime_config=cfg,
        _kv_dump_jobs=[],
        _deferred_kv_dump_jobs=[],
        _apply_config_cascade=MagicMock(),
    )
    with (
        patch(
            "vllm_ascend.runtime_guard.processor_bus.should_dump_kv_on_rank",
            return_value=True,
        ),
        patch("torch.distributed.all_reduce") as ar,
    ):
        assert RuntimeGuardProcessor._wave_head_merged_bus(proc, group) is False
    ar.assert_called_once()
    group.broadcast_object.assert_not_called()
    assert proc._deferred_kv_dump_jobs == []


def test_v18g_merged_bus_both_lanes_two_bcasts():
    """When both due: one AR + config bcast + dump bcast (dump stashed deferred)."""
    from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor

    jobs = [{"req_id": "r1", "incident_type": "token_repeat"}]
    cfg = MagicMock()
    cfg.hot_reload_enabled = True
    cfg.config_due_local.return_value = True
    cfg.build_config_sync_payload.return_value = (
        {"version": 1.0, "data": {"x": 1}},
        True,
    )
    cfg.apply_config_sync_payload.return_value = True
    group = MagicMock()
    group.world_size = 2
    group.cpu_group = object()
    group.is_first_rank = True
    group.rank_in_group = 0
    group.broadcast_object = MagicMock(side_effect=lambda obj, src=0: obj)
    proc = SimpleNamespace(
        runner=SimpleNamespace(),
        runtime_config=cfg,
        _kv_dump_jobs=list(jobs),
        _deferred_kv_dump_jobs=[],
        _apply_config_cascade=MagicMock(),
    )
    with (
        patch(
            "vllm_ascend.runtime_guard.processor_bus.should_dump_kv_on_rank",
            return_value=True,
        ),
        patch("torch.distributed.all_reduce"),
    ):
        assert RuntimeGuardProcessor._wave_head_merged_bus(proc, group) is True
    assert group.broadcast_object.call_count == 2
    assert proc._deferred_kv_dump_jobs == jobs
    proc._apply_config_cascade.assert_called_once()


# ------------------------------------------------- manual_dump drain (#5 / V19)


class _ConsumeRecorder:
    """runtime_config stand-in recording consume_manual_trigger calls."""

    def __init__(self, remaining: int = 2) -> None:
        self.remaining = remaining
        self.consume_calls = 0

    def consume_manual_trigger(self) -> bool:
        self.consume_calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            return True
        return False

    def manual_trigger_count(self) -> int:
        return self.remaining

    def dump_root(self):
        return "/tmp/ut-rg-report/kv_cache"

    def dump_get(self, key, default=None):
        if key == "free_headroom_bytes":
            return 0
        return default

    def report_include_block_ids(self) -> bool:
        return False

    def report_save_sensitive_info(self) -> bool:
        return False

    @property
    def report_dir(self):
        return "/tmp/ut-rg-report"


def _dump_ctx(incident, runtime_config, kv_reader, quota, *, req_ids=None) -> SimpleNamespace:
    # estimate_dump_bytes used for free-space gate; MagicMock must return int.
    if isinstance(kv_reader, MagicMock):
        kv_reader.estimate_dump_bytes.return_value = 0
    guard = MagicMock()
    # prepare only queues; without a truthy return, _queue_kv_dumps refunds and
    # skips arm (same contract as test_v19d).
    guard.queue_kv_dump.return_value = True
    runner_kw: dict = {
        "tp_rank": 0,
        "dp_rank": 0,
        "dcp_rank": 0,
        "dcp_size": 1,
        "runtime_guard": guard,
    }
    if req_ids is not None:
        runner_kw["input_batch"] = SimpleNamespace(req_ids=list(req_ids))
    return SimpleNamespace(
        incident=incident,
        runner=SimpleNamespace(**runner_kw),
        runtime_config=runtime_config,
        report_writer=MagicMock(),
        kv_reader=kv_reader,
        quota=quota,
        rank_tag="TP0",
        tokenizer=None,
        detail={},
        action_overrides={},
        submit_async=lambda job: None,
    )


def _quota_stub(ok: bool = True):
    q = MagicMock()
    q.try_consume.return_value = ok
    return q


def test_v19a4_manual_dump_arms_with_empty_block_ids(monkeypatch):
    """First prefill wave: req ids known, block table empty — still queue."""
    from vllm_ascend.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.runtime_guard.incident import Incident
    from vllm_ascend.runtime_guard.manual_trigger import MANUAL_TRIGGER_TYPE

    rc = _ConsumeRecorder(remaining=1)
    kv_reader = MagicMock()
    ctx = _dump_ctx(
        Incident(
            incident_type=MANUAL_TRIGGER_TYPE,
            req_id="__manual_trigger__",
            consume_quota=False,
            block_ids=[],
        ),
        rc,
        kv_reader,
        _quota_stub(),
        req_ids=["r1"],
    )
    ctx.action_overrides = {"dump_kv": {"scope": "all_requests"}}
    monkeypatch.setattr(
        "vllm_ascend.runtime_guard.kv_block_meta.block_ids_for_request",
        lambda *_a, **_k: [],
    )
    DumpKvAction().run(ctx)
    assert ctx.runner.runtime_guard.queue_kv_dump.call_count == 1
    assert ctx.runner.runtime_guard.queue_kv_dump.call_args.args[0]["req_id"] == "r1"


def test_v19a_manual_dump_forces_all_requests_no_consume_in_prepare(monkeypatch):
    """Manual dump ignores scope=request; consume happens in processor handle."""
    from vllm_ascend.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.runtime_guard.incident import Incident
    from vllm_ascend.runtime_guard.manual_trigger import MANUAL_TRIGGER_TYPE

    rc = _ConsumeRecorder(remaining=2)
    kv_reader = MagicMock()
    ctx = _dump_ctx(
        Incident(
            incident_type=MANUAL_TRIGGER_TYPE,
            req_id="__manual_trigger__",
            consume_quota=False,
            block_ids=[0],
        ),
        rc,
        kv_reader,
        _quota_stub(),
        req_ids=["r1", "r2"],
    )
    ctx.action_overrides = {"dump_kv": {"scope": "request"}}
    monkeypatch.setattr(
        "vllm_ascend.runtime_guard.kv_block_meta.block_ids_for_request",
        lambda _runner, req_id, req_idx=None, **kw: [0] if req_id == "r1" else [1],
    )
    DumpKvAction().run(ctx)  # arm only; D2H is flush_kv_dumps at sample end
    assert rc.consume_calls == 0  # consume moved to _handle_manual_trigger
    assert ctx.runner.runtime_guard.queue_kv_dump.call_count == 2
    queued = [c.args[0]["req_id"] for c in ctx.runner.runtime_guard.queue_kv_dump.call_args_list]
    assert queued == ["r1", "r2"]


def test_v19a2_manual_trigger_handle_consumes_one_count():
    from vllm_ascend.runtime_guard.manual_trigger import (
        MANUAL_TRIGGER_REQ_ID,
        MANUAL_TRIGGER_TYPE,
        TriggerEvent,
    )
    from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor

    p = _bare_processor()
    rc = _ConsumeRecorder(remaining=2)
    p.runtime_config = rc
    p._kv_dump_jobs = []

    def _handle_and_queue(*_a, **_k):
        p.queue_kv_dump(
            {
                "req_id": "r1",
                "incident_type": MANUAL_TRIGGER_TYPE,
                "consume_quota": False,
                "arm_id": "t",
                "wave": 3,
            }
        )

    p.action_executor = MagicMock()
    p.action_executor.handle.side_effect = _handle_and_queue
    p._get_report_tokenizer = MagicMock(return_value=None)
    p._batch_request_io_rows = MagicMock(return_value=[("r1", 0)])
    p.wave_tracker = MagicMock()
    p.wave_tracker.current_wave.return_value = 3
    with patch(
        "vllm_ascend.runtime_guard.processor_report.is_action_leader_rank",
        return_value=True,
    ), patch(
        "vllm_ascend.runtime_guard.processor_report.RequestIoSnapshotManager"
    ) as io_cls, patch(
        "vllm_ascend.runtime_guard.processor_report.block_ids_for_request",
        return_value=[0],
    ):
        io = MagicMock()
        snap = MagicMock()
        snap.as_detail_fields.return_value = {}
        io.snapshot.return_value = snap
        io_cls.get.return_value = io
        p._handle_manual_trigger(
            TriggerEvent(trigger_type=MANUAL_TRIGGER_TYPE, req_id=MANUAL_TRIGGER_REQ_ID)
        )
    assert p.action_executor.handle.called
    assert rc.consume_calls == 1
    assert rc.remaining == 1


def test_v19a3_manual_trigger_consumes_even_when_dump_not_armed():
    """Arm failure still burns manual_dump; DumpKvAction writes dump_skipped."""
    from vllm_ascend.runtime_guard.manual_trigger import (
        MANUAL_TRIGGER_REQ_ID,
        MANUAL_TRIGGER_TYPE,
        TriggerEvent,
    )

    p = _bare_processor()
    rc = _ConsumeRecorder(remaining=2)
    p.runtime_config = rc
    p._kv_dump_jobs = []
    p.action_executor = MagicMock()  # handle does not queue
    p._get_report_tokenizer = MagicMock(return_value=None)
    p._batch_request_io_rows = MagicMock(return_value=[("r1", 0)])
    p.wave_tracker = MagicMock()
    p.wave_tracker.current_wave.return_value = 1
    with patch(
        "vllm_ascend.runtime_guard.processor_report.is_action_leader_rank",
        return_value=True,
    ), patch(
        "vllm_ascend.runtime_guard.processor_report.RequestIoSnapshotManager"
    ) as io_cls, patch(
        "vllm_ascend.runtime_guard.processor_report.block_ids_for_request",
        return_value=[],
    ):
        io = MagicMock()
        snap = MagicMock()
        snap.as_detail_fields.return_value = {}
        io.snapshot.return_value = snap
        io_cls.get.return_value = io
        p._handle_manual_trigger(
            TriggerEvent(trigger_type=MANUAL_TRIGGER_TYPE, req_id=MANUAL_TRIGGER_REQ_ID)
        )
    assert p.action_executor.handle.called
    assert rc.consume_calls == 1
    assert rc.remaining == 1


def test_v19b_non_manual_incident_does_not_consume():
    from vllm_ascend.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.runtime_guard.incident import Incident

    rc = _ConsumeRecorder(remaining=2)
    kv_reader = MagicMock()
    ctx = _dump_ctx(
        Incident(
            incident_type="token_repeat",
            req_id="r1",
            consume_quota=True,
            block_ids=[0],
        ),
        rc,
        kv_reader,
        _quota_stub(),
    )
    DumpKvAction().run(ctx)
    assert rc.consume_calls == 0


def test_v19c_empty_block_ids_manual_still_queues(monkeypatch):
    """Manual arm with empty block_ids still queues (resolve at flush)."""
    from vllm_ascend.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.runtime_guard.incident import Incident
    from vllm_ascend.runtime_guard.manual_trigger import MANUAL_TRIGGER_TYPE

    rc = _ConsumeRecorder(remaining=2)
    kv_reader = MagicMock()
    quota = _quota_stub()
    ctx = _dump_ctx(
        Incident(incident_type=MANUAL_TRIGGER_TYPE, req_id="__manual_trigger__", consume_quota=False),
        rc,
        kv_reader,
        quota,
        req_ids=["r1"],
    )
    monkeypatch.setattr(
        "vllm_ascend.runtime_guard.kv_block_meta.block_ids_for_request",
        lambda *_a, **_k: [],
    )
    DumpKvAction().run(ctx)
    assert rc.consume_calls == 0  # manual count consumed in processor, not here
    quota.try_consume.assert_called_once_with(consume_quota=False)
    assert ctx.runner.runtime_guard.queue_kv_dump.call_count == 1

def test_v19d_dump_kv_queues_jobs_on_leader():
    from vllm_ascend.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.runtime_guard.incident import Incident

    guard = MagicMock()
    rc = _ConsumeRecorder(remaining=2)
    kv_reader = MagicMock()
    ctx = _dump_ctx(
        Incident(
            incident_type="token_repeat",
            req_id="r1",
            consume_quota=True,
            block_ids=[0],
        ),
        rc,
        kv_reader,
        _quota_stub(),
    )
    ctx.runner.runtime_guard = guard
    with patch(
        "vllm_ascend.runtime_guard.action.actions.runner_tp_rank",
        return_value=0,
    ):
        DumpKvAction().run(ctx)
    guard.queue_kv_dump.assert_called_once()
    job = guard.queue_kv_dump.call_args[0][0]
    assert job["req_id"] == "r1"
    assert job["incident_type"] == "token_repeat"
    assert "block_ids" not in job


def test_v19e_queue_fail_refunds_quota():
    from vllm_ascend.runtime_guard.action.actions import DumpKvAction
    from vllm_ascend.runtime_guard.incident import Incident

    rc = _ConsumeRecorder(remaining=2)
    kv_reader = MagicMock()
    quota = _quota_stub()
    ctx = _dump_ctx(
        Incident(
            incident_type="token_repeat",
            req_id="r1",
            consume_quota=True,
            block_ids=[0],
        ),
        rc,
        kv_reader,
        quota,
    )
    # Drop queue so _queue_kv_dumps returns False after try_consume.
    ctx.runner.runtime_guard = SimpleNamespace()
    with patch(
        "vllm_ascend.runtime_guard.action.actions.runner_tp_rank",
        return_value=0,
    ):
        DumpKvAction().run(ctx)
    quota.try_consume.assert_called_once_with(consume_quota=True)
    quota.refund.assert_called_once_with(consume_quota=True)


def test_v19f_multi_job_arm_refunds_once_when_all_fail(tmp_path: Path):
    """Same arm_id + consume_quota: all skips → single refund (not N)."""
    from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor
    from vllm_ascend.runtime_guard.request_state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    try:
        RequestGuardStore.get().mark_finished(["r1", "r2"], wave=0)
        quota = MagicMock()
        runner = SimpleNamespace(tp_rank=0, dp_rank=0, dcp_rank=0, dcp_size=1)
        proc = SimpleNamespace(
            runner=runner,
            runtime_config=SimpleNamespace(dump_root=lambda: str(tmp_path)),
            action_executor=SimpleNamespace(_kv_reader=MagicMock(), submit_heavy=None),
            quota=quota,
        )
        arm = "arm-shared"
        jobs = [
            {
                "req_id": "r1",
                "incident_type": "token_repeat",
                "consume_quota": True,
                "arm_id": arm,
            },
            {
                "req_id": "r2",
                "incident_type": "token_repeat",
                "consume_quota": True,
                "arm_id": arm,
            },
        ]
        RuntimeGuardProcessor._run_kv_dumps(proc, jobs)
        quota.refund.assert_called_once_with(consume_quota=True)
    finally:
        RequestGuardStore.reset_for_tests()


# ------------------------------------------- default-path merge (V21)


def test_v21_bootstrap_overwrites_existing_file_with_defaults(tmp_path: Path, monkeypatch):
    import vllm_ascend.runtime_config.config as cfg

    cfg_file = tmp_path / "runtime" / "config" / "runtime_config.json"
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text('{"detector": {"token_repeat": {"enabled": true}}}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rc = cfg.RuntimeConfig(config_path=None)
    assert rc.detector_get("token_repeat", "enabled", True) is False
    assert rc.ensure_persisted() is True
    on_disk = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert on_disk["detector"]["token_repeat"]["enabled"] is False
    assert rc.detector_get("token_repeat", "window", 0) == int(
        cfg._DEFAULTS["detector"]["token_repeat"]["window"]
    )
    assert rc.detector_get("output_substring", "enabled", True) is False


def test_v21b_default_path_missing_file_pure_defaults(tmp_path: Path, monkeypatch):
    import vllm_ascend.runtime_config.config as cfg

    monkeypatch.chdir(tmp_path)
    rc = cfg.RuntimeConfig(config_path=None)
    assert rc.detector_get("token_repeat", "enabled", True) is False
    assert rc.detector_get("output_substring", "enabled", True) is False


def test_v21c_pre_bootstrap_file_ignored_on_reload(tmp_path: Path):
    """A JSON written before this process's bootstrap must not leak in via the
    first hot-reload (the race where the background reloader reads the stale
    file before ``ensure_persisted`` overwrites it)."""
    import vllm_ascend.runtime_config.config as cfg

    cfg_file = tmp_path / "runtime_config.json"
    # Hand-edit written well before bootstrap: token_repeat enabled.
    cfg_file.write_text('{"detector": {"token_repeat": {"enabled": true}}}', encoding="utf-8")
    past = time.time() - 100
    os.utime(cfg_file, (past, past))

    rc = cfg.RuntimeConfig(
        config_path=str(cfg_file),
        report_dir=tmp_path / "report",
        ensure_file=False,  # production: persist deferred to ensure_persisted
        reload_interval_seconds=1,
        sync_mode="file",
    )
    # Bootstrap ignores the stale file: detector stays at its default (off).
    assert rc.detector_get("token_repeat", "enabled", True) is False
    # A non-forced reload must NOT apply the pre-bootstrap file either.
    assert rc.reload(force=False) is False
    assert rc.detector_get("token_repeat", "enabled", True) is False

    # Writer persists the effective startup config (defaults), overwriting stale.
    assert rc.ensure_persisted() is True
    on_disk = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert on_disk["detector"]["token_repeat"]["enabled"] is False

    # A genuine post-bootstrap hot-reload edit IS picked up.
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    data["detector"]["token_repeat"]["enabled"] = True
    cfg_file.write_text(json.dumps(data), encoding="utf-8")
    future = time.time() + 10
    os.utime(cfg_file, (future, future))
    assert rc.reload(force=False) is True
    assert rc.detector_get("token_repeat", "enabled", False) is True


# ------------------------------------------- configurable dump root (V23)


def test_v23_dump_root_default_json_and_startup_seed(tmp_path: Path, monkeypatch):
    import json

    import vllm_ascend.runtime_config.config as cfg

    monkeypatch.chdir(tmp_path)
    cfg_file = tmp_path / "runtime" / "config" / "runtime_config.json"
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text("{}", encoding="utf-8")

    # Default: derived <report_dir>/kv_cache.
    rc = cfg.RuntimeConfig(config_path=str(cfg_file))
    assert rc.dump_root() == rc.report_dir / "kv_cache"

    # JSON key wins and is hot-reload visible.
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    data["dump"] = {"dump_dir": str(tmp_path / "custom_dumps")}
    cfg_file.write_text(json.dumps(data), encoding="utf-8")
    assert rc.reload()
    assert rc.dump_root() == (tmp_path / "custom_dumps").resolve()

    # Startup dump_dir always seeds (preexisting file ignored at bootstrap).
    cfg_file.write_text("{}", encoding="utf-8")
    seeded = cfg.RuntimeConfig(
        config_path=str(cfg_file),
        dump_dir=str(tmp_path / "from_startup"),
        ensure_file=True,
        reload_interval_seconds=1,
        sync_mode="file",
    )
    assert seeded.dump_root() == (tmp_path / "from_startup").resolve()
    on_disk = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert on_disk["dump"]["dump_dir"] == str(tmp_path / "from_startup")
    # Clear via hot-reload after start.
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    data["dump"]["dump_dir"] = None
    cfg_file.write_text(json.dumps(data), encoding="utf-8")
    assert seeded.reload(force=True)
    assert seeded.dump_root() == seeded.report_dir / "kv_cache"


def test_v23b_report_writer_dump_dir_follows_provider(tmp_path: Path):
    from vllm_ascend.runtime_guard.report import ReportWriter

    root = tmp_path / "kv_root"

    def provider() -> str:
        return str(root)

    w = ReportWriter(tmp_path / "reports", dump_root_provider=provider)
    path = w.write(incident_type="token_repeat", req_id="r-1", detail={"x": 1})
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["dump_dir"] == str(root / "token_repeat" / "r-1")
    w2 = ReportWriter(tmp_path / "reports")
    path2 = w2.write(incident_type="token_repeat", req_id="r-1", detail={"x": 1}, dump_arm_wave=12)
    rec2 = json.loads(path2.read_text(encoding="utf-8"))
    assert rec2["dump_dir"] == str(tmp_path / "reports" / "kv_cache" / "token_repeat" / "r-1")
    assert rec2["dump_arm_wave"] == 12


def test_v23c_manual_trigger_report_dump_dir_uses_real_req_ids(tmp_path: Path):
    """Per-req manual reports: top-level req_id/dump_dir align (W3-2 / K-13)."""
    from vllm_ascend.runtime_guard.manual_trigger import (
        MANUAL_TRIGGER_REQ_ID,
        MANUAL_TRIGGER_TYPE,
    )
    from vllm_ascend.runtime_guard.report import ReportWriter

    root = tmp_path / "kv"
    w = ReportWriter(tmp_path / "reports", dump_root_provider=lambda: str(root))
    # One write per real req (as ``_handle_manual_trigger`` does now).
    path_a = w.write(
        incident_type=MANUAL_TRIGGER_TYPE,
        req_id="cmpl-a",
        dump_arm_wave=1,
        dump_attempted=True,
        detail={
            "trigger_req_id": MANUAL_TRIGGER_REQ_ID,
            "num_requests_in_batch": 2,
            "req_id": "cmpl-a",
            "prompt_token_count": 1,
            "block_ids": [1, 2],
        },
    )
    path_b = w.write(
        incident_type=MANUAL_TRIGGER_TYPE,
        req_id="cmpl-b",
        dump_arm_wave=1,
        dump_attempted=True,
        detail={
            "trigger_req_id": MANUAL_TRIGGER_REQ_ID,
            "num_requests_in_batch": 2,
            "req_id": "cmpl-b",
            "prompt_token_count": 2,
            "block_ids": [3],
        },
    )
    assert path_a is not None and path_b is not None
    rec_a = json.loads(path_a.read_text(encoding="utf-8"))
    rec_b = json.loads(path_b.read_text(encoding="utf-8"))
    assert rec_a["req_id"] == "cmpl-a"
    assert rec_a["dump_dir"] == str(root / MANUAL_TRIGGER_TYPE / "cmpl-a")
    assert rec_a["block_ids"] == [1, 2]
    assert "dump_dirs" not in rec_a
    assert rec_a["detail"]["trigger_req_id"] == MANUAL_TRIGGER_REQ_ID
    assert rec_b["req_id"] == "cmpl-b"
    assert rec_b["dump_dir"] == str(root / MANUAL_TRIGGER_TYPE / "cmpl-b")
    assert rec_b["block_ids"] == [3]


def test_v23d_manual_handle_writes_one_report_per_req(tmp_path: Path):
    """``_handle_manual_trigger`` one ActionExecutor.handle(report) per batch row."""
    from vllm_ascend.runtime_guard.manual_trigger import (
        MANUAL_TRIGGER_REQ_ID,
        MANUAL_TRIGGER_TYPE,
        TriggerEvent,
    )

    p = _bare_processor()
    rc = _ConsumeRecorder(remaining=1)
    p.runtime_config = rc
    p.action_executor = MagicMock()
    p._get_report_tokenizer = MagicMock(return_value=None)
    p._batch_request_io_rows = MagicMock(return_value=[("cmpl-a", 0), ("cmpl-b", 1)])
    p.wave_tracker = MagicMock()
    p.wave_tracker.current_wave.return_value = 5
    with patch(
        "vllm_ascend.runtime_guard.processor_report.is_action_leader_rank",
        return_value=True,
    ), patch(
        "vllm_ascend.runtime_guard.processor_report.RequestIoSnapshotManager"
    ) as io_cls, patch(
        "vllm_ascend.runtime_guard.processor_report.block_ids_for_request",
        side_effect=lambda _r, rid, *_a, **_k: [10] if rid == "cmpl-a" else [20],
    ):
        p.runtime_config.report_include_block_ids = lambda: True  # type: ignore[method-assign]
        io = MagicMock()
        snap = MagicMock()
        snap.as_detail_fields.return_value = {"prompt_token_count": 1}
        io.snapshot.return_value = snap
        io_cls.get.return_value = io
        p._handle_manual_trigger(
            TriggerEvent(trigger_type=MANUAL_TRIGGER_TYPE, req_id=MANUAL_TRIGGER_REQ_ID),
            inject_dump_kv=False,
            consume=False,
        )
    assert p.action_executor.handle.call_count == 2
    req_ids = [c.args[0].req_id for c in p.action_executor.handle.call_args_list]
    assert req_ids == ["cmpl-a", "cmpl-b"]
    for c in p.action_executor.handle.call_args_list:
        assert c.kwargs.get("action_override") == ["report"]
        assert c.kwargs.get("inject_manual_dump_kv") is False
        assert c.kwargs.get("write_report") is True


# ------------------------------------------- rank gate TP0-only (V25)


def test_v25b_dump_rank_tag_uses_global_dp_rank(monkeypatch):
    """External multi-DP: get_dp_group().rank_in_group is 0; use runner.dp_rank."""
    import vllm_ascend.runtime_guard.rank_gate as rank_gate

    monkeypatch.setattr(
        rank_gate,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1),
    )
    monkeypatch.setattr(
        rank_gate,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=0, is_last_rank=True),
    )
    # Even if get_dp_group would return 0, runner.dp_rank=1 must win.
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_dp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1),
        raising=False,
    )
    runner = SimpleNamespace(dp_rank=1, tp_rank=0, dcp_rank=0, dcp_size=1, pcp_rank=0)
    assert rank_gate.runner_dp_rank(runner) == 1
    assert rank_gate.dump_rank_tag(runner).startswith("dp1_tp")

    monkeypatch.setattr(
        rank_gate, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=2)
    )
    runner0 = SimpleNamespace(use_async_scheduling=True, tp_rank=0, tp_size=2)
    assert rank_gate.anomaly_check_rank_skip_reason(runner0) is None

    monkeypatch.setattr(
        rank_gate, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    assert rank_gate.anomaly_check_rank_skip_reason(runner0) == "not last PP rank"


def test_v25b_manager_skips_when_gated(tmp_path, monkeypatch):
    import vllm_ascend.runtime_config.config as cfg

    monkeypatch.chdir(tmp_path)
    rc = cfg.RuntimeConfig(config_path=None)
    rc._data["detector"]["logits_finite"]["enabled"] = True
    runner = SimpleNamespace(tp_rank=0, tp_size=1, input_batch=SimpleNamespace(req_ids=[]))

    gated = DetectorManager(runtime_config=rc, runner=runner, detection_gate=lambda: False)
    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        return []

    gated._logits_finite_det.check_all = _boom
    assert gated.check_before_sample(logits=None, input_batch=None) == []
    assert calls["n"] == 0

    open_mgr = DetectorManager(runtime_config=rc, runner=runner, detection_gate=lambda: True)
    open_mgr._logits_finite_det.check_all = _boom
    assert open_mgr.check_before_sample(logits=None, input_batch=None) == []
    assert calls["n"] == 1


# ------------------------------------------------- V26 after-sample CPU queue


def test_v26a_after_sample_cpu_does_not_block_return():
    from vllm_ascend.runtime_guard.detector.manager import AfterSampleCpuSnapshot
    from vllm_ascend.runtime_guard.incident import Incident
    from vllm_ascend.runtime_guard.request_state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    p = object.__new__(RuntimeGuardProcessor)
    p.detectors = MagicMock()
    p.wave_tracker = None
    p.runner = None
    p._handle_alert = MagicMock()
    p._reap_finished_requests = MagicMock()
    p._last_input_batch = None
    snap = AfterSampleCpuSnapshot(
        req_ids=["r1"],
        sampled_token_ids=[[1]],
        skip_req_ids=None,
    )
    logits = Incident(incident_type="logits_finite", req_id="r1")
    p.detectors.after_sample_hot_path.return_value = ([logits], snap)
    started = threading.Event()
    release = threading.Event()

    def _cpu(_snap):
        started.set()
        release.wait(timeout=2.0)
        return []

    p.detectors.run_after_sample_cpu.side_effect = _cpu
    q = ActionQueue(maxsize=8, name="ut-cpu-detect")
    p.action_executor = SimpleNamespace(action_queue=q)
    q.start()
    try:
        p.check_after_sample(sampled_token_ids=[[1]], req_ids=["r1"])
        handled = p._handle_alert.call_args[0][0]
        assert handled.incident_type == "logits_finite"
        assert started.wait(timeout=1.0)
        assert not release.is_set()
    finally:
        release.set()
        q.stop()
        RequestGuardStore.reset_for_tests()


def test_v26b_cpu_detect_dropped_not_inline():
    q = ActionQueue(maxsize=1, name="ut-cpu-drop")
    q.start()
    gate = threading.Event()
    started = threading.Event()
    try:
        def _block() -> None:
            started.set()
            gate.wait(timeout=2.0)

        q.submit(_block)
        assert started.wait(timeout=1.0)
        q.submit(lambda: None)  # occupies the only queue slot
        ran: list[int] = []
        assert q.submit(lambda: ran.append(1), drop_on_full=True) is False
        assert ran == []
    finally:
        gate.set()
        q.stop()
