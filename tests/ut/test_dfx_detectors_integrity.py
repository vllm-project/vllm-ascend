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

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.dfx.detector.block_kv import BlockKvDetector
from vllm_ascend.dfx.detector.logits_finite import LogitsFiniteDetector
from vllm_ascend.dfx.detector.position_alignment import (
    PositionAlignmentDetector,
    num_computed_before,
)
from vllm_ascend.dfx.dfx_types import ILL_TYPE_NAN
from vllm_ascend.dfx.kv_block_meta import KvBlockMetaTracker, block_ids_for_request
from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig


@pytest.fixture(autouse=True)
def _reset_kv_tracker():
    KvBlockMetaTracker.reset_for_tests()
    yield
    KvBlockMetaTracker.reset_for_tests()


def test_block_kv_wave_regression(tmp_path):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["block_kv"]["enabled"] = True
    det = BlockKvDetector(dfx_config=cfg, runner=SimpleNamespace())
    tracker = KvBlockMetaTracker.get()
    tracker.record_writes("req-a", [1], wave=5)
    alerts = det.check_writes("req-b", [1], wave=3)
    assert len(alerts) == 1
    assert alerts[0].anomaly_type == "block_kv"
    assert alerts[0].detail["num_violations"] == 1
    assert alerts[0].detail["violations"][0]["violation"] == "wave_regression"
    assert alerts[0].detail["violations"][0]["prev_writer_req_id"] == "req-a"
    assert alerts[0].detail["violations"][0]["new_writer_req_id"] == "req-b"


def test_block_kv_same_wave_writer_conflict(tmp_path):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["block_kv"]["enabled"] = True
    det = BlockKvDetector(dfx_config=cfg, runner=SimpleNamespace())
    tracker = KvBlockMetaTracker.get()
    tracker.record_writes("req-a", [2], wave=4)
    alerts = det.check_writes("req-b", [2], wave=4)
    assert len(alerts) == 1
    assert alerts[0].detail["violations"][0]["violation"] == "same_wave_writer_conflict"


def test_block_kv_multi_block_one_alert(tmp_path):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["block_kv"]["enabled"] = True
    det = BlockKvDetector(dfx_config=cfg, runner=SimpleNamespace())
    tracker = KvBlockMetaTracker.get()
    tracker.record_writes("req-a", [10, 11], wave=5)
    alerts = det.check_writes("req-b", [10, 11], wave=3)
    assert len(alerts) == 1
    assert alerts[0].detail["num_violations"] == 2
    assert {v["block_id"] for v in alerts[0].detail["violations"]} == {10, 11}


def test_position_alignment_detects_wrong_start(tmp_path):
    """Wave-before computed=4 → expected position starts at 4, not 3."""
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["position_alignment"]["enabled"] = True
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=["r0"],
            req_id_to_index={"r0": 0},
            num_computed_tokens_cpu=np.array([4], dtype=np.int32),
        ),
        query_start_loc=SimpleNamespace(np=np.array([0, 1], dtype=np.int64)),
    )
    det = PositionAlignmentDetector(dfx_config=cfg, runner=runner)
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"r0": 1}, total_num_scheduled_tokens=1)
    positions = torch.tensor([0], dtype=torch.int64)
    alerts = det.check_all(
        scheduler_output=scheduler_output,
        positions=positions,
        total_scheduled=1,
    )
    assert len(alerts) == 1
    assert alerts[0].anomaly_type == "position_alignment"
    assert alerts[0].detail["violation"] == "wrong_start"
    assert alerts[0].detail["expected_start"] == 4


def test_position_alignment_ok_when_aligned(tmp_path):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["position_alignment"]["enabled"] = True
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=["r0"],
            req_id_to_index={"r0": 0},
            num_computed_tokens_cpu=np.array([4], dtype=np.int32),
        ),
        query_start_loc=SimpleNamespace(np=np.array([0, 1], dtype=np.int64)),
    )
    det = PositionAlignmentDetector(dfx_config=cfg, runner=runner)
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"r0": 1}, total_num_scheduled_tokens=1)
    positions = torch.tensor([4], dtype=torch.int64)
    alerts = det.check_all(
        scheduler_output=scheduler_output,
        positions=positions,
        total_scheduled=1,
    )
    assert alerts == []


def test_position_alignment_uses_v2_num_computed_tokens_np(tmp_path):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["position_alignment"]["enabled"] = True
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=["r0"],
            req_id_to_index={"r0": 0},
            num_computed_tokens_np=np.array([10], dtype=np.int32),
            query_start_loc_np=np.array([0, 1], dtype=np.int64),
        ),
    )
    det = PositionAlignmentDetector(dfx_config=cfg, runner=runner)
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"r0": 1}, total_num_scheduled_tokens=1)
    assert (
        det.check_all(
            scheduler_output=scheduler_output,
            positions=torch.tensor([10], dtype=torch.int64),
            total_scheduled=1,
        )
        == []
    )
    alerts = det.check_all(
        scheduler_output=scheduler_output,
        positions=torch.tensor([9], dtype=torch.int64),
        total_scheduled=1,
    )
    assert len(alerts) == 1
    assert alerts[0].detail["expected_start"] == 10


def test_position_alignment_uses_explicit_v2_input_batch(tmp_path):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["position_alignment"]["enabled"] = True
    runner = SimpleNamespace(input_batch=None)
    input_batch = SimpleNamespace(
        req_ids=["r0"],
        num_computed_tokens_np=np.array([10], dtype=np.int32),
        query_start_loc_np=np.array([0, 1], dtype=np.int64),
    )
    det = PositionAlignmentDetector(dfx_config=cfg, runner=runner)
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"r0": 1}, total_num_scheduled_tokens=1)

    assert (
        det.check_all(
            scheduler_output=scheduler_output,
            positions=torch.tensor([10], dtype=torch.int64),
            total_scheduled=1,
            input_batch=input_batch,
        )
        == []
    )


def test_logits_finite_detects_nan(tmp_path):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["logits_finite"]["enabled"] = True
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["r0", "r1"]),
        logits_indices=torch.tensor([0, 1]),
        query_start_loc=SimpleNamespace(np=np.array([0, 1, 2], dtype=np.int64)),
    )
    det = LogitsFiniteDetector(dfx_config=cfg, runner=runner)
    logits = torch.tensor([[1.0, 2.0], [float("nan"), 1.0]])
    alerts = det.check_all(logits=logits, logits_indices=runner.logits_indices)
    assert len(alerts) == 1
    assert alerts[0].anomaly_type == "logits_finite"
    assert alerts[0].ill_type == ILL_TYPE_NAN
    assert alerts[0].detail["finite_kind"] == "nan"
    assert alerts[0].req_id == "r1"


@pytest.mark.parametrize(
    "bad, kind",
    [
        (float("inf"), "pos_inf"),
        (float("-inf"), "neg_inf"),
    ],
)
def test_logits_finite_detects_inf(tmp_path, bad, kind):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["logits_finite"]["enabled"] = True
    runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["r0"]))
    det = LogitsFiniteDetector(dfx_config=cfg, runner=runner)
    logits = torch.tensor([[1.0, bad]])
    alerts = det.check_all(logits=logits, logits_indices=None)
    assert len(alerts) == 1
    assert alerts[0].ill_type == ILL_TYPE_NAN
    assert alerts[0].detail["finite_kind"] == kind


def test_logits_finite_maps_explicit_v2_input_batch(tmp_path):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["logits_finite"]["enabled"] = True
    runner = SimpleNamespace(input_batch=None)
    input_batch = SimpleNamespace(
        req_ids=["r0", "r1"],
        query_start_loc_np=np.array([0, 1, 2], dtype=np.int64),
    )
    det = LogitsFiniteDetector(dfx_config=cfg, runner=runner)
    alerts = det.check_all(
        logits=torch.tensor([[1.0, 2.0], [float("nan"), 1.0]]),
        logits_indices=torch.tensor([0, 1]),
        input_batch=input_batch,
    )

    assert len(alerts) == 1
    assert alerts[0].req_id == "r1"
    assert alerts[0].req_idx == 1


def test_logits_finite_skips_when_finite(tmp_path):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["logits_finite"]["enabled"] = True
    runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["r0"]))
    det = LogitsFiniteDetector(dfx_config=cfg, runner=runner)
    logits = torch.tensor([[1.0, 2.0]])
    assert det.check_all(logits=logits, logits_indices=None) == []


def test_block_ids_for_request_uses_v2_state_index():
    block_rows = np.array(
        [
            [10, 11, 0],
            [20, 21, 22],
            [30, 0, 0],
        ],
        dtype=np.int32,
    )
    runner = SimpleNamespace(
        requests=None,
        input_batch=None,
        block_tables=SimpleNamespace(
            num_blocks=SimpleNamespace(np=np.array([[2, 3, 1]], dtype=np.int32)),
            block_tables=[SimpleNamespace(np=block_rows)],
        ),
    )
    input_batch = SimpleNamespace(
        req_ids=["r1"],
        idx_mapping_np=np.array([1], dtype=np.int32),
    )

    assert block_ids_for_request(
        runner,
        "r1",
        req_idx=0,
        input_batch=input_batch,
    ) == [20, 21, 22]


def test_block_ids_for_request_falls_back_to_execute_model_state_batch():
    """Report enrichment has no explicit batch; V2 must use execute_model_state."""
    block_rows = np.array([[20, 21, 0]], dtype=np.int32)
    input_batch = SimpleNamespace(
        req_ids=["r1"],
        idx_mapping_np=np.array([0], dtype=np.int32),
    )
    runner = SimpleNamespace(
        requests=None,
        input_batch=None,
        execute_model_state=SimpleNamespace(input_batch=input_batch),
        block_tables=SimpleNamespace(
            num_blocks=SimpleNamespace(np=np.array([[2]], dtype=np.int32)),
            block_tables=[SimpleNamespace(np=block_rows)],
        ),
    )
    assert block_ids_for_request(runner, "r1", req_idx=0) == [20, 21]


def test_v2_execute_model_notes_kv_only_on_success():
    """note_kv_block_writes must not run from finally after a failed forward."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "vllm_ascend/worker/v2/model_runner.py").read_text()
    i_fn = src.index("def execute_model(")
    i_next = src.index("\n    def sample(", i_fn)
    body = src[i_fn:i_next]
    i_note = body.index("self.dfx.note_kv_block_writes(")
    # Success-only: note sits in the else: branch immediately after except Exception.
    before_note = body[:i_note]
    i_except = before_note.rindex("except Exception:")
    i_else = before_note.rindex("else:")
    i_finally = body.index("finally:", i_note)
    i_finalize = body.index("self.dfx.finalize_dump_data(", i_finally)
    assert i_except < i_else < i_note < i_finally < i_finalize
    assert "self._dfx_scheduler_output = None" in body[i_except:i_else]


def test_runtime_config_knows_new_detector_sections(tmp_path):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    for name in ("block_kv", "position_alignment", "logits_finite"):
        assert name in DfxRuntimeConfig.DETECTOR_SECTIONS
        cfg._data["detector"][name]["enabled"] = True
    assert cfg.any_detector_enabled() is True


def test_manager_kv_block_runs_when_pending_dump(tmp_path):
    """block_kv must not be gated out after another detector arms dump."""
    from vllm_ascend.dfx.detector.manager import DetectorManager

    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["block_kv"]["enabled"] = True
    cfg._data["dump"]["enabled"] = True
    runner = SimpleNamespace(tp_rank=0)
    pending = {"busy": True}

    def gate(*, ignore_dump_busy: bool = False) -> bool:
        return not (pending["busy"] and not ignore_dump_busy)

    def skip(*, ignore_dump_busy: bool = False) -> str | None:
        if pending["busy"] and not ignore_dump_busy:
            return "pending_dump already armed"
        return None

    mgr = DetectorManager(
        dfx_config=cfg,
        runner=runner,
        detection_gate=gate,
        detection_skip_reason=skip,
    )
    tracker = KvBlockMetaTracker.get()
    tracker.record_writes("req-a", [7], wave=5)
    alerts = mgr.check_kv_block_writes("req-b", [7], wave=2)
    assert len(alerts) == 1
    assert alerts[0].detail["num_violations"] == 1


def test_same_step_stop_gate_skips_position_and_kv_after_logits(tmp_path):
    """Logits alert in check_before_sample must stop position + later block_kv."""
    from vllm_ascend.dfx.detector.manager import DetectorManager
    from vllm_ascend.dfx.request_state import RequestDfxStore

    RequestDfxStore.reset_for_tests()
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["stop_after_alert"] = True
    cfg._data["detector"]["logits_finite"]["enabled"] = True
    cfg._data["detector"]["position_alignment"]["enabled"] = True
    cfg._data["detector"]["block_kv"]["enabled"] = True
    input_batch = SimpleNamespace(
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
        num_computed_tokens_cpu=np.array([0], dtype=np.int32),
        query_start_loc_np=np.array([0, 1], dtype=np.int64),
    )
    runner = SimpleNamespace(
        tp_rank=0,
        input_batch=input_batch,
        query_start_loc=SimpleNamespace(np=np.array([0, 1], dtype=np.int64)),
    )
    mgr = DetectorManager(
        dfx_config=cfg,
        runner=runner,
        detection_gate=lambda **_k: True,
    )
    so = SimpleNamespace(num_scheduled_tokens={"r0": 1}, total_num_scheduled_tokens=1)
    # Wrong position (expected 0) would alert if stop gate failed mid-step.
    positions = torch.tensor([9], dtype=torch.int64)
    logits = torch.tensor([[float("nan"), 1.0]])
    alerts = mgr.check_before_sample(
        scheduler_output=so,
        logits=logits,
        positions=positions,
        total_scheduled_tokens=1,
        logits_indices=None,
        input_batch=input_batch,
    )
    assert [a.anomaly_type for a in alerts] == ["logits_finite"]
    assert "r0" in RequestDfxStore.get().stopped_req_ids()

    tracker = KvBlockMetaTracker.get()
    tracker.record_writes("other", [3], wave=9)
    assert mgr.check_kv_block_writes("r0", [3], wave=1) == []
    RequestDfxStore.reset_for_tests()


def test_num_computed_before_zero_is_valid():
    runner = SimpleNamespace(requests={"r0": SimpleNamespace(num_computed_tokens=0)})
    batch = SimpleNamespace(num_computed_tokens_np=np.array([0], dtype=np.int32))
    assert num_computed_before(runner, "r0", 0, 8, batch) == 0


def test_num_computed_before_prefers_batch_over_requests_zero():
    runner = SimpleNamespace(requests={"r0": SimpleNamespace(num_computed_tokens=0)})
    batch = SimpleNamespace(num_computed_tokens_np=np.array([10], dtype=np.int32))
    assert num_computed_before(runner, "r0", 0, 1, batch) == 10


def test_num_computed_before_missing_returns_none():
    runner = SimpleNamespace(requests=None, input_batch=None)
    assert num_computed_before(runner, "r0", 0, 1, None) is None


def test_position_alignment_skips_unknown_computed(tmp_path):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["position_alignment"]["enabled"] = True
    runner = SimpleNamespace(requests=None, input_batch=None)
    input_batch = SimpleNamespace(
        req_ids=["r0"],
        query_start_loc_np=np.array([0, 1], dtype=np.int64),
    )
    det = PositionAlignmentDetector(dfx_config=cfg, runner=runner)
    so = SimpleNamespace(num_scheduled_tokens={"r0": 1}, total_num_scheduled_tokens=1)
    assert (
        det.check_all(
            scheduler_output=so,
            positions=torch.tensor([0], dtype=torch.int64),
            total_scheduled=1,
            input_batch=input_batch,
        )
        == []
    )


def test_position_alignment_skips_mrope_nd(tmp_path):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["position_alignment"]["enabled"] = True
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=["r0"],
            num_computed_tokens_np=np.array([0], dtype=np.int32),
            query_start_loc_np=np.array([0, 1], dtype=np.int64),
        )
    )
    det = PositionAlignmentDetector(dfx_config=cfg, runner=runner)
    so = SimpleNamespace(num_scheduled_tokens={"r0": 1}, total_num_scheduled_tokens=1)
    assert (
        det.check_all(
            scheduler_output=so,
            positions=torch.zeros(3, 1, dtype=torch.int64),
            total_scheduled=1,
        )
        == []
    )


def _ungated_manager(cfg, runner):
    from vllm_ascend.dfx.detector.manager import DetectorManager

    return DetectorManager(
        dfx_config=cfg,
        runner=runner,
        detection_gate=lambda **_k: True,
    )


def test_manager_v2_check_before_sample_nan_with_runner_batch_none(tmp_path):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["logits_finite"]["enabled"] = True
    runner = SimpleNamespace(input_batch=None)
    mgr = _ungated_manager(cfg, runner)
    input_batch = SimpleNamespace(
        req_ids=["r0", "r1"],
        query_start_loc_np=np.array([0, 1, 2], dtype=np.int64),
    )
    so = SimpleNamespace(num_scheduled_tokens={"r0": 1, "r1": 1}, total_num_scheduled_tokens=2)
    alerts = mgr.check_before_sample(
        scheduler_output=so,
        logits=torch.tensor([[1.0, 2.0], [float("nan"), 1.0]]),
        positions=torch.tensor([0, 1], dtype=torch.int64),
        total_scheduled_tokens=2,
        logits_indices=torch.tensor([0, 1]),
        input_batch=input_batch,
    )
    finite = [a for a in alerts if a.anomaly_type == "logits_finite"]
    assert len(finite) == 1
    assert finite[0].req_id == "r1"


def test_manager_check_before_sample_nan_not_large_finite(tmp_path):
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["logits_finite"]["enabled"] = True
    runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["r0"]))
    mgr = _ungated_manager(cfg, runner)
    so = SimpleNamespace(num_scheduled_tokens={"r0": 1}, total_num_scheduled_tokens=1)
    finite = mgr.check_before_sample(
        scheduler_output=so,
        logits=torch.tensor([[1.0, -1e9]]),
        logits_indices=None,
        input_batch=runner.input_batch,
    )
    assert [a for a in finite if a.anomaly_type == "logits_finite"] == []
    nan_alerts = mgr.check_before_sample(
        scheduler_output=so,
        logits=torch.tensor([[float("nan"), 1.0]]),
        logits_indices=None,
        input_batch=runner.input_batch,
    )
    assert len([a for a in nan_alerts if a.anomaly_type == "logits_finite"]) == 1


def test_v1_sample_tokens_checks_before_grammar_bitmask():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "vllm_ascend/worker/model_runner_v1.py").read_text()
    i_check = src.index("self.dfx.check_before_sample")
    i_grammar = src.index("apply_grammar_bitmask(scheduler_output, grammar_output")
    assert i_check < i_grammar


def test_v2_sample_wraps_compute_logits_for_dfx_hook():
    from pathlib import Path

    runner_src = (Path(__file__).resolve().parents[2] / "vllm_ascend/worker/v2/model_runner.py").read_text()
    hooks_src = (Path(__file__).resolve().parents[2] / "vllm_ascend/dfx/runner_hooks.py").read_text()
    assert "need_pre_sample_hook" in runner_src
    assert "wrap_compute_logits_for_pre_sample" in runner_src
    assert "super().sample(" in runner_src
    assert "def wrap_compute_logits_for_pre_sample" in hooks_src
    assert "def need_pre_sample_hook" in hooks_src
    assert "def check_before_sample_from_batch" in hooks_src
    assert "can_run_anomaly_detection" in hooks_src
    assert "model.compute_logits = wrapped" in hooks_src
    # Must not keep a full copied upstream sample() body in the runner.
    assert "get_num_sampled_and_rejected" not in runner_src
    # wrap body must call the batch helper (not only define it earlier).
    wrap_body = hooks_src.split("def wrap_compute_logits_for_pre_sample", 1)[1]
    assert "check_before_sample_from_batch(" in wrap_body


def test_dfx_need_pre_sample_hook_skips_when_detection_gated():
    """Gate: detector on but can_run=False → no wrap."""
    from vllm_ascend.dfx.runner_hooks import need_pre_sample_hook

    class _Cfg:
        def __init__(self, logits=True, position=False):
            self._logits = logits
            self._position = position

        def detector_get(self, section, key, default=False):
            if section == "logits_finite" and key == "enabled":
                return self._logits
            if section == "position_alignment" and key == "enabled":
                return self._position
            return default

    dfx = SimpleNamespace(
        dfx_config=_Cfg(),
        dumper=SimpleNamespace(can_run_anomaly_detection=lambda: False),
    )
    assert need_pre_sample_hook(dfx) is False
    dfx.dumper.can_run_anomaly_detection = lambda: True
    assert need_pre_sample_hook(dfx) is True
    dfx.dfx_config = _Cfg(logits=False, position=False)
    assert need_pre_sample_hook(dfx) is False


def test_dfx_wrap_compute_logits_calls_check_and_restores():
    """``wrap_compute_logits_for_pre_sample`` calls check and restores the method."""
    from vllm_ascend.dfx.runner_hooks import wrap_compute_logits_for_pre_sample

    calls: list = []
    batch = object()

    class _Model:
        def compute_logits(self, hs):
            calls.append("orig")
            return hs + 1

    class _Dfx:
        def check_before_sample(self, **kwargs):
            calls.append(("check", float(kwargs["logits"].item()), kwargs["input_batch"] is batch))

    class _Runner:
        def __init__(self):
            self.model = _Model()
            self.dfx = _Dfx()
            self._dfx_scheduler_output = None

    runner = _Runner()
    hs = torch.tensor([1.0])
    with wrap_compute_logits_for_pre_sample(runner, batch):
        out = runner.model.compute_logits(hs)
        assert "compute_logits" in runner.model.__dict__
    assert float(out.item()) == 2.0
    assert calls[0] == "orig"
    assert calls[1] == ("check", 2.0, True)
    assert "compute_logits" not in runner.model.__dict__
    calls.clear()
    _ = runner.model.compute_logits(hs)
    assert calls == ["orig"]


def test_note_kv_block_writes_v2_input_batch_skips_prefix(tmp_path):
    from vllm_ascend.dfx.detector.manager import DetectorManager
    from vllm_ascend.dfx.processor import DfxProcessor

    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=tmp_path / "r", ensure_file=True)
    cfg._data["detector"]["block_kv"]["enabled"] = True
    runner = SimpleNamespace(
        input_batch=None,
        requests=None,
        block_size=16,
        vllm_config=None,
        block_tables=SimpleNamespace(
            num_blocks=SimpleNamespace(np=np.array([[2]], dtype=np.int32)),
            block_tables=[SimpleNamespace(np=np.array([[20, 21, 0]], dtype=np.int32))],
        ),
    )
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = runner
    proc.dfx_config = cfg
    proc.detectors = DetectorManager(
        dfx_config=cfg,
        runner=runner,
        detection_gate=lambda **_k: True,
    )
    proc.dumper = SimpleNamespace(current_wave=lambda: 3)
    proc._handle_alert = lambda *a, **k: None
    input_batch = SimpleNamespace(
        req_ids=["r1"],
        req_id_to_index={"r1": 0},
        idx_mapping_np=np.array([0], dtype=np.int32),
        num_computed_tokens_np=np.array([16], dtype=np.int32),
    )
    so = SimpleNamespace(num_scheduled_tokens={"r1": 1})
    proc.note_kv_block_writes(so, input_batch=input_batch)
    tracker = KvBlockMetaTracker.get()
    assert tracker.last_writer_req_id(21) == "r1"
    assert tracker.last_write_wave(21) == 3
    assert tracker.last_writer_req_id(20) is None


def test_forward_accounting_chain_writes_report_file(tmp_path):
    """Semi-integration: detect → _handle_alert → report file (no NPU forward)."""
    import json
    from unittest.mock import MagicMock, patch

    from vllm_ascend.dfx.detector.manager import DetectorManager
    from vllm_ascend.dfx.processor import DfxProcessor
    from vllm_ascend.dfx.report import DfxReportWriter
    from vllm_ascend.dfx.request_state import RequestDfxStore

    RequestDfxStore.reset_for_tests()
    report_dir = tmp_path / "report"
    cfg = DfxRuntimeConfig(tmp_path / "dfx.json", report_dir=report_dir, ensure_file=True)
    cfg._data["detector"]["stop_after_alert"] = True
    cfg._data["detector"]["logits_finite"]["enabled"] = True
    cfg._data["detector"]["block_kv"]["enabled"] = True
    cfg._data["dump"]["enabled"] = False

    input_batch = SimpleNamespace(
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
        idx_mapping_np=np.array([0], dtype=np.int32),
        num_computed_tokens_cpu=np.array([0], dtype=np.int32),
        num_computed_tokens_np=np.array([0], dtype=np.int32),
        query_start_loc_np=np.array([0, 1], dtype=np.int64),
    )
    runner = SimpleNamespace(
        tp_rank=0,
        input_batch=input_batch,
        requests=None,
        block_size=16,
        vllm_config=None,
        query_start_loc=SimpleNamespace(np=np.array([0, 1], dtype=np.int64)),
        block_tables=SimpleNamespace(
            num_blocks=SimpleNamespace(np=np.array([[1]], dtype=np.int32)),
            block_tables=[SimpleNamespace(np=np.array([[7, 0]], dtype=np.int32))],
        ),
    )
    proc = DfxProcessor.__new__(DfxProcessor)
    proc.runner = runner
    proc.dfx_config = cfg
    proc.detectors = DetectorManager(
        dfx_config=cfg,
        runner=runner,
        detection_gate=lambda **_k: True,
    )
    proc.dumper = SimpleNamespace(
        current_wave=lambda: 2,
        dump_count_snapshot=lambda **_k: (0, 0),
        dump_rank_tag=lambda: "tp0",
        dump_arm_wave_for_report=lambda: None,
        dump_arm_wave_for_req=lambda _rid: None,
        handle_anomaly_alert=lambda *a, **k: False,
    )
    proc.report_writer = DfxReportWriter(report_dir)
    proc._get_report_tokenizer = lambda: None
    proc._scheduler_output_for_step = None
    proc._enrich_detail_with_block_meta = lambda detail, *_a, **_k: detail

    so = SimpleNamespace(num_scheduled_tokens={"r0": 1}, total_num_scheduled_tokens=1)
    logits = torch.tensor([[float("nan"), 1.0]])

    with patch("vllm_ascend.dfx.processor.RequestIoSnapshotManager") as mgr_cls:
        mgr = MagicMock()
        mgr_cls.get.return_value = mgr
        snap = MagicMock()
        mgr.snapshot.return_value = snap
        mgr.merge_into_detail.side_effect = lambda detail, _snap: detail
        proc.check_before_sample(
            scheduler_output=so,
            logits=logits,
            positions=torch.tensor([0], dtype=torch.int64),
            total_scheduled_tokens=1,
            logits_indices=None,
            input_batch=input_batch,
        )

    reports = list(report_dir.glob("anomaly_*.log"))
    assert len(reports) == 1
    record = json.loads(reports[0].read_text(encoding="utf-8"))
    assert record["anomaly_type"] == "logits_finite"
    assert record["req_id"] == "r0"
    assert record["dump_attempted"] is False
    assert record["dump_armed"] is False
    assert record["detail"]["ill_type"] == ILL_TYPE_NAN
    assert record["detail"]["finite_kind"] == "nan"
    assert "r0" in RequestDfxStore.get().stopped_req_ids()

    # Same-step stop: wave regression would alert, but req is already stopped.
    KvBlockMetaTracker.get().record_writes("other", [7], wave=9)
    with patch("vllm_ascend.dfx.processor.RequestIoSnapshotManager") as mgr_cls:
        mgr_cls.get.return_value = MagicMock()
        proc.note_kv_block_writes(so, input_batch=input_batch)
    assert len(list(report_dir.glob("anomaly_*.log"))) == 1
    RequestDfxStore.reset_for_tests()
