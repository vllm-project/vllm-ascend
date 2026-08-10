#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
from unittest.mock import MagicMock, patch

from tests.ut.dfx_test_utils import make_dfx_config
from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.detector.manager import DetectorManager
from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager


def test_detector_manager_check_after_sample_aggregates_alerts(tmp_path):
    RequestIoSnapshotManager.reset_for_tests()
    cfg = make_dfx_config(tmp_path)
    runner = SimpleNamespace(tp_rank=0, input_batch=SimpleNamespace(req_ids=["r1"]), requests={})
    mgr = DetectorManager(dfx_config=cfg, runner=runner)

    token_alert = AnomalyAlert(anomaly_type="token_logprob", req_id="r1")
    out_alert = AnomalyAlert(anomaly_type="output_substring", req_id="r1")
    token_det = mgr.get("token_logprob")
    out_det = mgr.get("output_substring")
    assert token_det is not None and out_det is not None

    token_det.check_all = MagicMock(return_value=[token_alert])  # type: ignore[method-assign]
    out_det.check_all = MagicMock(return_value=[out_alert])  # type: ignore[method-assign]

    alerts = mgr.check_after_sample(
        sampled_token_ids=[[1, 2]],
        logprobs_lists=None,
        req_ids=["r1"],
    )
    assert alerts == [token_alert, out_alert]
    token_det.check_all.assert_called_once()
    out_det.check_all.assert_called_once()
    # Substring path must not re-append (sampled_token_ids=None).
    assert out_det.check_all.call_args.kwargs.get("sampled_token_ids") is None


def test_detector_manager_token_logprob_topk_if_enabled(tmp_path):
    cfg = make_dfx_config(tmp_path)
    runner = SimpleNamespace(tp_rank=0)
    mgr = DetectorManager(dfx_config=cfg, runner=runner)
    assert mgr.token_logprob_topk_if_enabled() is None

    cfg._data["detector"]["token_logprob"]["enabled"] = True
    cfg._data["detector"]["token_logprob"]["topk"] = 17
    with patch.object(mgr._token_det, "_get_ill_detector", return_value=object()):
        assert mgr.token_logprob_topk_if_enabled() == 17


class _FakeTok:
    """Deterministic char↔ord tokenizer for tests."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(c) for c in text]

    def decode(self, token_ids, skip_special_tokens: bool = False):
        return "".join(chr(int(t)) for t in token_ids)


def test_check_after_sample_feeds_substring_buffer(tmp_path):
    """check_after_sample appends tokens once; substring matches cumulative IO."""
    RequestIoSnapshotManager.reset_for_tests()
    cfg = make_dfx_config(tmp_path)
    cfg._data["detector"]["output_substring"]["enabled"] = True
    cfg._data["detector"]["output_substring"]["patterns"] = [[ord("x"), ord("y")]]

    runner = SimpleNamespace(
        tp_rank=0,
        input_batch=SimpleNamespace(req_ids=["r1"]),
        requests={"r1": SimpleNamespace()},
        vllm_config=SimpleNamespace(model_config=object()),
    )
    mgr = DetectorManager(dfx_config=cfg, runner=runner, tokenizer_provider=lambda: _FakeTok())

    # Step 1: append [x] — cumulative [x], no match yet.
    alerts1 = mgr.check_after_sample(sampled_token_ids=[[ord("x")]], logprobs_lists=None, req_ids=["r1"])
    assert alerts1 == []

    # Processor clears the per-wave IO cache each step; mimic that here.
    RequestIoSnapshotManager.get().clear_wave_cache()

    # Step 2: append [y] — cumulative [x, y] now matches [x, y].
    alerts2 = mgr.check_after_sample(sampled_token_ids=[[ord("y")]], logprobs_lists=None, req_ids=["r1"])
    assert len(alerts2) == 1
    assert alerts2[0].anomaly_type == "output_substring"
    assert alerts2[0].detail["matched_text"] == "xy"


def test_detector_manager_gates_detection(tmp_path):
    """detection_gate returning False suppresses all detection hooks."""
    RequestIoSnapshotManager.reset_for_tests()
    cfg = make_dfx_config(tmp_path)
    cfg._data["detector"]["token_logprob"]["enabled"] = True
    runner = SimpleNamespace(tp_rank=0, input_batch=SimpleNamespace(req_ids=["r1"]), requests={})
    mgr = DetectorManager(dfx_config=cfg, runner=runner, detection_gate=lambda: False)

    token_det = mgr.get("token_logprob")
    token_det.check_all = MagicMock(return_value=[AnomalyAlert(anomaly_type="token_logprob", req_id="r1")])

    # Gate off: hooks return [] and never touch detectors / IO buffer.
    assert mgr.check_after_spec(None, None) == []
    assert mgr.check_after_sample(sampled_token_ids=[[1]], logprobs_lists=None, req_ids=["r1"]) == []
    token_det.check_all.assert_not_called()


def test_check_after_sample_stop_after_alert_halts_on_alert(tmp_path):
    """Once a request alerts, later steps skip it (no endless reports); clear_finished resets."""
    RequestIoSnapshotManager.reset_for_tests()
    cfg = make_dfx_config(tmp_path)
    runner = SimpleNamespace(tp_rank=0, input_batch=SimpleNamespace(req_ids=["r1"]), requests={})
    mgr = DetectorManager(dfx_config=cfg, runner=runner)

    token_det = mgr.get("token_logprob")
    assert token_det is not None
    token_det.check_all = MagicMock(return_value=[AnomalyAlert(anomaly_type="token_logprob", req_id="r1")])

    # Step 1: r1 alerts → marked stopped.
    alerts = mgr.check_after_sample(sampled_token_ids=[[1]], logprobs_lists=None, req_ids=["r1"])
    assert [a.req_id for a in alerts] == ["r1"]
    assert token_det.check_all.call_count == 1
    # IO still accumulates on the alert step.
    assert RequestIoSnapshotManager.get().cumulative_output_count("r1") == 1

    # Step 2: r1 already alerted → no detector call, but IO still accumulates.
    assert mgr.check_after_sample(sampled_token_ids=[[2]], logprobs_lists=None, req_ids=["r1"]) == []
    assert token_det.check_all.call_count == 1
    assert RequestIoSnapshotManager.get().cumulative_output_count("r1") == 2

    # clear_finished resets so a request may be detected again.
    mgr.clear_finished("r1")
    alerts_again = mgr.check_after_sample(sampled_token_ids=[[3]], logprobs_lists=None, req_ids=["r1"])
    assert token_det.check_all.call_count == 2
    assert [a.req_id for a in alerts_again] == ["r1"]


def test_check_after_sample_stop_after_alert_keeps_detecting_until_alert(tmp_path):
    """No alert → the same request keeps being detected on every step."""
    RequestIoSnapshotManager.reset_for_tests()
    cfg = make_dfx_config(tmp_path)
    runner = SimpleNamespace(tp_rank=0, input_batch=SimpleNamespace(req_ids=["r1"]), requests={})
    mgr = DetectorManager(dfx_config=cfg, runner=runner)

    token_det = mgr.get("token_logprob")
    assert token_det is not None
    token_det.check_all = MagicMock(return_value=[])  # type: ignore[method-assign]

    assert mgr.check_after_sample(sampled_token_ids=[[1]], logprobs_lists=None, req_ids=["r1"]) == []
    assert mgr.check_after_sample(sampled_token_ids=[[2]], logprobs_lists=None, req_ids=["r1"]) == []
    assert token_det.check_all.call_count == 2


def test_check_after_sample_stop_after_alert_skips_by_req_id(tmp_path):
    """stop_after_alert skips by req_id; keeps full batch + original req_idx alignment."""
    RequestIoSnapshotManager.reset_for_tests()
    cfg = make_dfx_config(tmp_path)
    runner = SimpleNamespace(tp_rank=0, input_batch=SimpleNamespace(req_ids=["r1", "r2"]), requests={})
    mgr = DetectorManager(dfx_config=cfg, runner=runner)

    token_det = mgr.get("token_logprob")
    assert token_det is not None

    # Step 1: r1 alerts, r2 does not.
    def _alert_r1(sampled_token_ids=None, logprobs_lists=None, req_ids=None, skip_req_ids=None):
        return [AnomalyAlert(anomaly_type="token_logprob", req_id="r1", req_idx=0)]

    token_det.check_all = MagicMock(side_effect=_alert_r1)  # type: ignore[method-assign]
    mgr.check_after_sample(sampled_token_ids=[[1], [2]], logprobs_lists=None, req_ids=["r1", "r2"])
    assert token_det.check_all.call_args.kwargs["req_ids"] == ["r1", "r2"]

    # Step 2: full batch still passed; r1 is in skip_req_ids (do not remmap req_idx).
    mgr.check_after_sample(
        sampled_token_ids=[[3], [4], [5]],
        logprobs_lists=None,
        req_ids=["r1", "r2", "r3"],
    )
    kwargs = token_det.check_all.call_args.kwargs
    assert kwargs["req_ids"] == ["r1", "r2", "r3"]
    assert kwargs["sampled_token_ids"] == [[3], [4], [5]]
    assert "r1" in kwargs["skip_req_ids"]
    assert RequestIoSnapshotManager.get().cumulative_output_count("r1") == 2  # [1] then [3]
    assert RequestIoSnapshotManager.get().cumulative_output_count("r2") == 2  # [2] then [4]
    assert RequestIoSnapshotManager.get().cumulative_output_count("r3") == 1  # [5]


def test_detector_manager_apply_dfx_config_refreshes_token_logprob(tmp_path):
    cfg = make_dfx_config(tmp_path)
    cfg._data["detector"]["token_logprob"]["enabled"] = True
    mgr = DetectorManager(dfx_config=cfg, runner=SimpleNamespace(tp_rank=0))
    with patch.object(mgr._token_det, "refresh_from_config") as refresh:
        mgr.apply_dfx_config()
    refresh.assert_called_once()
