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

from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.dfx.detector.spec_acceptance import SpecAcceptanceDetector
from vllm_ascend.dfx.input_filters import InputFilterManager
from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager


def _make_detector(
    *,
    window: int = 1,
    low_threshold: float = 0.3,
    len_low_threshold: float = 1.4,
    high_threshold: float = 0.96,
    len_high_threshold: float = 2.8,
    req_ids: list[str] | None = None,
) -> SpecAcceptanceDetector:
    req_ids = req_ids or ["r1"]
    det = SpecAcceptanceDetector.__new__(SpecAcceptanceDetector)
    det._runner = SimpleNamespace(
        tp_rank=0,
        speculative_config=object(),
        input_batch=SimpleNamespace(
            req_ids=req_ids,
            req_output_token_ids=[[] for _ in req_ids],
            num_draft_tokens_per_req=None,
        ),
        requests={
            rid: SimpleNamespace(prev_num_draft_len=0, prompt_token_ids=[], output_token_ids=[]) for rid in req_ids
        },
    )
    det._dfx_config = None
    det._is_related_request = lambda rid, idx: True
    det._enabled = True
    det._history = defaultdict(deque)
    det._window = window
    det._low_threshold = low_threshold
    det._len_low_threshold = len_low_threshold
    det._high_threshold = high_threshold
    det._len_high_threshold = len_high_threshold
    det._short_log_ts = {}
    det._short_log_interval_s = 2.0
    return det


def _check_one(
    det: SpecAcceptanceDetector,
    *,
    req_id: str = "r1",
    req_idx: int = 0,
    draft_len: int,
    accepted: int,
    sampled: list[int],
):
    InputFilterManager.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    with patch("vllm_ascend.dfx.detector.spec_acceptance.get_pp_group") as get_pp:
        get_pp.return_value.is_last_rank = True
        return det.check_one(
            req_idx=req_idx,
            req_id=req_id,
            req_state=SimpleNamespace(
                prev_num_draft_len=draft_len,
                prompt_token_ids=[],
                output_token_ids=[],
            ),
            accepted_token_num=accepted,
            sampled_ids=sampled,
        )


def test_spec_no_alert_when_thresholds_not_met():
    """Normal accept rate inside band → no alert (moved from test_dumper)."""
    det = _make_detector(
        window=1,
        low_threshold=0.1,
        len_low_threshold=0.1,
        high_threshold=2.0,
        len_high_threshold=2.0,
    )
    # accepted_draft=1, draft=1 → rate=1.0, len=1.0 (inside band).
    assert _check_one(det, draft_len=1, accepted=2, sampled=[10, 11]) is None


def test_spec_alerts_on_low_acceptance():
    det = _make_detector(window=1)
    # accepted_draft=0, draft=10 → rate=0, len=0 → below low thresholds.
    alert = _check_one(det, draft_len=10, accepted=1, sampled=list(range(11)))
    assert alert is not None
    assert alert.anomaly_type == "spec_acceptance"
    assert alert.req_id == "r1"
    assert alert.detail["acceptance_rate"] == 0.0
    assert alert.detail["window"] == 1


def test_spec_alerts_on_high_acceptance():
    det = _make_detector(window=1)
    # accepted_draft=5, draft=5 → rate=1.0, len=5.0 → above high thresholds.
    alert = _check_one(det, draft_len=5, accepted=6, sampled=list(range(6)))
    assert alert is not None
    assert alert.detail["acceptance_rate"] == 1.0
    assert alert.detail["acceptance_len"] == 5.0


def test_spec_waits_for_window_before_alert():
    det = _make_detector(window=2)
    # First step would be low, but window not full → no alert.
    assert _check_one(det, draft_len=10, accepted=1, sampled=list(range(11))) is None
    assert len(det._history["r1"]) == 1
    # Second low step fills window → alert.
    alert = _check_one(det, draft_len=10, accepted=1, sampled=list(range(11)))
    assert alert is not None
    assert alert.detail["window"] == 2


def test_spec_check_all_honors_skip_req_ids():
    det = _make_detector(req_ids=["r1", "r2"], window=1)
    InputFilterManager.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    # Two rows: both would low-alert (accepted_draft=0, draft from sampled len-1).
    sampled = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    accepted = [1, 1]
    with patch("vllm_ascend.dfx.detector.spec_acceptance.get_pp_group") as get_pp:
        get_pp.return_value.is_last_rank = True
        # Force draft_len via requests.prev_num_draft_len.
        for rid in ("r1", "r2"):
            det._runner.requests[rid].prev_num_draft_len = 10
        alerts = det.check_all(sampled, accepted, skip_req_ids={"r1"})
    assert [a.req_id for a in alerts] == ["r2"]
    # Skipped rows never call check_one → no history for r1.
    assert "r1" not in det._history
    assert len(det._history["r2"]) == 1


def test_spec_clear_finished_drops_history():
    det = _make_detector(window=2)
    _check_one(det, draft_len=10, accepted=1, sampled=list(range(11)))
    assert "r1" in det._history
    det.clear_finished("r1")
    assert "r1" not in det._history
