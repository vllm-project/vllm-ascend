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

from types import SimpleNamespace

from vllm_ascend.dfx.detector.token_repeat import (
    TokenRepeatDetector,
    TokenRepeatState,
    push_token_repeat,
)
from vllm_ascend.dfx.dfx_types import ILL_TYPE_REPEAT
from vllm_ascend.dfx.input_filters import InputFilterManager
from vllm_ascend.dfx.io_snapshot import RequestIoSnapshotManager


def test_push_token_repeat_scores_and_sum():
    state = TokenRepeatState()
    # First unique token → score 0.
    assert push_token_repeat(state, 7, window=4, ignore=set()) == 0
    assert state.repeat_sum == 0
    # Same token again → score 1.
    assert push_token_repeat(state, 7, window=4, ignore=set()) == 1
    assert state.repeat_sum == 1
    assert list(state.content) == [7, 7]
    # Fill with more 7s.
    assert push_token_repeat(state, 7, window=4, ignore=set()) == 2
    assert push_token_repeat(state, 7, window=4, ignore=set()) == 3
    assert state.repeat_sum == 0 + 1 + 2 + 3
    # Window full of 7s → next 7 scores 4 (count in prior n), then slide.
    assert push_token_repeat(state, 7, window=4, ignore=set()) == 4
    assert len(state.content) == 4
    assert state.repeat_sum == 1 + 2 + 3 + 4


def test_push_ignores_token_ids_from_content_window():
    state = TokenRepeatState()
    ignore = {0}
    assert push_token_repeat(state, 5, window=4, ignore=ignore) == 0
    assert push_token_repeat(state, 0, window=4, ignore=ignore) == 0  # ignored
    assert list(state.content) == [5]
    assert push_token_repeat(state, 5, window=4, ignore=ignore) == 1
    assert list(state.content) == [5, 5]
    assert state.content_tokens_seen == 2


def _make_det(**kwargs) -> TokenRepeatDetector:
    det = TokenRepeatDetector.__new__(TokenRepeatDetector)
    det._dfx_config = None
    det._runner = SimpleNamespace(tp_rank=0)
    det._enabled = True
    det._window = kwargs.get("window", 4)
    det._repeat_sum_threshold = kwargs.get("repeat_sum_threshold", 6)
    det._min_tokens = kwargs.get("min_tokens", 4)
    det._consecutive_hits_thresh = kwargs.get("consecutive_hits", 1)
    det._ignore_token_ids = frozenset(kwargs.get("ignore_token_ids", ()))
    det._states = {}
    det._consumed_len = {}
    det._alerted = set()
    return det


def test_check_one_alerts_on_dense_repeat():
    InputFilterManager.reset_for_tests()
    det = _make_det(window=4, repeat_sum_threshold=6, min_tokens=4)
    # Unique stream: scores all 0 → no alert.
    assert det.check_one(0, "r1", [1, 2, 3, 4]) is None
    # Dense same-token stream over threshold.
    alert = det.check_one(0, "r2", [9, 9, 9, 9, 9, 9])
    assert alert is not None
    assert alert.anomaly_type == "token_repeat"
    assert alert.ill_type == ILL_TYPE_REPEAT
    assert alert.detail["repeat_sum"] > 6


def test_check_all_respects_skip_and_alert_once():
    InputFilterManager.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    det = _make_det(window=4, repeat_sum_threshold=6, min_tokens=4)
    alerts = det.check_all(
        sampled_token_ids=[[9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9]],
        req_ids=["a", "b"],
        skip_req_ids={"b"},
    )
    assert len(alerts) == 1
    assert alerts[0].req_id == "a"
    # Second call: already alerted → empty.
    assert (
        det.check_all(
            sampled_token_ids=[[9, 9, 9, 9, 9, 9]],
            req_ids=["a"],
        )
        == []
    )


def test_check_all_none_reads_cumulative_io_including_prior_append():
    """Manager-style path: IO already appended; pass None (same as substring)."""
    InputFilterManager.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()
    io = RequestIoSnapshotManager.get()
    # Spec step wrote accepted tokens into the shared cumulative stream.
    # Chunks must differ: append_output dedupes identical consecutive chunks.
    io.append_batch(["r1"], [[9, 9, 9]])
    det = _make_det(window=4, repeat_sum_threshold=6, min_tokens=4)
    # Sample step already appended by DetectorManager; detector gets None.
    io.append_batch(["r1"], [[9, 9, 9, 9]])
    assert io.cumulative_output_ids("r1") == [9, 9, 9, 9, 9, 9, 9]
    alerts = det.check_all(sampled_token_ids=None, req_ids=["r1"])
    assert len(alerts) == 1
    assert alerts[0].req_id == "r1"
    # Cursor advanced: another None call with no new ids → no re-alert / no work.
    assert det.check_all(sampled_token_ids=None, req_ids=["r1"]) == []
    assert det._consumed_len["r1"] == 7
