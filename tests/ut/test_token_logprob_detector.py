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

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.detector.registry import DetectorRegistry
from vllm_ascend.dfx.detector.token_logprob import TokenLogprobDetector
from vllm_ascend.dfx.dfx_types import ILL_TYPE_RARE
from vllm_ascend.dfx.input_filters import InputFilterManager


def _make_token_detector(*, window=4, stride=2, topk=2, thresh=1) -> TokenLogprobDetector:
    det = TokenLogprobDetector.__new__(TokenLogprobDetector)
    det._dfx_config = None
    det._runner = SimpleNamespace(tp_rank=0)
    det._enabled = True
    det._window = window
    det._stride = stride
    det._topk = topk
    det._ill_window_thresh = {1: thresh, 2: 1, 3: 2, 4: 1}
    det._buf = {}
    det._since_check = defaultdict(int)
    det._checked = set()
    det._ill_window_hits = defaultdict(lambda: defaultdict(int))
    det._ill_detector = MagicMock()
    det._ill_detector_init_failed = False
    return det


def test_row_to_topk_dict_keeps_highest_logprobs():
    row = TokenLogprobDetector._row_to_topk_dict(
        [10, 11, 12, -1],
        [-0.5, -0.1, -2.0, -0.01],
        topk=2,
    )
    assert list(row.keys()) == [11, 10]
    assert row[11] == -0.1


def test_check_one_fills_then_alerts_on_ill_hit():
    det = _make_token_detector(window=2, stride=2, thresh=1)
    ill = SimpleNamespace(is_ill=True, ill_type=ILL_TYPE_RARE)
    det._ill_detector.detector.return_value = ill

    assert (
        det.check_one(
            req_idx=0,
            req_id="r1",
            token_ids=[1],
            topk_logprobs=[{1: -0.1}],
            model_config={},
            detector=det._ill_detector,
            log_leader=True,
        )
        is None
    )
    alert = det.check_one(
        req_idx=0,
        req_id="r1",
        token_ids=[2],
        topk_logprobs=[{2: -0.2}],
        model_config={},
        detector=det._ill_detector,
        log_leader=True,
    )
    assert isinstance(alert, AnomalyAlert)
    assert alert.anomaly_type == "token_logprob"
    assert alert.ill_type == ILL_TYPE_RARE
    assert alert.detail["hits"] == 1
    assert alert.detail["window_token_ids"] == [1, 2]


def test_check_one_respects_stride_and_hit_thresh():
    det = _make_token_detector(window=2, stride=2, thresh=2)
    det._ill_detector.detector.return_value = SimpleNamespace(is_ill=True, ill_type=ILL_TYPE_RARE)

    # Fill window → first hit (below thresh=2)
    assert det.check_one(0, "r1", [1, 2], [{1: 0.0}, {2: 0.0}], {}, det._ill_detector) is None
    # Need stride more tokens before re-check
    assert det.check_one(0, "r1", [3], [{3: 0.0}], {}, det._ill_detector) is None
    alert = det.check_one(0, "r1", [4], [{4: 0.0}], {}, det._ill_detector)
    assert alert is not None
    assert alert.detail["hits"] == 2


def test_check_one_handles_detector_exception():
    det = _make_token_detector(window=1, stride=1, thresh=1)
    det._ill_detector.detector.side_effect = RuntimeError("boom")
    assert det.check_one(0, "r1", [1], [{1: 0.0}], {}, det._ill_detector) is None


class _FakeLogprobsLists:
    """Mimics the logprobs structure ``check_all`` reads per batch row."""

    logprob_token_ids = [
        [101, 102, 103],
        [104, 105, 106],
        [201, 202, 203],
        [204, 205, 206],
    ]
    logprobs = [
        [-0.1, -0.2, -0.3],
        [-0.4, -0.5, -0.6],
        [-0.7, -0.8, -0.9],
        [-1.0, -1.1, -1.2],
    ]
    cu_num_generated_tokens = None


def test_check_all_batch_entry_returns_alerts():
    InputFilterManager.reset_for_tests()
    det = _make_token_detector(window=2, stride=2, thresh=1)
    det._ill_detector.detector.return_value = SimpleNamespace(is_ill=True, ill_type=ILL_TYPE_RARE)

    alerts = det.check_all(
        sampled_token_ids=[[1, 2], [3, 4]],
        logprobs_lists=_FakeLogprobsLists(),
        req_ids=["r1", "r2"],
    )
    assert [a.req_id for a in alerts] == ["r1", "r2"]

    # skip_req_ids keeps batch-index alignment for the remaining requests.
    det = _make_token_detector(window=2, stride=2, thresh=1)
    det._ill_detector.detector.return_value = SimpleNamespace(is_ill=True, ill_type=ILL_TYPE_RARE)
    alerts = det.check_all(
        sampled_token_ids=[[1, 2], [3, 4]],
        logprobs_lists=_FakeLogprobsLists(),
        req_ids=["r1", "r2"],
        skip_req_ids={"r1"},
    )
    assert [a.req_id for a in alerts] == ["r2"]


def test_detector_registry_refresh_and_clear():
    registry = DetectorRegistry()
    a = MagicMock()
    a.anomaly_type = "spec_acceptance"
    b = MagicMock()
    b.anomaly_type = "token_logprob"
    registry.register(a)
    registry.register(b)
    assert len(registry) == 2
    assert registry.get("spec_acceptance") is a
    registry.refresh_all()
    a.refresh_from_config.assert_called_once()
    b.refresh_from_config.assert_called_once()
    registry.clear_finished("req-x")
    a.clear_finished.assert_called_once_with("req-x")
    b.clear_finished.assert_called_once_with("req-x")


def test_refresh_forces_token_logprob_off_when_msprobe_missing(tmp_path):
    from unittest.mock import patch

    from tests.ut.dfx_test_utils import make_dfx_config

    cfg = make_dfx_config(tmp_path)
    cfg._data["detector"]["token_logprob"]["enabled"] = True
    with patch.object(TokenLogprobDetector, "_get_ill_detector", return_value=None):
        det = TokenLogprobDetector(dfx_config=cfg, runner=SimpleNamespace(tp_rank=0))
    assert det.enabled is False
    assert cfg.detector_get("token_logprob", "enabled") is False


def test_refresh_retries_ill_detector_after_prior_failure(tmp_path):
    from unittest.mock import MagicMock, patch

    from tests.ut.dfx_test_utils import make_dfx_config

    cfg = make_dfx_config(tmp_path)
    cfg._data["detector"]["token_logprob"]["enabled"] = True
    det = TokenLogprobDetector.__new__(TokenLogprobDetector)
    det._dfx_config = cfg
    det._runner = SimpleNamespace(tp_rank=0)
    det._enabled = False
    det._window = 64
    det._stride = 32
    det._topk = 20
    det._ill_window_thresh = {1: 1, 2: 1, 3: 2, 4: 1}
    det._buf = {}
    det._since_check = defaultdict(int)
    det._checked = set()
    det._ill_window_hits = defaultdict(lambda: defaultdict(int))
    det._ill_detector = None
    det._ill_detector_init_failed = True

    fake = MagicMock()
    with patch.object(TokenLogprobDetector, "_get_ill_detector", return_value=fake):
        det.refresh_from_config()
    assert det.enabled is True
    assert det._ill_detector_init_failed is False
