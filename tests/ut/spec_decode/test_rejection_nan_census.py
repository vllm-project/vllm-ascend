# UT for the [dfx/nan] DFX census helpers in rejection_sampler_utils.
#
# The helpers are pure torch (no Triton launch) so they run in the CPU UT
# environment; the full rejection_sample wiring plus kernel behavior under
# NaN is covered by the NPU nightly e2e (test_rejection_sample_v2.py).
#
# Verified contract:
#   * clean inputs  -> zero log records even at DEBUG (no log spam on
#     healthy steps)
#   * any NaN       -> exactly one record per input tensor carrying
#     per-row req/step coordinates and NaN counts
#   * INFO level    -> zero records AND zero tensor work (gate first line)
#   * num_sampled==1 marks a fully-rejected request (accepted drafts + 1)

import logging

import torch

from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
    _log_accept0_census,
    _log_nan_census,
)

MODULE = "vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils"

V = 64  # small vocab; counts are shape-independent
K = 5
NUM_LOGITS = 12  # 2 requests * (K + 1) rows


class _Capture(logging.Handler):
    """Self-contained capture handler: does not depend on caplog or on the
    vllm_ascend handler tree configured by configure_ascend_logging."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _make_logger(level=logging.DEBUG):
    lg = logging.getLogger(MODULE)
    cap = _Capture()
    old_level = lg.level
    lg.addHandler(cap)
    lg.setLevel(level)
    return lg, cap, old_level


def _teardown_logger(lg, cap, old_level):
    lg.removeHandler(cap)
    lg.setLevel(old_level)


def _clean_target(num_logits=NUM_LOGITS):
    return torch.zeros(num_logits, V)


def _with_nan(base, row, count=1):
    t = base.clone()
    t[row, :count] = float("nan")
    return t


def _messages(cap):
    return [r.getMessage() for r in cap.records]


def _target_records(cap):
    return [m for m in _messages(cap) if "target_logits" in m]


def _draft_records(cap):
    return [m for m in _messages(cap) if "draft_logits" in m]


def test_clean_inputs_silent():
    lg, cap, old = _make_logger()
    try:
        _log_nan_census(lg, _clean_target(), torch.zeros(4, K, V), K)
        assert cap.records == [], f"clean inputs must stay silent, got {_messages(cap)}"
    finally:
        _teardown_logger(lg, cap, old)


def test_partial_nan_single_element():
    lg, cap, old = _make_logger()
    try:
        # row 6 = request 1, step 0 (rows_per_req = K + 1 = 6)
        _log_nan_census(lg, _with_nan(_clean_target(), 6, 1), None, K)
        recs = _target_records(cap)
        assert len(recs) == 1, recs
        msg = recs[0]
        assert "rows_with_nan=1/12" in msg
        assert "total_nan=1" in msg
        assert "req=1 step=0 nan=1" in msg
        assert _draft_records(cap) == []
    finally:
        _teardown_logger(lg, cap, old)


def test_whole_row_nan():
    lg, cap, old = _make_logger()
    try:
        _log_nan_census(lg, _with_nan(_clean_target(), 3, V), None, K)
        recs = _target_records(cap)
        assert len(recs) == 1, recs
        assert "total_nan=64" in recs[0]  # the whole-row upper bound
        assert "req=0 step=3 nan=64" in recs[0]
    finally:
        _teardown_logger(lg, cap, old)


def test_multiple_rows_nan():
    lg, cap, old = _make_logger()
    try:
        t = _with_nan(_clean_target(), 2, 3)   # req 0, step 2
        t = _with_nan(t, 7, 5)                 # req 1, step 1
        _log_nan_census(lg, t, None, K)
        recs = _target_records(cap)
        assert len(recs) == 1, recs
        assert "rows_with_nan=2/12" in recs[0]
        assert "total_nan=8" in recs[0]
        assert "req=0 step=2 nan=3" in recs[0]
        assert "req=1 step=1 nan=5" in recs[0]
    finally:
        _teardown_logger(lg, cap, old)


def test_draft_logits_nan():
    lg, cap, old = _make_logger()
    try:
        draft = torch.zeros(4, K, V)
        draft[1, 3, :7] = float("nan")         # slot 1, step 3
        _log_nan_census(lg, _clean_target(), draft, K)
        recs = _draft_records(cap)
        assert len(recs) == 1, recs
        assert "rows_with_nan=1" in recs[0]
        assert "total_nan=7" in recs[0]
        assert "slot=1 step=3 nan=7" in recs[0]
    finally:
        _teardown_logger(lg, cap, old)


def test_draft_logits_none_no_crash():
    lg, cap, old = _make_logger()
    try:
        _log_nan_census(lg, _clean_target(), None, K)
        assert cap.records == []
        # and with a poisoned target while draft is None:
        _log_nan_census(lg, _with_nan(_clean_target(), 0, 2), None, K)
        assert len(_target_records(cap)) == 1
        assert _draft_records(cap) == []
    finally:
        _teardown_logger(lg, cap, old)


def test_info_level_silent():
    lg, cap, old = _make_logger(level=logging.INFO)
    try:
        t = _with_nan(_clean_target(), 6, V)   # poisoned, but gate closed
        _log_nan_census(lg, t, torch.zeros(4, K, V), K)
        assert cap.records == [], f"INFO must stay silent, got {_messages(cap)}"
    finally:
        _teardown_logger(lg, cap, old)


def test_accept0_census():
    lg, cap, old = _make_logger()
    try:
        num_sampled = torch.tensor([1, 6, 3, 1], dtype=torch.int32)
        _log_accept0_census(lg, num_sampled, 4)
        recs = _messages(cap)
        assert len(recs) == 1, recs
        assert "accept0_reqs=2/4" in recs[0]
        assert "req_ids_first8=[0, 3]" in recs[0]
    finally:
        _teardown_logger(lg, cap, old)


def test_accept0_normal_silent():
    lg, cap, old = _make_logger()
    try:
        _log_accept0_census(lg, torch.tensor([6, 6], dtype=torch.int32), 2)
        assert cap.records == []
    finally:
        _teardown_logger(lg, cap, old)


def test_census_never_raises():
    """Malformed input must not take down the sampler (never-raise guard)."""
    lg, cap, old = _make_logger()
    try:
        _log_nan_census(lg, torch.zeros(0, V), None, K)          # zero rows
        _log_nan_census(lg, torch.zeros(3, 0), None, K)          # zero vocab
        _log_accept0_census(lg, torch.zeros(0, dtype=torch.int32), 0)
        # no exception escaped; failures would surface as census-failed DEBUGs
    finally:
        _teardown_logger(lg, cap, old)
