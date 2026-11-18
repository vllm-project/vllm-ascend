# UT for the [dfx/nan] DFX probes in observability/spec_decode_dfx.
#
# The probes are pure torch (no Triton launch) so they run in the CPU UT
# environment; the full rejection_sample wiring plus kernel behavior under
# NaN is covered by the NPU nightly e2e (test_rejection_sample_v2.py).
#
# Timing contract: the probes are called AFTER the accept0 verdict of the
# same step -- the verdict path runs with zero diagnostic kernels in
# flight, and every host sync lands after the verdict (observer-effect
# free for timing-sensitive NaN failures).
#
# Verified contract:
#   * clean inputs  -> zero log records even at DEBUG (no log spam on
#     healthy steps)
#   * any NaN       -> exactly one record per input tensor carrying
#     per-row req/step coordinates and NaN counts
#   * INFO level    -> zero records AND zero tensor work (gate first line)
#   * num_sampled==1 marks a fully-rejected request (accepted drafts + 1),
#     restricted to requests that actually had drafts under packed
#     cu_num_logits

import logging

import pytest
import torch

from vllm_ascend.observability import spec_decode_dfx as dfx

V = 64  # small vocab; counts are shape-independent
K = 5
NUM_LOGITS = 12  # 2 requests * (K + 1) rows


class _Capture:
    """Stand-in logger: records debug() calls without going through any
    handler tree, so the tests are independent of configure_ascend_logging."""

    def __init__(self, level: int):
        self.level = level
        self.records: list[str] = []

    def isEnabledFor(self, lvl: int) -> bool:
        return lvl >= self.level

    def debug(self, msg, *args):
        try:
            self.records.append(msg % args if args else msg)
        except Exception:
            self.records.append(msg)


@pytest.fixture
def capture(monkeypatch):
    cap = _Capture(logging.DEBUG)
    monkeypatch.setattr(dfx, "_logger", cap)
    return cap


@pytest.fixture
def capture_info(monkeypatch):
    cap = _Capture(logging.INFO)
    monkeypatch.setattr(dfx, "_logger", cap)
    return cap


def _clean_target(num_logits=NUM_LOGITS):
    return torch.zeros(num_logits, V)


def _with_nan(base, row, count=1):
    t = base.clone()
    t[row, :count] = float("nan")
    return t


def _messages(cap):
    return [r if isinstance(r, str) else r.getMessage() for r in cap.records]


def _target_records(cap):
    return [m for m in _messages(cap) if "target_logits" in m]


def _draft_records(cap):
    return [m for m in _messages(cap) if "draft_logits" in m]


def test_clean_inputs_silent(capture):
    dfx.log_nan_census(_clean_target(), torch.zeros(4, K, V), K)
    assert capture.records == [], f"clean inputs must stay silent, got {_messages(capture)}"


def test_partial_nan_single_element(capture):
    # row 6 = request 1, step 0 (rows_per_req = K + 1 = 6)
    dfx.log_nan_census(_with_nan(_clean_target(), 6, 1), None, K)
    recs = _target_records(capture)
    assert len(recs) == 1, recs
    msg = recs[0]
    assert "rows_with_nan=1/12" in msg
    assert "total_nan=1" in msg
    assert "req=1 step=0 nan=1" in msg
    assert _draft_records(capture) == []


def test_whole_row_nan(capture):
    dfx.log_nan_census(_with_nan(_clean_target(), 3, V), None, K)
    recs = _target_records(capture)
    assert len(recs) == 1, recs
    assert "total_nan=64" in recs[0]  # the whole-row upper bound
    assert "req=0 step=3 nan=64" in recs[0]


def test_multiple_rows_nan(capture):
    t = _with_nan(_clean_target(), 2, 3)  # req 0, step 2
    t = _with_nan(t, 7, 5)  # req 1, step 1
    dfx.log_nan_census(t, None, K)
    recs = _target_records(capture)
    assert len(recs) == 1, recs
    assert "rows_with_nan=2/12" in recs[0]
    assert "total_nan=8" in recs[0]
    assert "req=0 step=2 nan=3" in recs[0]
    assert "req=1 step=1 nan=5" in recs[0]


def test_draft_logits_nan(capture):
    draft = torch.zeros(4, K, V)
    draft[1, 3, :7] = float("nan")  # slot 1, step 3
    dfx.log_nan_census(_clean_target(), draft, K)
    recs = _draft_records(capture)
    assert len(recs) == 1, recs
    assert "rows_with_nan=1" in recs[0]
    assert "total_nan=7" in recs[0]
    assert "slot=1 step=3 nan=7" in recs[0]


def test_draft_logits_none_no_crash(capture):
    dfx.log_nan_census(_clean_target(), None, K)
    assert capture.records == []
    # and with a poisoned target while draft is None:
    dfx.log_nan_census(_with_nan(_clean_target(), 0, 2), None, K)
    assert len(_target_records(capture)) == 1
    assert _draft_records(capture) == []


def test_packed_rows_mixed_boundaries(capture):
    """Regression: target_logits is packed with cu_num_logits, so requests
    do NOT always occupy num_speculative_steps + 1 rows. With K=3 and
    boundaries [0, 1, 5] (a request without drafts followed by one with
    three drafts), a NaN at row 1 belongs to request 1 step 0 -- the
    uniform division (i // (K+1)) would misreport request 0 step 1."""
    boundaries = torch.tensor([0, 1, 5], dtype=torch.int64)
    t = torch.zeros(5, V)
    t[0, 0] = float("nan")  # request 0's only row
    t[1, 0] = float("nan")  # request 1, step 0
    t[4, :2] = float("nan")  # request 1, last row (bonus)
    dfx.log_nan_census(t, None, 3, boundaries)
    recs = _target_records(capture)
    assert len(recs) == 1, recs
    assert "rows_with_nan=3/5" in recs[0]
    assert "req=0 step=0 nan=1" in recs[0]
    assert "req=1 step=0 nan=1" in recs[0]
    assert "req=1 step=3 nan=2" in recs[0]


def test_packed_rows_uniform_boundaries_match_legacy(capture):
    """With uniform boundaries (every request K+1 rows) the packed path
    must agree with the legacy uniform division."""
    boundaries = torch.tensor([0, 6, 12], dtype=torch.int64)
    t = torch.zeros(12, V)
    t[6, 0] = float("nan")  # request 1, step 0
    dfx.log_nan_census(t, None, K, boundaries)
    recs = _target_records(capture)
    assert len(recs) == 1, recs
    assert "req=1 step=0 nan=1" in recs[0]


def test_info_level_silent(capture_info):
    t = _with_nan(_clean_target(), 6, V)  # poisoned, but gate closed
    dfx.log_nan_census(t, torch.zeros(4, K, V), K)
    dfx.log_accept0_census(torch.tensor([6], dtype=torch.int32), 1)
    assert capture_info.records == [], f"INFO must stay silent, got {_messages(capture_info)}"


def test_accept0_census(capture):
    num_sampled = torch.tensor([1, 6, 3, 1], dtype=torch.int32)
    dfx.log_accept0_census(num_sampled, 4)
    recs = _messages(capture)
    assert len(recs) == 1, recs
    assert "accept0_reqs=2/4" in recs[0]
    assert "req_ids_first8=[0, 3]" in recs[0]


def test_accept0_excludes_draftless_requests(capture):
    """Regression: with packed cu_num_logits, a request with no scheduled
    drafts (a single row) also stores num_sampled == 1 -- with nothing to
    reject. Both numerator and denominator must count only requests that
    actually had drafts. With cu=[0, 1, 5, 8]: request 0 has no drafts
    (num_sampled=1, excluded), request 1 has three drafts (num_sampled=6),
    request 2 has two drafts (num_sampled=1, counted)."""
    num_sampled = torch.tensor([1, 6, 1], dtype=torch.int32)
    cu = torch.tensor([0, 1, 5, 8], dtype=torch.int64)
    dfx.log_accept0_census(num_sampled, 3, cu)
    recs = _messages(capture)
    assert len(recs) == 1, recs
    assert "accept0_reqs=1/2(had_drafts)" in recs[0]
    assert "req_ids_first8=[2]" in recs[0]


def test_accept0_normal_silent(capture):
    dfx.log_accept0_census(torch.tensor([6, 6], dtype=torch.int32), 2)
    assert capture.records == []


def test_census_never_raises(capture):
    """Malformed input must not take down the sampler (never-raise guard)."""
    dfx.log_nan_census(torch.zeros(0, V), None, K)  # zero rows
    dfx.log_nan_census(torch.zeros(3, 0), None, K)  # zero vocab
    dfx.log_accept0_census(torch.zeros(0, dtype=torch.int32), 0)
    # no exception escaped; failures would surface as census-failed DEBUGs
