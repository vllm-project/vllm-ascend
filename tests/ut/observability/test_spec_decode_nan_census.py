# UT for the [dfx/nan] DFX probes in observability/spec_decode_dfx.
#
# The probes are pure torch (no Triton launch) so they run in the CPU UT
# environment; the full rejection_sample wiring plus kernel behavior under
# NaN is covered by the NPU nightly e2e (test_rejection_sample_v2.py).
#
# Timing contract (issue/consume split, per the async-scheduling review):
#   * issue_nan_census runs at the rejection_sample ENTRY, before any
#     verify kernel is enqueued -- its reductions enter the stream ahead
#     of every logits reader, so stream order pins them to THIS step's
#     data even when async scheduling recycles the logits buffers before
#     the exit runs.
#   * consume_nan_census runs at the EXIT, after the accept0 verdict's
#     sync; it reads ONLY the small snapshot tensors held by the pending
#     object, never the (possibly recycled) logits buffers.
#
# Verified contract:
#   * clean inputs  -> zero log records even at DEBUG (no log spam on
#     healthy steps)
#   * any NaN       -> exactly one record per input tensor carrying
#     per-row req/step coordinates and NaN counts
#   * INFO level    -> zero records AND zero tensor work (gate first line;
#     issue returns None)
#   * snapshot deferral: mutating the logits tensors AFTER issue must not
#     change what consume reports (the regression for async buffer reuse)
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


def _run_census(target, draft, steps, cu=None):
    """Two-phase path: issue at the entry, consume at the exit."""
    pending = dfx.issue_nan_census(target, draft, steps, cu)
    dfx.consume_nan_census(pending)


def test_clean_inputs_silent(capture):
    _run_census(_clean_target(), torch.zeros(4, K, V), K)
    assert capture.records == [], f"clean inputs must stay silent, got {_messages(capture)}"


def test_partial_nan_single_element(capture):
    # row 6 = request 1, step 0 (rows_per_req = K + 1 = 6)
    _run_census(_with_nan(_clean_target(), 6, 1), None, K)
    recs = _target_records(capture)
    assert len(recs) == 1, recs
    msg = recs[0]
    assert "rows_with_nan=1/12" in msg
    assert "total_nan=1" in msg
    assert "req=1 step=0 nan=1" in msg
    assert _draft_records(capture) == []


def test_whole_row_nan(capture):
    _run_census(_with_nan(_clean_target(), 3, V), None, K)
    recs = _target_records(capture)
    assert len(recs) == 1, recs
    assert "total_nan=64" in recs[0]  # the whole-row upper bound
    assert "req=0 step=3 nan=64" in recs[0]


def test_multiple_rows_nan(capture):
    t = _with_nan(_clean_target(), 2, 3)  # req 0, step 2
    t = _with_nan(t, 7, 5)  # req 1, step 1
    _run_census(t, None, K)
    recs = _target_records(capture)
    assert len(recs) == 1, recs
    assert "rows_with_nan=2/12" in recs[0]
    assert "total_nan=8" in recs[0]
    assert "req=0 step=2 nan=3" in recs[0]
    assert "req=1 step=1 nan=5" in recs[0]


def test_draft_logits_nan(capture):
    draft = torch.zeros(4, K, V)
    draft[1, 3, :7] = float("nan")  # slot 1, step 3
    _run_census(_clean_target(), draft, K)
    recs = _draft_records(capture)
    assert len(recs) == 1, recs
    assert "rows_with_nan=1" in recs[0]
    assert "total_nan=7" in recs[0]
    assert "slot=1 step=3 nan=7" in recs[0]


def test_draft_logits_none_no_crash(capture):
    _run_census(_clean_target(), None, K)
    assert capture.records == []
    # and with a poisoned target while draft is None:
    _run_census(_with_nan(_clean_target(), 0, 2), None, K)
    assert len(_target_records(capture)) == 1
    assert _draft_records(capture) == []


def test_draft_logits_dummy_skipped(capture):
    """The wiring normalizes draft_logits=None to a 1-element dummy tensor
    before calling issue; the dummy must not be treated as real logits."""
    dummy = torch.empty(1, 1, 1)
    _run_census(_clean_target(), dummy, K)
    assert capture.records == []
    _run_census(_with_nan(_clean_target(), 5, 3), dummy, K)
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
    _run_census(t, None, 3, boundaries)
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
    _run_census(t, None, K, boundaries)
    recs = _target_records(capture)
    assert len(recs) == 1, recs
    assert "req=1 step=0 nan=1" in recs[0]


def test_snapshot_survives_buffer_reuse(capture):
    """THE regression for the async-scheduling review finding: by consume
    time the logits buffers may already be recycled and rewritten by the
    next step. The pending snapshot must reflect the entry-time data, so
    overwriting the source tensors between issue and consume must not
    change the report."""
    t = _with_nan(_clean_target(), 6, 2)  # req 1, step 0, nan=2
    pending = dfx.issue_nan_census(t, None, K)
    # simulate the next step overwriting the recycled buffer: poison a
    # DIFFERENT row and clear the original one
    t[:] = 0.0
    t[0, :V] = float("nan")  # row 0 heavily poisoned post-issue
    dfx.consume_nan_census(pending)
    recs = _target_records(capture)
    assert len(recs) == 1, recs
    # entry-time snapshot: row 6, not the post-overwrite row 0
    assert "req=1 step=0 nan=2" in recs[0]
    assert "req=0" not in recs[0]


def test_issue_returns_none_at_info(capture_info):
    t = _with_nan(_clean_target(), 6, V)  # poisoned, but gate closed
    pending = dfx.issue_nan_census(t, torch.zeros(4, K, V), K)
    assert pending is None, "INFO gate must skip issue entirely (no tensor work)"
    dfx.consume_nan_census(pending)  # must tolerate None
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
    _run_census(torch.zeros(0, V), None, K)  # zero rows
    _run_census(torch.zeros(3, 0), None, K)  # zero vocab
    dfx.issue_nan_census(torch.zeros(0, V), torch.zeros(0, K, 0), K)
    dfx.consume_nan_census(None)  # degenerate pending
    dfx.log_accept0_census(torch.zeros(0, dtype=torch.int32), 0)
    # no exception escaped; failures would surface as census-failed DEBUGs
