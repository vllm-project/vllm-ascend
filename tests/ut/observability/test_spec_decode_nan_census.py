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
#   * census input set == verifier read set, enforced by three masks:
#     padded vocab columns beyond vocab_size are ignored (target and
#     draft), draft rows of greedy (temperature == 0) requests are zeroed,
#     and the draft census is restricted to the ACTIVE (slot, step) pairs
#     the verify kernels read -- draft_logits is a persistent
#     [max_num_reqs, K, V] buffer whose inactive slots / unscheduled steps
#     hold stale data that must NOT be reported (and must not take the
#     limited display slots); bonus rows read no draft and are zeroed
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


def _run_census(target, draft, steps, cu=None, slot=None, step=None, vocab_size=None, temperature=None):
    """Two-phase path: issue at the entry, consume at the exit."""
    pending = dfx.issue_nan_census(target, draft, steps, cu, slot, step, vocab_size, temperature)
    dfx.consume_nan_census(pending)


def _uniform_mappings(num_logits=NUM_LOGITS, k=K):
    """Slot/step mappings for a uniform batch: every request occupies
    k+1 packed rows (k draft rows + 1 bonus row at step == k), with
    request r parked in draft-buffer slot r."""
    rows_per_req = k + 1
    slot = torch.tensor([i // rows_per_req for i in range(num_logits)], dtype=torch.int32)
    step = torch.tensor([i % rows_per_req for i in range(num_logits)], dtype=torch.int32)
    return slot, step


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
    """Active draft NaN is reported with req/step/slot coordinates; a NaN
    in an INACTIVE slot (persistent buffer, not in idx_mapping) is stale
    data and must not be reported -- it must not take a display slot."""
    draft = torch.zeros(4, K, V)
    draft[0, 1, :7] = float("nan")  # slot 0 = req 0, step 1 -- ACTIVE
    draft[3, 2, :V] = float("nan")  # slot 3 not in idx_mapping -- STALE
    slot, step = _uniform_mappings()
    cu = torch.tensor([0, 6, 12], dtype=torch.int64)
    _run_census(_clean_target(), draft, K, cu, slot, step)
    recs = _draft_records(capture)
    assert len(recs) == 1, recs
    assert "rows_with_nan=1" in recs[0]
    assert "total_nan=7" in recs[0]
    assert "req=0 step=1 slot=0 nan=7" in recs[0]
    assert "slot=3" not in recs[0]


def test_draft_inactive_slot_not_reported(capture):
    """A poisoned slot that no request maps to this step is stale buffer
    content -- the census stays silent instead of reporting it."""
    draft = torch.zeros(4, K, V)
    draft[3, 2, :5] = float("nan")  # slot 3 unused by the active batch
    slot, step = _uniform_mappings()
    cu = torch.tensor([0, 6, 12], dtype=torch.int64)
    _run_census(_clean_target(), draft, K, cu, slot, step)
    assert capture.records == [], f"stale slot must stay silent, got {_messages(capture)}"


def test_draft_shorter_draft_stale_step_not_reported(capture):
    """Regression: a request scheduled fewer drafts than K leaves its
    higher step positions unpopulated this step. cu=[0, 6, 9]: req 0 has
    five drafts (rows 0-4 + bonus at 5), req 1 has TWO drafts (rows 6-7 +
    bonus at 8). A NaN at req 1's step 3 is stale -- only steps 0-1 were
    scheduled and only those are read by the verifier."""
    draft = torch.zeros(4, K, V)
    draft[1, 3, :5] = float("nan")  # beyond req 1's scheduled steps -- STALE
    draft[1, 0, :4] = float("nan")  # req 1, step 0 -- ACTIVE
    slot = torch.tensor([0] * 6 + [1] * 3, dtype=torch.int32)
    step = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.int32)
    cu = torch.tensor([0, 6, 9], dtype=torch.int64)
    _run_census(_clean_target(), draft, K, cu, slot, step)
    recs = _draft_records(capture)
    assert len(recs) == 1, recs
    assert "req=1 step=0 slot=1 nan=4" in recs[0]
    assert "step=3" not in recs[0]


def test_draft_bonus_row_not_reported(capture):
    """The bonus row (each request's last packed row) reads no draft in
    the verifier; its clamped gather is discarded, so a poisoned last
    draft step is reported at its own step, never as a bonus row."""
    draft = torch.zeros(4, K, V)
    draft[0, 4, :2] = float("nan")  # req 0, step 4 (last real draft) -- ACTIVE
    slot, step = _uniform_mappings()  # rows 5/11 carry step == K (bonus)
    cu = torch.tensor([0, 6, 12], dtype=torch.int64)
    _run_census(_clean_target(), draft, K, cu, slot, step)
    recs = _draft_records(capture)
    assert len(recs) == 1, recs
    assert "req=0 step=4 slot=0 nan=2" in recs[0]
    assert "step=5" not in recs[0]


def test_draft_full_scan_legacy_coords(capture):
    """Without row mappings the census falls back to the legacy full scan
    with uniform slot coordinates."""
    draft = torch.zeros(4, K, V)
    draft[1, 3, :7] = float("nan")  # slot 1, step 3
    _run_census(_clean_target(), draft, K)
    recs = _draft_records(capture)
    assert len(recs) == 1, recs
    assert "rows_with_nan=1" in recs[0]
    assert "total_nan=7" in recs[0]
    assert "slot=1 step=3 nan=7" in recs[0]


def test_padded_target_vocab_not_reported(capture):
    """The verifier masks columns to the effective vocab (min of target
    and draft widths); NaNs in padded target columns never reach it and
    must not be reported. Target width 64, draft width 32 -> effective
    vocab 32: a NaN at column 40 is invisible, one at column 5 is real."""
    target = torch.zeros(NUM_LOGITS, 64)
    target[6, 40] = float("nan")  # padded column -- invisible to the verifier
    target[7, 5] = float("nan")  # effective column -- real
    draft = torch.zeros(4, K, 32)
    slot, step = _uniform_mappings()
    cu = torch.tensor([0, 6, 12], dtype=torch.int64)
    _run_census(target, draft, K, cu, slot, step, vocab_size=32)
    recs = _target_records(capture)
    assert len(recs) == 1, recs
    assert "rows_with_nan=1/12" in recs[0]
    assert "total_nan=1" in recs[0]  # only the effective-column NaN
    assert "req=1 step=1 nan=1" in recs[0]  # row 7, not the padded row 6


def test_padded_draft_vocab_not_reported(capture):
    """Reverse padding: draft wider than target (effective vocab 32, draft
    width 64). The verifier reads only the effective vocab columns of the
    gathered draft rows; a NaN in a padded draft column must not be
    reported."""
    draft = torch.zeros(4, K, 64)
    draft[0, 1, 40] = float("nan")  # padded draft column -- invisible
    draft[0, 1, 5] = float("nan")  # effective column -- real
    slot, step = _uniform_mappings()
    cu = torch.tensor([0, 6, 12], dtype=torch.int64)
    _run_census(torch.zeros(NUM_LOGITS, 32), draft, K, cu, slot, step, vocab_size=32)
    recs = _draft_records(capture)
    assert len(recs) == 1, recs
    assert "total_nan=1" in recs[0]
    assert "req=0 step=1 slot=0 nan=1" in recs[0]


def test_greedy_draft_rows_not_reported(capture):
    """Mixed greedy/stochastic batch: the verifier skips draft reads for
    temperature == 0 requests entirely (argmax acceptance), so NaNs in
    their draft rows are not verification inputs. NaNs in the greedy
    TARGET rows are still reported (argmax reads them)."""
    draft = torch.zeros(4, K, V)
    draft[0, 1, :7] = float("nan")  # req 0 (greedy, slot 0) -- skipped
    draft[1, 2, :5] = float("nan")  # req 1 (stochastic, slot 1) -- real
    slot, step = _uniform_mappings()
    cu = torch.tensor([0, 6, 12], dtype=torch.int64)
    temperature = torch.tensor([0.0, 1.5, 1.0, 1.0], dtype=torch.float32)
    target = _clean_target()
    target[0, :3] = float("nan")  # greedy target NaN -- still a real input
    _run_census(target, draft, K, cu, slot, step, vocab_size=None, temperature=temperature)
    recs = _draft_records(capture)
    assert len(recs) == 1, recs
    assert "req=1 step=2 slot=1 nan=5" in recs[0]
    assert "slot=0" not in recs[0]
    trecs = _target_records(capture)
    assert len(trecs) == 1, trecs
    assert "req=0 step=0 nan=3" in trecs[0]


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
    # active-pair path with degenerate shapes
    _run_census(
        torch.zeros(0, V),
        torch.zeros(0, K, 0),
        K,
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(0, dtype=torch.int32),
        torch.zeros(0, dtype=torch.int32),
    )
    dfx.consume_nan_census(None)  # degenerate pending
    # degenerate masks: zero effective vocab, all-greedy temperature
    _run_census(_clean_target(), torch.zeros(4, K, V), K, vocab_size=0)
    _run_census(
        _with_nan(_clean_target(), 6, 2),
        torch.zeros(4, K, V),
        K,
        slot=torch.zeros(NUM_LOGITS, dtype=torch.int32),
        step=torch.zeros(NUM_LOGITS, dtype=torch.int32),
        temperature=torch.zeros(4, dtype=torch.float32),
    )
    dfx.log_accept0_census(torch.zeros(0, dtype=torch.int32), 0)
    # no exception escaped; failures would surface as census-failed DEBUGs
