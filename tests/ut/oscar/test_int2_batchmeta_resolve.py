# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the INT2 batch-metadata resolve mixin (module level).

Pins the slot-resolution contract: explicit finish/preempt frees; absence is
NOT finish; free-list reuse; fail-closed at capacity; immediate zero on
assignment; b-keyed fallback when the scheduler gives no request ids.
"""

from types import SimpleNamespace

import torch

from vllm_ascend.attention.int2.batchmeta_resolve import Int2BatchMetaResolveMixin


def _clean_env(monkeypatch):
    for var in (
        "R164_BATCHED_STAGE",
        "R196_EAGER_BATCH",
        "R203_METAFIX",
        "R166_SLOTALGO",
        "R167_IDFILL",
        "R170_ROWSLOT",
        "R144_TT_DYN_DBG",
    ):
        monkeypatch.delenv(var, raising=False)


class _MetaSelf(Int2BatchMetaResolveMixin):
    # class-level attrs the verbatim bodies expect on the backend class
    _r164_slotmap = None
    _r164_slotctr = None
    _r196_idfill_state = None
    _r196_idfill_state2 = None

    def __init__(self, mseq=4):
        self._tt_slotmap = {}
        self._tt_free = set()
        self._tt_next = 0
        self._tt_mseq = mseq
        self._tt_pref = {}
        self._tt_rcnt = {}
        self._tt_rcnt_seen = {}
        self._tt_ring_org = {}
        self._tt_known = set()
        self._tt_dbg = False
        self._tt_cur_rids = None
        self._tt_cur_slots = None
        self._hp = (
            torch.ones(mseq, 8, 2, 4, dtype=torch.float32),
            torch.ones(mseq, 8, 2, 4, dtype=torch.float32),
        )

    def _tt_hp_bufs(self, lid, device, dtype):
        return self._hp


def _md(rids, finished=(), preempted=(), seq_lens=None):
    return SimpleNamespace(
        request_ids=list(rids),
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
        seq_lens_list=list(seq_lens or [1] * len(rids)),
    )


class TestResolveSlots:
    def test_new_assignments_are_monotonic_then_free_list_reuse(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _MetaSelf(mseq=4)
        s._tt_resolve_slots(_md(["a", "b"]), 0, "cpu", torch.float32)
        assert s._tt_slotmap == {"a": 0, "b": 1}  # NEW: monotonic
        assert s._tt_cur_slots == [0, 1]
        s._tt_resolve_slots(_md(["a"], finished=["b"]), 0, "cpu", torch.float32)
        assert s._tt_slotmap == {"a": 0}  # finish frees slot 1
        s._tt_resolve_slots(_md(["a", "c"]), 0, "cpu", torch.float32)
        assert s._tt_slotmap["c"] == 1  # REUSE from free set

    def test_absence_is_not_finish_state_preserved(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _MetaSelf(mseq=4)
        s._tt_resolve_slots(_md(["a", "b"]), 0, "cpu", torch.float32)
        s._tt_pref["b"] = 17
        s._tt_resolve_slots(_md(["a"]), 0, "cpu", torch.float32)  # b absent, not finished
        assert s._tt_slotmap.get("b") == 1  # slot retained
        assert s._tt_pref["b"] == 17  # state retained
        s._tt_resolve_slots(_md(["a", "b"]), 0, "cpu", torch.float32)
        assert s._tt_slotmap["b"] == 1  # same slot on re-appearance

    def test_map_full_fails_closed_no_lru(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _MetaSelf(mseq=1)
        s._tt_resolve_slots(_md(["a"]), 0, "cpu", torch.float32)
        try:
            s._tt_resolve_slots(_md(["a", "b"]), 0, "cpu", torch.float32)
        except RuntimeError as e:
            assert "slot capacity exceeded" in str(e)
        else:
            raise AssertionError("expected fail-closed RuntimeError")

    def test_preempt_frees_like_finish(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _MetaSelf(mseq=4)
        s._tt_resolve_slots(_md(["a", "b"]), 0, "cpu", torch.float32)
        s._tt_pref["b"] = 5
        s._tt_resolve_slots(_md(["a"], preempted=["b"]), 0, "cpu", torch.float32)
        assert "b" not in s._tt_slotmap and "b" not in s._tt_pref

    def test_assignment_and_finish_zero_hp_rows(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _MetaSelf(mseq=4)
        hk, hv = s._hp
        hk[0] = 9.0
        hv[1] = 9.0
        s._tt_resolve_slots(_md(["a", "b"]), 0, "cpu", torch.float32)
        assert hk[0].abs().max().item() == 0.0  # zeroed on assignment
        assert hv[1].abs().max().item() == 0.0
        hk[1] = 7.0
        s._tt_resolve_slots(_md(["a"], finished=["b"]), 0, "cpu", torch.float32)
        assert hk[1].abs().max().item() == 0.0  # zeroed on finish

    def test_no_request_ids_keeps_bkeyed_fallback(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _MetaSelf(mseq=4)
        s._tt_resolve_slots(SimpleNamespace(request_ids=None), 0, "cpu", torch.float32)
        assert s._tt_cur_rids is None and s._tt_cur_slots is None


def test_reuse_assignment_is_deterministic(monkeypatch):
    # two independent resolvers fed the same lifecycle must hand out the
    # same slot sequence (lowest free slot first), independent of set
    # iteration order (PYTHONHASHSEED varies per process)
    _clean_env(monkeypatch)
    seqs = []
    for _ in range(2):
        s = _MetaSelf(mseq=8)
        s._tt_resolve_slots(_md(["a", "b", "c"]), None, "cpu", torch.float32)
        s._tt_resolve_slots(_md(["a", "b"], finished=["c"]), None, "cpu", torch.float32)
        s._tt_resolve_slots(_md(["a", "b", "d"]), None, "cpu", torch.float32)
        seqs.append(dict(s._tt_slotmap))
    assert seqs[0] == seqs[1]
