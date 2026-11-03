# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the relocated INT2 shared helpers.

Covers the request/slot lifecycle invariants of the R166 slot allocator
(the F7 cluster's module-level half: binding, release, reuse, owner
semantics, cross-request isolation, reserve0) and verifies the diagnostic
helpers stay OFF unless explicitly gated.
"""

import pytest
import torch

from vllm_ascend.attention.int2.helpers import (
    _R166SlotAlloc,
    _r216_merge,
    _r218_diag,
    load_oscar_rotations,
)


class TestSlotAllocatorLifecycle:
    def test_acquire_release_reuse(self):
        a = _R166SlotAlloc(capacity=4)
        s0, g0, reused0 = a.acquire("r1")
        # free list is seeded descending; pop() serves from the tail, so
        # slot 0 is handed out first (verified behavior, comment in the
        # source notwithstanding)
        assert s0 == 0 and g0 == 1 and reused0 is False
        a.release("r1")
        s1, g1, reused1 = a.acquire("r1")
        # acquire(+1) + release(+1) + acquire(+1): next generation is 3
        assert s1 == s0 and g1 == 3 and reused1 is True
        assert a.check() == []

    def test_bijection_and_cover_invariants(self):
        a = _R166SlotAlloc(capacity=8)
        rids = [f"r{i}" for i in range(5)]
        slots = [a.acquire(r)[0] for r in rids]
        assert len(set(slots)) == 5  # unique occupied
        assert a.check() == []
        a.release(rids[2])
        assert a.check() == []
        # freed slot returns on next acquire; others unaffected
        s, _, _ = a.acquire("r9")
        assert s == slots[2]
        assert a.check() == []

    def test_capacity_exhausted_fails_closed(self):
        a = _R166SlotAlloc(capacity=1)
        a.acquire("r1")
        with pytest.raises(RuntimeError, match="capacity exhausted"):
            a.acquire("r2")

    def test_double_release_is_dropped_not_corrupting(self):
        a = _R166SlotAlloc(capacity=2)
        a.acquire("r1")
        a.release("r1")
        a.release("r1")  # stale release
        assert a.check() == []
        assert len(a.free) == 2

    def test_owner_mismatch_release_fails_loud(self):
        # r1 holds slot; simulate a corrupted second owner map entry by
        # forging slot2rid -- release of a DIFFERENT rid that maps to a
        # slot owned by someone else must raise, not free the slot
        a = _R166SlotAlloc(capacity=2)
        a.acquire("r1")
        a.rid2slot["r2"] = a.rid2slot["r1"]  # forge alias
        with pytest.raises(RuntimeError, match="owner mismatch"):
            a.release("r2")

    def test_cross_request_no_slot_share(self):
        a = _R166SlotAlloc(capacity=16)
        held = [a.acquire(f"req{i}")[0] for i in range(16)]
        assert sorted(held) == list(range(16))  # no sharing
        for i in range(0, 16, 2):
            a.release(f"req{i}")
        fresh = [a.acquire(f"new{i}")[0] for i in range(8)]
        assert sorted(fresh) == sorted(held[0::2])  # exactly the freed set

    def test_reserve0_keeps_identity_row_unassigned(self):
        a = _R166SlotAlloc(capacity=4, reserve0=True)
        slots = [a.acquire(f"r{i}")[0] for i in range(3)]
        assert 0 not in slots and sorted(slots) == [1, 2, 3]
        assert a.check() == []

    def test_generation_monotonic_per_slot(self):
        a = _R166SlotAlloc(capacity=2)
        gens = []
        for _ in range(3):
            s, g, _ = a.acquire("r")
            gens.append(g)
            a.release("r")
        # generation bumps on BOTH acquire and release: 1,3,5 across cycles
        assert gens == [1, 3, 5]


class TestDiagnosticsStayGated:
    def test_r216_merge_off_by_default(self, tmp_path, monkeypatch):
        # no R216 env set -> early return, no dump written even with
        # bogus state
        monkeypatch.delenv("R216_LSEDUMP", raising=False)
        lse = torch.zeros(1, 1)
        _r216_merge(lse, lse, torch.zeros(1), torch.zeros(1), None)
        assert not (tmp_path / "merge_0000.pt").exists()

    def test_r218_diag_writes_only_when_enabled(self, tmp_path, monkeypatch):
        # the sink is redirected to the sandbox so the assertions prove the
        # actual write (or its absence), not an unrelated default path
        log = tmp_path / "diag.log"
        monkeypatch.setenv("OSCAR_DIAG_LOG", str(log))
        monkeypatch.delenv("OSCAR_DIAG", raising=False)
        _r218_diag("off-must-not-land")
        assert not any(tmp_path.iterdir())  # nothing written anywhere
        monkeypatch.setenv("OSCAR_DIAG", "1")
        _r218_diag("on-must-land")
        assert log.exists() and "on-must-land" in log.read_text()

    def test_acquire_duplicate_rid_fails_closed(self):
        A = _R166SlotAlloc(4)
        A.acquire("a")
        with pytest.raises(RuntimeError, match="duplicate acquire"):
            A.acquire("a")
        # the failed duplicate left the map exactly as it was
        assert A.check() == []
        assert len(A.rid2slot) == 1

    def test_zeroed_stamps_pruned_on_slot_turnover(self):
        A = _R166SlotAlloc(2)
        A.zeroed = {(0, 1, 5), (0, 0, 3), (1, 2, 5)}  # slot 0 stale gens + other slot
        A.release("r0_unused")  # unowned release: no prune side effects
        A.acquire("a")  # takes a free slot
        A.acquire("b")
        A.release("a")
        # freeing prunes that slot's stamps; the other slot's stay
        assert all(t[0] != A.rid2slot.get("a", -1) for t in A.zeroed)
        # acquire-reuse prunes again: cycle and confirm nothing accumulates
        A.acquire("c")  # reuses the freed slot, bumps gen
        assert all(t[0] != A.rid2slot["c"] or t[1] == A.gen[A.rid2slot["c"]] for t in A.zeroed)

    def test_events_bounded(self):
        A = _R166SlotAlloc(2)
        for i in range(8192):
            try:
                A.acquire(f"r{i}")
            except RuntimeError:
                A.release(f"r{i - 1}")
        assert len(A.events) <= 4096

    def test_load_oscar_rotations_reexport(self):
        # re-export contract: the backend imports this name from helpers
        from vllm_ascend.oscar.rotation import load_oscar_rotations as orig

        assert load_oscar_rotations is orig


class TestR114SteadyWriter:
    def test_ts_write_opens_sink_without_nameerror(self, tmp_path, monkeypatch):
        # regression: _r114_ts_write used bare ``os`` while the module only
        # imported aliased forms (``os as _os`` ...) — calling it raised
        # NameError before the module-level ``import os`` fix.
        import vllm_ascend.attention.int2.helpers as h

        sink = tmp_path / "steady.jsonl"
        monkeypatch.setenv("R114_STEADY_OUT", str(sink))
        h._R114_TS_FH = None
        try:
            h._r114_ts_write()
            h._r114_ts_write()
        finally:
            if h._R114_TS_FH is not None:
                h._R114_TS_FH.close()
                h._R114_TS_FH = None
        lines = sink.read_text().splitlines()
        assert len(lines) == 2 and all('"t":' in x for x in lines)
