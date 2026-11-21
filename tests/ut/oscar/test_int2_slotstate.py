# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the INT2 slot-state mixin (module level).

Small deterministic fixtures against the verbatim-relocated methods, driven
through a minimal fake ``self`` carrying only the attributes the methods
read. These pin the request-slot lifecycle semantics: stable request id ->
reusable slot, owner-checked release, zero-on-reuse, and the stable
layer-ordinal registry. Device-dependent arms (NPU ring zeroing inside the
real HP buffers) are covered by the device battery.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_ascend.attention.int2.helpers import _R166SlotAlloc
from vllm_ascend.attention.int2.slotstate import Int2SlotStateMixin

_L0 = SimpleNamespace(layer_name="model.layers.0.self_attn")
_L1 = SimpleNamespace(layer_name="model.layers.1.self_attn")


@pytest.fixture(autouse=True)
def _fresh_class_state():
    # the allocator/slotmap live on the CLASS (shared across impls by
    # design); reset between tests so each test starts from a clean map
    _SlotSelf._r166_alloc = None
    _SlotSelf._r164_slotmap = None
    _SlotSelf._oscar_name2ord = None
    yield
    _SlotSelf._r166_alloc = None
    _SlotSelf._r164_slotmap = None


def _clean_env(monkeypatch):
    for var in (
        "R164_PREFILL_SAFE",
        "R164_WCHK",
        "R164_WRITEAUDIT",
        "R166_STREAMFIX",
        "R166_SLOTCHECK",
        "R137_RK_PROOF",
        "R164_SLOTMAP",
        "R166_SLOTALGO",
    ):
        monkeypatch.delenv(var, raising=False)


class _SlotSelf(Int2SlotStateMixin):
    # class-level attrs the verbatim bodies expect on the backend class
    _r166_alloc: Any = None
    _r164_slotmap: Any = None
    _oscar_name2ord: Any = None

    def __init__(self, mseq=8, max_hp=32, nkv=2, hd=4):
        self._tt_ready = False
        self._tt_mseq = mseq
        self._tt_pref = {}
        self._tt_rcnt = {}
        self._tt_rcnt_seen = {}
        self._tt_ring_org = {}
        self._tt_cur_rids = None
        self._tt_cur_slots = None
        self._oscar_R_k = None
        self._oscar_R_v = None
        # HP buffers with the real layout: [max_seqs, max_hp, nkv, hd]
        self._hp = (
            torch.ones(mseq, max_hp, nkv, hd, dtype=torch.float32),
            torch.ones(mseq, max_hp, nkv, hd, dtype=torch.float32),
        )

    def _tt_hp_bufs(self, lid, device, dtype):
        return self._hp


def _md(rids, finished=(), preempted=()):
    return SimpleNamespace(request_ids=list(rids), finished_req_ids=set(finished), preempted_req_ids=set(preempted))


class TestRidSlotMapping:
    def test_bkeyed_fallback_returns_row(self):
        s = _SlotSelf()
        assert s._tt_rid(3) == 3 and s._tt_slot(3) == 3  # verbatim b-keyed path

    def test_bound_maps_route_through_current_lists(self):
        s = _SlotSelf()
        s._tt_cur_rids = ["rA", "rB", "rC"]
        s._tt_cur_slots = [5, 0, 7]
        assert [s._tt_rid(b) for b in range(3)] == ["rA", "rB", "rC"]
        assert [s._tt_slot(b) for b in range(3)] == [5, 0, 7]


class TestLayerOrdinal:
    def test_parses_model_layers_index(self):
        s = _SlotSelf()
        assert s._oscar_layer_ord(SimpleNamespace(layer_name="model.layers.7.self_attn")) == 7

    def test_registry_survives_object_recreation(self):
        s = _SlotSelf()
        a = s._oscar_layer_ord(SimpleNamespace(layer_name="model.layers.2.mlp"))
        b = s._oscar_layer_ord(SimpleNamespace(layer_name="model.layers.2.mlp"))
        assert a == b == 2

    def test_fail_closed_without_layer_name(self):
        s = _SlotSelf()
        try:
            s._oscar_layer_ord(SimpleNamespace())
        except RuntimeError as e:
            assert "layer_name" in str(e)
        else:
            raise AssertionError("expected fail-closed RuntimeError")

    def test_fail_closed_on_unparsable_name(self):
        s = _SlotSelf()
        try:
            s._oscar_layer_ord(SimpleNamespace(layer_name="encoder.h.three"))
        except RuntimeError as e:
            assert "parse" in str(e)
        else:
            raise AssertionError("expected fail-closed RuntimeError")

    def test_fail_closed_on_rotation_row_overflow(self):
        s = _SlotSelf()
        s._oscar_R_k = torch.zeros(4, 2, 2)
        s._oscar_R_v = torch.zeros(4, 2, 2)
        try:
            s._oscar_layer_ord(SimpleNamespace(layer_name="model.layers.9.mlp"))
        except RuntimeError as e:
            assert "R_k rows" in str(e)
        else:
            raise AssertionError("expected config-mismatch RuntimeError")


class TestR166BindSlots:
    def test_bind_assigns_and_mirrors_slotmap(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _SlotSelf(mseq=4)
        s._r166_bind_slots(_md(["a", "b"]), "cpu", torch.float32, _L0)
        A = _SlotSelf._r166_alloc
        # allocator pops the LOWEST free slot first (free list is stored
        # high-to-low and .pop() takes the tail) — pinned as-implemented
        assert A.rid2slot == {"a": 0, "b": 1}
        assert s._tt_cur_rids == ["a", "b"]
        assert s._tt_cur_slots == [0, 1]
        assert _SlotSelf._r164_slotmap == {"a": 0, "b": 1}

    def test_finished_releases_and_slot_is_reused(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _SlotSelf(mseq=4)
        s._r166_bind_slots(_md(["a", "b"]), "cpu", torch.float32, _L0)
        A: Any = type(s)._r166_alloc
        s._r166_bind_slots(_md(["a"], finished=["b"]), "cpu", torch.float32, _L0)
        assert "b" not in A.rid2slot and 1 in A.free
        assert _SlotSelf._r164_slotmap.get("b") is None
        s._r166_bind_slots(_md(["a", "c"]), "cpu", torch.float32, _L0)
        assert A.rid2slot["c"] == 1  # reuse of the freed slot
        assert A.gen[1] == 3  # acquire + release + acquire

    def test_reused_slot_ring_rows_are_zeroed(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _SlotSelf(mseq=4)
        hk, hv = s._hp
        hk[1] = 5.0  # dirty prior-generation rows
        hv[1] = 5.0
        s._r166_bind_slots(_md(["a", "b"]), "cpu", torch.float32, _L0)
        s._r166_bind_slots(_md(["a"], finished=["b"]), "cpu", torch.float32, _L0)
        s._r166_bind_slots(_md(["a", "c"]), "cpu", torch.float32, _L0)
        assert hk[1].abs().max().item() == 0.0  # zero-on-reuse fired
        assert hv[1].abs().max().item() == 0.0
        A: Any = type(s)._r166_alloc
        assert (1, 3, 0) in A.zeroed  # once-only stamp (slot, gen, lid=0)

    def test_tier_state_reset_for_reacquired_request(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _SlotSelf(mseq=4)
        s._tt_pref["b"] = 9
        s._tt_rcnt["b"] = 4
        s._tt_ring_org["b"] = 1
        s._r166_bind_slots(_md(["a", "b"]), "cpu", torch.float32, _L0)
        assert "b" not in s._tt_pref  # fresh acquisition wipes state
        s._r166_bind_slots(_md(["a"], finished=["b"]), "cpu", torch.float32, _L0)
        s._tt_pref["b"] = 9  # re-prefill re-registers state
        s._r166_bind_slots(_md(["a", "b"]), "cpu", torch.float32, _L0)
        assert "b" not in s._tt_pref and "b" not in s._tt_rcnt

    def test_absence_is_not_finish(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _SlotSelf(mseq=4)
        s._r166_bind_slots(_md(["a", "b"]), "cpu", torch.float32, _L0)
        A: Any = type(s)._r166_alloc
        s._r166_bind_slots(_md(["a"]), "cpu", torch.float32, _L0)  # b absent, NOT finished
        assert "b" in A.rid2slot  # owner retains its slot
        assert _SlotSelf._r164_slotmap.get("b") == A.rid2slot["b"]


class TestAllocatorInvariants:
    def test_bijection_and_cover_after_churn(self):
        A = _R166SlotAlloc(4)
        for live in (["a"], ["a", "b"], ["a", "b", "c"], ["c"]):
            for r in [x for x in ("a", "b") if x not in live]:
                A.release(r)
            for r in live:
                if r not in A.rid2slot:
                    A.acquire(r)
        assert A.check() == []  # invariants hold after churn

    def test_release_unowned_is_dropped_not_raised(self):
        A = _R166SlotAlloc(2)
        A.release("ghost")  # stale/double release: no raise
        assert "ghost" not in A.rid2slot

    def test_acquire_fail_closed_when_exhausted(self):
        A = _R166SlotAlloc(1)
        A.acquire("a")
        try:
            A.acquire("b")
        except RuntimeError as e:
            assert "fail-closed" in str(e)
        else:
            raise AssertionError("expected fail-closed RuntimeError")


class TestRebindLocked:
    def test_move_to_free_slot_preserves_cover(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _SlotSelf(mseq=4)
        s._r166_bind_slots(_md(["a"]), "cpu", torch.float32, _L0)
        A: Any = type(s)._r166_alloc
        old = A.rid2slot["a"]  # a at slot 0
        A._lk.acquire()
        try:
            # static-by-convention: must be called through the CLASS
            _SlotSelf._r170_rebind_locked(A, "a", 2)  # 2 is free
        finally:
            A._lk.release()
        assert A.rid2slot["a"] == 2 and A.slot2rid[2] == "a"
        assert old in A.free and A.gen[2] == 1  # first use of slot 2
        assert A.check() == []

    def test_rebind_conflict_fail_closed(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _SlotSelf(mseq=4)
        s._r166_bind_slots(_md(["a", "b"]), "cpu", torch.float32, _L0)
        A: Any = type(s)._r166_alloc
        A._lk.acquire()
        try:
            _SlotSelf._r170_rebind_locked(A, "a", A.rid2slot["b"])  # occupied by b
        except RuntimeError as e:
            assert "rebind conflict" in str(e)
        else:
            raise AssertionError("expected fail-closed RuntimeError")
        finally:
            A._lk.release()


class TestWriteHp:
    def test_writes_rotated_window_in_place(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _SlotSelf(mseq=2, max_hp=32)
        hk, hv = s._hp
        k = torch.full((3, 2, 4), 2.0)
        v = torch.full((3, 2, 4), 3.0)
        s._tt_write_hp(0, 1, 4, (k, v))  # row 1, window [4,7)
        assert torch.equal(hk[1, 4:7], torch.full((3, 2, 4), 2.0))
        assert torch.equal(hv[1, 4:7], torch.full((3, 2, 4), 3.0))
        assert hk[1, :4].abs().max().item() == 1.0  # outside window untouched


class TestR164MonotonicBind:
    def test_monotonic_no_reuse_and_fail_closed(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("R164_SLOTMAP", "1")  # the slotmap policy gate
        s = _SlotSelf(mseq=2)
        s._r164_bind_slots(_md(["a", "b"]))
        assert _SlotSelf._r164_slotmap == {"a": 0, "b": 1}
        # finish does NOT free a slot under the monotonic policy (no reuse by design)
        s._r164_bind_slots(_md(["a"], finished=["b"]))
        assert _SlotSelf._r164_slotmap.get("b") == 1  # binding retained
        try:
            s._r164_bind_slots(_md(["a", "c"]))  # next=2 >= mseq=2
        except RuntimeError as e:
            assert "monotonic slot exhaustion" in str(e)
        else:
            raise AssertionError("expected fail-closed at mseq")
        assert "c" not in _SlotSelf._r164_slotmap  # nothing half-assigned


class TestOscarDbg:
    # the gated debug print lands in helpers with this module (its first
    # consumer); no-op unless SGLANG_OSCAR_DEBUG is set
    def test_noop_without_env(self, monkeypatch, capsys):
        monkeypatch.delenv("SGLANG_OSCAR_DEBUG", raising=False)
        from vllm_ascend.attention.int2.helpers import _oscar_dbg

        assert _oscar_dbg("hello", 1) is None
        assert capsys.readouterr().out == ""

    def test_prints_when_gated_on(self, monkeypatch, capsys):
        monkeypatch.setenv("SGLANG_OSCAR_DEBUG", "1")
        from vllm_ascend.attention.int2.helpers import _oscar_dbg

        _oscar_dbg("hello", 2)
        out = capsys.readouterr().out
        assert "hello 2" in out


class TestSharedAllocatorMultiLayer:
    def test_every_layer_impl_zeros_its_own_reused_rows(self, monkeypatch):
        # two per-layer impls share the CLASS-level allocator; the second
        # impl's bind (request already mapped by the first) must still
        # stamp-and-zero its own ring rows for the reused slot
        _clean_env(monkeypatch)
        s0 = _SlotSelf(mseq=4)
        s1 = _SlotSelf(mseq=4)
        md = _md(["a"])
        s0._r166_bind_slots(md, "cpu", torch.float32, _L0)
        s1._r166_bind_slots(md, "cpu", torch.float32, _L1)
        hk0, _ = s0._hp
        hk1, _ = s1._hp
        hk0[0].fill_(7.0)  # stale data from the previous owner
        hk1[0].fill_(9.0)
        md2 = _md(["b"], finished=["a"])  # slot 0 freed then reused by "b"
        s0._r166_bind_slots(md2, "cpu", torch.float32, _L0)
        s1._r166_bind_slots(md2, "cpu", torch.float32, _L1)
        # both impls acquired no rows this step themselves, but BOTH must
        # have zeroed the reused slot's rows on their own buffers
        assert hk0[0].abs().max().item() == 0.0
        assert hk1[0].abs().max().item() == 0.0
        A = _SlotSelf._r166_alloc
        assert (A.rid2slot["b"], A.gen[A.rid2slot["b"]], 0) in A.zeroed
        assert (A.rid2slot["b"], A.gen[A.rid2slot["b"]], 1) in A.zeroed
