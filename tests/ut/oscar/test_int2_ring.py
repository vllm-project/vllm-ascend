# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the INT2 prefix/recent ring mixin (module level).

Small deterministic fixtures against the verbatim-relocated methods, driven
through a minimal fake ``self`` carrying only the attributes the methods
read. These pin the virtual-ring index/window semantics (origin, modulo
position, wraparound gather order) and the registry handoff contract for the
two diag names the ring shares with the batch-metadata update module
(_R218_DISC / _R218_XAS). Device-dependent arms are covered by the device
battery.
"""

import pathlib
from typing import Any

import torch

import vllm_ascend.attention.int2.ctx as ctx
from vllm_ascend.attention.int2.ring import Int2RingMixin

_AUDIT = pathlib.Path("/tmp/r218_xaudit_on")


class _RingSelf(Int2RingMixin):
    # class-level attrs the methods cache on (mirrors the backend class
    # attribute layout the verbatim bodies expect)
    _r198_idx_cache: dict[Any, Any] = {}

    def __init__(self, org, pref, recent=8, mhp=32, mseq=4):
        self._tt_ring_org = org  # rid -> origin token
        self._tt_pref = pref  # rid -> prefix len
        self._TT_RECENT = recent
        self._tt_mhp = mhp
        self._tt_mseq = mseq

    def _tt_rid(self, b):
        return "r0"

    def _tt_slot(self, b):
        return 0


class TestRingPositionMath:
    def test_pos_is_prefix_plus_modulo_offset(self):
        s = _RingSelf({"r1": 10}, {"r1": 100})
        assert s._r198_pos("r1", 10) == 100  # origin maps to ring head
        assert s._r198_pos("r1", 13) == 103
        # wraparound: token origin+cap lands back on the same physical row
        assert s._r198_pos("r1", 10 + 8) == 100
        assert s._r198_pos("r1", 10 + 8 + 3) == 103

    def test_pos_no_origin_returns_none(self):
        s = _RingSelf({}, {})
        assert s._r198_pos("r1", 5) is None

    def test_win_ridx_formula(self):
        s = _RingSelf({"r1": 10}, {"r1": 6}, recent=8, mhp=32)
        hk = torch.zeros(1, 32, 2, 4)
        idx = s._r198_win_ridx("r1", 0, 14, 4, hk)  # sl=14, rcnt=4
        t0 = 14 - 4
        expect = [0 * 32 + 6 + ((t0 + j - 10) % 8) for j in range(4)]
        assert idx.tolist() == expect


class TestPrefixRecentWindow:
    def _hk_hv(self, rows=16, nkv=2, hd=4):
        hk = torch.arange(rows * nkv * hd, dtype=torch.float32).reshape(1, rows, nkv, hd)
        hv = hk * 10
        return hk, hv

    def test_legacy_cat_arm(self, monkeypatch):
        monkeypatch.delenv("R198_VRING", raising=False)
        s = _RingSelf({}, {})  # no origin -> legacy
        hk, hv = self._hk_hv()
        pr_k, pr_v = s._r198_pr_window("r1", 0, 10, 4, 6, hk, hv)
        assert torch.equal(pr_k, torch.cat([hk[0, :4], hk[0, 4:10]], dim=0))
        assert torch.equal(pr_v, torch.cat([hv[0, :4], hv[0, 4:10]], dim=0))

    def test_vring_gather_equals_legacy_when_contiguous(self, monkeypatch):
        _RingSelf._r198_idx_cache = {}
        # VRING arm with a window that does not wrap must produce exactly
        # the legacy cat content. Ring origin X = token at the ring HEAD =
        # oldest live token (sl - rcnt); rows then map 1:1 in order.
        monkeypatch.setenv("R198_VRING", "1")
        sl, pref, rcnt = 10, 4, 6
        s = _RingSelf({"r1": sl - rcnt}, {"r1": pref}, recent=32, mhp=32)
        hk, hv = self._hk_hv(rows=32)
        pr_k_v, _ = s._r198_pr_window("r1", 0, sl, pref, rcnt, hk, hv)
        expect = torch.cat([hk[0, :pref], hk[0, pref : pref + rcnt]], dim=0)
        assert torch.allclose(pr_k_v, expect)

    def test_vring_wraparound_row_order(self, monkeypatch):
        # window spanning the ring boundary gathers ROTATED row order:
        # physical row = pref + ((t - X) % cap); with cap=8, X=14 and
        # tokens 18..23 the rows are 8,9,10,11,4,5 -- legacy cat order is
        # NOT valid there, the modulo map is the contract
        _RingSelf._r198_idx_cache = {}
        monkeypatch.setenv("R198_VRING", "1")
        cap, pref, X, sl, rcnt = 8, 4, 14, 24, 6
        s = _RingSelf({"r1": X}, {"r1": pref}, recent=cap, mhp=32)
        hk, hv = self._hk_hv(rows=32)
        pr_k_v, _ = s._r198_pr_window("r1", 0, sl, pref, rcnt, hk, hv)
        rows = [pref + ((sl - rcnt + j - X) % cap) for j in range(rcnt)]
        assert rows == [8, 9, 10, 11, 4, 5]  # spans the boundary
        expect = torch.cat([hk[0, :pref], hk[0, rows]], dim=0)
        assert torch.allclose(pr_k_v, expect)

    def test_zero_window_returns_none(self, monkeypatch):
        monkeypatch.delenv("R198_VRING", raising=False)
        s = _RingSelf({}, {})
        hk, hv = self._hk_hv()
        assert s._r198_pr_window("r1", 0, 0, 0, 0, hk, hv) == (None, None)


class TestRegistryHandoff:
    """The ring shares two gated-diag names with the batch-metadata update
    module through ctx._R2XX_G. In the monolith both sides met in one module
    globals(); after the split only the registry keeps the handoff working
    (the first extraction silently isolated them)."""

    def _batched_ring_bails(self, monkeypatch):
        monkeypatch.setenv("R198_VRING", "1")
        s = _RingSelf({}, {})  # no origin -> pos None
        q = torch.zeros(1, 2)
        k = torch.zeros(1, 2, 2, 4)
        v = torch.zeros(1, 2, 2, 4)
        return s._r197_batched_ring(0, q, k, v, [5], 1)

    def test_disc_append_goes_to_the_shared_registry(self, monkeypatch):
        prev = _AUDIT.read_text() if _AUDIT.exists() else None
        ctx._R2XX_G.pop("_R218_DISC", None)
        _AUDIT.write_text("")  # arm the audit gate
        try:
            assert self._batched_ring_bails(monkeypatch) is False
            disc = ctx._R2XX_G.get("_R218_DISC")
            assert disc and disc[-1].startswith("ring_BAIL b=0 rid=r0 sl=5")
        finally:
            if prev is None:
                _AUDIT.unlink(missing_ok=True)
            else:
                _AUDIT.write_text(prev)
            ctx._R2XX_G.pop("_R218_DISC", None)

    def test_disc_reader_and_cleaner_share_the_object(self):
        # lifecycle contract: the batchmeta dump site clears IN PLACE
        # (.clear() on the shared list), never re-binds a fresh list
        import vllm_ascend.attention.int2.ring as ring_mod

        ctx._R2XX_G.pop("_R218_DISC", None)
        try:
            shared = ring_mod._R2XX_G.setdefault("_R218_DISC", [])
            shared.append("x")
            ctx._R2XX_G["_R218_DISC"].clear()  # cleaner side
            assert shared == [] and ctx._R2XX_G["_R218_DISC"] is shared
        finally:
            ctx._R2XX_G.pop("_R218_DISC", None)

    def test_diagnostics_off_leaves_registry_untouched(self, monkeypatch):
        # audit file absent: the bail still returns False (legacy fallback)
        # but no _R218_DISC key is created; the XAS reader creates no key
        _AUDIT.unlink(missing_ok=True)
        before = set(ctx._R2XX_G)
        assert self._batched_ring_bails(monkeypatch) is False
        s = _RingSelf({}, {})
        assert s._r198_win_idx("r0", 3, 10, 4, 2, 8, "cpu") is None
        assert set(ctx._R2XX_G) == before and "_R218_DISC" not in ctx._R2XX_G
        assert "_R218_XAS" not in ctx._R2XX_G
