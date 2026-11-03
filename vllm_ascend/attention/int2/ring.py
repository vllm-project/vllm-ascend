# SPDX-License-Identifier: Apache-2.0
"""Prefix/recent ring mixin for the INT2 backend (window indexing, batched ring gather, restage)."""

from __future__ import annotations

import os as _r144_ovl_os
from typing import Any

import torch
import torch_npu  # noqa: F401 - registers torch.npu

from vllm_ascend.attention.int2.ctx import (  # noqa: F401
    _OSCAR_CFG,
    _R2XX_G,
)
from vllm_ascend.attention.int2.helpers import (  # noqa: F401
    _r166_tr,
)


class Int2RingMixin:
    # Host-state annotations only: the _TT_* slot/HP state and the cache
    # interfaces (key_cache/value_cache, scale caches) are owned by the
    # backend assembly / sibling mixins (_ensure_int2_scale_buffers lives on
    # the quant-cache mixin, _oscar_layer_ord on the slot-state mixin).
    # Bare annotations create no runtime attributes; they pin types for
    # mypy --check-untyped-defs.
    _TT_RECENT: int
    _tt_mseq: int
    _tt_mhp: int
    _tt_pref: dict[Any, Any]
    _tt_rcnt: dict[Any, Any]
    _tt_rcnt_seen: dict[Any, Any]
    _tt_ring_org: dict[Any, Any]
    # One-shot read-probe latch (None until first _r198_win_idx call).
    _r218_xaudit_on: bool | None
    _tt_rid: Any
    _tt_slot: Any
    _tt_hp_bufs: Any
    _r198_idx_cache: Any
    _r203_ring_idx: Any
    _r155_cu_mod: Any
    _r156_diag_cu: Any
    _ensure_int2_scale_buffers: Any
    _oscar_layer_ord: Any
    key_cache: Any
    value_cache: Any
    k_scale_cache: Any
    k_zero_cache: Any
    v_scale_cache: Any
    v_zero_cache: Any
    vllm_config: Any

    @staticmethod
    def _r198_vring_on():
        return _r144_ovl_os.environ.get("R198_VRING", "0") == "1"

    def _r198_pos(self, rid, t, pref=None):
        """Physical ring position of token t (0-based) for this request.
        Returns None when the row has no VRING origin (caller: legacy)."""
        X = self._tt_ring_org.get(rid)
        if X is None:
            return None
        if pref is None:
            pref = self._tt_pref.get(rid, X)
        cap = self._TT_RECENT
        return pref + ((int(t) - X) % cap)

    def _r198_win_idx(self, rid, slot, sl, rcnt, nkv, hd, device):
        """[npr=pref+rcnt] flat index tensor over the row's HP buffer for the
        full S/R window (prefix identity + rotated ring). Cached per
        (slot, sl, pref, X, rcnt) — one build per row per step, shared by all
        36 layers. Returns None when the row lacks a VRING origin (legacy
        contiguous window is then correct by construction)."""
        X = self._tt_ring_org.get(rid)
        # R218 read-side key probe: WHAT KEY AND X THE WINDOW READ ACTUALLY USES
        # (slot == row index under the R164 slot(b)==b contract, so the caller's
        # rid can be compared against the row it claims to be). The marker is
        # latched once per process -- this runs per row per step.
        _xaudit = getattr(type(self), "_r218_xaudit_on", None)
        if _xaudit is None:
            _xaudit = type(self)._r218_xaudit_on = _r144_ovl_os.path.exists("/tmp/r218_xaudit_on")
        if _xaudit and int(slot) >= 24 and _R2XX_G.get("_R218_XAS", {}).get("n", 0) <= 60:
            try:
                import json as _jr

                with open("/tmp/r218_rd.jsonl", "a") as _fr:
                    _fr.write(
                        _jr.dumps(
                            {
                                "st": _R2XX_G.get("_R218_XAS", {}).get("n", -1),
                                "fn": "win_idx",
                                "slot": int(slot),
                                "rid": repr(rid),
                                "sl": int(sl),
                                "rcnt": int(rcnt),
                                "X": (None if X is None else int(X)),
                                "pref": int(self._tt_pref.get(rid, -1)) if X is not None else None,
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
        if X is None:
            return None
        pref = self._tt_pref.get(rid, X)
        key = (int(slot), int(sl), int(pref), int(X), int(rcnt), str(device), int(nkv), int(hd))
        _c = type(self)._r198_idx_cache
        hit = _c.get(key)
        if hit is not None:
            return hit
        cap = self._TT_RECENT
        mhp = int(self._tt_mhp)
        idx = [slot * mhp + j for j in range(pref)]
        t0 = int(sl) - int(rcnt)
        for j in range(int(rcnt)):
            idx.append(slot * mhp + pref + ((t0 + j - X) % cap))
        t = torch.tensor(idx, dtype=torch.int64, device=device)
        if len(_c) > 4096:
            _c.clear()
        _c[key] = t
        return t

    def _r198_win_ridx(self, rid, slot, sl, rcnt, hk):
        """[rcnt] flat ring-only index tensor for the recent window (no
        prefix), over hk.view(-1, nkv*hd). None when no VRING origin."""
        X = self._tt_ring_org.get(rid)
        if X is None:
            return None
        pref = self._tt_pref.get(rid, X)
        cap = self._TT_RECENT
        mhp = int(self._tt_mhp)
        t0 = int(sl) - int(rcnt)
        idx = [slot * mhp + pref + ((t0 + j - X) % cap) for j in range(int(rcnt))]
        return torch.tensor(idx, dtype=torch.int64, device=hk.device)

    def _r198_pr_window(self, rid, slot, sl, pref, rcnt, hk, hv):
        """(pr_k, pr_v) for the full S/R window [0, pref+rcnt) of one row.
        VRING: one cached flat gather (prefix identity + rotated ring),
        layout-identical to the legacy cat. Legacy: the verbatim cat.
        Returns None,None when n_pr == 0."""
        n_pr = int(pref) + int(rcnt)
        if n_pr <= 0:
            return None, None
        if _r144_ovl_os.environ.get("R198_VRING", "0") == "1" and self._tt_ring_org.get(rid) is not None:
            _wi = self._r198_win_idx(rid, slot, sl, rcnt, int(hk.shape[2]), int(hk.shape[3]), hk.device)
            _nf = int(hk.shape[2]) * int(hk.shape[3])
            return (
                hk.view(-1, _nf).index_select(0, _wi).view(n_pr, hk.shape[2], hk.shape[3]),
                hv.view(-1, _nf).index_select(0, _wi).view(n_pr, hv.shape[2], hv.shape[3]),
            )
        return (
            torch.cat([hk[slot, :pref], hk[slot, pref : pref + rcnt]], dim=0),
            torch.cat([hv[slot, :pref], hv[slot, pref : pref + rcnt]], dim=0),
        )

    def _r197_batched_ring(self, layer, query, key_rot, value_rot, seq_lens, batch_size):
        """R197 Fix-1 (R197_EAGER_MIX=1, default OFF): batched HP-recent ring
        maintenance for the EAGER three-tier middle path. Per-row semantics
        IDENTICAL to the legacy loop it replaces (append below cap: write the
        new token at ring[slot, pref+rcnt]; at cap: slide the window left by
        one and write the newest at its tail; R147_RCNT_FIX counter discipline
        verbatim), but with O(1) dispatched ops instead of ~12 per row:
          appends -> ONE flat index_copy_ per tensor over all rows
          slides  -> ONE index_select (reads the cap-1 window into fresh
                     storage, so the subsequent index_copy_ has NO overlap
                     hazard) + ONE index_copy_ for the shift + ONE for the
                     tails, per tensor.
        Compute-stream index kernels throughout (same stream-ordering class as
        the R166_STREAMFIX=ring mul-out idiom). Device traffic is ~half the
        legacy cat+mul-out sequence (1 read + 1 write of the window vs 2+2).
        Returns True when handled; any contract mismatch returns False BEFORE
        any ring write (caller runs the legacy loop)."""
        import os as _os

        if key_rot.shape[0] < batch_size or value_rot.shape[0] < batch_size:
            return False
        # R198 VRING fast path: ALL rows (growing or full) append at their
        # rotating position pos(sl-1); the slide lists vanish. Rows without
        # a VRING origin bail to the legacy classification below (returns
        # False BEFORE any write when mixed with vring rows? — no: handle
        # vring rows here and leave non-vring rows untouched is UNSAFE (the
        # legacy fallback re-runs every row). Bail entirely instead.
        if _os.environ.get("R198_VRING", "0") == "1":
            _vr_pos = []
            _vr_b = []
            for b in range(batch_size):
                rid = self._tt_rid(b)
                slot = int(self._tt_slot(b))
                sl = int(seq_lens[b])
                p = self._r198_pos(rid, sl - 1)
                if p is None or not (0 <= slot < int(self._tt_mseq)) or not (0 <= p < int(self._tt_mhp)):
                    # R218 disc probe: ANY row without a VRING origin makes the
                    # caller re-run EVERY row through the legacy left-packed
                    # slide, while the rows that DO have X keep reading through
                    # the frozen-X rotating map -> layout desync.
                    try:
                        if _os.path.exists("/tmp/r218_xaudit_on"):
                            _R2XX_G.setdefault("_R218_DISC", []).append(f"ring_BAIL b={b} rid={rid} sl={sl} p={p}")
                    except Exception:
                        pass
                    return False
                _vr_b.append(b)
                _vr_pos.append(slot * int(self._tt_mhp) + p)
            hk, hv = self._tt_hp_bufs(self._oscar_layer_ord(layer), query.device, query.dtype)
            if hk.dtype != key_rot.dtype:
                return False
            nkv = int(hk.shape[2])
            hd = int(hk.shape[3])
            hkf = hk.view(-1, nkv, hd)
            hvf = hv.view(-1, nkv, hd)
            # R203_METAFIX: _vr_pos/_vr_b are LAYER-INVARIANT within a step
            # (same rows, same sl; only the per-layer DATA differs) but the
            # torch.tensor(list, device) build is a blocking pageable H2D —
            # measured ~1.5ms/layer x 36 layers per mixed step (r203d census,
            # the dominant surviving to+copy_ chain after the idfill fix).
            # Single-slot class cache keyed on the full content tuple +
            # device: layer 1 builds, layers 2..36 reuse; a different step's
            # positions miss and rebuild. METAFIX=0 keeps the exact original
            # build + list-advanced-index form.
            if _os.environ.get("R203_METAFIX", "0") == "1":
                _ck = (tuple(_vr_pos), tuple(_vr_b), str(query.device))
                _cs = getattr(type(self), "_r203_ring_idx", None)
                if _cs is not None and _cs[0] == _ck:
                    _pi, _ib = _cs[1], _cs[2]
                else:
                    _pi = torch.tensor(_vr_pos, dtype=torch.int64, device=query.device)
                    _ib = torch.tensor(_vr_b, dtype=torch.int64, device=query.device)
                    type(self)._r203_ring_idx = (_ck, _pi, _ib)
                hkf.index_copy_(0, _pi, key_rot.index_select(0, _ib))
                hvf.index_copy_(0, _pi, value_rot.index_select(0, _ib))
            else:
                _pi = torch.tensor(_vr_pos, dtype=torch.int64, device=query.device)
                hkf.index_copy_(0, _pi, key_rot[_vr_b])
                hvf.index_copy_(0, _pi, value_rot[_vr_b])
            # host counters — legacy discipline, verbatim (rcnt grows to cap
            # then pins; used only for tier-boundary math, never positions)
            for b in _vr_b:
                rid = self._tt_rid(b)
                rcnt = self._tt_rcnt.get(rid, 0)
                cap = self._TT_RECENT
                if rcnt < cap:
                    if _os.environ.get("R147_RCNT_FIX", "0") == "1":
                        sl = int(seq_lens[b])
                        if self._tt_rcnt_seen.get(rid) != sl:
                            self._tt_rcnt_seen[rid] = sl
                            self._tt_rcnt[rid] = rcnt + 1
                    else:
                        self._tt_rcnt[rid] = rcnt + 1
                else:
                    self._tt_rcnt[rid] = cap
            return True
        lord = self._oscar_layer_ord(layer)
        dev = query.device
        hk, hv = self._tt_hp_bufs(lord, dev, query.dtype)
        if hk.dtype != key_rot.dtype:
            return False
        mh = int(hk.shape[1])
        nkv = int(hk.shape[2])
        hd = int(hk.shape[3])
        if key_rot.dim() != 3 or key_rot.shape[1] != nkv or key_rot.shape[2] != hd:
            return False
        cap = self._TT_RECENT
        rcf = _os.environ.get("R147_RCNT_FIX", "0") == "1"
        app_b, app_idx = [], []
        sld_b, sld_slot, sld_pref = [], [], []
        for b in range(batch_size):
            sl = int(seq_lens[b])
            rid = self._tt_rid(b)
            slot = int(self._tt_slot(b))
            pref = self._tt_pref.get(rid, 0)
            rcnt = self._tt_rcnt.get(rid, 0)
            if not (0 <= slot < int(hk.shape[0])):
                return False
            if pref < 0 or pref + cap > mh or pref + rcnt >= mh:
                return False
            if rcnt < cap:
                app_b.append(b)
                app_idx.append(slot * mh + pref + rcnt)
            else:
                sld_b.append(b)
                sld_slot.append(slot)
                sld_pref.append(pref)
        hkf = hk.view(-1, nkv, hd)
        hvf = hv.view(-1, nkv, hd)
        # all allocations/materializations FIRST — after the first in-place
        # write below, an exception must NOT fall back to the legacy loop
        # (a re-run would slide already-shifted rows a second time).
        _ai = None
        _akv = _avv = None
        if app_b:
            _ai = torch.tensor(app_idx, dtype=torch.int64, device=dev)
            _akv = key_rot[app_b]
            _avv = value_rot[app_b]
        _srcf = _dstf = _tail = _stgk = _stgv = _skv = _svv = None
        if sld_b:
            _sl = torch.tensor(sld_slot, dtype=torch.int64, device=dev)
            _pf = torch.tensor(sld_pref, dtype=torch.int64, device=dev)
            _ar = torch.arange(cap - 1, dtype=torch.int64, device=dev)
            _base = _sl * mh
            _srcf = ((_base + _pf + 1).view(-1, 1) + _ar.view(1, -1)).view(-1)
            _dstf = ((_base + _pf).view(-1, 1) + _ar.view(1, -1)).view(-1)
            _tail = _base + _pf + (cap - 1)
            _stgk = hkf.index_select(0, _srcf)
            _stgv = hvf.index_select(0, _srcf)
            _skv = key_rot[sld_b]
            _svv = value_rot[sld_b]
        # writes (allocation-free)
        if _ai is not None:
            hkf.index_copy_(0, _ai, _akv)
            hvf.index_copy_(0, _ai, _avv)
        if _srcf is not None:
            hkf.index_copy_(0, _dstf, _stgk)
            hkf.index_copy_(0, _tail, _skv)
            hvf.index_copy_(0, _dstf, _stgv)
            hvf.index_copy_(0, _tail, _svv)
        # host counters — legacy discipline, verbatim
        for b in app_b:
            sl = int(seq_lens[b])
            rid = self._tt_rid(b)
            rcnt = self._tt_rcnt.get(rid, 0)
            if rcf:
                if self._tt_rcnt_seen.get(rid) != sl:
                    self._tt_rcnt_seen[rid] = sl
                    self._tt_rcnt[rid] = rcnt + 1
            else:
                self._tt_rcnt[rid] = rcnt + 1
        for b in sld_b:
            self._tt_rcnt[self._tt_rid(b)] = cap
        return True

    def _r198_restage(impl, Bcap, _me):
        """R198 ISTAGE: eagerly re-stage ALL layers' mid carriers for this
        bucket on the CALLER's current stream (the update stream inside
        _r164_update_meta). Byte-identical gather sequence to the capture
        region's staging (which R198_ISTAGE removed from the graph). Reads
        the CURRENT bt_mirror. Returns #layers staged."""
        _cp = getattr(type(impl), "_r113_cpool_batched", None) or {}
        _idm = _me["bt_mirror_flat"]
        _n = 0
        for (_lidK, _bcapK), _c in _cp.items():
            if _bcapK != Bcap or _c.get("r198_src") is None:
                continue
            (kc, vc, ksc, kzc, vsc, vzc) = _c["r198_src"]
            Bc, S_pad, nkv, _bsz, _nbm = _c["r198_geo"]
            k_r91, v_r91 = _c["k"], _c["v"]
            ks, ko, vs, vo = _c["ks"], _c["ko"], _c["vs"], _c["vo"]
            _gk = kc.index_select(0, _idm)
            k_r91.copy_(_gk.permute(0, 2, 1, 3))
            _gv = vc.index_select(0, _idm)
            v_r91.copy_(_gv.permute(0, 2, 1, 3))
            if (
                _r144_ovl_os.environ.get("R198_DBG", "0") == "1"
                and _n == 0
                and globals().setdefault("_R198_RS_CHK", [0])[0] < 3
            ):
                globals()["_R198_RS_CHK"][0] += 1
                try:
                    torch.npu.synchronize()
                    _eqRS = bool(torch.equal(_gk.permute(0, 2, 1, 3).view(torch.uint8), k_r91.view(torch.uint8)))
                    _srcS = _gk.permute(0, 2, 1, 3)[:2, 0, 0, :8].to(torch.int32).flatten().tolist()
                    _dstS = k_r91[:2, 0, 0, :8].to(torch.int32).flatten().tolist()
                    _eqCS = bool(torch.equal(_gk.permute(0, 2, 1, 3).contiguous(), k_r91))
                    import json as _jRS

                    with open("/tmp/r198_dbg.jsonl", "a") as _fRS:
                        _fRS.write(
                            _jRS.dumps(
                                {
                                    "rs_chk": 1,
                                    "lid": _lidK,
                                    "bk": _bcapK,
                                    "eq_after_copy": _eqRS,
                                    "eq_contig": _eqCS,
                                    "src_bytes": _srcS,
                                    "dst_bytes": _dstS,
                                    "k_ptr": k_r91.data_ptr(),
                                    "kc_ptr": kc.data_ptr(),
                                    "idm_head": [int(x) for x in _idm[:4].tolist()],
                                    "idm_len": int(_idm.numel()),
                                    "k_shape": list(k_r91.shape),
                                }
                            )
                            + "\n"
                        )
                except Exception as _eRS:
                    try:
                        with open("/tmp/r198_dbg.jsonl", "a") as _fRS:
                            _fRS.write(f"RS_ERR {_eRS!r}\n")
                    except Exception:
                        pass

            def _sc(_src_t, _idm=_idm, _Bc=Bc, _S=S_pad, _nkv=nkv):
                return _src_t.index_select(0, _idm).view(_Bc, _S, _nkv).transpose(1, 2)

            ks.copy_(_sc(ksc))
            ko.copy_(-_sc(kzc) * ks)
            vs.copy_(_sc(vsc))
            vo.copy_(-_sc(vzc) * vs)
            _n += 1
        return _n

    def _r155_cu_ring_row(self, md, lid, b, mid_start, rcnt, sl):
        """R167_CUSYNC/CUDEF (default OFF): this CU reads RING rows and writes
        PAGED blocks on the UPDATE stream while the graph replay concurrently
        touches both (in-graph slide rewrites rings; in-graph mid staging
        reads paged). Probe-proven at ctx3072: NOCU=8/8 vs CU=families; PFEV
        does not cover it (the replay executes on an internal stream so a
        default-stream record fires early). CUSYNC = full device sync before
        the quantize (read-side); CUDEF = run the CU's ops on the DEFAULT
        stream (covered by the pre-replay current_stream sync for the
        write-side too). CU fires only at 64-token boundaries per request.
        R155 A2-lite v2 (R155_CUEAGER): LAZY eviction-batched quantize.
        A decode token's paged INT2 row is first READ by the middle tier only
        when it leaves the recent window (rcnt steps after it was produced).
        The eager per-step CU therefore wastes ~0.62ms/layer in-graph AND on
        the update-stream event chain. Here: every 64 steps (aligned on the
        eviction boundary), batch-quantize the NEXT 64 ring rows that are
        decode tokens (prefill rows were already quantized at prefill time).
        Same CU kernel, same input bytes (ring rows store key_rot/value_rot
        verbatim), same slots => E-class; rows land >=1 step pre-eviction.
        Canonical ctx1047+NT128 NEVER triggers (all rows prefill-written)."""
        import os as _os

        # R167_CUSYNC: full sync before the quantize (CU ring-read vs the
        # in-flight replay slide). R167_CUDEF: run the CU ops on the DEFAULT
        # stream (write-side covered by the next pre-replay sync).
        _cudef_on = _os.environ.get("R167_CUDEF", "0") == "1"
        if _os.environ.get("R167_CUSYNC", "0") == "1" or _cudef_on:
            _np167cs = torch.npu  # torch_npu at module top registers the submodule

            _np167cs.synchronize()
        pref = _OSCAR_CFG.prefix_window
        t0 = int(sl) - int(rcnt)  # newest middle-visible token idx
        _lc_n = globals().setdefault("_R166_LC_N", [0])
        if (t0 >= 0 and (t0 % 64) == 0) or _lc_n[0] < 3:
            _lc_n[0] += 1
            _r166_tr("lazy_cu_gate", t0=int(t0), sl=int(sl), rcnt=int(rcnt), lid=int(lid), b=int(b))
        if t0 < 0 or (t0 % 64) != 0:
            return
        init_ctx = int(getattr(md, "_r155_init_ctx", -1))
        if init_ctx < 0:
            # first call: remember the prompt length (all tokens < this were
            # quantized by the prefill path already; sl counts 1 decode tok)
            init_ctx = int(sl) - 1
            md._r155_init_ctx = init_ctx
        _cup168 = _os.environ.get("R168_CUPROBE", "0") == "1"
        if _cup168:
            try:
                _cnt168 = globals().setdefault("_R168_CU_N", [0])
                if _cnt168[0] < 4000:
                    _cnt168[0] += 1
                    import json as _j168

                    with open("/tmp/r168_cu.jsonl", "a") as _f168:
                        _f168.write(
                            _j168.dumps(
                                {
                                    "n": _cnt168[0],
                                    "lid": int(lid),
                                    "b": int(b),
                                    "sl": int(sl),
                                    "rcnt": int(rcnt),
                                    "ms": int(mid_start),
                                    "t0": int(t0),
                                    "g64": int(t0 % 64 == 0),
                                    "init": int(init_ctx),
                                    "mdid": id(md) % 1000003,
                                    "fresh": int(getattr(md, "_r155_init_ctx", -1) < 0),
                                }
                            )
                            + "\n"
                        )
            except Exception:
                pass
        if t0 + 64 <= init_ctx:
            return  # next 64 evictees are all prefill
        bt_live = md.block_tables
        bs = int(self.vllm_config.cache_config.block_size)
        hk, hv = self._tt_hp_bufs(lid, self.key_cache.device, torch.bfloat16)
        _slot_r = self._tt_slot(b)
        t_lo = max(t0, init_ctx)
        t_hi = min(t0 + 64, int(sl))
        if t_hi <= t_lo:
            return
        toks = torch.arange(t_lo, t_hi, device=bt_live.device)
        slots_t = (bt_live[b].to(torch.long).index_select(0, toks // bs) * bs + (toks % bs)).contiguous()
        # ring pos pref+j holds token (sl-rcnt)+j = t0+j
        j_lo = t_lo - t0
        j_hi = t_hi - t0
        kc, vc = self.key_cache, self.value_cache
        if getattr(self, "k_scale_cache", None) is None or self.k_scale_cache.shape[0] != kc.shape[0]:
            self.k_scale_cache = None
            self._ensure_int2_scale_buffers(kc, kc.shape[1], kc.shape[2], torch.bfloat16, kc.device)
        mod = getattr(type(self), "_r155_cu_mod", None)
        if mod is None:
            from vllm_ascend.oscar import r116_torch_op as _r116op

            mod = _r116op.preload(kc.device)
            type(self)._r155_cu_mod = mod
        _ck168 = None
        if _cup168:
            try:
                _ck168 = (
                    int(slots_t.sum().item()),
                    int(t_lo),
                    int(t_hi),
                    int(hk[_slot_r, pref + j_lo : pref + j_hi].sum().item()),
                )
            except Exception:
                _ck168 = None
        if _cudef_on:
            import contextlib as _cl167cd

            _np167cd = torch.npu  # torch_npu at module top registers the submodule

            _cu_ctx = torch.npu.stream(_np167cd.default_stream())
        else:
            import contextlib as _cl167cd

            _cu_ctx = _cl167cd.nullcontext()
        with _cu_ctx:
            if _os.environ.get("R198_VRING", "0") == "1":
                # R198 VRING: gather the evicting tokens from their rotating
                # positions (tokens [t_lo, t_hi) -> pos pref+((t-X)%cap))
                _Xcu = self._tt_ring_org.get(self._tt_rid(b), pref)
                _capcu = self._TT_RECENT
                _mhpCu = int(self._tt_mhp)
                _posCu = [_slot_r * _mhpCu + pref + ((_tCu - _Xcu) % _capcu) for _tCu in range(t_lo, t_hi)]
                _idxCu = torch.tensor(_posCu, dtype=torch.int64, device=hk.device)
                _nfCu = int(hk.shape[2]) * int(hk.shape[3])
                _ckCu = hk.view(-1, _nfCu).index_select(0, _idxCu).view(-1, hk.shape[2], hk.shape[3])
                _cvCu = hv.view(-1, _nfCu).index_select(0, _idxCu).view(-1, hv.shape[2], hv.shape[3])
            else:
                _ckCu = hk[_slot_r, pref + j_lo : pref + j_hi].contiguous()
                _cvCu = hv[_slot_r, pref + j_lo : pref + j_hi].contiguous()
            rc_cu, _ws_cu, lr_cu, _c1, _c2, _c3, _c4 = mod.cache_update_aclnn(
                _ckCu,
                _cvCu,
                slots_t,
                kc,
                vc,
                self.k_scale_cache,
                self.k_zero_cache,
                self.v_scale_cache,
                self.v_zero_cache,
            )
        if not (int(rc_cu) == 0 and int(lr_cu) == 0):
            raise RuntimeError(f"[R155-CUEAGER] batch CU rc={rc_cu} lr={lr_cu} toks=[{t_lo},{t_hi}) layer={lid}")
        if _cup168:
            try:
                import json as _j168b

                with open("/tmp/r168_cu.jsonl", "a") as _f168b:
                    _f168b.write(
                        _j168b.dumps(
                            {
                                "WRITE": 1,
                                "lid": int(lid),
                                "b": int(b),
                                "sl": int(sl),
                                "tlo": int(t_lo),
                                "t_hi": int(t_hi),
                                "slot": int(_slot_r),
                                "ck": _ck168,
                                "blk": [int(x) for x in bt_live[b][(t_lo // bs) : (t_hi // bs + 1)].tolist()],
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
        if _os.environ.get("R156_DIAG", "0") == "1":
            _cu = getattr(type(self), "_r156_diag_cu", None)
            if _cu is None:
                _cu = type(self)._r156_diag_cu = [0]
            _cu[0] += 1

    def _r163_stage_sr(impl, update_stream, md, _lid, Bcap, mid_start, rcnt, nkv):
        """R163 SROFF: snapshot the batched S/R carriers from the HP ring on
        the update stream BEFORE the event the graph waits on — removes the
        graph's live HP-ring reads (the R160/R161 cross-stream torn race)."""
        _scp = getattr(type(impl), "_r163_srcarrier", None)
        if not _scp:
            return
        _ent = _scp.get((_lid, int(Bcap)))
        if _ent is None:
            return
        hk, hv = impl._tt_hp_bufs(_lid, _ent["dev"], _ent["dtype"])
        with torch.npu.stream(update_stream):
            for _b in range(int(Bcap)):
                _slot = impl._tt_slot(_b)
                # Under R198_VRING the recent window is virtual/rotated, so
                # plain contiguous slices copy rotated tokens into the S/R
                # carrier; _r198_pr_window handles both layouts (and returns
                # None only for rows without a VRING origin, where the
                # contiguous gather is correct by construction).
                _rid = impl._tt_rid(_b)
                _sl = int(md.seq_lens_list[_b]) if _b < len(md.seq_lens_list) else 0
                _pw = impl._r198_pr_window(_rid, _slot, _sl, mid_start, rcnt, hk, hv)
                if _pw is not None:
                    _ks, _vs = _pw
                else:
                    _ks = torch.cat([hk[_slot, :mid_start], hk[_slot, mid_start : mid_start + rcnt]], dim=0)
                    _vs = torch.cat([hv[_slot, :mid_start], hv[_slot, mid_start : mid_start + rcnt]], dim=0)
                _ent["k"][_b].copy_(_ks.permute(1, 0, 2))
                _ent["v"][_b].copy_(_vs.permute(1, 0, 2))
