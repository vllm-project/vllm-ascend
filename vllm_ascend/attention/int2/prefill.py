# SPDX-License-Identifier: Apache-2.0
"""Three-tier prefill execution mixin (HP prefix+recent writes, middle
quantize, attention core)."""

from __future__ import annotations

import os
import os as _r144_ovl_os
import sys
from typing import Any

import torch
import torch_npu

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.int2.ctx import (  # noqa: F401
    _R2XX_G,
    _odiag_o,
)
from vllm_ascend.attention.int2.helpers import (  # noqa: F401
    _oscar_dbg,
    _r166_tr,
    _r203_d2h_install,
    _r203_qsl_host,
    _r218_diag,
)


class Int2PrefillMixin:
    # Host-state annotations only: bare annotations create no runtime
    # attributes, so the MRO at assembly is unaffected. The five
    # type(self)-assigned probe/ latch names carry None defaults so
    # they exist before any assignment path runs.
    _r204_nlayers: Any = None
    _r204_sync2_fired: Any = None
    _r204_sync_fired: Any = None
    _r218_call_drids: Any = None
    _r218_call_rids: Any = None
    _TT_PREFIX: int
    _TT_RECENT: int
    _dequantize_packed_int2: Any
    _ensure_tt: Any
    _forward_int2_decode_threetier_middle: Any
    _oscar_R_k: Any
    _oscar_R_v: Any
    _oscar_layer_ord: Any
    _r164_bind_slots: Any
    _r164_slotmap: Any
    _r198_pos: Any
    _r198_win_ridx: Any
    _r218_st_clock: Any
    _r218_st_len: Any
    _r218_st_tick: Any
    _r218_stage_assign: Any
    _r218_stage_bufs: Any
    _r218_stage_lookup: Any
    _tt_hp_bufs: Any
    _tt_hp_k: Any
    _tt_pref: dict[str, Any]
    _tt_rcnt: dict[str, Any]
    _tt_rcnt_seen: Any
    _tt_resolve_slots: Any
    _tt_rid: Any
    _tt_ring_org: Any
    _tt_slot: Any
    _tt_write_hp: Any
    head_size: int
    k_scale_cache: Any
    k_zero_cache: Any
    key_cache: Any
    num_heads: int
    num_kv_heads: int
    scale: float
    v_scale_cache: Any
    v_zero_cache: Any
    value_cache: Any
    vllm_config: Any

    def _forward_int2_prefill_threetier(self, query, key_float, value_float, attn_metadata, output, layer):
        """Three-tier prefill: write HP prefix+recent, quantize middle to INT2,
        then run attention on the already-rotated float KV via attn core."""
        _r166_tr(
            "prefill3t",
            T=int(query.shape[0]),
            bs=int(len(getattr(attn_metadata, "seq_lens_list", []) or [])),
            st=str(getattr(attn_metadata, "attn_state", "?")),
            lid=int(self._oscar_layer_ord(layer)) if layer is not None else -1,
            rids=[str(r) for r in (getattr(attn_metadata, "request_ids", None) or [])][:16],
        )
        self._ensure_tt(query.device)
        # R144 P0: map batch-row b -> stable rid + reusable slot (gated, default OFF).
        if os.environ.get("R144_TT_REQID", "0") == "1":
            self._tt_resolve_slots(attn_metadata, self._oscar_layer_ord(layer), query.device, query.dtype)
        self._r164_bind_slots(attn_metadata, query.device, query.dtype, layer)
        # R165 FINAL FIX (R164_EAGER_WAIT, default OFF): eager steps that write
        # rings MUST wait the prior graph replay (its late copy-engine slide
        # scatter can land AFTER this step's ring writes and overwrite them
        # with shifted window content — the N-1 block-shaped corruption).
        if _r144_ovl_os.environ.get("R164_EAGER_WAIT", "0") == "1":
            # The replay-fence event, when present, is registered under the
            # shared registry key by whoever records it (graph launchers);
            # absent key => nothing to wait (same as the stash import that
            # used to live here and resolved to nothing outside the dev tree).
            try:
                _e = _R2XX_G.get("R161_E_REPLAY_PREV")
                if _e is not None:
                    _e.wait(torch.npu.current_stream())
            except Exception:
                pass

        if _r144_ovl_os.environ.get("R197_DBG", "0") == "1":  # R197 Fix-0: was unconditional per-layer-call stderr
            print(
                f"[3T-PF-DBG] ENTER num_reqs={len(attn_metadata.seq_lens_list)} "
                f"attn_state={attn_metadata.attn_state} query.shape={query.shape}",
                file=sys.stderr,
                flush=True,
            )
        q_start = None
        if _r144_ovl_os.environ.get("R203_METAFIX", "0") == "1":
            # R203: per-step mirror (see _r203_qsl_host) — one D2H per step
            # instead of one full-stream sync per layer; None => base read.
            q_start = _r203_qsl_host(attn_metadata)
        if _r144_ovl_os.environ.get("R203_D2HTRACE", "0") == "1":
            _r203_d2h_install()
        if q_start is None or not isinstance(q_start, list):
            q_start = attn_metadata.query_start_loc
            if hasattr(q_start, "cpu"):
                q_start = q_start.cpu()
            if not isinstance(q_start, list):
                q_start = q_start.tolist()
        num_reqs = len(attn_metadata.seq_lens_list)

        # Apply R_k/R_v rotation ONCE to the full float K/V
        R_k = R_v = None
        if self._oscar_R_k is not None and self._oscar_layer_ord(layer) < self._oscar_R_k.shape[0]:
            R_k = self._oscar_R_k[self._oscar_layer_ord(layer)]
            R_v = self._oscar_R_v[self._oscar_layer_ord(layer)]

        # Rotate Q
        query_rot = query
        if R_k is not None:
            query_rot = torch.matmul(query.contiguous(), R_k)
        key_rot = key_float.contiguous()
        value_rot = value_float.contiguous()
        if R_k is not None:
            key_rot = torch.matmul(key_rot, R_k)
            value_rot = torch.matmul(value_rot, R_v)

        # R144 G3 §1: detect decode-token rows mixed into a ChunkedPrefill batch.
        # vllm V1 may batch an already-decoding request's 1-token "prefill" with
        # another request's real prefill in one ChunkedPrefill batch (decode rows
        # are the first num_decodes rows). The 1-token row must NOT reinitialize
        # the three-tier state (it would clobber pref 128->1 and zero the HP prefix
        # of the real prefill). Instead, append the new token to HP recent (decode
        # semantics) and leave _tt_pref unchanged. R109_TT_PREFIX=0 "worked" only
        # because pref=0 is always block-aligned — it was a diagnostic workaround,
        # not a fix; this restores canonical TT_PREFIX=128 boundaries per request.
        _num_dec = getattr(attn_metadata, "num_decodes", 0) or 0
        _r164_mix_rows = []
        _r218_cont_rows = []  # R218 FIX2b: split-prefill continuation rows
        # R218 STAGE: rows of THIS call are protected from stage eviction —
        # a split row written by the previous call re-appears here as the
        # continuation, so its stage must survive this call's own writes.
        try:
            type(self)._r218_call_rids = {self._tt_rid(_bb) for _bb in range(num_reqs)}
            type(self)._r218_call_drids = {self._tt_rid(_bb) for _bb in range(_num_dec)}
        except Exception:
            type(self)._r218_call_rids = None
            type(self)._r218_call_drids = None
        for b in range(num_reqs):
            s0, s1 = int(q_start[b]), int(q_start[b + 1])
            sl = s1 - s0
            # R144 P0: stable request identity + reusable HP slot for this row.
            _rid = self._tt_rid(b)
            _slot = self._tt_slot(b)
            # Decode-mixed gate: keep the EXACT pre-P0 logic (b < num_decodes)
            # and (rid in _tt_pref); under REQID=OFF _rid==b so this is byte-
            # identical to the G3 baseline. P0 only re-keys the dict test from
            # b to rid (correct key under REQID=ON); it does NOT generalize the
            # gate to (rid in _tt_pref) alone -- that would change chunked-
            # prefill continuation semantics (multi-token continuation chunks
            # have rid in _tt_pref at b>=num_dec) and is a separate decision.
            _is_decode_in_mixed = (b < _num_dec) and (_rid in self._tt_pref)
            # R218 STAGE (see _r218_stage_assign): stage THIS chunk's rotated
            # BF16 K/V so a later continuation chunk of the same prompt can
            # attend BF16 history instead of the INT2 paged cache.
            if not _is_decode_in_mixed:
                try:
                    _totS = int(attn_metadata.seq_lens_list[b])
                    _prevS = _totS - sl
                    _prefS = int(self._tt_pref.get(_rid, 0))
                    _siS = self._r218_stage_assign(
                        _slot,
                        _rid,
                        protect=getattr(type(self), "_r218_call_rids", None),
                        dead=getattr(type(self), "_r218_call_drids", None),
                    )
                    if os.environ.get("R144_PF_PROBE", "0") == "2":
                        try:
                            with _odiag_o(
                                "r218_stage{}.log".format(_r144_ovl_os.environ.get("R218_STAGE_TAG", ""))
                            ) as _fst:
                                _fst.write(
                                    "W L={} b={} rid={} slot={} sl={} tot={} pref={} si={}\n".format(
                                        self._oscar_layer_ord(layer), b, _rid, _slot, sl, _totS, _prefS, _siS
                                    )
                                )
                        except Exception:
                            pass
                    # Store only the MIDDLE span [pref, tot) at stage offset
                    # (pos - pref): the prefix and the recent window are already
                    # BF16 in the HP ring, so the continuation reads them from
                    # there. The FIRST chunk of a prompt covers [0, sl) and
                    # therefore carries the head of the middle too -- clip the
                    # source by the prefix overlap instead of skipping the row.
                    _i0S = _prevS - _prefS
                    _j0S = max(0, _i0S)  # destination offset
                    _srS = _j0S - _i0S  # source offset (prefix part)
                    _nS = sl - _srS
                    if _siS is not None and _nS > 0:
                        _skS, _svS, _capS = self._r218_stage_bufs(
                            self._oscar_layer_ord(layer), query.device, query.dtype
                        )
                        # A first chunk is usually LONGER than the middle the
                        # continuation will ever read (mid_len = tot-pref-rcnt
                        # <= cap). Clip instead of skipping: only the first
                        # `cap` middle positions are ever needed.
                        if _j0S >= _capS:
                            _nS = 0
                        elif _j0S + _nS > _capS:
                            _nS = _capS - _j0S
                        if _nS > 0 and _j0S + _nS <= _capS:
                            _skS[_siS, _j0S : _j0S + _nS].copy_(key_rot[s0 + _srS : s0 + _srS + _nS])
                            _svS[_siS, _j0S : _j0S + _nS].copy_(value_rot[s0 + _srS : s0 + _srS + _nS])
                            self._r218_st_len[_siS] = _j0S + _nS
                            self._r218_st_tick[0] += 1
                            self._r218_st_clock[_siS] = self._r218_st_tick[0]
                except Exception:
                    pass
            if os.environ.get("R144_PF_PROBE", "0") == "2":  # R218 FIX2 marker (whitelisted channel)
                try:
                    with open("/tmp/r218_fix2m.log", "a") as _fm:
                        _fm.write(
                            "M b={} sl={} rid={} slot={} pref={} rcnt={} sll={} num_dec={} dm={} st={}\n".format(
                                b,
                                sl,
                                _rid,
                                _slot,
                                self._tt_pref.get(_rid),
                                self._tt_rcnt.get(_rid),
                                attn_metadata.seq_lens_list[b] if b < len(attn_metadata.seq_lens_list) else -9,
                                _num_dec,
                                _is_decode_in_mixed,
                                attn_metadata.attn_state,
                            )
                        )
                except Exception:
                    pass
            # R144 DIAG (file): log branch decision
            # R197 Fix-0: the R144 per-row prefill probe wrote an UNCONDITIONAL
            # open+write+close per row per layer (335MB / 2.37M lines per leg,
            # ~31us/line host) on the mixed/prefill critical path. Now gated
            # behind R144_PF_PROBE (default OFF).
            if os.environ.get("R144_PF_PROBE", "0") == "1":
                try:
                    with open("/tmp/r144_prefill_probe.log", "a") as _pf:
                        _pf.write(
                            f"[R144-PF] L={getattr(layer, 'layer_idx', 0)} b={b} rid={_rid} slot={_slot} sl={sl} "
                            f"num_dec={_num_dec} decode_mixed={_is_decode_in_mixed} "
                            f"cur_pref={self._tt_pref.get(_rid, -1)} cur_rcnt={self._tt_rcnt.get(_rid, -1)} "
                            f"state={attn_metadata.attn_state}\n"
                        )
                except Exception:
                    pass
            # ---------------------------------------------------------------
            # R218 FIX2 -- CHUNKED-PREFILL CONTINUATION (env R218_CHUNKAPP,
            # default 1). A request whose PREFILL is split across scheduler
            # steps re-enters here with sl = the CHUNK length (s1-s0) while it
            # already owns three-tier state. The "genuine prefill" path below
            # recomputes pref/rcnt FROM THE CHUNK (pref=min(128,sl),
            # rcnt=min(896,sl-pref)), zeroes the slot (R137_HP_ZERO) and writes
            # only the chunk -- destroying the prefix+recent window written by
            # the earlier chunk. Device-proven at ctx1024/NS=32: slot16
            # rid=832822b1 is zeroed at sl=1015 AND again at sl=9, and the row
            # then decodes with pref=9/rcnt=0 (mid_len=0) -- attending 9 of
            # 1024 tokens, so its FIRST sampled token already differs.
            # A continuation chunk must APPEND at the sequence position, using
            # the same window rule as the decode append (:8028-8041) and the
            # same VRING positions as the read side (_r198_win_idx). Fresh
            # requests (prev == 0) and slot reuse by another request (prev == 0,
            # new rid) still take the init path, which must keep zeroing.
            # R218 FIX2/FIX2b on/off switch. R218_CHUNKAPP is NOT forwarded by the
            # serve launcher (fixed -e whitelist), so the A/B control rides the
            # whitelisted R144_PF_PROBE channel: value 9 = fix DISABLED.
            _capp = os.environ.get("R218_CHUNKAPP", "1")
            if os.environ.get("R144_PF_PROBE", "0") == "9":
                _capp = "0"
            if _capp != "0" and (not _is_decode_in_mixed) and (_rid in self._tt_pref):
                _tot = int(attn_metadata.seq_lens_list[b])
                _prev_sll = _tot - sl
                _prev_own = int(self._tt_pref.get(_rid, 0))
                if os.environ.get("R144_PF_PROBE", "0") == "2":  # R218 FIX2 detail log
                    try:
                        with _odiag_o("r218_fix2.log") as _f2:
                            _f2.write(
                                "L={} b={} rid={} slot={} sl={} sll={} prev_sll={} prev_own={} dm={} "
                                "state={} cpref={} crcnt={} nblk={}\n".format(
                                    getattr(layer, "layer_idx", 0),
                                    b,
                                    _rid,
                                    _slot,
                                    sl,
                                    _tot,
                                    _prev_sll,
                                    _prev_own,
                                    _is_decode_in_mixed,
                                    attn_metadata.attn_state,
                                    self._tt_pref.get(_rid),
                                    self._tt_rcnt.get(_rid),
                                    len(attn_metadata.block_tables) if hasattr(attn_metadata, "block_tables") else -1,
                                )
                            )
                    except Exception:
                        pass
                _prev = _prev_sll if _capp != "2" else max(_prev_sll, _prev_own)
                if _prev > 0:
                    _pref2 = int(self._tt_pref.get(_rid, 0))
                    _cap2 = int(self._TT_RECENT)
                    _lid2 = self._oscar_layer_ord(layer)
                    _nk2 = key_rot[s0 : s0 + sl]
                    _nv2 = value_rot[s0 : s0 + sl]
                    _p0 = (
                        self._r198_pos(_rid, _prev, _pref2)
                        if _r144_ovl_os.environ.get("R198_VRING", "0") == "1"
                        else None
                    )
                    try:
                        _xa1 = globals().setdefault("_R218_XA_ON", [None])[0]
                        if _xa1 is None:
                            _xa1 = globals()["_R218_XA_ON"][0] = _r144_ovl_os.path.exists("/tmp/r218_xaudit_on")
                        if _xa1:
                            _R2XX_G.setdefault("_R218_DISC", []).append(
                                "cont_{} b={} sl={} prev={}".format("legacy" if _p0 is None else "vring", b, sl, _prev)
                            )
                    except Exception:
                        pass
                    if _p0 is None:
                        # legacy contiguous ring: append at pref+rcnt; on overflow
                        # keep the last cap tokens (same rule as the decode append)
                        _r2 = int(self._tt_rcnt.get(_rid, 0))
                        if _r2 + sl <= _cap2:
                            self._tt_write_hp(_lid2, _slot, _pref2 + _r2, (_nk2, _nv2))
                        elif sl >= _cap2:
                            self._tt_write_hp(_lid2, _slot, _pref2, (_nk2[sl - _cap2 :], _nv2[sl - _cap2 :]))
                        else:
                            _m2 = _cap2 - sl
                            _hk2, _hv2 = self._tt_hp_bufs(_lid2, query.device, query.dtype)
                            _hk2[_slot, _pref2 : _pref2 + _cap2] = torch.cat(
                                [_hk2[_slot, _pref2 + _r2 - _m2 : _pref2 + _r2], _nk2], dim=0
                            )
                            _hv2[_slot, _pref2 : _pref2 + _cap2] = torch.cat(
                                [_hv2[_slot, _pref2 + _r2 - _m2 : _pref2 + _r2], _nv2], dim=0
                            )
                    else:
                        # R198 VRING: rotated positions; split when the run wraps.
                        # A chunk LONGER than the ring can never be stored whole
                        # -- a ring keeps only its last cap tokens -- so clamp the
                        # run to [tot-cap, tot) (one exact lap) before writing.
                        # The previous version wrote the raw remainder and
                        # overflowed the slot: "1536 and 896 cannot broadcast"
                        # (aclnnInplaceCopy 161002) whenever a continuation chunk
                        # exceeded the space left to the ring end, i.e. for every
                        # ctx whose prompt length does not divide the step budget.
                        _hi2 = _pref2 + _cap2
                        if sl >= _cap2:
                            _p0v = self._r198_pos(_rid, _tot - _cap2, _pref2)
                            _run2 = _cap2
                            _nk2v = _nk2[sl - _cap2 :]
                            _nv2v = _nv2[sl - _cap2 :]
                        else:
                            _p0v = _p0
                            _run2 = sl
                            _nk2v = _nk2
                            _nv2v = _nv2
                        if _p0v + _run2 <= _hi2:
                            self._tt_write_hp(_lid2, _slot, _p0v, (_nk2v, _nv2v))
                        else:
                            _n12 = _hi2 - _p0v
                            self._tt_write_hp(_lid2, _slot, _p0v, (_nk2v[:_n12], _nv2v[:_n12]))
                            self._tt_write_hp(_lid2, _slot, _pref2, (_nk2v[_n12:], _nv2v[_n12:]))
                    self._tt_rcnt[_rid] = min(_cap2, _tot - _pref2)
                    self._tt_rcnt_seen[_rid] = _tot
                    # FIX2b: the core below attends a row against its OWN chunk
                    # K/V only, so this row's history would be lost. Record it
                    # and recompute its output after the core (same overwrite
                    # pattern the MIXDEC rows use).
                    _r218_cont_rows.append((b, s0, s0 + sl))
                    continue
            if _is_decode_in_mixed:
                # R164 FIX (env R164_MIXDEC, default ON): a decode row mixed
                # into a ChunkedPrefill batch MUST get DECODE semantics — its
                # output has to attend the full three-tier history. The old
                # path only appended the ring token and then let
                # _attn_core_prefill attend the 1-token chunk KV -> garbage
                # logits (the historical B>1 "+1"/off-by-one family: the row
                # decodes one step early in a split batch). Now: record the
                # row; after the prefill core, run the eager three-tier
                # decode on the decode subset (which also appends the ring).
                if os.environ.get("R164_MIXDEC", "1") != "1":
                    # legacy behavior (diagnostic A/B leg)
                    _d_pref = self._tt_pref.get(_rid, 0)
                    _d_rcnt = self._tt_rcnt.get(_rid, 0)
                    _cap = self._TT_RECENT
                    _dnk = key_rot[s0 : s0 + 1]
                    _dnv = value_rot[s0 : s0 + 1]
                    _dp198 = (
                        self._r198_pos(_rid, int(sl) - 1, _d_pref) if os.environ.get("R198_VRING", "0") == "1" else None
                    )
                    try:
                        _xa2 = globals().setdefault("_R218_XA_ON", [None])[0]
                        if _xa2 is None:
                            _xa2 = globals()["_R218_XA_ON"][0] = _r144_ovl_os.path.exists("/tmp/r218_xaudit_on")
                        if _xa2:
                            _R2XX_G.setdefault("_R218_DISC", []).append(
                                "mix_{} b={} sl={}".format("vring" if _dp198 is not None else "legacy", b, sl)
                            )
                    except Exception:
                        pass
                    if _dp198 is not None:
                        # R198 VRING: rotating append, no slide
                        self._tt_write_hp(self._oscar_layer_ord(layer), _slot, _dp198, (_dnk, _dnv))
                        if _d_rcnt < _cap:
                            if os.environ.get("R147_RCNT_FIX", "0") == "1":
                                if self._tt_rcnt_seen.get(_rid) != s1:
                                    self._tt_rcnt_seen[_rid] = s1
                                    self._tt_rcnt[_rid] = _d_rcnt + 1
                            else:
                                self._tt_rcnt[_rid] = _d_rcnt + 1
                        else:
                            self._tt_rcnt[_rid] = _cap
                    elif _d_rcnt < _cap:
                        self._tt_write_hp(self._oscar_layer_ord(layer), _slot, _d_pref + _d_rcnt, (_dnk, _dnv))
                        if os.environ.get("R147_RCNT_FIX", "0") == "1":
                            if self._tt_rcnt_seen.get(_rid) != s1:
                                self._tt_rcnt_seen[_rid] = s1
                                self._tt_rcnt[_rid] = _d_rcnt + 1
                        else:
                            self._tt_rcnt[_rid] = _d_rcnt + 1
                    else:
                        _dhk, _dhv = self._tt_hp_bufs(self._oscar_layer_ord(layer), query.device, query.dtype)
                        _dnk_c = torch.cat([_dhk[_slot, _d_pref + 1 : _d_pref + _cap], _dnk], dim=0)
                        _dnv_c = torch.cat([_dhv[_slot, _d_pref + 1 : _d_pref + _cap], _dnv], dim=0)
                        if _r144_ovl_os.environ.get("R166_STREAMFIX", "0") in ("1", "ring", "both"):
                            torch.mul(_dnk_c, 1.0, out=_dhk.narrow(0, _slot, 1).narrow(1, _d_pref, _cap)[0])
                            torch.mul(_dnv_c, 1.0, out=_dhv.narrow(0, _slot, 1).narrow(1, _d_pref, _cap)[0])
                        else:
                            _dhk[_slot, _d_pref : _d_pref + _cap] = _dnk_c
                            _dhv[_slot, _d_pref : _d_pref + _cap] = _dnv_c
                    pass
                else:
                    try:
                        with open("/tmp/r164_mixdec.log", "a") as _mf:
                            _mf.write(f"MIXDEC row b={b} rid={_rid} s0={s0} sl={sl}\n")
                    except Exception:
                        pass
                    _r164_mix_rows.append((b, s0))
                continue
            # Genuine prefill (first prefill or large chunk): init three-tier state.
            pref = min(self._TT_PREFIX, sl)
            rcnt = min(self._TT_RECENT, max(0, sl - pref))
            self._tt_pref[_rid] = pref
            self._tt_rcnt[_rid] = rcnt
            # R198 VRING: record the ring origin for this (re)init. The
            # prefill writes the window left-packed at [pref, pref+rcnt);
            # X = max(pref, sl - cap) makes the rotating formula
            # pos(t) = pref + ((t - X) % cap) reproduce exactly that layout
            # for the tokens just written (growing rows: X=pref, pos=t;
            # full rows: X=sl-cap, window start maps to pref).
            if _r144_ovl_os.environ.get("R198_VRING", "0") == "1":
                self._tt_ring_org[_rid] = max(pref, sl - self._TT_RECENT)
                # R218 (2026-10-17) TESTED AND REJECTED: mirroring the origin under the row
                # index b here (on the theory that _tt_rid(b) degrades to int b because
                # _tt_cur_rids can be a subset of the batch) changed NOTHING -- mK135
                # (marker) and mK136 (clean) stayed uniq=2 with victims {29,30,31}@34 and
                # the identical sha.  The int fallback is a probe artifact: the read
                # consumers and the init share one per-instance _tt_cur_rids, so they agree
                # row-by-row.  Original comment kept below for the record.  R218_KSFIX:
                # _tt_rid(b) is NOT total.  _tt_cur_rids is set from
                # several places and can be a SUBSET of the batch (mixed/ragged
                # steps -- e.g. 28 entries for a 32-row batch), so for the rows
                # it does not cover the identity silently degrades from the
                # stable string rid to the int row index b.  Every READ consumer
                # resolves the map through _tt_rid(b) while the driver's append
                # resolves it through md.request_ids[b] (the string, which HITS),
                # so those rows read the LEGACY left-packed window while the
                # append keeps writing the ROTATING one -- the ring desyncs for
                # exactly those rows and the greedy margin flips a few steps in.
                # Mirror the origin (plus pref/rcnt, which the read consults too)
                # under the row index: the R164 contract slot(b)==b makes b a
                # stable identity, each new occupant re-inits and overwrites it,
                # and the driver's own fallback chain already tries b.
                # R218_XAUDIT (default OFF): record the (re)init that DEFINES
                # the row's ring origin. The read consumers must see exactly
                # this X for every later step of the same rid; a stale/other
                # value rotates the recent window by (X_step - X_init) % cap.
                _xai = globals().setdefault("_R218_XA_ON", [None])[0]
                if _xai is None:
                    _xai = globals()["_R218_XA_ON"][0] = _r144_ovl_os.path.exists("/tmp/r218_xaudit_on")
                if _xai:
                    # NB: env CANNOT gate this -- RAW/r216_serve_full4.sh:90 forwards an
                    # explicit -e whitelist, so a NEW R218_* name silently never reaches
                    # the container (same failure class as the bash var-prefix trap).
                    # A file marker needs no launcher edit and the peer never creates it.
                    try:
                        import json as _jxi

                        _xiS = globals().setdefault("_R218_XA", {"n": 0})
                        _xiS["n"] += 1
                        with open("/tmp/r218_xaudit.jsonl", "a") as _fxi:
                            _fxi.write(
                                _jxi.dumps(
                                    {
                                        "ev": "init",
                                        "n": _xiS["n"],
                                        "rid": str(_rid),
                                        "b": int(b),
                                        "slot": int(_slot),
                                        "sl": int(sl),
                                        "pref": int(pref),
                                        "rcnt": int(rcnt),
                                        "X": int(max(pref, sl - self._TT_RECENT)),
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass

            # R137 P0.2 (generalized via R144 P0 slot reuse): zero this slot's
            # per-layer HP region on new prefill so an A->B->A generate (or any
            # 2nd generate in-process, or a reused slot) does not read stale
            # rotated KV from the prior occupant. Cheap: one zero kernel per
            # (layer, slot) at prefill (eager, not captured). Env-gated
            # (R137_HP_ZERO, default 1) for diagnostics.
            if os.environ.get("R137_HP_ZERO", "1") == "1":
                _zhk, _zhv = self._tt_hp_bufs(self._oscar_layer_ord(layer), query.device, query.dtype)
                _zhk[_slot].zero_()
                _zhv[_slot].zero_()
                if _r144_ovl_os.environ.get("R164_ZEROAUDIT", "0") == "1":
                    try:
                        with open("/tmp/r164_zeros.log", "a") as _zf:
                            _zf.write(
                                f"Z lid={self._oscar_layer_ord(layer)} impl={id(self) % 100000:05d} "
                                f"slot={_slot} rid={str(_rid)[-8:]} sl={sl}\n"
                            )
                    except Exception:
                        pass

            # Write HP prefix: positions [0, pref)
            if pref > 0:
                self._tt_write_hp(
                    self._oscar_layer_ord(layer), _slot, 0, (key_rot[s0 : s0 + pref], value_rot[s0 : s0 + pref])
                )

            # Write HP recent: positions [pref, pref+rcnt)
            if rcnt > 0:
                self._tt_write_hp(
                    self._oscar_layer_ord(layer),
                    _slot,
                    pref,
                    (key_rot[s0 + sl - rcnt : s0 + sl], value_rot[s0 + sl - rcnt : s0 + sl]),
                )
                if _r144_ovl_os.environ.get("R164_SYNCCHECK", "0") == "1" and rcnt >= 512:
                    # ordering-correct post-write verification: full sync THEN
                    # checksum (the async checksum raced the engine copy and
                    # read pre-write zeros — retracted evidence). One sync per
                    # completing prefill layer only.
                    try:
                        # torch.npu is registered by the module-top torch_npu import
                        torch.npu.synchronize()
                        _hkC = self._tt_hp_k[self._oscar_layer_ord(layer)]
                        _s1 = round(float(_hkC[_slot, pref : pref + 8].float().sum().item()), 1)
                        _s2 = round(float(_hkC[_slot, pref + rcnt - 8 : pref + rcnt].float().sum().item()), 1)
                        with open("/tmp/r164_synccheck.log", "a") as _fc:
                            _fc.write(
                                f"C lid={self._oscar_layer_ord(layer)} slot={_slot} rid={str(_rid)[-8:]} "
                                f"head8={_s1} tail8={_s2} rcnt={rcnt}\n"
                            )
                    except Exception:
                        pass

            # R170_PFDUMP (default OFF): after the genuine-prefill full-window
            # ring write, sync + dump this layer's slot rows. Paired with the
            # replay-time RINGDUMP: stale-HERE => the write itself failed or
            # landed on wrong rows; fresh-here but stale-at-first-replay =>
            # clobbered by a late copy between prefill and first replay.
            try:
                import os as _os170pf

                if _os170pf.environ.get("R170_PFDUMP", "0") == "1":
                    torch.npu.synchronize()
                    # route the row dump through the gated diag sink (the
                    # stash module it used to write to does not exist outside
                    # the dev tree)
                    import json as _js170pf

                    _pfk = self._tt_hp_k.get(self._oscar_layer_ord(layer))
                    if _pfk is not None:
                        _rows170 = min(int(_pfk.shape[1]), 1024)
                        with _odiag_o("r170_pfdump.jsonl") as _f170pf:
                            _f170pf.write(
                                _js170pf.dumps(
                                    {
                                        "src": "pfdump",
                                        "lid": int(self._oscar_layer_ord(layer)),
                                        "slot": int(_slot),
                                        "sl": int(sl),
                                        "pref": int(pref),
                                        "rcnt": int(rcnt),
                                        "ring_k": [
                                            round(float(x), 2)
                                            for x in _pfk[_slot, :_rows170].float().abs().sum(dim=(1, 2)).tolist()
                                        ],
                                    }
                                )
                                + "\n"
                            )
            except Exception:
                pass
            _oscar_dbg(
                f"[3T-PF] b={b} sl={sl} pref={pref} rcnt={rcnt}",
                flush=True,
            )
            # R218 QDUMP (DIAGNOSTIC_ONLY, default OFF): content ground truth
            # for the INT2 middle — read back the just-written pages for THIS
            # row's middle span, dequantize, and compare against the rotated
            # BF16 K/V the quantizer consumed (key_rot/value_rot slices).
            # All read paths (eager per-row + both graph launchers) share the
            # same pages, so a mismatch here indicts the WRITE side; a match
            # with large coserr indicts 2-bit fidelity itself.
            if _r144_ovl_os.environ.get("R218_QDUMP", "0") == "1" and self._oscar_layer_ord(layer) == 0 and sl > 1500:
                try:
                    _qdS = globals().setdefault("_R218_QD_N", set())
                    if _rid not in _qdS:
                        _qdS.add(_rid)
                        _mid0, _mid1 = pref, sl - rcnt
                        _bsQ = self.vllm_config.cache_config.block_size
                        _bloQ, _blhiQ = _mid0 // _bsQ, (_mid1 + _bsQ - 1) // _bsQ
                        _btQ = attn_metadata.block_tables[b, _bloQ:_blhiQ]
                        _phQ = self.key_cache.shape[-1]
                        _TCQ = (_blhiQ - _bloQ) * _bsQ
                        _offQ = _mid0 - _bloQ * _bsQ
                        _nqQ = _mid1 - _mid0
                        _kbQ = self.key_cache[_btQ].contiguous()
                        _kfQ = self._dequantize_packed_int2(
                            _kbQ.view(_TCQ, self.num_kv_heads, _phQ),
                            self.k_scale_cache[_btQ].contiguous().view(_TCQ, self.num_kv_heads),
                            self.k_zero_cache[_btQ].contiguous().view(_TCQ, self.num_kv_heads),
                            key_rot.dtype,
                        )
                        _vbQ = self.value_cache[_btQ].contiguous()
                        _vfQ = self._dequantize_packed_int2(
                            _vbQ.view(_TCQ, self.num_kv_heads, _phQ),
                            self.v_scale_cache[_btQ].contiguous().view(_TCQ, self.num_kv_heads),
                            self.v_zero_cache[_btQ].contiguous().view(_TCQ, self.num_kv_heads),
                            value_rot.dtype,
                        )
                        _ktQ = key_rot[s0 + _mid0 : s0 + _mid1].float()
                        _vtQ = value_rot[s0 + _mid0 : s0 + _mid1].float()
                        _kdQ = _kfQ[_offQ : _offQ + _nqQ].float()
                        _vdQ = _vfQ[_offQ : _offQ + _nqQ].float()
                        _cosk = torch.nn.functional.cosine_similarity(
                            _ktQ.reshape(-1, _ktQ.shape[-1]), _kdQ.reshape(-1, _kdQ.shape[-1]), dim=-1
                        )
                        _cosv = torch.nn.functional.cosine_similarity(
                            _vtQ.reshape(-1, _vtQ.shape[-1]), _vdQ.reshape(-1, _vdQ.shape[-1]), dim=-1
                        )
                        _relk = ((_ktQ - _kdQ).norm() / (_ktQ.norm() + 1e-9)).item()
                        _relv = ((_vtQ - _vdQ).norm() / (_vtQ.norm() + 1e-9)).item()
                        with open("/tmp/r218_qdump.log", "a") as _fq:
                            _fq.write(
                                "QD rid={} slot={} sl={} mid=[{},{}) n={} bt0={} "
                                "cosK p5={:.4f} p50={:.4f} mean={:.4f} | cosV p5={:.4f} p50={:.4f} mean={:.4f} "
                                "| relK={:.4f} relV={:.4f} kabs={:.4f} dabs={:.4f}\n".format(
                                    str(_rid)[-8:],
                                    _slot,
                                    sl,
                                    _mid0,
                                    _mid1,
                                    _nqQ,
                                    _btQ[:4].tolist(),
                                    _cosk.quantile(0.05).item(),
                                    _cosk.median().item(),
                                    _cosk.mean().item(),
                                    _cosv.quantile(0.05).item(),
                                    _cosv.median().item(),
                                    _cosv.mean().item(),
                                    _relk,
                                    _relv,
                                    _ktQ.abs().mean().item(),
                                    _kdQ.abs().mean().item(),
                                )
                            )
                        torch.save(
                            {
                                "kt": _ktQ.cpu(),
                                "kd": _kdQ.cpu(),
                                "vt": _vtQ.cpu(),
                                "vd": _vdQ.cpu(),
                                "bt": _btQ.cpu(),
                                "mid": (_mid0, _mid1),
                            },
                            "/tmp/r218_qdump_{}.pt".format(str(_rid)[-6:]),
                        )
                except Exception as _eQ:
                    try:
                        with open("/tmp/r218_qdump.log", "a") as _fq:
                            _fq.write("QD ERR {!r}\n".format(_eQ))
                    except Exception:
                        pass
            if _r144_ovl_os.environ.get("R197_DBG", "0") == "1":  # R197 Fix-0
                print(f"[3T-PF-DBG] b={b} sl={sl} pref={pref} rcnt={rcnt} s0={s0} s1={s1}", file=sys.stderr, flush=True)

        # Run attention on the (already rotated) float K/V via attn core
        # (no rotation inside -- already done above). R201: under MIXDEC the
        # decode-mixed rows' outputs are overwritten by the sub-batch below —
        # pass them as skip_rows so the mask-free fast path doesn't compute
        # discarded 1-token chunk-local attention for them.
        _skip201 = (
            [_b201 for (_b201, _s201) in _r164_mix_rows]
            if (_r164_mix_rows and os.environ.get("R164_MIXDEC", "1") == "1")
            else None
        )
        if _R2XX_G.setdefault("_R218_FIRES", set()).isdisjoint({"P2"}):
            _R2XX_G["_R218_FIRES"].add("P2")
            _r218_diag(
                "FIRE-P2 core_prefill caller qrot_none={} krot_none={}".format(query_rot is None, key_rot is None)
            )
        _ret = self._attn_core_prefill(query_rot, key_rot, value_rot, attn_metadata, output, layer, skip_rows=_skip201)
        if _r218_cont_rows:
            # ---------------------------------------------------------------
            # R218 FIX2b -- SPLIT-PREFILL CONTINUATION ATTENTION.
            # `_attn_core_prefill` attends each request slice against ITS OWN
            # chunk K/V (key = key_float[s0:s1]) with a block-diagonal causal
            # mask built from query_start_loc -- so a continuation chunk
            # (sl < total) would attend only its own sl tokens and lose the
            # whole history (its last-position hidden state feeds the FIRST
            # sampled token). For those rows ONLY, attend the full three-tier
            # history -- prefix + INT2 middle + rotated VRING recent window,
            # exactly the parts the decode fn assembles -- against the chunk
            # queries with a causal mask shifted by (total - sl). The KV array
            # is in logical order and spans [0, total) contiguously
            # (prefix [0,pref), middle [pref, total-rcnt), recent
            # [total-rcnt, total)), so the mask is `key_idx <= total-sl+i`.
            # Disjoint from the MIXDEC rows (those are the 1-token decode rows).
            _lidC = self._oscar_layer_ord(layer)
            _RvC = None
            if self._oscar_R_v is not None and _lidC < self._oscar_R_v.shape[0]:
                _RvC = self._oscar_R_v[_lidC].T
            for _bC, _s0C, _s1C in _r218_cont_rows:
                _ridC = self._tt_rid(_bC)
                _slotC = self._tt_slot(_bC)
                _prefC = int(self._tt_pref.get(_ridC, 0))
                _rcntC = int(self._tt_rcnt.get(_ridC, 0))
                _totC = int(attn_metadata.seq_lens_list[_bC])
                _slC = _s1C - _s0C
                _hkC, _hvC = self._tt_hp_bufs(_lidC, query.device, query.dtype)
                _pk = []
                _pv = []
                if _prefC > 0:
                    _pk.append(_hkC[_slotC, :_prefC])
                    _pv.append(_hvC[_slotC, :_prefC])
                _midEndC = _totC - _rcntC
                if _midEndC > _prefC:
                    # R218 STAGE first: if this request's in-flight prompt is
                    # still staged in BF16, attend THOSE values (identical to
                    # what an unsplit prefill would attend). Only when the
                    # stage was recycled/incomplete do we fall back to the
                    # INT2 paged read (pre-fix behavior).
                    _siC, _slenC = self._r218_stage_lookup(_slotC, _ridC)
                    _needC = _midEndC - _prefC
                    if os.environ.get("R144_PF_PROBE", "0") == "2":
                        try:
                            with _odiag_o(
                                "r218_stage{}.log".format(_r144_ovl_os.environ.get("R218_STAGE_TAG", ""))
                            ) as _fst:
                                _fst.write(
                                    "R L={} b={} rid={} slot={} pref={} midEnd={} si={} len={} need={}\n".format(
                                        _lidC, _bC, _ridC, _slotC, _prefC, _midEndC, _siC, _slenC, _needC
                                    )
                                )
                        except Exception:
                            pass
                    if _siC is not None and _slenC >= _needC:
                        _skC, _svC, _capSC = self._r218_stage_bufs(_lidC, query.device, query.dtype)
                        _pk.append(_skC[_siC, :_needC])
                        _pv.append(_svC[_siC, :_needC])
                    else:
                        _bsC = self.vllm_config.cache_config.block_size
                        _, _, _, _phC = self.key_cache.shape
                        _bloC = _prefC // _bsC
                        _blHiC = (_midEndC + _bsC - 1) // _bsC
                        _btC = attn_metadata.block_tables[_bC, _bloC:_blHiC]
                        _kbC = self.key_cache[_btC].contiguous()
                        _vbC = self.value_cache[_btC].contiguous()
                        _ksC = self.k_scale_cache[_btC].contiguous()
                        _kzC = self.k_zero_cache[_btC].contiguous()
                        _vsC = self.v_scale_cache[_btC].contiguous()
                        _vzC = self.v_zero_cache[_btC].contiguous()
                        _TC = (_blHiC - _bloC) * _bsC
                        _kfC = self._dequantize_packed_int2(
                            _kbC.view(_TC, self.num_kv_heads, _phC),
                            _ksC.view(_TC, self.num_kv_heads),
                            _kzC.view(_TC, self.num_kv_heads),
                            query.dtype,
                        )
                        _vfC = self._dequantize_packed_int2(
                            _vbC.view(_TC, self.num_kv_heads, _phC),
                            _vsC.view(_TC, self.num_kv_heads),
                            _vzC.view(_TC, self.num_kv_heads),
                            query.dtype,
                        )
                        _offC = _prefC - _bloC * _bsC
                        _pk.append(_kfC[_offC : _offC + (_midEndC - _prefC)])
                        _pv.append(_vfC[_offC : _offC + (_midEndC - _prefC)])
                if _rcntC > 0:
                    _wiC = (
                        self._r198_win_ridx(_ridC, _slotC, _totC, _rcntC, _hkC)
                        if _r144_ovl_os.environ.get("R198_VRING", "0") == "1"
                        else None
                    )
                    if _wiC is not None:
                        _nfC = int(_hkC.shape[2]) * int(_hkC.shape[3])
                        _pk.append(_hkC.view(-1, _nfC).index_select(0, _wiC).view(_rcntC, _hkC.shape[2], _hkC.shape[3]))
                        _pv.append(_hvC.view(-1, _nfC).index_select(0, _wiC).view(_rcntC, _hvC.shape[2], _hvC.shape[3]))
                    else:
                        _pk.append(_hkC[_slotC, _prefC : _prefC + _rcntC])
                        _pv.append(_hvC[_slotC, _prefC : _prefC + _rcntC])
                if not _pk:
                    continue  # no history at all -> chunk-local is correct
                _kC = torch.cat(_pk, dim=0)
                _vC = torch.cat(_pv, dim=0)
                if _kC.shape[0] < _slC:
                    continue
                _arC = torch.arange(_kC.shape[0], device=query.device).view(1, -1)
                _qpc = torch.arange(_totC - _slC, _totC, device=query.device).view(-1, 1)
                _mC = _arC <= _qpc
                if os.environ.get("R144_PF_PROBE", "0") == "2":  # R218 FIX2b geometry
                    try:
                        with _odiag_o("r218_fix2b.log") as _f2b:
                            _f2b.write(
                                "L={} b={} slot={} slC={} totC={} prefC={} rcntC={} nkC={} "
                                "npref={} nmid={} nrec={} qdt={}\n".format(
                                    _lidC,
                                    _bC,
                                    _slotC,
                                    _slC,
                                    _totC,
                                    _prefC,
                                    _rcntC,
                                    int(_kC.shape[0]),
                                    int(_prefC),
                                    int(max(0, (_totC - _rcntC) - _prefC)),
                                    int(_rcntC),
                                    str(query.dtype),
                                )
                            )
                    except Exception:
                        pass
                _oC = torch.nn.functional.scaled_dot_product_attention(
                    query_rot[_s0C:_s1C].unsqueeze(0).transpose(1, 2),
                    _kC.unsqueeze(0).transpose(1, 2),
                    _vC.unsqueeze(0).transpose(1, 2),
                    attn_mask=_mC,
                    scale=self.scale,
                    is_causal=False,
                )
                _oC = _oC.transpose(1, 2).squeeze(0).contiguous()
                if _RvC is not None:
                    _oC = torch.matmul(_oC, _RvC)
                output[_s0C:_s1C] = _oC.view(_slC, self.num_heads, self.head_size)
        if _r164_mix_rows and os.environ.get("R164_MIXDEC", "1") == "1":
            # R164 FIX: decode rows mixed into this ChunkedPrefill batch get
            # REAL decode semantics — the eager three-tier decode on the
            # subset (ring append + S/R + INT2 middle + merge + unrot), then
            # scatter into the mixed batch's output rows. Subset rows are the
            # FIRST rows (b < num_decodes, 1 token each) so slot/b mapping is
            # preserved verbatim.
            import copy as _cp164

            _dn = len(_r164_mix_rows)
            # R203_METAFIX: build the MIXDEC sub-metadata ONCE per step. The
            # per-layer copy.copy created a FRESH object each layer, which
            # defeated the R196 idfill step-memo (object-identity keyed) —
            # the whole IDFILL-v2/FLATFIX flat refresh (several blocking
            # torch.tensor H2Ds + flat copies) re-ran 36x per mixed step,
            # writing layer-invariant subset values. All inputs derive from
            # the step-constant parent (mix rows, seq_lens_list, request_ids,
            # block_tables slice); the cache key re-checks them, and any
            # mismatch or stash failure falls back to a fresh copy (base
            # behavior).
            _sub_key164 = None
            _sub_md = None
            if _r144_ovl_os.environ.get("R203_METAFIX", "0") == "1":
                try:
                    _rids164k = getattr(attn_metadata, "request_ids", None)
                    _sub_key164 = (
                        tuple(_r164_mix_rows),
                        tuple(int(attn_metadata.seq_lens_list[b]) for b, _ in _r164_mix_rows),
                        tuple(_rids164k[b] for b, _ in _r164_mix_rows) if _rids164k is not None else None,
                    )
                    _c164 = getattr(attn_metadata, "_r203_submd_cache", None)
                    if _c164 is not None and _c164[0] == _sub_key164 and _c164[1] is attn_metadata:
                        _sub_md = _c164[2]
                except Exception:
                    _sub_md = None
            if _sub_md is None:
                _sub_md = _cp164.copy(attn_metadata)
                _sub_md.seq_lens_list = [int(attn_metadata.seq_lens_list[b]) for b, _ in _r164_mix_rows]
                _rids164 = getattr(attn_metadata, "request_ids", None)
                if _rids164 is not None:
                    _sub_md.request_ids = [_rids164[b] for b, _ in _r164_mix_rows]
                try:  # noqa: SIM105
                    _sub_md.block_tables = attn_metadata.block_tables[0:_dn]
                except Exception:
                    pass
                # R203: this copy is a SUBSET view of the step — its idfill
                # refresh must not run (a padded subset refresh would clobber
                # the MAIN refresh's real slot rows with identity values; the
                # main metadata's full-width padded refresh already covers
                # every live row of the step). See the gate in bind_slots.
                try:  # noqa: SIM105
                    _sub_md._r203_is_sub = True
                except Exception:
                    pass
                if _sub_key164 is not None:
                    try:  # noqa: SIM105
                        attn_metadata._r203_submd_cache = (_sub_key164, attn_metadata, _sub_md)
                    except Exception:
                        pass
            # RAW q rows: the decode fn rotates q internally (line ~6874);
            # passing the pre-rotated query_rot double-rotates Q.
            if _r144_ovl_os.environ.get("R164_MIXAUDIT", "0") == "1":
                try:
                    _cs = getattr(self, "_tt_cur_slots", None)
                    _cr = getattr(self, "_tt_cur_rids", None)
                    _req_ids = getattr(attn_metadata, "request_ids", None) or []
                    _sm_head = [(_r, type(self)._r164_slotmap.get(_r)) for _r, _ in [(x, 0) for x in _req_ids]][:3]
                    with open("/tmp/r164_mixaudit.log", "a") as _mf2:
                        _mf2.write(
                            f"MIX lid={self._oscar_layer_ord(layer)} rows={_r164_mix_rows} "
                            f"cur_slots={list(_cs or [])} cur_rids={list(_cr or [])[:3]} "
                            f"sm={_sm_head}\n"
                        )
                except Exception:
                    pass
            # R204_MIXCHAIN (default OFF): decode-mixed rows are the FIRST _dn
            # TOKEN rows of the packed batch and each carries exactly ONE token,
            # so their token offsets are s0 == 0.._dn-1 — the three per-row
            # torch.cat()s (each ~B slice views + a cat kernel) and the per-row
            # output scatter loop collapse to contiguous VIEWS plus ONE direct
            # write of the sub-batch output rows. Values are identical (a cat
            # of consecutive 1-row slices IS the contiguous slice); the decode
            # fn only READS _sub_q/_sub_k/_sub_v and writes _sub_out rows
            # [0,batch) exactly once, so writing straight into the framework
            # output view is order-equivalent to the deferred scatter.
            _mixc204 = _r144_ovl_os.environ.get("R204_MIXCHAIN", "0") == "1" and [
                s0 for _, s0 in _r164_mix_rows
            ] == list(range(_dn))
            if _mixc204:
                _sub_q = query[:_dn]
                _sub_k = key_rot[:_dn]
                _sub_v = value_rot[:_dn]
                _sub_out = output[:_dn]
            else:
                _sub_q = torch.cat([query[s0 : s0 + 1] for _, s0 in _r164_mix_rows], 0)
                _sub_k = torch.cat([key_rot[s0 : s0 + 1] for _, s0 in _r164_mix_rows], 0)
                _sub_v = torch.cat([value_rot[s0 : s0 + 1] for _, s0 in _r164_mix_rows], 0)
                _sub_out = torch.empty_like(_sub_q)
            self._forward_int2_decode_threetier_middle(_sub_q, _sub_k, _sub_v, _sub_md, _sub_out, layer)
            if not _mixc204:
                for _i164, (_b164, _s0164) in enumerate(_r164_mix_rows):
                    output[_s0164 : _s0164 + 1] = _sub_out[_i164 : _i164 + 1]
            # R204_MIXSYNC (default OFF): a mixed step's eager writes (HP ring
            # appends for the decode rows + the R167/R169 flat refreshes) are
            # the producers for the FIRST decode-graph replay that follows the
            # mixed phase. The replay's internal stream is covered by neither
            # stream FIFO nor event edges (R198 dbg6/7 carrier-write race
            # root — the same exposure class), so the transition is a
            # submission race the incumbent only wins by host-time accident
            # (intermittent stuck rows 19/30, [first,0,0,...] forever: the
            # raced replay slides/maintains the ring with stale flat values —
            # delivered-base rate ~1/11 legs, e.g. r203e_mf_b32_r2). ONE full
            # device sync at the LAST layer ordinal of every mixed step makes
            # the producers unconditionally visible before the next step is
            # scheduled — the same primitive R198_ISTAGE uses for restage.
            # Layer count is model-config-derived (never a literal); fires
            # once per mixed step (the impl owning the last ordinal).
            if _r144_ovl_os.environ.get("R204_MIXSYNC", "0") == "1" and not _EXTRA_CTX.capturing:
                try:
                    _nl204 = getattr(type(self), "_r204_nlayers", None)
                    if _nl204 is None:
                        _mc204 = self.vllm_config.model_config
                        # vLLM ModelConfig exposes get_total_num_hidden_layers()
                        # (TP-agnostic); HF configs carry num_hidden_layers.
                        try:
                            _nl204 = int(_mc204.get_total_num_hidden_layers())
                        except Exception:
                            _nl204 = 0
                        if _nl204 <= 0:
                            _nl204 = int(getattr(getattr(_mc204, "hf_config", None), "num_hidden_layers", 0) or 0)
                        if _nl204 <= 0:
                            _nl204 = int(getattr(getattr(_mc204, "hf_text_config", None), "num_hidden_layers", 0) or 0)
                        type(self)._r204_nlayers = _nl204
                    if _nl204 > 0 and self._oscar_layer_ord(layer) == _nl204 - 1:
                        # one-shot engagement marker via the FILE channel —
                        # V1 EngineCore swallows post-init stderr prints
                        # (probe-proven r204f: stderr silent, file visible)
                        if not getattr(type(self), "_r204_sync_fired", False):
                            type(self)._r204_sync_fired = True
                            try:
                                with open("/tmp/r204_mixsync.log", "a") as _pf204:
                                    _pf204.write(f"[R204-MIXSYNC] engaged pid={_r144_ovl_os.getpid()} nl={_nl204}\n")
                            except Exception:
                                pass
                        torch.npu.synchronize()
                except Exception:
                    pass
        # R204_MIXSYNC2 (default OFF): the SAME last-layer full-device sync as
        # R204_MIXSYNC, but for ANY prefill step — including PURE prefill
        # steps (no decode rows mixed). A pure-prefill step's eager writes
        # (HP ring appends, R167/R169 flat refreshes, staging) are producers
        # for the FIRST decode-graph replay exactly like a mixed step's, and
        # MIXSYNC never fires for them (its gate requires _r164_mix_rows) —
        # leaving the prefill->replay transition a submission race. Observed
        # as the deterministic single-row token-1 flip on the ragged frozen
        # config (frzcan_gr/grc1/grc2 rev row5, LCP=1/64; eager clean).
        if _r144_ovl_os.environ.get("R204_MIXSYNC2", "0") == "1" and not _r164_mix_rows and not _EXTRA_CTX.capturing:
            try:
                _nl204b = getattr(type(self), "_r204_nlayers", None)
                if _nl204b is None:
                    _mc204b = self.vllm_config.model_config
                    try:
                        _nl204b = int(_mc204b.get_total_num_hidden_layers())
                    except Exception:
                        _nl204b = 0
                    if _nl204b <= 0:
                        _nl204b = int(getattr(getattr(_mc204b, "hf_config", None), "num_hidden_layers", 0) or 0)
                    if _nl204b <= 0:
                        _nl204b = int(getattr(getattr(_mc204b, "hf_text_config", None), "num_hidden_layers", 0) or 0)
                    type(self)._r204_nlayers = _nl204b
                if _nl204b > 0 and self._oscar_layer_ord(layer) == _nl204b - 1:
                    if not getattr(type(self), "_r204_sync2_fired", False):
                        type(self)._r204_sync2_fired = True
                        try:
                            with open("/tmp/r204_mixsync.log", "a") as _pf204b:
                                _pf204b.write(f"[R204-MIXSYNC2] engaged pid={_r144_ovl_os.getpid()} nl={_nl204b}\n")
                        except Exception:
                            pass
                    torch.npu.synchronize()
            except Exception:
                pass
        return _ret

    def _attn_core_prefill(self, query, key_float, value_float, attn_metadata, output, layer, skip_rows=None):
        """Attention core for prefill (no rotation). Q/K/V are already in the
        correct (rotated) space. Runs SDPA or NPU FIA and applies R_v.T inverse
        to output. skip_rows: batch rows whose output the caller overwrites
        (R164_MIXDEC decode-mixed subset; R201_PFNOMASK skips their slices)."""
        num_tokens = query.shape[0]
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        query = query.contiguous()
        key_float = key_float.contiguous()
        value_float = value_float.contiguous()

        # R165 FIX (multi-request prefill batches): flat-batch is_causal=True
        # lets request B's tokens attend request A's leading tokens whenever
        # a decode row and another request's prefill chunk share one step
        # (split arrivals / chunked prefill). Single-request batches keep the
        # EXACT original call (canonical bit-path); multi-request batches get
        # an explicit per-request causal mask (block-diagonal + causal).
        # R201: the [T,T] mask BUILD is now LAZY — only materialized when the
        # R201_PFNOMASK fast path is off or fails (the fast path never reads
        # it; kprof r201c: the eager build cost GreaterEqual 1.78ms + Equal
        # 0.82ms per layer even when the masked SDPA itself was skipped).
        _nreq_pf = len(attn_metadata.seq_lens_list)
        _mask_pf = None

        def _r201_build_mask():
            _qs = attn_metadata.query_start_loc
            if hasattr(_qs, "cpu"):
                _qs = _qs.cpu()
            _qs = _qs.tolist() if not isinstance(_qs, list) else _qs
            _T = int(num_tokens)
            _ar = torch.arange(_T, device=query.device)
            # request id per token position
            _rid_pf = torch.zeros(_T, dtype=torch.long, device=query.device)
            for _rpf in range(len(_qs) - 1):
                _s0p, _s1p = int(_qs[_rpf]), int(_qs[_rpf + 1])
                if _s1p > _s0p:
                    _rid_pf[_s0p:_s1p] = _rpf
            return (_ar.view(-1, 1) >= _ar.view(1, -1)) & (_rid_pf.view(-1, 1) == _rid_pf.view(1, -1))

        # R201_PFNOMASK (default OFF): multi-request prefill batches attend
        # PER-REQUEST with the plain causal kernel instead of one masked SDPA
        # over the concatenation. The block-diagonal+causal mask is exactly
        # "query i attends key j iff same request AND j <= i" — per-request
        # causal slices over query_start_loc compute the identical set with
        # NO [T,T] mask tensor and no cross-request score waste. Standalone
        # probe (RAW/r201_probe_pfnm.py, 2x4096 GQA 32/8/128 bf16): masked
        # 11.87ms vs loop 2.87ms per layer call (mask build alone 2.73ms);
        # maxdiff 1.0e-3 rel 2.7e-4 (bf16 kernel-order class). The TND FIA
        # fallback tiling-fails (561002) at this geometry, so the loop is the
        # only mask-free form. Single-request batches keep the EXACT original
        # call below (canonical bit-path untouched). Any exception falls back
        # to the delivered mask path (fail-open to base behavior).
        # skip_rows: decode-mixed 1-token rows (R164_MIXDEC subset) whose
        # output the caller overwrites via the eager three-tier sub-batch
        # right after this core — computing them here is discarded work
        # (kprof r201c: 30 tiny SDPA+Triu launches per layer per mixed step).
        if _nreq_pf > 1 and os.environ.get("R201_PFNOMASK", "0") == "1":
            try:
                _en201 = globals().setdefault("_R201_ENGAGED", [0])
                _en201[0] += 1
                if _en201[0] <= 3 or (_en201[0] % 1000) == 0:
                    try:
                        with open("/tmp/r201_engaged.log", "a") as _f201:
                            _f201.write(
                                "PFNOMASK n={} nreq={} T={} skip={}\n".format(
                                    _en201[0], _nreq_pf, num_tokens, len(skip_rows or ())
                                )
                            )
                    except Exception:
                        pass
                _qs201 = None
                if os.environ.get("R203_METAFIX", "0") == "1":
                    # R203: per-step mirror — same step-constant tensor as the
                    # threetier-prefill entry read (one D2H per step total).
                    _qs201 = _r203_qsl_host(attn_metadata)
                if _qs201 is None or not isinstance(_qs201, list):
                    _qs201 = attn_metadata.query_start_loc
                    if hasattr(_qs201, "cpu"):
                        _qs201 = _qs201.cpu()
                    _qs201 = _qs201.tolist() if not isinstance(_qs201, list) else _qs201
                _Rv201 = None
                if self._oscar_R_v is not None and self._oscar_layer_ord(layer) < self._oscar_R_v.shape[0]:
                    _Rv201 = self._oscar_R_v[self._oscar_layer_ord(layer)].T
                _w201 = 0
                _nskip201 = 0
                for _r201 in range(len(_qs201) - 1):
                    _s0201, _s1201 = int(_qs201[_r201]), int(_qs201[_r201 + 1])
                    if _s1201 <= _s0201:
                        continue
                    if skip_rows is not None and _r201 in skip_rows and (_s1201 - _s0201) == 1:
                        # leave this output row to the MIXDEC sub-batch
                        # scatter (it attends full three-tier history; the
                        # chunk-local value here is discarded by overwrite)
                        _nskip201 += 1
                        continue
                    _o201 = torch.nn.functional.scaled_dot_product_attention(
                        query[_s0201:_s1201].unsqueeze(0).transpose(1, 2),
                        key_float[_s0201:_s1201].unsqueeze(0).transpose(1, 2),
                        value_float[_s0201:_s1201].unsqueeze(0).transpose(1, 2),
                        attn_mask=None,
                        scale=self.scale,
                        is_causal=True,
                    )
                    _o201 = _o201.transpose(1, 2).squeeze(0).contiguous()
                    if _Rv201 is not None:
                        _o201 = torch.matmul(_o201, _Rv201)
                    # per-slice direct write (skipped rows keep whatever the
                    # MIXDEC scatter writes; positions stay row-aligned)
                    output[_s0201:_s1201] = _o201.view(_s1201 - _s0201, self.num_heads, self.head_size)
                    _w201 += _s1201 - _s0201
                if _w201 == 0 and _nskip201 == 0:
                    raise RuntimeError("R201_PFNOMASK: no non-empty request slices")
                return output
            except Exception as _e201:
                _oscar_dbg(
                    f"[INT2-PREFILL] R201_PFNOMASK fast path failed ({_e201}); falling back to mask path", flush=True
                )
        if _nreq_pf > 1 and _mask_pf is None:
            # lazy mask build — only the fallback path reads it (R201)
            try:
                _mask_pf = _r201_build_mask()
            except Exception:
                _mask_pf = None
        try:
            q_sdpa = query.unsqueeze(0)
            k_sdpa = key_float.unsqueeze(0)
            v_sdpa = value_float.unsqueeze(0)
            q_sdpa = q_sdpa.transpose(1, 2)
            k_sdpa = k_sdpa.transpose(1, 2)
            v_sdpa = v_sdpa.transpose(1, 2)
            if _mask_pf is not None:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa, attn_mask=_mask_pf, scale=self.scale, is_causal=False
                )
            else:
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa, attn_mask=None, scale=self.scale, is_causal=True
                )
            attn_output = attn_output.transpose(1, 2).squeeze(0)
        except Exception as e:
            _oscar_dbg(f"[INT2-PREFILL] SDPA failed: {e}, trying NPU kernel", flush=True)
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                query,
                key_float,
                value_float,
                atten_mask=attn_metadata.attn_mask,
                input_layout="TND",
                actual_seq_lengths=actual_seq_qlen,
                actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
            )

        if self._oscar_R_v is not None:
            if self._oscar_layer_ord(layer) < self._oscar_R_v.shape[0]:
                R_v = self._oscar_R_v[self._oscar_layer_ord(layer)]
                attn_output = torch.matmul(attn_output, R_v.T)
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output
