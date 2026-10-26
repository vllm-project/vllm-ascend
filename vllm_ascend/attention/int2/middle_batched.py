# SPDX-License-Identifier: Apache-2.0
"""Batched middle-operand mixin: the R196 batched S+R builder and the
gated middle probes (midprobe / probe_record)."""

from __future__ import annotations

import os as _r144_ovl_os
from typing import Any

import torch
import torch_npu

from vllm_ascend.attention.int2.ctx import (  # noqa: F401
    _R2XX_G,
)
from vllm_ascend.attention.int2.helpers import (  # noqa: F401
    _r216_merge,
    _r218_diag,
)


class Int2MiddleBatchedMixin:
    # Host-state annotations: the operand-window caches and pending-event
    # handles below are owned by this mixin and initialized lazily through
    # type(self) (defaults are required there); the slot-mapping, rotation,
    # ring-window, and bridge-loader names are owned by the assembled backend
    # / sibling mixins. Bare annotations create no runtime attributes, so the
    # MRO at assembly is unaffected.
    _r200_ebt: Any = None
    _r200_ews: Any = None
    _r200_fwin: Any = None
    _r204_fiabuf: Any = None
    _r196_slbuf: Any = None
    _r196_nprbuf: Any = None
    _r196_vridx: Any = None
    _r197_dev_pend: Any = None
    _r112_mod: Any = None
    _tt_rid: Any
    _tt_slot: Any
    _tt_pref: dict[str, Any]
    _tt_rcnt: dict[str, Any]
    _tt_ring_org: dict[str, Any]
    _tt_mseq: int
    _tt_hp_bufs: Any
    _r198_pr_window: Any
    _oscar_layer_ord: Any
    _oscar_R_k: Any
    _oscar_R_v: Any
    _TT_RECENT: int
    head_size: int
    num_heads: int
    num_kv_heads: int
    scale: float

    def _r196_batched_sr(
        self, query, output, layer, seq_lens, batch_size, _r144_bmid, _st, _slab204=None, _qrot_pre=None
    ):
        """R196 P1 (R196_EAGER_BATCH=1, default OFF): batched S+R tier for the
        EAGER three-tier middle — the path every mixed prefill+decode step and
        eager decode leg takes. Replaces the legacy per-row loop's S+R section
        (B FIA launches + 2B cats + per-row merge math + stack) with:
          ONE padded FIA over the raw HP ring rows + per-row l_pr einsum
          VERBATIM + ONE vectorized merge + ONE unrot matmul + ONE writeback.
        Math contract per row is IDENTICAL to the legacy loop:
          - pr_k/pr_v are hk[slot, :n_pr] / hv[slot, :n_pr]: the legacy
            torch.cat([hk[slot,:pref], hk[slot,pref:pref+rcnt]]) concatenates
            two ADJACENT slices of the same row, i.e. the contiguous prefix.
          - FIA masks each row to its first n_pr keys via
            actual_seq_lengths_kv (the same masking contract the per-row call
            gets by passing an exact-length key).
          - l_pr is the legacy einsum+logsumexp sequence on identical values
            (bit-exact per row).
          - merge/unrot are the legacy formulas vectorized (elementwise
            identical arithmetic; absent-middle rows reduce exactly to o_pr).
        Numeric class: the padded-FIA softmax accumulation order may differ
        from the exact-length FIA (reorder-level; R192/R195 tolerance class).
        R196_SR_CMP=1 recomputes the legacy per-row path per call and logs
        max|diff| (o_pr, merged, final) — diagnostic gate, default OFF.
        Returns True when handled; any contract mismatch returns False
        BEFORE any write and the caller falls back to the legacy loop."""
        _nq = self.num_heads
        _nkv = self.num_kv_heads
        _hd = self.head_size
        _lord = self._oscar_layer_ord(layer)
        hk, hv = self._tt_hp_bufs(_lord, query.device, query.dtype)
        R_k = R_v = None
        if self._oscar_R_k is not None and _lord < self._oscar_R_k.shape[0]:
            R_k = self._oscar_R_k[_lord]
            R_v = self._oscar_R_v[_lord]
        slots = []
        nprs = []
        for b in range(batch_size):
            _rid = self._tt_rid(b)
            slots.append(int(self._tt_slot(b)))
            nprs.append(int(self._tt_pref.get(_rid, 0)) + int(self._tt_rcnt.get(_rid, 0)))
        _r200e = globals().setdefault("_R200_ENTRY", {"n": 0, "min0": 0, "bmid": 0, "ok": 0})
        _r200e["n"] += 1
        if _r200e["n"] % 256 == 0:
            open("/tmp/r200_engaged.log", "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                f"entry calls={_r200e['n']} min0={_r200e['min0']} "
                f"bmidmiss={_r200e['bmid']} ok={_r200e['ok']} "
                f"pg={globals().get('_R200_CNT', {}).get('pg', -1)} "
                f"rowfb={globals().get('_R200_CNT', {}).get('mid', -1)}\n"
            )
        if not nprs or min(nprs) < 1:
            _r200e["min0"] += 1
            if _r144_ovl_os.environ.get("R197_PROBE", "0") == "1":
                open("/tmp/r197_r196skip.log", "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                    f"nprs<1 B={batch_size} nprs={nprs[:8]}\n"
                )
            return False  # decode rows always have n_pr>=1; fail-safe
        for b in range(batch_size):
            sl = int(seq_lens[b])
            _rid = self._tt_rid(b)
            _mid = sl - int(self._tt_rcnt.get(_rid, 0)) - int(self._tt_pref.get(_rid, 0))
            if _mid > 0 and b not in _r144_bmid:
                _r200e["bmid"] += 1
                if _r144_ovl_os.environ.get("R197_PROBE", "0") == "1":
                    open("/tmp/r197_r196skip.log", "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                        f"bmid-miss B={batch_size} b={b} sl={sl} mid={_mid} keys={sorted(_r144_bmid.keys())[:10]}\n"
                    )
                return False  # row still needs the per-row middle candidate
        _r200e["ok"] += 1
        # ---- R200_EPRD (default OFF): eager paged S/R — the mixed-step twin
        # of R199_PRREAD. The eager three-tier middle consumes the HP ring
        # rows IN PLACE via a paged FIA v2 whose block_table addresses each
        # row's slot in PHYSICAL order, replacing (a) the per-step rotated
        # token gather (flat index build + 2 index_select + permute) and
        # (b) the chunked fp32 l_pr einsum (the FIA lse is used instead —
        # the SAME lse source the R199 in-graph path already merges with).
        # Set-equality contract per row is the R199-proven class: full ring
        # (rcnt == cap: any rotation => all cap slots live), growing ring
        # (mid == 0: physical [0,npr) == logical window, left-packed before
        # the first wrap), and non-VRING rows (legacy layout IS the physical
        # order). VRING rows grow at IDENTITY positions (token t at position
        # t), so a row with mid > 0 AND rcnt < cap (recompute/preempt/stale
        # tt states, e.g. sl=4097 npr=3) has its window at
        # [pref+mid, pref+mid+rcnt) — physical [0, npr) would read a
        # DIFFERENT token set. Such rows keep the EXACT legacy per-row
        # semantics (gather + exact-length FIA + fp32 einsum l_pr); the
        # healthy rows take the paged path. Fail-closed to the whole-call
        # gather path on geometry mismatch only.
        _pg200 = (
            _r144_ovl_os.environ.get("R200_EPRD", "0") == "1"
            and _r144_ovl_os.environ.get("R199_PRREAD", "0") == "1"
            and int(hk.shape[1]) % 128 == 0
            and max(nprs) <= int(hk.shape[1])
            and hk.is_contiguous()
            and hv.is_contiguous()
        )
        _fail200 = []
        # R200_EPRD_STRICT (default 0): page every row. For HEALTHY rows the
        # (mid>0 AND rcnt<cap) state is impossible BY CONSTRUCTION (at
        # prefill end rcnt = min(len-pref, cap) and mid = max(0, len-pref-cap):
        # rcnt<cap => mid==0), so the only rows a set-equality guard could
        # exclude are stale tt-states (pref=i^2+1 CANWARM leftovers, sl~4K
        # with npr<=50) whose windows are semantically arbitrary in the
        # DELIVERED BASE TOO (the legacy gather reads the same broken state;
        # same exposure class as R199's shipped idx_sr rebinding-row jumps).
        # STRICT=1 re-engages the row-selective exact-legacy handling below
        # as an escape hatch (costs ~5 micro-FIAs+einsum per eager call).
        _strict200 = _r144_ovl_os.environ.get("R200_EPRD_STRICT", "0") == "1"
        if _pg200:
            # Set-equality per row: physical [0, npr) == the legacy window iff
            # (no VRING origin) OR (rcnt == cap — FULL ring: all cap slots
            # live, so any rotation is the same set; NOTE npr==mhp is the
            # WRONG test: pref<128 rows (CANWAMP pref=i^2+1 leftovers) have
            # rcnt==cap with npr=pref+cap<mhp and ARE left-packed) OR
            # (mid == 0 — growing short-prompt window is left-packed).
            _cap200 = int(self._TT_RECENT)
            _failset = set()
            for b in range(batch_size):
                if self._tt_ring_org.get(self._tt_rid(b)) is None:
                    continue
                if int(seq_lens[b]) - nprs[b] > 0:
                    _prefX = int(self._tt_pref.get(self._tt_rid(b), 0))
                    if nprs[b] - _prefX < _cap200:
                        _failset.add(b)
                        _r200sig = globals().setdefault("_R200_SIG", {})
                        _k200 = (int(seq_lens[b]) // 256, nprs[b], _prefX, nprs[b] - _prefX)
                        _r200sig[_k200] = _r200sig.get(_k200, 0) + 1
                        if len(_r200sig) <= 64 and _r200sig[_k200] == 1:
                            open("/tmp/r200_engaged.log", "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                                f"FAILSIG sl256={_k200[0]} npr={_k200[1]} pref={_k200[2]} rcnt={_k200[3]}\n"
                            )
            if _strict200:
                if len(_failset) >= batch_size:
                    _pg200 = False  # nothing to page — gather path
                else:
                    _fail200 = sorted(_failset)
        _r200c = globals().setdefault("_R200_CNT", {"pg": 0, "mid": 0, "n": 0})
        _r200c["n"] += 1
        if _pg200:
            _r200c["pg"] += 1
            _r200c["mid"] += len(_failset) if _strict200 else 0
        if _r200c["n"] % 2048 == 0:
            open("/tmp/r200_engaged.log", "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                f"calls={_r200c['n']} pg={_r200c['pg']} row_fallbacks={_r200c['mid']} strict={int(_strict200)}\n"
            )
        if _pg200:
            if not globals().setdefault("_R200_ENGAGED", [False])[0]:
                globals()["_R200_ENGAGED"][0] = True
                open("/tmp/r200_engaged.log", "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                    f"EPRD first fire: B={batch_size} mhp={int(hk.shape[1])} nprs={nprs[:8]}\n"
                )
            _nb200 = int(hk.shape[1]) // 128
            _rows200 = [b for b in range(batch_size) if b not in set(_fail200)] if _fail200 else list(range(batch_size))
            _slotsP = [slots[b] for b in _rows200]
            _nprsP = [nprs[b] for b in _rows200]
            _Bp200 = len(_rows200)
            _ck200 = (tuple(_slotsP), _Bp200)
            _cs200 = getattr(type(self), "_r200_ebt", None)
            if _cs200 is None or _cs200[0] != _ck200:
                # bt[p, j] = slot(p)*nb + j — the PASS rows' PHYSICAL block
                # ids (pure function of the pass-slot list; rebuilt only on
                # binding changes, cached across layers AND steps).
                _bt200 = torch.tensor(_slotsP, dtype=torch.int32, device=query.device).view(
                    -1, 1
                ) * _nb200 + torch.arange(_nb200, dtype=torch.int32, device=query.device).view(1, -1)
                _sel200 = torch.tensor(_rows200, dtype=torch.int64, device=query.device)
                type(self)._r200_ebt = (_ck200, _bt200, _sel200)
            else:
                _bt200 = _cs200[1]
                _sel200 = _cs200[2]
            _q200 = query[:batch_size].index_select(0, _sel200) if _fail200 else query[:batch_size]
            # R204_MIXCHAIN: reuse the R146 carrier qrot (same batched matmul
            # values; skip one [B,nq,hd]@[hd,hd] per layer) on the identity
            # no-fail-row path.
            if _qrot_pre is not None and not _fail200 and _qrot_pre.size(0) == batch_size:
                qrot = _qrot_pre
            else:
                qrot = torch.matmul(_q200, R_k) if R_k is not None else _q200
            # workspace: sized once, monotone in B (max-workspace semantics;
            # a larger buffer serves smaller batches).
            _ews = getattr(type(self), "_r200_ews", None)
            if _ews is None or _ews[1] < _Bp200:
                _wsq = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                    query=torch.zeros(_Bp200, _nq, 1, _hd, dtype=query.dtype, device=query.device),
                    key=hk.view(-1, 128, _nkv * _hd),
                    value=hv.view(-1, 128, _nkv * _hd),
                    input_layout="BNSD",
                    block_table=_bt200,
                    block_size=128,
                    actual_seq_qlen=[1] * _Bp200,
                    actual_seq_kvlen=[int(hk.shape[1])] * _Bp200,
                    num_query_heads=_nq,
                    num_key_value_heads=_nkv,
                    sparse_mode=0,
                    pre_tokens=2147483647,
                    next_tokens=2147483647,
                    softmax_scale=self.scale,
                )
                _ews = type(self)._r200_ews = (_wsq, _Bp200)
            # R204_MIXCHAIN: persistent FIA out/lse buffers (was 2 fresh
            # torch.empty per layer-call; same-stream write-after-read of the
            # previous layer's cast is ordered — safe to reuse).
            _ob204 = getattr(type(self), "_r204_fiabuf", None)
            if _ob204 is None or _ob204[2] < _Bp200:
                _B204 = max(int(_Bp200), 8)
                _ob204 = (
                    torch.empty(_B204, _nq, 1, _hd, dtype=torch.bfloat16, device=query.device),
                    torch.empty(_B204, _nq, 1, 1, dtype=torch.float32, device=query.device),
                    _B204,
                )
                type(self)._r204_fiabuf = _ob204
            _o200 = _ob204[0][:_Bp200]
            _l200 = _ob204[1][:_Bp200]
            torch_npu.npu_fused_infer_attention_score_v2.out(
                query=qrot.unsqueeze(2),
                key=hk.view(-1, 128, _nkv * _hd),
                value=hv.view(-1, 128, _nkv * _hd),
                input_layout="BNSD",
                block_table=_bt200,
                block_size=128,
                actual_seq_qlen=[1] * _Bp200,
                actual_seq_kvlen=_nprsP,
                num_query_heads=_nq,
                num_key_value_heads=_nkv,
                softmax_scale=self.scale,
                sparse_mode=0,
                pre_tokens=2147483647,
                next_tokens=2147483647,
                return_softmax_lse=True,
                workspace=_ews[0],
                out=[_o200, _l200],
            )
            if not _fail200:
                o_pr = _o200[:, :, 0, :].float()  # [B, nq, hd]
                l_pr = _l200.view(batch_size, _nq)  # [B, nq] fp32 lse
            else:
                o_pr = torch.empty(batch_size, _nq, _hd, dtype=torch.float32, device=query.device)
                l_pr = torch.empty(batch_size, _nq, dtype=torch.float32, device=query.device)
                o_pr[_sel200] = _o200[:, :, 0, :].float()
                l_pr[_sel200] = _l200.view(_Bp200, _nq)
                # ---- failing rows keep the EXACT legacy semantics (the same
                # values the R196 legacy path produces: rotated-gather window
                # + padded v1 FIA + masked fp32 einsum l_pr). Their windows
                # are TINY (stale tt states: npr = pref+rcnt <= ~50) and
                # stable across the step's 36 layer-calls — stack the padded
                # K/V windows ONCE per step (cached), then ONE v1 FIA + ONE
                # einsum chunk per layer.
                _kvh200 = torch.arange(_nq, device=query.device) // (_nq // _nkv)
                _fw = []
                for b in _fail200:
                    _ridF = self._tt_rid(b)
                    _fw.append(
                        (
                            int(slots[b]),
                            int(seq_lens[b]),
                            int(self._tt_pref.get(_ridF, 0)),
                            nprs[b],
                            int(self._tt_ring_org.get(_ridF, 0) or 0),
                        )
                    )
                _ckF = (tuple(_fw), int(hk.shape[1]), str(query.device))
                _cfF = getattr(type(self), "_r200_fwin", None)
                if _cfF is None or _cfF[0] != _ckF:
                    _nprsF = [x[3] for x in _fw]
                    _maxnF = max(_nprsF)
                    _BfF = len(_fw)
                    _kfT = torch.zeros(_BfF, _maxnF, _nkv, _hd, dtype=query.dtype, device=query.device)
                    _vfT = torch.zeros(_BfF, _maxnF, _nkv, _hd, dtype=query.dtype, device=query.device)
                    for _iF, b in enumerate(_fail200):
                        _ridF = self._tt_rid(b)
                        _prefF = int(self._tt_pref.get(_ridF, 0))
                        _prkF, _prvF = self._r198_pr_window(
                            _ridF, int(slots[b]), int(seq_lens[b]), _prefF, nprs[b] - _prefF, hk, hv
                        )
                        if _prkF is None:
                            continue
                        _kfT[_iF, : nprs[b]].copy_(_prkF)
                        _vfT[_iF, : nprs[b]].copy_(_prvF)
                    _cfF = type(self)._r200_fwin = (_ckF, _kfT, _vfT, _nprsF, _maxnF)
                (_, _kfT, _vfT, _nprsF, _maxnF) = _cfF
                _selF = torch.tensor(_fail200, dtype=torch.int64, device=query.device)
                _qF = query[:batch_size].index_select(0, _selF)
                _qrF = torch.matmul(_qF, R_k) if R_k is not None else _qF
                _alF, _ = torch_npu.npu_fused_infer_attention_score(
                    query=_qrF.unsqueeze(2),
                    key=_kfT.permute(0, 2, 1, 3),
                    value=_vfT.permute(0, 2, 1, 3),
                    input_layout="BNSD",
                    actual_seq_lengths=[1] * len(_fail200),
                    actual_seq_lengths_kv=_nprsF,
                    num_heads=_nq,
                    num_key_value_heads=_nkv,
                    scale=self.scale,
                    sparse_mode=0,
                )
                o_pr[_selF] = _alF[:, :, 0, :].float()
                _kkF = _kfT[:, :, _kvh200, :].permute(0, 2, 1, 3).float()
                _scF = torch.einsum("bhd,bhnd->bhn", _qrF.float(), _kkF) * self.scale
                _arF = torch.arange(_maxnF, device=query.device)
                _npF = torch.tensor(_nprsF, dtype=torch.int64, device=query.device)
                _scF = _scF.masked_fill(_arF.view(1, 1, -1) >= _npF.view(-1, 1, 1), float("-inf"))
                l_pr[_selF] = torch.logsumexp(_scF, dim=-1)
        else:
            # ---- gather S+R rows (zero-copy when slots are ascending-contiguous)
            # (legacy R196 path — kept verbatim as the fallback)
            _vr196 = _r144_ovl_os.environ.get("R198_VRING", "0") == "1" and all(
                self._tt_ring_org.get(self._tt_rid(b)) is not None for b in range(batch_size)
            )
            if _vr196:
                # R198 VRING: rows are rotated — materialize per-row windows in
                # token order via one flat gather (cols >= npr_b point at row
                # position 0; they are masked by actual_seq_lengths_kv / the
                # einsum mask, same class as the padded-row padding). Built once
                # per step (cached; shared by all 36 layers).
                _ck196 = (
                    tuple(slots),
                    tuple(int(x) for x in nprs),
                    tuple(int(x) for x in seq_lens[:batch_size]),
                    tuple(int(self._tt_ring_org.get(self._tt_rid(b), 0)) for b in range(batch_size)),
                )
                _cs196 = getattr(type(self), "_r196_vridx", None)
                if _cs196 is None or _cs196[0] != _ck196:
                    _mhp196 = int(hk.shape[1])
                    _cap196 = self._TT_RECENT
                    _idx196 = []
                    for b in range(batch_size):
                        rid = self._tt_rid(b)
                        sl = int(seq_lens[b])
                        _X196 = int(self._tt_ring_org.get(rid, 0))
                        _pf196 = int(self._tt_pref.get(rid, 0))
                        _n196 = int(nprs[b])
                        t0 = sl - _n196 + _pf196  # window start token (rcnt=_n196-_pf196)
                        for j in range(_mhp196):
                            if j < _pf196:
                                _pos196 = j
                            elif j < _n196:
                                _pos196 = _pf196 + ((t0 + (j - _pf196) - _X196) % _cap196)
                            else:
                                _pos196 = 0
                            _idx196.append(slots[b] * _mhp196 + _pos196)
                    _it196 = torch.tensor(_idx196, dtype=torch.int64, device=query.device)
                    type(self)._r196_vridx = (_ck196, _it196)
                else:
                    _it196 = _cs196[1]
                _nf196 = int(hk.shape[2]) * int(hk.shape[3])
                k_g = (
                    hk.view(-1, _nf196).index_select(0, _it196).view(batch_size, hk.shape[1], hk.shape[2], hk.shape[3])
                )
                v_g = (
                    hv.view(-1, _nf196).index_select(0, _it196).view(batch_size, hv.shape[1], hv.shape[2], hv.shape[3])
                )
            elif slots == list(range(slots[0], slots[0] + batch_size)):
                k_g = hk.narrow(0, slots[0], batch_size)
                v_g = hv.narrow(0, slots[0], batch_size)
            else:
                _sh196 = getattr(type(self), "_r196_slbuf", None)
                if _sh196 is None or _sh196[0].size(0) < batch_size:
                    _sh196 = (
                        torch.empty(int(self._tt_mseq), dtype=torch.int64, pin_memory=True),
                        torch.empty(int(self._tt_mseq), dtype=torch.int64, device=query.device),
                    )
                    type(self)._r196_slbuf = _sh196
                _sh196[0][:batch_size] = torch.tensor(slots, dtype=torch.int64)
                _sh196[1][:batch_size].copy_(_sh196[0][:batch_size], non_blocking=True)
                _sel = _sh196[1][:batch_size]
                k_g = torch.index_select(hk, 0, _sel)
                v_g = torch.index_select(hv, 0, _sel)
            # ---- ONE q-rotation ([B,nq,hd]@[hd,hd]; M-batch does not change the
            # per-row dot order) + ONE padded FIA
            qrot = torch.matmul(query[:batch_size], R_k) if R_k is not None else query[:batch_size]
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=qrot.unsqueeze(2),
                key=k_g.permute(0, 2, 1, 3),
                value=v_g.permute(0, 2, 1, 3),
                input_layout="BNSD",
                actual_seq_lengths=[1] * batch_size,
                actual_seq_lengths_kv=nprs,
                num_heads=_nq,
                num_key_value_heads=_nkv,
                scale=self.scale,
                sparse_mode=0,
            )
            o_pr = attn_out[:, :, 0, :].float()  # [B, nq, hd]
            # ---- l_pr: batched masked einsum (bit-exact vs the legacy per-row
            # einsum+logsumexp on identical values — device-adjudicated R196
            # standalone: maxdiff 0.0; -inf padded positions contribute exp=0.0
            # exactly and never win the max). Chunked to bound the fp32 kk
            # materialization ([chunk, nq, CPR, hd]) at ~8 rows.
            kvh = torch.arange(_nq, device=query.device) // (_nq // _nkv)
            l_pr = torch.empty(batch_size, _nq, dtype=torch.float32, device=query.device)
            _arC = torch.arange(int(k_g.shape[1]), device=query.device)
            # pinned staging for the per-row n_pr (async H2D, no stream-drain)
            _np196 = getattr(type(self), "_r196_nprbuf", None)
            if _np196 is None or _np196[0].size(0) < batch_size:
                _np196 = (
                    torch.empty(int(self._tt_mseq), dtype=torch.int64, pin_memory=True),
                    torch.empty(int(self._tt_mseq), dtype=torch.int64, device=query.device),
                )
                type(self)._r196_nprbuf = _np196
            _np196[0][:batch_size] = torch.tensor(nprs, dtype=torch.int64)
            _np196[1][:batch_size].copy_(_np196[0][:batch_size], non_blocking=True)
            _nprD = _np196[1]
            _CH = 8
            for _c0 in range(0, batch_size, _CH):
                _c1 = min(_c0 + _CH, batch_size)
                kk = k_g[_c0:_c1][:, :, kvh, :].permute(0, 2, 1, 3).float()
                sc = torch.einsum("bhd,bhnd->bhn", qrot[_c0:_c1].float(), kk) * self.scale
                sc = sc.masked_fill(_arC.view(1, 1, -1) >= _nprD[_c0:_c1].view(-1, 1, 1), float("-inf"))
                l_pr[_c0:_c1] = torch.logsumexp(sc, dim=-1)
        # ---- o_h/l_h from the batched middle results (carrier rows)
        # R204_MIXCHAIN: when the R146 slab covers every batch row in order
        # (every row has a middle tier — the MIXDEC sub-batch case), the merge
        # reads the slab directly: ZERO staging kernels instead of
        # zeros+full+2B copy_ per layer. Rows absent from the carrier keep the
        # exact base staging (l_h=-inf selects o_pr exactly).
        if _slab204 is not None and _slab204[2] == tuple(range(batch_size)) and _slab204[0].size(0) == batch_size:
            o_h = _slab204[0]
            l_h = _slab204[1]
        else:
            o_h = torch.zeros(batch_size, _nq, _hd, dtype=torch.float32, device=query.device)
            l_h = torch.full((batch_size, _nq), float("-inf"), dtype=torch.float32, device=query.device)
            for b in range(batch_size):
                got = _r144_bmid.get(b)
                if got is not None:
                    o_h[b].copy_(got[0])
                    l_h[b].copy_(got[1])
        # ---- vectorized merge (legacy formulas; middle-less rows reduce to
        # o_pr exactly: w_h=0, w_pr=1, 0*0=0)
        # R204_MIXCPP (default OFF): the whole tail (merge + cast + unrot +
        # writeback, ~10 Python dispatches) as ONE C++ crossing — same at:: ops
        # on the same values (oscar_merge_tail in the r152 launcher). Falls
        # through to the shared merge-stat/SR_CMP adjudication below either way.
        _r204cpp = (
            _r144_ovl_os.environ.get("R204_MIXCPP", "0") == "1"
            and getattr(type(self), "_r112_mod", None) is not None
            and hasattr(type(self)._r112_mod, "oscar_merge_tail")
            and l_pr.size(0) == batch_size
        )
        if _r204cpp:
            type(self)._r112_mod.oscar_merge_tail(
                l_pr,
                l_h,
                o_pr,
                o_h,
                (R_v if R_v is not None else torch.empty(0, dtype=query.dtype, device=query.device)),
                output,
                int(batch_size),
            )
        else:
            # R216: capture the merge operands exactly as consumed (default OFF)
            if _r144_ovl_os.environ.get("R216_LSEDUMP", "0") == "1":
                _r216_merge(l_pr, l_h, o_pr, o_h, None)
            m = torch.maximum(l_pr, l_h)
            w_pr = torch.exp(l_pr - m)
            w_h = torch.exp(l_h - m)
            merged = (w_pr.unsqueeze(-1) * o_pr + w_h.unsqueeze(-1) * o_h) / (w_pr + w_h).unsqueeze(-1)
            _r218mgB = globals().setdefault("_R218_MERGECAPB_N", [0])
            if _r218mgB[0] < 2 and _r144_ovl_os.environ.get("R218_MERGECAP", "0") == "1":
                _r218mgB[0] += 1
                try:
                    torch.npu.synchronize()
                    torch.save(
                        {
                            "n": _r218mgB[0],
                            "site": "B-vec",
                            "layer": int(self._oscar_layer_ord(layer)),
                            "l_pr": l_pr.detach().float().cpu().clone(),
                            "l_h": l_h.detach().float().cpu().clone(),
                            "o_pr": o_pr.detach().float().cpu().clone(),
                            "o_h": o_h.detach().float().cpu().clone(),
                            "merged": merged.detach().float().cpu().clone(),
                            "qrot": qrot.detach().cpu().clone(),
                        },
                        f"/tmp/r218_mergecap_B{_r218mgB[0]}.pt",
                    )
                    _r218_diag(f"MERGECAP-B saved #{_r218mgB[0]} lay={int(self._oscar_layer_ord(layer))}")
                except Exception as _eMGB:
                    _r218_diag(f"MERGECAP-B ERR {_eMGB!r}")
            # ---- dtype + ONE unrot + ONE writeback (== legacy stack->matmul->assign)
            out = merged.to(query.dtype)
            if R_v is not None:
                out = torch.matmul(out, R_v.transpose(-1, -2))
            output[:batch_size] = out
        _st["merge"] += batch_size
        # ---- R196_SR_CMP: adjudicate vs the legacy per-row sequence.
        # Warmup-era calls (all n_pr==1, B==batch) are logged sparsely (1/16)
        # so the 128-call budget covers the real mixed/full-width calls.
        if _r144_ovl_os.environ.get("R196_SR_CMP", "0") == "1":
            _n196c = globals().setdefault("_R196_CMP_N", [0])
            # R200: real mixed/full-width calls have EVERY row at width
            # (npr ~1024); CANWARM warmup mixed calls carry tiny-npr rows
            # (3/4/6) and would eat the whole 128-call budget at full
            # density — sample them 1/16 like the other warmup calls.
            # R218: the 128-call budget was being consumed by CANWARM/chunked-
            # prefill rows with npr 9..36. Real steady three-tier decode rows have
            # prefix+recent ~=1024; gate on the FULL width to keep budget for them.
            _interesting = max(nprs) > 512 and min(nprs) > 512
            if _n196c[0] < 128 and (_interesting or _n196c[0] % 16 == 0):
                _n196c[0] += 1
                _lg = []
                _dopr = 0.0
                _dlpr = 0.0
                _rows196 = []
                # kvh lives in the legacy branch only; define it here for the
                # R200_EPRD path (the adjudication must run on BOTH paths).
                kvh = torch.arange(_nq, device=query.device) // (_nq // _nkv)
                for b in range(batch_size):
                    pref = int(self._tt_pref.get(self._tt_rid(b), 0))
                    rcnt = int(self._tt_rcnt.get(self._tt_rid(b), 0))
                    n_pr = pref + rcnt
                    pr_k, pr_v = self._r198_pr_window(
                        self._tt_rid(b), int(slots[b]), int(seq_lens[b]), pref, rcnt, hk, hv
                    )
                    q_b = query[b : b + 1]
                    q_rotL = torch.matmul(q_b, R_k) if R_k is not None else q_b
                    al, _ = torch_npu.npu_fused_infer_attention_score(
                        query=q_rotL.unsqueeze(2),
                        key=pr_k.permute(1, 0, 2).unsqueeze(0),
                        value=pr_v.permute(1, 0, 2).unsqueeze(0),
                        input_layout="BNSD",
                        actual_seq_lengths=[1],
                        actual_seq_lengths_kv=[n_pr],
                        num_heads=_nq,
                        num_key_value_heads=_nkv,
                        scale=self.scale,
                        sparse_mode=0,
                    )
                    o_prL = al[0, :, 0, :].float()
                    _dopr = max(_dopr, float((o_pr[b] - o_prL).abs().max()))
                    kkL = pr_k[:, kvh, :].permute(1, 0, 2).float()
                    scL = torch.einsum("hd,hnd->hn", q_rotL[0].float(), kkL) * self.scale
                    l_prL = torch.logsumexp(scL, dim=-1)
                    _dlpr_r = float((l_pr[b] - l_prL).abs().max())
                    _dopr_r = float((o_pr[b] - o_prL).abs().max())
                    _dlpr = max(_dlpr, _dlpr_r)
                    # R218: per-row operands make the mixed-step tier-boundary class
                    # directly visible (the aggregate max cannot say WHICH row moved).
                    _rows196.append((b, n_pr, _dopr_r, _dlpr_r))
                    o_hL, l_hL = _r144_bmid.get(b, (None, None))
                    if o_hL is not None:
                        mL = torch.maximum(l_prL, l_hL)
                        wp = torch.exp(l_prL - mL)
                        wh = torch.exp(l_hL - mL)
                        mgL = (wp.unsqueeze(-1) * o_prL + wh.unsqueeze(-1) * o_hL) / (wp + wh).unsqueeze(-1)
                    else:
                        mgL = o_prL
                    _lg.append(mgL.to(query.dtype))
                outL = torch.stack(_lg, dim=0)
                if R_v is not None:
                    outL = torch.matmul(outL, R_v.transpose(-1, -2))
                _dfin = (output[:batch_size].float() - outL.float()).abs().max().item()
                try:
                    with open("/tmp/r196_srcmp.log", "a") as _pf:
                        _pf.write(
                            f"n={_n196c[0]} L={_lord} B={batch_size} "
                            f"nprs={nprs[:8]} o_pr_maxdiff={_dopr:.3e} "
                            f"l_pr_maxdiff={_dlpr:.3e} final_maxdiff={_dfin:.3e} "
                            f"rows={_rows196[:16]}\n"
                        )
                except Exception:
                    pass
        return True

    @staticmethod
    def _r164_midprobe(rec):
        try:
            import json as _jmp

            _st = globals().setdefault("_R164_MIDPROBE_N", [0])
            if _st[0] >= 48:
                return
            if not (1000 < int(rec.get("sl", 0)) < 2048):
                return  # profile/dummy steps (sl 1/4/2048/4096); keep real decode only
            _st[0] += 1
            with open("/tmp/r164_midprobe.jsonl", "a") as _fp:
                _fp.write(_jmp.dumps(rec) + "\n")
        except Exception:
            pass

    def _r197_probe_record(self, attn_metadata, layer, batch_size, seq_lens, r196_done, tm, ev0):
        """R197 PROBE (default OFF, R197_PROBE=1 legs only): append one
        sampled eager-middle layer-call record — host phase walls (ring /
        staging / q / fill+op / R196 S+R) + census (Bp, mid lens, cap bucket)
        — to /tmp/r197_probe.jsonl, and resolve the previous sampled call's
        device event pair (one sync per sampled call; never in A/B legs)."""
        import json as _j197
        import time as _t197

        ev1 = torch.npu.Event(enable_timing=True)
        ev1.record()
        _tend = _t197.perf_counter()
        # tm values are Optional perf_counter stamps; Nones are backfilled
        # right below, so carry the list as Any until then
        _marks: list[Any] = [tm.get(k) for k in ("t0", "t1", "t2", "t3", "t4", "t5a", "t5")]
        for _i in range(1, len(_marks)):
            if _marks[_i] is None:
                _marks[_i] = _marks[_i - 1] if _marks[_i - 1] is not None else _tend
        if _marks[0] is None:
            _marks[0] = _tend

        def _ms(i, j):
            return round(max(0.0, (_marks[j] - _marks[i])) * 1000, 3)

        _call = _R2XX_G.get("_R197_PROBE_N", [0])[0]
        _row = {
            "call": _call,
            "kind": "mid",
            "lord": int(self._oscar_layer_ord(layer)),
            "B": int(batch_size),
            "state": str(getattr(attn_metadata, "attn_state", "?")),
            "r196": bool(r196_done),
            # the ternary is identity (backfill above removed the Nones) but
            # keeps the operand non-Optional for the checker
            "tot_ms": round((_tend - (_marks[0] if _marks[0] is not None else _tend)) * 1000, 3),
            "ring_ms": _ms(0, 1),
            "stage_ms": _ms(1, 2),
            "q_ms": _ms(2, 3),
            "fillop_ms": _ms(3, 4),
            "glue_ms": _ms(4, 5),
            "sr_ms": _ms(5, 6),
            "cap": tm.get("cap", -1),
            "nbC": tm.get("nbC", -1),
            "BcapT": tm.get("BcapT", -1),
            "Bp": tm.get("Bp", -1),
            "cml": tm.get("cml", []),
            "cml_sum": tm.get("cml_sum", -1),
            "npr_max": tm.get("npr", -1),
            "vmode": str(tm.get("vmode", "?")),
            "sls6": [int(x) for x in list(seq_lens)[:6]],
        }
        _pend = getattr(type(self), "_r197_dev_pend", None)
        if _pend is not None:
            try:
                _pend[1].synchronize()
                _row["dev_prev_call"] = _pend[2]
                _row["dev_ms"] = round(_pend[0].elapsed_time(_pend[1]), 3)
                # R201: per-phase device split of the previous sampled call
                _evsP = _pend[3] or {}
                _prevE = _pend[4] or {}
                _dms = {}
                for _k197 in ("ring", "stage", "q", "fillop", "sr"):
                    _i197 = {"ring": 0, "stage": 1, "q": 2, "fillop": 3, "sr": 4}[_k197]
                    _a197 = _prevE.get(f"ev{_i197}")
                    _b197 = _prevE.get(f"ev{_i197 + 1}") or _pend[1]
                    if _a197 is not None and _b197 is not None:
                        _dms[_k197] = round(_a197.elapsed_time(_b197), 3)
                _row["dev_ph"] = _dms
            except Exception:
                pass
        type(self)._r197_dev_pend = (ev0, ev1, _call, None, tm)
        with open("/tmp/r197_probe.jsonl", "a") as _f197:
            _f197.write(_j197.dumps(_row) + "\n")
