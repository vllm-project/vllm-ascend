# SPDX-License-Identifier: Apache-2.0
"""Batch-metadata maintenance mixin: row-slot migration, the flat metadata
fills (mdfill / update_meta), flat refresh and the R196 idfill memo."""

from __future__ import annotations

import contextlib
import os as _r144_ovl_os
from typing import Any

import torch
import torch_npu  # noqa: F401 - registers torch.npu

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.int2.ctx import (  # noqa: F401
    _OSCAR_CFG,
    _R2XX_G,
    get_graph_params,
)
from vllm_ascend.attention.int2.helpers import (  # noqa: F401
    _r166_tr,
    _r199_pool_rowsum,
    _r199_sc_mismatch,
)


class Int2BatchMetaUpdateMixin:
    # Sibling-mixin annotations: _r170_rebind_locked lives on the slot-state
    # mixin and the R196 idfill memo on the staging mixin; bare annotations
    # create no runtime attributes, they pin types for mypy --check-untyped-defs.
    _r170_rebind_locked: Any
    wait_stream: Any
    _r196_idfill_state: Any
    _r196_idfill_state2: Any

    def _r170_rowslot_migrate(cls, update_stream, forward_context, num_tokens):
        """R170_ROWSLOT pre-replay migration (default OFF; called from
        _update_graph_params_r156 under the flag).

        Full-decode graphs baked hk[row] at capture. Before each pure-decode
        replay (num_prefills==0; N <= num_tokens covers padded buckets), move
        ring content so the request at row b lives at slot b:
          evicts: any live NON-decoding owner of a row/pad/scratch slot moves
                  to a free slot first (pad rows N..Bcap bake live ring
                  slides; slot cap-1 is the reserved scratch when Bcap < cap)
          permutation: movers direct-move in dependency order; ONE stash per
                  dependency cycle (temps = #cycles, not #movers — tight-cap
                  swaps must not exhaust the free pool; host smoke-tested in
                  RAW/r170_migrate_smoke2.py)
        A plan-building simulation mirrors the allocator maps so every step's
        precondition is verified against the exact intermediate state before
        any device copy is enqueued; the real rebinds then commit in the same
        order (guards re-check). All copies enqueue on update_stream (ordered
        after eager prefill writes and the previous replay by R161_FENCE
        F1/F3). Fast path: zero copies when already aligned.
        Returns the number of ring moves (0 = already aligned)."""
        import os as _os170m

        try:
            if torch.npu.is_current_stream_capturing():
                # R170h5b: capture-time md is SYNTHETIC (placeholder rids,
                # duplicates legal) — migrating against it livelocks the plan
                # walk (mf_r170p3 B>1 warmup crash). Capture-side slot
                # binding is already correct via _tt_slot's slot=row bake.
                return 0
        except Exception:
            pass
        A = getattr(cls, "_r166_alloc", None)
        hpk = getattr(cls, "_r170_hpk", None)
        hpv = getattr(cls, "_r170_hpv", None)
        if A is None or not hpk:
            # INT2 impl binds through its concrete class; resolve and retry.
            _sub = globals().get("AscendINT2AttentionBackendImpl")
            if _sub is not None and _sub is not cls:
                A = A or getattr(_sub, "_r166_alloc", None)
                hpk = hpk or getattr(_sub, "_r170_hpk", None)
                hpv = hpv or getattr(_sub, "_r170_hpv", None)
        # R171 FIX (defect B): union the rings across ALL impl instances via
        # the registry — the single legacy alias holds only the LAST
        # instance's dict (one lid), which made _cp copy one layer and left
        # the other N-1 layers' row slots holding the previous generation's
        # window (foreign-KV walk). Same lid in two registry dicts = two live
        # rings for one layer ordinal -> fail-closed (allocator/impl
        # inconsistency), never silently pick one.
        _reg171 = getattr(cls, "_r171_hp_reg", None) or getattr(
            globals().get("AscendINT2AttentionBackendImpl"), "_r171_hp_reg", None
        )
        if _reg171:
            # R171_HPREG_MAX: bisector cap - copy only the first N layer
            # ordinals (sorted). N=1 reproduces the legacy single-ring
            # behavior through the new code path (hang discriminator).
            try:
                _mx171 = int(_os170m.environ.get("R171_HPREG_MAX", "0"))
            except Exception:
                _mx171 = 0
            if _mx171 > 0:
                _ks171 = sorted({_l for _dk, _dv in _reg171 for _l in _dk})
                _keep171 = set(_ks171[:_mx171])
            else:
                _keep171 = None
            _uk = {}
            for _dk, _dv in _reg171:
                for _lid, _rk in _dk.items():
                    if _keep171 is not None and _lid not in _keep171:
                        continue
                    if _lid in _uk:
                        raise RuntimeError(
                            "[R171-HPREG] layer ordinal {!r} registered twice (two live rings — fail-closed)".format(
                                _lid
                            )
                        )
                    _uk[_lid] = (_rk, _dv.get(_lid))
            if _uk:
                hpk = {_l: _t[0] for _l, _t in _uk.items()}
                hpv = {_l: _t[1] for _l, _t in _uk.items()}
        if A is None or not hpk:
            return 0
        md = None
        _amd = getattr(forward_context, "attn_metadata", None) or {}
        for _k in _amd:
            md = _amd[_k]
            break
        rids = list(getattr(md, "request_ids", None) or [])
        if not rids:
            return 0
        if int(getattr(md, "num_prefills", 0) or 0) != 0:
            return 0  # mixed/prefill step: FULL graphs never replay these
        N = len(rids)
        Bcap = int(num_tokens)
        if Bcap < N:
            return 0  # rids exceed graph rows: not a decode replay of this md
        if Bcap > A.cap:
            raise RuntimeError(f"[R170-ROWSLOT] graph rows {Bcap} exceed ring capacity {A.cap} (fail-closed)")
        _log170 = _os170m.environ.get("R170_ROWSLOT_LOG", "0") == "1"
        _nmove = 0
        with A._lk:
            r2s, s2r = A.rid2slot, A.slot2rid
            rowset = set(rids)
            movers = []  # (rid, cur, row)
            for b, r in enumerate(rids):
                cur = r2s.get(r)
                if cur is not None and cur != b:
                    movers.append((r, cur, b))
            if len(set(rids)) != N:
                raise RuntimeError(
                    "[R170-ROWSLOT] duplicate rids in decode batch (fail-closed) rids={!r}".format(rids[:16])
                )
            targets = {b for (_r, _c, b) in movers}
            pad_range = set(range(N, Bcap))
            scratch = {A.cap - 1} if Bcap < A.cap else set()
            _touched = targets | pad_range | scratch
            # Plan-building simulation: mirror of the allocator maps that
            # every planned step mutates as it is appended, so each step's
            # precondition is verified against the exact intermediate state.
            sim_free = set(s for s in A.free if s not in _touched)
            sim_r2s = dict(r2s)
            sim_s2r = dict(s2r)
            plan = []  # ("mv", rid, cur, dst) in copy/commit order

            def _sim_rebind(rid, cur, dst):
                if sim_s2r.get(dst) is not None:
                    raise RuntimeError(f"[R170-ROWSLOT] plan step targets occupied slot {dst} (fail-closed)")
                sim_free.discard(dst)
                del sim_s2r[cur]
                sim_r2s[rid] = dst
                sim_s2r[dst] = rid
                sim_free.add(cur)

            # 1) evict live NON-decoding owners off touched slots first
            #    (their vacated slots then serve as direct-move targets).
            # R172 FIX (B32@4K livelock): eviction/stash targets must come
            # from slots OUTSIDE _touched. min(sim_free) could pick a row
            # slot just vacated by a previous eviction (rebind re-adds the
            # vacated cur), so eviction chains cascaded INTO rows 0..N-1,
            # re-occupying the decoding movers' targets — the round-robin
            # stash then rotated movers through the row pool forever
            # (r172 i2_4096_p1: plan loop exceeded bound, 35 live rings,
            # 20 movers, rows free-then-refilled drum). Picking outside
            # _touched keeps rows monotonically freed for the direct moves.
            def _pick_free_outside():
                # R172: priority 1 = free slot not a mover TARGET (safe park,
                # bounded stash debt); priority 2 = outside the whole touched
                # range; None only when every free slot is a mover target.
                _c = [s for s in sim_free if s not in targets]
                if _c:
                    return min(_c)
                _c = [s for s in sim_free if s not in _touched]
                return min(_c) if _c else None

            for s in sorted(_touched):
                x = sim_s2r.get(s)
                if x is None or x in rowset:
                    continue  # decoding owner is itself a mover
                t = _pick_free_outside()
                if t is None:
                    t = min(sim_free) if sim_free else None
                if t is None:
                    raise RuntimeError(
                        f"[R170-ROWSLOT] no free slot to evict live owner {x!r} "
                        f"off slot {s} (cap={A.cap}, free={len(A.free)}; fail-closed)"
                    )
                sim_free.discard(t)
                plan.append(("mv", x, s, t))
                _sim_rebind(x, s, t)
            # 2) in-place permutation: direct-move every mover whose row slot
            #    is free in the sim; stash ONE mover per dependency cycle to
            #    a free slot, then re-scan. temps = #cycles, not #movers
            #    (tight-cap swaps must not exhaust the free pool).
            pending = [list(m) for m in movers]
            _guard = 0
            _stash_i = 0
            while pending:
                _prog = True
                while _prog:
                    _prog = False
                    _i = 0
                    while _i < len(pending):
                        _r, _c, _b = pending[_i]
                        if sim_s2r.get(_b) is None:
                            plan.append(("mv", _r, _c, _b))
                            _sim_rebind(_r, _c, _b)
                            pending.pop(_i)
                            _prog = True
                        else:
                            _i += 1
                if pending:
                    # R170h5 fix: round-robin the stash across pending movers.
                    # Always stashing pending[0] livelocks when [0]'s target is
                    # held by a settled occupant (e.g. duplicate rid at two
                    # rows); rotating guarantees every cycle head gets stashed.
                    _si = _stash_i % len(pending)
                    _stash_i += 1
                    _r, _c, _b = pending[_si]
                    # R172 FIX: stash OUTSIDE _touched (see eviction note) —
                    # stashing into a vacated row slot re-occupies a mover
                    # target and feeds the rotation drum. _pick_free_outside
                    # already prefers non-target slots.
                    _t = _pick_free_outside()
                    if _t is None:
                        _t = min(sim_free) if sim_free else None
                    if _t is None:
                        raise RuntimeError(
                            f"[R170-ROWSLOT] no free temp slot for mover {_r!r} "
                            f"(cap={A.cap}, free={len(A.free)}; fail-closed)"
                        )
                    sim_free.discard(_t)
                    plan.append(("mv", _r, _c, _t))
                    _sim_rebind(_r, _c, _t)
                    pending[_si][1] = _t
                _guard += 1
                if _guard > 4 * (N + 1):
                    try:
                        import json as _j170d

                        _j170d.dump(
                            {
                                "rids": [str(x) for x in rids],
                                "r2s": {str(k): v for k, v in r2s.items()},
                                "s2r": {str(k): str(v) for k, v in s2r.items()},
                                "free": sorted(A.free),
                                "cap": A.cap,
                                "touched": sorted(_touched),
                                "Bcap": Bcap,
                                "pending": [list(x) for x in pending],
                                "plan": [[m[0], m[1], m[2], m[3]] for m in plan],
                            },
                            open("/tmp/r170_rowslot_debug.json", "w"),  # noqa: SIM115 - fire-and-forget diagnostic write
                        )
                    except Exception:
                        pass
                    raise RuntimeError(
                        "[R170-ROWSLOT] plan loop exceeded bound "
                        "(fail-closed; state dumped to /tmp/r170_rowslot_debug.json)"
                    )
            for b, r in enumerate(rids):
                if sim_r2s.get(r) != b:
                    raise RuntimeError(
                        f"[R170-ROWSLOT] plan failed to align rid {r!r} at row "
                        f"{b} (got {sim_r2s.get(r)!r}; fail-closed)"
                    )
            if not plan:
                return 0
            if hpv is None:
                hpv = {}

            def _cp(dst, src):
                for _lid in hpk:
                    _hk = hpk.get(_lid)
                    if _hk is None:
                        continue
                    _hv = hpv.get(_lid)
                    _hk[dst].copy_(_hk[src])
                    if _hv is not None:
                        _hv[dst].copy_(_hv[src])

            with torch.npu.stream(update_stream):
                for _k170, _r, _c, _d in plan:
                    _cp(_d, _c)
            # commit the real map rebinds in the SAME order; every guard
            # re-checks against the exact state the sim verified.
            for _k170, _r, _c, _d in plan:
                cls._r170_rebind_locked(A, _r, _d)
            _sm = getattr(cls, "_r164_slotmap", None)
            if _sm is None:
                _sub2 = globals().get("AscendINT2AttentionBackendImpl")
                _sm = getattr(_sub2, "_r164_slotmap", None) if _sub2 else None
            if _sm is not None:
                for _k170, _r, _c, _d in plan:
                    _sm[_r] = _d
            _nmove = len(plan)
            if _log170:
                try:
                    with open("/tmp/r170_rowslot.jsonl", "a") as _f170:
                        import json as _j170

                        _f170.write(
                            _j170.dumps(
                                {
                                    "n": _nmove,
                                    "Bcap": Bcap,
                                    "N": N,
                                    "plan": [[str(_r), _c, _d] for (_k, _r, _c, _d) in plan],
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
        return _nmove

    def _r172_wrapper_flatrefresh(cls, update_stream, forward_context, num_tokens):
        """R172_FLATREFRESH (default OFF): defect-A cure at the TRUE fill
        point under FULL mixed graphs.

        R169_FLATFIX never fires there — its trigger is an eager
        Prefill-state forward, which does not exist when prefills are
        captured in-graph (r172b: /tmp/r169_flatfix.log ABSENT; census 14
        empties at row0/step0 L0-L13). Consequence: the FIRST decode replay
        of generation 1 reads CAPTURE-ZERO flats (empty keys), and the first
        decode replay of generation N>1 reads generation N-1's fills (STALE
        mid window) — the rows1-7 diverge@5 vs row0 diverge@20 residual.

        This refresh runs in the acl_graph wrapper's post-need_sync
        pre-replay window (device quiescent, same stream the replay uses —
        the site R171 validated for the ring migrate). It fires ONLY when
        the decoding request-set changes (generation boundary), and writes
        the same flat set the prefill-side FLATFIX would have: slot/idx_sr/
        p_src/p_dst/p_new (ring geometry, post-migrate slots) + askv +
        bt_mirror (mid window), with geometry from the bucket's captured
        _r164_baked_geom. Returns number of refreshed buckets (0 = skipped).
        """
        import os as _os172f

        try:
            if _os172f.environ.get("R172_FLATREFRESH", "0") != "1":
                return 0
            if _os172f.environ.get("R164_BATCHED_STAGE", "0") != "1":
                return 0
            _sub = globals().get("AscendINT2AttentionBackendImpl")
            _mp = getattr(cls, "_r164_meta", None) or (getattr(_sub, "_r164_meta", None) if _sub else None)
            _bg = getattr(cls, "_r164_baked_geom", None) or (getattr(_sub, "_r164_baked_geom", None) if _sub else None)
            if not _mp or not _bg:
                return 0
            md = None
            _amd = getattr(forward_context, "attn_metadata", None) or {}
            for _k in _amd:
                md = _amd[_k]
                break
            rids = list(getattr(md, "request_ids", None) or [])
            if not rids:
                return 0
            if int(getattr(md, "num_prefills", 0) or 0) != 0:
                return 0  # mixed/prefill step: FULL graphs never replay these
            N = len(rids)
            Bcap = int(num_tokens)
            if Bcap not in _mp or Bcap < N:
                return 0
            # fire only at request-set boundaries (generation start/finish)
            _sig = (Bcap, tuple(sorted(str(r) for r in rids)))
            _prev = getattr(cls, "_r172_fr_prev", None)
            if _prev is None:
                _prev = {}
                cls._r172_fr_prev = _prev
            if _prev.get(Bcap) == _sig:
                return 0
            _bgF = _bg.get(Bcap)
            if _bgF is None:
                return 0  # baked geometry not recorded (needs FLATFIX=1 on)
            if int(_bgF.get("bsz", 0) or 0) <= 0 and getattr(md, "block_tables", None) is not None:
                return 0  # page size unknown: no partial refresh (fail-soft)
            _meI = _mp[Bcap]
            _sm = getattr(cls, "_r164_slotmap", None) or (getattr(_sub, "_r164_slotmap", None) if _sub else None) or {}
            _slI = [_sm.get(r, b) for b, r in enumerate(rids)]
            _sls = list(getattr(md, "seq_lens_list", None) or [])
            _bt = getattr(md, "block_tables", None)
            with torch.npu.stream(update_stream):
                _slT = torch.tensor(_slI, dtype=torch.int64, device=_meI["slot"].device)
                _meI["slot"].copy_(_slT)
                _mhp = int(_meI.get("mhp", _meI.get("npr", 1)))
                _tc = _meI.get("t_const")
                if _tc is None or _tc.device != _slT.device:
                    _tc = torch.arange(int(_meI["npr"]), dtype=torch.int64, device=_slT.device)
                _meI["idx_sr"].copy_((_slT.view(-1, 1) * _mhp + _tc.view(1, -1)).reshape(-1))
                _prefF = int(_meI["prefp"])
                _capF = int(_meI["capw"])
                _tw1 = _meI.get("t_w1")
                if _tw1 is None or _tw1.device != _slT.device:
                    _tw1 = torch.arange(_prefF + 1, _prefF + _capF, dtype=torch.int64, device=_slT.device)
                _tw0 = _meI.get("t_w0")
                if _tw0 is None or _tw0.device != _slT.device:
                    _tw0 = torch.arange(_prefF, _prefF + _capF - 1, dtype=torch.int64, device=_slT.device)
                _meI["p_src"].copy_((_slT.view(-1, 1) * _mhp + _tw1.view(1, -1)).reshape(-1))
                _meI["p_dst"].copy_((_slT.view(-1, 1) * _mhp + _tw0.view(1, -1)).reshape(-1))
                _meI["p_new"].copy_(_slT * _mhp + (_prefF + _capF - 1))
                # mid window: askv + bt_mirror with the baked geometry
                _msF = int(_bgF["mid_start"])
                _rcF = int(_bgF["rcnt"])
                _nbF = int(_meI["nb_max"])
                _avN = int(_meI["askv"].shape[0])
                _avF = [0] * _avN
                for _rF in range(min(len(_sls), _avN)):
                    _mlF = int(_sls[_rF]) - _rcF - _msF
                    _avF[_rF] = _mlF if _mlF > 0 else 0
                _meI["askv"].copy_(torch.tensor(_avF, dtype=torch.int64, device=_meI["askv"].device))
                if _bt is not None:
                    _bszF = int(_bgF.get("bsz", 0) or 0)
                    _widF = int(_bt.shape[1])
                    _loF = _msF // _bszF if _bszF > 0 else 0
                    _ncF = 0
                    if 0 <= _loF < _widF:
                        _ncF = min(_nbF, _widF - _loF)
                        _meI["bt_mirror"][:, :_ncF].fill_(0)
                        _acF = min(len(_sls), int(_meI["bt_mirror"].shape[0]))
                        _meI["bt_mirror"][0:_acF, 0:_ncF].copy_(_bt[0:_acF, _loF : _loF + _ncF])
                        _meI["bt_mirror"].clamp_(min=0)
            _prev[Bcap] = _sig
            _frN = globals().setdefault("_R172_FR_N", [0])
            if _frN[0] < 32:
                _frN[0] += 1
                import json as _j172f

                with open("/tmp/r172_flatrefresh.log", "a") as _f172:
                    _f172.write(
                        _j172f.dumps(
                            {
                                "n": _frN[0],
                                "bk": Bcap,
                                "N": N,
                                "slots": _slI[:8],
                                "askv": _avF[:8],
                                "mid_start": _msF,
                                "rcnt": _rcF,
                            }
                        )
                        + "\n"
                    )
            return 1
        except Exception as _e172f:
            _frE = globals().setdefault("_R172_FR_E", [0])
            if _frE[0] < 8:
                _frE[0] += 1
                print("[R172_FLATREFRESH_ERR] {!r}".format(_e172f), flush=True)
            return 0

    def _r170_mdfill(update_stream, forward_context, num_tokens):
        """R170_MDFILL (default OFF; driver calls at bucket-first replay):
        PRE-FILL the layer-invariant metadata flat (bt-mirror, askv, slot,
        asq, idx_sr, ring-slide indices) with THIS step's real values,
        WITHOUT recording E_inputs_ready. Defect-A root: the standard
        post-replay update order means a bucket-first replay reads a flat
        whose fills are not yet enqueued (captured wait no-ops on a
        never-recorded event) => lse=-9999/out=0 empty layer prefix
        (12-19 layers, timing-dependent). Pre-filling real values makes the
        first replay correct by construction; the standard update still
        runs post-replay (fills again + records; lifecycle unchanged).
        Fence: R161 on_update_enter orders update_stream after the eager
        prefill writers (F1) and previous replay (F3) — same edges the
        standard update path applies before touching the flat."""
        # R218 STREAM MODE (default "cur"): enqueue the pre-fill on the CALLER's
        # current stream (== the stream the replay is about to be launched on),
        # so the fills are FIFO-ordered before the replay by construction and
        # NO fence event is involved. The old "upd" mode filled on update_stream
        # behind an on_update_enter wait; walking the live deadlock (py-spy:
        # host parked in _quantize_kv_to_int2's torch.npu.synchronize, the
        # device parked on update_stream) showed that path can leave
        # update_stream waiting on an event the (now absent) replay never
        # records -- the MDFILL warm0->warm1 hang. "upd" kept for A/B.
        if _r144_ovl_os.environ.get("R170_MDFSTREAM", "cur") == "upd":
            # the R161 fence hook is injected via the shared registry by the
            # backend assembly (the monolith resolved the same name from
            # attention_v1's module globals); absent -> no fence, matching the
            # monolith's no-op SimpleNamespace stub
            _r161_f = _R2XX_G.get("_r161_f")
            if _r161_f is not None:
                _r161_f.on_update_enter(update_stream, forward_context)
            _mds170 = update_stream
        else:
            _mds170 = torch.npu.current_stream()
        gp = get_graph_params()
        ap = gp.attn_params.get(num_tokens)
        if not ap or len(ap) == 0 or ap[0][0] != "OSCAR_MID_R164":
            return False
        p0 = ap[0]
        md = None
        for _k in forward_context.attn_metadata:
            md = forward_context.attn_metadata[_k]
            break
        if md is None:
            return False
        p0[21]._r164_update_meta(
            _mds170, md, p0[21], int(p0[17]), int(p0[18]), int(p0[19]), num_tokens, record_event=False
        )
        return True

    def _r164_update_meta(update_stream, md, impl, Bcap, mid_start, rcnt, num_tokens, record_event=True):
        type(impl)._r168_impl_ref = impl  # R168: instance ref for runner-level device reads
        """R164 BATCHED_STAGE replay driver: refresh the layer-invariant device
        metadata ONCE per step on the update stream and record the single
        shared E_inputs_ready (all per-layer captured waits reference it).

        Per-step device work (total, ALL layers): bt-mirror copy+clamp from the
        live md.block_tables (device->device; NULL/-1 padding clamped to block
        0, which only feeds rows beyond askv -> never read), askv copy from a
        ping-pong pinned buffer (host mid_len list), asq fill(1), ONE event
        record. No H2D per layer, no D2H, no host synchronize."""
        import os as _os

        Bcap = int(Bcap)
        _mp = getattr(type(impl), "_r164_meta", None)
        _me = _mp.get(Bcap) if _mp else None
        if _me is None:
            raise RuntimeError(f"[R164] metadata pool missing Bcap={Bcap} (fail-closed)")
        nb_max = int(_me["nb_max"])
        block_size = int(impl.vllm_config.cache_config.block_size)
        blk_lo = int(mid_start) // block_size
        _bt = md.block_tables
        if _bt.shape[1] < blk_lo + nb_max:
            raise RuntimeError(
                f"[R164] block_tables width {tuple(_bt.shape)} < required {blk_lo + nb_max} (fail-closed)"
            )
        _rows = int(_bt.shape[0])
        sl_list = md.seq_lens_list
        _pin = _me["askv_pin"][_me["ping"] & 1]
        _pinS = _me["slot_pin"][_me["ping"] & 1]
        _me["ping"] = _me["ping"] + 1
        for _i in range(Bcap):
            if _i < len(sl_list):
                _ml = int(sl_list[_i]) - int(rcnt) - int(mid_start)
                _pin[_i] = _ml if _ml > 0 else 0
            else:
                _pin[_i] = 0
        # Slot source of truth = THIS STEP's request_ids (row order), NOT the
        # impl's _tt_cur_slots — during pure-graph replays no eager forward
        # runs and the instance state is stale from the last mixed step (the
        # graph then reads never-written rings -> [261,0,0,...] degenerate).
        _smap = getattr(type(impl), "_r164_slotmap", None) or {}
        _srids = getattr(md, "request_ids", None) or []
        # R166 PADFIX: absent rows (Bactive < Bcap) previously fell back to
        # slot=row-index — a LIVE ring under the allocator. The captured
        # in-graph slide runs for ALL Bcap rows, so a pad row's slide WROTE a
        # live request's ring every padded replay (the all-rows-wrong
        # padded-bucket corruption, E29/E34). Point pad rows at the LAST
        # reserved scratch slot (never allocated to any request).
        _scratch = int(impl._tt_mseq) - 1 if getattr(impl, "_tt_mseq", 0) else Bcap
        for _i in range(Bcap):
            _rs = _srids[_i] if _i < len(_srids) else None
            _pinS[_i] = int(_smap[_rs]) if (_rs is not None and _rs in _smap) else _scratch
        _ev = type(impl)._r164_events.get(num_tokens)
        if _ev is None:
            raise RuntimeError(f"[R164] shared event missing nt={num_tokens} (fail-closed)")
        # R218 GRAPH-PATH MIDDLE OPERAND/OUTPUT CAPTURE (DIAGNOSTIC_ONLY,
        # default OFF). Runs at the TOP of update_meta: with ISTAGE=0 the
        # carriers still hold the PREVIOUS replay's in-graph-staged operands and
        # the op pool holds that replay's out/lse. Saved together with a fresh
        # gather of the same mirror so offline analysis can (a) verify the
        # staged bytes equal a live gather and (b) adjudicate the V1 launch
        # against an fp32 rebuild of exactly these operands.
        if _os.environ.get("R218_CARDMP", "0") == "1":
            try:
                _cd = globals().setdefault("_R218_CD_N", [0])
                _cdmax = int(_os.environ.get("R218_CARDMP_MAX", "8"))
                _cdlay = int(_os.environ.get("R218_CARDMP_LID", "0"))
                # R218_CARDMP_SKIP (default 0): first step index to dump, so a
                # leg can target the mid-decode steps (e.g. first-divergence)
                # instead of always step 0..MAX.
                _cdskip = int(_os.environ.get("R218_CARDMP_SKIP", "0"))
                # R218_CARDMP_STAG (default OFF): one line per update_meta
                # call (per decode step) with the live seq-len vector — the
                # admission-schedule map for targeting SKIP at a specific
                # leg/row transition (the graph fn itself only runs at
                # capture, so it cannot log per replay step).
                if _os.environ.get("R218_CARDMP_STAG", "0") == "1":
                    try:
                        with open(f"/tmp/r218_cardstag_{_os.environ.get('R218_BRANCH_TAG', 'x')}.log", "a") as _fs:
                            _fs.write(
                                f"[ST] step={int(_cd[0])} Bcap={int(Bcap)} mid={int(mid_start)} "
                                f"rcnt={int(rcnt)} sl={[int(x) for x in (sl_list or [])][: int(Bcap)]}\n"
                            )
                    except Exception:
                        pass
                if _cdskip <= _cd[0] < _cdskip + _cdmax:
                    _cdir = _os.environ.get("R218_CARDMP_DIR", "/tmp/r218_card")
                    _os.makedirs(_cdir, exist_ok=True)
                    _cpC = getattr(type(impl), "_r113_cpool_batched", None) or {}
                    _opC = getattr(type(impl), "_r113_opool_batched", None) or {}
                    _cC = _cpC.get((_cdlay, int(Bcap)))
                    if _cC is not None:
                        _mfC = _me["bt_mirror_flat"]
                        _snC = {
                            "step": int(_cd[0]),
                            "Bcap": int(Bcap),
                            "nb_max": int(nb_max),
                            "blk_lo": int(blk_lo),
                            "mid_start": int(mid_start),
                            "rcnt": int(rcnt),
                            "sl_host": [int(x) for x in (sl_list or [])][: int(Bcap)],
                            "pin_askv": [int(x) for x in _pin.tolist()],
                            "askv_dev": _me["askv"].detach().cpu().clone(),
                            "slot": _me["slot"].detach().cpu().clone(),
                            "bt_mirror": _me["bt_mirror"].detach().cpu().clone(),
                            "bt": _cC["bt"].detach().cpu().clone(),
                            # op geometry, for offline fp32 adjudication
                            "nq": int(impl.num_heads),
                            "nkv": int(impl.num_kv_heads),
                            "hd": int(impl.head_size),
                            "bs": int(block_size),
                            "scale": 1.0 / float(int(impl.head_size)) ** 0.5,
                        }
                        for _nmC in ("k", "v", "ks", "ko", "vs", "vo"):
                            _snC[_nmC] = _cC[_nmC].detach().cpu().clone()
                        # R218: middle Q (graph-pool content of the PREVIOUS
                        # replay — the same replay that produced out_h/lse_h).
                        _cqS = getattr(type(impl), "_r218_cardq", None) or {}
                        _qSC = _cqS.get(int(_cdlay))
                        if _qSC is not None:
                            with contextlib.suppress(Exception):
                                _snC["q"] = _qSC.detach().cpu().clone()
                        _snC["gather_k"] = impl.key_cache.index_select(0, _mfC).detach().cpu().clone()
                        _snC["gather_v"] = impl.value_cache.index_select(0, _mfC).detach().cpu().clone()
                        _snC["gather_ks"] = impl.k_scale_cache.index_select(0, _mfC).detach().cpu().clone()
                        _snC["gather_vs"] = impl.v_scale_cache.index_select(0, _mfC).detach().cpu().clone()
                        _okC = (_cdlay, int(impl.num_heads), int(impl.head_size), str(_me["askv"].device), int(Bcap))
                        if _okC in _opC:
                            _oS, _lS, _ = _opC[_okC]
                            _snC["out_h"] = _oS.detach().cpu().clone()
                            _snC["lse_h"] = _lS.detach().cpu().clone()
                        import torch as _tC

                        # R218 V1RELAY (DIAGNOSTIC_ONLY, default OFF):
                        # re-issue launch_lse_v1_own EAGERLY on the EXACT
                        # same carrier/q/askv tensors the replayed graph used,
                        # into scratch out/lse. card_lse vs relay_lse
                        # bit-compares replay vs eager for the SAME inputs
                        # (isolates kernel-replay divergence from
                        # reference-rebuild artifacts such as q-stash
                        # aliasing).
                        if _os.environ.get("R218_V1RELAY", "0") == "1":
                            try:
                                _cqS2 = getattr(type(impl), "_r218_cardq", None) or {}
                                _qS2 = _cqS2.get(int(_cdlay))
                                _v1m = getattr(type(impl), "_r135_mod", None)
                                if _qS2 is not None and _v1m is not None:
                                    _rp = getattr(type(impl), "_r218_relaypool", None)
                                    if _rp is None:
                                        type(impl)._r218_relaypool = _rp = {}
                                    _rkey = (int(Bcap), int(impl.num_heads), int(impl.head_size))
                                    if _rkey not in _rp:
                                        _rp[_rkey] = (
                                            _tC.zeros(
                                                Bcap,
                                                int(impl.num_heads),
                                                1,
                                                int(impl.head_size),
                                                dtype=_tC.bfloat16,
                                                device=_me["askv"].device,
                                            ),
                                            _tC.zeros(
                                                Bcap,
                                                int(impl.num_heads),
                                                1,
                                                dtype=_tC.float32,
                                                device=_me["askv"].device,
                                            ),
                                        )
                                    _roS, _rlS = _rp[_rkey]
                                    _frr = _v1m.launch_lse_v1_own(
                                        _qS2,
                                        _cC["k"],
                                        _cC["v"],
                                        _cC["bt"],
                                        _cC["ks"],
                                        _cC["ko"],
                                        _cC["vs"],
                                        _cC["vo"],
                                        _roS,
                                        _rlS,
                                        _me["asq"],
                                        _me["askv"],
                                        None,
                                        int(impl.num_heads),
                                        1.0 / float(int(impl.head_size)) ** 0.5,
                                        int(impl.num_kv_heads),
                                        "BNSD",
                                        int(block_size),
                                        1,
                                        1,
                                        _tC.zeros(0, dtype=_tC.uint8, device=_me["askv"].device),
                                        0,
                                        0,
                                        0,
                                        True,
                                    )
                                    _snC["relay_rc"] = [int(x) for x in _frr]
                                    _snC["relay_lse"] = _rlS.detach().cpu().clone()
                                    _snC["relay_out"] = _roS.detach().cpu().clone()
                                    # R218 FRELAY: same operands through the
                                    # VALIDATED launch_lse_fast (r216: dlse
                                    # 7.8e-4 with call-site-captured q) —
                                    # v1-vs-fast on identical tensors decides
                                    # whether the v1 kernel is wrong or the
                                    # offline reference's q is untrusted.
                                    if _os.environ.get("R218_FRELAY", "0") == "1":
                                        try:
                                            _f112 = getattr(type(impl), "_r112_mod", None)
                                            if _f112 is not None:
                                                _askvL = [int(x) for x in _me["askv"].detach().cpu().tolist()]
                                                _roS.zero_()
                                                _rlS.fill_(-1e30)
                                                _fr2 = _f112.launch_lse_fast(
                                                    _qS2,
                                                    _cC["k"],
                                                    _cC["v"],
                                                    _cC["bt"],
                                                    _cC["ks"],
                                                    _cC["ko"],
                                                    _cC["vs"],
                                                    _cC["vo"],
                                                    _roS,
                                                    _rlS,
                                                    int(impl.num_heads),
                                                    1.0 / float(int(impl.head_size)) ** 0.5,
                                                    int(impl.num_kv_heads),
                                                    "BNSD",
                                                    int(block_size),
                                                    1,
                                                    1,
                                                    _askvL,
                                                    _tC.zeros(0, dtype=_tC.uint8, device=_me["askv"].device),
                                                    0,
                                                    0,
                                                    0,
                                                    True,
                                                )
                                                _snC["frelay_rc"] = [int(x) for x in _fr2]
                                                _snC["frelay_lse"] = _rlS.detach().cpu().clone()
                                                _snC["frelay_out"] = _roS.detach().cpu().clone()
                                        except Exception as _eF:
                                            _snC["frelay_err"] = repr(_eF)
                            except Exception as _eR:
                                with contextlib.suppress(Exception):
                                    _snC["relay_err"] = repr(_eR)
                        _tC.save(_snC, f"{_cdir}/card_{int(_cd[0]):04d}.pt")
                # R218 FIX: increment EVERY call (was inside the skip window,
                # so any SKIP>0 froze the counter at 0 and never dumped).
                _cd[0] += 1
            except Exception:
                pass
        if _os.environ.get("R164_MIDPROBE", "0") == "1":
            # paged-cache liveness probe: are the middle blocks nonzero?
            try:
                import json as _jmp2

                _st2 = globals().setdefault("_R164_MIDPROBE2_N", [0])
                if _st2[0] < 400:
                    _st2[0] += 1
                    _btv = _bt[0, max(0, blk_lo) : blk_lo + 3].tolist()
                    _sums: list[Any] = []
                    for _bi in _btv:
                        try:
                            _sums.append(round(float(impl.key_cache[int(_bi)].to(torch.int16).sum().item()), 1))
                        except Exception:
                            _sums.append(None)
                    _askv_dev = _me["askv"].detach().to("cpu").tolist()
                    _ringsums = None
                    _ringb = None
                    try:
                        _hk0 = next(iter(impl._tt_hp_k.values()))
                        _nprR = int(_hk0.shape[1])
                        _sl = [int(x) for x in _pinS.tolist()]
                        _ringsums = [round(float(_hk0[s, :_nprR].float().sum().item()), 1) for s in _sl]
                        # per-block sums of the first 4 recent blocks + last 2
                        # blocks (16-row granularity, block_size-aligned)
                        _pf = _OSCAR_CFG.prefix_window
                        _bsz = 16
                        # tail-64 audit: the last 4 blocks = the 64 most
                        # recent slide appends (decode tokens). A contaminated
                        # static-input buffer (mixed-chunk K/V) shows here.
                        _ringb = [[round(float(_hk0[s, _nprR - 64 : _nprR].float().sum().item()), 1)] for s in _sl]
                        if False:
                            _unused = 0
                    except Exception:
                        pass
                    _host_btv = None
                    _host_sums: list[Any] | None = None
                    _hb = getattr(type(impl), "_r166_host_bt", None)
                    if _hb is not None:
                        _host_btv = [int(x) for x in _hb[0, blk_lo : blk_lo + 3].tolist()]
                        _host_sums = []
                        for _bi in _host_btv:
                            try:
                                _host_sums.append(
                                    round(float(impl.key_cache[int(_bi)].to(torch.int16).sum().item()), 1)
                                )
                            except Exception:
                                _host_sums.append(None)
                    with open("/tmp/r164_midprobe.jsonl", "a") as _fp2:
                        _fp2.write(
                            _jmp2.dumps(
                                {
                                    "stage": "updater_meta",
                                    "Bcap": Bcap,
                                    "bt3": _btv,
                                    "kc_sums": _sums,
                                    "host_bt3": _host_btv,
                                    "host_kc": _host_sums,
                                    "askv_host": [int(x) for x in (md.seq_lens_list[:Bcap] or [])],
                                    "askv_dev": [int(x) for x in _askv_dev],
                                    "slot_dev": [int(x) for x in _me["slot"].detach().to("cpu").tolist()],
                                    "ring_sums": _ringsums,
                                    "ring_blocks": _ringb,
                                    "slots": [int(x) for x in _pinS.tolist()],
                                }
                            )
                            + "\n"
                        )
            except Exception:
                pass
        # R166 tierbuf rolling dump: in-graph per-tier sums (o_pr/l_pr/o_h/l_h)
        # survive EngineCore SIGKill (atexit dead) — dump at the updater.
        _tbN = globals().setdefault("_R166_TBDUMP_N", [0])
        _tb = getattr(type(impl), "_r164_tdbuf", None)
        _scb = getattr(type(impl), "_r199_scbuf", None)
        if _tb is not None or _scb is not None:
            _tbN[0] += 1
            if _tb is not None and _tbN[0] % 128 == 24:
                try:
                    _np166 = torch.npu  # torch_npu at module top registers the submodule

                    _np166.synchronize()
                    _cpu = _tb.detach().to("cpu")
                    _tbrows166 = {}
                    for _l in range(_cpu.shape[0]):
                        for _b in range(_cpu.shape[1]):
                            _v = _cpu[_l, _b]
                            if float(_v.abs().sum()) != 0.0:
                                _tbrows166[f"L{_l}b{_b}"] = [round(float(x), 4) for x in _v.tolist()]
                    if _tbrows166:
                        import json as _j166t

                        with open("/tmp/r166_tierdump.jsonl", "a") as _f166t:
                            _f166t.write(_j166t.dumps({"n": _tbN[0], "rows": _tbrows166}) + "\n")
                except Exception:
                    pass
            # R199_SELFCHECK accumulator dump (own cadence, per-window): max
            # |Δout|/|Δlse| + per-row attribution of paged-vs-legacy in-graph
            # S/R over the last window of steps.
            if _scb is not None and _tbN[0] % 8 == 4:
                try:
                    _np199 = torch.npu  # torch_npu at module top registers the submodule

                    _np199.synchronize()
                    import json as _j199

                    _scrD = getattr(type(impl), "_r199_scrow", None)
                    _scgD = getattr(type(impl), "_r199_scbig", None)
                    with open("/tmp/r199_selfcheck.jsonl", "a") as _f199:
                        _f199.write(
                            _j199.dumps(
                                {
                                    "n": _tbN[0],
                                    "max_dout": float(_scb[0].item()),
                                    "max_dlse": float(_scb[1].item()),
                                    "calls": float(_scb[2].item()),
                                    "row_dout": (
                                        [round(float(x), 4) for x in _scrD.tolist()] if _scrD is not None else None
                                    ),
                                    "slots": [int(x) for x in _me["slot"].tolist()],
                                    "idx_sr_row0": (
                                        [int(x) for x in _me["idx_sr"][:6].tolist()]
                                        if _me.get("idx_sr") is not None
                                        else None
                                    ),
                                    "idx_sr_padrow": (
                                        [
                                            int(x)
                                            for x in _me["idx_sr"][
                                                8 * int(_me.get("npr", 0)) : 8 * int(_me.get("npr", 0)) + 6
                                            ].tolist()
                                        ]
                                        if _me.get("idx_sr") is not None and int(_me.get("npr", 0))
                                        else None
                                    ),
                                    "bt_sr_padrow": (
                                        [int(x) for x in _me["bt_sr"][8].tolist()]
                                        if _me.get("bt_sr") is not None
                                        else None
                                    ),
                                    "slot_comp_mismatch": _r199_sc_mismatch(_me),
                                    "lse_pg_rows": _r199_pool_rowsum(impl, "pg"),
                                    "lse_lg_rows": _r199_pool_rowsum(impl, ""),
                                    "bt_sr_row0": (
                                        [int(x) for x in _me["bt_sr"][0].tolist()]
                                        if _me.get("bt_sr") is not None
                                        else None
                                    ),
                                    "big_steps_per_row": (
                                        [int(x) for x in _scgD.tolist()] if _scgD is not None else None
                                    ),
                                    "n_active": int(len(sl_list) if sl_list is not None else -1),
                                    "sls_head": [int(x) for x in (sl_list or [])[:12]],
                                }
                            )
                            + "\n"
                        )
                    # per-window mode: reset accumulators after each dump
                    _scb.zero_()
                    if _scrD is not None:
                        _scrD.zero_()
                    if _scgD is not None:
                        _scgD.zero_()
                except Exception:
                    pass
        # R166 reverify: when were the wverify-confirmed blocks rewritten?
        _rvN = globals().setdefault("_R166_RV_N", [0])
        _rvB = _R2XX_G.get("_R166_WV_BLOCKS", None)
        if _rvB and _rvN[0] < 400:
            _rvN[0] += 1
            if _rvN[0] % 36 == 1:
                _rowsRV = []
                for (_lRV, _bRV), _srcRV in list(_rvB.items())[:12]:
                    with contextlib.suppress(Exception):
                        _rowsRV.append(
                            (_lRV, _bRV, _srcRV, round(float(impl.key_cache[_bRV].to(torch.int16).sum().item()), 1))
                        )
                _r166_tr("reverify", n=_rvN[0], blocks=_rowsRV)
        # R218_UPSYNC (DIAGNOSTIC_ONLY, default OFF): bisect the missing
        # ordering edge at binding-change steps (b233: a CARDMP dump at the
        # rev-admission window HEALS the deterministic rev-row5 flip, so a
        # sync closes it; the dump does D2H+I/O so which edge is unknown).
        #   1 = device sync BEFORE the update_stream fills (protect the
        #       PREVIOUS in-flight replay's reads from the fills — dbg6 shape)
        #   2 = (see the second site, after _ev.record) make the fills
        #       visible before THIS step's replay launch (POSTSYNC shape)
        if _os.environ.get("R218_UPSYNC", "0") == "1":
            torch.npu.synchronize()
        # R218_UPSYNC=3 (CANDIDATE FIX shape, default OFF): the shippable edge
        # — make update_stream WAIT (device-side, no host block) for everything
        # already enqueued on the caller's stream (the previous replay) before
        # the fills overwrite bt_sr/slot/bt_mirror. up1 (full sync) heals the
        # deterministic rev-row5 flip; this is the same ordering at ~zero cost.
        if _os.environ.get("R218_UPSYNC", "0") == "3":
            update_stream.wait_stream(torch.npu.current_stream())
        # R218_UPSYNC=4 (REBIND-GATED sync, default OFF): device-sync ONLY
        # when a row that was present in the previous step changed its ring
        # slot -- the ragged rebind/rev-admission shape the every-step sync
        # (=1) was added for. The gate exists because =1/=3 also CORRUPT the
        # uniform cells' admission/split steps (b254/b255, 2026-10-17 21:55:
        # ctx2048 chunked rows {17,18,19} diverge@4 + flaky {1,2,6} under
        # up1 AND up3 while up0 is tie-only; ctx8192 up1 row0 early-diverges
        # where the morning up0 run was uniq=1): delaying the host fills
        # past the previous replay lets them clobber replay-advanced state
        # for freshly admitted rows. Append-only admissions (uniform waves,
        # split continuations) keep existing rows' slots stable -> no sync
        # -> up0 behaviour; ragged turnover re-slots existing rows -> sync
        # -> up1 behaviour exactly where it heals.
        if _os.environ.get("R218_UPSYNC", "0") == "4":
            _cur4 = None
            try:
                _cur4 = [int(_x) for _x in _pinS.tolist()]
            except Exception:
                _cur4 = None
            if _cur4 is not None:
                # compare only rows ACTIVE in both steps (_rows = this
                # step's scheduled rows; _pinS is fixed-Bcap so appended
                # admissions would otherwise look like rebinds -- the
                # first =4 gate fired on every admission and reproduced
                # the up1 corruption exactly, b256 g2)
                _pa4, _ps4 = _me.get("r218_up4", (None, None))
                _ca4 = int(_rows)
                if _ps4 is not None and (_ca4 < _pa4 or any(_ps4[_i] != _cur4[_i] for _i in range(min(_pa4, _ca4)))):
                    torch.npu.synchronize()
                _me["r218_up4"] = (_ca4, _cur4[:_ca4])
        with torch.npu.stream(update_stream):
            # R166_BT_HOST (default OFF): fill bt_mirror from the HOST block
            # table (scheduler source of truth) instead of the device
            # md.block_tables — MIDPROBE-proven STALE in this composition
            # (constant capture-era block ids pointing at recycled/garbage
            # blocks while the request's real blocks hold the PFQUANT writes;
            # kc_sums 4M garbage vs zeros expected). Ping-pong pinned H2D on
            # the update stream = single-stream ordered with the gather.
            _bt_host_done = False
            if _os.environ.get("R166_BT_HOST", "0") == "1":
                _ht = getattr(type(impl), "_r166_host_bt", None)
                if _ht is not None:
                    _pinB = _me.get("bt_pin")
                    if _pinB is None:
                        _pinB = _me["bt_pin"] = (
                            torch.zeros((Bcap, nb_max), dtype=torch.int64).pin_memory(),
                            torch.zeros((Bcap, nb_max), dtype=torch.int64).pin_memory(),
                        )
                    _pb = _pinB[_me["ping"] & 1]
                    _nr = min(Bcap, int(_ht.shape[0]))
                    for _i in range(Bcap):
                        if _i < _nr and _i < len(sl_list):
                            _row = _ht[_i, blk_lo : blk_lo + nb_max].tolist()
                            for _j, _v in enumerate(_row):
                                _pb[_i, _j] = int(_v)
                            for _j in range(len(_row), nb_max):
                                _pb[_i, _j] = -1
                        else:
                            _pb[_i].fill_(0)
                    _me["bt_mirror"].copy_(_pb, non_blocking=True)
                    _me["bt_mirror"].clamp_(min=0)
                    _bt_host_done = True
                    _r166_tr(
                        "bt_host_fill",
                        Bcap=Bcap,
                        rows=_nr,
                        head=[int(_pb[0, _k].item()) for _k in range(min(4, nb_max))],
                    )
            if not _bt_host_done:
                if _rows >= Bcap:
                    _me["bt_mirror"].copy_(_bt[0:Bcap, blk_lo : blk_lo + nb_max])
                else:
                    # fewer rows than the captured bucket (dynamic leave): pad the
                    # tail rows with -1 -> clamp -> block 0; those rows carry
                    # askv=0 (neutral) anyway.
                    _me["bt_mirror"][0:_rows].copy_(_bt[0:_rows, blk_lo : blk_lo + nb_max])
                    _me["bt_mirror"][_rows:Bcap].fill_(0)
                _me["bt_mirror"].clamp_(min=0)
            _me["askv"].copy_(_pin, non_blocking=True)
            _me["slot"].copy_(_pinS, non_blocking=True)
            _me["asq"].fill_(1)
            # R199_PRREAD: refresh the S/R paged block table from the live
            # slot flat — bt_sr[b] = slot(b)*nb_sr + j, j=0..nb_sr-1 (PHYSICAL
            # order over the ring row). CHANGE-GATED (host compare on the
            # pinned slot list): steady decode rewrites NOTHING (the ring
            # rotation lives in the data, not the block table — unlike idx_sr,
            # which rotates every step), so the per-step eager-write exposure
            # against the replay's internal stream is eliminated; writes land
            # only on binding changes, inside the same event edge as the
            # idx_sr refresh (record_event below).
            if _me.get("pg199"):
                _slNow199 = [int(x) for x in _pinS.tolist()]
                # R218 FIX CANDIDATE (2026-10-17, marker-gated, default OFF):
                # refresh bt_sr EVERY step instead of only on a slot change.
                # The gate's premise ("the ring rotation lives in the data, not
                # the block table") is true for a rotation but NOT for a SLOT
                # REBIND: if the copy_ below is not ordered against the replay's
                # internal-stream read, the replay reads the PREVIOUS slot's
                # blocks for the rebinding rows -- a disjoint token set, which is
                # exactly the gross S/R divergence R199_SELFCHECK measures
                # (max_dout 26..35 on contiguous row blocks in 4 windows, all
                # other windows at bf16 noise 0.03125). Marker /tmp/r218_btsr_on
                # (env CANNOT gate this: the launcher forwards an -e whitelist).
                _btsrAlways = False
                try:
                    _btsrAlways = _r144_ovl_os.path.exists("/tmp/r218_btsr_on")
                except Exception:
                    _btsrAlways = False
                if _btsrAlways or (_slNow199 != _me.get("r199_slots_last")):
                    _me["r199_slots_last"] = list(_slNow199)
                    _nbQ = int(_me["nb_sr"])
                    _tBQ = _me.get("t_bsr")
                    if _tBQ is None or _tBQ.numel() != _nbQ:
                        _tBQ = _me["t_bsr"] = torch.arange(_nbQ, dtype=torch.int64, device=_me["slot"].device)
                    _me["bt_sr"].copy_(_me["slot"].view(-1, 1) * _nbQ + _tBQ.view(1, -1))
            # R198 ISTAGE trigger: re-stage the mid carriers ONLY when
            # something could have changed them. Steady pure-decode leaves
            # the mid pages IMMUTABLE (prefill-quantized pages; decode tokens
            # reach the mid tier only via the ring-row CU), so the per-replay
            # in-graph re-gather is skipped entirely between events:
            #   (a) request-set / slot-binding change (new, retired,
            #       preempt-recomputed rows — the sig carries (rid, slot)),
            #   (b) per-row page-set or mid_len change (nb growth crosses a
            #       page boundary; recompute resets mid_len),
            #   (c) CU gate fire (t0 % 64 == 0 AND beyond the prompt: CU
            #       writes pages only when decode tokens reach the mid tier),
            #   (d) first use of this bucket (capture warmup / phase start).
            if _os.environ.get("R198_ISTAGE", "0") == "1" and not getattr(_EXTRA_CTX, "capturing", False):
                # (capturing guard: the restage path host-syncs, which is
                # illegal/unstable inside a capture pass — skip during
                # capture; the first REAL replay restages (trigger d))
                _sig198 = (
                    Bcap,
                    tuple(
                        sorted((str(_srids[_i]) if _i < len(_srids) else str(_i), int(_pinS[_i])) for _i in range(Bcap))
                    ),
                )
                # page-set proxy: per-row nb (ceil(mid_len / block_size));
                # mid_len itself grows EVERY step without changing the
                # gathered page set/content in the prefill-covered regime.
                _bsz198 = int(impl.vllm_config.cache_config.block_size)
                _ml198 = [(int(_pin[_i]) + _bsz198 - 1) // _bsz198 for _i in range(Bcap)]
                # per-PHASE prompt estimate: reset whenever the request
                # set changes (the capture-phase value covers warmup rows
                # and would gate the CU condition on for the whole run —
                # the dbg4 restage-every-step cadence bug)
                _sls198 = [int(x) for x in (sl_list or [])][:Bcap]
                _init198 = _me.get("r198_init_sl")
                if _init198 is None or _me.get("r198_sig") != _sig198:
                    _init198 = _me["r198_init_sl"] = min(_sls198) if _sls198 else 0
                _cu198 = False
                for _i198 in range(min(len(sl_list), Bcap)):
                    _t0198 = int(sl_list[_i198]) - int(rcnt)
                    if _t0198 >= 0 and (_t0198 % 64) == 0 and _t0198 + 64 > _init198 - 1:
                        _cu198 = True
                        break
                _rs198 = False
                if _os.environ.get("R198_DBG", "0") == "2" and globals().setdefault("_R198_POISON", [0])[0] >= 2:
                    pass  # poison persists: suppress further restages
                elif (
                    _me.get("r198_sig") != _sig198
                    or _me.get("r198_ml") != _ml198
                    or _cu198
                    or _os.environ.get("R198_FORCE", "0") == "1"
                ):
                    # R198_FORCE (DIAGNOSTIC_ONLY): restage every step. If this
                    # makes ISTAGE=1 bitwise repeatable, the residual is
                    # CARRIER STALENESS, not a race — then narrow the trigger.
                    # ORDERING (dbg6 device-proof): the restage's carrier
                    # writes MUST NOT overlap the in-flight replay's carrier
                    # reads. The replay executes on an INTERNAL stream that
                    # neither stream-FIFO nor the event edges cover (R167
                    # class: E_replay_prev records fire early; dbg7: current-
                    # stream FIFO does NOT heal) — a full device sync before
                    # the restage is the only complete edge (dbg6: heals).
                    # Restage events are rare (request-set / page-set
                    # changes; ~2 + row-page-crossings per generation), so
                    # the sync stall amortizes to <0.5ms/step.
                    _np198 = torch.npu  # torch_npu at module top registers the submodule

                    _np198.synchronize()
                    _rs198 = True
                    _n198 = type(impl)._r198_restage(impl, Bcap, _me)
                    _me["r198_sig"] = _sig198
                    _me["r198_ml"] = _ml198
                    if _os.environ.get("R198_POSTSYNC", "0") == "1":
                        # MISSING-EDGE TEST (DIAGNOSTIC_ONLY): the restage
                        # writes the carriers on THIS (update) stream; the
                        # replay reads them on an INTERNAL stream that no
                        # stream-FIFO/event covers. The pre-restage sync only
                        # protects the PREVIOUS replay. This sync makes the
                        # carrier writes visible before the next launch.
                        _npPS = torch.npu  # torch_npu at module top registers the submodule

                        _npPS.synchronize()
                    if _os.environ.get("R198_DBG", "0") == "2" and globals().setdefault("_R198_POISON", [0])[0] == 1:
                        # wiring probe: poison ONE layer's k carrier after a
                        # correct restage — the deterministic degenerate
                        # stream must CHANGE if the graph reads the pool
                        globals()["_R198_POISON"][0] = 2
                        try:
                            _cpZ = getattr(type(impl), "_r113_cpool_batched", None) or {}
                            for (_lz, _bz), _cz in _cpZ.items():
                                if _bz != Bcap:
                                    continue
                                _cz["k"].fill_(0x55)
                                import json as _jZ

                                with open("/tmp/r198_dbg.jsonl", "a") as _fZ:
                                    _fZ.write(_jZ.dumps({"poison": 1, "lid": _lz, "k_ptr": _cz["k"].data_ptr()}) + "\n")
                                break
                        except Exception as _eZ:
                            pass
                    if _os.environ.get("R198_DBG", "0") == "2":
                        globals().setdefault("_R198_POISON", [0])[0] = max(
                            globals().setdefault("_R198_POISON", [0])[0], 1
                        )
                    if _os.environ.get("R198_DBG", "0") == "1":
                        _st198 = globals().setdefault("_R198_DBG_N", [0])
                        _st198[0] += 1
                        if _st198[0] <= 40:
                            try:
                                import json as _j198

                                with open("/tmp/r198_dbg.jsonl", "a") as _f198:
                                    _f198.write(
                                        _j198.dumps(
                                            {
                                                "n": _st198[0],
                                                "bk": Bcap,
                                                "layers": _n198,
                                                "mir": [int(x) for x in _me["bt_mirror"][0, :6].tolist()],
                                                "askv0": int(_pin[0]),
                                                "nb0": _ml198[0] if _ml198 else -1,
                                            }
                                        )
                                        + "\n"
                                    )
                            except Exception:
                                pass
                    if _os.environ.get("R198_DBG", "0") == "1" and globals().setdefault("_R198_DBG_C", [0])[0] < 3:
                        # one-shot content check: fresh gather vs live
                        # carriers for ONE layer, few steps after a restage
                        globals()["_R198_DBG_C"][0] += 1
                        try:
                            _cpP = getattr(type(impl), "_r113_cpool_batched", None) or {}
                            for (_lkP, _bkP), _cP in _cpP.items():
                                if _bkP != Bcap or _cP.get("r198_src") is None:
                                    continue
                                (kcP, vcP, kscP, _z1, _z2, _z3) = _cP["r198_src"]
                                _gP = kcP.index_select(0, _me["bt_mirror_flat"])
                                _freshP = _gP.permute(0, 2, 1, 3)
                                _neP = _freshP.view(torch.uint8) != _cP["k"].view(torch.uint8)
                                _dP = int(_neP.sum().item())
                                _totP = _cP["k"].numel()
                                # position anatomy: mismatching PAGES (rows
                                # of _cP["k"] = carrier pages) vs live nb
                                _pgP = _neP.view(_cP["k"].shape[0], -1).any(-1)
                                _pgwP = _pgP.view(int(_cP["r198_geo"][0]), int(_cP["r198_geo"][4]))
                                _nbP = max(_ml198) if _ml198 else 0
                                _liveP = int(_pgwP[:, :_nbP].sum().item())
                                _padP = int(_pgwP[:, _nbP:].sum().item())
                                _rowsP = [int(x) for x in _pgwP.any(-1).nonzero().flatten()[:8].tolist()]
                                _firstP = None
                                _nzP = _neP.nonzero()
                                if _nzP.shape[0] > 0:
                                    _f0P = _nzP[0].tolist()
                                    _firstP = {
                                        "page": _f0P[0],
                                        "row": _f0P[0] // int(_cP["r198_geo"][4]),
                                        "col": _f0P[0] % int(_cP["r198_geo"][4]),
                                        "car": int(_cP["k"].view(-1)[_nzP[0][0] + 0].item())
                                        if _cP["k"].dim() == 1
                                        else None,
                                    }
                                import json as _jP

                                with open("/tmp/r198_dbg.jsonl", "a") as _fP:
                                    _fP.write(
                                        _jP.dumps(
                                            {
                                                "chk": 1,
                                                "lid": _lkP,
                                                "k_ptr": _cP["k"].data_ptr(),
                                                "kc_ptr": kcP.data_ptr(),
                                                "idm_head": [int(x) for x in _me["bt_mirror_flat"][:4].tolist()],
                                                "mismatch_bytes": _dP,
                                                "total": _totP,
                                                "live_nb": _nbP,
                                                "bad_pages_live": _liveP,
                                                "bad_pages_pad": _padP,
                                                "bad_rows_first8": _rowsP,
                                                "first": _firstP,
                                            }
                                        )
                                        + "\n"
                                    )
                                break
                        except Exception as _eP:
                            try:
                                with open("/tmp/r198_dbg.jsonl", "a") as _fP:
                                    _fP.write("CHK_ERR {!r}\n".format(_eP))
                            except Exception:
                                pass
                # R198_DBG=3 STALENESS PROBE (DIAGNOSTIC_ONLY): on a few steps,
                # compare the LIVE carriers against a FRESH gather of the same
                # pages NOW (i.e. after this step's writers ran). Non-zero bytes
                # on a step where no restage fired == the carriers ARE stale, and
                # the position/page anatomy says which writer went uncovered.
                if _os.environ.get("R198_DBG", "0") == "3" and globals().setdefault("_R198_ST", [0])[0] < int(
                    _os.environ.get("R198_DBG_MAX", "12")
                ):
                    globals()["_R198_ST"][0] += 1
                    try:
                        import json as _jS

                        import torch as _tS

                        _tS.npu.synchronize()
                        _cpS = getattr(type(impl), "_r113_cpool_batched", None) or {}
                        for (_lkS, _bkS), _cS in _cpS.items():
                            if _bkS != Bcap or _cS.get("r198_src") is None:
                                continue
                            _kcS = _cS["r198_src"][0]
                            _gS = _kcS.index_select(0, _me["bt_mirror_flat"])
                            _frS = _gS.permute(0, 2, 1, 3)
                            _neS = _frS.view(torch.uint8) != _cS["k"].view(torch.uint8)
                            _pgS = _neS.view(_cS["k"].shape[0], -1).any(-1)
                            _gwS = _pgS.view(int(_cS["r198_geo"][0]), int(_cS["r198_geo"][4]))
                            _nbS = max(_ml198) if _ml198 else 0
                            # MIRROR vs LIVE: the eager restage gathers with
                            # the HOST-visible mirror bt_mirror; the in-graph
                            # staging (ISTAGE=0) reads md.block_tables on the
                            # device. If the mirror != the live block table
                            # for this step, the carriers carry WRONG PAGES.
                            _mismBT = -1
                            try:
                                _btv = _bt[0 : min(_rows, Bcap), blk_lo : blk_lo + nb_max].to(torch.int64)
                                _bmv = _me["bt_mirror"][0 : min(_rows, Bcap), :nb_max].to(torch.int64)
                                _mismBT = int((_btv != _bmv).sum().item())
                            except Exception:
                                _mismBT = -2
                            with open("/tmp/r198_stale.jsonl", "a") as _fS:
                                _fS.write(
                                    _jS.dumps(
                                        {
                                            "bt_mismatch": _mismBT,
                                            "bt_live_head": [int(x) for x in _bt[0, blk_lo : blk_lo + 4].tolist()]
                                            if hasattr(_bt, "tolist")
                                            else None,
                                            "bt_mirror_head": [int(x) for x in _me["bt_mirror"][0, :4].tolist()],
                                            "sl": [int(x) for x in (sl_list or [])[:2]],
                                            "restaged": bool(_rs198),
                                            "cu198": bool(_cu198),
                                            "nb": _nbS,
                                            "mismatch_bytes": int(_neS.sum().item()),
                                            "total": int(_cS["k"].numel()),
                                            "bad_pages_live": int(_gwS[:, :_nbS].sum().item()),
                                            "bad_pages_pad": int(_gwS[:, _nbS:].sum().item()),
                                            "bad_rows": [int(x) for x in _gwS.any(-1).nonzero().flatten()[:8].tolist()],
                                            "t0": int(sl_list[0]) - int(rcnt) if sl_list else -1,
                                        }
                                    )
                                    + "\n"
                                )
                            break
                    except Exception as _eS:
                        with contextlib.suppress(Exception):
                            open("/tmp/r198_stale.jsonl", "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                                "ERR {!r}\n".format(_eS)
                            )
            # idx_sr refresh via EAGER tensor arithmetic on the update
            # stream (NO captured index arithmetic — capture-time folding of
            # the arange chain baked slot=0 indices; the graph must read a
            # PLAIN eagerly-preallocated tensor).
            _nprD = _me.get("npr")
            if _nprD:
                # R167_STRIDEFIX: the flat slot stride MUST be the TRUE ring
                # row width (_tt_mhp = pref+recent+4), NOT npr — the legacy
                # npr stride misaligned every slot>0 by 4 rows (S/R gather read
                # [4 slack zeros + rows 0..1019]; the in-graph slide wrote
                # prefix rows 124-127). Identical-prompt fixtures hid it as a
                # false majority ("row0 the odd one out").
                _strD = _nprD
                if _os.environ.get("R167_STRIDEFIX", "0") == "1":
                    _mhpD = _me.get("mhp")
                    if not _mhpD:
                        raise RuntimeError("[R167-STRIDEFIX] mhp not preallocated (fail-closed)")
                    _strD = int(_mhpD)
                _tC = _me.get("t_const")
                if _tC is None or _tC.numel() != _nprD:
                    _tC = _me["t_const"] = torch.arange(_nprD, dtype=torch.int64, device=_me["slot"].device)
                _prefD = _me.get("prefp", 0)
                _capD = _me.get("capw", 0)
                try:
                    if _os.path.exists("/tmp/r218_xaudit_on"):
                        _R2XX_G.setdefault("_R218_DISC", []).append(
                            "drv_%s"
                            % ("vring" if (_os.environ.get("R198_VRING", "0") == "1" and _capD > 1) else "legacy")
                        )
                except Exception:
                    pass
                if _os.environ.get("R198_VRING", "0") == "1" and _capD > 1:
                    # R198 VRING driver: rotated S/R read pattern + rotating
                    # append position. Per row (host, cheap):
                    #   X_b    = ring origin (impl._tt_ring_org[rid];
                    #            default prefp = left-packed/growing)
                    #   head_b = prefp + ((sl_b - _capD - X_b) % _capD)
                    #           (window = last _capD tokens, read start pos)
                    #   pnew_b = prefp + ((sl_b - 1 - X_b) % _capD)
                    # idx_sr[:, :prefp] = slot*strD + arange(prefp)
                    # idx_sr[:, prefp:] = slot*strD + prefp
                    #                       + (head + j) % capw,  j=0..capw-1
                    _org198 = getattr(impl, "_tt_ring_org", None) or {}
                    _srids198 = getattr(md, "request_ids", None) or []
                    _pinH = _me.get("r198_pin")
                    if _pinH is None:
                        _pinH = _me["r198_pin"] = (
                            (
                                torch.zeros(Bcap, dtype=torch.int64).pin_memory(),
                                torch.zeros(Bcap, dtype=torch.int64).pin_memory(),
                            ),
                            (
                                torch.zeros(Bcap, dtype=torch.int64).pin_memory(),
                                torch.zeros(Bcap, dtype=torch.int64).pin_memory(),
                            ),
                        )
                    # per-VALUE ping-pong pairs (head, pnew), each with the
                    # askv_pin 2-step depth discipline: buffer reuse is 2
                    # steps apart, matching the proven askv/slot pattern.
                    _ph = _pinH[0][_me["ping"] & 1]
                    _ph2 = _pinH[1][_me["ping"] & 1]
                    for _i198 in range(Bcap):
                        _rs198 = _srids198[_i198] if _i198 < len(_srids198) else None
                        # dual key-space: _tt_ring_org is keyed by _tt_rid(b)
                        # (int row index under R144_TT_REQID=0, string rid
                        # under =1); try the metadata rid first, then the
                        # row index, then the left-packed default.
                        _X198 = int(_org198.get(_rs198, _org198.get(_i198, _prefD)))
                        _sl198 = int(sl_list[_i198]) if _i198 < len(sl_list) else 0
                        _ph[_i198] = _prefD + ((_sl198 - _capD - _X198) % _capD)
                        _ph2[_i198] = _prefD + ((_sl198 - 1 - _X198) % _capD)
                    # R218 XDUMP (marker /tmp/r218_xdump_on, default OFF,
                    # diagnostic): the per-row ring origin the DRIVER resolves
                    # vs the one the impl map holds. A driver fallback to prefp
                    # (X missing for that rid) anchors the rotating window
                    # differently from what prefill wrote -> the window drops
                    # the wrong token every step. Rows are recorded compactly:
                    # [row, X_driver, X_impl(-1 = missing), sl, head, p_new].
                    if _os.environ.get("R218_XDUMP", "0") == "1" or _os.path.exists("/tmp/r218_xdump_on"):
                        try:
                            import json as _jxd

                            _rxs = globals().setdefault("_R218_XD", [0])
                            if _rxs[0] < 3000:
                                _rxs[0] += 1
                                _xr = []
                                for _iX in range(Bcap):
                                    _rsX = _srids198[_iX] if _iX < len(_srids198) else None
                                    _xiX = _org198.get(_rsX, _org198.get(_iX, None))
                                    _xr.append(
                                        [
                                            _iX,
                                            int(_org198.get(_rsX, _org198.get(_iX, _prefD))),
                                            (int(_xiX) if _xiX is not None else -1),
                                            (int(sl_list[_iX]) if _iX < len(sl_list) else -1),
                                            int(_ph[_iX]),
                                            int(_ph2[_iX]),
                                        ]
                                    )
                                with open("/tmp/r218_xdump.jsonl", "a") as _fxd:
                                    _fxd.write(
                                        _jxd.dumps(
                                            {
                                                "n": _rxs[0],
                                                "Bcap": int(Bcap),
                                                "nrids": len(_srids198),
                                                "nsl": len(sl_list),
                                                "pref": int(_prefD),
                                                "cap": int(_capD),
                                                "npr": int(_nprD),
                                                "org_keys": len(_org198),
                                                "rows": _xr,
                                            }
                                        )
                                        + "\n"
                                    )
                        except Exception:
                            pass
                    _headD = _me["slot"].new_empty(Bcap)
                    _headD.copy_(_ph, non_blocking=True)
                    _tP = _me.get("t_pref198")
                    if _tP is None or _tP.numel() != _prefD:
                        _tP = _me["t_pref198"] = torch.arange(_prefD, dtype=torch.int64, device=_me["slot"].device)
                    _tW = _me.get("t_w198")
                    if _tW is None or _tW.numel() != _capD:
                        _tW = _me["t_w198"] = torch.arange(_capD, dtype=torch.int64, device=_me["slot"].device)
                    # R218 FIX CANDIDATE (2026-10-17, marker /tmp/r218_ringord_on,
                    # default OFF): _headD is the FLAT position
                    # pref + ((sl-cap-X) % cap); taking it % cap rotates the
                    # window enumeration by pref. The slot SET is unchanged (a
                    # bijection over the same ring), so the attention math is
                    # invariant -- but the FP REDUCTION ORDER differs from the
                    # legacy left-packed window, the ~1e-3 deltas compound
                    # through the layers and greedy decode flips.
                    # BF16-native-measured at ctx2048 NS=32 NT=64 (same prompts,
                    # mN301 sha 6482906ed26cd4e3): VRING=0 32/32 rows exact,
                    # VRING=1 29/32 (rows 29,30,31 first differ at token 34);
                    # EDGEFENCE / bt_sr-always-refresh / SELFCHECK variants do
                    # not move it (all RED with the same sha), i.e. it is pure
                    # ordering, not a race and not stale metadata.
                    # Subtract the flat prefix offset before the modulo =>
                    # TEMPORAL order, identical to the legacy path (and BF16).
                    _rord218 = False
                    try:
                        # R218 2026-10-17: the file marker is process-GLOBAL to
                        # the overlay (two sessions share it), so also accept a
                        # per-leg env gate R218_RINGORD=1 (default 0) -- same
                        # behaviour, no cross-session interference.
                        _rord218 = _os.environ.get("R218_RINGORD", "0") == "1" or _os.path.exists(
                            "/tmp/r218_ringord_on"
                        )
                    except Exception:
                        _rord218 = False
                    if _rord218:
                        try:
                            _rn218 = globals().setdefault("_R218_RORD_N", [0])
                            if _rn218[0] < 6:
                                _rn218[0] += 1
                                with open("/tmp/r218_ringord.jsonl", "a") as _frn:
                                    _frn.write(
                                        f'{{"n": {_rn218[0]}, "pref": {int(_prefD)}, "cap": {int(_capD)}, '
                                        f' "npr": {int(_nprD)}}}\n'
                                    )
                        except Exception:
                            pass
                        _ring2 = ((_headD.view(-1, 1) - _prefD) + _tW.view(1, -1)) % _capD
                    else:
                        _ring2 = (_headD.view(-1, 1) + _tW.view(1, -1)) % _capD
                    _slotV = _me["slot"].view(-1, 1) * _strD
                    _iv2 = _me["idx_sr"].view(Bcap, _nprD)
                    _iv2[:, :_prefD].copy_(_slotV + _tP.view(1, -1))
                    _iv2[:, _prefD:].copy_(_slotV + _prefD + _ring2)
                    _me["p_new"].copy_(_ph2, non_blocking=True)
                    _me["p_new"].add_(_me["slot"] * _strD)
                else:
                    _me["idx_sr"].copy_((_me["slot"].view(-1, 1) * _strD + _tC.view(1, -1)).reshape(-1))
                    if _capD > 1:
                        _tW1 = _me.get("t_w1")
                        if _tW1 is None or _tW1.numel() != _capD - 1:
                            _tW1 = _me["t_w1"] = torch.arange(
                                _prefD + 1, _prefD + _capD, dtype=torch.int64, device=_me["slot"].device
                            )
                            _me["t_w0"] = torch.arange(
                                _prefD, _prefD + _capD - 1, dtype=torch.int64, device=_me["slot"].device
                            )
                        _slotV = _me["slot"].view(-1, 1) * _strD
                        _me["p_src"].copy_((_slotV + _tW1.view(1, -1)).reshape(-1))
                        _me["p_dst"].copy_((_slotV + _me["t_w0"].view(1, -1)).reshape(-1))
                        _me["p_new"].copy_(_me["slot"] * _strD + (_prefD + _capD - 1))
            if record_event:
                _ev.record(update_stream)
                # R218_UPSYNC=2 (DIAGNOSTIC_ONLY): second bisect site — sync
                # AFTER the fills+record so the next replay (launched by the
                # caller on return) cannot race the just-written metadata
                # (first site: sync BEFORE the fills, see R218_UPSYNC=1).
                if _os.environ.get("R218_UPSYNC", "0") == "2":
                    torch.npu.synchronize()
            # R170_MDFILL pre-fill path passes record_event=False: the flat
            # holds real values pre-replay, so the captured no-op wait is
            # harmless AND the 1:1 record<->wait lifecycle is untouched
            # (extra records before the next captured wait hang the device —
            # h18/h19 EVFIX postmortems).
        # R167_RINGAUDIT (default OFF, diagnostic): per-step driver audit — md
        # (rids/slots/askv host+device), ring 64-row segment sums at the
        # current slots (post-replay state; sync = perturbing, diag legs only),
        # the paged content of each row's first bt blocks, the tier matrix.
        if _os.environ.get("R167_RINGAUDIT", "0") == "1":
            try:
                import json as _j167

                _raS = globals().setdefault("_R167_RA", {"n": 0, "prev_rids": None})
                _raS["n"] += 1
                _firstk = int(_os.environ.get("R167_FIRSTK", "8"))
                _rk = tuple(str(r) for r in (_srids or []))
                _rids_changed = _rk != _raS["prev_rids"]
                _raS["prev_rids"] = _rk
                if _raS["n"] <= _firstk or _rids_changed or (_raS["n"] % 32) == 1:
                    _np167 = torch.npu  # torch_npu at module top registers the submodule

                    _np167.synchronize()
                    _rec167 = {
                        "n": _raS["n"],
                        "Bcap": Bcap,
                        "rids": [str(r)[:14] for r in (_srids or [])][:Bcap],
                        "sl_host": [int(x) for x in sl_list[:Bcap]],
                        "askv_host": [int(x) for x in _pin.tolist()[:Bcap]],
                        "slots_host": [int(x) for x in _pinS.tolist()[:Bcap]],
                        "askv_dev": [int(x) for x in _me["askv"].tolist()],
                        "slot_dev": [int(x) for x in _me["slot"].tolist()],
                        "rids_changed": bool(_rids_changed),
                    }
                    _hk0 = impl._tt_hp_k.get(0)
                    if _hk0 is not None:
                        _nprR = int(_hk0.shape[1])
                        _segs = []
                        for _s167x in [int(x) for x in _pinS.tolist()[:Bcap]]:
                            _row167 = _hk0[_s167x]
                            _segs.append(
                                [
                                    round(float(_row167[_i167 : _i167 + 64].float().sum().item()), 1)
                                    for _i167 in range(0, _nprR, 64)
                                ]
                            )
                        _rec167["ring_seg_L0"] = _segs
                        _rec167["ring_npr"] = _nprR
                    _btm = _me["bt_mirror"]
                    _pb167 = []
                    for _b167 in range(Bcap):
                        _bl167 = [int(x) for x in _btm[_b167, :3].tolist()]
                        _pb167.append(
                            {
                                "blk": _bl167,
                                "kc": [
                                    round(float(impl.key_cache[_x167].to(torch.int16).sum().item()), 1)
                                    if _x167 >= 0
                                    else None
                                    for _x167 in _bl167
                                ],
                            }
                        )
                    _rec167["paged_L0"] = _pb167
                    _tdb = getattr(type(impl), "_r164_tdbuf", None)
                    if _tdb is not None:
                        _cpu167 = _tdb.detach().to("cpu")
                        _rows167 = {}
                        for _l167 in range(_cpu167.shape[0]):
                            for _b167x in range(min(Bcap, _cpu167.shape[1])):
                                _v167 = _cpu167[_l167, _b167x]
                                if float(_v167.abs().sum()) != 0.0:
                                    _rows167[f"L{_l167}b{_b167x}"] = [
                                        round(float(_x167), 3) for _x167 in _v167.tolist()
                                    ]
                        _rec167["tier"] = _rows167
                    with open("/tmp/r167_ringaudit.jsonl", "a") as _f167:
                        _f167.write(_j167.dumps(_rec167) + "\n")
            except Exception as _e167:
                try:
                    with open("/tmp/r167_ringaudit.jsonl", "a") as _f167e:
                        _f167e.write("RINGAUDIT_ERR " + repr(_e167) + "\n")
                except Exception:
                    pass
        if _os.environ.get("R164_TIERDUMP", "0") == "1":
            # DIAGNOSTIC legs only (NT small): host-sync + D2H of the tier
            # checksum buffer written IN-GRAPH this step. EngineCore is
            # SIGKILLed at engine shutdown -> atexit cannot run; dump here.
            try:
                import json as _jtd

                _npu_td = torch.npu  # torch_npu at module top registers the submodule

                _npu_td.synchronize()
                _td = getattr(type(impl), "_r164_tdbuf", None)
                if _td is not None:
                    _cpu = _td.detach().to("cpu")
                    _tdrows = {}
                    for _l in range(_cpu.shape[0]):
                        for _b in range(_cpu.shape[1]):
                            _v = _cpu[_l, _b]
                            if float(_v.abs().sum()) == 0.0:
                                continue
                            _tdrows[f"L{_l}b{_b}"] = [round(float(x), 4) for x in _v.tolist()]
                    with open("/tmp/r164_tierdump.jsonl", "a") as _ftd:
                        _ftd.write(_jtd.dumps({"step": _me["ping"] - 1, "Bcap": Bcap, "rows": _tdrows}) + "\n")
            except Exception as _etd:
                try:
                    with open("/tmp/r164_tierdump.jsonl", "a") as _ftd2:
                        _ftd2.write("dump_error " + repr(_etd) + "\n")
                except Exception:
                    pass
        _pb = _R2XX_G.get("_R164_PASSBUF")
        if _pb is not None and not globals().get("_R164_PASSDUMPED"):
            globals()["_R164_PASSDUMPED"] = True
            try:
                import json as _jpd

                _npd = torch.npu  # torch_npu at module top registers the submodule

                _npd.synchronize()
                _n = min(_R2XX_G["_R164_PASSN"][0], 4096)
                with open("/tmp/r164_passive.jsonl", "w") as _fdp:
                    for _i, _r in enumerate(_pb[:_n].cpu().tolist()):
                        _fdp.write(
                            _jpd.dumps({"i": _i, "L": int(_r[0]), "srk": round(_r[1], 1), "row0": round(_r[2], 3)})
                            + "\n"
                        )
            except Exception:
                pass
        _cb = _R2XX_G.get("_R164_WCHKBUF")
        if _cb is not None and not globals().get("_R164_WCHKDUMPED"):
            globals()["_R164_WCHKDUMPED"] = True
            try:
                import json as _jw

                _npw = torch.npu  # torch_npu at module top registers the submodule

                _npw.synchronize()
                _n = min(_R2XX_G["_R164_WCHKN"][0], 8192)
                with open("/tmp/r164_wchk.jsonl", "w") as _fw:
                    for _i, _r in enumerate(_cb[:_n].cpu().tolist()):
                        _fw.write(_jw.dumps({"i": _i, "key": int(_r[0]), "dst8": round(_r[1], 1)}) + "\n")
            except Exception:
                pass
        _mk = getattr(type(impl), "_r164_mk", None)
        if _mk is None:
            type(impl)._r164_mk = _mk = set()
        if "R164" not in _mk:
            _mk.add("R164")
            with contextlib.suppress(Exception):
                open("/tmp/r164_markers.log", "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                    f"ACTIVE marker=OSCAR_MID_R164 Bcap={Bcap} nb_max={nb_max} "
                    f"blk_lo={blk_lo} update_enqueues_per_step=6 "
                    f"in_graph_fill_groups=36 (was 36*B={36 * Bcap})\n"
                )

        if _os.environ.get("R168_OUTLSE", "0") == "1":
            # R168: dump pooled OUT/LSE INSIDE the EngineCore process (the
            # front-end harness cannot see these class-level pools across
            # process boundaries). One file per replay step. Default OFF.
            try:
                _sn168 = globals().setdefault("_R168_SNAP_PROC_N", [0])
                _si168 = _sn168[0]
                _sn168[0] += 1
                if _si168 < 1100:
                    import torch as _t168

                    _t168.npu.synchronize()
                    _rec168 = {}
                    _op168 = getattr(type(impl), "_r113_opool_batched", None) or {}
                    for _k2 in _op168:
                        _tup = _op168[_k2]
                        _rec168[f"L{int(_k2[0])}c{int(_k2[4])}"] = {
                            "out": _tup[0].detach().to(torch.float16).cpu(),
                            "lse": _tup[1].detach().float().cpu().clone(),
                        }
                    _rids168 = list(getattr(md, "request_ids", None) or [])[:8]
                    _sls168 = [int(x) for x in (getattr(md, "seq_lens_list", None) or [])[:8]]
                    _snap168 = {"s": _si168, "layers": _rec168, "nt": int(num_tokens), "rids": _rids168, "sls": _sls168}
                    if _os.environ.get("R169_FLATDUMP", "0") == "1":
                        # R169: LIVE R164 flats AS THIS SNAPSHOT LEAVES THEM
                        # (post-refresh, post-sync) = the exact inputs the
                        # NEXT replay reads. Host expectations alongside.
                        # snapshot[k].flats == inputs consumed by replay k+1.
                        _btmd169 = _bt[0:Bcap, blk_lo : blk_lo + nb_max].clamp(min=0)
                        _fl169 = {
                            "askv": _me["askv"].tolist(),
                            "askv_exp": _pin.tolist(),
                            "slot": _me["slot"].tolist(),
                            "slot_exp": _pinS.tolist(),
                            "bt_mirror": _me["bt_mirror"].tolist(),
                            "bt_md": _btmd169.tolist(),
                            "bt_miss_md": int((_me["bt_mirror"] != _btmd169).sum().item()),
                            "mid_start": int(mid_start),
                            "rcnt": int(rcnt),
                            "blk_lo": int(blk_lo),
                            "nb_max": int(nb_max),
                            "ping": int(_me["ping"]) & 1,
                        }
                        _hb169 = getattr(type(impl), "_r169_host_bt", None)
                        if _hb169 is not None and int(_hb169.dim()) == 2 and int(_hb169.shape[1]) >= blk_lo + nb_max:
                            _rows169 = []
                            for _r169 in range(min(Bcap, int(_hb169.shape[0]))):
                                _rr169 = _hb169[_r169, blk_lo : blk_lo + nb_max].tolist()
                                _rows169.append([int(v) if int(v) >= 0 else 0 for v in _rr169])
                            _fl169["bt_host"] = _rows169
                            _miss169 = 0
                            for _r169, _row169 in enumerate(_rows169):
                                _miss169 += sum(
                                    1 for _c169 in range(nb_max) if _fl169["bt_mirror"][_r169][_c169] != _row169[_c169]
                                )
                            _fl169["bt_miss_host"] = _miss169
                        _snap168["flats"] = _fl169
                    if _os.environ.get("R218_ROWCMP", "0") == "1" and _si168 < int(
                        _os.environ.get("R218_ROWCMP_N", "40")
                    ):
                        # R218 ROW-PAGE FINGERPRINT (DIAGNOSTIC_ONLY): rows with
                        # the SAME askv share prompt+length, so their middle
                        # pages must be byte-identical. Any hash difference
                        # localizes the defect to the WRITE side (quant/scatter/
                        # slot->page mapping), before attention reads anything.
                        try:
                            import hashlib as _hRC
                            import json as _jRC

                            _askRC = [int(x) for x in _me["askv"][:Bcap].tolist()]
                            _bszRC = int(impl.vllm_config.cache_config.block_size)
                            _rowsRC = []
                            for _rRC in range(min(Bcap, 32)):
                                _nbRC = (_askRC[_rRC] + _bszRC - 1) // _bszRC
                                _pgRC = [int(x) for x in _me["bt_mirror"][_rRC, :_nbRC].tolist()]
                                if not _pgRC:
                                    continue
                                _kRC = impl.key_cache[_pgRC].detach().cpu().numpy().tobytes()
                                _vRC = impl.value_cache[_pgRC].detach().cpu().numpy().tobytes()
                                _rowsRC.append(
                                    {
                                        "r": _rRC,
                                        "askv": _askRC[_rRC],
                                        "nb": _nbRC,
                                        "pages": _pgRC[:6],
                                        "kh": _hRC.md5(_kRC).hexdigest()[:16],
                                        "vh": _hRC.md5(_vRC).hexdigest()[:16],
                                        "ksz": int(len(_kRC)),
                                    }
                                )
                            with open("/tmp/r218_rowcmp.jsonl", "a") as _fRC:
                                _fRC.write(_jRC.dumps({"s": _si168, "rows": _rowsRC}) + "\n")
                        except Exception as _eRC:
                            with contextlib.suppress(Exception):
                                open("/tmp/r218_rowcmp.jsonl", "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                                    "ERR {!r}\n".format(_eRC)
                                )
                    if _os.environ.get("R218_WINCMP", "0") == "1" and _si168 < int(
                        _os.environ.get("R218_WINCMP_N", "3")
                    ):
                        # R218 S/R-WINDOW CONTENT FINGERPRINT (DIAGNOSTIC_ONLY):
                        # rows that share prompt+sl MUST read byte-identical
                        # prefix/recent windows. Hashes L0 only (first layer,
                        # allocated once) so the D2H cost stays ~2MB/row.
                        try:
                            import hashlib as _hW
                            import json as _jW

                            _hpkW = getattr(impl, "_tt_hp_k", None)
                            _hk0 = None if not _hpkW else _hpkW.get(0)
                            _rowsW = []
                            if _hk0 is not None:
                                _slW = [int(x) for x in (getattr(md, "seq_lens_list", None) or [])]
                                for _bW in range(min(Bcap, 32)):
                                    try:
                                        _ridW = impl._tt_rid(_bW)
                                    except Exception:
                                        _ridW = None
                                    if _ridW is None:
                                        continue
                                    _pW = int(impl._tt_pref.get(_ridW, 0))
                                    _rW = int(impl._tt_rcnt.get(_ridW, 0))
                                    _sW = int(impl._tt_slot(_bW))
                                    _XW = impl._tt_ring_org.get(_ridW)
                                    _aW = _hk0[_sW, :_pW].detach().float().cpu().numpy().tobytes()
                                    _bW2 = _hk0[_sW, _pW : _pW + _rW].detach().float().cpu().numpy().tobytes()
                                    _rowsW.append(
                                        {
                                            "b": _bW,
                                            "rid": str(_ridW),
                                            "slot": _sW,
                                            "pref": _pW,
                                            "rcnt": _rW,
                                            "X": (None if _XW is None else int(_XW)),
                                            "sl": (_slW[_bW] if _bW < len(_slW) else -1),
                                            "kp": _hW.md5(_aW).hexdigest()[:12],
                                            "kr": _hW.md5(_bW2).hexdigest()[:12],
                                        }
                                    )
                            if (
                                _os.environ.get("R218_WINSAVE", "0") == "1"
                                and _si168 < int(_os.environ.get("R218_WINSAVE_N", "2"))
                                and _hk0 is not None
                            ):
                                # R218: save the EXACT gathered S/R window
                                # (what the merge consumes) for the first rows
                                # so a host-side compare can measure the
                                # rotation offset between two rows.
                                import torch as _tWS

                                _nfW = int(_hk0.shape[2]) * int(_hk0.shape[3])
                                _flatW = _hk0.view(-1, _nfW)
                                for _bS in range(min(Bcap, int(_os.environ.get("R218_WINSAVE_ROWS", "8")))):
                                    try:
                                        _ridS = impl._tt_rid(_bS)
                                    except Exception:
                                        continue
                                    if _ridS is None:
                                        continue
                                    _pS = int(impl._tt_pref.get(_ridS, 0))
                                    _rS = int(impl._tt_rcnt.get(_ridS, 0))
                                    _sS = int(impl._tt_slot(_bS))
                                    _slS = int(_slW[_bS]) if _bS < len(_slW) else -1
                                    _idxS = impl._r198_win_idx(
                                        _ridS, _sS, _slS, _rS, int(_hk0.shape[2]), int(_hk0.shape[3]), _hk0.device
                                    )
                                    if _idxS is None:
                                        continue
                                    _win = _flatW.index_select(0, _idxS).detach().float().cpu()
                                    _tWS.save(
                                        {
                                            "b": _bS,
                                            "rid": str(_ridS),
                                            "slot": _sS,
                                            "sl": _slS,
                                            "pref": _pS,
                                            "rcnt": _rS,
                                            "X": (
                                                None
                                                if impl._tt_ring_org.get(_ridS) is None
                                                else int(impl._tt_ring_org.get(_ridS))
                                            ),
                                            "win": _win,
                                        },
                                        f"/tmp/r218_win_s{_si168}_b{_bS}.pt",
                                    )
                            with open("/tmp/r218_wincmp.jsonl", "a") as _fW:
                                _fW.write(_jW.dumps({"s": _si168, "rows": _rowsW}) + "\n")
                        except Exception as _eW:
                            with contextlib.suppress(Exception):
                                open("/tmp/r218_wincmp.jsonl", "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                                    "ERR {!r}\n".format(_eW)
                                )
                    torch.save(_snap168, f"/tmp/r168_outlse_p{_si168}.pt")
            except Exception as _e168:
                print("[R168-OUTLSE] proc snap fail: {}".format(_e168), flush=True)

        if _os.path.exists("/tmp/r218_xaudit_on"):
            # R218_XAUDIT (file-marker gated): the frozen ring-origin map AS THE READ
            # CONSUMERS SEE IT, once per replay step, keyed the same way they
            # key it (_tt_rid(b)). X_step must equal the X_init recorded at
            # 7859 for the same rid; any other value means the recent window
            # is read rotated by (X_step - X_init) % cap. Column 8 is the
            # DRIVER's first lookup key (md.request_ids[b], a STRING) to show
            # whether that lookup can hit the int-keyed map at all.
            try:
                import json as _jxa

                # NB: separate key from the INIT site's _R218_XA -- sharing it made the
                # init counter (3956 records) push n past the cap and the step probe
                # silently emitted nothing (b122 mK129: 3956 inits, 0 steps).
                _xaS = _R2XX_G.setdefault("_R218_XAS", {"n": 0})
                _xaS["n"] += 1
                if _xaS["n"] <= 48:
                    _orgA = getattr(impl, "_tt_ring_org", None) or {}
                    _prfA = getattr(impl, "_tt_pref", None) or {}
                    _rcnA = getattr(impl, "_tt_rcnt", None) or {}
                    _ridsA = list(getattr(md, "request_ids", None) or [])
                    _slA = [int(x) for x in (getattr(md, "seq_lens_list", None) or [])]
                    _rowsA = []
                    # ARTIFACT GUARD (R218 2026-10-17): _tt_rid(b) == _tt_cur_rids[b]
                    # raises IndexError when _tt_cur_rids is SHORTER than the batch
                    # (_r166_bind_slots sets it to attn_metadata.request_ids whenever
                    # that list is non-empty, without checking R144_TT_REQID). An
                    # earlier version of this probe caught that and substituted the int
                    # row index, which then MISSES the string-keyed map -- producing a
                    # bogus "rows 28..31 have X=-1 / no VRING origin" and two rounds of
                    # dead-end analysis. Now: fall back to the metadata's own string rid
                    # (the product's real key) and RECORD the length desync explicitly.
                    _curA = getattr(impl, "_tt_cur_rids", None)
                    for _bA in range(Bcap):
                        if _curA is not None and _bA < len(_curA):
                            _kA = _curA[_bA]
                        elif _bA < len(_ridsA):
                            _kA = _ridsA[_bA]
                        else:
                            _kA = _bA
                        _rowsA.append(
                            [
                                int(_bA),
                                str(_kA),
                                int(_slA[_bA]) if _bA < len(_slA) else -1,
                                str(_ridsA[_bA]) if _bA < len(_ridsA) else "",
                                int(_orgA[_kA]) if _kA in _orgA else -1,
                                int(_prfA[_kA]) if _kA in _prfA else -1,
                                int(_rcnA[_kA]) if _kA in _rcnA else -1,
                                (int(_orgA[_ridsA[_bA]]) if _bA < len(_ridsA) and _ridsA[_bA] in _orgA else -2),
                            ]
                        )
                    _slotHostA = []
                    for _b2 in range(Bcap):
                        try:
                            _slotHostA.append(int(impl._tt_slot(_b2)))
                        except Exception:
                            _slotHostA.append(-9)
                    with open("/tmp/r218_xaudit.jsonl", "a") as _fxa:
                        _fxa.write(
                            _jxa.dumps(
                                {
                                    "ev": "step",
                                    "n": _xaS["n"],
                                    "Bcap": int(Bcap),
                                    "cap": int(getattr(impl, "_TT_RECENT", -1)),
                                    "prefD": int(_me.get("prefp", -1)),
                                    "capD": int(_me.get("capw", -1)),
                                    "nprD": int(_me.get("npr", -1)),
                                    "cur_rids_len": (len(_curA) if _curA is not None else -1),
                                    "md_rids_len": len(_ridsA),
                                    "sls_len": len(_slA),
                                    "slot_dev": _me["slot"].tolist(),
                                    "slot_host": _slotHostA,
                                    "disc": _R2XX_G.setdefault("_R218_DISC", []).copy(),
                                    "rows": _rowsA,
                                }
                            )
                            + "\n"
                        )
                    _R2XX_G.setdefault("_R218_DISC", []).clear()
            except Exception as _eA:
                with contextlib.suppress(Exception):
                    open("/tmp/r218_xaudit.jsonl", "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                        "XAUDIT_ERR " + repr(_eA) + "\n"
                    )

    def _r196_idfill_memo(self, attn_metadata, set_key=False):
        """R196 P1 (active only under R196_EAGER_BATCH=1, default OFF):
        step-memo for the R167_IDFILL-v2 / R169_FLATFIX flat refresh inside
        _r164_bind_slots. The refresh's inputs (request_ids -> slots,
        seq_lens, bucket = f(len(rids))) are LAYER-INVARIANT within one
        forward: the 36 per-layer calls write byte-identical values 36x,
        each paying ~2 blocking torch.tensor H2Ds plus flat copies.
        Identity = (same attn_metadata OBJECT, same derived key): a strong
        ref to the metadata is held in the memo state so object identity
        cannot be spoofed by address reuse, and the key re-derives
        rid/slot/len content. set_key=True is called at the END of the
        refresh body (inside its try) so a mid-refresh failure re-runs on
        the next layer instead of memoizing a partial write.
        R203_METAFIX: the memo state is now a per-OBJECT entry LIST, not a
        single slot. One mixed step has TWO live metadata objects (the main
        batch + the R164 MIXDEC sub-batch copy); with a single slot they
        alternately evicted each other, so BOTH the main and the subset
        flat refresh re-ran on every layer (~72 refreshes/step, each with
        blocking H2D torch.tensor builds). Entries hold strong refs (same
        anti-address-reuse contract); the list is bounded and cleared when
        it grows past the cap (stale cross-step entries then simply miss)."""
        if _r144_ovl_os.environ.get("R196_EAGER_BATCH", "0") != "1":
            return False
        try:
            _rids = tuple(getattr(attn_metadata, "request_ids", None) or ())
            _sls = tuple(int(x) for x in (getattr(attn_metadata, "seq_lens_list", None) or ()))
            _sm = getattr(type(self), "_r164_slotmap", None) or {}
            _sl = tuple(_sm.get(r, i) for i, r in enumerate(_rids))
            _key = (_rids, _sl, _sls)
        except Exception:
            return False
        _multi = _r144_ovl_os.environ.get("R203_METAFIX", "0") == "1"
        if set_key:
            if _multi:
                _lst = getattr(type(self), "_r196_idfill_state2", None)
                if _lst is None:
                    type(self)._r196_idfill_state2 = _lst = []
                _lst[:] = [(md, k) for (md, k) in _lst if md is not attn_metadata]
                _lst.append((attn_metadata, _key))
                if len(_lst) > 8:
                    del _lst[0]
            else:
                type(self)._r196_idfill_state = (attn_metadata, _key)
            return False
        if _multi:
            _lst = getattr(type(self), "_r196_idfill_state2", None) or []
            return any(md is attn_metadata and k == _key for (md, k) in _lst)
        _st196 = getattr(type(self), "_r196_idfill_state", None)
        return _st196 is not None and _st196[0] is attn_metadata and _st196[1] == _key
