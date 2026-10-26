# SPDX-License-Identifier: Apache-2.0
"""HP-buffer state and request-slot management mixin (alloc / bind / release /
cross-request reuse; prefix+recent buffer ownership)."""

from __future__ import annotations

import contextlib
import os
import os as _r144_ovl_os
from typing import Any

import torch
import torch_npu

from vllm_ascend.attention.int2.ctx import (  # noqa: F401
    _OSCAR_CFG,
    _R2XX_G,
    _odiag_p,
)
from vllm_ascend.attention.int2.helpers import (  # noqa: F401
    _R170_ROWSLOT_ON,
    _oscar_dbg,
    _R166SlotAlloc,
)


class Int2SlotStateMixin:
    # Host-state annotations only: the _TT_* class constants, the shared lock,
    # the rotation tensors, vllm_config/key_cache plumbing, and the R113-R200
    # machinery attributes are owned by the backend assembly / sibling mixins.
    # Bare annotations create no runtime attributes; they pin types for
    # mypy --check-untyped-defs on the mixin bodies.
    _TT_MAX_SEQS: int
    _TT_MAX_HP: int
    _TT_RECENT: int
    _tt_ready: bool
    _tt_hp_k: dict[Any, Any]
    _tt_hp_v: dict[Any, Any]
    _tt_hp_k_shared: Any
    _tt_hp_v_shared: Any
    _lk: Any
    _oscar_R_k: Any
    _oscar_R_v: Any
    _oscar_name2ord: Any
    _r113_cpool_batched: Any
    _r137_rk_proof: Any
    _r145_carriers_cls: Any
    _r163_bsrpool: Any
    _r164_idfill_done: Any
    _r164_meta: Any
    _r164_slotctr: Any
    _r164_slotmap: Any
    _r164_srcarrier: Any
    _r164_tdbuf: Any
    _r166_alloc: Any
    _r170_hp_mirror: Any
    _r196_idfill_memo: Any
    _r198_idx_cache: Any
    _r200_srcfree: Any
    vllm_config: Any
    key_cache: Any

    def _ensure_tt(self, device):
        """Lazy-init three-tier HP buffers + per-seq state (one-time)."""
        if self._tt_ready:
            return
        self._tt_ready = True
        # R138: HP buffers keyed by the STABLE layer ordinal
        # (self._oscar_layer_ord(layer), name-parsed; was id(layer) in R137).
        # layer.layer_idx is unset in this vllm build -> getattr default 0 ->
        # all layers shared one buffer -> absolute-correctness divergence +
        # B2 cross-generate degeneracy.
        self._tt_hp_k = {}  # stable_ord -> [max_seqs, max_hp, H, D] bf16
        self._tt_hp_v = {}
        # R144 P0: per-request state keyed by STABLE request identity (rid),
        # not batch-row b. HP buffer dim-0 indexed by a reusable slot in
        # [0, max_seqs). See _tt_resolve_slots / _tt_rid / _tt_slot. Gated by
        # R144_TT_REQID (default OFF -> b-keyed path verbatim, G3 preserved).
        self._tt_pref = {}  # rid -> int (prefix_end)   [b -> int when OFF]
        self._tt_rcnt = {}  # rid -> int (hp_recent count)
        # R147 P0-B: rid -> last seq position this request's shared _tt_rcnt was
        # advanced for (once-per-step guard under R147_RCNT_FIX).
        self._tt_rcnt_seen = {}
        # R198 VRING (R198_VRING=1, default OFF): rid -> ring origin X. The
        # recent-window ring becomes VIRTUAL: token t of a request lives at
        # physical ring position pref + ((t - X) % cap), X set at prefill
        # init = max(pref, sl - cap). Kills the per-step physical slide
        # (~6GB/step of pure layout copies at B32); every window read goes
        # through rotated indices instead. Rows without an X (never
        # prefill-initialized under VRING) keep the legacy left-packed
        # layout via X=pref.
        self._tt_ring_org = {}
        # R218 STAGE: BF16 staging of in-flight (split) prefill prompts.
        self._r218_st_k = {}
        self._r218_st_v = {}
        self._r218_st_slot = {}  # HP slot -> stage idx
        self._r218_st_rid = {}  # stage idx -> rid
        self._r218_st_len = {}  # stage idx -> staged token count
        self._r218_st_clock = {}  # stage idx -> LRU tick
        self._r218_st_tick = [0]
        # per-step cache of rotated window index tensors (eager readers);
        # keyed (slot, sl, pref, X, rcnt) — cleared when it grows past 4096.
        type(self)._r198_idx_cache = {}
        if os.environ.get("R144_TT_REQID", "0") == "1":
            # R144 P0: capacity from vLLM runtime config (no hardcoded 4/8).
            self._tt_mseq = int(self.vllm_config.scheduler_config.max_num_seqs)
        else:
            # vLLM 0.31's V2 runner may append one dummy padding request to
            # query_start_loc (batch width = max_num_seqs + 1 in the worst
            # case), so the b-keyed HP pool needs one extra row. Keeping the
            # engine width as the lower bound also prevents a small env cap
            # from under-sizing the pool.
            _mns = int(getattr(self.vllm_config.scheduler_config, "max_num_seqs", 0) or 0)
            self._tt_mseq = max(self._TT_MAX_SEQS, 4, _mns + 1)
        # R205_HPFIT (default OFF): right-size the HP ring pool from the engine's
        # REAL concurrency instead of the env cap. The canonical
        # R109_TT_MAX_SEQS=96 over-provisions 3x for B<=32 legs: the pool is
        # [ms, mhp, nkv, hd] bf16 x2 x 36 layers = 14.5GB at ms=96/mhp=1024 vs
        # 4.8GB at ms=36 — the single largest non-KV consumer on the card
        # (B32@4K actual HBM 50.3GB = 15.3 weights + 7.1 KV + 14.5 HP + 11.4
        # graphs, npu-smi continuous-sampler proven), eating the INT2 KV
        # density win at long context. Shrunk cap keeps a MIGRATOR MARGIN:
        # the R170 rowslot permutation planner needs free temp slots
        # (stash-per-cycle + eviction parks) — cap == max_num_seqs is
        # deterministic fail-closed starvation (r205 smoke B32/TT32: 188
        # migrate failures/leg, whole-generation [first,0,0,...] degeneracy;
        # free=0 at every steady decode step). min(env, mns+margin): the env
        # value stays an upper bound, fail-open to the env cap on any config
        # read error.
        if os.environ.get("R205_HPFIT", "0") == "1":
            try:
                _mns205: Any = int(self.vllm_config.scheduler_config.max_num_seqs)
                _marg205: Any = int(os.environ.get("R205_HPFIT_MARGIN", "8"))
                self._tt_mseq = max(4, min(self._tt_mseq, _mns205 + _marg205))
            except Exception:
                _mns205 = _marg205 = None
            # engagement marker on the FILE channel (V1 EngineCore swallows
            # post-init stderr; r204f-proven pattern)
            try:
                with open("/tmp/r205_hpfit.log", "a") as _f205:
                    _f205.write("hpfit engaged mseq={} mns={} margin={}\n".format(self._tt_mseq, _mns205, _marg205))
            except Exception:
                pass
        self._tt_mhp = self._TT_MAX_HP
        # R144 P0: per-request slot bookkeeping keyed by STABLE req_id (string
        # from InputBatch.req_ids, explicit add/remove lifecycle via
        # SchedulerOutput). NOT block_tables[b][0] (physical block address --
        # recycles on prefix-cache collision / block-pool reuse, so two requests
        # sharing a prefix collide; unsafe). NOT LRU (a heuristic, not a
        # deterministic finish signal). Used only when R144_TT_REQID=1; inert OFF.
        self._tt_slotmap = {}  # req_id -> slot in [0, max_seqs)
        self._tt_free = set()  # freed slots reusable for new requests
        self._tt_next = 0  # next never-used slot ordinal
        self._tt_cur_slots = None  # per-call b -> slot list (None => use b)
        self._tt_cur_rids = None  # per-call b -> req_id list (None => use b)
        if _R170_ROWSLOT_ON:
            # R170_ROWSLOT: class-level aliases the pre-replay updater reads
            # (update_graph_params reaches this class, not the instance). The
            # dict objects are aliased by identity, so lazy per-layer
            # _tt_hp_bufs() inserts stay visible. Mirrored onto BOTH the
            # concrete impl class and the base (the updater is a classmethod
            # invoked through AscendAttentionBackendImpl).
            import vllm_ascend.attention.attention_v1 as _m170i

            for _c170 in dict.fromkeys([type(self), _m170i.AscendAttentionBackendImpl]):
                # deliberately dynamic class-attribute mirrors: the targets
                # include the foreign base class, so plain attribute syntax
                # cannot type-check against it
                setattr(_c170, "_r170_hpk", self._tt_hp_k)  # noqa: B010 - dynamic mirror by design
                setattr(_c170, "_r170_hpv", self._tt_hp_v)  # noqa: B010 - dynamic mirror by design
                setattr(_c170, "_r170_mseq", int(self._tt_mseq))  # noqa: B010 - dynamic mirror by design
                # R171 FIX (defect B): the legacy alias above is OVERWRITTEN by
                # every layer impl instance (N instances, each with its OWN
                # dict holding ONLY its own lid) — the class attr ends up
                # pointing at the LAST instance's dict, so _r170_rowslot_migrate
                # copied just ONE layer's rings; layers 0..N-2 kept the previous
                # generation's window at their row slots (the E27/E28
                # foreign-KV walk). Registry: append EVERY instance's (k,v)
                # dict pair once; the migrator unions lids across all of them.
                _reg171 = getattr(_c170, "_r171_hp_reg", None)
                if _reg171 is None:
                    # create the registry in a local and setattr that: the
                    # mirror target includes the foreign base class, where
                    # plain attribute syntax cannot type-check, and the
                    # stored list stays identity-aliased with _reg171
                    _reg171 = []
                    setattr(_c170, "_r171_hp_reg", _reg171)  # noqa: B010 - dynamic mirror by design
                if not any(k is self._tt_hp_k for k, _v in _reg171):
                    _reg171.append((self._tt_hp_k, self._tt_hp_v))
        # req_ids that currently have an assigned slot (the "live" set). A
        # req_id leaving this set == finished/preempted (explicit finish signal:
        # known-but-absent-from-current-batch) -> free its slot + drop its
        # _tt_pref/_tt_rcnt; HP region zeroed by the next occupant's genuine
        # prefill (R137_HP_ZERO, A->B->A safe). vLLM V1 keeps a live request in
        # InputBatch.req_ids every step, so absence == finish/preempt; a
        # re-prefilled (recomputed) request correctly reinitializes its state.
        self._tt_known = set()
        # R144 P0 dynamic-lifecycle instrumentation (gated by R144_TT_DEBUG=1).
        _tt_dbg = os.environ.get("R144_TT_DEBUG", "0") == "1"
        self._tt_dbg = _tt_dbg
        self._tt_fin = 0  # finish events (req_id left the batch)
        self._tt_reuse = 0  # slot reuse events (new req_id got a freed slot)
        self._tt_log = []  # [(event, req_id_prefix, slot), ...]
        # R144 P1 StageV2: per-layer persistent device carriers keyed by slot
        # (qrot/out/lse). Pre-allocated once, reused every step -> alloc=0/cat=0/
        # zeros=0 per step; addresses stable across replays (graph-safe). Gated
        # by R144_BATCHED_STAGE_V2 (default OFF; existing path preserved).
        self._r144_carriers = {}
        # R145: fixed-capacity tensor-ABI carriers keyed by capacity bucket (shared
        # across layers; one graph per bucket). Default OFF (R145_TENSOR=0).
        # R150 P3: hoisted to CLASS level — 36 per-layer impl instances would
        # otherwise each allocate their own carrier set (36x HBM). All
        # instances alias the same dict; buckets stay lazy.
        if os.environ.get("R150_SHARED_CARRIERS", "1") == "1":
            if not hasattr(type(self), "_r145_carriers_cls"):
                type(self)._r145_carriers_cls = {}
            self._r145_carriers = type(self)._r145_carriers_cls
        else:
            self._r145_carriers = {}
        _oscar_dbg(
            f"[3T] init max_seqs={self._tt_mseq} max_hp={self._tt_mhp} reqid={os.environ.get('R144_TT_REQID', '0')}",
            flush=True,
        )

    def _tt_hp_bufs(self, lid, device, dtype):
        """Lazily allocate per-layer HP BF16 buffers, keyed by the stable layer
        ordinal (lid = self._oscar_layer_ord(layer))."""
        if lid not in self._tt_hp_k:
            ms, mh = self._tt_mseq, self._tt_mhp
            nk = getattr(self, "num_kv_heads", 8)
            hd = getattr(self, "head_size", 128)
            self._tt_hp_k[lid] = torch.zeros(ms, mh, nk, hd, dtype=dtype, device=device)
            self._tt_hp_v[lid] = torch.zeros(ms, mh, nk, hd, dtype=dtype, device=device)
            # R170_RINGDUMP (default OFF): class-level mirror so the
            # acl_graph consumption-point probe can read the per-instance
            # rings (instance attr; trap-21 class-attr lesson).
            try:
                import os as _os170m

                if _os170m.environ.get("R170_RINGDUMP", "0") == "1":
                    _m = getattr(type(self), "_r170_hp_mirror", None)
                    if _m is None:
                        type(self)._r170_hp_mirror = _m = {}
                    _m.setdefault(lid, (self._tt_hp_k[lid], self._tt_hp_v[lid]))
            except Exception:
                pass
            # R170_RINGDUMP support: class-level alias of the instance dict
            # (same object, no copy; probe-only visibility).
            type(self)._tt_hp_k_shared = self._tt_hp_k
            type(self)._tt_hp_v_shared = self._tt_hp_v
            _oscar_dbg(f"[3T] alloc HP lid={lid} shape=({ms},{mh},{nk},{hd}) {dtype}", flush=True)
        return self._tt_hp_k[lid], self._tt_hp_v[lid]

    def _tt_write_hp(self, lid, b, start, kv_rot):
        # R165 ROOT FIX (R164_PREFILL_SAFE, default OFF): the slice-assign
        # `hk[b, s:s+n] = kv` dispatches a COPY-ENGINE D2D async to the compute
        # stream; kv is a VIEW whose storage the caching allocator can recycle
        # BEFORE the copy executes => the write lands as ZEROS (device-proven:
        # dst8=0.0 x36 with correct enqueue-time src8). Safe mode: clone the
        # source (compute-stream copy to owned storage) + keep it alive past a
        # stream sync so the engine copy reads valid memory.
        if _r144_ovl_os.environ.get("R164_PREFILL_SAFE", "0") == "1":
            try:
                _ka = globals().setdefault("_R164_WKA", [])
                _k = (kv_rot[0].clone(), kv_rot[1].clone())
                _dst = self._tt_hp_k[lid]
                # v2: COMPUTE-kernel writes (elementwise mul) — stream-ordered
                # on the compute stream; the copy-engine path (v1) has
                # unbounded async latency and read recycled storage.
                _n0 = _k[0].shape[0]
                _n1 = _k[1].shape[0]
                _dst[b, start : start + _n0] = _k[0] * torch.ones_like(_k[0])
                self._tt_hp_v[lid][b, start : start + _n1] = _k[1] * torch.ones_like(_k[1])
                _ka.append(_k)
                if len(_ka) > 256:
                    del _ka[:128]
                return
            except Exception:
                pass
        # R165 device-side write verification: enqueue a checksum of the JUST-
        # written destination rows into a passive side buffer (same stream =>
        # executes right after the copy). If the post-write checksum already
        # differs from src8, the write copy itself is the corruption point.
        if (
            _r144_ovl_os.environ.get("R164_WCHK", "0") == "1"
            and kv_rot
            and kv_rot[0] is not None
            and kv_rot[0].shape[0] > 0
        ):
            try:
                _dst = self._tt_hp_k[lid]
                _cb = _R2XX_G.setdefault("_R164_WCHKBUF", None)
                if _cb is None:
                    _cb = torch.zeros(8192, 2, dtype=torch.float32, device=_dst.device)
                    _R2XX_G["_R164_WCHKBUF"] = _cb
                    _R2XX_G["_R164_WCHKN"] = [0]
                _n = _R2XX_G["_R164_WCHKN"][0]
                if _n < 8192:
                    _cb[_n, 0] = lid * 1000 + b
                    _cb[_n, 1] = _dst[b, start : start + 8].float().sum()
                    _R2XX_G["_R164_WCHKN"][0] = _n + 1
                    if _n % 2048 == 2047:
                        # rolling dump segment (one sync per 2048 writes —
                        # covers the FINAL leg's slot window too)
                        try:
                            _npr = torch.npu  # torch_npu at module top registers the submodule

                            _npr.synchronize()
                            import json as _jr

                            with open("/tmp/r164_wchk.jsonl", "a") as _fr2:
                                for _i in range(_n - 2047, _n + 1):
                                    _fr2.write(
                                        _jr.dumps(
                                            {
                                                "i": _i,
                                                "key": int(_cb[_i, 0].item()),
                                                "dst8": round(float(_cb[_i, 1].item()), 1),
                                            }
                                        )
                                        + "\n"
                                    )
                        except Exception:
                            pass
            except Exception:
                pass
        # R165 write-audit: every ring write in (mixed) prefill steps, one line
        # per call — layer, slot, start_row, nrows. Correlate with the decode
        # row's slot to find the 64-row chunk bleed (blocks 49-53 defect).
        if _r144_ovl_os.environ.get("R164_WRITEAUDIT", "0") == "1":
            try:
                n = kv_rot[0].shape[0] if kv_rot and kv_rot[0] is not None else -1
                # content checksum of the SOURCE rows at write time — decides
                # garbage-source vs read-race (per-layer garbage hypothesis)
                _cs = round(float(kv_rot[0][:8].float().sum().item()), 1) if n > 0 else None
                _wt = self._tt_hp_k.get(lid)
                _w = _wt.shape[1] if _wt is not None else -1
                _tid = id(_wt) % 1000000 if _wt is not None else -1
                with open("/tmp/r164_writes.log", "a") as _wf:
                    _wf.write(
                        f"W lid={lid} impl={id(self) % 100000:05d} tid={_tid} "
                        f"slot={b} start={start} nrows={n} w={_w} src8={_cs}\n"
                    )
            except Exception:
                pass
        """Write rotated kv_rot[0..n] to HP buffer at positions [start, start+n)."""
        hk, hv = self._tt_hp_bufs(lid, kv_rot[0].device, kv_rot[0].dtype)
        n = kv_rot[0].shape[0]
        # R166_STREAMFIX (default OFF): COMPUTE-stream ring write. The plain
        # `.copy_()` slice-assign dispatches a COPY-ENGINE D2D that is async
        # to the compute stream (R165 copy_d2d_baseformat_opapi evidence) and
        # is therefore NOT covered by the pre-replay default-stream
        # synchronize — the first decode replay's S/R read can beat the
        # prefill ring write (R166 causal proof: probes forcing syncs at
        # these writes -> B1 canonical 3/3; unperturbed -> 0/3, streams
        # diverge at low-margin tokens). Elementwise out= a destination VIEW
        # launches a vector kernel ON the compute stream: stream-ordered
        # write AND stream-ordered source lifetime (recycled key_rot storage
        # is only reused by later same-stream ops). No host syncs.
        _sfmode = _r144_ovl_os.environ.get("R166_STREAMFIX", "0")
        if _sfmode in ("1", "ring", "both"):
            try:
                torch.mul(kv_rot[0], 1.0, out=hk.narrow(0, b, 1).narrow(1, start, n)[0])
                torch.mul(kv_rot[1], 1.0, out=hv.narrow(0, b, 1).narrow(1, start, n)[0])
                return
            except Exception:
                pass
        hk[b, start : start + n].copy_(kv_rot[0])
        hv[b, start : start + n].copy_(kv_rot[1])

    def _tt_rid(self, b):
        """Stable request identity for batch row b. When R144_TT_REQID=1,
        _tt_resolve_slots mapped row b -> its req_id (the stable vLLM request
        identity plumbed from InputBatch.req_ids via AscendMetadata.request_ids).
        When OFF, returns b (verbatim b-keyed path)."""
        if self._tt_cur_rids is not None:
            return self._tt_cur_rids[b]
        return b

    def _tt_slot(self, b):
        """HP-buffer row (slot) for batch row b. When R144_TT_REQID=1, row b ->
        its request's reusable slot in [0, max_seqs). When OFF, returns b."""
        if _R170_ROWSLOT_ON:
            # R170_ROWSLOT: while the ACL graph capture runs, bake the
            # ROW-INDEX contract — the replayed graph's row b must read
            # hk[row b], independent of the runtime rid->slot binding
            # (the pre-replay migrator keeps row content at row slots).
            try:
                if torch.npu.is_current_stream_capturing():
                    return b
            except Exception:
                pass
        if self._tt_cur_slots is not None:
            return self._tt_cur_slots[b]
        return b

    def _r166_bind_slots(self, attn_metadata, device, dtype, layer):
        """R166 product slot allocator (see contract in _r164_bind_slots).
        Class-level state so every per-layer impl instance shares ONE map
        (per-instance maps split the view across the 36 impls — the R165
        instance-mismatch audit class). Ring zero-on-reuse is done ONCE per
        acquire per (layer impl, slot): each impl zeroes its own ring rows
        (rings are per-instance); the allocator records which impls have
        zeroed via a per-(slot,gen) stamp set to keep it once-only."""
        import os as _os

        A = getattr(type(self), "_r166_alloc", None)
        lid = self._oscar_layer_ord(layer)
        if A is None or A.cap != int(self._tt_mseq):
            # (re)init: capacity from the REAL mseq (max_num_seqs / TT_MAX_SEQS)
            # R205 note: do NOT reserve slot 0 here — the R170 rowslot migrator's
            # row contract rebinds decode row b to SLOT b (0 included) and its
            # cover invariant requires every slot free-or-owned; a reserved row
            # fail-closes every plan (r205n_h1: 254 PREMIG_FAIL -> eager
            # fallback every step, tpot 183 vs 128).
            type(self)._r166_alloc = A = _R166SlotAlloc(int(self._tt_mseq))
        rids = getattr(attn_metadata, "request_ids", None)
        if not rids:
            return
        finished = getattr(attn_metadata, "finished_req_ids", None) or set()
        preempted = getattr(attn_metadata, "preempted_req_ids", None) or set()
        gone = set(finished) | set(preempted)
        # 1) explicit release (owner-checked) — absence is NOT finish
        for r in gone:
            A.release(r)
        # 2) acquire for new rids; zero-on-reuse runs ONCE PER IMPL: the
        # allocator is class-level (shared across the per-layer impls), so
        # only the FIRST impl of a step acquires — every impl must still
        # stamp-and-zero its OWN ring rows for a reused slot, which is what
        # the per-(slot, gen, lid) stamp set exists for
        zero_pairs = []  # (lid, slot) whose ring rows must be zeroed now
        for r in rids:
            if r not in A.rid2slot:
                slot, _gen, _new = A.acquire(r)
            else:
                slot = A.rid2slot[r]
            gen = A.gen[slot]
            if gen > 0 and (slot, gen, lid) not in A.zeroed:
                A.zeroed.add((slot, gen, lid))
                zero_pairs.append((lid, slot))
        # 2b) MIRROR into _r164_slotmap: the graph replay driver resolves
        # slot ids from that dict (empty dict => row-index fallback => the
        # graph reads never-written rings => [261,x,0,0,...] degenerate rows
        # — R166 ml_matrix evidence).
        _sm = getattr(type(self), "_r164_slotmap", None)
        if _sm is None:
            type(self)._r164_slotmap = _sm = {}
        for r in gone:
            _sm.pop(r, None)
        for r in rids:
            _sm[r] = A.rid2slot[r]
        # 3) bind current rows
        self._tt_cur_rids = list(rids)
        self._tt_cur_slots = [A.rid2slot[r] for r in rids]
        # 4) ring zero-once for reused slots (this impl's rows only)
        if zero_pairs:
            try:
                hk, hv = self._tt_hp_bufs(zero_pairs[0][0], device, dtype or torch.bfloat16)
                for _l, _s in zero_pairs:
                    if _l == lid:
                        hk[_s].zero_()
                        hv[_s].zero_()
            except Exception:
                pass
        # 5) tier state reset for reused requests (pref/rcnt fresh at prefill)
        for r in rids:
            if getattr(A, "just_acquired", None) and r in A.just_acquired:
                self._tt_pref.pop(r, None)
                self._tt_rcnt.pop(r, None)
                self._tt_rcnt_seen.pop(r, None)
                self._tt_ring_org.pop(r, None)  # R198 VRING
        A.just_acquired = set()
        # 6) gated invariant check (cheap host-side sets; fail-loud)
        if _os.environ.get("R166_SLOTCHECK", "0") == "1":
            viol = A.check()
            if viol:
                raise RuntimeError("[R166-SLOTCHECK] invariant violation: " + "; ".join(viol))

    def _r164_bind_slots(self, attn_metadata, device=None, dtype=None, layer=None):
        """R164_SLOTMAP (env-gated): STABLE PER-REQUEST ring slots.

        Root cause fixed here: with split-arrival batches, a prefill-only step
        indexes its metadata rows 0..k-1, which are NOT the persistent engine
        rows — writing ring slot = md-row clobbers another request's ring and
        leaves later engine rows unwritten (graph replay then reads a
        cross-wired/zero ring: SR_KV 2.23M vs 861K at the same position).
        Bind md rows -> arrival-order slots via AscendMetadata.request_ids so
        every request owns exactly one stable HP slot across prefill/decode/
        graph capture. Model-neutral: no batch/layer/head constants."""
        # ---- R166 product allocator (R166_SLOTALGO=1, default OFF): real
        # free-set with owner/generation semantics. Replaces the monotonic
        # no-reuse policy (monotonic capacity must cover the process-lifetime
        # request count — infeasible for single-engine multi-leg matrices and
        # forbidden as a final design). Contract (directive §4):
        #   identity   = AscendMetadata.request_ids (InputBatch.req_ids);
        #   release    = ONLY explicit finished_req_ids | preempted_req_ids
        #                (absence is NOT finish — no LRU, no grace heuristics);
        #   acquire    = pop from the real free set (never len(map));
        #   exhaustion = fail-closed;
        #   reuse      = zero the ring rows ONCE at acquire + bump generation;
        #   stale-gen write (a write for slot s arriving with an older
        #                generation than the slot's current) = fail-loud;
        #   invariants = req->slot and slot->req bijection, occupied unique,
        #                free^occupied empty, free|occupied = full capacity.
        import os as _os

        # ---- EAGER PRE-ALLOCATION (visibility probe r164_visibility_probe:
        # tensors allocated DURING the capture region — on ANY stream — are
        # graph-pool and are RESET to capture-time contents at every replay;
        # only eagerly-pre-allocated tensors carry live content into the
        # graph). The first EAGER forward therefore creates, ONCE:
        #   (a) the HP rings for EVERY layer ordinal, and
        #   (b) the R164 metadata flat for EVERY graph bucket,
        # both in the REGULAR device pool. The capture side then only
        # REFERENCES them (fail-closed if a bucket was not pre-created).
        # PER-INSTANCE guard: each of the N layer impls owns its own
        # _tt_hp_k/_tt_hp_v dicts — a class-level guard would prealloc only
        # the FIRST impl's rings and leave the rest to lazy in-capture
        # allocation (graph pool -> frozen at replay).
        if (
            not getattr(self, "_r164_prealloc_done", False)
            and _os.environ.get("R164_BATCHED_STAGE", "0") == "1"
            and device is not None
        ):
            self._r164_prealloc_done = True
            _mc = self.vllm_config.model_config
            _nl = int(getattr(_mc, "num_hidden_layers", 0) or 0)
            if _nl <= 0:
                _nl = int(getattr(getattr(_mc, "hf_text_config", None), "num_hidden_layers", 0) or 0)
            if _nl <= 0:
                raise RuntimeError("[R164] num_hidden_layers not derivable (fail-closed)")
            if layer is None:
                raise RuntimeError("[R164] ring prealloc requires the layer handle (fail-closed)")
            # ONLY this impl's OWN ring (each impl owns exactly one ordinal;
            # allocating all ordinals per instance = N^2 memory).
            self._tt_hp_bufs(self._oscar_layer_ord(layer), device, dtype or torch.bfloat16)
            # NOTE: R167_IDFILL v2 lives AFTER the prealloc block (per-forward
            # execution) — see the block following the prealloc gate below.
            _bsz = int(self.vllm_config.cache_config.block_size)
            _mml = int(getattr(_mc, "max_model_len", 0) or 0)
            if _mml <= 0:
                raise RuntimeError("[R164] max_model_len not derivable (fail-closed)")
            _prefp = _OSCAR_CFG.prefix_window
            _rcntp = int(self._TT_RECENT)
            _nbmp = (_mml - _prefp - _rcntp + _bsz - 1) // _bsz
            if _nbmp <= 0:
                raise RuntimeError("[R164] nb_max<=0 (fail-closed)")
            _mp2 = getattr(type(self), "_r164_meta", None)
            if _mp2 is None:
                type(self)._r164_meta = _mp2 = {}
            _bset = {
                int(x)
                for x in _os.environ.get("R164_BCAPS", _os.environ.get("R116_CG_SIZES", "1,2,4,8,16"))
                .replace(" ", "")
                .split(",")
                if x != ""
            }
            with contextlib.suppress(Exception):
                _bset.add(int(self.vllm_config.scheduler_config.max_num_seqs))
            for _bcap in sorted(_bset):
                if _bcap <= 0 or _bcap in _mp2:
                    continue
                _nprp2 = _prefp + _rcntp
                _capw = _rcntp  # slide window == recent capacity
                _fl = torch.zeros(
                    _bcap * _nbmp + 3 * _bcap + _bcap * _nprp2 + 2 * _bcap * (_capw - 1) + _bcap,
                    dtype=torch.int64,
                    device=device,
                )
                # R199_PRREAD: persistent S/R paged block table (int32, the
                # FIA contract; separate from the int64 flat). Row b holds the
                # 128-token block ids of slot(b)'s ring row IN PHYSICAL ORDER —
                # the arange init IS the identity pattern (slot=b), matching
                # the R167_IDFILL first-replay discipline.
                _nbSR199 = (_nprp2 + 127) // 128
                _pg199p = (
                    _os.environ.get("R199_PRREAD", "0") == "1" and _nprp2 % 128 == 0 and (_prefp + _capw) % 128 == 0
                )
                _mp2[_bcap] = {
                    "bt_sr": (
                        torch.arange(_bcap * _nbSR199, dtype=torch.int32, device=device).view(_bcap, _nbSR199)
                        if _pg199p
                        else None
                    ),
                    "nb_sr": int(_nbSR199) if _pg199p else 0,
                    "pg199": bool(_pg199p),
                    "flat": _fl,
                    "nb_max": int(_nbmp),
                    "npr": int(_nprp2),
                    "ping": 0,
                    "bt_mirror": _fl[: _bcap * _nbmp].view(_bcap, _nbmp),
                    "bt_mirror_flat": _fl[: _bcap * _nbmp],
                    "askv": _fl[_bcap * _nbmp : _bcap * _nbmp + _bcap],
                    "asq": _fl[_bcap * _nbmp + _bcap : _bcap * _nbmp + 2 * _bcap],
                    "slot": _fl[_bcap * _nbmp + 2 * _bcap : _bcap * _nbmp + 3 * _bcap],
                    "idx_sr": _fl[_bcap * _nbmp + 3 * _bcap : _bcap * _nbmp + 3 * _bcap + _bcap * _nprp2],
                    "p_src": _fl[
                        _bcap * _nbmp + 3 * _bcap + _bcap * _nprp2 : _bcap * _nbmp
                        + 3 * _bcap
                        + _bcap * _nprp2
                        + _bcap * (_capw - 1)
                    ],
                    "p_dst": _fl[
                        _bcap * _nbmp + 3 * _bcap + _bcap * _nprp2 + _bcap * (_capw - 1) : _bcap * _nbmp
                        + 3 * _bcap
                        + _bcap * _nprp2
                        + 2 * _bcap * (_capw - 1)
                    ],
                    "p_new": _fl[_bcap * _nbmp + 3 * _bcap + _bcap * _nprp2 + 2 * _bcap * (_capw - 1) :],
                    "capw": int(_capw),
                    "prefp": int(_prefp),
                    # R167: TRUE ring row width (flat-index stride; see R167_STRIDEFIX).
                    "mhp": int(self._tt_mhp),
                    "askv_pin": (
                        torch.zeros(_bcap, dtype=torch.int64).pin_memory(),
                        torch.zeros(_bcap, dtype=torch.int64).pin_memory(),
                    ),
                    "slot_pin": (
                        torch.arange(_bcap, dtype=torch.int64).pin_memory(),
                        torch.arange(_bcap, dtype=torch.int64).pin_memory(),
                    ),
                }
            # srcarrier + bsrpool per bucket (bucket-shared across layers;
            # EAGER allocation mandatory — capture-region allocs are
            # graph-pool frozen at replay).
            _nprp = _prefp + _rcntp
            _Hp = int(getattr(self, "num_heads", 0) or 0)
            _Dhp = int(getattr(self, "head_size", 0) or 0)
            _nkvp = int(getattr(self, "num_kv_heads", 0) or 0)
            if min(_Hp, _Dhp, _nkvp) <= 0:
                raise RuntimeError("[R164] head geometry not derivable (fail-closed)")
            _dtp = getattr(_mc, "dtype", None) or torch.bfloat16
            _scp3 = getattr(type(self), "_r164_srcarrier", None)
            if _scp3 is None:
                type(self)._r164_srcarrier = _scp3 = {}
            _srp3 = getattr(type(self), "_r163_bsrpool", None)
            if _srp3 is None:
                type(self)._r163_bsrpool = _srp3 = {}
            _bset = {
                int(x)
                for x in _os.environ.get("R164_BCAPS", _os.environ.get("R116_CG_SIZES", "1,2,4,8,16"))
                .replace(" ", "")
                .split(",")
                if x != ""
            }
            with contextlib.suppress(Exception):
                _bset.add(int(self.vllm_config.scheduler_config.max_num_seqs))
            for _bcap in sorted(_bset):
                if _bcap <= 0:
                    continue
                # R200_SRCFREE (default OFF): under PRREAD the in-graph S/R
                # carrier is DEAD (the paged FIA consumes the ring rows in
                # place) — skip allocating it (and the legacy contiguous
                # bsrpool entry below, whose workspace is sized FROM the
                # carrier) when the bucket's pg199 geometry holds and no
                # diagnostic needs them. R199_SELFCHECK recomputes the legacy
                # rotated-gather path, so it re-engages the allocation.
                _sf200 = (
                    _os.environ.get("R200_SRCFREE", "0") == "1"
                    and _mp2[_bcap]["pg199"]
                    and _os.environ.get("R199_SELFCHECK", "0") != "1"
                )
                if _sf200:
                    type(self)._r200_srcfree = True
                if (_bcap,) not in _scp3 and not _sf200:
                    _scp3[(_bcap,)] = {
                        "k": torch.zeros(_bcap, _nkvp, _nprp, _Dhp, dtype=_dtp, device=device),
                        "v": torch.zeros(_bcap, _nkvp, _nprp, _Dhp, dtype=_dtp, device=device),
                    }
                # R198 ISTAGE: mid carrier pool per (lid, Bcap) — EAGER
                # allocation, same contract as the srcarrier above (the
                # restager writes these between replays; a pool first created
                # inside the capture region interacts with the aclgraph
                # private pool and intermittently corrupts the carriers at
                # replay — device-proven r1-healthy/r2-degenerate on the
                # capture-time side-stream variant).
                if (
                    _os.environ.get("R198_ISTAGE", "0") == "1"
                    and getattr(self, "key_cache", None) is not None
                    and _bcap <= 32
                ):
                    # (_bcap<=32: only the graph-captured bucket sizes — the
                    # CG ladder; preallocating the max_num_seqs=96 bucket
                    # would add ~13GB of carriers across 36 layers for a
                    # bucket that is never captured)
                    _cp198 = getattr(type(self), "_r113_cpool_batched", None)
                    if _cp198 is None:
                        type(self)._r113_cpool_batched = _cp198 = {}
                    _ck198p = (self._oscar_layer_ord(layer), _bcap)
                    if _ck198p not in _cp198:
                        _pck198 = int(self.key_cache.shape[-1])
                        _S198 = int(_nbmp) * int(_bsz)
                        _cp198[_ck198p] = {
                            "nb_max": int(_nbmp),
                            "S_pad": _S198,
                            "B": _bcap,
                            "k": torch.zeros(_bcap * _nbmp, _nkvp, _bsz, _pck198, dtype=torch.int8, device=device),
                            "v": torch.zeros(_bcap * _nbmp, _nkvp, _bsz, _pck198, dtype=torch.int8, device=device),
                            "ks": torch.zeros(_bcap, _nkvp, _S198, dtype=torch.bfloat16, device=device),
                            "ko": torch.zeros(_bcap, _nkvp, _S198, dtype=torch.bfloat16, device=device),
                            "vs": torch.zeros(_bcap, _nkvp, _S198, dtype=torch.bfloat16, device=device),
                            "vo": torch.zeros(_bcap, _nkvp, _S198, dtype=torch.bfloat16, device=device),
                            "askv_dev": torch.zeros(_bcap, dtype=torch.int64, device=device),
                            "asq_dev": torch.ones(_bcap, dtype=torch.int64, device=device),
                            "bt": torch.arange(_bcap * _nbmp, dtype=torch.int32, device=device)
                            .view(_bcap, _nbmp)
                            .contiguous(),
                            "bt_src": None,
                        }
                _bk3 = (_bcap, _nprp, _Hp, _Dhp, str(device))
                if _bk3 not in _srp3 and not _sf200:
                    _kb3 = _scp3[(_bcap,)]["k"]
                    _ws3 = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                        query=torch.zeros(_bcap, _Hp, 1, _Dhp, dtype=_dtp, device=device),
                        key=_kb3,
                        value=_kb3,
                        input_layout="BNSD",
                        block_size=0,
                        actual_seq_lengths=[1] * _bcap,
                        actual_seq_lengths_kv=[_nprp] * _bcap,
                        num_heads=_Hp,
                        num_key_value_heads=_nkvp,
                        sparse_mode=0,
                        pre_tokens=2147483647,
                        next_tokens=2147483647,
                        scale=float(_Hp and (1.0 / (_Dhp**0.5))),
                    )
                    _srp3[_bk3] = (
                        _ws3,
                        torch.empty(_bcap, _Hp, 1, _Dhp, dtype=torch.bfloat16, device=device),
                        torch.empty(_bcap, _Hp, 1, 1, dtype=torch.float32, device=device),
                    )
                # R199_PRREAD: paged-FIA pool entry (key prefixed "pg"). The
                # workspace is sized from THIS impl's ring viewed as a paged
                # pool ([ms*mhp/128, 128, nkv*D]) with the real bt_sr —
                # shape-identical across layers, so the first impl's sizing
                # serves every layer (same contract as _bk3 above).
                _pg199q = _mp2[_bcap]["pg199"]
                _bk3q = ("pg", _bcap, _nprp, _Hp, _Dhp, str(device))
                if _pg199q and _bk3q not in _srp3:
                    _hkq = self._tt_hp_k.get(self._oscar_layer_ord(layer))
                    if _hkq is None:
                        raise RuntimeError("[R199] ring not preallocated (fail-closed)")
                    _kvq = _hkq.view(-1, 128, _nkvp * _Dhp)
                    _ws3q = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                        query=torch.zeros(_bcap, _Hp, 1, _Dhp, dtype=_dtp, device=device),
                        key=_kvq,
                        value=_kvq,
                        input_layout="BNSD",
                        block_table=_mp2[_bcap]["bt_sr"],
                        block_size=128,
                        actual_seq_qlen=[1] * _bcap,
                        actual_seq_kvlen=[_nprp] * _bcap,
                        num_query_heads=_Hp,
                        num_key_value_heads=_nkvp,
                        sparse_mode=0,
                        pre_tokens=2147483647,
                        next_tokens=2147483647,
                        softmax_scale=float(_Hp and (1.0 / (_Dhp**0.5))),
                    )
                    _srp3[_bk3q] = (
                        _ws3q,
                        torch.empty(_bcap, _Hp, 1, _Dhp, dtype=torch.bfloat16, device=device),
                        torch.empty(_bcap, _Hp, 1, 1, dtype=torch.float32, device=device),
                    )
            # R167: EAGER tier-probe buffer (graph-pool freeze law: a buffer
            # lazily allocated inside the capture region reads ZERO at eager
            # post-replay reads — the R166 TIERDUMP-atexit/env blocker).
            if _os.environ.get("R167_STAGEPROBE", "0") == "1" or _os.environ.get("R164_TIERDUMP", "0") == "1":
                _bmax167 = max(_bset)
                type(self)._r164_tdbuf = torch.zeros(_nl, _bmax167, 10, dtype=torch.float32, device=device)
            # R167_IDFILL (first-generation fix): the first replay of each
            # bucket can NEVER see the driver fills (post-replay update order;
            # the captured E_inputs_ready wait no-ops on a never-recorded
            # event). Pre-fill SAFE IDENTITY values (slot=row, askv=0 =>
            # mid masked neutral, bt=0) + the idx/p arrays: replay-1 reads the
            # identity slots — which for the FIRST generation ARE the real
            # slots (monotonic allocator starts at 0). NO eager event record
            # (NPU ExternalEvent is 1:1 record<->wait).
            if _os.environ.get("R167_IDFILL", "0") == "1":
                for _bcapI in sorted(_bset):
                    if _bcapI <= 0 or _bcapI in getattr(type(self), "_r164_idfill_done", set()):
                        continue
                    _dfd = getattr(type(self), "_r164_idfill_done", None)
                    if _dfd is None:
                        type(self)._r164_idfill_done = _dfd = set()
                    _dfd.add(_bcapI)
                    _meI = _mp2[_bcapI]
                    _slotI = torch.arange(_bcapI, dtype=torch.int64, device=device)
                    _meI["slot"].copy_(_slotI)
                    _meI["askv"].fill_(0)
                    _meI["asq"].fill_(1)
                    _meI["bt_mirror"].fill_(0)
                    _nprI = int(_nprp2)
                    _capI = int(_capw)
                    _prefI = int(_prefp)
                    _mhpI = int(self._tt_mhp)
                    _meI["t_const"] = torch.arange(_nprI, dtype=torch.int64, device=device)
                    _meI["idx_sr"].copy_((_slotI.view(-1, 1) * _mhpI + _meI["t_const"].view(1, -1)).reshape(-1))
                    _meI["t_w1"] = torch.arange(_prefI + 1, _prefI + _capI, dtype=torch.int64, device=device)
                    _meI["t_w0"] = torch.arange(_prefI, _prefI + _capI - 1, dtype=torch.int64, device=device)
                    _meI["p_src"].copy_((_slotI.view(-1, 1) * _mhpI + _meI["t_w1"].view(1, -1)).reshape(-1))
                    _meI["p_dst"].copy_((_slotI.view(-1, 1) * _mhpI + _meI["t_w0"].view(1, -1)).reshape(-1))
                    _meI["p_new"].copy_(_slotI * _mhpI + (_prefI + _capI - 1))
        # R167_IDFILL v2: refresh the flats at every PREFILL-state eager
        # forward with the REAL slots (bind_slots sees the new generation's
        # slotmap). Root of the per-generation first-replay stale-flat race
        # (B4-l0 rows 2,3 token-3 flips): each generation's first replay can
        # read the PREVIOUS generation's fills when it loses the submission
        # race — the real rings then miss one slide append. Pre-filling the
        # REAL slots/idx at prefill time makes that replay correct regardless
        # of how the race lands.
        if (
            os.environ.get("R167_IDFILL", "0") == "1"
            and os.environ.get("R164_BATCHED_STAGE", "0") == "1"
            and not (os.environ.get("R203_METAFIX", "0") == "1" and getattr(attn_metadata, "_r203_is_sub", False))
            and getattr(attn_metadata, "attn_state", None) is not None
            and str(getattr(attn_metadata, "attn_state", "")).endswith("Prefill")
            and device is not None
            and not self._r196_idfill_memo(attn_metadata)
        ):
            try:
                _mpI2 = getattr(type(self), "_r164_meta", None) or {}
                _ridsI2 = list(getattr(attn_metadata, "request_ids", None) or [])
                if _ridsI2 and _mpI2:
                    _bkI2 = None
                    for _bk2 in sorted(_mpI2):
                        if _bk2 >= len(_ridsI2):
                            _bkI2 = _bk2
                            break
                    if _bkI2 is not None:
                        _meI2 = _mpI2[_bkI2]
                    else:
                        _meI2 = None
                    if _bkI2 is not None and _meI2 is not None:
                        _smI2 = getattr(type(self), "_r164_slotmap", None) or {}
                        _slI2 = [_smI2.get(r, i) for i, r in enumerate(_ridsI2)]
                        # R203_METAFIX width PAD: the refresh copies
                        # len(rids)-wide tensors into bucket-width flats —
                        # with ChunkedPrefill/mixed batches (len 2..31 into
                        # the 32-bucket) the FIRST copy_ threw
                        # aclnnInplaceCopy 161002 ("N and 32 cannot
                        # broadcast"); the swallow-all except kept the
                        # step-memo from setting, so the SAME failing refresh
                        # re-ran on EVERY layer (~1.2-1.8ms/layer of blocking
                        # torch.tensor H2Ds; r203d census: 2160 exceptions,
                        # 2074 wasted tensor builds per leg) AND the R167
                        # raced-first-replay protection NEVER ENGAGED on this
                        # channel (r203b/e METAFIX legs: elevated token-3
                        # degenerate rows = the B4-l0 class the refresh was
                        # built to fix). PAD to bucket width instead: live
                        # rows carry their REAL slots; absent rows carry the
                        # R167_IDFILL identity (slot=row, askv=0 masked,
                        # bt=0, physical idx) — the same semantics the
                        # first-generation v1 fill uses, so a raced first
                        # replay reads real rings for live rows and neutral
                        # identity for absent ones.
                        if os.environ.get("R203_METAFIX", "0") == "1" and len(_slI2) < int(_meI2["slot"].shape[0]):
                            _wI2 = int(_meI2["slot"].shape[0])
                            _slI2 = list(_slI2) + list(range(len(_slI2), _wI2))
                        _mdI2_bt = getattr(attn_metadata, "block_tables", None)
                        _slsI2 = list(getattr(attn_metadata, "seq_lens_list", None) or [])
                        _curI2 = torch.npu.current_stream()
                        with torch.npu.stream(_curI2):
                            _slT2 = torch.tensor(_slI2, dtype=torch.int64, device=device)
                            _meI2["slot"].copy_(_slT2)
                            # R199_PRREAD: keep bt_sr consistent with the raced
                            # first-replay protection above (bt_sr is a pure
                            # function of the slot flat).
                            if _meI2.get("pg199"):
                                _nbQ2 = int(_meI2["nb_sr"])
                                _tBQ2 = _meI2.get("t_bsr")
                                if _tBQ2 is None or _tBQ2.numel() != _nbQ2:
                                    _tBQ2 = _meI2["t_bsr"] = torch.arange(_nbQ2, dtype=torch.int64, device=device)
                                _meI2["bt_sr"].copy_(_meI2["slot"].view(-1, 1) * _nbQ2 + _tBQ2.view(1, -1))
                            _mhp2 = int(_meI2.get("mhp", _meI2.get("npr", 1)))
                            if _os.environ.get("R198_VRING", "0") == "1":
                                # R198 VRING: rotated patterns (same formula
                                # as _r164_update_meta; protects a raced
                                # first post-prefill replay)
                                _pf2 = int(_meI2["prefp"])
                                _cp2 = int(_meI2["capw"])
                                _npr2 = int(_meI2.get("npr", _pf2 + _cp2))
                                _org2 = getattr(self, "_tt_ring_org", None) or {}
                                _idx2 = []
                                _pn2 = []
                                # R203 pad: iterate the FULL bucket width; rows
                                # beyond the live rids get identity positions
                                # (X=pref, sl=pref+cap => pos==j for all j).
                                _nI2 = len(_slI2) if os.environ.get("R203_METAFIX", "0") == "1" else len(_ridsI2)
                                for _i2 in range(_nI2):
                                    _s2 = int(_slI2[_i2])
                                    if _i2 < len(_ridsI2):
                                        _r2 = _ridsI2[_i2]
                                        _X2 = int(_org2.get(_r2, _org2.get(_i2, _pf2)))
                                        _sl2 = int(_slsI2[_i2]) if _i2 < len(_slsI2) else 0
                                    else:
                                        _X2 = _pf2
                                        _sl2 = _pf2 + _cp2
                                    for _j2 in range(_npr2):
                                        _pos2 = (
                                            _j2 if _j2 < _pf2 else _pf2 + ((_sl2 - _cp2 + (_j2 - _pf2) - _X2) % _cp2)
                                        )
                                        _idx2.append(_s2 * _mhp2 + _pos2)
                                    _pn2.append(_s2 * _mhp2 + _pf2 + ((_sl2 - 1 - _X2) % _cp2))
                                _meI2["idx_sr"].view(-1)[: len(_idx2)].copy_(
                                    torch.tensor(_idx2, dtype=torch.int64, device=device)
                                )
                                _meI2["p_new"][: len(_pn2)].copy_(torch.tensor(_pn2, dtype=torch.int64, device=device))
                            else:
                                _meI2["idx_sr"].copy_(
                                    (_slT2.view(-1, 1) * _mhp2 + _meI2["t_const"].view(1, -1)).reshape(-1)
                                )
                                _meI2["p_src"].copy_(
                                    (_slT2.view(-1, 1) * _mhp2 + _meI2["t_w1"].view(1, -1)).reshape(-1)
                                )
                                _meI2["p_dst"].copy_(
                                    (_slT2.view(-1, 1) * _mhp2 + _meI2["t_w0"].view(1, -1)).reshape(-1)
                                )
                                _meI2["p_new"].copy_(_slT2 * _mhp2 + (int(_meI2["prefp"]) + int(_meI2["capw"]) - 1))
                            if _os.environ.get("R169_FLATFIX", "0") == "1" and _mdI2_bt is not None:
                                # R169_FLATFIX: ALSO refresh the two
                                # middle-tier input flats (askv + bt_mirror)
                                # at every prefill-state forward, so the FIRST
                                # post-prefill replay reads THIS generation's
                                # mid window instead of the previous
                                # generation's (or the R167_IDFILL identity).
                                # Geometry = the values BAKED into the
                                # bucket's captured staging (_r164_baked_geom)
                                # — the same numbers the updater uses. Same
                                # stream/ordering as the proven IDFILL slot
                                # refresh (FIFO before the next replay
                                # launch); no sync, no new events.
                                _bgF = getattr(type(self), "_r164_baked_geom", {}).get(int(_bkI2))
                                if _bgF is not None:
                                    _msF = int(_bgF["mid_start"])
                                    _rcF = int(_bgF["rcnt"])
                                    _nbF = int(_meI2["nb_max"])
                                    _bszF = int(self.vllm_config.cache_config.block_size)
                                    _avN = int(_meI2["askv"].shape[0])
                                    _avF = [0] * _avN
                                    for _rF in range(min(len(_slsI2), _avN)):
                                        _mlF = int(_slsI2[_rF]) - _rcF - _msF
                                        _avF[_rF] = _mlF if _mlF > 0 else 0
                                    _meI2["askv"].copy_(
                                        torch.tensor(_avF, dtype=torch.int64, device=_meI2["askv"].device)
                                    )
                                    _ncF = 0
                                    _widF = int(_mdI2_bt.shape[1])
                                    _loF = _msF // _bszF if _bszF > 0 else 0
                                    if 0 <= _loF < _widF:
                                        _ncF = min(_nbF, _widF - _loF)
                                        _meI2["bt_mirror"][:, :_ncF].fill_(0)
                                        _acF = min(len(_slsI2), int(_meI2["bt_mirror"].shape[0]))
                                        _meI2["bt_mirror"][0:_acF, 0:_ncF].copy_(_mdI2_bt[0:_acF, _loF : _loF + _ncF])
                                        _meI2["bt_mirror"].clamp_(min=0)
                                    _ffN = globals().setdefault("_R169_FF_N", [0])
                                    if _ffN[0] < 24:
                                        _ffN[0] += 1
                                        import json as _jF

                                        with open("/tmp/r169_flatfix.log", "a") as _fF:
                                            _fF.write(
                                                _jF.dumps(
                                                    {
                                                        "n": _ffN[0],
                                                        "bk": int(_bkI2),
                                                        "baked": _bgF,
                                                        "sls": _slsI2[:8],
                                                        "askv": _avF[:8],
                                                        "lo": _loF,
                                                        "nc": _ncF,
                                                    }
                                                )
                                                + "\n"
                                            )
                # R196: refresh completed for this (metadata, key) — later
                # layers of the same forward skip the redundant rewrite.
                self._r196_idfill_memo(attn_metadata, set_key=True)
            except Exception:
                if _os.environ.get("R203_D2HTRACE", "0") == "1":
                    import traceback as _tb203

                    try:
                        with open("/tmp/r203_d2h.log", "a") as _f203e:
                            _f203e.write("IDFILL_EXC {}\n".format(_tb203.format_exc()[-400:]))
                    except Exception:
                        pass
        import os as _os166

        if _os166.environ.get("R170_ROWSLOT", "0") == "1" and layer is not None:
            # R170_ROWSLOT: row-slot contract. Bind rid->slot with the R166
            # free-list allocator (acquire/release lifecycle, zero-on-reuse);
            # _r170_rowslot_migrate (pre-replay updater) then keeps each
            # decoding request's ring at its decode-ROW slot, matching what
            # FULL decode graphs bake under this flag. Supersedes both the
            # monotonic no-reuse policy and the legacy b-keyed path.
            return self._r166_bind_slots(attn_metadata, device, dtype, layer)
        if (
            _os166.environ.get("R166_SLOTALGO", "0") == "1"
            and _os166.environ.get("R164_SLOTMAP", "0") == "1"
            and layer is not None
        ):
            return self._r166_bind_slots(attn_metadata, device, dtype, layer)
        if _os.environ.get("R164_SLOTMAP", "0") != "1":
            return
        rids = getattr(attn_metadata, "request_ids", None)
        if not rids:
            return
        m = getattr(type(self), "_r164_slotmap", None)
        if m is None:
            type(self)._r164_slotmap = m = {}
        # R165-FINAL slot policy: MONOTONIC counter — NO reuse, NO frees.
        # Every reuse policy tried (absent-free, finish-signal, grace+free-
        # list) had edge cases that clobbered live rings across leg
        # transitions; a never-reused slot cannot collide by construction.
        # Ring cost ~151MB/slot for the model; mseq must cover the process
        # lifetime request count (fail-closed with a clear message).
        cur_rids = []
        cur_slots = []
        _ctr = getattr(type(self), "_r164_slotctr", None)
        if _ctr is None:
            type(self)._r164_slotctr = _ctr = [0]
        for _rid in rids:
            if _rid not in m:
                _slot = _ctr[0]
                if _slot >= int(self._tt_mseq):
                    raise RuntimeError(
                        f"[R164-SLOTMAP] monotonic slot exhaustion "
                        f"(mseq={self._tt_mseq}, next={_slot}) — raise "
                        "R109_TT_MAX_SEQS to cover the process lifetime "
                        "request count (fail-closed; no reuse by design)"
                    )
                m[_rid] = _slot
                _ctr[0] = _slot + 1
            cur_rids.append(_rid)
            cur_slots.append(m[_rid])
        self._tt_cur_rids = cur_rids
        self._tt_cur_slots = cur_slots

    def _r170_rebind_locked(A: Any, rid, ns):
        """Move rid's slot binding to ns. Caller holds A._lk. Invariants of
        _R166SlotAlloc are preserved (bijection, free/occupancy cover); gen
        bump on ns invalidates stale (slot,gen,lid) zero-once entries so a
        future acquire of ns still zero-fills."""
        cur = A.rid2slot.get(rid)
        if cur is None or cur == ns:
            return
        owner = A.slot2rid.get(ns)
        if owner is not None and owner != rid:
            raise RuntimeError(
                f"[R170-ROWSLOT] rebind conflict: slot {ns} owned by {owner!r}, cannot move {rid!r} there (fail-closed)"
            )
        if owner is None and ns not in A.free:
            raise RuntimeError(
                f"[R170-ROWSLOT] rebind target {ns} neither free nor owned "
                "(allocator cover invariant broken; fail-closed)"
            )
        if ns in A.free:
            A.free.remove(ns)
        del A.slot2rid[cur]
        A.rid2slot[rid] = ns
        A.slot2rid[ns] = rid
        A.free.append(cur)
        A.gen[ns] += 1
        A.events.append((rid, ns, A.gen[ns], "r170-move"))

    def _oscar_layer_ord(self, layer):
        """R138: STABLE per-layer ordinal for R_k/R_v indexing, _tt_hp keying,
        and the capture ``_lid`` (carrier-pool key).

        Replaces the R137 first-seen ``id(layer) -> ordinal`` map (visitation-
        order-dependent; fragile to object-id reuse) with an EXPLICIT,
        order-independent registry keyed on ``layer.layer_name``: the parsed
        checkpoint layer index (``model.layers.N...`` -> N), the same stable
        source used at L1627 for hamming_sparse. Name->ord is a pure function
        of the name, so:

          * Capture and Replay always agree: the ``_lid`` baked into the marker
            param is the same stable ordinal regardless of capture/replay order
            (the R138 ``Capture/Replay order-change invariance`` goal);
          * the ``_r113_cpool`` / ``_tt_hp`` mapping is invariant when Capture
            visits layers in a different order than Replay;
          * survives layer-object recreation (same name -> same ord).

        For the standard transformer forward (layers 0..N-1 in order) the
        ordinals are 0..N-1 == the R137 first-seen values, so R_k indexing is
        unchanged (no regression). Runtime-derived: no model-name / 36 / D128 /
        NKV literals. Fail-closed if ``layer_name`` is absent/unparsable, or
        (when R_k is loaded) the parsed ordinal >= R_k rows (config mismatch).

        Optional proof (R137_RK_PROOF=1): on first sight of a layer name,
        record its ordinal, R_k slice data_ptr + sha -- proving distinct
        layers do NOT share a rotation slice.
        """
        name = getattr(layer, "layer_name", None)
        if not name:
            raise RuntimeError(
                "[R138-RK] layer has no layer_name; stable registry cannot derive a per-layer ordinal (fail-closed)"
            )
        m = getattr(type(self), "_oscar_name2ord", None)
        if m is None:
            m = {}
            type(self)._oscar_name2ord = m
        o = m.get(name)
        if o is None:
            try:
                o = int(name.split(".")[2])
            except (ValueError, IndexError):
                raise RuntimeError(f"[R138-RK] cannot parse layer index from layer_name {name!r} (fail-closed)")
            if self._oscar_R_k is not None and o >= self._oscar_R_k.shape[0]:
                raise RuntimeError(
                    f"[R138-RK] layer ordinal {o} >= R_k rows "
                    f"{self._oscar_R_k.shape[0]} (more distinct layers than "
                    f"rotation rows -- config mismatch, fail-closed)"
                )
            m[name] = o
            if os.environ.get("R137_RK_PROOF", "0") == "1" and self._oscar_R_k is not None:
                try:
                    import hashlib as _hl
                    import json as _js

                    sl = self._oscar_R_k[o]
                    rec = {
                        "ord": o,
                        "name": name,
                        "rk_ptr": int(sl.data_ptr()),
                        "rk_sha12": _hl.sha256(sl.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[
                            :12
                        ],
                        "rv_ptr": int(self._oscar_R_v[o].data_ptr()),
                        "rows": int(self._oscar_R_k.shape[0]),
                    }
                    type(self)._r137_rk_proof = getattr(type(self), "_r137_rk_proof", [])
                    type(self)._r137_rk_proof.append(rec)
                    with open(os.environ.get("R137_RK_PROOF_OUT", _odiag_p("r138_rk_proof.jsonl")), "a") as _f:
                        _f.write(_js.dumps(rec) + "\n")
                except Exception as _e:
                    _oscar_dbg(f"[R138-RK] proof failed: {_e}", flush=True)
        return o
