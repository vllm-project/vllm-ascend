# SPDX-License-Identifier: Apache-2.0
"""INT2 middle-executor mixin: op input layout, batched middle calls,
OUT/LSE reception. Numeric path device-bound."""

from __future__ import annotations

import os
import os as _r144_ovl_os
from typing import Any

import torch
import torch_npu

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.int2.ctx import (  # noqa: F401
    _OSCAR_CFG,
    _R2XX_G,
    _odiag_o,
    _odiag_p,
    _odiag_w,
)
from vllm_ascend.attention.int2.helpers import (  # noqa: F401
    _r150_tr,
    _r166_tr,
    _r216_mid,
    _r218_diag,
)


class Int2MiddleExecMixin:
    # Host-state annotations: the probe flags / module handles / shared
    # operand caches below are owned by this mixin and initialized lazily
    # through type(self) (defaults are required there); the slot-mapping,
    # cache-interface, loader, and sibling-builder names are owned by the
    # assembled backend / sibling mixins (bare annotations only - no runtime
    # attributes, so no MRO shadowing at assembly).
    _r113_warmed: Any
    _r115_mu_calls: int = 0
    _r144_qshadow_n: int = 0
    _r146_dummy_blob: Any
    _r146_mirror: Any = None
    _r147_shared: Any = None
    _r147_stage: Any = None
    _r185c_dumped: Any = None
    _r189_fill_probed: Any = None
    _r189_mod: Any
    _r189_real_probed: Any = None
    _r189_v3_probed: Any = None
    _r190_probed: Any = None
    _r191_fx: Any = None
    _r191_ident: Any = None
    _r191_p2: Any = None
    _r191_p3: Any = None
    _r191_p4: Any = None
    _r215_gm_done: Any = None
    _r218_mid_R: Any = None
    _r218_qperm: Any = None
    _TT_MAX_SEQS: Any
    _TT_PREFIX: Any
    _TT_RECENT: Any
    _dequantize_packed_int2: Any
    _ensure_tt: Any
    _oscar_R_k: Any
    _oscar_R_v: Any
    _oscar_layer_ord: Any
    _r109_middle_candidate: Any
    _r112_ensure_fast: Any
    _r112_mod: Any
    _r115_aclnn_mod: Any
    _r135_ensure_v1: Any
    _r135_mod: Any
    _r144_carriers: Any
    _r145_carriers: Any
    _r164_bind_slots: Any
    _r164_midprobe: Any
    _r196_batched_sr: Any
    _r197_batched_ring: Any
    _r197_probe_record: Any
    _r198_pos: Any
    _r198_pr_window: Any
    _tt_hp_bufs: Any
    _tt_hp_k: Any
    _tt_mseq: Any
    _tt_pref: Any
    _tt_rcnt: Any
    _tt_rcnt_seen: Any
    _tt_resolve_slots: Any
    _tt_rid: Any
    _tt_ring_org: Any
    _tt_slot: Any
    _tt_write_hp: Any
    head_size: Any
    k_scale_cache: Any
    k_zero_cache: Any
    key_cache: Any
    num_heads: Any
    num_kv_heads: Any
    scale: Any
    v_scale_cache: Any
    v_zero_cache: Any
    value_cache: Any
    vllm_config: Any

    def _forward_int2_decode_threetier_middle(self, query, key_rot, value_rot, attn_metadata, output, layer):
        """R109 B: three-tier decode where ONLY the INT2 middle executor is the
        candidate fused op. Tier boundaries/HP writes identical to the existing
        path (first block below is shared logic re-run). Merge (task §6.3):
          M = max(L_i)
          w_i = exp(L_i - M)
          O = sum(w_i O_i)/sum(w_i)
        All O_i stay in the ROTATED-V basis; R_v.T applied once after merge."""
        import os as _os
        import sys as _sys

        # R197 Fix-0: was an UNCONDITIONAL file append per middle layer-call
        # (~31us host each). Gated behind R190_PROBE (default OFF).
        if _os.environ.get("R190_PROBE", "0") == "1":
            _odiag_w("r190_probe.log", "[R190] _forward_int2_decode_threetier_middle CALLED\n")
        # R191 P1: trusted-baseline identity audit — loaded module file + content hash +
        # overlay env + impl class, fail-closed (md5 must be diffed against the manifest
        # record; probe deltas are expected and listed in the R191 report).
        if _os.environ.get("R191_AUDIT", "0") == "1" and not getattr(type(self), "_r191_ident", False):
            type(self)._r191_ident = True
            try:
                import hashlib as _hl191

                _p191 = __file__
                with open(_p191, "rb") as _f191:
                    _b191 = _f191.read()
                _odiag_w(
                    "r191_audit.log",
                    f"[IDENTITY] pid={_os.getpid()} file={_p191} "
                    f"md5={_hl191.md5(_b191).hexdigest()} sha256={_hl191.sha256(_b191).hexdigest()[:16]} "
                    f"overlay={_os.environ.get('R148_OVERLAY', '?')} cls={type(self).__name__} "
                    f"msd={_os.environ.get('R174_INT2_MSD', '?')} vendor={_os.environ.get('R159K_R146VENDOR', '?')}\n",
                )
            except Exception:
                pass
        # R190 PROBE: at the method entry — run LSE with the REAL captured fixture (not synthetic).
        # R189 showed synthetic→GREEN, real→RED. This tests whether the REAL fixture alone
        # (with the product cube state at the method entry) triggers the fault.
        if _os.environ.get("R190_FIXTURE_PROBE") == "1" and not getattr(type(self), "_r190_probed", False):
            type(self)._r190_probed = True
            _odiag_w("r190_probe.log", "[R190] probe entry reached, loading module if needed\n")
            # Load the launcher module if not already loaded (self-contained, no dependency on R189 L0 probe)
            if getattr(type(self), "_r189_mod", None) is None:
                import torch.utils.cpp_extension as _ce190

                _R152 = _os.environ.get("R152_LAUNCHER_DIR", "")
                _TN = os.path.dirname(torch_npu.__file__)
                _INC2 = [
                    os.path.join(_TN, "include"),
                    os.path.join(_TN, "include", "third_party", "op-plugin"),
                    "/usr/local/Ascend/cann-9.0.0/aarch64-linux/include",
                    "/usr/local/Ascend/cann-9.0.0/include",
                    _os.environ.get("R183_COMPOSITE_OPP", "") + "/op_api/include",
                ]
                _LD2 = [
                    "-L/usr/local/Ascend/cann-9.0.0/lib64",
                    "-lascendcl",
                    "-lopapi",
                    "-L" + os.path.join(_TN, "lib"),
                    "-ltorch_npu",
                    "-L" + _os.environ.get("R183_COMPOSITE_OPP", "") + "/op_api/lib",
                    "-lcust_opapi",
                    "-Wl,-rpath,/usr/local/Ascend/cann-9.0.0/lib64",
                    "-Wl,-rpath," + _os.environ.get("R183_COMPOSITE_OPP", "") + "/op_api/lib",
                ]
                _os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/r191_ext"
                _os.makedirs(_os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
                _odiag_w("r190_probe.log", "[R190] loading launcher module...\n")
                type(self)._r189_mod = _ce190.load(
                    name="r191mod",
                    sources=[_R152 + "/int2_op_launch_lse_fast_r152.cpp"],
                    extra_include_paths=_INC2,
                    extra_ldflags=_LD2,
                    build_directory=_os.environ["TORCH_EXTENSIONS_DIR"],
                    verbose=False,
                )
                _odiag_w("r190_probe.log", "[R190] module loaded\n")
            import json as _j190

            _fb = _odiag_p("r185_makeacl.bin")
            _fj = _odiag_p("r185_makeacl.json")
            try:
                with open(_fj) as _fjH:
                    _J = _j190.load(_fjH)
                with open(_fb, "rb") as _fbH:
                    _raw = _fbH.read()

                def _ld190(nm):
                    t = [x for x in _J["tensors"] if x["name"] == nm][0]
                    o, nb, sh, dt = t["offset"], t["nbytes"], t["shape"], t["dtype"]
                    b = _raw[o : o + nb]
                    if "BFloat16" in dt:
                        a = torch.frombuffer(b, dtype=torch.uint16).reshape(sh).view(torch.bfloat16)
                    elif "signed char" in dt:
                        a = torch.frombuffer(b, dtype=torch.int8).reshape(sh)
                    elif dt == "int":
                        a = torch.frombuffer(b, dtype=torch.int32).reshape(sh)
                    else:
                        a = torch.frombuffer(b, dtype=torch.uint8).reshape(sh)
                    return a.clone().to(device=query.device)

                _rq, _rk, _rv, _rbt, _rks, _rko, _rvs, _rvo = (
                    _ld190("q"),
                    _ld190("k"),
                    _ld190("v"),
                    _ld190("bt"),
                    _ld190("ks"),
                    _ld190("ko"),
                    _ld190("vs"),
                    _ld190("vo"),
                )
                _ro = torch.zeros(1, 32, 1, 128, dtype=torch.bfloat16, device=query.device)
                _rl = torch.zeros(1, 32, 1, dtype=torch.float32, device=query.device)
                _rr = type(self)._r189_mod.launch_lse_fast(
                    _rq,
                    _rk,
                    _rv,
                    _rbt,
                    _rks,
                    _rko,
                    _rvs,
                    _rvo,
                    _ro,
                    _rl,
                    32,
                    1.0 / (128**0.5),
                    8,
                    "BNSD",
                    128,
                    1,
                    1,
                    [2176],
                    torch.zeros(0, dtype=torch.bfloat16, device=query.device),
                    0,
                    0,
                    0,
                    True,
                )
                torch.npu.synchronize()
                _odiag_w(
                    "r190_probe.log",
                    f"[R190-FIXTURE] METHOD-ENTRY-REAL rc={int(_rr[0])} "
                    f"finite={torch.isfinite(_ro).all().item()} out_absmax={float(_ro.abs().max()):.4g}\n",
                )
            except Exception as _pe190:
                _odiag_w("r190_probe.log", f"[R190-FIXTURE] METHOD-ENTRY-REAL CRASH: {str(_pe190)[:100]}\n")
        if self._oscar_layer_ord(layer) == 0:
            _r166_tr(
                "decode3t",
                bs=int(len(attn_metadata.seq_lens_list)),
                sls=[int(x) for x in attn_metadata.seq_lens_list][:16],
                st=str(getattr(attn_metadata, "attn_state", "?")),
            )
        self._ensure_tt(query.device)
        # R144 P0: map batch-row b -> stable rid + reusable slot (gated, default OFF).
        if _os.environ.get("R144_TT_REQID", "0") == "1":
            self._tt_resolve_slots(attn_metadata, self._oscar_layer_ord(layer), query.device, query.dtype)
        self._r164_bind_slots(attn_metadata, query.device, query.dtype, layer)
        # R144 DIAG (file-based, stderr suppressed under R116_QUIET): entry of
        # threetier-middle decode path. Log first 6 decode calls w/ per-row boundaries.
        _pr_k_row = {}
        _o_pr_row = {}
        _l_pr_row = {}
        _o_h_row = {}
        _l_h_row = {}
        _m_row = {}
        _r144dm_n = globals().setdefault("_R144_DM_N", [0])
        if _r144dm_n[0] < 6:
            _r144dm_n[0] += 1
            try:
                _bsz = len(attn_metadata.seq_lens_list)
                _sls = list(attn_metadata.seq_lens_list)
                _prefs = [self._tt_pref.get(self._tt_rid(i), -1) for i in range(_bsz)]
                _rcnts = [self._tt_rcnt.get(self._tt_rid(i), -1) for i in range(_bsz)]
                _mids = []
                for i in range(_bsz):
                    _mids.append(int(_sls[i] - _rcnts[i] - _prefs[i]))
                with open("/tmp/r144_dm_probe.log", "a") as _pf:
                    _pf.write(
                        f"[R144-DM] call#{_r144dm_n[0]} L={self._oscar_layer_ord(layer)} bs={_bsz} "
                        f"sls={_sls} pref={_prefs} rcnt={_rcnts} mid_len={_mids} "
                        f"BM={_os.environ.get('R144_BATCHED_MIDDLE', '?')}\n"
                    )
            except Exception:
                pass
        # R113: pre-grow the fast-launcher workspace during the EAGER warmup
        # (before any aclgraph capture / task-update). torch::empty inside those
        # regions deadlocks; replay updates need ws for the MAX askv, not the
        # capture placeholder. 128 MiB covers askv<=1408 (fast_smoke: 32 MiB at
        # askv=561).
        if (
            not _EXTRA_CTX.capturing
            and getattr(type(self), "_r113_warmed", None) is None
            and _os.environ.get("R112_LSE_FAST", "0") == "1"
        ):
            if self._r112_ensure_fast(query.device):
                type(self)._r112_mod.warm_ws(query, 1024 << 20)
            # R135: pre-grow the V1 (device-tensor) binding's shared workspace
            # during eager warmup, before any aclgraph capture. torch::empty
            # inside the binding (g_ws grow) deadlocks under capture/replay.
            # R161 P1-A: the BATCHED middle launch (R161_BATCHED_MID) passes a
            # [B,...] query/out/lse/askv to launch_lse_v1_own; its workspace
            # scales with B (softmax intermediates ~ B*S*H*D). 128 MiB only
            # covers B=1; B>=4 capture would try to grow g_ws INSIDE the
            # NPUGraph capture -> futex deadlock. Pre-grow to 2 GiB so every
            # capture bucket (B1..B16) finds g_ws sufficient (no in-capture
            # alloc, no deadlock).
            if _os.environ.get("R133_GRAPH_V1", "0") == "1" and self._r135_ensure_v1(query.device):
                type(self)._r135_mod.warm_ws(query, 128 << 20)
            type(self)._r113_warmed = True
        layer_idx = getattr(layer, "layer_idx", 0)
        batch_size = len(attn_metadata.seq_lens_list)
        seq_lens = attn_metadata.seq_lens_list
        # R197 PROBE (default OFF): per-phase timing of the EAGER mixed-step
        # middle chain — ring / staging / q / fill+op / R196 S+R — plus a
        # row census (Bp, mid lens, cap bucket). Sampled; one device event
        # pair per sampled call resolved at the NEXT sampled call (probe legs
        # only; never in A/B). Evidence file: /tmp/r197_probe.jsonl.
        _r197s = False
        if _os.environ.get("R197_PROBE", "0") == "1" and not _EXTRA_CTX.capturing:
            try:
                _r197n = _R2XX_G.setdefault("_R197_PROBE_N", [0])
                _r197n[0] += 1
                if _r197n[0] <= 72 or (_r197n[0] % 40) == 0:
                    _r197s = True
                    import time as _t197

                    _r197t: dict[str, Any] = {"t0": _t197.perf_counter()}
                    _r197ev0 = torch.npu.Event(enable_timing=True)
                    _r197ev0.record()
                    _r197t["ev0"] = _r197ev0
            except Exception:
                _r197s = False
        _st = _R2XX_G.setdefault("_R109_STATS", {"middle_calls": 0, "middle_ok": 0, "merge": 0, "full": 0})
        if layer_idx == 0:
            if _os.environ.get("R197_DBG", "0") == "1":  # R197 Fix-0
                print(
                    f"[R109-3T-DEC] ENTER batch_size={batch_size} seq_lens={seq_lens} "
                    f"tt_pref={dict(self._tt_pref)} tt_rcnt={dict(self._tt_rcnt)} "
                    f"stats={_st}",
                    file=_sys.stderr,
                    flush=True,
                )
            try:
                import json as _j

                _st_x = dict(_st)
                _st_x["cu_calls"] = getattr(type(self), "_r115_cu_calls", 0)
                _st_x["mu_calls"] = getattr(type(self), "_r115_mu_calls", 0)
                with open(_os.environ.get("R116_STATS_OUT", _odiag_p("r116_route_stats.json")), "w") as _f:
                    _j.dump(_st_x, _f)
            except Exception:
                pass

        # R147 P0-B PROBE: per-layer tier-boundary trace (entry state, pre-increment).
        _r147n = globals().setdefault("_R147_P0B_N", [0])
        if _os.environ.get("R147_PROBE", "0") == "1" and _r147n[0] < 6000:
            _r147n[0] += 1
            _rows147 = []
            for b in range(batch_size):
                _rid2 = self._tt_rid(b)
                _sl2 = int(seq_lens[b])
                _pr2 = self._tt_pref.get(_rid2, 0)
                _rc2 = self._tt_rcnt.get(_rid2, 0)
                _rows147.append(f"{_rid2}:{_sl2}:{_pr2}:{_rc2}:{_sl2 - _rc2 - _pr2}")
            try:
                with open("/tmp/r147_p0b_probe.log", "a") as _pf:
                    _pf.write(
                        f"c{_r147n[0]} L{self._oscar_layer_ord(layer)} B{batch_size} "
                        f"F{_os.environ.get('R147_RCNT_FIX', '0')} id{str(id(self))[-6:]} "
                        f"d{dict(self._tt_rcnt)} {' '.join(_rows147)}\n"
                    )
            except Exception:
                pass
        # HP-recent window maintenance — identical to the existing three-tier path
        # R197 Fix-1 (R197_EAGER_MIX=1, default OFF): batched ring update for the
        # eager path — the legacy per-row loop costs ~0.5ms/row/layer of HOST
        # dispatch (probe: ring_ms 0.55*B per layer-call; 505ms/mixed step at
        # B=31). Same writes to the same ring positions; falls back verbatim.
        _r197ring = False
        if (
            _os.environ.get("R197_EAGER_MIX", "0") == "1"
            and batch_size >= 2
            and not _EXTRA_CTX.capturing
            and _os.environ.get("R164_EAGER_SLIDE", "0") != "1"
        ):
            try:
                _r197ring = self._r197_batched_ring(layer, query, key_rot, value_rot, seq_lens, batch_size)
            except Exception:
                _r197ring = False
        if not _r197ring:
            for b in range(batch_size):
                sl = int(seq_lens[b])
                _rid = self._tt_rid(b)
                _slot = self._tt_slot(b)
                pref = self._tt_pref.get(_rid, 0)
                rcnt = self._tt_rcnt.get(_rid, 0)
                cap = self._TT_RECENT
                newk = key_rot[b : b + 1]
                newv = value_rot[b : b + 1]
                _p198 = self._r198_pos(_rid, sl - 1, pref) if _os.environ.get("R198_VRING", "0") == "1" else None
                if _os.environ.get("R218_RINGDBG", "0") == "1" and self._oscar_layer_ord(layer) == 0:
                    # R218 decode-ring layout census (DIAGNOSTIC_ONLY): every row's
                    # write convention this step. A row with p=None while its peers
                    # rotate gets the LEGACY left-packed write -> read/write layout
                    # desync (the window rotates by sl-1-X while the read uses X).
                    try:
                        import json as _jrd

                        with open("/tmp/r218_ringdbg.jsonl", "a") as _frd:
                            _frd.write(
                                _jrd.dumps(
                                    {
                                        "b": b,
                                        "rid": str(_rid),
                                        "slot": int(_slot),
                                        "sl": int(sl),
                                        "pref": int(pref),
                                        "rcnt": int(rcnt),
                                        "p": (None if _p198 is None else int(_p198)),
                                        "X": (
                                            None
                                            if self._tt_ring_org.get(_rid) is None
                                            else int(self._tt_ring_org.get(_rid))
                                        ),
                                        "mode": (
                                            "vring"
                                            if _p198 is not None
                                            else ("slide" if rcnt >= self._TT_RECENT else "append")
                                        ),
                                        "src": "row",
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                if _p198 is not None:
                    # R198 VRING: rotating append (see _r198_pos); no slide.
                    self._tt_write_hp(self._oscar_layer_ord(layer), _slot, _p198, (newk, newv))
                    if rcnt < cap:
                        if _os.environ.get("R147_RCNT_FIX", "0") == "1":
                            if self._tt_rcnt_seen.get(_rid) != sl:
                                self._tt_rcnt_seen[_rid] = sl
                                self._tt_rcnt[_rid] = rcnt + 1
                        else:
                            self._tt_rcnt[_rid] = rcnt + 1
                    else:
                        self._tt_rcnt[_rid] = cap
                elif _os.environ.get("R164_EAGER_SLIDE", "0") == "1" and rcnt >= min(cap, max(0, sl - pref)):
                    # R165 CANDIDATE FIX (default OFF): graph-equivalent fixed-window
                    # semantics — once the recent window holds min(cap, sl-pref) rows
                    # (i.e. ring as full as prefill left it), ALWAYS slide instead of
                    # appending (append writes INSIDE the window when rcnt<cap and the
                    # per-layer counter drifts — the 16-row block misalignment seen as
                    # the B8 N-1 signature).
                    _ek, _ev = self._tt_hp_bufs(self._oscar_layer_ord(layer), query.device, query.dtype)
                    _nk = torch.cat([_ek[_slot, pref + 1 : pref + cap], newk], dim=0)
                    _nv = torch.cat([_ev[_slot, pref + 1 : pref + cap], newv], dim=0)
                    if _os.environ.get("R166_STREAMFIX", "0") in ("1", "ring", "both"):
                        torch.mul(_nk, 1.0, out=_ek.narrow(0, _slot, 1).narrow(1, pref, cap)[0])
                        torch.mul(_nv, 1.0, out=_ev.narrow(0, _slot, 1).narrow(1, pref, cap)[0])
                    else:
                        _ek[_slot, pref : pref + cap] = _nk
                        _ev[_slot, pref : pref + cap] = _nv
                    self._tt_rcnt[_rid] = cap
                elif rcnt < cap:
                    # R147 P0-B FIX (R147_RCNT_FIX=1): _tt_rcnt is a REQUEST-scoped token
                    # count shared by all 36 per-layer HP buffers. Each layer appends its
                    # own rotated K/V copy at the SHARED position (pref+rcnt) — but the
                    # shared counter advanced once per LAYER before, which (a) shrank
                    # mid_end = sl - rcnt by 1 per layer inside one decode step and
                    # (b) in append mode (rcnt < cap, i.e. prompt < pref+cap tokens)
                    # left each per-layer HP recent window with 35/36 never-written
                    # (zero) slots that the tier merge then attended over. Advance the
                    # counter once per decode step: the first layer call observing this
                    # request's new seq_len (per-rid seen-map; seq_len is monotone).
                    self._tt_write_hp(self._oscar_layer_ord(layer), _slot, pref + rcnt, (newk, newv))
                    if _os.environ.get("R147_RCNT_FIX", "0") == "1":
                        if self._tt_rcnt_seen.get(_rid) != sl:
                            self._tt_rcnt_seen[_rid] = sl
                            self._tt_rcnt[_rid] = rcnt + 1
                    else:
                        self._tt_rcnt[_rid] = rcnt + 1
                else:
                    hk, hv = self._tt_hp_bufs(self._oscar_layer_ord(layer), query.device, query.dtype)
                    _nk2 = torch.cat([hk[_slot, pref + 1 : pref + cap], newk], dim=0)
                    _nv2 = torch.cat([hv[_slot, pref + 1 : pref + cap], newv], dim=0)
                    if _os.environ.get("R166_STREAMFIX", "0") in ("1", "ring", "both"):
                        torch.mul(_nk2, 1.0, out=hk.narrow(0, _slot, 1).narrow(1, pref, cap)[0])
                        torch.mul(_nv2, 1.0, out=hv.narrow(0, _slot, 1).narrow(1, pref, cap)[0])
                    else:
                        hk[_slot, pref : pref + cap] = _nk2
                        hv[_slot, pref : pref + cap] = _nv2

        if _os.environ.get("R147_PROBE", "0") == "1" and _r147n[0] <= 120:
            try:
                with open("/tmp/r147_p0b_probe.log", "a") as _pf:
                    _pf.write(f"  post c{_r147n[0]} d{dict(self._tt_rcnt)}\n")
            except Exception:
                pass
        if _r197s:
            try:
                _r197t["t1"] = _t197.perf_counter()  # post ring-maintenance
                _r197t["ev1"] = torch.npu.Event(enable_timing=True)
                _r197t["ev1"].record()
            except Exception:
                pass
        # ---------------------------------------------------------------
        # R218 KVFP (R144_PF_PROBE=7, layer 0 only): per-row content
        # fingerprint of the HP row THIS request's decode reads. Every row of
        # the harness carries the SAME prompt, so HK[slot, 0:pref) (the prefix
        # region, written once at prefill and never rewritten) MUST be
        # bit-identical for every row at every step; a row whose prefix
        # fingerprint differs is reading foreign/zero data out of its HP slot
        # (bad rid->slot binding, a foreign write into the row, or a handover
        # that did not zero). The ring region is printed too (rows with equal
        # `rcnt` must also agree there). One index_select + one D2H per step.
        if self._oscar_layer_ord(layer) == 0 and _os.environ.get("R144_PF_PROBE", "0") == "7":
            try:
                _hkF, _hvF = self._tt_hp_bufs(self._oscar_layer_ord(layer), query.device, query.dtype)
                _mhF = int(_hkF.shape[1])
                _slF = [int(self._tt_slot(b)) for b in range(batch_size)]
                _prF = [int(self._tt_pref.get(self._tt_rid(b), 0)) for b in range(batch_size)]
                _p0F = min(_prF) if _prF else 0
                _iF = torch.tensor(_slF, dtype=torch.int64, device=query.device)
                _selF = _hkF.index_select(0, _iF).float()
                _pfS = _selF[:, :_p0F].abs().sum(dim=(1, 2, 3)).tolist()
                _rgS = _selF[:, _p0F:_mhF].abs().sum(dim=(1, 2, 3)).tolist()
                _rgF = _selF[:, _p0F:_mhF].sum(dim=(1, 2, 3)).tolist()
                _capF = int(self._TT_RECENT)
                _wcF = "0"
                with open("/tmp/r218_kvfp.log", "a") as _fF:
                    for b in range(batch_size):
                        _ridF = self._tt_rid(b)
                        _slb = int(seq_lens[b])
                        _Xb = self._tt_ring_org.get(_ridF)
                        _rcb = int(self._tt_rcnt.get(_ridF, 0) or 0)
                        # content-independent identity fields: where THIS row's
                        # newest token was just written, and where its read
                        # window starts (both modulo the ring capacity)
                        if _Xb is None:
                            _wpos = _prF[b] + _rcb - 1
                            _w0 = _prF[b]
                        else:
                            _wpos = _prF[b] + ((_slb - 1 - int(_Xb)) % _capF)
                            _w0 = _prF[b] + ((_slb - _rcb - int(_Xb)) % _capF)
                        _fF.write(
                            f"b={b} rid={_ridF} slot={_slF[b]} sl={_slb} pref={_prF[b]} "
                            f"rcnt={_rcb} X={_Xb} "
                            f"psum={float(_pfS[b]):.8e} rabs={float(_rgS[b]):.8e} "
                            f"rsum={float(_rgF[b]):.8e} "
                            f"mhp={_mhF} wcap={_capF} wpos={_wpos} w0={_w0}\n"
                        )
            except Exception:
                pass
        merged_all = []
        fused_mu_done = False
        capt: dict[str, Any] | None = None
        _st = _R2XX_G.setdefault("_R109_STATS", {"middle_calls": 0, "middle_ok": 0, "merge": 0, "full": 0})
        if _os.environ.get("R109_CAPTURE", "0") == "1" and _st["middle_calls"] == int(
            _os.environ.get("R109_CAPT_AT", "40")
        ):
            capt = {"layer": layer_idx, "seq_lens": list(seq_lens), "tiers": []}

        # ---------------- R144 G3: batched middle (ONE aclnn B=B for all reqs) ----
        # Replaces the per-request _r109_middle_candidate loop (B B=1 calls -> 1 B=B
        # call) when R144_BATCHED_MIDDLE=1 + B>=2. Per-row blk_lo/mid_len/actualSeqKv
        # (mid_start=pref varies per request). Same prefix/recent/merge semantics.
        # Gated default OFF; B=1 path untouched. Graph replay values come from
        # device tensors (block_tables/seq_lens are device/host-stable metadata).
        _r144_bmid = {}
        _r145_handled = False  # R147: set by the R146 fill path; early-init for the
        _carr: Any = None  # merge section's guards on no-middle steps
        _srB147 = False  # R147: batched S+R path handled-all-rows flag
        # R204_MIXCHAIN (default OFF): per-layer slab view of the middle-op
        # out/lse carriers, handed to _r196_batched_sr so the o_h/l_h merge
        # staging reads the carrier slab directly (2 casts/layer) instead of
        # 2 per-row copy_ kernels per row; None = base per-row staging.
        _slab204 = None
        if layer_idx == 0 and _os.environ.get("R197_DBG", "0") == "1":  # R197 Fix-0
            print(
                f"[R144-GATE] layer0 batch_size={batch_size} BM={_os.environ.get('R144_BATCHED_MIDDLE', '0')} "
                f"r112mod={getattr(type(self), '_r112_mod', None) is not None} "
                f"mid_cpp={_os.environ.get('R113_MIDDLE_CPP', '0')} lse_fast={_os.environ.get('R112_LSE_FAST', '0')}",
                file=_sys.stderr,
                flush=True,
            )
        # R146_PERTURB=3 (OFF-leg mirror): reproduce the R146 staging work (carrier allocs +
        # per-layer staging device ops + python) in the RT=0 leg so layout/timing second-order
        # effects cancel in ON-vs-OFF token comparison. No semantic change (own tensors).
        if _os.environ.get("R146_PERTURB", "0") == "3" and _os.environ.get("R145_TENSOR", "0") != "1":
            _bsM = self.vllm_config.cache_config.block_size
            _nqM = self.num_heads
            _nkvM = self.num_kv_heads
            _hdM = self.head_size
            _mseqM = int(self._tt_mseq)
            _capM = 512
            _nbCM = _capM // _bsM
            _BcapTM = max(_mseqM, batch_size)
            _mir = getattr(type(self), "_r146_mirror", None)
            if _mir is None:
                _kvM = _BcapTM * _nbCM
                _d4M = _hdM // 4
                _mir = {
                    "qrot": torch.zeros(_BcapTM, _nqM, 1, _hdM, dtype=torch.bfloat16, device=query.device),
                    "qsrc": torch.zeros(_BcapTM, _nqM, _hdM, dtype=torch.bfloat16, device=query.device),
                    "out": torch.zeros(_BcapTM, _nqM, 1, _hdM, dtype=torch.bfloat16, device=query.device),
                    "lse": torch.zeros(_BcapTM, _nqM, 1, dtype=torch.float32, device=query.device),
                    "K": torch.zeros(_kvM, _bsM, _nkvM, _d4M, dtype=torch.uint8, device=query.device),
                    "V": torch.zeros(_kvM, _bsM, _nkvM, _d4M, dtype=torch.uint8, device=query.device),
                    "Kc": torch.zeros(_kvM, _nkvM, _bsM, _d4M, dtype=torch.uint8, device=query.device),
                    "Vc": torch.zeros(_kvM, _nkvM, _bsM, _d4M, dtype=torch.uint8, device=query.device),
                    "kS": torch.zeros(_BcapTM, _nkvM, _capM, dtype=torch.bfloat16, device=query.device),
                    "kO": torch.zeros(_BcapTM, _nkvM, _capM, dtype=torch.bfloat16, device=query.device),
                    "vS": torch.zeros(_BcapTM, _nkvM, _capM, dtype=torch.bfloat16, device=query.device),
                    "vO": torch.zeros(_BcapTM, _nkvM, _capM, dtype=torch.bfloat16, device=query.device),
                    "bt": torch.zeros(_BcapTM, _nbCM, dtype=torch.int32, device=query.device),
                    "asq": torch.ones(_BcapTM, dtype=torch.int64, device=query.device),
                    "askv": torch.ones(_BcapTM, dtype=torch.int64, device=query.device),
                    "btStage": torch.zeros(_BcapTM, 16, dtype=torch.int64, device=query.device),
                    "btSel": torch.zeros(_BcapTM, 16, dtype=torch.int64, device=query.device),
                    "arange": torch.arange(_nbCM, dtype=torch.int64, device=query.device).view(1, _nbCM),
                    "blklo": torch.zeros(_BcapTM, dtype=torch.int64, device=query.device),
                    "bidx": torch.zeros(_BcapTM, dtype=torch.int64, device=query.device),
                    "offs": torch.zeros(_BcapTM, _nbCM, dtype=torch.int64, device=query.device),
                    "offsC": torch.zeros(_BcapTM, _nbCM, dtype=torch.int64, device=query.device),
                    "IDX64": torch.zeros(_BcapTM, _nbCM, dtype=torch.int64, device=query.device),
                    "ksgS": torch.zeros(_kvM, _bsM, _nkvM, dtype=torch.bfloat16, device=query.device),
                    "kzgS": torch.zeros(_kvM, _bsM, _nkvM, dtype=torch.bfloat16, device=query.device),
                    "vsgS": torch.zeros(_kvM, _bsM, _nkvM, dtype=torch.bfloat16, device=query.device),
                    "vzgS": torch.zeros(_kvM, _bsM, _nkvM, dtype=torch.bfloat16, device=query.device),
                    "lensPin": torch.ones(_BcapTM, dtype=torch.int64, pin_memory=True),
                    "blkloPin": torch.zeros(_BcapTM, dtype=torch.int64, pin_memory=True),
                    "bidxPin": torch.zeros(_BcapTM, dtype=torch.int64, pin_memory=True),
                }
                _mir["idxFlat"] = _mir["IDX64"].view(-1)
                _mir["bt"].copy_(
                    torch.arange(_BcapTM * _nbCM, dtype=torch.int32, device=query.device).view(_BcapTM, _nbCM)
                )
                type(self)._r146_mirror = _mir
            # per-layer mirrored staging (same op sequence as R146 staging + fills)
            _btFwM = attn_metadata.block_tables
            _msxM = min(int(_btFwM.size(0)), int(_mir["btStage"].size(0)))
            _mir["btStage"][:_msxM, :].copy_(_btFwM[:_msxM, :])
            for _iM in range(_BcapTM):
                _mir["lensPin"][_iM] = 1
                _mir["blkloPin"][_iM] = 0
                _mir["bidxPin"][_iM] = 0
            _mir["askv"].copy_(_mir["lensPin"])
            _mir["blklo"].copy_(_mir["blkloPin"])
            _mir["bidx"].copy_(_mir["bidxPin"])
            torch.add(_mir["blklo"].view(-1, 1), _mir["arange"], out=_mir["offs"])
            torch.index_select(_mir["btStage"], 0, _mir["bidx"], out=_mir["btSel"])
            torch.clamp(_mir["offs"], max=int(_mir["btStage"].size(1)) - 1, out=_mir["offsC"])
            torch.gather(_mir["btSel"], 1, _mir["offsC"], out=_mir["IDX64"])
            torch.index_select(query, 0, _mir["bidx"], out=_mir["qsrc"])
        if (
            _os.environ.get("R144_BATCHED_MIDDLE", "0") == "1"
            and (
                batch_size >= 2
                # R151 P0 fix: B=1 routed through the SAME capacity-carrier
                # batched path (BcapT always >= Bp+1 sentinel; CAP-wide scales
                # satisfy the r146-stride scale-stride contract). The legacy
                # B=1 tight path (scales [1,nkv,mid_len]) violates that
                # contract under R146_STRIDE=1 => OOB scale reads => NaN
                # o_h/l_h => argmax-0 zeros (R150 FIRST_FAIL#2, root-closed
                # R151: B1+middle natural with carrier path / R146S=0).
                or (_os.environ.get("R151_B1_CARRIER", "0") == "1" and batch_size >= 1)
            )
            and getattr(type(self), "_r112_mod", None) is not None
            and _os.environ.get("R113_MIDDLE_CPP", "0") == "1"
            and _os.environ.get("R112_LSE_FAST", "0") == "1"
        ):
            import math as _m144

            _bs = self.vllm_config.cache_config.block_size
            _nq = self.num_heads
            _nkv = self.num_kv_heads
            _hd = self.head_size
            _Rk144 = None
            if self._oscar_R_k is not None and self._oscar_layer_ord(layer) < self._oscar_R_k.shape[0]:
                _Rk144 = self._oscar_R_k[self._oscar_layer_ord(layer)]
            # R218 Plan-F middle-Q contract (paired with the vector pair-pack
            # kernel): the live INT2 packer stores quarter-strided D lanes, so the
            # K tile seen by bmm1 is natural-Krot[..., qperm]. The middle carrier
            # Q must therefore be natural-Qrot[..., qperm]. Express that as a
            # COLUMN-permuted rotation matrix (R[:, qperm]); prefix/recent FIA and
            # SR_CMP references keep the natural _Rk144. Cached per layer tensor.
            _RkMid144 = _Rk144
            if (
                _Rk144 is not None
                and _os.environ.get("R218_QPERM", "0") == "1"
                and _Rk144.size(0) == 128
                and _Rk144.size(1) == 128
            ):
                _lordQ = self._oscar_layer_ord(layer)
                _qkeyQ = (_lordQ, _Rk144.data_ptr(), tuple(_Rk144.shape))
                _qcacheQ = getattr(type(self), "_r218_mid_R", None)
                if _qcacheQ is None or _qcacheQ[0] != _qkeyQ:
                    if getattr(type(self), "_r218_qperm", None) is None:
                        _qq = 32
                        _pi = torch.empty(128, dtype=torch.long, device=_Rk144.device)
                        _pi[0 : 2 * _qq : 2] = torch.arange(_qq, device=_Rk144.device)
                        _pi[1 : 2 * _qq : 2] = torch.arange(_qq, 2 * _qq, device=_Rk144.device)
                        _pi[2 * _qq :: 2] = torch.arange(2 * _qq, 3 * _qq, device=_Rk144.device)
                        _pi[2 * _qq + 1 :: 2] = torch.arange(3 * _qq, 128, device=_Rk144.device)
                        type(self)._r218_qperm = _pi
                    type(self)._r218_mid_R = (_qkeyQ, _Rk144.index_select(1, type(self)._r218_qperm).contiguous())
                _RkMid144 = type(self)._r218_mid_R[1]
            # R144 P1 StageV2: persistent carriers (qrot/out/lse) keyed by slot.
            # When ON: per-row Q rotation writes into the carrier via out= (no
            # per-row alloc), the carrier slice replaces torch.cat (no cat) and
            # torch.zeros (no fresh out/lse). alloc=0/cat=0/zeros=0 per step;
            # addresses stable across replays (graph-safe). Default OFF.
            _V2 = _os.environ.get("R144_BATCHED_STAGE_V2", "0") == "1"
            _carr = None
            if _V2:
                _lid2 = self._oscar_layer_ord(layer)
                _carr = self._r144_carriers.get(_lid2)
                if _carr is None:
                    _mml = int(getattr(self.vllm_config.model_config, "max_model_len", 4096))
                    _mnb = (_mml + _bs - 1) // _bs  # max blocks per row
                    _kv_cap = self._tt_mseq * _mnb  # total gathered-block capacity
                    _d4 = _hd // 4  # int2 packed: 4 vals/byte
                    _carr = {
                        "qrot": torch.zeros(self._tt_mseq, _nq, 1, _hd, dtype=query.dtype, device=query.device),
                        "out": torch.zeros(self._tt_mseq, _nq, 1, _hd, dtype=torch.bfloat16, device=query.device),
                        "lse": torch.zeros(self._tt_mseq, _nq, 1, dtype=torch.float32, device=query.device),
                        # R149 P1: async LSE contract — carriers are initialized
                        # ONCE at creation (fills + stream sync HERE, off the
                        # serving critical path); every eager op call is then
                        # BARE dz=False (NO fills adjacent, NO sync). Evidence:
                        # fills-adjacent no-sync drops 3/3000 (EE/EG/GEE); bare
                        # calls 0/2000 (NOFILL/GENOFILL) incl graph interleave.
                        # R144 P1 launcher StageV2: persistent gather/bt carriers.
                        # K/V native-layout carrier matches the cache layout
                        # [num_blocks, block, nkv, D/4]; index_select_out writes
                        # directly (no alloc). Kc/Vc are the op-layout contiguous
                        # carriers [kv_cap, nkv, block, D/4] filled via transpose-
                        # copy (permute{0,2,1,3}); op requires contiguous [nb,nkv,block,D/4].
                        "K": torch.zeros(_kv_cap, _bs, _nkv, _d4, dtype=torch.uint8, device=query.device),
                        "V": torch.zeros(_kv_cap, _bs, _nkv, _d4, dtype=torch.uint8, device=query.device),
                        "Kc": torch.zeros(_kv_cap, _nkv, _bs, _d4, dtype=torch.uint8, device=query.device),
                        "Vc": torch.zeros(_kv_cap, _nkv, _bs, _d4, dtype=torch.uint8, device=query.device),
                        "kS": torch.zeros(self._tt_mseq, _nkv, _mml, dtype=torch.bfloat16, device=query.device),
                        "kO": torch.zeros(self._tt_mseq, _nkv, _mml, dtype=torch.bfloat16, device=query.device),
                        "vS": torch.zeros(self._tt_mseq, _nkv, _mml, dtype=torch.bfloat16, device=query.device),
                        "vO": torch.zeros(self._tt_mseq, _nkv, _mml, dtype=torch.bfloat16, device=query.device),
                        "bt": torch.zeros(self._tt_mseq, _mnb, dtype=torch.int32, device=query.device),
                    }
                    self._r144_carriers[_lid2] = _carr
            _bl = []
            _nb_l = []
            _ml_l = []
            _q_l = []
            _bidx: list[Any] = []
            # ================= R145: fixed-capacity TENSOR-ABI batched middle =================
            # R145_TENSOR=1 (default OFF): sentinel-row capacity contract + device int64
            # askv carrier + optional ONE captured graph per bucket (R145_GRAPH=1).
            # Takes precedence over the V1/V2 paths below; the per-row loop below is
            # skipped when this path handled the step.
            _R145T = _os.environ.get("R145_TENSOR", "0") == "1"
            if _os.environ.get("R218_BRANCH", "0") == "1":
                try:
                    # _fbB: _fb is already bound as a PATH string (r185_makeacl.bin)
                    # in this method -- reusing the name for the handle trips mypy
                    with open("/tmp/r218_branch_{}.log".format(_os.environ.get("R218_BRANCH_TAG", "x")), "a") as _fbB:
                        _fbB.write(
                            f"[BR] pre-dispatch L={self._oscar_layer_ord(layer)} B={batch_size} R145T={_R145T} "
                            f"R146F={_os.environ.get('R146_FILL', '0') == '1'} sl0={int(seq_lens[0])} "
                            f"pr0={self._tt_pref.get(self._tt_rid(0), -1)} "
                            f"rc0={self._tt_rcnt.get(self._tt_rid(0), -1)}\n"
                        )
                except Exception:
                    pass
            _R145G = _os.environ.get("R145_GRAPH", "0") == "1"
            _R146F = _os.environ.get("R146_FILL", "0") == "1"  # R146 P2: static graph-captured fills
            _R146S = _os.environ.get("R146_STRIDE", "0") == "1"  # R146 P1: stride=seqInnerSize .o (sentinel-free)
            _r145_handled = False
            if _R145T and not _R146F and _os.environ.get("R146_PERTURB", "0") == "1":
                _dummy = torch.ones(8, 8, dtype=torch.bfloat16, device=query.device)
                _dummy2 = torch.matmul(_dummy, _dummy)
            if _R145T and not _R146F and _os.environ.get("R146_PERTURB", "0") == "2":
                # allocation-footprint probe: ONCE, allocate a carrier-sized blob (~MBs) and
                # hold it on the impl; never used. If tokens flip => layout/address sensitivity.
                if getattr(type(self), "_r146_dummy_blob", None) is None:
                    _blob = torch.zeros(3 * 1024 * 1024, dtype=torch.uint8, device=query.device)
                    type(self)._r146_dummy_blob = _blob
            if _R145T and not _R146F:
                _step = int(_os.environ.get("R145_CAP_STEP", "256"))
                _mml_top = int(getattr(self.vllm_config.model_config, "max_model_len", 4096))
                # R150 P3: doubling buckets (256,512,...,8192) — one carrier set
                # per bucket, lazy; min legal bucket still selected per step.
                if _os.environ.get("R150_POW2_LADDER", "1") == "1":
                    _ladder = []
                    _lv = _step
                    while _lv < _mml_top:
                        _ladder.append(_lv)
                        _lv *= 2
                    _ladder.append(_lv)
                else:
                    _ladder = list(range(_step, _mml_top + _step, _step))
                _cb = []
                _cbl = []
                _cnb = []
                _cml = []
                for _bb in range(batch_size):
                    _sl = int(seq_lens[_bb])
                    _pr = self._tt_pref.get(self._tt_rid(_bb), 0)
                    _rc = self._tt_rcnt.get(self._tt_rid(_bb), 0)
                    _ms = _pr
                    _me = _sl - _rc
                    _ml = int(_me - _ms)
                    if _ml <= 0 or (_ms % _bs) != 0:
                        continue
                    _blo = _ms // _bs
                    _bhi = (_me + _bs - 1) // _bs
                    _nb2 = _bhi - _blo
                    if _nb2 <= 0:
                        continue
                    _cb.append(int(_bb))
                    _cbl.append(int(_blo))
                    _cnb.append(int(_nb2))
                    _cml.append(int(_ml))
                _cap = 0
                for _L in _ladder:
                    if max(_cml, default=0) <= _L:
                        _cap = _L
                        break
                if _cap == 0 and len(_cb) > 0 and _ladder:
                    _cap = _ladder[-1]  # over-ladder: use top bucket (fail-closed would be better)
                if len(_cb) >= 1 and _cap > 0:
                    _Bp = len(_cb)
                    _Bcap = max(int(self._tt_mseq), _Bp)
                    _BcapT = _Bcap + 1
                    _nbC = _cap // _bs
                    _carr = self._r145_carriers.get(_cap)
                    if _carr is None:
                        _kv_cap = _BcapT * _nbC
                        _d4 = _hd // 4
                        _carr = {
                            "qrot": torch.zeros(_BcapT, _nq, 1, _hd, dtype=query.dtype, device=query.device),
                            "out": torch.zeros(_BcapT, _nq, 1, _hd, dtype=torch.bfloat16, device=query.device),
                            "lse": torch.zeros(_BcapT, _nq, 1, dtype=torch.float32, device=query.device),
                            "K": torch.zeros(_kv_cap, _bs, _nkv, _d4, dtype=torch.uint8, device=query.device),
                            "V": torch.zeros(_kv_cap, _bs, _nkv, _d4, dtype=torch.uint8, device=query.device),
                            "Kc": torch.zeros(_kv_cap, _nkv, _bs, _d4, dtype=torch.uint8, device=query.device),
                            "Vc": torch.zeros(_kv_cap, _nkv, _bs, _d4, dtype=torch.uint8, device=query.device),
                            "kS": torch.zeros(_BcapT, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                            "kO": torch.zeros(_BcapT, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                            "vS": torch.zeros(_BcapT, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                            "vO": torch.zeros(_BcapT, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                            "bt": torch.zeros(_BcapT, _nbC, dtype=torch.int32, device=query.device),
                            "asq": torch.ones(_BcapT, dtype=torch.int64, device=query.device),
                            "askv": torch.ones(_BcapT, dtype=torch.int64, device=query.device),
                            "graph": None,
                        }
                        _carr["askv"][_BcapT - 1] = _cap  # sentinel row: stride == storage CAP
                        # bt content is address-layout-constant: bt[r,j] = r*nbC + j
                        _carr["bt"].copy_(
                            torch.arange(_BcapT * _nbC, dtype=torch.int32, device=query.device).view(_BcapT, _nbC)
                        )
                        self._r145_carriers[_cap] = _carr
                        try:
                            # hoist: a newline inside f-string braces is a
                            # SyntaxError on Python <= 3.11
                            _mbC = sum(v.numel() * v.element_size() for v in _carr.values() if torch.is_tensor(v)) / 1e6
                            with open("/tmp/r145_probe.log", "a") as _pf:
                                _pf.write(
                                    f"[R145-CARR] cap={_cap} BcapT={_BcapT} nbC={_nbC} kv_cap={_kv_cap} MB={_mbC:.1f}\n"
                                )
                        except Exception:
                            pass
                    # q rotation into carrier rows (eager, per layer)
                    for _ci, _bb in enumerate(_cb):
                        _qb = query[_bb : _bb + 1]
                        if _qb.dim() == 3:
                            _qb = _qb.unsqueeze(2)
                        if _Rk144 is not None:
                            torch.matmul(_qb, _RkMid144, out=_carr["qrot"][_ci : _ci + 1])
                        else:
                            _carr["qrot"][_ci : _ci + 1].copy_(_qb)
                    # R191 P2: run the REAL fixture LSE right BEFORE the v3 carrier fill.
                    # Ladder: R190 entry GREEN -> P2(PRE-FILL) ? -> P3(POST-FILL) ? -> live RED.
                    # GREEN here + RED at P3 pins the predecessor to the v3 fill itself.
                    import os as _os191a

                    if _os191a.environ.get("R191_PROBE") == "1" and not getattr(type(self), "_r191_p2", False):
                        type(self)._r191_p2 = True
                        _A191 = _odiag_p("r191_audit.log")
                        try:
                            if getattr(type(self), "_r189_mod", None) is None:
                                import torch.utils.cpp_extension as _ce191

                                _R152 = _os.environ.get("R152_LAUNCHER_DIR", "")
                                _TN2 = os.path.dirname(torch_npu.__file__)
                                _INC9 = [
                                    os.path.join(_TN2, "include"),
                                    os.path.join(_TN2, "include", "third_party", "op-plugin"),
                                    "/usr/local/Ascend/cann-9.0.0/aarch64-linux/include",
                                    "/usr/local/Ascend/cann-9.0.0/include",
                                    _os.environ.get("R183_COMPOSITE_OPP", "") + "/op_api/include",
                                ]
                                _LD9 = [
                                    "-L/usr/local/Ascend/cann-9.0.0/lib64",
                                    "-lascendcl",
                                    "-lopapi",
                                    "-L" + os.path.join(_TN2, "lib"),
                                    "-ltorch_npu",
                                    "-L" + _os.environ.get("R183_COMPOSITE_OPP", "") + "/op_api/lib",
                                    "-lcust_opapi",
                                    "-Wl,-rpath,/usr/local/Ascend/cann-9.0.0/lib64",
                                    "-Wl,-rpath," + _os.environ.get("R183_COMPOSITE_OPP", "") + "/op_api/lib",
                                ]
                                _os191a.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/r191_ext"
                                _os191a.makedirs(_os191a.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
                                type(self)._r189_mod = _ce191.load(
                                    name="r191mod",
                                    sources=[_R152 + "/int2_op_launch_lse_fast_r152.cpp"],
                                    extra_include_paths=_INC9,
                                    extra_ldflags=_LD9,
                                    build_directory=_os191a.environ["TORCH_EXTENSIONS_DIR"],
                                    verbose=False,
                                )
                            import hashlib as _hl191a
                            import json as _j191

                            _fb = _odiag_p("r185_makeacl.bin")
                            _fj = _odiag_p("r185_makeacl.json")
                            with open(_fj) as _fjH:
                                _J = _j191.load(_fjH)
                            with open(_fb, "rb") as _fbH:
                                _raw = _fbH.read()

                            def _ld191(nm):
                                t = [x for x in _J["tensors"] if x["name"] == nm][0]
                                o, nb, sh, dt = t["offset"], t["nbytes"], t["shape"], t["dtype"]
                                b = _raw[o : o + nb]
                                if "BFloat16" in dt:
                                    a = torch.frombuffer(b, dtype=torch.uint16).reshape(sh).view(torch.bfloat16)
                                elif "signed char" in dt:
                                    a = torch.frombuffer(b, dtype=torch.int8).reshape(sh)
                                elif dt == "int":
                                    a = torch.frombuffer(b, dtype=torch.int32).reshape(sh)
                                else:
                                    a = torch.frombuffer(b, dtype=torch.uint8).reshape(sh)
                                return a.clone().to(device=query.device)

                            _rq, _rk, _rv, _rbt, _rks, _rko, _rvs, _rvo = (
                                _ld191("q"),
                                _ld191("k"),
                                _ld191("v"),
                                _ld191("bt"),
                                _ld191("ks"),
                                _ld191("ko"),
                                _ld191("vs"),
                                _ld191("vo"),
                            )
                            _ro = torch.zeros(1, 32, 1, 128, dtype=torch.bfloat16, device=query.device)
                            _rl = torch.zeros(1, 32, 1, dtype=torch.float32, device=query.device)

                            def _h191a(t):
                                return _hl191a.sha256(
                                    t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                                ).hexdigest()[:12]

                            open(_A191, "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                                f"[FIXTURE_HASH] q={_h191a(_rq)} k={_h191a(_rk)} v={_h191a(_rv)} "
                                f"bt={_h191a(_rbt)} ks={_h191a(_rks)}\n"
                            )
                            _rr = type(self)._r189_mod.launch_lse_fast(
                                _rq,
                                _rk,
                                _rv,
                                _rbt,
                                _rks,
                                _rko,
                                _rvs,
                                _rvo,
                                _ro,
                                _rl,
                                32,
                                1.0 / (128**0.5),
                                8,
                                "BNSD",
                                128,
                                1,
                                1,
                                [2176],
                                torch.zeros(0, dtype=torch.bfloat16, device=query.device),
                                0,
                                0,
                                0,
                                True,
                            )
                            torch.npu.synchronize()
                            open(_A191, "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                                f"[P2_PRE-FILL] rc={int(_rr[0])} "
                                f"finite={torch.isfinite(_ro).all().item()} "
                                f"absmax={float(_ro.abs().max()):.4g}\n"
                            )
                            type(self)._r191_fx = (_rq, _rk, _rv, _rbt, _rks, _rko, _rvs, _rvo)
                        except Exception as _e191a:
                            open(_A191, "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                                f"[P2_PRE-FILL] CRASH: {str(_e191a)[:120]}\n"
                            )
                    _r3 = type(self)._r112_mod.oscar_middle_prep_v3_fill(
                        attn_metadata.block_tables,
                        _cb,
                        _cbl,
                        _cnb,
                        _cml,
                        self.key_cache,
                        self.value_cache,
                        self.k_scale_cache,
                        self.k_zero_cache,
                        self.v_scale_cache,
                        self.v_zero_cache,
                        _carr["K"],
                        _carr["V"],
                        _carr["Kc"],
                        _carr["Vc"],
                        _carr["kS"],
                        _carr["kO"],
                        _carr["vS"],
                        _carr["vO"],
                        _carr["bt"],
                        _carr["askv"],
                        _Bp,
                        _BcapT,
                        _nbC,
                        _cap,
                        _nkv,
                        _bs,
                    )
                    # R185 OPERAND SNAPSHOT (env R185_OPDUMP): capture the EXACT V3 carrier tensors
                    # entering oscar_middle_op_v3 (the crash-path middle LSE call), for standalone
                    # MSD_ON replay. Carriers just filled by oscar_middle_prep_v3_fill; full sync
                    # first. Read-only; single-shot (first middle-tier V3 call = crash call).
                    import os as _os185c

                    # R215: R185_OPDUMP_MINB — only capture once the mixed batch has at
                    # least this many real rows (default 1 = R185 behavior). The B=1 base
                    # path was captured in R185/R192; R215 needs a B>=2 GROUP-M capture.
                    if (
                        _os185c.environ.get("R185_OPDUMP") == "1"
                        and not getattr(type(self), "_r185c_dumped", False)
                        and int(_Bp) >= int(_os185c.environ.get("R185_OPDUMP_MINB", "1"))
                    ):
                        type(self)._r185c_dumped = True
                        torch.npu.synchronize()
                        _odc = _os185c.environ.get("R185_OPDUMP_DIR", "/tmp/r185")
                        _os185c.makedirs(_odc, exist_ok=True)
                        _sc185c = 1.0 / _m144.sqrt(_hd)
                        _snap = {
                            "q": _carr["qrot"].detach().cpu().clone(),
                            "k": _carr["Kc"].detach().cpu().clone(),
                            "v": _carr["Vc"].detach().cpu().clone(),
                            "bt": _carr["bt"].detach().cpu().clone(),
                            "ks": _carr["kS"].detach().cpu().clone(),
                            "ko": _carr["kO"].detach().cpu().clone(),
                            "vs": _carr["vS"].detach().cpu().clone(),
                            "vo": _carr["vO"].detach().cpu().clone(),
                            "out_h": _carr["out"].detach().cpu().clone(),
                            "lse_h": _carr["lse"].detach().cpu().clone(),
                            "askv": list(int(x) for x in _carr["askv"].cpu().tolist()),
                            "asq": list(int(x) for x in _carr["asq"].cpu().tolist())
                            if _carr["asq"].numel() > 0
                            else [],
                            "nq": int(_nq),
                            "scale": float(_sc185c),
                            "nkv": int(_nkv),
                            "block_size": int(_bs),
                            "cap": int(_cap),
                            "nbC": int(_nbC),
                            "layer_idx": int(layer_idx),
                            "b_list": list(int(x) for x in _cb),
                            "q_shape": list(_carr["qrot"].shape),
                            "q_stride": list(_carr["qrot"].stride()),
                            "k_shape": list(_carr["Kc"].shape),
                            "k_stride": list(_carr["Kc"].stride()),
                            "bt_shape": list(_carr["bt"].shape),
                            "bt_stride": list(_carr["bt"].stride()),
                            "ks_shape": list(_carr["kS"].shape),
                            "ks_stride": list(_carr["kS"].stride()),
                            "bt_min": int(_carr["bt"].min()),
                            "bt_max": int(_carr["bt"].max()),
                            "ks_absmax": float(_carr["kS"].abs().max()),
                            "ko_absmax": float(_carr["kO"].abs().max()),
                        }
                        _pc = f"{_odc}/r185_v3_l{layer_idx}_cap{_cap}.pt"
                        torch.save(_snap, _pc)
                        print(
                            f"[R185-V3-OPDUMP] l={layer_idx} cap={_cap} nbC={_nbC} nq={_nq} nkv={_nkv} bs={_bs} "
                            f"askv={_snap['askv']} bt_min={_snap['bt_min']} bt_max={_snap['bt_max']} "
                            f"ks_absmax={_snap['ks_absmax']:.6g} ko_absmax={_snap['ko_absmax']:.6g} "
                            f"k_shape={_snap['k_shape']} bt_shape={_snap['bt_shape']} -> {_pc}",
                            flush=True,
                        )
                    if _R145G:
                        if _carr["graph"] is None:
                            _s145 = torch.npu.Stream()
                            _s145.wait_stream(torch.npu.current_stream())
                            with torch.npu.stream(_s145):
                                for _ in range(3):
                                    type(self)._r112_mod.oscar_middle_op_v3(
                                        _carr["qrot"],
                                        _carr["Kc"],
                                        _carr["Vc"],
                                        _carr["bt"],
                                        _carr["kS"],
                                        _carr["kO"],
                                        _carr["vS"],
                                        _carr["vO"],
                                        _carr["out"],
                                        _carr["lse"],
                                        _carr["asq"],
                                        _carr["askv"],
                                        _nq,
                                        1.0 / _m144.sqrt(_hd),
                                        _nkv,
                                        _bs,
                                        True,
                                    )
                            torch.npu.current_stream().wait_stream(_s145)
                            torch.npu.synchronize()
                            _g145 = torch.npu.NPUGraph()
                            with torch.npu.graph(_g145, stream=_s145):
                                _rc145 = type(self)._r112_mod.oscar_middle_op_v3(
                                    _carr["qrot"],
                                    _carr["Kc"],
                                    _carr["Vc"],
                                    _carr["bt"],
                                    _carr["kS"],
                                    _carr["kO"],
                                    _carr["vS"],
                                    _carr["vO"],
                                    _carr["out"],
                                    _carr["lse"],
                                    _carr["asq"],
                                    _carr["askv"],
                                    _nq,
                                    1.0 / _m144.sqrt(_hd),
                                    _nkv,
                                    _bs,
                                    True,
                                )
                            torch.npu.synchronize()
                            if int(_rc145[0]) != 0:
                                raise RuntimeError(f"[R145] graph capture getws rc={_rc145[0]}")
                            _carr["graph"] = _g145
                            try:
                                with open("/tmp/r145_probe.log", "a") as _pf:
                                    _pf.write(f"[R145-GRAPH] captured cap={_cap} rc={int(_rc145[0])}\n")
                            except Exception:
                                pass
                        _carrG = _carr  # R146 graph arm: carriers created above
                        assert _carrG is not None
                        _carrG["graph"].replay()
                        _r2, _lr2 = 0, 0
                    else:
                        # R185 FIX: MSD_ON crashes under the Tensor-variant API (oscar_middle_op_v3,
                        # device-resident asq/askv -> 507015 in the MSD_ON kernel). The non-Tensor
                        # launch_lse_fast (host askv) replays the SAME operands rc=0 finite
                        # (R185 standalone replay). For B=1 MSD_ON, route through the non-Tensor
                        # variant. Default OFF (env R185_NON_TENSOR_FIX); MSD_OFF keeps the V3 path.
                        # R191 P3: REAL fixture LSE right AFTER the v3 carrier fill (predecessor
                        # just ran on this stream). RED here vs GREEN at P2 => the v3 fill is
                        # the minimal predecessor. GREEN here vs live RED => live data is the
                        # trigger, not the fill.
                        import os as _os191b

                        if (
                            _os191b.environ.get("R191_PROBE") == "1"
                            and getattr(type(self), "_r191_fx", None) is not None
                            and not getattr(type(self), "_r191_p3", False)
                        ):
                            type(self)._r191_p3 = True
                            _A191 = _odiag_p("r191_audit.log")
                            try:
                                _fx = type(self)._r191_fx
                                _ro3 = torch.zeros(1, 32, 1, 128, dtype=torch.bfloat16, device=query.device)
                                _rl3 = torch.zeros(1, 32, 1, dtype=torch.float32, device=query.device)
                                _rr3p = type(self)._r189_mod.launch_lse_fast(
                                    _fx[0],
                                    _fx[1],
                                    _fx[2],
                                    _fx[3],
                                    _fx[4],
                                    _fx[5],
                                    _fx[6],
                                    _fx[7],
                                    _ro3,
                                    _rl3,
                                    32,
                                    1.0 / (128**0.5),
                                    8,
                                    "BNSD",
                                    128,
                                    1,
                                    1,
                                    [2176],
                                    torch.zeros(0, dtype=torch.bfloat16, device=query.device),
                                    0,
                                    0,
                                    0,
                                    True,
                                )
                                torch.npu.synchronize()
                                open(_A191, "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                                    f"[P3_POST-FILL] rc={int(_rr3p[0])} "
                                    f"finite={torch.isfinite(_ro3).all().item()} "
                                    f"absmax={float(_ro3.abs().max()):.4g}\n"
                                )
                            except Exception as _e191b:
                                open(_A191, "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                                    f"[P3_POST-FILL] CRASH: {str(_e191b)[:120]}\n"
                                )
                        if (
                            _os.environ.get("R174_INT2_MSD") == "1"
                            and _os.environ.get("R185_NON_TENSOR_FIX", "0") == "1"
                            and len(_cb) == 1
                        ):
                            print(
                                f"[R185-FIX] routing MSD_ON B=1 through non-Tensor launch_lse_fast l={layer_idx}",
                                flush=True,
                            )
                            _askv_host = [int(_carr["askv"][0].item())]
                            # R191 P4: live args + live-content hashes at the Grouped 1/G launch.
                            if _os.environ.get("R191_AUDIT", "0") == "1" and not getattr(type(self), "_r191_p4", False):
                                type(self)._r191_p4 = True
                                _A191 = _odiag_p("r191_audit.log")
                                try:
                                    import hashlib as _hl191c

                                    def _h191c(t):
                                        return _hl191c.sha256(
                                            t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                                        ).hexdigest()[:12]

                                    open(_A191, "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                                        f"[P4_LIVE_ARGS] askv={_askv_host} nq={int(_nq)} nkv={int(_nkv)} bs={int(_bs)} "
                                        f"qshape={list(_carr['qrot'].shape)} qstride={list(_carr['qrot'].stride())} "
                                        f"kshape={list(_carr['Kc'].shape)} kstride={list(_carr['Kc'].stride())} "
                                        f"bt=[{int(_carr['bt'].min())},{int(_carr['bt'].max())}] "
                                        f"qh={_h191c(_carr['qrot'])} kh={_h191c(_carr['Kc'].view(torch.int8))} "
                                        f"vh={_h191c(_carr['Vc'].view(torch.int8))} "
                                        f"bth={_h191c(_carr['bt'])} ksh={_h191c(_carr['kS'])} "
                                        f"[GROUPED_1G_LIVE] pid={_os.getpid()}\n"
                                    )
                                except Exception as _e191c:
                                    open(_A191, "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                                        f"[P4_ERR] {_e191c}\n"
                                    )
                            _k8v = _carr["Kc"].view(torch.int8)
                            _v8v = _carr["Vc"].view(torch.int8)
                            # R189 PROBE: before real middle LSE
                            if (
                                os.environ.get("R189_PROBE") == "1"
                                and getattr(type(self), "_r189_mod", None) is not None
                                and not getattr(type(self), "_r189_real_probed", False)
                            ):
                                type(self)._r189_real_probed = True
                                _pd = _carr["qrot"].device
                                _pD, _pnq, _pnkv, _pbs, _pnb = 128, 32, 8, 128, 17
                                _pS = _pnb * _pbs
                                _pq = (torch.randn(1, _pnq, 1, _pD, device=_pd) * 0.5).bfloat16()
                                _pk = torch.randint(
                                    0, 256, (_pnb, _pnkv, _pbs, _pD // 4), device=_pd, dtype=torch.uint8
                                ).view(torch.int8)
                                _pv = torch.randint(
                                    0, 256, (_pnb, _pnkv, _pbs, _pD // 4), device=_pd, dtype=torch.uint8
                                ).view(torch.int8)
                                _pbt = torch.arange(_pnb, dtype=torch.int32, device=_pd).unsqueeze(0)
                                _pks = (torch.rand(1, _pnkv, _pS, device=_pd) * 0.05 + 0.001).bfloat16()
                                _pko = (torch.randn(1, _pnkv, _pS, device=_pd) * 0.02).bfloat16()
                                _pvs = (torch.rand(1, _pnkv, _pS, device=_pd) * 0.05 + 0.001).bfloat16()
                                _pvo = (torch.randn(1, _pnkv, _pS, device=_pd) * 0.02).bfloat16()
                                _po = torch.zeros(1, _pnq, 1, _pD, dtype=torch.bfloat16, device=_pd)
                                _pl = torch.zeros(1, _pnq, 1, dtype=torch.float32, device=_pd)
                                try:
                                    _pr = type(self)._r189_mod.launch_lse_fast(
                                        _pq,
                                        _pk,
                                        _pv,
                                        _pbt,
                                        _pks,
                                        _pko,
                                        _pvs,
                                        _pvo,
                                        _po,
                                        _pl,
                                        _pnq,
                                        1.0 / (_pD**0.5),
                                        _pnkv,
                                        "BNSD",
                                        _pbs,
                                        1,
                                        1,
                                        [_pS],
                                        torch.zeros(0, dtype=torch.bfloat16, device=_pd),
                                        0,
                                        0,
                                        0,
                                        True,
                                    )
                                    torch.npu.synchronize()
                                    _odiag_w(
                                        "r190_probe.log",
                                        f"[R189-PROBE] BEFORE-REAL-LSE rc={int(_pr[0])} "
                                        f"finite={torch.isfinite(_po).all().item()}\n",
                                    )
                                except Exception as _pe:
                                    _odiag_w("r190_probe.log", f"[R189-PROBE] BEFORE-REAL-LSE CRASH: {str(_pe)[:80]}\n")

                            _rr3 = type(self)._r112_mod.launch_lse_fast(
                                _carr["qrot"],
                                _k8v,
                                _v8v,
                                _carr["bt"],
                                _carr["kS"],
                                _carr["kO"],
                                _carr["vS"],
                                _carr["vO"],
                                _carr["out"],
                                _carr["lse"],
                                _nq,
                                1.0 / _m144.sqrt(_hd),
                                _nkv,
                                "BNSD",
                                _bs,
                                1,
                                1,
                                _askv_host,
                                torch.zeros(0, dtype=torch.uint8, device=query.device),
                                0,
                                0,
                                0,
                                True,
                            )
                        else:
                            _rr3 = type(self)._r112_mod.oscar_middle_op_v3(
                                _carr["qrot"],
                                _carr["Kc"],
                                _carr["Vc"],
                                _carr["bt"],
                                _carr["kS"],
                                _carr["kO"],
                                _carr["vS"],
                                _carr["vO"],
                                _carr["out"],
                                _carr["lse"],
                                _carr["asq"],
                                _carr["askv"],
                                _nq,
                                1.0 / _m144.sqrt(_hd),
                                _nkv,
                                _bs,
                                True,
                            )
                        _r2, _lr2 = int(_rr3[0]), int(_rr3[2])
                    if not (_r2 == 0 and _lr2 == 0):
                        raise RuntimeError(f"[R145] middle v3 NOT ok rc={_r2} lr={_lr2}")
                    # R218 MIDCAP: one-shot capture of the EXACT op operands at
                    # the B=1 direct-launch site (the stage-wrapper LSEDUMP hook
                    # never fires on this path) for offline fp32 adjudication.
                    _r218mc = globals().setdefault("_R218_MIDCAP", [0])
                    if _r218mc[0] < 2 and _os.environ.get("R216_LSEDUMP", "0") == "1":
                        _r218mc[0] += 1
                        try:
                            _bpQ = len(_cb)
                            _nbCQ = int(_carr["bt"].shape[1]) if _carr["bt"].dim() > 1 else int(_carr["bt"].shape[0])
                            torch.npu.synchronize()
                            torch.save(
                                {
                                    "idx": 9000 + _r218mc[0],
                                    "layer": int(self._oscar_layer_ord(layer)),
                                    "cap": 0,
                                    "nbC": _nbCQ,
                                    "bp": _bpQ,
                                    "bidx": [int(x) for x in _cb],
                                    "replay": False,
                                    "nq": int(_nq),
                                    "nkv": int(_nkv),
                                    "bs": int(_bs),
                                    "hd": int(_hd),
                                    "scale": float(1.0 / _m144.sqrt(_hd)),
                                    "askv": list(_askv_host),
                                    "asq": [int(x) for x in _carr["asq"][:_bpQ].cpu().tolist()],
                                    "q": _carr["qrot"][:_bpQ].detach().cpu().clone(),
                                    "k": _carr["Kc"][: _bpQ * _nbCQ].detach().cpu().clone(),
                                    "v": _carr["Vc"][: _bpQ * _nbCQ].detach().cpu().clone(),
                                    "bt": _carr["bt"][:_bpQ].detach().cpu().clone(),
                                    "ks": _carr["kS"][:_bpQ].detach().cpu().clone(),
                                    "ko": _carr["kO"][:_bpQ].detach().cpu().clone(),
                                    "vs": _carr["vS"][:_bpQ].detach().cpu().clone(),
                                    "vo": _carr["vO"][:_bpQ].detach().cpu().clone(),
                                    "out_h": _carr["out"][:_bpQ].detach().cpu().clone(),
                                    "lse_h": _carr["lse"][:_bpQ].detach().cpu().clone(),
                                },
                                f"/tmp/r218_midcap_{_r218mc[0]}.pt",
                            )
                            _r218_diag(
                                f"MIDCAP saved #{_r218mc[0]} lay={int(self._oscar_layer_ord(layer))} "
                                f"bp={_bpQ} nbC={_nbCQ}"
                            )
                        except Exception as _eMC:
                            _r218_diag("MIDCAP ERR {!r}".format(_eMC))
                    for _i, _bb in enumerate(_cb):
                        _r218_diag("MIDSITE-10937")
                        _r144_bmid[_bb] = (_carr["out"][_i, :, 0, :].float(), _carr["lse"][_i, :, 0].float())
                    _st["middle_calls"] += 1
                    _st["middle_ok"] += 1
                    _r145_handled = True
                    try:
                        with open("/tmp/r145_probe.log", "a") as _pf:
                            _pf.write(
                                f"[R145-BATCH] L={layer_idx} B={batch_size} Bp={_Bp} cap={_cap} "
                                f"bidx={_cb} ml={_cml} graph={_R145G} rc={_r2} lr={_lr2}\n"
                            )
                    except Exception:
                        pass
            # ================= R146: static graph-captured fills + optional sentinel-free stride =====
            # R146_FILL=1 (default OFF; requires R145_TENSOR=1): replaces the R145 per-row eager
            # fill (~8,900 tiny kernels/step at B8 = 221ms device + dominant host dispatch) with
            # STATIC-shape fills (full-width CAP gathers; dynamic content ONLY via device tensors:
            # block-id index carrier IDX64 [BcapT,nbC], lens->askv, bidx, blk_lo — 3 pinned H2D per
            # STEP shared across 36 layers) + ONE NPUGraph per (cap,layer) capturing fill+op.
            # Tokens beyond each row's logical len carry stale-but-masked content (rows independent,
            # tail masked by the op) — same contract as R145. R146_STRIDE=1 additionally uses the
            # r146_stride candidate .o (stride=seqInnerSize; NO sentinel row; BcapT=Bcap).
            if _R145T and _R146F and _os.environ.get("R147_ADAPTER", "0") == "1":
                # R147 P5: thin hook — all carrier/staging/graph logic lives in
                # OscarGraphAdapter (oscar_graph_adapter.py); shapes derived from
                # runtime config; layer ordinal passed EXPLICITLY.
                from vllm_ascend.attention.oscar_graph_adapter import OscarGraphAdapter as _OGA

                _adp = getattr(self, "_r147_adapter", None)
                if _adp is None:
                    _adp = _OGA(self)
                    self._r147_adapter = _adp
                _r147res = _adp.middle_decode(
                    layer=layer,
                    layer_ord=self._oscar_layer_ord(layer),
                    query=query,
                    seq_lens=seq_lens,
                    attn_metadata=attn_metadata,
                    batch_size=batch_size,
                    block_size=_bs,
                    num_heads=_nq,
                    num_kv_heads=_nkv,
                    head_size=_hd,
                    stats=_st,
                    r144_bmid=_r144_bmid,
                    Rk=_RkMid144,
                    graph_on=_R145G,
                    stride_on=_R146S,
                )
                if _r147res is not None:
                    _r145_handled = True
                    _carr = _r147res["carr"]
                    _cb = _r147res["cb"]
                    _cbl = _r147res["cbl"]
                    _cnb = _r147res["cnb"]
                    _cml = _r147res["cml"]
                    _cap = _r147res["cap"]
                    _Bp = _r147res["Bp"]
                    _BcapT = _r147res["BcapT"]
            if _R145T and _R146F and _os.environ.get("R147_ADAPTER", "0") != "1":
                _step = int(_os.environ.get("R145_CAP_STEP", "256"))
                _mml_top = int(getattr(self.vllm_config.model_config, "max_model_len", 4096))
                _ladder = list(range(_step, _mml_top + _step, _step))
                _cb = []
                _cbl = []
                _cnb = []
                _cml = []
                _r197fa = _os.environ.get("R197_EAGER_MIX", "0") == "1"
                for _bb in range(batch_size):
                    _sl = int(seq_lens[_bb])
                    _pr = self._tt_pref.get(self._tt_rid(_bb), 0)
                    _rc = self._tt_rcnt.get(self._tt_rid(_bb), 0)
                    _ms = _pr
                    _me = _sl - _rc
                    _ml = int(_me - _ms)
                    if _ml <= 0:
                        continue
                    if (_ms % _bs) != 0:
                        # R197 Fix-3 (R197_EAGER_MIX=1, default OFF): a row
                        # whose mid START is not block-aligned was DROPPED from
                        # the batched carrier (then fell to the per-row middle
                        # candidate at ~2-4ms/row/layer — the surviving mixed-
                        # step ramp; it also broke the R196 S+R gate, silently
                        # reverting EVERY such step to the legacy per-row loop).
                        # Floor-align its start: the <=bs-1 boundary tokens
                        # move from the BF16 S tier to the INT2 H tier — same
                        # tokens attended exactly once (the three-tier boundary
                        # is an implementation artifact; TT_PREFIX itself is
                        # env-tunable). Aligned rows (pref=128) are unchanged.
                        if not _r197fa or _os.environ.get("R197_NOALIGN", "0") == "1":
                            if _r144_ovl_os.environ.get("R197_PROBE", "0") == "1":
                                open("/tmp/r197_cbskip.log", "a").write(  # noqa: SIM115 - fire-and-forget diagnostic write
                                    f"cbskip B={batch_size} bb={_bb} sl={_sl} pr={_pr} rc={_rc} ml={_ml}\n"
                                )
                            continue
                        _ms -= _ms % _bs
                        _ml = int(_me - _ms)
                    _blo = _ms // _bs
                    _bhi = (_me + _bs - 1) // _bs
                    _nb2 = _bhi - _blo
                    if _nb2 <= 0:
                        continue
                    _cb.append(int(_bb))
                    _cbl.append(int(_blo))
                    _cnb.append(int(_nb2))
                    _cml.append(int(_ml))
                _cap = 0
                for _L in _ladder:
                    if max(_cml, default=0) <= _L:
                        _cap = _L
                        break
                if _cap == 0 and len(_cb) > 0 and _ladder:
                    _cap = _ladder[-1]
                if len(_cb) >= 1 and _cap > 0:
                    _Bp = len(_cb)
                    # R205_HPFIT: the mid-op carrier/launch width must stay at
                    # the VALIDATED width (env R109_TT_MAX_SEQS, = 96 on every
                    # composed-stack leg ever run) even when the HP pool itself
                    # is right-sized to max_num_seqs+margin. oscar_middle_op_v3
                    # is NOT width-safe at other launch widths (R201 MIXTRIM
                    # adjudication: every narrow width rc=0 but garbage —
                    # r205 h1/ms=40 + TT=32 smoke collapsed exactly this way,
                    # silently). Carriers are workspace (gathered per step),
                    # not residency — keeping them 96-wide costs nothing in
                    # product terms and leaves the captured decode graph's
                    # full-width launch untouched.
                    if os.environ.get("R205_HPFIT", "0") == "1":
                        _Bcap = max(int(self._TT_MAX_SEQS), int(self._tt_mseq), _Bp)
                    else:
                        _Bcap = max(int(self._tt_mseq), _Bp)
                    _BcapT = _Bcap if _R146S else _Bcap + 1  # P1: no sentinel row
                    _nbC = _cap // _bs
                    _hoist147 = _os.environ.get("R147_HOIST", "0") == "1"
                    _carr = self._r145_carriers.get(("r146", _cap))
                    if _hoist147:
                        # R147 P1-B: carriers are CONTENT carriers (staged once per
                        # step, consumed sequentially by 36 same-stream layer replays)
                        # -- safe to share ONE per (cap, BcapT) across the per-layer
                        # backend instances. R146 allocated a full set PER INSTANCE
                        # (36x HBM: e.g. B16 3 buckets 68.1MB -> 2.45GB) and re-
                        # allocated on every BcapT change (capture churn). Keyed by
                        # (cap, BcapT) so shape changes get a fresh carrier and fresh
                        # graphs (graphs stay per-layer inside the shared dict).
                        _sh147 = getattr(type(self), "_r147_shared", None)
                        if _sh147 is None:
                            _sh147 = {}
                            type(self)._r147_shared = _sh147
                        _carr = _sh147.get(("r146", _cap, _BcapT))
                    if _carr is None:
                        _kv_cap = _BcapT * _nbC
                        _d4 = _hd // 4
                        _mseq_i = int(self._tt_mseq)
                        _btd1 = max(int(attn_metadata.block_tables.size(1)), _nbC)
                        _carr = {
                            "qrot": torch.zeros(_BcapT, _nq, 1, _hd, dtype=query.dtype, device=query.device),
                            "qsrc": torch.zeros(_BcapT, _nq, _hd, dtype=query.dtype, device=query.device),
                            "out": torch.zeros(_BcapT, _nq, 1, _hd, dtype=torch.bfloat16, device=query.device),
                            "lse": torch.zeros(_BcapT, _nq, 1, dtype=torch.float32, device=query.device),
                            "K": torch.zeros(_kv_cap, _bs, _nkv, _d4, dtype=torch.uint8, device=query.device),
                            "V": torch.zeros(_kv_cap, _bs, _nkv, _d4, dtype=torch.uint8, device=query.device),
                            "Kc": torch.zeros(_kv_cap, _nkv, _bs, _d4, dtype=torch.uint8, device=query.device),
                            "Vc": torch.zeros(_kv_cap, _nkv, _bs, _d4, dtype=torch.uint8, device=query.device),
                            "kS": torch.zeros(_BcapT, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                            "kO": torch.zeros(_BcapT, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                            "vS": torch.zeros(_BcapT, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                            "vO": torch.zeros(_BcapT, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                            "bt": torch.zeros(_BcapT, _nbC, dtype=torch.int32, device=query.device),
                            "asq": torch.ones(_BcapT, dtype=torch.int64, device=query.device),
                            "askv": torch.ones(_BcapT, dtype=torch.int64, device=query.device),
                            # R146 static-staging carriers (content-only dynamism)
                            "btStage": torch.zeros(max(_mseq_i, _BcapT), _btd1, dtype=torch.int64, device=query.device),
                            "btSel": torch.zeros(_BcapT, _btd1, dtype=torch.int64, device=query.device),
                            "offsC": torch.zeros(_BcapT, _nbC, dtype=torch.int64, device=query.device),
                            "arange": torch.arange(_nbC, dtype=torch.int64, device=query.device).view(1, _nbC),
                            "blklo": torch.zeros(_BcapT, dtype=torch.int64, device=query.device),
                            "bidx": torch.zeros(_BcapT, dtype=torch.int64, device=query.device),
                            "offs": torch.zeros(_BcapT, _nbC, dtype=torch.int64, device=query.device),
                            "IDX64": torch.zeros(_BcapT, _nbC, dtype=torch.int64, device=query.device),
                            "idxFlat": None,  # set below (view)
                            # alloc-free gather staging (shared across layer graphs; same-stream
                            # sequential replays make reuse safe)
                            "ksgS": torch.zeros(_kv_cap, _bs, _nkv, dtype=torch.bfloat16, device=query.device),
                            "kzgS": torch.zeros(_kv_cap, _bs, _nkv, dtype=torch.bfloat16, device=query.device),
                            "vsgS": torch.zeros(_kv_cap, _bs, _nkv, dtype=torch.bfloat16, device=query.device),
                            "vzgS": torch.zeros(_kv_cap, _bs, _nkv, dtype=torch.bfloat16, device=query.device),
                            "lensPin": torch.ones(_BcapT, dtype=torch.int64, pin_memory=True),
                            "blkloPin": torch.zeros(_BcapT, dtype=torch.int64, pin_memory=True),
                            "bidxPin": torch.zeros(_BcapT, dtype=torch.int64, pin_memory=True),
                            "graphs": {},
                        }
                        _carr["idxFlat"] = _carr["IDX64"].view(-1)
                        _carr["bt"].copy_(
                            torch.arange(_BcapT * _nbC, dtype=torch.int32, device=query.device).view(_BcapT, _nbC)
                        )
                        if not _R146S:
                            _carr["askv"][_BcapT - 1] = _cap  # sentinel row (R145 contract)
                        if _hoist147:
                            type(self)._r147_shared[("r146", _cap, _BcapT)] = _carr
                        self._r145_carriers[("r146", _cap)] = _carr  # instance alias
                        try:
                            # hoist: a newline inside f-string braces is a
                            # SyntaxError on Python <= 3.11
                            _mbC = sum(v.numel() * v.element_size() for v in _carr.values() if torch.is_tensor(v)) / 1e6
                            with open("/tmp/r146_probe.log", "a") as _pf:
                                _pf.write(
                                    f"[R146-CARR] cap={_cap} BcapT={_BcapT} nbC={_nbC} stride={_R146S} MB={_mbC:.1f}\n"
                                )
                        except Exception:
                            pass
                    # ---- per-STEP staging (eager; content-only updates) ----
                    # R147 P1-A: tier boundaries are layer-INVARIANT (probe-proven:
                    # per-layer backend instances advance their rcnt in lockstep, so
                    # _cb/_cbl/_cnb/_cml are identical for all 36 layer calls of a
                    # step; R146's 'ml shrinks 1/layer' premise refuted). Under
                    # R147_HOIST=1 the staging below (btStage copy + 3 pinned H2D +
                    # add/index_select/clamp/gather chain) runs ONCE per step and the
                    # 35 later layers reuse IDX64/askv/btStage/bidx/blklo verbatim.
                    _stKey147 = None
                    if _hoist147:
                        _stKey147 = (int(_cap), int(_BcapT), tuple(int(x) for x in seq_lens), tuple(_cb), tuple(_cml))
                        _st147: dict[str, Any] | None = getattr(type(self), "_r147_stage", None)
                        if _st147 is None:
                            _st147 = {}
                            type(self)._r147_stage = _st147
                        if _st147.get("key") == _stKey147:
                            _st147["hits"] = _st147.get("hits", 0) + 1
                    else:
                        _st147 = None
                    _btFw = attn_metadata.block_tables
                    _vmode = _os.environ.get("R146_VANILLA", "0")
                    _msx = min(int(_btFw.size(0)), int(_carr["btStage"].size(0)))
                    if _hoist147 and _st147 is not None and _st147.get("key") == _stKey147:
                        _vmode = "hoisted"  # skip to per-layer section; staging already ran
                    if _vmode != "2" and _vmode != "hoisted":  # mode 2: skip ALL staging device ops
                        _carr["btStage"][:_msx, :].copy_(_btFw[:_msx, :])
                        if _msx < int(_carr["btStage"].size(0)):
                            _carr["btStage"][_msx:].copy_(_carr["btStage"][0:1])
                    if (
                        _msx < int(_carr["btStage"].size(0)) and _vmode != "hoisted"
                    ):  # sentinel/padding rows mirror row 0
                        _carr["btStage"][_msx:].copy_(_carr["btStage"][0:1])
                    if _vmode != "hoisted":
                        for _i in range(_BcapT):
                            _r146v = _cml[_i] if _i < _Bp else (1 if _R146S else (1 if _i < _BcapT - 1 else _cap))
                            _carr["lensPin"][_i] = _r146v
                            _carr["blkloPin"][_i] = _cbl[_i] if _i < _Bp else 0
                            _carr["bidxPin"][_i] = _cb[_i] if _i < _Bp else 0
                    if _vmode != "5" and _vmode != "hoisted":  # mode 5: python-only staging (no H2D/add)
                        _carr["askv"].copy_(_carr["lensPin"])
                        _carr["blklo"].copy_(_carr["blkloPin"])
                        _carr["bidx"].copy_(_carr["bidxPin"])  # blocking (ordered) H2D
                        torch.add(_carr["blklo"].view(-1, 1), _carr["arange"], out=_carr["offs"])
                    # ROW-SELECT by bidx first (carrier row i -> framework slot bidx[i]), then
                    # gather dim1 with clamped offsets (blocks beyond a row's extent hold stale
                    # but VALID block ids; content beyond logical len is masked by the op).
                    if _vmode not in ("2", "3", "hoisted"):  # mode 3: skip gather-chain ops
                        torch.index_select(_carr["btStage"], 0, _carr["bidx"], out=_carr["btSel"])
                        torch.clamp(_carr["offs"], max=int(_carr["btStage"].size(1)) - 1, out=_carr["offsC"])
                        torch.gather(_carr["btSel"], 1, _carr["offsC"], out=_carr["IDX64"])
                        if _hoist147 and _st147 is not None:
                            _st147["key"] = _stKey147  # staging complete; later layers reuse
                    if _r197s:
                        try:
                            _r197t["t2"] = _t197.perf_counter()  # post staging
                            _r197t["ev2"] = torch.npu.Event(enable_timing=True)
                            _r197t["ev2"].record()
                            _r197t["vmode"] = _vmode
                        except Exception:
                            pass
                    # ---- per-LAYER: eager qsrc/qrot + graph(fill+op) ----
                    if _os.environ.get("R146_VANILLA", "0") == "1":
                        # staging present; middle = vanilla R145 math (per-row rot + C++ fill +
                        # op) into a vanilla-shaped carrier; NOTHING of my fill/op runs.
                        _carr3 = self._r145_carriers.get(_cap)
                        if _carr3 is None:
                            _kv3 = (_Bcap + 1) * _nbC
                            _d43 = _hd // 4
                            _carr3 = {
                                "qrot": torch.zeros(_Bcap + 1, _nq, 1, _hd, dtype=query.dtype, device=query.device),
                                "out": torch.zeros(_Bcap + 1, _nq, 1, _hd, dtype=torch.bfloat16, device=query.device),
                                "lse": torch.zeros(_Bcap + 1, _nq, 1, dtype=torch.float32, device=query.device),
                                "K": torch.zeros(_kv3, _bs, _nkv, _d43, dtype=torch.uint8, device=query.device),
                                "V": torch.zeros(_kv3, _bs, _nkv, _d43, dtype=torch.uint8, device=query.device),
                                "Kc": torch.zeros(_kv3, _nkv, _bs, _d43, dtype=torch.uint8, device=query.device),
                                "Vc": torch.zeros(_kv3, _nkv, _bs, _d43, dtype=torch.uint8, device=query.device),
                                "kS": torch.zeros(_Bcap + 1, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                                "kO": torch.zeros(_Bcap + 1, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                                "vS": torch.zeros(_Bcap + 1, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                                "vO": torch.zeros(_Bcap + 1, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                                "bt": torch.zeros(_Bcap + 1, _nbC, dtype=torch.int32, device=query.device),
                                "asq": torch.ones(_Bcap + 1, dtype=torch.int64, device=query.device),
                                "askv": torch.ones(_Bcap + 1, dtype=torch.int64, device=query.device),
                            }
                            _carr3["askv"][_Bcap] = _cap
                            _carr3["bt"].copy_(
                                torch.arange((_Bcap + 1) * _nbC, dtype=torch.int32, device=query.device).view(
                                    _Bcap + 1, _nbC
                                )
                            )
                            self._r145_carriers[_cap] = _carr3
                        for _ci, _bb in enumerate(_cb):
                            _qb = query[_bb : _bb + 1]
                            if _qb.dim() == 3:
                                _qb = _qb.unsqueeze(2)
                            if _Rk144 is not None:
                                torch.matmul(_qb, _RkMid144, out=_carr3["qrot"][_ci : _ci + 1])
                            else:
                                _carr3["qrot"][_ci : _ci + 1].copy_(_qb)
                        type(self)._r112_mod.oscar_middle_prep_v3_fill(
                            attn_metadata.block_tables,
                            _cb,
                            _cbl,
                            _cnb,
                            _cml,
                            self.key_cache,
                            self.value_cache,
                            self.k_scale_cache,
                            self.k_zero_cache,
                            self.v_scale_cache,
                            self.v_zero_cache,
                            _carr3["K"],
                            _carr3["V"],
                            _carr3["Kc"],
                            _carr3["Vc"],
                            _carr3["kS"],
                            _carr3["kO"],
                            _carr3["vS"],
                            _carr3["vO"],
                            _carr3["bt"],
                            _carr3["askv"],
                            _Bp,
                            _Bcap + 1,
                            _nbC,
                            _cap,
                            _nkv,
                            _bs,
                        )
                        # R189 PROBE: AFTER carrier fill, BEFORE middle LSE — is cube state still GREEN?
                        if (
                            os.environ.get("R189_PROBE") == "1"
                            and getattr(type(self), "_r189_mod", None) is not None
                            and not getattr(type(self), "_r189_fill_probed", False)
                        ):
                            type(self)._r189_fill_probed = True
                            _pd = _carr3["qrot"].device
                            _pD, _pnq, _pnkv, _pbs, _pnb = 128, 32, 8, 128, 17
                            _pS = _pnb * _pbs
                            _pq = (torch.randn(1, _pnq, 1, _pD, device=_pd) * 0.5).bfloat16()
                            _pk = torch.randint(
                                0, 256, (_pnb, _pnkv, _pbs, _pD // 4), device=_pd, dtype=torch.uint8
                            ).view(torch.int8)
                            _pv = torch.randint(
                                0, 256, (_pnb, _pnkv, _pbs, _pD // 4), device=_pd, dtype=torch.uint8
                            ).view(torch.int8)
                            _pbt = torch.arange(_pnb, dtype=torch.int32, device=_pd).unsqueeze(0)
                            _pks = (torch.rand(1, _pnkv, _pS, device=_pd) * 0.05 + 0.001).bfloat16()
                            _pko = (torch.randn(1, _pnkv, _pS, device=_pd) * 0.02).bfloat16()
                            _pvs = (torch.rand(1, _pnkv, _pS, device=_pd) * 0.05 + 0.001).bfloat16()
                            _pvo = (torch.randn(1, _pnkv, _pS, device=_pd) * 0.02).bfloat16()
                            _po = torch.zeros(1, _pnq, 1, _pD, dtype=torch.bfloat16, device=_pd)
                            _pl = torch.zeros(1, _pnq, 1, dtype=torch.float32, device=_pd)
                            try:
                                _pr = type(self)._r189_mod.launch_lse_fast(
                                    _pq,
                                    _pk,
                                    _pv,
                                    _pbt,
                                    _pks,
                                    _pko,
                                    _pvs,
                                    _pvo,
                                    _po,
                                    _pl,
                                    _pnq,
                                    1.0 / (_pD**0.5),
                                    _pnkv,
                                    "BNSD",
                                    _pbs,
                                    1,
                                    1,
                                    [_pS],
                                    torch.zeros(0, dtype=torch.bfloat16, device=_pd),
                                    0,
                                    0,
                                    0,
                                    True,
                                )
                                torch.npu.synchronize()
                                _odiag_w(
                                    "r190_probe.log",
                                    f"[R189-PROBE] AFTER-FILL-BEFORE-LSE rc={int(_pr[0])} "
                                    f"finite={torch.isfinite(_po).all().item()}\n",
                                )
                            except Exception as _pe:
                                _odiag_w(
                                    "r190_probe.log", f"[R189-PROBE] AFTER-FILL-BEFORE-LSE CRASH: {str(_pe)[:80]}\n"
                                )
                        _rr3v = type(self)._r112_mod.oscar_middle_op_v3(
                            _carr3["qrot"],
                            _carr3["Kc"],
                            _carr3["Vc"],
                            _carr3["bt"],
                            _carr3["kS"],
                            _carr3["kO"],
                            _carr3["vS"],
                            _carr3["vO"],
                            _carr3["out"],
                            _carr3["lse"],
                            _carr3["asq"],
                            _carr3["askv"],
                            _nq,
                            1.0 / _m144.sqrt(_hd),
                            _nkv,
                            _bs,
                            True,
                        )
                        # R215 GROUP-M in-vivo capture (R215_GMDUMP=1, default OFF,
                        # single-shot, MINB-gated): the V1CAP carriers of a REAL
                        # mixed-step mid call PRE-op, plus the product's own out/lse
                        # POST-op — the standalone GROUP-M replay ground truth.
                        if (
                            os.environ.get("R215_GMDUMP") == "1"
                            and not getattr(type(self), "_r215_gm_done", False)
                            and int(_Bp) >= int(os.environ.get("R215_GMDUMP_MINB", "4"))
                        ):
                            type(self)._r215_gm_done = True
                            try:
                                torch.npu.synchronize()
                                _od215 = os.environ.get("R215_GMDUMP_DIR", _odiag_p("r215_gmdump"))
                                os.makedirs(_od215, exist_ok=True)
                                _snap215 = {
                                    "q": _carr3["qrot"].detach().cpu().clone(),
                                    "k": _carr3["Kc"].detach().cpu().clone(),
                                    "v": _carr3["Vc"].detach().cpu().clone(),
                                    "bt": _carr3["bt"].detach().cpu().clone(),
                                    "ks": _carr3["kS"].detach().cpu().clone(),
                                    "ko": _carr3["kO"].detach().cpu().clone(),
                                    "vs": _carr3["vS"].detach().cpu().clone(),
                                    "vo": _carr3["vO"].detach().cpu().clone(),
                                    "out_h": _carr3["out"].detach().cpu().clone(),
                                    "lse_h": _carr3["lse"].detach().cpu().clone(),
                                    "askv": [int(x) for x in _carr3["askv"].cpu().tolist()],
                                    "nq": int(_nq),
                                    "scale": float(1.0 / _m144.sqrt(_hd)),
                                    "nkv": int(_nkv),
                                    "block_size": int(_bs),
                                    "Bp": int(_Bp),
                                    "cap": int(_cap),
                                    "nbC": int(_nbC),
                                    "layer_idx": int(layer_idx),
                                    "b_list": [int(x) for x in _cb],
                                }
                                _pc215 = f"{_od215}/r215_gm_l{layer_idx}_cap{_cap}.pt"
                                torch.save(_snap215, _pc215)
                                print(
                                    f"[R215-GMDUMP] l={layer_idx} Bp={_Bp} cap={_cap} "
                                    f"askv_valid={_snap215['askv'][:_Bp]} -> {_pc215}",
                                    flush=True,
                                )
                            except Exception as _e215:
                                print(f"[R215-GMDUMP] failed: {str(_e215)[:120]}", flush=True)
                        for _i, _bb in enumerate(_cb):
                            _r218_diag("MIDSITE-11279")
                            _r144_bmid[_bb] = (_carr3["out"][_i, :, 0, :].float(), _carr3["lse"][_i, :, 0].float())
                        _st["middle_calls"] += 1
                        _st["middle_ok"] += 1
                        _r145_handled = True
                        _r2, _lr2 = int(_rr3v[0]), int(_rr3v[2])
                    if not _r145_handled:
                        torch.index_select(query, 0, _carr["bidx"], out=_carr["qsrc"])
                    if _Rk144 is not None:
                        torch.matmul(_carr["qsrc"].unsqueeze(2), _RkMid144, out=_carr["qrot"])
                    else:
                        _carr["qrot"].copy_(_carr["qsrc"].unsqueeze(2))
                    if _r197s:
                        try:
                            _r197t["t3"] = _t197.perf_counter()  # post q gather+rot
                            _r197t["ev3"] = torch.npu.Event(enable_timing=True)
                            _r197t["ev3"].record()
                        except Exception:
                            pass
                    _kc = self.key_cache
                    _vc = self.value_cache
                    _kfl = _carr["K"].view(-1, _bs, _nkv, _hd // 4)
                    _vfl = _carr["V"].view(-1, _bs, _nkv, _hd // 4)
                    _kcfl = _carr["Kc"]
                    _vcfl = _carr["Vc"]
                    _ksC = _carr["kS"]
                    _koC = _carr["kO"]
                    _vsC = _carr["vS"]
                    _voC = _carr["vO"]
                    _idxF = _carr["idxFlat"]
                    _btw = _carr["K"].view(_BcapT, _nbC, _bs, _nkv, _hd // 4)
                    _vbw = _carr["V"].view(_BcapT, _nbC, _bs, _nkv, _hd // 4)
                    _ksw = _carr["kS"]
                    _kov = _carr["kO"]
                    # R197 Fix-2 (R197_EAGER_MIX=1, default OFF): eager fill width
                    # = the REAL row count Bp instead of BcapT — the carrier's
                    # padding rows (askv=1, compute-masked by the op) only need
                    # VALID block ids (bt is layout-constant), never fresh
                    # content, so gathering them is pure waste (BcapT=96 vs
                    # Bp=2..31 in mixed steps). Under capture the width MUST
                    # stay frozen at BcapT (replay re-executes whatever was
                    # captured; a Bp-frozen slice would under-fill later steps).
                    _BpF = _BcapT
                    if _os.environ.get("R197_EAGER_MIX", "0") == "1" and not _EXTRA_CTX.capturing:
                        _BpF = max(int(_Bp), 1)
                    _nF = _BpF * _nbC
                    _idxS = _idxF[:_nF]
                    _d4 = _hd // 4

                    def _r146_fill_layer():
                        # R204_MIXCPP (default OFF): the same ~15-op fill as ONE
                        # C++ crossing (oscar_fill_v3 in the r152 launcher —
                        # value-identical ATen port; ~15 Python dispatches ->
                        # 1 pybind call per layer).
                        if (
                            _os.environ.get("R204_MIXCPP", "0") == "1"
                            and getattr(type(self), "_r112_mod", None) is not None
                            and hasattr(type(self)._r112_mod, "oscar_fill_v3")
                            and not _EXTRA_CTX.capturing
                        ):
                            type(self)._r112_mod.oscar_fill_v3(
                                _kc,
                                _vc,
                                self.k_scale_cache,
                                self.k_zero_cache,
                                self.v_scale_cache,
                                self.v_zero_cache,
                                _idxS,
                                _kfl,
                                _vfl,
                                _kcfl,
                                _vcfl,
                                _carr["ksgS"],
                                _carr["kzgS"],
                                _carr["vsgS"],
                                _carr["vzgS"],
                                _carr["kS"],
                                _carr["kO"],
                                _carr["vS"],
                                _carr["vO"],
                                int(_BpF),
                                int(_nbC),
                                int(_bs),
                                int(_nkv),
                                int(_d4),
                            )
                            return
                        torch.index_select(_kc, 0, _idxS, out=_kfl[:_nF])
                        torch.index_select(_vc, 0, _idxS, out=_vfl[:_nF])
                        _kcfl[:_nF].copy_(_btw[:_BpF].permute(0, 1, 3, 2, 4).reshape(_nF, _nkv, _bs, _d4))
                        _vcfl[:_nF].copy_(_vbw[:_BpF].permute(0, 1, 3, 2, 4).reshape(_nF, _nkv, _bs, _d4))
                        # NOTE: uint8 permute-copy keeps packed INT2 bytes verbatim (no unpack).
                        torch.index_select(self.k_scale_cache, 0, _idxS, out=_carr["ksgS"][:_nF])
                        _carr["kS"][:_BpF].copy_(_carr["ksgS"][:_nF].view(_BpF, _nbC * _bs, _nkv).permute(0, 2, 1))
                        torch.index_select(self.k_zero_cache, 0, _idxS, out=_carr["kzgS"][:_nF])
                        _carr["kO"][:_BpF].copy_(_carr["kzgS"][:_nF].view(_BpF, _nbC * _bs, _nkv).permute(0, 2, 1))
                        _carr["kO"][:_BpF].mul_(_carr["kS"][:_BpF])
                        _carr["kO"][:_BpF].neg_()
                        torch.index_select(self.v_scale_cache, 0, _idxS, out=_carr["vsgS"][:_nF])
                        _carr["vS"][:_BpF].copy_(_carr["vsgS"][:_nF].view(_BpF, _nbC * _bs, _nkv).permute(0, 2, 1))
                        torch.index_select(self.v_zero_cache, 0, _idxS, out=_carr["vzgS"][:_nF])
                        _carr["vO"][:_BpF].copy_(_carr["vzgS"][:_nF].view(_BpF, _nbC * _bs, _nkv).permute(0, 2, 1))
                        _carr["vO"][:_BpF].mul_(_carr["vS"][:_BpF])
                        _carr["vO"][:_BpF].neg_()

                    def _r146_op(_dz):
                        # R189 PROBE: before the V3 middle LSE — is cube state GREEN or RED?
                        if (
                            os.environ.get("R189_PROBE") == "1"
                            and getattr(type(self), "_r189_mod", None) is not None
                            and not getattr(type(self), "_r189_v3_probed", False)
                        ):
                            type(self)._r189_v3_probed = True
                            _pd = _carr["qrot"].device
                            _pD, _pnq, _pnkv, _pbs, _pnb = 128, 32, 8, 128, 17
                            _pS = _pnb * _pbs
                            _pq = (torch.randn(1, _pnq, 1, _pD, device=_pd) * 0.5).bfloat16()
                            _pk = torch.randint(
                                0, 256, (_pnb, _pnkv, _pbs, _pD // 4), device=_pd, dtype=torch.uint8
                            ).view(torch.int8)
                            _pv = torch.randint(
                                0, 256, (_pnb, _pnkv, _pbs, _pD // 4), device=_pd, dtype=torch.uint8
                            ).view(torch.int8)
                            _pbt = torch.arange(_pnb, dtype=torch.int32, device=_pd).unsqueeze(0)
                            _pks = (torch.rand(1, _pnkv, _pS, device=_pd) * 0.05 + 0.001).bfloat16()
                            _pko = (torch.randn(1, _pnkv, _pS, device=_pd) * 0.02).bfloat16()
                            _pvs = (torch.rand(1, _pnkv, _pS, device=_pd) * 0.05 + 0.001).bfloat16()
                            _pvo = (torch.randn(1, _pnkv, _pS, device=_pd) * 0.02).bfloat16()
                            _po = torch.zeros(1, _pnq, 1, _pD, dtype=torch.bfloat16, device=_pd)
                            _pl = torch.zeros(1, _pnq, 1, dtype=torch.float32, device=_pd)
                            try:
                                _pr = type(self)._r189_mod.launch_lse_fast(
                                    _pq,
                                    _pk,
                                    _pv,
                                    _pbt,
                                    _pks,
                                    _pko,
                                    _pvs,
                                    _pvo,
                                    _po,
                                    _pl,
                                    _pnq,
                                    1.0 / (_pD**0.5),
                                    _pnkv,
                                    "BNSD",
                                    _pbs,
                                    1,
                                    1,
                                    [_pS],
                                    torch.zeros(0, dtype=torch.bfloat16, device=_pd),
                                    0,
                                    0,
                                    0,
                                    True,
                                )
                                torch.npu.synchronize()
                                _odiag_w(
                                    "r190_probe.log",
                                    f"[R189-PROBE] BEFORE-V3-LSE rc={int(_pr[0])} "
                                    f"finite={torch.isfinite(_po).all().item()}\n",
                                )
                            except Exception as _pe:
                                _odiag_w("r190_probe.log", f"[R189-PROBE] BEFORE-V3-LSE CRASH: {str(_pe)[:80]}\n")
                        # R201_MIXTRIM (default OFF): eager mixed steps launch
                        # the middle op on the LIVE rows only. The persistent
                        # carriers are BcapT-wide so the captured decode graph
                        # can launch full-width; the EAGER call only ever
                        # consumes rows [0, Bp) (downstream bmid reads), and
                        # every mixed step's decode subset is Bp = 2..31 rows
                        # while the op's device time is FLAT in the launch
                        # width (fixed grid/tiling => B=3 cost == B=31 cost,
                        # R201 probe dev 9.6-9.9ms/layer for B 3..31). Dim-0
                        # slices keep each row's math identical (rows are
                        # independent single-query attends); padding-row
                        # out/lse stay stale, read by nobody in the eager
                        # path (the captured graph recomputes its own rows).
                        # Capture keeps the full-width launch (replay executes
                        # whatever was captured).
                        # R201_T_NOMIDE (DIAGNOSTIC_ONLY/NEVER_PROMOTE/default
                        # OFF): skip the eager middle-op launch entirely
                        # (timing-only; out/lse stay at fill value -> outputs
                        # WRONG). Sizes the op's device share of the mixed
                        # step on the delivered config (R202 budget).
                        if _os.environ.get("R201_T_NOMIDE", "0") == "1" and not _EXTRA_CTX.capturing:
                            return (0, 0, 0)
                        _mtV201 = _os.environ.get("R201_MIXTRIM", "0")
                        _mt201 = _mtV201 in ("1", "2") and not _EXTRA_CTX.capturing
                        if _mt201 and int(_Bp) < int(_carr["qrot"].shape[0]):
                            if _mtV201 == "2":
                                # R201 adjudicator: sliced launch, snapshot,
                                # full launch, compare rows [0,Bp). Logs to
                                # /tmp/r201m_trimcmp.log. DIAGNOSTIC ONLY.
                                _nbC201 = int(_carr["bt"].shape[1])
                                _rT = type(self)._r112_mod.oscar_middle_op_v3(
                                    _carr["qrot"][:_Bp],
                                    _carr["Kc"][: _Bp * _nbC201],
                                    _carr["Vc"][: _Bp * _nbC201],
                                    _carr["bt"][:_Bp],
                                    _carr["kS"][:_Bp],
                                    _carr["kO"][:_Bp],
                                    _carr["vS"][:_Bp],
                                    _carr["vO"][:_Bp],
                                    _carr["out"][:_Bp],
                                    _carr["lse"][:_Bp],
                                    _carr["asq"][:_Bp],
                                    _carr["askv"][:_Bp],
                                    _nq,
                                    1.0 / _m144.sqrt(_hd),
                                    _nkv,
                                    _bs,
                                    _dz,
                                )
                                _oT = _carr["out"][:_Bp].clone()
                                _lT = _carr["lse"][:_Bp].clone()
                                _rF = type(self)._r112_mod.oscar_middle_op_v3(
                                    _carr["qrot"],
                                    _carr["Kc"],
                                    _carr["Vc"],
                                    _carr["bt"],
                                    _carr["kS"],
                                    _carr["kO"],
                                    _carr["vS"],
                                    _carr["vO"],
                                    _carr["out"],
                                    _carr["lse"],
                                    _carr["asq"],
                                    _carr["askv"],
                                    _nq,
                                    1.0 / _m144.sqrt(_hd),
                                    _nkv,
                                    _bs,
                                    _dz,
                                )
                                torch.npu.current_stream().synchronize()
                                _do201 = (_oT.float() - _carr["out"][:_Bp].float()).abs().max().item()
                                _dl201 = (_lT - _carr["lse"][:_Bp]).abs().max().item()
                                try:
                                    with open("/tmp/r201m_trimcmp.log", "a") as _f201c:
                                        _f201c.write(
                                            f"L={_lord146 if '_lord146' in dir() else '?'} Bp={_Bp} "
                                            f"nbC={_nbC201} cap={_cap} rcT={int(_rT[0])}/{int(_rT[2])} "
                                            f"rcF={int(_rF[0])}/{int(_rF[2])} "
                                            f"max|d_out|={_do201:.3e} max|dlse|={_dl201:.3e}\n"
                                        )
                                except Exception:
                                    pass
                                return _rF
                            _nbC201 = int(_carr["bt"].shape[1])
                            return type(self)._r112_mod.oscar_middle_op_v3(
                                _carr["qrot"][:_Bp],
                                _carr["Kc"][: _Bp * _nbC201],
                                _carr["Vc"][: _Bp * _nbC201],
                                _carr["bt"][:_Bp],
                                _carr["kS"][:_Bp],
                                _carr["kO"][:_Bp],
                                _carr["vS"][:_Bp],
                                _carr["vO"][:_Bp],
                                _carr["out"][:_Bp],
                                _carr["lse"][:_Bp],
                                _carr["asq"][:_Bp],
                                _carr["askv"][:_Bp],
                                _nq,
                                1.0 / _m144.sqrt(_hd),
                                _nkv,
                                _bs,
                                _dz,
                            )
                        # R202_APROBE (DIAGNOSTIC_ONLY/default OFF): time the
                        # full-width middle-op launch under controlled askv
                        # patterns to decide whether the FLAT ~7ms/layer cost
                        # is row-count-bound (dead rows cost full rows =>
                        # kernel dead-row skip / width fix pays) or key-count-
                        # bound (askv already bounds work => lever dead).
                        # Patterns: B = all-dead (askv=1 except sentinel row),
                        # C = single live row, then A = the REAL launch (last,
                        # so downstream reads correct values). Timed with npu
                        # events; askv restored before A. /tmp/r202_aprobe.log
                        if _os.environ.get("R202_APROBE", "0") == "1" and not _EXTRA_CTX.capturing and int(_Bp) >= 2:
                            try:
                                _apN = globals().setdefault("_R202_APROBE_N", [0])
                                _apN[0] += 1
                                _apElig = None
                                if _apN[0] <= 400:
                                    _apEl = int(_carr["askv"][0].item())
                                if _apEl is not None and _apEl >= 1000:
                                    _apC = _carr
                                    _apBw = int(_apC["askv"].shape[0])
                                    _apOrig = _apC["askv"].clone()
                                    _apCapV = int(_apOrig[_apBw - 1].item())
                                    _apLive0 = int(_apOrig[0].item())

                                    def _ap_launch():
                                        return type(self)._r112_mod.oscar_middle_op_v3(
                                            _apC["qrot"],
                                            _apC["Kc"],
                                            _apC["Vc"],
                                            _apC["bt"],
                                            _apC["kS"],
                                            _apC["kO"],
                                            _apC["vS"],
                                            _apC["vO"],
                                            _apC["out"],
                                            _apC["lse"],
                                            _apC["asq"],
                                            _apC["askv"],
                                            _nq,
                                            1.0 / _m144.sqrt(_hd),
                                            _nkv,
                                            _bs,
                                            _dz,
                                        )

                                    _apT = {}
                                    for _apPat in ("B", "C"):
                                        if _apPat == "B":
                                            _apC["askv"].fill_(1)
                                            _apC["askv"][_apBw - 1] = _apCapV
                                        else:
                                            _apC["askv"].fill_(1)
                                            _apC["askv"][0] = _apLive0
                                            _apC["askv"][_apBw - 1] = _apCapV
                                        _e0 = torch.npu.Event(enable_timing=True)
                                        _e1 = torch.npu.Event(enable_timing=True)
                                        torch.npu.current_stream().synchronize()
                                        _e0.record()
                                        _ap_launch()
                                        _e1.record()
                                        torch.npu.current_stream().synchronize()
                                        _apT[_apPat] = _e0.elapsed_time(_e1)
                                    _apC["askv"].copy_(_apOrig)
                                    _e0 = torch.npu.Event(enable_timing=True)
                                    _e1 = torch.npu.Event(enable_timing=True)
                                    torch.npu.current_stream().synchronize()
                                    _e0.record()
                                    _rA = _ap_launch()
                                    _e1.record()
                                    torch.npu.current_stream().synchronize()
                                    _apT["A"] = _e0.elapsed_time(_e1)
                                    with open("/tmp/r202_aprobe.log", "a") as _apF:
                                        _apF.write(
                                            f"L={(_lord146 if '_lord146' in dir() else '?')} Bp={int(_Bp)} "
                                            f"W={_apBw} live0={_apLive0} cap={_apCapV} "
                                            f"tA(real)={_apT['A']:.2f}ms tB(all-dead)={_apT['B']:.2f}ms "
                                            f"tC(1-live)={_apT['C']:.2f}ms rcA={int(_rA[0])}\n"
                                        )
                            except Exception:
                                pass
                        return type(self)._r112_mod.oscar_middle_op_v3(
                            _carr["qrot"],
                            _carr["Kc"],
                            _carr["Vc"],
                            _carr["bt"],
                            _carr["kS"],
                            _carr["kO"],
                            _carr["vS"],
                            _carr["vO"],
                            _carr["out"],
                            _carr["lse"],
                            _carr["asq"],
                            _carr["askv"],
                            _nq,
                            1.0 / _m144.sqrt(_hd),
                            _nkv,
                            _bs,
                            _dz,
                        )

                    # R146 debug bisect: R146_STAGE_ONLY=1 -> use THIS staging (IDX64/askv/
                    # btStage/bidx/blklo) but run R145's proven per-row C++ fill + eager op.
                    if _os.environ.get("R146_STAGE_ONLY", "0") == "1":
                        _r3dbg = type(self)._r112_mod.oscar_middle_prep_v3_fill(
                            attn_metadata.block_tables,
                            _cb,
                            _cbl,
                            _cnb,
                            _cml,
                            _kc,
                            _vc,
                            self.k_scale_cache,
                            self.k_zero_cache,
                            self.v_scale_cache,
                            self.v_zero_cache,
                            _carr["K"],
                            _carr["V"],
                            _carr["Kc"],
                            _carr["Vc"],
                            _carr["kS"],
                            _carr["kO"],
                            _carr["vS"],
                            _carr["vO"],
                            _carr["bt"],
                            _carr["askv"],
                            _Bp,
                            _BcapT,
                            _nbC,
                            _cap,
                            _nkv,
                            _bs,
                        )
                        _rrdbg = _r146_op(True)
                        _r2, _lr2 = int(_rrdbg[0]), int(_rrdbg[2])
                        if not (_r2 == 0 and _lr2 == 0):
                            raise RuntimeError(f"[R146-DBg] rc={_r2} lr={_lr2}")
                        for _i, _bb in enumerate(_cb):
                            _r218_diag("MIDSITE-11550")
                            _r144_bmid[_bb] = (_carr["out"][_i, :, 0, :].float(), _carr["lse"][_i, :, 0].float())
                        _st["middle_calls"] += 1
                        _st["middle_ok"] += 1
                        _r145_handled = True
                        # fall through to skip normal path this layer
                    if not _r145_handled and _R145G and False:  # noqa: SIM223 - parked arm, deliberately unreachable
                        pass
                    # R146_CMP=1: run BOTH paths this layer and diff out/lse per row.
                    if _os.environ.get("R146_CMP", "0") == "1" and not _r145_handled:
                        _carr2 = self._r145_carriers.get(_cap)
                        if _carr2 is None:
                            _kv_cap2 = (_Bcap + 1) * _nbC
                            _d4b = _hd // 4
                            _carr2 = {
                                "qrot": torch.zeros(_Bcap + 1, _nq, 1, _hd, dtype=query.dtype, device=query.device),
                                "out": torch.zeros(_Bcap + 1, _nq, 1, _hd, dtype=torch.bfloat16, device=query.device),
                                "lse": torch.zeros(_Bcap + 1, _nq, 1, dtype=torch.float32, device=query.device),
                                "K": torch.zeros(_kv_cap2, _bs, _nkv, _d4b, dtype=torch.uint8, device=query.device),
                                "V": torch.zeros(_kv_cap2, _bs, _nkv, _d4b, dtype=torch.uint8, device=query.device),
                                "Kc": torch.zeros(_kv_cap2, _nkv, _bs, _d4b, dtype=torch.uint8, device=query.device),
                                "Vc": torch.zeros(_kv_cap2, _nkv, _bs, _d4b, dtype=torch.uint8, device=query.device),
                                "kS": torch.zeros(_Bcap + 1, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                                "kO": torch.zeros(_Bcap + 1, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                                "vS": torch.zeros(_Bcap + 1, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                                "vO": torch.zeros(_Bcap + 1, _nkv, _cap, dtype=torch.bfloat16, device=query.device),
                                "bt": torch.zeros(_Bcap + 1, _nbC, dtype=torch.int32, device=query.device),
                                "asq": torch.ones(_Bcap + 1, dtype=torch.int64, device=query.device),
                                "askv": torch.ones(_Bcap + 1, dtype=torch.int64, device=query.device),
                                "graph": None,
                            }
                            _carr2["askv"][_Bcap] = _cap
                            _carr2["bt"].copy_(
                                torch.arange((_Bcap + 1) * _nbC, dtype=torch.int32, device=query.device).view(
                                    _Bcap + 1, _nbC
                                )
                            )
                            self._r145_carriers[_cap] = _carr2
                        for _ci, _bb in enumerate(_cb):
                            _qb = query[_bb : _bb + 1]
                            if _qb.dim() == 3:
                                _qb = _qb.unsqueeze(2)
                            if _Rk144 is not None:
                                torch.matmul(_qb, _RkMid144, out=_carr2["qrot"][_ci : _ci + 1])
                            else:
                                _carr2["qrot"][_ci : _ci + 1].copy_(_qb)
                        type(self)._r112_mod.oscar_middle_prep_v3_fill(
                            attn_metadata.block_tables,
                            _cb,
                            _cbl,
                            _cnb,
                            _cml,
                            self.key_cache,
                            self.value_cache,
                            self.k_scale_cache,
                            self.k_zero_cache,
                            self.v_scale_cache,
                            self.v_zero_cache,
                            _carr2["K"],
                            _carr2["V"],
                            _carr2["Kc"],
                            _carr2["Vc"],
                            _carr2["kS"],
                            _carr2["kO"],
                            _carr2["vS"],
                            _carr2["vO"],
                            _carr2["bt"],
                            _carr2["askv"],
                            _Bp,
                            _Bcap + 1,
                            _nbC,
                            _cap,
                            _nkv,
                            _bs,
                        )
                        _rr2 = type(self)._r112_mod.oscar_middle_op_v3(
                            _carr2["qrot"],
                            _carr2["Kc"],
                            _carr2["Vc"],
                            _carr2["bt"],
                            _carr2["kS"],
                            _carr2["kO"],
                            _carr2["vS"],
                            _carr2["vO"],
                            _carr2["out"],
                            _carr2["lse"],
                            _carr2["asq"],
                            _carr2["askv"],
                            _nq,
                            1.0 / _m144.sqrt(_hd),
                            _nkv,
                            _bs,
                            True,
                        )
                        # mine (no graph; eager, same step)
                        _r146_fill_layer()
                        _rrm = _r146_op(True)
                        # input diffs
                        _qi = (_carr["qrot"][:_Bp].float() - _carr2["qrot"][:_Bp].float()).abs().max().item()
                        _ki = (_carr["Kc"][: _Bp * _nbC].int() - _carr2["Kc"][: _Bp * _nbC].int()).abs().max().item()
                        _si = (_carr["kS"][:_Bp].float() - _carr2["kS"][:_Bp].float()).abs().max().item()
                        _oi = (_carr["kO"][:_Bp].float() - _carr2["kO"][:_Bp].float()).abs().max().item()
                        _ai = int((_carr["askv"][:_Bp].int() - _carr2["askv"][:_Bp].int()).abs().max().item())
                        _do = (_carr["out"][:_Bp].float() - _carr2["out"][:_Bp].float()).abs().max().item()
                        _dl = (_carr["lse"][:_Bp].float() - _carr2["lse"][:_Bp].float()).abs().max().item()
                        try:
                            with open("/tmp/r146_cmp.log", "a") as _pf:
                                _pf.write(
                                    f"L={layer_idx} B={batch_size} Bp={_Bp} cap={_cap} ml={_cml} "
                                    f"qrot={_qi} Kc={_ki} kS={_si} kO={_oi} askv={_ai} "
                                    f"OUTdiff={_do} LSEdiff={_dl} rc2={int(_rr2[0])} rcm={int(_rrm[0])}\n"
                                )
                        except Exception:
                            pass
                        for _i, _bb in enumerate(_cb):
                            _r218_diag("MIDSITE-11626")
                            _r144_bmid[_bb] = (_carr2["out"][_i, :, 0, :].float(), _carr2["lse"][_i, :, 0].float())
                        _st["middle_calls"] += 1
                        _st["middle_ok"] += 1
                        _r145_handled = True
                    _lord146 = self._oscar_layer_ord(layer)  # layer_idx attr is MISSING on
                    # this layer object (defaults 0 for all 36) — MUST key graphs by the real
                    # ordinal: each graph captures ITS layer's key/value/scale-cache addresses.
                    _g146 = _carr["graphs"].get(_lord146)
                    _r2, _lr2 = 0, 0
                    if not _r145_handled and _R145G and _g146 is not None:
                        _g146.replay()
                        _r2, _lr2 = 0, 0
                        if _os.environ.get("R146_CMPG", "0") == "1":
                            # per-step replay-vs-eager: compare against a fresh eager recompute
                            _or146 = _carr["out"][:_Bp].clone()
                            _lr146 = _carr["lse"][:_Bp].clone()
                            _r146_fill_layer()
                            _r146_op(True)
                            _do = (_carr["out"][:_Bp].float() - _or146.float()).abs().max().item()
                            _dl = (_carr["lse"][:_Bp].float() - _lr146.float()).abs().max().item()
                            if _do != 0.0 or _dl != 0.0:
                                try:
                                    with open("/tmp/r146_cmpg2.log", "a") as _pf:
                                        _pf.write(
                                            f"L={_lord146} Bp={_Bp} cap={_cap} ml={_cml} OUTdiff={_do} LSEdiff={_dl}\n"
                                        )
                                except Exception:
                                    pass
                            _carr["out"][:_Bp].copy_(_or146)
                            _carr["lse"][:_Bp].copy_(_lr146)
                    elif not _r145_handled:
                        _r146_fill_layer()
                        if _os.environ.get("R149_LSE_ASYNC", "0") == "1":
                            # R149 P1 ASYNC CONTRACT: BARE dz=False call — zero syncs,
                            # zero adjacent fills (carriers were initialized once at
                            # creation; R147_LSE_GUARD below still fail-louds on drops).
                            _rr146 = _r146_op(False)
                        elif _os.environ.get("R148_LSE_FIX", "0") == "1":
                            # R148 P0 FIX: the eager dz=True pattern (zero_/fill_ of the
                            # out/lse carriers inside the launcher, immediately followed by
                            # GetWorkspaceSize+launch in the SAME extension call) drops the
                            # LSE write (deterministic on the first such call per process,
                            # ~1-3% after; rc=0, OUT bit-exact). Contract proven in
                            # R148/RAW/p0: issue the fills as separate ATen ops, let them
                            # COMPLETE (stream sync), then call the op with dz=False.
                            # One stream sync per EAGER first-use call only (rare, bucket
                            # churn); graph replays and steady decode are untouched.
                            _carr["out"].zero_()
                            _carr["lse"].fill_(-9999.0)
                            torch.npu.current_stream().synchronize()
                            _rr146 = _r146_op(False)
                        else:
                            _rr146 = _r146_op(True)
                        # R196 P1 (default OFF): the rc int()s and the LSE-guard
                        # .item() below are full stream-drain syncs — 3 per
                        # layer-call serialized the eager middle (mixed steps).
                        # Under R196_EAGER_BATCH they run on layer ordinal 0 of
                        # every step plus each carrier bucket's FIRST eager use
                        # (the flake class the guard exists for); other layers
                        # skip the host syncs (rc assumed 0; a layer-0 failure
                        # fail-closes the step exactly as before).
                        _r196chk = True
                        if _os.environ.get("R196_EAGER_BATCH", "0") == "1":
                            if _carr.get("_r196_seen") is None:
                                _carr["_r196_seen"] = True
                            elif int(self._oscar_layer_ord(layer)) != 0:
                                _r196chk = False
                        if _r196chk:
                            _r2, _lr2 = int(_rr146[0]), int(_rr146[2])
                        else:
                            _r2, _lr2 = 0, 0
                        # R147 P0-A guard: flaky-artifact class = eager dz=True call leaves
                        # LSE unwritten at fill value (-9999) with rc=0 (~1% under a
                        # replay-then-eager interleave; serving first-use eager runs
                        # BEFORE any replay so unexposed). Fail loudly ONCE per first
                        # use instead of feeding -9999 into the tier merge.
                        if _r196chk and _r2 == 0 and _os.environ.get("R147_LSE_GUARD", "1") == "1":
                            _mn147 = float(_carr["lse"][:_Bp].min().item())
                            if _mn147 < -9000.0:
                                raise RuntimeError(
                                    f"[R147-LSE-GUARD] L={_lord146} Bp={_Bp} cap={_cap} "
                                    f"eager first-use left LSE at fill value (min={_mn147}) "
                                    f"— known flaky eager-side non-write; retry step"
                                )
                        if _os.environ.get("R146_CMPG", "0") == "1":  # R196: clones feed CMPG probes only
                            _oe146 = _carr["out"][:_Bp].clone()
                            _le146 = _carr["lse"][:_Bp].clone()
                        if _os.environ.get("R146_CMPG", "0") == "1":
                            _carr["graphs"].pop(_lord146, None)  # force fresh capture diag
                        if _R145G and _g146 is None:
                            _s146 = torch.npu.Stream()
                            _s146.wait_stream(torch.npu.current_stream())
                            with torch.npu.stream(_s146):
                                for _ in range(2):
                                    _r146_fill_layer()
                                    _r146_op(False)  # R146 rule: NO zero/fill INSIDE capture (LSE drop)
                            torch.npu.current_stream().wait_stream(_s146)
                            torch.npu.synchronize()
                            _g146n = torch.npu.NPUGraph()
                            with torch.npu.graph(_g146n, stream=_s146):
                                _r146_fill_layer()
                                _rc146 = _r146_op(False)
                            torch.npu.synchronize()
                            if int(_rc146[0]) != 0:
                                raise RuntimeError(f"[R146] graph capture rc={int(_rc146[0])}")
                            _carr["graphs"][_lord146] = _g146n
                            _g146n.replay()  # produce this step's output (capture didn't execute)
                            _r2, _lr2 = 0, 0
                            if _os.environ.get("R146_CMPG", "0") == "1":
                                _do = (_carr["out"][:_Bp].float() - _oe146.float()).abs().max().item()
                                _dl = (_carr["lse"][:_Bp].float() - _le146.float()).abs().max().item()
                                try:
                                    with open("/tmp/r146_cmpg.log", "a") as _pf:
                                        _pf.write(
                                            f"L={layer_idx} Bp={_Bp} cap={_cap} ml={_cml} OUTdiff={_do} LSEdiff={_dl}\n"
                                        )
                                except Exception:
                                    pass
                                _carr["out"][:_Bp].copy_(_oe146)
                                _carr["lse"][:_Bp].copy_(_le146)
                    if not (_r2 == 0 and _lr2 == 0):
                        raise RuntimeError(f"[R146] middle NOT ok rc={_r2} lr={_lr2}")
                    if _r197s:
                        try:
                            _r197t["t4"] = _t197.perf_counter()  # post fill+op(+rc)
                            _r197t["ev4"] = torch.npu.Event(enable_timing=True)
                            _r197t["ev4"].record()
                            _r197t.update(
                                cap=int(_cap),
                                nbC=int(_nbC),
                                BcapT=int(_BcapT),
                                Bp=len(_cb),
                                cml=list(_cml)[:12],
                                cml_sum=int(sum(_cml)),
                                npr=int(
                                    max(
                                        (
                                            self._tt_pref.get(self._tt_rid(x), 0)
                                            + self._tt_rcnt.get(self._tt_rid(x), 0)
                                            for x in _cb
                                        ),
                                        default=0,
                                    ),
                                ),
                            )
                        except Exception:
                            pass
                    if _os.environ.get("R146_DUMP", "0") == "1":
                        _cs = float(_carr["out"][:_Bp].float().abs().sum().item())
                        _cl = float(_carr["lse"][:_Bp].float().abs().sum().item())
                        try:
                            with open("/tmp/r146_dump.log", "a") as _pf:
                                _pf.write(
                                    f"L={_lord146} Bp={_Bp} cap={_cap} ml={_cml} "
                                    f"outsum={_cs:.4f} lsesum={_cl:.4f} "
                                    f"bidx={_cb} lens={_carr['askv'][:_Bp].int().tolist()}\n"
                                )
                        except Exception:
                            pass
                    # R216: in-vivo capture at the REAL decode convergence point
                    # (fires for BOTH graph-replay and eager mid calls). Default OFF.
                    if _os.environ.get("R216_LSEDUMP", "0") == "1":
                        _r216_mid(
                            _carr,
                            _Bp,
                            _cb,
                            _cml,
                            _cap,
                            _nbC,
                            _lord146,
                            _nq,
                            _nkv,
                            _bs,
                            _hd,
                            1.0 / _m144.sqrt(_hd),
                            bool((not _r145_handled) and (_g146 is not None)),
                            lmod=getattr(type(self), "_r112_mod", None),
                        )
                    if _r144_ovl_os.environ.get("R204_MIXCHAIN", "0") == "1":
                        # R204: ONE slab cast per layer replaces the 2-per-row
                        # bf16->fp32 cast kernels (2B kernels/layer); the dict
                        # values become free VIEWS into the slab (read-only
                        # consumers). The slab is also handed to the batched
                        # S+R merge (skips the 2B o_h/l_h copy_ staging).
                        _osl204 = _carr["out"][: len(_cb), :, 0, :].float()
                        _lsl204 = _carr["lse"][: len(_cb), :, 0].float()
                        for _i, _bb in enumerate(_cb):
                            _r218_diag("MIDSITE-11781")
                            _r144_bmid[_bb] = (_osl204[_i], _lsl204[_i])
                        _slab204 = (_osl204, _lsl204, tuple(_cb))
                    else:
                        for _i, _bb in enumerate(_cb):
                            _r218_diag("MIDSITE-11785")
                            _r144_bmid[_bb] = (_carr["out"][_i, :, 0, :].float(), _carr["lse"][_i, :, 0].float())
                    _st["middle_calls"] += 1
                    _st["middle_ok"] += 1
                    _r145_handled = True
            for _bb in range(batch_size):
                if _R145T:
                    continue
                _sl = int(seq_lens[_bb])
                _pr = self._tt_pref.get(self._tt_rid(_bb), 0)
                _rc = self._tt_rcnt.get(self._tt_rid(_bb), 0)
                _ms = _pr
                _me = _sl - _rc
                _ml = int(_me - _ms)
                # R144 DIAG (file): log every row's boundary decision (first decode step)
                _r144gs_n = globals().setdefault("_R144_GS_N", [0])
                if _r144gs_n[0] < (batch_size * 2):
                    _r144gs_n[0] += 1
                    try:
                        with open("/tmp/r144_gate_probe.log", "a") as _pf:
                            _pf.write(
                                f"[R144-GATE] L={layer_idx} bb={_bb} sl={_sl} pref={_pr} "
                                f"rcnt={_rc} ms={_ms} me={_me} ml={_ml} bs={_bs} "
                                f"ms%bs={_ms % _bs} skip={(_ml <= 0 or (_ms % _bs) != 0)}\n"
                            )
                    except Exception:
                        pass
                if _ml <= 0 or (_ms % _bs) != 0:
                    continue
                _blk_lo = _ms // _bs
                _blk_hi = (_me + _bs - 1) // _bs
                _nb = _blk_hi - _blk_lo
                if _nb <= 0:
                    continue
                _qb = query[_bb : _bb + 1]
                if _qb.dim() == 3:
                    _qb = _qb.unsqueeze(2)
                if _V2:
                    # write rotated Q directly into the persistent carrier (no alloc)
                    _ci = len(_bidx)
                    if _Rk144 is not None:
                        torch.matmul(_qb, _RkMid144, out=_carr["qrot"][_ci : _ci + 1])
                    else:
                        _carr["qrot"][_ci : _ci + 1].copy_(_qb)
                else:
                    if _Rk144 is not None:
                        _qb = torch.matmul(_qb, _RkMid144)
                    _q_l.append(_qb)
                _bidx.append(int(_bb))
                _bl.append(int(_blk_lo))
                _nb_l.append(int(_nb))
                _ml_l.append(int(_ml))
            if layer_idx == 0 and _os.environ.get("R197_DBG", "0") == "1":  # R197 Fix-0
                print(
                    f"[R144-COLLECT] layer0 collected {len(_bidx)} reqs bidx={_bidx} "
                    f"blk_lo={_bl} nb={_nb_l} mid_len={_ml_l} V2={_V2}",
                    file=_sys.stderr,
                    flush=True,
                )
            if len(_bidx) >= 1:
                if _V2:
                    _Bp = len(_bidx)
                    _q_stack = _carr["qrot"][:_Bp]
                    _out_s = _carr["out"][:_Bp]
                    _lse_s = _carr["lse"][:_Bp]
                    # R144 P1 sha1 q-shadow: build V1 q (matmul alloc + cat) from
                    # the SAME query rows, compare to V2 carrier q byte-by-byte.
                    if _os.environ.get("R144_V2_SHADOW", "0") == "1":
                        type(self)._r144_qshadow_n = getattr(type(self), "_r144_qshadow_n", 0) + 1
                        if getattr(type(self), "_r144_qshadow_n", 0) <= 8:
                            try:
                                _qv1_l = []
                                for _ci2, _bb2 in enumerate(_bidx):
                                    _qb1 = query[_bb2 : _bb2 + 1]
                                    if _qb1.dim() == 3:
                                        _qb1 = _qb1.unsqueeze(2)
                                    if _Rk144 is not None:
                                        _qb1 = torch.matmul(_qb1, _RkMid144)
                                    _qv1_l.append(_qb1)
                                _qv1 = torch.cat(_qv1_l, 0).contiguous()
                                _qv2 = _q_stack.contiguous()
                                _c1 = _qv1.cpu().contiguous()
                                _c2 = _qv2.cpu().contiguous()
                                import hashlib as _hl2

                                _h1 = _hl2.sha256(_c1.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
                                _h2 = _hl2.sha256(_c2.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
                                _fd = -1
                                _p1 = _c1.view(-1).to(torch.float32)
                                _p2 = _c2.view(-1).to(torch.float32)
                                for _k in range(_p1.numel()):
                                    if _p1[_k].item() != _p2[_k].item():
                                        _fd = _k
                                        break
                                with open("/tmp/r144_v2_qshadow.log", "a") as _qf:
                                    _qf.write(
                                        f"[QSHADOW L={layer_idx} Bp={_Bp} v1={list(_qv1.shape)} "
                                        f"v2={list(_qv2.shape)} h1={_h1} h2={_h2} match={_h1 == _h2} "
                                        f"FIRST_DIFF_ELEM={_fd}\n"
                                    )
                            except Exception as _qe:
                                try:
                                    with open("/tmp/r144_v2_qshadow.log", "a") as _qf:
                                        _qf.write(f"[QSHADOW err] {type(_qe).__name__}: {_qe}\n")
                                except Exception:
                                    pass
                else:
                    _q_stack = torch.cat(_q_l, 0).contiguous()  # [B', nq, 1, hd]
                    _Bp = _q_stack.size(0)
                    _out_s = torch.zeros(_Bp, _nq, 1, _hd, dtype=torch.bfloat16, device=query.device)
                    _lse_s = torch.zeros(_Bp, _nq, 1, dtype=torch.float32, device=query.device)
                if _V2:
                    try:
                        _r2, _ws2, _lr2, _ = type(self)._r112_mod.oscar_middle_prep_batched_v2(
                            attn_metadata.block_tables,
                            _bidx,
                            _bl,
                            _nb_l,
                            _ml_l,
                            self.key_cache,
                            self.value_cache,
                            self.k_scale_cache,
                            self.k_zero_cache,
                            self.v_scale_cache,
                            self.v_zero_cache,
                            _q_stack,
                            _out_s,
                            _lse_s,
                            _carr["K"],
                            _carr["V"],
                            _carr["Kc"],
                            _carr["Vc"],
                            _carr["kS"],
                            _carr["kO"],
                            _carr["vS"],
                            _carr["vO"],
                            _carr["bt"],
                            _nq,
                            1.0 / _m144.sqrt(_hd),
                            _nkv,
                            _bs,
                        )
                    except Exception as _e:
                        try:
                            import traceback as _tb

                            with open("/tmp/r144_v2_err.log", "a") as _ef:
                                _ef.write(
                                    f"[V2-ERR] L={layer_idx} Bp={len(_bidx)} "
                                    f"bidx={_bidx} bl={_bl} nb={_nb_l} ml={_ml_l}\n"
                                    f"{type(_e).__name__}: {_e}\n"
                                    f"{''.join(_tb.format_exception(_e))}\n"
                                )
                        except Exception:
                            pass
                        raise
                else:
                    _r2, _ws2, _lr2, _ = type(self)._r112_mod.oscar_middle_prep_batched(
                        attn_metadata.block_tables,
                        _bidx,
                        _bl,
                        _nb_l,
                        _ml_l,
                        self.key_cache,
                        self.value_cache,
                        self.k_scale_cache,
                        self.k_zero_cache,
                        self.v_scale_cache,
                        self.v_zero_cache,
                        _q_stack,
                        _out_s,
                        _lse_s,
                        _nq,
                        1.0 / _m144.sqrt(_hd),
                        _nkv,
                        _bs,
                    )
                if not (int(_r2) == 0 and int(_lr2) == 0):
                    try:
                        with open("/tmp/r144_v2_err.log", "a") as _ef:
                            _ef.write(
                                f"[V2-RC] L={layer_idx} Bp={len(_bidx)} rc={_r2} lr={_lr2} "
                                f"bidx={_bidx} bl={_bl} nb={_nb_l} ml={_ml_l}\n"
                            )
                    except Exception:
                        pass
                    raise RuntimeError(f"[R144-BATCH] middle_prep_batched NOT ok rc={_r2} lr={_lr2}")
                for _i, _bb in enumerate(_bidx):
                    _r218_diag("MIDSITE-11922")
                    _r144_bmid[_bb] = (_out_s[_i, :, 0, :].float(), _lse_s[_i, :, 0].float())
                _st["middle_calls"] += 1  # ONE call for all requests (was batch_size)
                _st["middle_ok"] += 1
                # R144 DIAG (file-based): batched call result + per-row blk_lo/nb/mid_len
                try:
                    with open("/tmp/r144_batch_probe.log", "a") as _pf:
                        _pf.write(
                            f"[R144-BATCH] L={layer_idx} B={batch_size} Bp={_Bp} "
                            f"rc={_r2} lr={_lr2} bidx={_bidx} blk_lo={_bl} "
                            f"nb={_nb_l} mid_len={_ml_l} bs_cfg={_bs}\n"
                        )
                except Exception:
                    pass

        # ---- R147 P5 thin hook: batched S+R merge via OscarGraphAdapter ----
        if (
            _os.environ.get("R147_SR_BATCH", "0") == "1"
            and _r145_handled
            and _R145T
            and len(_r144_bmid) > 0
            and _os.environ.get("R147_ADAPTER", "0") == "1"
        ):
            from vllm_ascend.attention.oscar_graph_adapter import OscarGraphAdapter as _OGA

            _adp = getattr(self, "_r147_adapter", None)
            if _adp is None:
                _adp = _OGA(self)
                self._r147_adapter = _adp
            _Rv147h = None
            if self._oscar_R_v is not None and self._oscar_layer_ord(layer) < self._oscar_R_v.shape[0]:
                _Rv147h = self._oscar_R_v[self._oscar_layer_ord(layer)]
            _srB147 = _adp.sr_merge(
                layer=layer,
                layer_ord=self._oscar_layer_ord(layer),
                query=query,
                seq_lens=seq_lens,
                output=output,
                batch_size=batch_size,
                r144_bmid=_r144_bmid,
                carr=_carr,
                cb=_cb,
                cml=_cml,
                stats=_st,
                Rk=_Rk144,
                Rv=_Rv147h,
                handled=_r145_handled,
            )
        elif (
            _os.environ.get("R147_SR_BATCH", "0") == "1"
            and _r145_handled
            and _R145T
            and len(_r144_bmid) > 0
            and _os.environ.get("R147_ADAPTER", "0") != "1"
        ):
            # (R147_SR_BATCH=1, default OFF; requires the R146 fill/carrier path active so
            #  middle results live in the shared carrier). Per layer: ONE padded FIA for
            #  all rows + per-row l_pr einsum (legacy op sequence verbatim, values read
            #  from the padded S+R carrier slices) + ONE vectorized merge in carrier
            #  space + one unrot matmul + one index_copy writeback (~18*B tiny ops/layer
            #  -> ~6+3B). Per-element math order identical to the legacy per-row formula;
            #  absent tiers selected by exact torch.where (no arithmetic on absent
            #  branch); FIA stays eager (host actual_seq_lengths). R147_SR_CMP=1
            #  recomputes the legacy per-row path per layer and logs max|diff|.
            _srB147 = _os.environ.get("R147_SR_BATCH", "0") == "1" and _r145_handled and _R145T and len(_r144_bmid) > 0
            if _srB147:
                _Rv147 = None
                if self._oscar_R_v is not None and self._oscar_layer_ord(layer) < self._oscar_R_v.shape[0]:
                    _Rv147 = self._oscar_R_v[self._oscar_layer_ord(layer)]
                _nqS = self.num_heads
                _nkvS = self.num_kv_heads
                _hdS = self.head_size
                _prefC = int(self._TT_PREFIX)
                _capS = int(self._TT_RECENT)
                _CPR = _prefC + _capS  # static S+R width bound
                _lordS = self._oscar_layer_ord(layer)
                _BpS = len(_cb)
                _BT = max(int(_carr["out"].size(0)), batch_size)
                _hkS, _hvS = self._tt_hp_bufs(_lordS, query.device, query.dtype)
                _shS = getattr(type(self), "_r147_shared", None)
                if _shS is None:
                    _shS = {}
                    type(self)._r147_shared = _shS
                _src = _shS.get(("sr", _BT, _CPR))
                if _src is None:
                    _src = {
                        "prk": torch.zeros(_BT, _CPR, _nkvS, _hdS, dtype=query.dtype, device=query.device),
                        "prv": torch.zeros(_BT, _CPR, _nkvS, _hdS, dtype=query.dtype, device=query.device),
                        "qrot": torch.zeros(_BT, _nqS, _hdS, dtype=query.dtype, device=query.device),
                        "opr": torch.zeros(_BT, _nqS, _hdS, dtype=torch.float32, device=query.device),
                        "lpr": torch.full((_BT, _nqS), float("-inf"), dtype=torch.float32, device=query.device),
                        "lh": torch.full((_BT, _nqS), float("-inf"), dtype=torch.float32, device=query.device),
                        "oh": torch.zeros(_BT, _nqS, _hdS, dtype=torch.float32, device=query.device),
                        "prlens": torch.ones(_BT, dtype=torch.int64, device=query.device),
                        "hOKd": torch.zeros(_BT, dtype=torch.int64, device=query.device),
                        "merged": torch.zeros(_BT, _nqS, _hdS, dtype=torch.float32, device=query.device),
                        "allidx": torch.arange(_BT, dtype=torch.int64, device=query.device),
                        "outsel": torch.zeros(_BT, _nqS, _hdS, dtype=query.dtype, device=query.device),
                        "rvT": torch.zeros(_hdS, _hdS, dtype=query.dtype, device=query.device),
                    }
                    _shS[("sr", _BT, _CPR)] = _src
                # R148 P1: stage per-layer R_v^T into the shared carrier (graph reads
                # it LIVE at replay => ONE glue graph serves ALL layers of a bucket).
                if _Rv147 is not None:
                    _src["rvT"].copy_(_Rv147.transpose(-1, -2))
                # rotate Q for ALL framework rows: per-row matmul loop = legacy op
                # sequence verbatim (bit-exact); B ops, small.
                for _bS in range(batch_size):
                    if _Rk144 is not None:
                        torch.matmul(query[_bS : _bS + 1], _Rk144, out=_src["qrot"][_bS : _bS + 1])
                    else:
                        _src["qrot"][_bS : _bS + 1].copy_(query[_bS : _bS + 1])
                # stage S+R content (HP buffers are per-layer; 2 copies per row)
                _prs_list = []
                for _bS in range(batch_size):
                    _ridS = self._tt_rid(_bS)
                    _slotS = self._tt_slot(_bS)
                    _prS = self._tt_pref.get(_ridS, 0)
                    _rcS = self._tt_rcnt.get(_ridS, 0)
                    _nS = min(_prS + _rcS, _CPR)
                    _prs_list.append(int(_nS))
                    if _nS > 0:
                        _src["prk"][_bS, :_prS] = _hkS[_slotS, :_prS]
                        _src["prk"][_bS, _prS:_nS] = _hkS[_slotS, _prS : _prS + (_nS - _prS)]
                        _src["prv"][_bS, :_prS] = _hvS[_slotS, :_prS]
                        _src["prv"][_bS, _prS:_nS] = _hvS[_slotS, _prS : _prS + (_nS - _prS)]
                _prs_list = _prs_list + [1] * (_BT - batch_size)  # pad rows len 1
                for _bS in range(_BT):
                    _src["prlens"][_bS] = _prs_list[_bS]
                    if _bS >= batch_size:  # pad rows reuse row-0 first token (masked)
                        _src["prk"][_bS, 0] = _src["prk"][0, 0]
                        _src["prv"][_bS, 0] = _src["prv"][0, 0]
                # ONE FIA for all rows (eager; host lens list)
                _attnS, _ = torch_npu.npu_fused_infer_attention_score(
                    query=_src["qrot"].unsqueeze(2),
                    key=_src["prk"].permute(0, 2, 1, 3),
                    value=_src["prv"].permute(0, 2, 1, 3),
                    input_layout="BNSD",
                    actual_seq_lengths=[1] * _BT,
                    actual_seq_lengths_kv=_prs_list,
                    num_heads=_nqS,
                    num_key_value_heads=_nkvS,
                    scale=self.scale,
                    sparse_mode=0,
                )
                _src["opr"].copy_(_attnS[:, :, 0, :])
                # l_h / o_h into carrier space: rows present in _cb take the R146 op
                # result; other rows keep l=-inf (exact-selected away below).
                _src["lh"].fill_(float("-inf"))
                _src["oh"].zero_()
                _src["hOKd"].zero_()
                for _iS in range(_BpS):
                    _bS = _cb[_iS]
                    _src["lh"][_bS].copy_(_carr["lse"][_iS, :, 0].float())
                    _src["oh"][_bS].copy_(_carr["out"][_iS, :, 0, :].float())
                    _src["hOKd"][_bS] = 1
                # per-row l_pr einsum VERBATIM (legacy op sequence on identical values)
                _kvhS = torch.arange(_nqS, device=query.device) // (_nqS // _nkvS)
                for _iS in range(_BpS):
                    _bS = _cb[_iS]
                    _ridS = self._tt_rid(_bS)
                    _prS = self._tt_pref.get(_ridS, 0)
                    _nS = _prs_list[_bS]
                    _qrowS = _src["qrot"][_bS]  # [nq, hd] rotated (legacy q_rot)
                    _kkS = _src["prk"][_bS, :_nS][:, _kvhS, :].permute(1, 0, 2).float()
                    _scS = torch.einsum("hd,hnd->hn", _qrowS.float(), _kkS) * self.scale
                    _src["lpr"][_bS].copy_(torch.logsumexp(_scS, dim=-1))
                for _bS in range(batch_size):
                    if _bS in _cb:
                        continue
                    _ridS = self._tt_rid(_bS)
                    _prS = self._tt_pref.get(_ridS, 0)
                    _nS = _prs_list[_bS]
                    if _nS <= 0:
                        continue
                    _qrowS = _src["qrot"][_bS]
                    _kkS = _src["prk"][_bS, :_nS][:, _kvhS, :].permute(1, 0, 2).float()
                    _scS = torch.einsum("hd,hnd->hn", _qrowS.float(), _kkS) * self.scale
                    _src["lpr"][_bS].copy_(torch.logsumexp(_scS, dim=-1))
                # vectorized merge (identical per-element formula & order as legacy)
                _mS = torch.maximum(_src["lpr"], _src["lh"])
                _wprS = torch.exp(_src["lpr"] - _mS)
                _whS = torch.exp(_src["lh"] - _mS)
                _mgS = (_wprS.unsqueeze(-1) * _src["opr"] + _whS.unsqueeze(-1) * _src["oh"]) / (_wprS + _whS).unsqueeze(
                    -1
                )
                _prOK = _src["prlens"] > 0
                _hOK = _src["hOKd"] > 0
                # NPU torch.where does not broadcast the cond dim; expand explicitly.
                _bothE = (_prOK & _hOK).view(-1, 1, 1).expand(-1, _nqS, _hdS)
                _hE = _hOK.view(-1, 1, 1).expand(-1, _nqS, _hdS)
                if _os.environ.get("R148_GLUE_GRAPH", "0") == "1":
                    # R148 P1 merge-glue graph: the FULL merge sequence (max/exp/weighted
                    # sum VERBATIM + where-select + bf16 cast + V-unrot -> persistent
                    # outsel) is captured per (layer_ord, BT, CPR); replay reads the
                    # persistent carriers LIVE (R134 pattern). EAGER stays: FIA +
                    # per-row einsum + staging (host lens / dynamic slices). Framework
                    # writeback stays eager (address changes per step/layer). Graph keyed
                    # by layer ordinal (per-layer R_v address baked in matmul).
                    def _glue_body():
                        _mG = torch.maximum(_src["lpr"], _src["lh"])
                        _wprG = torch.exp(_src["lpr"] - _mG)
                        _whG = torch.exp(_src["lh"] - _mG)
                        _mgG = (_wprG.unsqueeze(-1) * _src["opr"] + _whG.unsqueeze(-1) * _src["oh"]) / (
                            _wprG + _whG
                        ).unsqueeze(-1)
                        _prOKG = _src["prlens"] > 0
                        _hOKG = _src["hOKd"] > 0
                        _bothEG = (_prOKG & _hOKG).view(-1, 1, 1).expand(-1, _nqS, _hdS)
                        _hEG = _hOKG.view(-1, 1, 1).expand(-1, _nqS, _hdS)
                        _mrgG = torch.where(_bothEG, _mgG, torch.where(_hEG, _src["oh"], _src["opr"]))
                        _mgBG = _mrgG.to(query.dtype)
                        torch.matmul(_mgBG, _src["rvT"], out=_src["outsel"])

                    _gluG = _src.get("graph_" + ("rv" if _Rv147 is not None else "norv"))
                    if _gluG is None:
                        # capture protocol: real fill already staged above; warmup x2
                        # on side stream, then capture (NO fill inside capture).
                        _sG = torch.npu.Stream()
                        _sG.wait_stream(torch.npu.current_stream())
                        with torch.npu.stream(_sG):
                            for _ in range(2):
                                _glue_body()
                        torch.npu.current_stream().wait_stream(_sG)
                        torch.npu.synchronize()
                        _gG = torch.npu.NPUGraph()
                        with torch.npu.graph(_gG, stream=_sG):
                            _glue_body()
                        torch.npu.synchronize()
                        _src["graph_" + ("rv" if _Rv147 is not None else "norv")] = _gG
                        _gluG = _gG
                        _gluG.replay()  # capture didn't execute; produce this step
                    else:
                        _gluG.replay()
                else:
                    _mergedS = torch.where(_bothE, _mgS, torch.where(_hE, _src["oh"], _src["opr"]))
                    _src["merged"].copy_(_mergedS)
                    # legacy dtype path: merged -> query.dtype, THEN R_v^T matmul (bf16),
                    # then output[:B] (== stack->matmul->assign on identical values).
                    _mgB = _src["merged"].to(query.dtype)
                    if _Rv147 is not None:
                        torch.matmul(_mgB, _Rv147.transpose(-1, -2), out=_src["outsel"])
                    else:
                        _src["outsel"].copy_(_mgB)
                output[:batch_size] = _src["outsel"][:batch_size]
                _st["merge"] += batch_size
                if _os.environ.get("R147_SR_CMP", "0") == "1":
                    # per-layer bit-exact adjudication vs legacy per-row recompute
                    _lg = []
                    for b in range(batch_size):
                        sl = int(seq_lens[b])
                        _rid = self._tt_rid(b)
                        _slot = self._tt_slot(b)
                        pref = self._tt_pref.get(_rid, 0)
                        rcnt = self._tt_rcnt.get(_rid, 0)
                        n_pr = pref + rcnt
                        pr_k, pr_v = self._r198_pr_window(_rid, _slot, int(seq_lens[b]), pref, rcnt, _hkS, _hvS)
                        q_b = query[b : b + 1]
                        q_rotL = torch.matmul(q_b, _Rk144) if _Rk144 is not None else q_b
                        al, _ = torch_npu.npu_fused_infer_attention_score(
                            query=q_rotL.unsqueeze(2),
                            key=pr_k.permute(1, 0, 2).unsqueeze(0),
                            value=pr_v.permute(1, 0, 2).unsqueeze(0),
                            input_layout="BNSD",
                            actual_seq_lengths=[1],
                            actual_seq_lengths_kv=[n_pr],
                            num_heads=self.num_heads,
                            num_key_value_heads=self.num_kv_heads,
                            scale=self.scale,
                            sparse_mode=0,
                        )
                        o_prL = al[0, :, 0, :].float()
                        kk = pr_k[:, _kvhS, :].permute(1, 0, 2).float()
                        sc = torch.einsum("hd,hnd->hn", q_rotL[0].float(), kk) * self.scale
                        l_prL = torch.logsumexp(sc, dim=-1)
                        o_hL, l_hL = _r144_bmid.get(b, (None, None))
                        if o_prL is not None and o_hL is not None:
                            mL = torch.maximum(l_prL, l_hL)
                            wp = torch.exp(l_prL - mL)
                            wh = torch.exp(l_hL - mL)
                            mgL = (wp.unsqueeze(-1) * o_prL + wh.unsqueeze(-1) * o_hL) / (wp + wh).unsqueeze(-1)
                        elif o_hL is None:
                            mgL = o_prL
                        else:
                            mgL = o_hL
                        _lg.append(mgL.to(query.dtype))
                    outL = torch.stack(_lg, dim=0)
                    if _Rv147 is not None:
                        outL = torch.matmul(outL, _Rv147.transpose(-1, -2))
                    _diffS = (output[:batch_size].float() - outL.float()).abs().max().item()
                    try:
                        with open("/tmp/r147_srcmp.log", "a") as _pf:
                            _pf.write(
                                f"L={_lordS} B={batch_size} Bp={_BpS} maxdiff={_diffS} "
                                f"prs={_prs_list[:batch_size]} cml={_cml}\n"
                            )
                    except Exception:
                        pass
        # ---- R196 P1 (default OFF): batched S+R tier for the eager middle ----
        # One padded FIA over the raw HP rows + verbatim per-row l_pr einsum +
        # vectorized merge + ONE unrot/writeback, replacing the per-row loop
        # below (B FIA launches + 2B cats + per-row merge math). Falls back to
        # the legacy loop (returns False before any write) on any contract
        # mismatch. See _r196_batched_sr for the math-contract derivation.
        _r196_done = False
        if (
            _os.environ.get("R196_EAGER_BATCH", "0") == "1"
            and not _srB147
            and not fused_mu_done
            and batch_size >= 2
            and not _EXTRA_CTX.capturing
        ):
            if _r197s:
                try:  # noqa: SIM105 - timing arm fails open by design
                    _r197t["t5a"] = _t197.perf_counter()
                except Exception:
                    pass
            # R204_MIXCHAIN: when the R146 carrier covers every batch row in
            # order, hand the batched S+R tier the carrier's already-rotated q
            # (built by the SAME batched matmul _r196 would re-run — identical
            # values, one fewer [B,nq,hd]@[hd,hd] matmul per layer).
            _qrot_pre204 = None
            if (
                _slab204 is not None
                and _r144_ovl_os.environ.get("R204_MIXCHAIN", "0") == "1"
                and _carr is not None
                and _slab204[2] == tuple(range(batch_size))
            ):
                _qrot_pre204 = _carr["qrot"][:batch_size].squeeze(2)
            _r196_done = self._r196_batched_sr(
                query, output, layer, seq_lens, batch_size, _r144_bmid, _st, _slab204=_slab204, _qrot_pre=_qrot_pre204
            )
            if _r197s:
                try:
                    _r197t["t5"] = _t197.perf_counter()  # post R196 batched S+R
                    _r197t["ev5"] = torch.npu.Event(enable_timing=True)
                    _r197t["ev5"].record()
                except Exception:  # noqa: SIM105 - event arm fails open by design
                    pass
        if _r197s:
            try:  # noqa: SIM105 - probe arm fails open by design
                self._r197_probe_record(attn_metadata, layer, batch_size, seq_lens, _r196_done, _r197t, _r197ev0)
            except Exception:
                pass
        for b in range(batch_size):
            if _r196_done:
                break  # R196 batched path handled all rows (output written)
            sl = int(seq_lens[b])
            _rid = self._tt_rid(b)
            _slot = self._tt_slot(b)
            pref = self._tt_pref.get(_rid, 0)
            rcnt = self._tt_rcnt.get(_rid, 0)
            mid_start = pref
            mid_end = sl - rcnt
            mid_len = mid_end - mid_start

            # ---- S+R tiers: exact BF16 from HP buffers, explicit FP32 logsumexp ----
            hk, hv = self._tt_hp_bufs(self._oscar_layer_ord(layer), query.device, query.dtype)
            pr_k, pr_v = self._r198_pr_window(_rid, _slot, sl, pref, rcnt, hk, hv)
            if pr_k is None:
                pr_k = torch.empty(0, self.num_kv_heads, self.head_size, device=query.device, dtype=query.dtype)
                pr_v = torch.empty(0, self.num_kv_heads, self.head_size, device=query.device, dtype=query.dtype)
            n_pr = pref + rcnt
            if _os.environ.get("R164_PASSIVE", "0") == "1" and int(sl) > 1000 and pref == 128:
                # PASSIVE probe: pr_k window mass into a device ring buffer —
                # NO host sync here (the .item() syncs of earlier audits perturb
                # the race); dumped once at exit via atexit.
                try:
                    _pb = _R2XX_G.setdefault("_R164_PASSBUF", None)
                    if _pb is None:
                        _pb = torch.zeros(4096, 3, dtype=torch.float32, device=pr_k.device)
                        _R2XX_G["_R164_PASSBUF"] = _pb
                        _R2XX_G["_R164_PASSN"] = [0]
                        import atexit as _ax

                        def _dump(_pb=_pb):  # bind now: same registry buffer
                            try:
                                import json as _jd

                                n = min(_R2XX_G["_R164_PASSN"][0], 4096)
                                cpu = _pb[:n].cpu().tolist()
                                with open("/tmp/r164_passive.jsonl", "w") as _fd:
                                    for i, r in enumerate(cpu):
                                        _fd.write(
                                            _jd.dumps(
                                                {"i": i, "L": int(r[0]), "srk": round(r[1], 1), "row0": round(r[2], 3)}
                                            )
                                            + "\n"
                                        )
                            except Exception:
                                pass

                        _ax.register(_dump)
                    _n = _R2XX_G["_R164_PASSN"][0]
                    if _n < 4096:
                        _pb[_n, 0] = float(self._oscar_layer_ord(layer))
                        _pb[_n, 1] = pr_k.float().sum()
                        _pb[_n, 2] = hk[_slot, 0, 0, :4].float().sum()
                        _R2XX_G["_R164_PASSN"][0] = _n + 1
                except Exception:
                    pass
            if _os.environ.get("R164_TIERDUMP", "0") == "1":
                _pr_k_row[b] = round(float(pr_k.float().sum()), 4)

            R_k = R_v = None
            if self._oscar_R_k is not None and self._oscar_layer_ord(layer) < self._oscar_R_k.shape[0]:
                R_k = self._oscar_R_k[self._oscar_layer_ord(layer)]
                R_v = self._oscar_R_v[self._oscar_layer_ord(layer)]
            q_b = query[b : b + 1]
            q_rot = torch.matmul(q_b, R_k) if R_k is not None else q_b  # [1, nq, hd]
            if _R2XX_G.setdefault("_R218_FIRES", set()).isdisjoint({"Q2"}):
                _R2XX_G["_R218_FIRES"].add("Q2")
                try:
                    _q2d = (
                        float((q_rot - q_b).abs().max())
                        if R_k is not None and not torch.npu.is_current_stream_capturing()
                        else -1.0
                    )
                except Exception:
                    _q2d = -2.0
                _r218_diag("FIRE-Q2 threetier per-row qrot Rk_none={} qd={:.4e}".format(R_k is None, _q2d))

            if _srB147:
                continue  # batched path handled all rows
            o_pr = None
            l_pr = None
            if n_pr > 0:
                # R154 V0-A (M2-class, env-gated default OFF): FIA_v2 with
                # return_softmax_lse=True returns the S+R LSE directly (op-level
                # evidence: lse == legacy einsum logsumexp <=1 fp32 ULP; out
                # bit-exact vs v1) -> drop the cast[10.5MB]+einsum+mul+logsumexp
                # recompute chain. Golden+error-matrix+128-tok identity gate
                # before any promotion (R154_M2 rule).
                if _os.environ.get("R154_LSE_REUSE", "0") == "1":
                    attn_out_pr, _lse_pr = torch_npu.npu_fused_infer_attention_score_v2(
                        query=q_rot.unsqueeze(2),  # [1, nq, 1, hd]
                        key=pr_k.permute(1, 0, 2).unsqueeze(0),  # [1, nkv, n_pr, hd]
                        value=pr_v.permute(1, 0, 2).unsqueeze(0),  # [1, nkv, n_pr, hd]
                        input_layout="BNSD",
                        actual_seq_qlen=[1],
                        actual_seq_kvlen=[n_pr],
                        num_query_heads=self.num_heads,
                        num_key_value_heads=self.num_kv_heads,
                        softmax_scale=self.scale,
                        return_softmax_lse=True,
                    )
                else:
                    attn_out_pr, _lse_pr = torch_npu.npu_fused_infer_attention_score(
                        query=q_rot.unsqueeze(2),  # [1, nq, 1, hd]
                        key=pr_k.permute(1, 0, 2).unsqueeze(0),  # [1, nkv, n_pr, hd]
                        value=pr_v.permute(1, 0, 2).unsqueeze(0),  # [1, nkv, n_pr, hd]
                        input_layout="BNSD",
                        actual_seq_lengths=[1],
                        actual_seq_lengths_kv=[n_pr],
                        num_heads=self.num_heads,
                        num_key_value_heads=self.num_kv_heads,
                        scale=self.scale,
                        sparse_mode=0,
                    )
                _lse_pr = None if (_lse_pr is None or _lse_pr.numel() == 0) else _lse_pr
                o_pr = attn_out_pr[0, :, 0, :].float()  # [nq, hd] rotated basis
                # R154 V0-A (M2-class, env-gated default OFF): reuse the FIA's own
                # softmax-LSE return instead of re-computing scores with an
                # einsum+logsumexp chain (cast of kk [nq,n_pr,hd] to fp32 alone
                # materializes ~10.5MB/layer). Numeric class: same mathematical
                # quantity, different accumulation order -> golden + error matrix
                # + 128-tok identity gate before promotion (R154_M2 rule).
                if (
                    _os.environ.get("R154_LSE_REUSE", "0") == "1"
                    and _lse_pr is not None
                    and _lse_pr.numel() == self.num_heads
                ):
                    l_pr = _lse_pr.reshape(-1).float()  # [nq]
                    if _os.environ.get("R154_LSE_PROBE", "0") == "1":
                        # R154 M2 golden probe: legacy einsum lse vs FIA lse vs fp64 ref
                        _prb = globals().setdefault("_R154_LSE_PROBE", [0])
                        if _prb[0] < 64:
                            _prb[0] += 1
                            kvh_p = torch.arange(self.num_heads, device=query.device) // (
                                self.num_heads // self.num_kv_heads
                            )
                            kk_p = pr_k[:, kvh_p, :].permute(1, 0, 2).float()
                            sc_p = torch.einsum("hd,hnd->hn", q_rot[0].float(), kk_p) * self.scale
                            l_leg = torch.logsumexp(sc_p, dim=-1)
                            sc64 = torch.einsum("hd,hnd->hn", q_rot[0].double(), kk_p.double()) * float(self.scale)
                            l_ref = torch.logsumexp(sc64, dim=-1)
                            d_leg = (l_leg.double() - l_ref).abs()
                            d_new = (l_pr.double() - l_ref).abs()
                            try:
                                import json as _js154

                                with open("/tmp/r154_lse_probe.jsonl", "a") as _pf:
                                    _pf.write(
                                        _js154.dumps(
                                            {
                                                "n": int(_prb[0]),
                                                "n_pr": int(n_pr),
                                                "leg_max_err": float(d_leg.max()),
                                                "new_max_err": float(d_new.max()),
                                                "leg_mean_err": float(d_leg.mean()),
                                                "new_mean_err": float(d_new.mean()),
                                                "cross_max": float((l_leg - l_pr).abs().max()),
                                            }
                                        )
                                        + "\n"
                                    )
                            except Exception:
                                pass
                else:
                    # explicit FP32 logsumexp over the SAME prefix+recent tokens
                    kvh = torch.arange(self.num_heads, device=query.device) // (self.num_heads // self.num_kv_heads)
                    kk = pr_k[:, kvh, :].permute(1, 0, 2).float()  # [nq, n_pr, hd]
                    scores = torch.einsum("hd,hnd->hn", q_rot[0].float(), kk) * self.scale
                    l_pr = torch.logsumexp(scores, dim=-1)  # [nq]

            # ---- H tier: candidate over persistent INT2 middle ----
            o_h = None
            l_h = None
            _r218h = _R2XX_G.setdefault("_R218_FIRES", set())
            if "H" not in _r218h:
                _r218h.add("H")
                _r218_diag(
                    f"FIRE-H consumer mid_len={int(mid_len)} b_in_bmid={b in _r144_bmid} "
                    f"nbmid={len(_r144_bmid)} gates: r145h={globals().get('_r145_handled')} "
                    f"R145T={globals().get('_R145T')} capt={bool(_EXTRA_CTX.capturing)}"
                )
            if mid_len > 0:
                if b in _r144_bmid:
                    # R144 G3: use the batched middle result (ONE aclnn for all reqs)
                    o_h, l_h = _r144_bmid[b]
                    if _os.environ.get("R164_MIDPROBE", "0") == "1":
                        self._r164_midprobe(
                            {
                                "src": "bmid",
                                "L": int(layer_idx),
                                "b": int(b),
                                "sl": int(sl),
                                "mid": [int(mid_start), int(mid_end)],
                                "o_h": float(o_h.sum()),
                                "l_h": float(l_h.sum()),
                            }
                        )
                else:
                    got = self._r109_middle_candidate(
                        query,
                        attn_metadata,
                        layer,
                        b,
                        mid_start,
                        mid_end,
                        q_rot_pre=(q_rot if _os.environ.get("R113_QROT_DEDUPE", "0") == "1" else None),
                    )
                    _r218_gr = _R2XX_G.setdefault("_R218_MIDC_N", [0])
                    if _r218_gr[0] <= 3:
                        _r218_diag("MIDC-RESULT got_none={}".format(got is None))
                    if got is not None:
                        o_h, l_h = got[0].float(), got[1].float()
                    if _os.environ.get("R164_MIDPROBE", "0") == "1":
                        _blkid = None
                        _btrow = None
                        _btsums = None
                        try:
                            _blkid = int(
                                attn_metadata.block_tables[
                                    b, mid_start // self.vllm_config.cache_config.block_size
                                ].item()
                            )
                            _btrow = attn_metadata.block_tables[b, :8].tolist()
                            _btsums = [float(self.key_cache[_bi].to(torch.int16).sum().item()) for _bi in _btrow]
                        except Exception:
                            pass
                        self._r164_midprobe(
                            {
                                "src": "candidate",
                                "L": int(layer_idx),
                                "b": int(b),
                                "sl": int(sl),
                                "mid": [int(mid_start), int(mid_end)],
                                "blk": _blkid,
                                "bt_row": _btrow,
                                "bt_kc_sums": _btsums,
                                "ksc_blk_sum": (
                                    None if _blkid is None else float(self.k_scale_cache[_blkid].float().sum().item())
                                ),
                                "o_h": (None if o_h is None else float(o_h.sum())),
                                "l_h": (None if l_h is None else float(l_h.sum())),
                            }
                        )

            if capt is not None and b == 0:
                capt["tiers"].append(
                    {
                        "sl": sl,
                        "pref": pref,
                        "rcnt": rcnt,
                        "mid_len": mid_len,
                        "mid_start": mid_start,
                        "mid_end": mid_end,
                        "o_pr": None if o_pr is None else o_pr.detach().cpu(),
                        "l_pr": None if l_pr is None else l_pr.detach().cpu(),
                        "o_h": None if o_h is None else o_h.detach().cpu(),
                        "l_h": None if l_h is None else l_h.detach().cpu(),
                    }
                )

            if _os.environ.get("R164_TIERDUMP", "0") == "1":
                _o_pr_row[b] = None if o_pr is None else round(float(o_pr.sum()), 4)
                _l_pr_row[b] = None if l_pr is None else round(float(l_pr.sum()), 4)
                _o_h_row[b] = None if o_h is None else round(float(o_h.sum()), 4)
                _l_h_row[b] = None if l_h is None else round(float(l_h.sum()), 4)
            # ---- exact merge in rotated basis (task §6.3) ----
            if (
                _os.environ.get("R115_MERGE_UNROT", "0") == "1"
                and o_pr is not None
                and o_h is not None
                and l_pr is not None
                and l_h is not None
                and batch_size == 1
                and R_v is not None
                and getattr(type(self), "_r115_aclnn_mod", None) is not None
            ):
                # R115 Phase-4: ONE kernel = stable 2-tier merge + once V-unrot,
                # writing the framework output buffer directly (isolated gate:
                # 4/6 bit-exact vs python golden, rest <=2ULP, nrmse<=1.1e-5).
                _mod = type(self)._r115_aclnn_mod
                _ones = torch.ones(1, dtype=torch.int32, device=o_pr.device)
                _oc = output[:batch_size].view(batch_size, o_pr.shape[0], 1, o_pr.shape[1])
                r_mu, lr_mu, _c_mu = _mod.merge_unrot_aclnn(
                    o_pr.view(1, o_pr.shape[0], 1, o_pr.shape[1]).to(torch.bfloat16),
                    o_h.view(1, o_h.shape[0], 1, o_h.shape[1]).to(torch.bfloat16),
                    l_pr.view(1, -1, 1).contiguous(),
                    l_h.view(1, -1, 1).contiguous(),
                    _ones,
                    _ones,
                    R_v.contiguous(),
                    _oc,
                )
                if r_mu != 0 or lr_mu != 0:
                    raise RuntimeError(f"R115 merge_unrot failed rc={r_mu} lr={lr_mu}")
                type(self)._r115_mu_calls = getattr(type(self), "_r115_mu_calls", 0) + 1
                fused_mu_done = True
            elif o_pr is None:
                merged = o_h
                _r218bg = _R2XX_G.setdefault("_R218_FIRES", set())
                if "MB1" not in _r218bg:
                    _r218bg.add("MB1")
                    _r218_diag(f"MERGE-BRANCH o_pr=None lay={int(layer_idx)}")
            elif o_h is None:
                merged = o_pr
                _r218bg = _R2XX_G.setdefault("_R218_FIRES", set())
                if "MB2" not in _r218bg:
                    _r218bg.add("MB2")
                    _r218_diag(f"MERGE-BRANCH o_h=None lay={int(layer_idx)}")
            else:
                m = torch.maximum(l_pr, l_h)
                w_pr = torch.exp(l_pr - m)
                w_h = torch.exp(l_h - m)
                merged = (w_pr.unsqueeze(-1) * o_pr + w_h.unsqueeze(-1) * o_h) / (w_pr + w_h).unsqueeze(-1)
                _r218mg = globals().setdefault("_R218_MERGECAP_N", [0])
                if _r218mg[0] < 2 and _os.environ.get("R218_MERGECAP", "0") == "1" and int(mid_len) >= 128:
                    _r218mg[0] += 1
                    try:
                        torch.npu.synchronize()
                        torch.save(
                            {
                                "n": _r218mg[0],
                                "site": "A-row",
                                "layer": int(layer_idx),
                                "mid": [int(mid_start), int(mid_end)],
                                "l_pr": None if l_pr is None else l_pr.detach().float().cpu().clone(),
                                "l_h": None if l_h is None else l_h.detach().float().cpu().clone(),
                                "o_pr": None if o_pr is None else o_pr.detach().float().cpu().clone(),
                                "o_h": None if o_h is None else o_h.detach().float().cpu().clone(),
                                "merged": merged.detach().float().cpu().clone(),
                                "q": query[b : b + 1].detach().cpu().clone(),
                            },
                            f"/tmp/r218_mergecap_A{_r218mg[0]}.pt",
                        )
                        _r218_diag(f"MERGECAP-A saved #{_r218mg[0]} lay={int(layer_idx)}")
                    except Exception as _eMG:
                        _r218_diag("MERGECAP ERR {!r}".format(_eMG))
                _st["merge"] += 1
                # R150 FIRST_FAIL#2 probe: B1-with-middle zeros — tier finiteness
                _r150ff_n = globals().setdefault("_R150_FF_N", [0])
                if _r150ff_n[0] < 4 and l_pr is not None and l_h is not None:
                    _r150ff_n[0] += 1
                    try:
                        import torch as _t150

                        _r150_tr(
                            "merge_probe",
                            bn=int(_r150ff_n[0]),
                            bs=int(b) if isinstance(b, int) else -1,
                            lpr_finite=bool(_t150.isfinite(l_pr).all().item()),
                            lh_finite=bool(_t150.isfinite(l_h).all().item()),
                            lpr_max=float(l_pr.max().item()),
                            lh_max=float(l_h.max().item()),
                            lpr_min=float(l_pr.min().item()),
                            lh_min=float(l_h.min().item()),
                            opr_finite=bool(_t150.isfinite(o_pr).all().item()),
                            oh_finite=bool(_t150.isfinite(o_h).all().item()),
                            oh_absmax=float(o_h.abs().max().item()),
                            merged_finite=bool(_t150.isfinite(merged).all().item()),
                        )
                    except Exception as _e150ff:
                        _r150_tr("merge_probe_err", err=str(_e150ff)[:200])
            if not fused_mu_done:
                if _os.environ.get("R164_TIERDUMP", "0") == "1" and merged is not None:
                    _m_row[b] = round(float(merged.float().sum()), 4)
                assert merged is not None  # merge arms above guarantee a value
                merged_all.append(merged.to(query.dtype))

        if fused_mu_done:
            out_b = None
            out_b_rot = None
        elif _srB147 or _r196_done:
            out_b = None
            out_b_rot = None  # batched path wrote output already
        else:
            if _R2XX_G.setdefault("_R218_FIRES", set()).isdisjoint({"O2"}):
                _R2XX_G["_R218_FIRES"].add("O2")
                _r218_diag(f"FIRE-O2 threetier final unrot Rv_none={R_v is None} rows={len(merged_all)}")
            out_b = torch.stack(merged_all, dim=0)  # [B, nq, hd] rotated basis
            # R111 S5: keep BOTH bases for the capture (pre/post unrotation)
            out_b_rot = out_b
            if R_v is not None:
                out_b = torch.matmul(out_b, R_v.transpose(-1, -2))
            output[:batch_size] = out_b

        # ---------------- R111 rich S3/S4/S5/S6 capture (task-owned) ----------------
        try:
            if _os.environ.get("R111_CAPTURE", "0") == "1":
                _r111_cap_layers = set(int(x) for x in _os.environ.get("R111_CAPT_LAYERS", "0").split(",") if x != "")
                _r111_max_steps = int(_os.environ.get("R111_CAPT_STEPS", "2"))
                if layer_idx in _r111_cap_layers and batch_size >= 1:
                    _r111_dir = _odiag_p("direct_raw")
                    _os.makedirs(_r111_dir, exist_ok=True)
                    # step index = count of existing files for this layer (fs=truth)
                    _r111_step = 0
                    while _os.path.exists(f"{_r111_dir}/r111_s345_capt_L{layer_idx}_S{_r111_step}.pt"):
                        _r111_step += 1
                    if _r111_step < _r111_max_steps:
                        b = 0
                        _rid0 = self._tt_rid(b)
                        _slot0 = self._tt_slot(b)
                        sl = int(seq_lens[b])
                        pref = self._tt_pref.get(_rid0, 0)
                        rcnt = self._tt_rcnt.get(_rid0, 0)
                        mid_start = pref
                        mid_end = sl - rcnt
                        mid_len = mid_end - mid_start
                        hk, hv = self._tt_hp_bufs(self._oscar_layer_ord(layer), query.device, query.dtype)
                        pr_k, pr_v = self._r198_pr_window(_rid0, _slot0, sl, pref, rcnt, hk, hv)
                        # dequantized middle (A-path identical recipe) for the
                        # same-quant full-span FP64 reference
                        kf_mid = vf_mid = None
                        if mid_len > 0:
                            block_size = self.vllm_config.cache_config.block_size
                            _, _, _, packed_head = self.key_cache.shape
                            mb_lo = mid_start // block_size
                            mb_hi = (mid_end + block_size - 1) // block_size
                            bt_m = attn_metadata.block_tables[b, mb_lo:mb_hi]
                            kb = self.key_cache[bt_m].contiguous()
                            vb = self.value_cache[bt_m].contiguous()
                            ks = self.k_scale_cache[bt_m].contiguous()
                            kz = self.k_zero_cache[bt_m].contiguous()
                            vs = self.v_scale_cache[bt_m].contiguous()
                            vz = self.v_zero_cache[bt_m].contiguous()
                            T = (mb_hi - mb_lo) * block_size
                            kf_mid = self._dequantize_packed_int2(
                                kb.view(T, self.num_kv_heads, packed_head),
                                ks.view(T, self.num_kv_heads),
                                kz.view(T, self.num_kv_heads),
                                query.dtype,
                            )
                            vf_mid = self._dequantize_packed_int2(
                                vb.view(T, self.num_kv_heads, packed_head),
                                vs.view(T, self.num_kv_heads),
                                vz.view(T, self.num_kv_heads),
                                query.dtype,
                            )
                            off = mid_start - mb_lo * block_size
                            kf_mid = kf_mid[off : off + mid_len]
                            vf_mid = vf_mid[off : off + mid_len]
                        R_k_c = R_k.detach().cpu() if R_k is not None else None
                        R_v_c = R_v.detach().cpu() if R_v is not None else None
                        import torch as _t

                        _t.save(
                            {
                                "path": "B",
                                "layer": layer_idx,
                                "step": _r111_step,
                                "sl": sl,
                                "pref": pref,
                                "rcnt": rcnt,
                                "mid_start": mid_start,
                                "mid_end": mid_end,
                                "mid_len": mid_len,
                                "q_raw": query[:1].detach().cpu(),  # pre-rotation [1,nq,hd]
                                "q_rot": (torch.matmul(query[:1], R_k) if R_k is not None else query[:1])
                                .float()
                                .cpu(),  # [1,nq,hd] fp32
                                "pr_k": pr_k.detach().cpu(),
                                "pr_v": pr_v.detach().cpu(),
                                "kf_mid": None if kf_mid is None else kf_mid.detach().cpu(),
                                "vf_mid": None if vf_mid is None else vf_mid.detach().cpu(),
                                "o_pr": None if o_pr is None else o_pr.float(),
                                "l_pr": None if l_pr is None else l_pr.float(),
                                "o_h": None if o_h is None else o_h.float(),
                                "l_h": None if l_h is None else l_h.float(),
                                "merged_rot": (
                                    None if out_b_rot is None else out_b_rot[0].float().cpu()
                                ),  # [nq,hd] rotated basis
                                "out_unrot": (None if out_b is None else out_b[0].float().cpu()),  # [nq,hd] final basis
                                "R_k": R_k_c,
                                "R_v": R_v_c,
                                "scale": float(self.scale),
                                "nq": int(self.num_heads),
                                "nkv": int(self.num_kv_heads),
                                "hd": int(self.head_size),
                            },
                            f"{_r111_dir}/r111_s345_capt_L{layer_idx}_S{_r111_step}.pt",
                        )
                        print(
                            f"[R111-CAPT-B] L{layer_idx} S{_r111_step} sl={sl} mid={mid_len}",
                            file=_sys.stderr,
                            flush=True,
                        )
        except Exception as _e:
            print(f"[R111-CAPT-B] failed: {_e}", file=_sys.stderr, flush=True)

        # R164 TIERDUMP (eager decode path; default OFF): per-row per-tier
        # checksums aligned with the graph-side dump (same field order), so a
        # mixed-step eager decode row can be compared against the B1 pure
        # graph decode at the SAME sl (same request+position+layer).
        if _os.environ.get("R164_TIERDUMP", "0") == "1":
            try:
                import json as _jtde

                # torch_npu at module top registers torch.npu; alias keeps the
                # synchronize call short without importing the runtime submodule
                _npu_tde = torch.npu

                _npu_tde.synchronize()
                _lid_t = self._oscar_layer_ord(layer)
                for _bte in range(batch_size):
                    _hkR = self._tt_hp_k.get(_lid_t)
                    _row = {
                        "stage": "eager_decode",
                        "L": _lid_t,
                        "impl": id(self) % 100000,
                        "tid": (None if _hkR is None else id(_hkR) % 1000000),
                        "w": (None if _hkR is None else int(_hkR.shape[1])),
                        "b": _bte,
                        "sl": int(seq_lens[_bte]) if _bte < len(seq_lens) else -1,
                        "f": [],
                    }
                    _fL = _row["f"]
                    _q_t = q_rot[_bte] if (q_rot.dim() > 2 and q_rot.shape[0] > _bte) else q_rot
                    _fL.append(float(_q_t.float().sum()))
                    for _t in (
                        _pr_k_row.get(_bte),
                        _o_pr_row.get(_bte),
                        _l_pr_row.get(_bte),
                        _o_h_row.get(_bte),
                        _l_h_row.get(_bte),
                        _m_row.get(_bte),
                    ):
                        _fL.append(None if _t is None else round(float(_t), 4))
                    with open("/tmp/r164_tierdump.jsonl", "a") as _fe:
                        _fe.write(_jtde.dumps(_row) + "\n")
            except Exception as _ete:
                try:
                    with open("/tmp/r164_tierdump.jsonl", "a") as _fe2:
                        _fe2.write("eager_dump_error " + repr(_ete) + "\n")
                except Exception:
                    pass
        # R111 S6 fingerprint: per-layer/step hash of the attention output (all layers)
        try:
            if _os.environ.get("R111_HASH", "1") == "1" and out_b is not None:
                import hashlib as _hl

                _h = _hl.sha256(out_b.detach().cpu().numpy().tobytes()).hexdigest()[:16]
                with _odiag_o("r111_b_hashes.jsonl") as _f:
                    _f.write(f'{{"p":"B","L":{layer_idx},"sl":{int(seq_lens[0])},"h":"{_h}"}}\n')
        except Exception:
            pass

        # R170E eager OUT/LSE snapshot (R171 second-lp probe; default OFF):
        # the R168 snapshot hook lives in the graph mid-path updater and
        # never fires eager (g0_gold ran OUTLSE=1 with ZERO records), so an
        # eager per-layer golden needs its own emitter here. SAME
        # /tmp/r168_outlse_p%d.pt format (layers "L{lid}c{B}": out fp16 +
        # lse fp32, rids/sls for cross-leg matching) so the census/diff
        # tooling consumes both. B1 only; flush on layer-0 wrap; cap 1100.
        try:
            # R171: also require R152_EAGER=1 — this hook must NEVER fire
            # during graph capture (torch.npu.synchronize()/.cpu() inside a
            # captured region is a capture violation); eager legs set both
            # envs, graph OUTLSE legs set only R168_OUTLSE.
            if (
                os.environ.get("R168_OUTLSE", "0") == "1"
                and _OSCAR_CFG.eager
                and batch_size == 1
                and (l_pr is not None or l_h is not None)
            ):
                _eN = globals().setdefault("_R168E_N", [0])
                if _eN[0] < 1100:
                    _lidE = int(self._oscar_layer_ord(layer))
                    if _lidE == 0 and globals().get("_R168E_BUF"):
                        torch.save(dict(globals()["_R168E_BUF"]), f"/tmp/r168_outlse_p{_eN[0]}.pt")
                        _eN[0] += 1
                        globals()["_R168E_BUF"] = None
                    _eb = globals().get("_R168E_BUF")
                    if _eb is None:
                        _eb = globals()["_R168E_BUF"] = {
                            "s": _eN[0],
                            "nt": int(query.shape[0]),
                            "rids": [self._tt_rid(0)],
                            "sls": [int(seq_lens[0])],
                            "layers": {},
                        }
                    if l_pr is not None and l_h is not None:
                        _mE = torch.maximum(l_pr, l_h)
                        _lseE = _mE + torch.log(torch.exp(l_pr - _mE) + torch.exp(l_h - _mE))
                    else:
                        _lseE = (l_pr if l_pr is not None else l_h).clone()
                    _outE = out_b_rot if out_b_rot is not None else output[:1]
                    torch.npu.synchronize()
                    _eb["layers"][f"L{_lidE}c{batch_size}"] = {
                        "out": _outE[0].detach().to(torch.float16).unsqueeze(0).unsqueeze(-2).cpu(),
                        "lse": _lseE.detach().float().view(1, -1, 1).cpu(),
                        # R171 tier-split fields (additive; census reads only
                        # "lse"/"out"): the graph mid-path pool lse IS the
                        # middle-INT2 tier alone, so eager-vs-pool must use
                        # lse_h — the merged "lse" above is NOT comparable.
                        "lse_pr": (None if l_pr is None else l_pr.detach().float().view(1, -1, 1).cpu()),
                        "lse_h": (None if l_h is None else l_h.detach().float().view(1, -1, 1).cpu()),
                        "basis": "rot" if out_b_rot is not None else "final",
                    }
        except Exception as _eE:
            print("[R168E-OUTLSE] eager snap fail: {}".format(_eE), flush=True)

        if capt is not None:
            import torch as _t

            _t.save(capt, _odiag_p("r111_s04_capt.pt"))
            print(
                f"[R109-CAPT] saved layer={capt['layer']} tiers={len(capt['tiers'])} stats={_st}",
                file=_sys.stderr,
                flush=True,
            )
        return output
