# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the OSCAR INT2 attention backend.

Relocated verbatim (module level) from the monolithic attention_v1 overlay
of the frozen chain: slot allocator, tier metadata/trace records, the R216
LSE-dump diagnostic, and small shared-device-buffer helpers. The heavy
numerical paths stay in the backend methods that own them.
"""

from __future__ import annotations

import contextlib

# aliased imports carried verbatim from the monolith overlay (the moved
# function bodies reference these alias names directly)
import json as _r150_json
import os
import os as _os
import os as _r144_ovl_os
import os as _r150_os
from collections import deque
from typing import Any

import torch


def _r218_diag(msg):
    """Gated file diagnostic (PROVEN to survive the EngineCore child, where
    this module's stdout/stderr prints vanish -- see the R144-era
    /tmp/r144_int2_fwd_probe.log workaround; same mechanism). Off unless
    OSCAR_DIAG is set; OSCAR_DIAG_LOG redirects the sink (tests point it at
    a sandbox path to prove both directions without touching /tmp)."""
    try:
        if not os.environ.get("OSCAR_DIAG"):
            return
        import time as _r218_t

        _r218_path = os.environ.get("OSCAR_DIAG_LOG", "/tmp/r218_diag.log")
        with open(_r218_path, "a") as _r218_f:
            _r218_f.write(f"{_r218_t.strftime('%m-%d %H:%M:%S')} pid={_r144_ovl_os.getpid()} {msg}\n")
    except Exception:
        pass


_R170_ROWSLOT_ON = _r144_ovl_os.environ.get("R170_ROWSLOT", "0") == "1"

_R170_MID2_ON = _r144_ovl_os.environ.get("R170_MIDDUMP2", "0") == "1"

_R170_EDGEFENCE_RAW = _r144_ovl_os.environ.get("R170_EDGEFENCE", "0")

_R170_EDGEFENCE_MODE = int(_R170_EDGEFENCE_RAW) if _R170_EDGEFENCE_RAW.isdigit() else 0

_R170_EDGEFENCE_N = 0


def _r199_sc_mismatch(_me):
    """R199 diagnostic helper: rows where idx_sr's slot component differs
    from bt_sr's (the two S/R read paths disagreeing on WHICH ring row)."""
    try:
        _iv = _me.get("idx_sr")
        _bv = _me.get("bt_sr")
        if _iv is None or _bv is None:
            return None
        _n = int(_me.get("npr", 0) or 1)
        _nb = int(_me.get("nb_sr", 0) or 1)
        _rows = min(_bv.shape[0], _iv.numel() // _n)
        return [r for r in range(_rows) if int(_iv[r * _n].item()) // _n != int(_bv[r, 0].item()) // _nb]
    except Exception:
        return "ERR"


def _r199_pool_rowsum(impl, prefix):
    """R199 diagnostic helper: per-row lse sums of the LAST replay from the
    given bsrpool entry (pg=paged main path, ""=legacy selfcheck side)."""
    try:
        _p = getattr(type(impl), "_r163_bsrpool", None) or {}
        _n = int(next(iter([k[2] for k in _p if isinstance(k, tuple) and len(k) >= 3 and k[0] == prefix]), 0) or 0)
        for _k, _v in _p.items():
            if isinstance(_k, tuple) and len(_k) >= 3 and _k[0] == prefix:
                return [round(float(x), 2) for x in _v[2].view(-1, _v[2].shape[1]).sum(1).tolist()[:12]]
        return None
    except Exception:
        return "ERR"


def _r150_tr(ev, **kw):
    if _r150_os.environ.get("R150_TRACE", "0") != "1":
        return
    try:
        kw["ev"] = ev
        kw["pid"] = _r150_os.getpid()
        with open(f"/tmp/r150_trace_{_r150_os.getpid()}.jsonl", "a") as _f:
            _f.write(_r150_json.dumps(kw) + "\n")
    except Exception:
        pass


def _r166_tr(ev, **kw):
    if _r150_os.environ.get("R166_REFTRACE", "0") != "1":
        return
    try:
        _c = globals().setdefault("_R166_TR_N", [0])
        _c[0] += 1
        kw["ev"] = ev
        kw["n"] = _c[0]
        kw["pid"] = _r150_os.getpid()
        with open(f"/tmp/r166_trace_{_r150_os.getpid()}.jsonl", "a") as _f:
            _f.write(_r150_json.dumps(kw) + "\n")
    except Exception:
        pass


def _r203_qsl_host(md):
    try:
        _n = len(getattr(md, "seq_lens_list", None) or ())
        _c = getattr(md, "_r203_qsl_cache", None)
        if _c is not None and len(_c) == _n + 1:
            return _c
        _q = md.query_start_loc
        if hasattr(_q, "cpu"):
            _q = _q.cpu()
        if not isinstance(_q, list):
            _q = _q.tolist()
        _q = list(_q)
        if len(_q) == _n + 1:
            with contextlib.suppress(Exception):
                md._r203_qsl_cache = _q
        return _q
    except Exception:
        return None


_R216_STATE: dict[str, Any] = {"n": 0, "pending": None, "ledger": []}


def _r216_env() -> tuple[bool, Any]:
    import os as _o216

    return _o216.environ.get("R216_LSEDUMP", "0") == "1", _o216.environ


def _r216_mid(carr, bp, cb, cml, cap, nbC, lord, nq, nkv, bs, hd, scale, replay, lmod=None):
    try:
        on, env = _r216_env()
        # R218 bp=1 entry probe (DIAGNOSTIC_ONLY): records that the middle
        # convergence point is actually reached at B=1 and what it sees.
        try:
            if on and int(bp) < 2 and int(_R216_STATE.get("p1", 0)) < 40:
                _R216_STATE["p1"] = int(_R216_STATE.get("p1", 0)) + 1
                with open("/tmp/r216_bp1.log", "a") as _f1:
                    _f1.write(
                        f"bp={int(bp)} idx={int(_R216_STATE['n'])} "
                        f"minbp={env.get('R216_LSEDUMP_MINBP', '?')} "
                        f"cap={int(cap)} nbC={int(nbC)}\n"
                    )
        except Exception:
            pass
        # R218: capture floor is env-tunable (default 2 = R216 behavior). NS=1
        # decode replays carry bp=1, which the R216 floor silently suppressed;
        # R216_LSEDUMP_MINBP=1 enables the B=1 base-path adjudication.
        if not on or int(bp) < int(env.get("R216_LSEDUMP_MINBP", "2")):
            return
        st = _R216_STATE
        idx = st["n"]
        every = max(int(env.get("R216_LSEDUMP_EVERY", "36")), 1)
        do_ledger = (idx % every) == 0
        skip = int(env.get("R216_LSEDUMP_SKIP", "0"))
        stride = max(int(env.get("R216_LSEDUMP_STRIDE", "1")), 1)
        maxf = int(env.get("R216_LSEDUMP_MAX", "4"))
        do_full = (skip <= idx < skip + stride * maxf) and ((idx - skip) % stride == 0)
        lse_v: Any = None
        askv: Any = None
        if do_ledger or do_full:
            askv = [int(x) for x in carr["askv"][:bp].cpu().tolist()]
        if do_ledger:
            lse_v = carr["lse"][:bp].detach().float().cpu().clone()
        # cadenced light ledger (one D2H-sync per EVERY calls — a per-call sync
        # would perturb the leg's timing, and this defect class is load-sensitive)
        if do_ledger:
            {
                "idx": idx,
                "layer": int(lord),
                "cap": int(cap),
                "nbC": int(nbC),
                "bp": int(bp),
                "bidx": [int(x) for x in cb],
                "cml": [int(x) for x in cml],
                "askv": askv,
                "replay": bool(replay),
                "nq": int(nq),
                "nkv": int(nkv),
                "bs": int(bs),
                "hd": int(hd),
                "scale": float(scale),
                "lse": lse_v.tolist(),
                "out_amax": float(carr["out"][:bp].detach().float().abs().max().item()),
            }
            st["ledger"].append(
                {
                    "i": idx,
                    "l": int(lord),
                    "bp": int(bp),
                    "rp": int(bool(replay)),
                    "askv": askv,
                    "lse_min": float(lse_v.min()),
                    "lse_max": float(lse_v.max()),
                    "lse": [round(float(x), 3) for x in lse_v.flatten().tolist()],
                }
            )
        if len(st["ledger"]) >= 50:
            import json as _j216
            import os as _o216b

            d = env.get("R216_LSEDUMP_DIR", "/tmp/r216_lsedump")
            _o216b.makedirs(d, exist_ok=True)
            with open(f"{d}/ledger.jsonl", "a") as _f216:
                for _r216e in st["ledger"]:
                    _f216.write(_j216.dumps(_r216e) + "\n")
            st["ledger"] = []
        # full dump: the exact op operands + outputs for the offline fp32 ref
        # (SKIP/STRIDE spread — idx 0..~50 are boot warmups with dummy data)
        if do_full:
            import os as _o216c

            import torch as _t216

            _t216.npu.synchronize()  # inputs are pre-op stable; out/lse post-op
            d = env.get("R216_LSEDUMP_DIR", "/tmp/r216_lsedump")
            _o216c.makedirs(d, exist_ok=True)
            torch_mod = _t216
            snap = {
                "idx": idx,
                "layer": int(lord),
                "cap": int(cap),
                "nbC": int(nbC),
                "bp": int(bp),
                "bidx": [int(x) for x in cb],
                "replay": bool(replay),
                "nq": int(nq),
                "nkv": int(nkv),
                "bs": int(bs),
                "hd": int(hd),
                "scale": float(scale),
                "askv": askv,
                "asq": [int(x) for x in carr["asq"][:bp].cpu().tolist()],
                "q": carr["qrot"][:bp].detach().cpu().clone(),
                "k": carr["Kc"][: bp * int(nbC)].detach().cpu().clone(),
                "v": carr["Vc"][: bp * int(nbC)].detach().cpu().clone(),
                "bt": carr["bt"][:bp].detach().cpu().clone(),
                "ks": carr["kS"][:bp].detach().cpu().clone(),
                "ko": carr["kO"][:bp].detach().cpu().clone(),
                "vs": carr["vS"][:bp].detach().cpu().clone(),
                "vo": carr["vO"][:bp].detach().cpu().clone(),
                "out_h": carr["out"][:bp].detach().cpu().clone(),
                "lse_h": carr["lse"][:bp].detach().cpu().clone(),
            }
            torch_mod.save(snap, f"{d}/mid_{idx:04d}.pt")
        # R216 RELAY: re-launch the SAME op on the SAME carrier inputs (the
        # exact ABI the product used) after zeroing out/lse — discriminates
        # stateful/call-order corruption (relay heals it) from input-driven
        # (relay reproduces it). ALIGNED arm sets askv[:bp]=cap. Product
        # state (out/lse/askv) restored afterwards.
        if do_full and env.get("R216_RELAY", "0") == "1" and lmod is not None:
            import torch as _t216r

            _t216r.npu.synchronize()
            _o0 = carr["out"].clone()
            _l0 = carr["lse"].clone()
            _a0 = carr["askv"].clone()

            def _relay_mha(tag, nq_mha):
                # R216 ratio ladder: slice q/out/lse to nq_mha heads with
                # numKvHeads=8 -> ratio = nq_mha/8 (1 = MHA base path, 2, 4...).
                # Localizes grouped-member handling vs shared MSD machinery.
                q_t = carr["qrot"][:, :nq_mha].contiguous()
                o_m = torch.zeros_like(q_t, dtype=torch.bfloat16)
                l_m = torch.full((q_t.shape[0], nq_mha, 1), -9999.0, dtype=torch.float32, device=q_t.device)
                r = lmod.oscar_middle_op_v3(
                    q_t,
                    carr["Kc"],
                    carr["Vc"],
                    carr["bt"],
                    carr["kS"],
                    carr["kO"],
                    carr["vS"],
                    carr["vO"],
                    o_m,
                    l_m,
                    carr["asq"],
                    carr["askv"],
                    nq_mha,
                    scale,
                    nkv,
                    bs,
                    False,
                )
                torch.npu.synchronize()
                o = o_m[:bp].detach().float().cpu().clone()
                lse = l_m[:bp].detach().float().cpu().clone()
                norms = o.reshape(bp, nq_mha, hd).norm(dim=-1)
                import json as _j216m

                with open(f"{d}/relay.jsonl", "a") as _f:
                    _f.write(
                        _j216m.dumps(
                            {
                                "idx": idx,
                                "tag": tag,
                                "rc": int(r[0]),
                                "lrc": int(r[2]),
                                "width": int(o.shape[0]),
                                "zero_out_of": f"{int((norms < 1e-6).sum())}/{bp * nq_mha}",
                                "out_absmax": float(o.abs().max()),
                                "lse_min": float(lse.min()),
                                "lse_max": float(lse.max()),
                                "lse0_6": [round(float(x), 2) for x in lse[0, :6].flatten().tolist()],
                            }
                        )
                        + "\n"
                    )
                torch.save({"idx": idx, "tag": tag, "out": o, "lse": lse}, f"{d}/relayout_{idx:04d}_{tag}.pt")

            def _relay(tag, align=False, deadlive=False, rows=False, askv_set=None):
                Bw = bp if rows else carr["qrot"].shape[0]
                o_t = carr["out"][:Bw]
                l_t = carr["lse"][:Bw]
                o_t[:].zero_()
                l_t[:].fill_(-9999.0)
                if align:
                    carr["askv"][:bp] = cap
                if deadlive:
                    carr["askv"][bp:] = carr["askv"][bp - 1] if bp > 0 else 1
                if askv_set is not None:
                    carr["askv"][:bp] = askv_set
                _t216r.npu.current_stream().synchronize()
                if rows:
                    r = lmod.oscar_middle_op_v3(
                        carr["qrot"][:bp],
                        carr["Kc"][: bp * int(nbC)],
                        carr["Vc"][: bp * int(nbC)],
                        carr["bt"][:bp],
                        carr["kS"][:bp],
                        carr["kO"][:bp],
                        carr["vS"][:bp],
                        carr["vO"][:bp],
                        o_t,
                        l_t,
                        carr["asq"][:bp],
                        carr["askv"][:bp],
                        nq,
                        scale,
                        nkv,
                        bs,
                        False,
                    )
                else:
                    r = lmod.oscar_middle_op_v3(
                        carr["qrot"],
                        carr["Kc"],
                        carr["Vc"],
                        carr["bt"],
                        carr["kS"],
                        carr["kO"],
                        carr["vS"],
                        carr["vO"],
                        carr["out"],
                        carr["lse"],
                        carr["asq"],
                        carr["askv"],
                        nq,
                        scale,
                        nkv,
                        bs,
                        False,
                    )
                _t216r.npu.synchronize()
                o = o_t.detach().float().cpu().clone()
                lse = l_t.detach().float().cpu().clone()
                norms = o[:bp].reshape(bp, nq, hd).norm(dim=-1)
                nzw = int((norms < 1e-6).sum())
                import json as _j216r

                with open(f"{d}/relay.jsonl", "a") as _f:
                    _f.write(
                        _j216r.dumps(
                            {
                                "idx": idx,
                                "tag": tag,
                                "rc": int(r[0]),
                                "lrc": int(r[2]),
                                "width": int(o.shape[0]),
                                "zero_out_of": f"{nzw}/{o.shape[0] * nq}",
                                "out_absmax": float(o.abs().max()),
                                "lse_min": float(lse.min()),
                                "lse_max": float(lse.max()),
                                "lse0_6": [round(float(x), 2) for x in lse[0, :6].flatten().tolist()],
                            }
                        )
                        + "\n"
                    )
                _t216r.save({"idx": idx, "tag": tag, "out": o, "lse": lse}, f"{d}/relayout_{idx:04d}_{tag}.pt")

            _arms = env.get("R216_RELAY_ARMS", "plain,mha8,bladder")
            try:
                if "plain" in _arms:
                    _relay("plain")
                if "rep" in _arms:
                    # R218 KERNEL-REPEATABILITY arm (DIAGNOSTIC_ONLY): two extra
                    # launches on the very same operands. If repA != repB the op
                    # itself is run-to-run nondeterministic; if plain==repA==repB
                    # but out_h differs, the product's in-launch path (replay
                    # context) is the carrier; if all equal, the jitter is
                    # upstream in the carrier OPERANDS (host-side state).
                    _relay("repA")
                    _relay("repB")
                if "aligned" in _arms:
                    _relay("aligned", align=True)
                if "deadlive" in _arms:
                    _relay("deadlive", deadlive=True)
                if "rows" in _arms:
                    _relay("rows", rows=True)
                if "uniform" in _arms:
                    _relay("uniform", align=True, deadlive=True)
                # tile-count bisect: S<=sInner -> tileCount==1 -> R61 gate FAILS
                # -> BASE path. S=2 windows -> grouped. The out-completeness
                # difference pins the bug to the grouped pipeline.
                if "s" in _arms:
                    _relay("s1024", askv_set=1024)
                    _relay("s2048", askv_set=2048)
                    _relay("s3072", askv_set=3072)
                if "mha8" in _arms:
                    _relay_mha("mha8", 8)
                if "gqa" in _arms:
                    _relay_mha("gqa16", 16)
                    _relay_mha("gqa24", 24)
                # R216 B-width ladder on the FULL 32-head GQA form: B=2 barely
                # passes the R61 gate (16 groups / 40 cores, minimal pipeline
                # interaction); wider = deeper inter-group pipelining.
                if "bladder" in _arms:
                    for _bw in (2, 4, 8, 16):
                        if _bw <= bp:
                            _bqw = carr["qrot"][:_bw].contiguous()
                            _bko = torch.zeros_like(carr["out"][:_bw])
                            _bkl = torch.full((_bw, nq, 1), -9999.0, dtype=torch.float32, device=_bqw.device)
                            _bka = carr["askv"][:_bw].contiguous()
                            _bks = carr["asq"][:_bw].contiguous()
                            _rbl = lmod.oscar_middle_op_v3(
                                _bqw,
                                carr["Kc"][: _bw * int(nbC)],
                                carr["Vc"][: _bw * int(nbC)],
                                carr["bt"][:_bw],
                                carr["kS"][:_bw],
                                carr["kO"][:_bw],
                                carr["vS"][:_bw],
                                carr["vO"][:_bw],
                                _bko,
                                _bkl,
                                _bks,
                                _bka,
                                nq,
                                scale,
                                nkv,
                                bs,
                                False,
                            )
                            torch.npu.synchronize()
                            torch.save(
                                {"idx": idx, "tag": f"b{_bw}", "out": _bko.float().cpu(), "lse": _bkl.float().cpu()},
                                f"{d}/relayout_{idx:04d}_b{_bw}.pt",
                            )
            finally:
                carr["out"].copy_(_o0)
                carr["lse"].copy_(_l0)
                carr["askv"].copy_(_a0)
                _t216r.npu.synchronize()
        st["pending"] = {"idx": idx, "layer": int(lord), "bp": int(bp), "bidx": [int(x) for x in cb]}
        st["n"] = idx + 1
    except Exception:
        pass


def _r216_merge(l_pr, l_h, o_pr, o_h, bte):
    try:
        on, env = _r216_env()
        if not on:
            return
        st = _R216_STATE
        pend = st.get("pending")
        st["pending"] = None
        if pend is None or int(l_h.shape[0]) != int(pend.get("bp", -1)):
            return
        # save ONLY for the full-dump set (a per-call save would perturb the leg)
        idx = int(pend["idx"])
        skip = int(env.get("R216_LSEDUMP_SKIP", "0"))
        stride = max(int(env.get("R216_LSEDUMP_STRIDE", "1")), 1)
        maxf = int(env.get("R216_LSEDUMP_MAX", "4"))
        if not (skip <= idx < skip + stride * maxf and (idx - skip) % stride == 0):
            return
        import os as _o216d

        d = env.get("R216_LSEDUMP_DIR", "/tmp/r216_lsedump")
        _o216d.makedirs(d, exist_ok=True)
        import torch as _t216d

        _t216d.npu.synchronize()
        rec = {
            "idx": pend["idx"],
            "layer": pend["layer"],
            "l_pr": l_pr.detach().float().cpu().clone(),
            "l_h": l_h.detach().float().cpu().clone(),
            "o_pr_amax": float(o_pr.detach().float().abs().max().item()),
            "o_h_amax": float(o_h.detach().float().abs().max().item()),
        }
        # R218 quality diagnostic (default OFF): keep the FULL merge operands plus
        # the exact vectorized merge result for the captured calls. The light amax
        # record cannot distinguish a rotated/misplaced o_pr from a numerically
        # correct one; the tensors + mid_*.pt reference can.
        if env.get("R218_MERGEDUMP", "0") == "1":
            _o_pr_c = o_pr.detach().float().cpu().clone()
            _o_h_c = o_h.detach().float().cpu().clone()
            _m = torch.maximum(l_pr, l_h)
            _wp = torch.exp(l_pr - _m)
            _wh = torch.exp(l_h - _m)
            rec.update(
                {
                    "o_pr": _o_pr_c,
                    "o_h": _o_h_c,
                    "merged": ((_wp.unsqueeze(-1) * _o_pr_c + _wh.unsqueeze(-1) * _o_h_c) / (_wp + _wh).unsqueeze(-1)),
                }
            )
        _t216d.save(rec, f"{d}/merge_{pend['idx']:04d}.pt")
    except Exception:
        pass


def _r203_d2h_install():
    if getattr(torch.Tensor, "_r203_patched", False):
        return
    try:
        import sys as _sys203
        import time as _t203

        _TH = 3e-4
        _st = {"n": 0}
        _o_cpu = torch.Tensor.cpu
        _o_tl = torch.Tensor.tolist
        _o_it = torch.Tensor.item

        def _w(dur, kind, site, shape):
            if _st["n"] >= 6000:
                return
            _st["n"] += 1
            try:
                with open("/tmp/r203_d2h.log", "a") as _f:
                    _f.write("{:.1f}us {} {} {}\n".format(dur * 1e6, kind, site, shape))
            except Exception:
                pass

        def _site():
            fr = _sys203._getframe(2)
            return f"{fr.f_code.co_filename.rsplit('/', 1)[-1]}:{fr.f_lineno}:{fr.f_code.co_name}"

        def _cpu(self, *a, **k):
            _t0 = _t203.perf_counter()
            _r = _o_cpu(self, *a, **k)
            _d = _t203.perf_counter() - _t0
            if _d > _TH and getattr(self, "is_npu", False):
                _w(_d, "cpu", _site(), tuple(self.shape))
            return _r

        def _tl(self, *a, **k):
            _t0 = _t203.perf_counter()
            _r = _o_tl(self, *a, **k)
            _d = _t203.perf_counter() - _t0
            if _d > _TH and getattr(self, "is_npu", False):
                _w(_d, "tolist", _site(), tuple(self.shape))
            return _r

        def _it(self, *a, **k):
            _t0 = _t203.perf_counter()
            _r = _o_it(self, *a, **k)
            _d = _t203.perf_counter() - _t0
            if _d > _TH and getattr(self, "is_npu", False):
                _w(_d, "item", _site(), tuple(self.shape))
            return _r

        torch.Tensor.cpu = _cpu
        torch.Tensor.tolist = _tl
        torch.Tensor.item = _it
        # r203b post-mortem: 2 x (aten::to 0.9-1.3ms + copy_ + lift_fresh +
        # copy_ ~0.35ms) chains per pass-2 layer SURVIVE the cpu/tolist/item
        # census — wrap Tensor.to and the torch.tensor factory to catch them.
        _o_to = torch.Tensor.to
        _o_tensor = torch.tensor

        def _to(self, *a, **k):
            _t0 = _t203.perf_counter()
            _r = _o_to(self, *a, **k)
            _d = _t203.perf_counter() - _t0
            if _d > _TH and getattr(self, "is_npu", False):
                _w(_d, "to", _site(), tuple(self.shape))
            return _r

        def _tensor(*a, **k):
            _t0 = _t203.perf_counter()
            _r = _o_tensor(*a, **k)
            _d = _t203.perf_counter() - _t0
            if _d > _TH and getattr(_r, "is_npu", False):
                _w(_d, "tensor", _site(), tuple(_r.shape))
            return _r

        torch.Tensor.to = _to
        torch.tensor = _tensor
        torch.Tensor._r203_patched = True
    except Exception:
        pass


class _R166SlotAlloc:
    """Free-set slot allocator with owner/generation semantics (R166).

    Invariants (check()):
      - rid2slot and slot2rid are exact inverses (bijection);
      - occupied slots unique;
      - free ∩ occupied == ∅;
      - free ∪ occupied == [0, capacity);
      - every free slot has gen >= 1 and no owner;
    Release requires an owner match (stale/double release fails loud).
    """

    def __init__(self, capacity, reserve0=False):
        self.cap = int(capacity)
        # R205_HPFIT reserve0: slot 0 is the NEVER-ASSIGNED identity/scratch row
        # the IDFILL v1 fill + R203 width-pad rows target (see _ensure_tt). At
        # full-range ms the pad's slot=row values landed on never-assigned low
        # rows by accident (LIFO assigns high-first); at ms~=max_num_seqs they
        # overlap LIVE slots and the refresh clobbers real rings (r205 h1:
        # 30/32 rows dead by token 2). Reserving 0 makes every identity write
        # idempotent and ownerless at ANY ms.
        self.reserve0 = bool(reserve0)
        self.rid2slot = {}
        self.slot2rid = {}
        self.free = list(range(self.cap - 1, 0 if self.reserve0 else -1, -1))  # LIFO stack, highest first
        self.gen = [0] * self.cap
        self.zeroed = set()
        self.just_acquired = set()
        self.events: deque = deque(maxlen=4096)
        # audit trail (rid, slot, gen, ev); bounded so a long-lived
        # server cannot grow it without bound
        # R168 P2: acquire runs in the forward path while release runs in the
        # runner _update_states (async-scheduling thread) — the multi-step
        # map/stack mutations are NOT GIL-atomic as a group. One lock guards
        # acquire/release/check (observed leak ~0.15 slot/gen pre-lock).
        import threading as _th168

        self._lk = _th168.Lock()

    def acquire(self, rid):
        with self._lk:
            if rid in self.rid2slot:
                # duplicate acquire without an intervening release would
                # overwrite rid2slot and leak the old slot in slot2rid --
                # a caller lifecycle bug; fail closed on it
                raise RuntimeError(
                    f"[R166-SLOT] rid {rid!r} already holds slot "
                    f"{self.rid2slot[rid]} (duplicate acquire without release; "
                    f"release requires explicit finished/preempted)"
                )
            if not self.free:
                raise RuntimeError(
                    f"[R166-SLOT] capacity exhausted ({self.cap}) with no free slot — "
                    f"cannot assign {rid!r} (fail-closed; release requires explicit "
                    f"finished/preempted)"
                )
            slot = self.free.pop()
            reused = self.gen[slot] > 0
            self.gen[slot] += 1
            self.rid2slot[rid] = slot
            self.slot2rid[slot] = rid
            self.just_acquired.add(rid)
            # gen just moved: every stamp for this slot is obsolete
            self.zeroed = {t for t in self.zeroed if t[0] != slot}
            self.events.append((rid, slot, self.gen[slot], "reuse" if reused else "new"))
            return slot, self.gen[slot], reused

    def release(self, rid):
        with self._lk:
            slot = self.rid2slot.pop(rid, None)
            if slot is None:
                self.events.append((rid, -1, -1, "release-unowned(dropped)"))
                return
            owner = self.slot2rid.pop(slot, None)
            if owner != rid:
                # restore and fail loud: slot owned by someone else
                self.slot2rid[slot] = owner
                self.rid2slot[rid] = slot
                raise RuntimeError(
                    f"[R166-SLOT] release owner mismatch: rid={rid!r} maps to slot {slot} but slot owned by {owner!r}"
                )
            self.gen[slot] += 1
            self.free.append(slot)
            # slot freed: its stamps are obsolete (next occupant re-stamps)
            self.zeroed = {t for t in self.zeroed if t[0] != slot}
            self.events.append((rid, slot, self.gen[slot], "release"))

    def gen_of(self, slot):
        return self.gen[slot]

    def check(self):
        with self._lk:
            return self._check_locked()

    def _check_locked(self):
        v = []
        r2s, s2r = self.rid2slot, self.slot2rid
        if len(s2r) != len(r2s):
            v.append(f"map sizes differ {len(r2s)} vs {len(s2r)}")
        for rid, slot in r2s.items():
            if s2r.get(slot) != rid:
                v.append(f"not inverse: rid={rid!r}->slot={slot} owner={s2r.get(slot)!r}")
        occ = set(s2r.keys())
        fr = set(self.free)
        if len(self.free) != len(fr):
            v.append(f"free list has duplicates ({len(self.free)} entries, {len(fr)} unique)")
        if fr & occ:
            v.append("free^occupied nonempty: {}".format(sorted(fr & occ)[:8]))
        _capv = self.cap - (1 if self.reserve0 else 0)
        if len(occ) + len(fr) != _capv:
            v.append(f"cover broken: |occ|+{len(fr)}|free|={len(occ) + len(fr)} != cap {_capv}")
        return v


_R144_TT_DYN: dict[str, Any] = {"fin": 0, "reuse": 0, "new": 0, "log": []}


def _r136_p0_dump(mode, layer_idx, step, scalars, tensors):
    import hashlib as _hl
    import json as _js

    if _os.environ.get("R136_P0_DIAG", "0") != "1":
        return
    try:
        _nohash = _os.environ.get("R136_P0_NOHASH", "0") == "1"
        if not _nohash:
            torch.npu.synchronize()
        rec = {"mode": mode, "layer": int(layer_idx), "step": int(step)}
        rec.update({k: v for k, v in scalars.items()})
        for name, t in tensors.items():
            if t is None:
                rec[name] = None
                continue
            try:
                tc = t.contiguous()
                rec[name + "_shape"] = list(tc.shape)
                rec[name + "_dtype"] = str(tc.dtype).replace("torch.", "")
                if _nohash:
                    rec[name + "_hash"] = "skip"
                    continue
                # byte-view hash works for ANY dtype (bf16/int8/int32) — no numpy
                # ScalarType issue. view(uint8) is a pure reinterpretation.
                rec[name + "_hash"] = _hl.sha256(tc.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
            except Exception as _e:
                rec[name + "_hash"] = "ERR:" + str(_e)[:60]
        _p = _os.environ.get("R136_P0_OUT", "/tmp/r136_p0_diag.jsonl")
        with open(_p, "a") as _f:
            _f.write(_js.dumps(rec) + "\n")
    except Exception as _e:
        print(f"[R136-P0] dump failed: {_e}", flush=True)


_R114_TS_FH = None


def _r156_shared_askv(device, b=0):
    """R156-H1: ONE shared askv scalar per (device, batch-row). Default path
    (R156_H1=0) returns a fresh per-layer tensor (R135 contract); H1 returns
    the shared scalar so the 36 captured V1 launches of batch-row b read the
    SAME address and the updater fills once per step. mid_len is layer-
    invariant per (step, batch-row) but NOT across rows => keyed by b."""
    if _os.environ.get("R156_H1", "0") != "1":
        return torch.zeros(1, dtype=torch.int64, device=device)
    d = globals().setdefault("_R156_ASKV", {})
    k = (str(device), int(b))
    if k not in d:
        d[k] = torch.zeros(1, dtype=torch.int64, device=device)
    return d[k]


def _r156_shared_asq(device, b=0):
    if _os.environ.get("R156_H1", "0") != "1":
        return torch.zeros(1, dtype=torch.int64, device=device)
    d = globals().setdefault("_R156_ASQ", {})
    k = (str(device), int(b))
    if k not in d:
        d[k] = torch.zeros(1, dtype=torch.int64, device=device)
    return d[k]


def _r114_ts_write():
    global _R114_TS_FH
    if _R114_TS_FH is None:
        import json as _json

        _R114_TS_FH = open(os.environ.get("R114_STEADY_OUT", "/tmp/r114_steady.jsonl"), "a", buffering=1)  # noqa: SIM115 - persistent append handle by design
    import json as _json
    import time as _time

    _R114_TS_FH.write(_json.dumps({"t": _time.time_ns(), "pid": _os.getpid()}) + "\n")


def _oscar_dbg(*args, **kwargs):
    # gated debug print for INT2 paths; no-op unless SGLANG_OSCAR_DEBUG set
    import os as _os

    if _os.environ.get("SGLANG_OSCAR_DEBUG"):
        print(*args, **kwargs)


def _oscar_diag_write(relname, text):
    """Gated diagnostic append (see vllm_ascend/oscar/diag.py)."""
    # local tree module; isolated single-file runs resolve the installed pkg instead
    from vllm_ascend.oscar.diag import write  # type: ignore[import-untyped]

    write(relname, text)


from vllm_ascend.oscar.rotation import (  # noqa: E402,F401 - bottom-of-module re-export is the hub contract
    load_oscar_rotations,  # noqa: F401,E402 - bottom-of-module re-export is the hub contract
)
