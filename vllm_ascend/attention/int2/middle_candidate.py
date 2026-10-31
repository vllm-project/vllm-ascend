# SPDX-License-Identifier: Apache-2.0
"""Middle operand-assembly mixin: the R109 LSE sibling-candidate
preparation (per-row middle window operands for the executor)."""

from __future__ import annotations

from typing import Any

import torch
import torch_npu  # noqa: F401 - registers torch.npu

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.int2.ctx import (  # noqa: F401
    _OSCAR_CFG,
    _R2XX_G,
    _odiag_o,
    _odiag_p,
)
from vllm_ascend.attention.int2.helpers import (  # noqa: F401
    _r136_p0_dump,
    _r218_diag,
)


class Int2MiddleCandidateMixin:
    # Host-state annotations: the per-class pools/latches/counters below are
    # owned by this mixin and initialized lazily through type(self) (defaults
    # are required there); the cache-interface, rotation, TT-window, and
    # bridge-loader names are owned by the assembled backend / sibling mixins.
    # Bare annotations create no runtime attributes, so the MRO at assembly
    # is unaffected.
    _r112_opool: Any
    _r118_ucpool: Any
    _r118_ubtpool: Any
    _r185b_dumped: Any = None
    _r112_pool: Any
    _r113_cpool: Any
    _r113_btpool: Any
    _r135_adev: Any
    _r112_mod: Any = None
    _r135_mod: Any = None
    _r91_mod: Any = None
    _oscar_R_k: Any
    _oscar_layer_ord: Any
    _tt_hp_bufs: Any
    _r91_ensure_loaded: Any
    _r112_ensure_fast: Any
    _r135_ensure_v1: Any
    vllm_config: Any
    _TT_RECENT: int
    head_size: int
    num_heads: int
    num_kv_heads: int
    key_cache: Any
    value_cache: Any
    k_scale_cache: Any
    k_zero_cache: Any
    v_scale_cache: Any
    v_zero_cache: Any

    def _r109_middle_candidate(self, query, attn_metadata, layer, b, mid_start, mid_end, q_rot_pre=None):
        """Run the LSE sibling candidate over ONLY the INT2 middle token range
        [mid_start, mid_end) of sequence b, reading the persistent paged INT2
        cache. Returns (O_H [nq, hd] bf16 rotated-basis, L_H [nq] fp32) or None.
        Requires mid_start % block_size == 0 (pref=64 with bs=16 holds)."""
        import math as _m
        import os as _os

        _st = _R2XX_G.setdefault("_R109_STATS", {"middle_calls": 0, "middle_ok": 0, "merge": 0, "full": 0})
        _r218_mc = _R2XX_G.setdefault("_R218_MIDC_N", [0])
        if _r218_mc[0] < 3:
            _r218_mc[0] += 1
            _r218_diag(
                f"MIDC-ENTER #{_r218_mc[0]} lay={getattr(layer, 'layer_idx', -1)} "
                f"mid=[{mid_start},{mid_end}) Rk={self._oscar_R_k is not None} "
                f"R113cpp={_os.environ.get('R113_MIDDLE_CPP', '0')} "
                f"R112fast={_os.environ.get('R112_LSE_FAST', '0')}"
            )
        if not self._r91_ensure_loaded(query.device):
            if _st["middle_calls"] == 0:
                print("[R109-MID-DBG] ensure_loaded FAILED (first call)", flush=True)
            return None
        if _st["middle_calls"] < 3:
            print(
                f"[R109-MID-DBG] enter layer={getattr(layer, 'layer_idx', -1)} "
                f"mid=[{mid_start},{mid_end}) len={mid_end - mid_start}",
                flush=True,
            )
        if mid_end <= mid_start:
            if _st["middle_calls"] < 3:
                print(f"[R109-MID-RET] len gate mid=[{mid_start},{mid_end})", flush=True)
            return None
        block_size = self.vllm_config.cache_config.block_size
        if mid_start % block_size != 0:
            if _st["middle_calls"] < 3:
                print(f"[R109-MID-RET] bs gate bs={block_size} mid_start={mid_start}", flush=True)
            return None
        _CP = type(self).__name__ + ":" + str(id(type(self).__dict__.get("_r109_middle_candidate", None)))
        if _st["middle_calls"] < 3:
            print(f"[R109-MID-CP1] past gates bs={block_size} cls={_CP}", flush=True)
        num_block, _, num_kv_heads, packed_head = self.key_cache.shape
        head_size = self.head_size
        nq = self.num_heads
        nkv = self.num_kv_heads
        layer_idx = getattr(layer, "layer_idx", 0)
        device = query.device

        block_table = attn_metadata.block_tables
        blk_lo = mid_start // block_size
        blk_hi = (mid_end + block_size - 1) // block_size
        # ---- R113 Fallback-1: single-call C++ middle prep + launch (bit-exact:
        # same ATen op sequence inside C++, ONE pybind crossing instead of ~35
        # Python dispatches). Env-gated, default OFF; requires fast launcher.
        if _os.environ.get("R113_MIDDLE_CPP", "0") == "1" and _os.environ.get("R112_LSE_FAST", "0") == "1":
            if not self._r112_ensure_fast(query.device):
                raise RuntimeError("[R113-F1] fast module load FAILED (fail-closed)")
            _nb = blk_hi - blk_lo
            _mid_len = int(mid_end - mid_start)
            if getattr(type(self), "_r112_opool", None) is None:
                type(self)._r112_opool = {}
            _pool2 = type(self)._r112_opool
            _ok = (int(nq), int(head_size), str(device))
            if _ok not in _pool2:
                # R116: unify the shared pool on the 3-tuple shape (the graph
                # path stores (out, lse, empty_mask); a 2-tuple from this eager
                # fast path crashed the later graph unpack — R113 latent bug).
                _pool2[_ok] = (
                    torch.zeros(1, nq, 1, head_size, dtype=torch.bfloat16, device=device),
                    torch.zeros(1, nq, 1, dtype=torch.float32, device=device),
                    torch.zeros(0, dtype=torch.uint8, device=device),
                )
            out_h, lse_h, _ = _pool2[_ok]
            if q_rot_pre is not None and _os.environ.get("R113_QROT_DEDUPE", "0") == "1":
                query_bnsd = q_rot_pre.unsqueeze(2) if q_rot_pre.dim() == 3 else q_rot_pre
            else:
                _Rk = None
                if self._oscar_R_k is not None and self._oscar_layer_ord(layer) < self._oscar_R_k.shape[0]:
                    _Rk = self._oscar_R_k[self._oscar_layer_ord(layer)]
                query_bnsd = query[b : b + 1]
                if query_bnsd.dim() == 3:
                    query_bnsd = query_bnsd.unsqueeze(2)
                if _Rk is not None:
                    query_bnsd = torch.matmul(query_bnsd, _Rk)
            _bt_pre = block_table[b, blk_lo : blk_lo + _nb]
            if (
                _os.environ.get("R118_GRAPH_INPUT_FIX", "0") == "1"
                and not _EXTRA_CTX.capturing
                and int(_bt_pre.shape[0]) == int(_nb)
            ):
                # R118 unified middle numerics (default OFF): stage the middle
                # carriers with the EXACT op sequence the graph replay updater
                # uses (_r113_update_oscar_mid) and launch via launch_lse_fast
                # with the same carrier shapes/args. This removes the eager-vs-
                # graph middle-op numeric divergence (oh/lh differ every step ->
                # logits noise -> greedy argmax flips at low-margin steps, the
                # mini idx-8 root cause). Perf: Python staging is slower than
                # the C++ prep — correctness legs only, default OFF.
                _rcnt_l = int(self._TT_RECENT)
                _pref_l = _OSCAR_CFG.prefix_window
                _mlen_l = int(getattr(self.vllm_config.model_config, "max_model_len", 2048))
                _nbm = (_mlen_l - _pref_l - _rcnt_l + block_size - 1) // block_size
                if _nbm <= 0:
                    raise RuntimeError("[R118-FIX] nb_max<=0 (config)")
                if getattr(type(self), "_r118_ucpool", None) is None:
                    type(self)._r118_ucpool = {}
                _cp_l = type(self)._r118_ucpool
                if layer_idx not in _cp_l:
                    _cp_l[layer_idx] = {
                        "nb_max": _nbm,
                        "k": torch.zeros(
                            _nbm, nkv, block_size, self.key_cache.shape[-1], dtype=torch.int8, device=device
                        ),
                        "v": torch.zeros(
                            _nbm, nkv, block_size, self.key_cache.shape[-1], dtype=torch.int8, device=device
                        ),
                        "ks": torch.zeros(1, nkv, _nbm * block_size, dtype=torch.bfloat16, device=device),
                        "ko": torch.zeros(1, nkv, _nbm * block_size, dtype=torch.bfloat16, device=device),
                        "vs": torch.zeros(1, nkv, _nbm * block_size, dtype=torch.bfloat16, device=device),
                        "vo": torch.zeros(1, nkv, _nbm * block_size, dtype=torch.bfloat16, device=device),
                    }
                _c_l = _cp_l[layer_idx]
                bt_gather = _bt_pre
                _nb_eff = int(_nb)
                _mid_eff = min(int(_mid_len), _nb_eff * block_size)
                _c_l["k"][:_nb_eff] = self.key_cache[bt_gather].permute(0, 2, 1, 3).contiguous().view(torch.int8)
                _c_l["v"][:_nb_eff] = self.value_cache[bt_gather].permute(0, 2, 1, 3).contiguous().view(torch.int8)
                _sn = _nb_eff * block_size
                ksb = self.k_scale_cache[bt_gather].reshape(_sn, nkv)
                kzb = self.k_zero_cache[bt_gather].reshape(_sn, nkv)
                vsb = self.v_scale_cache[bt_gather].reshape(_sn, nkv)
                vzb = self.v_zero_cache[bt_gather].reshape(_sn, nkv)
                _c_l["ks"][0, :, :_mid_eff] = ksb[:_mid_eff].t().bfloat16()
                _c_l["ko"][0, :, :_mid_eff] = (-kzb[:_mid_eff].t() * _c_l["ks"][0, :, :_mid_eff]).bfloat16()
                _c_l["vs"][0, :, :_mid_eff] = vsb[:_mid_eff].t().bfloat16()
                _c_l["vo"][0, :, :_mid_eff] = (-vzb[:_mid_eff].t() * _c_l["vs"][0, :, :_mid_eff]).bfloat16()
                if getattr(type(self), "_r118_ubtpool", None) is None:
                    type(self)._r118_ubtpool = {}
                _bk_l = (int(nq), int(head_size), _nbm)
                _bp_l = type(self)._r118_ubtpool
                if _bk_l not in _bp_l:
                    _bp_l[_bk_l] = torch.arange(_nbm, dtype=torch.int32, device=device).view(1, -1).contiguous()
                _empty = torch.zeros(0, dtype=torch.uint8, device=device)
                # R185 OPERAND SNAPSHOT (env R185_OPDUMP): capture the EXACT device tensors entering
                # launch_lse_fast at the LSE input-consumption point (the crash-path call site inside
                # _r109_middle_candidate), for standalone MSD_ON replay. Full sync first; read-only;
                # single-shot (first middle-tier call = crash call). bt here is identity arange; the
                # real page indices (bt_gather) are captured for the record.
                import os as _os185b

                if _os185b.environ.get("R185_OPDUMP") == "1" and not getattr(type(self), "_r185b_dumped", False):
                    type(self)._r185b_dumped = True
                    torch.npu.synchronize()
                    _od185b = _os185b.environ.get("R185_OPDUMP_DIR", "/tmp/r185")
                    _os185b.makedirs(_od185b, exist_ok=True)
                    _sc185 = 1.0 / _m.sqrt(head_size)
                    _snap = {
                        "q": query_bnsd.detach().cpu().clone(),
                        "k": _c_l["k"][:_nb_eff].detach().cpu().clone(),
                        "v": _c_l["v"][:_nb_eff].detach().cpu().clone(),
                        "bt": _bp_l[_bk_l].detach().cpu().clone(),
                        "ks": _c_l["ks"][0, :, :_mid_eff].detach().cpu().clone(),
                        "ko": _c_l["ko"][0, :, :_mid_eff].detach().cpu().clone(),
                        "vs": _c_l["vs"][0, :, :_mid_eff].detach().cpu().clone(),
                        "vo": _c_l["vo"][0, :, :_mid_eff].detach().cpu().clone(),
                        "out_h": out_h.detach().cpu().clone(),
                        "lse_h": lse_h.detach().cpu().clone(),
                        "askv": [int(_mid_eff)],
                        "nq": int(nq),
                        "scale": float(_sc185),
                        "nkv": int(nkv),
                        "layout": "BNSD",
                        "block_size": int(block_size),
                        "layer_idx": int(layer_idx),
                        "b": int(b),
                        "mid_start": int(mid_start),
                        "mid_end": int(mid_end),
                        "mid_len": int(_mid_len),
                        "mid_eff": int(_mid_eff),
                        "nb_eff": int(_nb_eff),
                        "nbm": int(_nbm),
                        "head_size": int(head_size),
                        "bt_gather": bt_gather.detach().cpu().clone(),
                        "q_shape": list(query_bnsd.shape),
                        "q_stride": list(query_bnsd.stride()),
                        "k_shape": list(_c_l["k"][:_nb_eff].shape),
                        "k_stride": list(_c_l["k"][:_nb_eff].stride()),
                        "ks_shape": list(_c_l["ks"][0, :, :_mid_eff].shape),
                        "ks_stride": list(_c_l["ks"][0, :, :_mid_eff].stride()),
                    }
                    _p185 = f"{_od185b}/r185_opdump_l{layer_idx}_b{b}_ms{mid_start}.pt"
                    torch.save(_snap, _p185)
                    print(
                        f"[R185-OPDUMP] l={layer_idx} b={b} mid=[{mid_start},{mid_end}] mid_eff={_mid_eff} "
                        f"nb_eff={_nb_eff} askv=[{_mid_eff}] bt(identity) bt_gather_min={int(bt_gather.min())} "
                        f"bt_gather_max={int(bt_gather.max())} "
                        f"ks_absmax={float(_c_l['ks'][0, :, :_mid_eff].abs().max())} "
                        f"ko_absmax={float(_c_l['ko'][0, :, :_mid_eff].abs().max())} "
                        f"k_int8_absmax={int(_c_l['k'][:_nb_eff].abs().max())} "
                        f"-> {_p185}",
                        flush=True,
                    )
                fr = type(self)._r112_mod.launch_lse_fast(
                    query_bnsd,
                    _c_l["k"],
                    _c_l["v"],
                    _bp_l[_bk_l],
                    _c_l["ks"],
                    _c_l["ko"],
                    _c_l["vs"],
                    _c_l["vo"],
                    out_h,
                    lse_h,
                    nq,
                    1.0 / _m.sqrt(head_size),
                    nkv,
                    "BNSD",
                    block_size,
                    1,
                    1,
                    [int(_mid_eff)],
                    _empty,
                    0,
                    0,
                    0,
                    True,
                )
            else:
                fr = type(self)._r112_mod.oscar_middle_prep(
                    block_table,
                    int(b),
                    int(blk_lo),
                    int(_nb),
                    int(_mid_len),
                    self.key_cache,
                    self.value_cache,
                    self.k_scale_cache,
                    self.k_zero_cache,
                    self.v_scale_cache,
                    self.v_zero_cache,
                    query_bnsd,
                    out_h,
                    lse_h,
                    nq,
                    1.0 / _m.sqrt(head_size),
                    nkv,
                    block_size,
                )
            # R117 P0-B diag (env-gated, default OFF): eager-side mirror of the
            # graph updater dump — same source hashes per step for the ladder.
            if _os.environ.get("R117_MID_DIAG", "0") == "1":
                try:
                    import hashlib as _hl
                    import json as _js

                    torch.npu.synchronize()

                    def _h(t):
                        return _hl.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:12]

                    _cb = (mid_end + 511) // 128  # current (partial) middle block
                    _btg = block_table[b, blk_lo : blk_lo + _nb]
                    _hk, _hv = self._tt_hp_bufs(self._oscar_layer_ord(layer), query.device, query.dtype)
                    _rec = {
                        "mode": "eager",
                        "layer": int(layer_idx),
                        "cnt": globals().setdefault("_R117_MID_E_N", [0])[0],
                        "mid": [int(mid_start), int(mid_end)],
                        "askv": int(_mid_len),
                        "nb_now": int(_nb),
                        "bt": _btg.cpu().tolist(),
                        "src_k": _h(self.key_cache[_btg]),
                        "src_v": _h(self.value_cache[_btg]),
                        "q": _h(query_bnsd),
                        "curblk_k": _h(self.key_cache[block_table[b, _cb] : block_table[b, _cb] + 1]),
                        "slotmap": _h(attn_metadata.slot_mapping.view(torch.uint8))
                        if attn_metadata.slot_mapping is not None
                        else "none",
                        "src_ks": _h(self.k_scale_cache[_btg]),
                        "ring_k": _h(_hk[b]),
                        "ring_v": _h(_hv[b]),
                        "rk_pre": _h(_hk[b, :128]),
                        "rk_r0": _h(_hk[b, 128:320]),
                        "rk_r1": _h(_hk[b, 320:512]),
                        "rk_r2": _h(_hk[b, 512:624]),
                        "rk_tail": _h(_hk[b, 624:640]),
                        "rv_tail": _h(_hv[b, 624:640]),
                        "rk_slots": "|".join(_h(_hk[b, s : s + 1]) for s in range(624, 640)),
                        "rv_slots": "|".join(_h(_hv[b, s : s + 1]) for s in range(624, 640)),
                    }
                    globals()["_R117_MID_E_N"][0] += 1
                    _p = _os.environ.get("R117_MID_E_DIAG_OUT", "/tmp/r117_mid_e_diag.jsonl")
                    if _os.environ.get("R118_RAW", "0") == "1" and _rec["cnt"] <= 12:
                        _rec["raw_tail_k"] = _hk[b, 639].float().tolist()
                        _rec["raw_tail_v"] = _hv[b, 639].float().tolist()
                        _rec["raw_q0"] = (
                            query_bnsd[0, 0, :8].float().tolist()
                            if query_bnsd.dim() == 3
                            else query_bnsd[0, :8].float().tolist()
                        )
                    with open(_p, "a") as _f:
                        _f.write(_js.dumps(_rec) + "\n")
                except Exception as _e:
                    print(f"[R117-MID-DIAG] eager diag failed: {_e}", flush=True)
            _st["middle_calls"] += 1
            if not (int(fr[0]) == 0 and int(fr[2]) == 0):
                raise RuntimeError(f"[R113-F1] middle_prep NOT ok: {tuple(int(x) for x in fr)}")
            _st["middle_ok"] += 1
            # R218 MIDCAP2: one-shot capture around the F1 C++ middle call for
            # offline fp32 adjudication of the ACTUALLY-executed eager middle.
            _r218m2 = globals().setdefault("_R218_MIDCAP2", [0])
            if (
                _r218m2[0] < 2 and _os.environ.get("R216_LSEDUMP", "0") == "1" and int(mid_start) >= 128
            ):  # real middle only; skip warmup dummies
                _r218m2[0] += 1
                try:
                    _btg2 = block_table[b, blk_lo:blk_hi]
                    torch.npu.synchronize()
                    torch.save(
                        {
                            "idx": 9100 + _r218m2[0],
                            "layer": int(self._oscar_layer_ord(layer)),
                            "bp": 1,
                            "nbC": int(_nb),
                            "bidx": [int(b)],
                            "replay": False,
                            "nq": int(nq),
                            "nkv": int(nkv),
                            "bs": int(block_size),
                            "hd": int(head_size),
                            "scale": float(1.0 / _m.sqrt(head_size)),
                            "askv": [int(_mid_len)],
                            "q": query_bnsd.detach().cpu().clone(),
                            "k": self.key_cache[_btg2].detach().cpu().clone(),
                            "v": self.value_cache[_btg2].detach().cpu().clone(),
                            "ks": self.k_scale_cache[_btg2].detach().cpu().clone(),
                            "kz": self.k_zero_cache[_btg2].detach().cpu().clone(),
                            "vs": self.v_scale_cache[_btg2].detach().cpu().clone(),
                            "vz": self.v_zero_cache[_btg2].detach().cpu().clone(),
                            "btrow": _btg2.detach().cpu().clone(),
                            "out_h": out_h.detach().cpu().clone(),
                            "lse_h": lse_h.detach().cpu().clone(),
                            "mid": [int(mid_start), int(mid_end)],
                        },
                        f"/tmp/r218_midcap2_{_r218m2[0]}.pt",
                    )
                    _r218_diag(
                        f"MIDCAP2 saved #{_r218m2[0]} lay={int(self._oscar_layer_ord(layer))} "
                        f"nb={int(_nb)} midlen={int(_mid_len)}"
                    )
                except Exception as _eM2:
                    _r218_diag(f"MIDCAP2 ERR {_eM2!r}")
            if _os.environ.get("R117_MID_DIAG", "0") == "1":
                try:
                    import hashlib as _hl2
                    import json as _js2

                    torch.npu.synchronize()
                    _oh = _hl2.sha256(out_h.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:12]
                    _lh = _hl2.sha256(lse_h.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:12]
                    _l = _js2.dumps(
                        {
                            "mode": "eager_oh",
                            "cnt": _st["middle_calls"] - 1,
                            "oh": _oh,
                            "lh": _lh,
                            "mid": [int(mid_start), int(mid_end)],
                        }
                    )
                    with open(_os.environ.get("R117_MID_E_DIAG_OUT", "/tmp/r117_mid_e_diag.jsonl"), "a") as _f2:
                        _f2.write(_l + chr(10))
                except Exception:
                    pass
            return out_h[0, :, 0, :], lse_h[0, :, 0]
        # R110 D6a FIX: op indexes the packed K/V GM by PHYSICAL block id from bt;
        # the gather compacts blocks to [0..n), so the LAUNCH bt must be identity
        # arange while the GATHER keeps the ORIGINAL physical ids.
        bt_gather = block_table[b, blk_lo:blk_hi]
        # R112_BUFPOOL: identity bt is a pure function of (nb) — cache it (stream-safe:
        # read-only operand). Keyed on device too.
        if _os.environ.get("R112_BUFPOOL", "0") == "1":
            if getattr(type(self), "_r112_pool", None) is None:
                type(self)._r112_pool = {}
            _pool = type(self)._r112_pool
            _bk = (int(blk_hi - blk_lo), str(device))
            if _bk not in _pool:
                _pool[_bk] = (
                    torch.arange(blk_hi - blk_lo, dtype=torch.int32, device=block_table.device).view(1, -1).contiguous()
                )
            bt = _pool[_bk]
        else:
            bt = torch.arange(blk_hi - blk_lo, dtype=torch.int32, device=block_table.device).view(1, -1).contiguous()

        k_packed = self.key_cache[bt_gather]
        v_packed = self.value_cache[bt_gather]
        k_r91 = k_packed.permute(0, 2, 1, 3).contiguous().view(torch.int8)
        v_r91 = v_packed.permute(0, 2, 1, 3).contiguous().view(torch.int8)
        k_scale_blk = self.k_scale_cache[bt_gather]
        k_zero_blk = self.k_zero_cache[bt_gather]
        v_scale_blk = self.v_scale_cache[bt_gather]
        v_zero_blk = self.v_zero_cache[bt_gather]
        S = (blk_hi - blk_lo) * block_size
        mid_len = int(mid_end - mid_start)  # R110 D6b: kernel strides scale rows by max(askv)=mid_len
        k_scale_flat = k_scale_blk.reshape(S, nkv)[:mid_len].t().unsqueeze(0).bfloat16().contiguous()
        k_zero_flat = k_zero_blk.reshape(S, nkv)[:mid_len].t().unsqueeze(0).bfloat16().contiguous()
        k_offset_flat = (-k_zero_flat * k_scale_flat).contiguous()
        v_scale_flat = v_scale_blk.reshape(S, nkv)[:mid_len].t().unsqueeze(0).bfloat16().contiguous()
        v_zero_flat = v_zero_blk.reshape(S, nkv)[:mid_len].t().unsqueeze(0).bfloat16().contiguous()
        v_offset_flat = (-v_zero_flat * v_scale_flat).contiguous()
        askv_list = [int(mid_end - mid_start)]

        R_k = None
        if self._oscar_R_k is not None and self._oscar_layer_ord(layer) < self._oscar_R_k.shape[0]:
            R_k = self._oscar_R_k[self._oscar_layer_ord(layer)]
        query_bnsd = query[b : b + 1]
        if query_bnsd.dim() == 3:
            query_bnsd = query_bnsd.unsqueeze(2)
        if R_k is not None:
            query_bnsd = torch.matmul(query_bnsd, R_k)

        if _st["middle_calls"] < 3:
            print(f"[R109-MID-CP2] pre-launch askv={askv_list} bt={bt.shape} k={k_r91.shape}", flush=True)
        # R109 S2 operand capture: dump ALL middle operands on the FIRST call only
        # R112: skipped in fast mode EXCEPT when R112_CAPT=1 (correctness legs)
        if _os.environ.get("R112_LSE_FAST", "0") != "1" or _os.environ.get("R112_CAPT", "0") == "1":
            try:
                _dump_p = _odiag_p("r111_s02_operands.pt")
                if not _os.path.exists(_dump_p):
                    import torch as _t

                    _t.save(
                        {
                            "k_r91": k_r91.cpu(),
                            "v_r91": v_r91.cpu(),
                            "k_scale": k_scale_flat.cpu(),
                            "k_offset": k_offset_flat.cpu(),
                            "v_scale": v_scale_flat.cpu(),
                            "v_offset": v_offset_flat.cpu(),
                            "bt": bt.cpu(),
                            "askv": askv_list,
                            "q_rot": query_bnsd.detach().cpu(),
                            "nq": nq,
                            "nkv": nkv,
                            "bs": block_size,
                            "scale": 1.0 / _m.sqrt(head_size),
                        },
                        _dump_p,
                    )
                    print(f"[R109-S2] operands dumped askv={askv_list}", flush=True)
            except Exception as _e:
                print(f"[R109-S2] dump failed: {_e}", flush=True)
            # R111: bind S2 operands of THIS run to the rich S3-S7 capture (first call
            # = layer 0 step 0 of decode, the same invocation the capture fires on)
            try:
                if _os.environ.get("R111_S2DUMP", "1") == "1":
                    _dump2 = _odiag_p("r111_s02_operands.pt")
                    if not _os.path.exists(_dump2):
                        import torch as _t2

                        _t2.save(
                            {
                                "k_r91": k_r91.cpu(),
                                "v_r91": v_r91.cpu(),
                                "k_scale": k_scale_flat.cpu(),
                                "k_offset": k_offset_flat.cpu(),
                                "v_scale": v_scale_flat.cpu(),
                                "v_offset": v_offset_flat.cpu(),
                                "bt": bt.cpu(),
                                "askv": askv_list,
                                "q_rot": query_bnsd.detach().cpu(),
                                "layer": layer_idx,
                                "nq": nq,
                                "nkv": nkv,
                                "bs": block_size,
                                "scale": 1.0 / _m.sqrt(head_size),
                            },
                            _dump2,
                        )
                        print(f"[R111-S2] operands dumped L{layer_idx} askv={askv_list}", flush=True)
            except Exception as _e2:
                print(f"[R111-S2] dump failed: {_e2}", flush=True)

        # R112 output buffer pool (R112_BUFPOOL=1): out/lse/mask keyed (nq,nkv,hd);
        # stream-safe — same-stream in-order guarantees merge (enqueued earlier)
        # reads before the next op's fill_/write.
        if _os.environ.get("R112_BUFPOOL", "0") == "1":
            if getattr(type(self), "_r112_opool", None) is None:
                type(self)._r112_opool = {}
            _pool2 = type(self)._r112_opool
            _ok = (int(nq), int(head_size), str(device))
            if _ok not in _pool2:
                _pool2[_ok] = (
                    torch.zeros(1, nq, 1, head_size, dtype=torch.bfloat16, device=device),
                    torch.zeros(1, nq, 1, dtype=torch.float32, device=device),
                    torch.zeros(0, dtype=torch.uint8, device=device),
                )
            out_h, lse_h, empty_mask = _pool2[_ok]
        else:
            out_h = torch.zeros(1, nq, 1, head_size, dtype=torch.bfloat16, device=device)
            lse_h = torch.zeros(1, nq, 1, dtype=torch.float32, device=device)
            empty_mask = torch.zeros(0, dtype=torch.uint8, device=device)

        if _os.environ.get("R112_LSE_FAST", "0") == "1":
            # R112 fast path: sync-free launch, fail-closed on status, no per-call
            # file counter, no readback heuristic (lse_changed/lse_finite replaced
            # by periodic explicit verify below — R111 hardening item).
            if not self._r112_ensure_fast(query.device):
                raise RuntimeError("[R112-MID] fast module load FAILED (fail-closed)")
            if _os.environ.get("R133_GRAPH_V1", "0") == "1":
                if type(self)._r135_mod is None and not self._r135_ensure_v1(device):
                    raise RuntimeError("[R135-MID] V1 binding load FAILED (fail-closed)")
                if _os.environ.get("R136_UNIFIED", "0") == "1":
                    # R136 UNIFIED eager (F_unified): stage the middle operands into
                    # the SAME capacity pool (_r113_cpool) + capacity-shaped bt
                    # (arange(nb_max)) + capacity scales that the V1 GRAPH capture +
                    # replay updater use, then launch_lse_v1_own. This makes eager
                    # and graph pass BYTE-IDENTICAL-SHAPED operands -> identical
                    # kernel tiling -> bit-exact output (closes staging #4: the
                    # tight-vs-capacity operand-shape divergence, P0 root). All
                    # sizes runtime-derived from config/cache-shape (no literals).
                    _nb_now = (int(mid_len) + block_size - 1) // block_size
                    _max_len = int(getattr(self.vllm_config.model_config, "max_model_len", 2048))
                    _rcap = int(self._TT_RECENT)
                    _nb_max = (_max_len - int(mid_start) - _rcap + block_size - 1) // block_size
                    if _nb_max < _nb_now:
                        _nb_max = _nb_now  # safety: capacity >= current need
                    _S_pad = _nb_max * block_size
                    if getattr(type(self), "_r113_cpool", None) is None:
                        type(self)._r113_cpool = {}
                    _cp = type(self)._r113_cpool
                    if layer_idx not in _cp:
                        _cp[layer_idx] = {
                            "nb_max": _nb_max,
                            "S_pad": _S_pad,
                            "k": torch.zeros(
                                _nb_max, nkv, block_size, self.key_cache.shape[-1], dtype=torch.int8, device=device
                            ),
                            "v": torch.zeros(
                                _nb_max, nkv, block_size, self.key_cache.shape[-1], dtype=torch.int8, device=device
                            ),
                            "ks": torch.zeros(1, nkv, _S_pad, dtype=torch.bfloat16, device=device),
                            "ko": torch.zeros(1, nkv, _S_pad, dtype=torch.bfloat16, device=device),
                            "vs": torch.zeros(1, nkv, _S_pad, dtype=torch.bfloat16, device=device),
                            "vo": torch.zeros(1, nkv, _S_pad, dtype=torch.bfloat16, device=device),
                            "askv_dev": torch.zeros(1, dtype=torch.int64, device=device),
                            "asq_dev": torch.zeros(1, dtype=torch.int64, device=device),
                        }
                    _c = _cp[layer_idx]
                    if _c["nb_max"] < _nb_now:
                        raise RuntimeError(f"[R136-UNI] pool nb_max={_c['nb_max']} < nb_now={_nb_now}")
                    # capacity bt = identity arange(nb_max) (R110 D6a; R119 fix#5)
                    if getattr(type(self), "_r113_btpool", None) is None:
                        type(self)._r113_btpool = {}
                    _bp = type(self)._r113_btpool
                    _bk2 = (int(nq), int(head_size), int(_c["nb_max"]))
                    if _bk2 not in _bp:
                        _bp[_bk2] = (
                            torch.arange(_c["nb_max"], dtype=torch.int32, device=device).view(1, -1).contiguous()
                        )
                    _bt_u = _bp[_bk2]
                    _bt_u.copy_(torch.arange(_c["nb_max"], dtype=torch.int32, device=device).view(1, -1))
                    _btg = block_table[b, blk_lo : blk_lo + _nb_now]
                    _c["k"][:_nb_now] = self.key_cache[_btg].permute(0, 2, 1, 3).contiguous().view(torch.int8)
                    _c["v"][:_nb_now] = self.value_cache[_btg].permute(0, 2, 1, 3).contiguous().view(torch.int8)
                    _Sn = _nb_now * block_size
                    _ksb = self.k_scale_cache[_btg].reshape(_Sn, nkv)
                    _kzb = self.k_zero_cache[_btg].reshape(_Sn, nkv)
                    _vsb = self.v_scale_cache[_btg].reshape(_Sn, nkv)
                    _vzb = self.v_zero_cache[_btg].reshape(_Sn, nkv)
                    _c["ks"][0, :, : int(mid_len)] = _ksb[: int(mid_len)].t().bfloat16()
                    _c["ko"][0, :, : int(mid_len)] = (
                        -_kzb[: int(mid_len)].t() * _c["ks"][0, :, : int(mid_len)]
                    ).bfloat16()
                    _c["vs"][0, :, : int(mid_len)] = _vsb[: int(mid_len)].t().bfloat16()
                    _c["vo"][0, :, : int(mid_len)] = (
                        -_vzb[: int(mid_len)].t() * _c["vs"][0, :, : int(mid_len)]
                    ).bfloat16()
                    _c["askv_dev"].fill_(int(mid_len))
                    _c["asq_dev"].fill_(1)
                    _r136_p0_dump(
                        "unified",
                        layer_idx,
                        _R2XX_G.setdefault("_R109_STATS", {}).get("middle_calls", 0),
                        {
                            "mid_start": int(mid_start),
                            "mid_len": int(mid_len),
                            "nb_now": int(_nb_now),
                            "nb_max": int(_c["nb_max"]),
                            "S_pad": int(_c["S_pad"]),
                            "block_size": int(block_size),
                            "askv": int(mid_len),
                        },
                        {
                            "q": query_bnsd,
                            "k": _c["k"][:_nb_now],
                            "v": _c["v"][:_nb_now],
                            "bt": _bt_u,
                            "ks": _c["ks"][0, :, : int(mid_len)],
                            "ko": _c["ko"][0, :, : int(mid_len)],
                            "vs": _c["vs"][0, :, : int(mid_len)],
                            "vo": _c["vo"][0, :, : int(mid_len)],
                        },
                    )
                    fr = type(self)._r135_mod.launch_lse_v1_own(
                        query_bnsd,
                        _c["k"],
                        _c["v"],
                        _bt_u,
                        _c["ks"],
                        _c["ko"],
                        _c["vs"],
                        _c["vo"],
                        out_h,
                        lse_h,
                        _c["asq_dev"],
                        _c["askv_dev"],
                        None,
                        nq,
                        1.0 / _m.sqrt(head_size),
                        nkv,
                        "BNSD",
                        block_size,
                        1,
                        1,
                        empty_mask,
                        0,
                        0,
                        0,
                        True,
                    )
                else:
                    # R135 V1 eager path (F_eager, TIGHT operands). Default.
                    if getattr(type(self), "_r135_adev", None) is None:
                        type(self)._r135_adev = {}
                    _adev = type(self)._r135_adev
                    _dk = str(device)
                    if _dk not in _adev:
                        _adev[_dk] = (
                            torch.tensor([1], dtype=torch.int64, device=device),
                            torch.tensor([int(block_size)], dtype=torch.int64, device=device),
                        )
                    _asq_d, _askv_d = _adev[_dk]
                    _askv_d.fill_(int(askv_list[0]))
                    _r136_p0_dump(
                        "eager",
                        layer_idx,
                        _R2XX_G.setdefault("_R109_STATS", {}).get("middle_calls", 0),
                        {
                            "mid_start": int(mid_start),
                            "mid_len": int(mid_len),
                            "nb": int(blk_hi - blk_lo),
                            "block_size": int(block_size),
                            "askv": int(askv_list[0]),
                        },
                        {
                            "q": query_bnsd,
                            "k": k_r91,
                            "v": v_r91,
                            "bt": bt,
                            "ks": k_scale_flat,
                            "ko": k_offset_flat,
                            "vs": v_scale_flat,
                            "vo": v_offset_flat,
                        },
                    )
                    fr = type(self)._r135_mod.launch_lse_v1_own(
                        query_bnsd,
                        k_r91,
                        v_r91,
                        bt,
                        k_scale_flat,
                        k_offset_flat,
                        v_scale_flat,
                        v_offset_flat,
                        out_h,
                        lse_h,
                        _asq_d,
                        _askv_d,
                        None,
                        nq,
                        1.0 / _m.sqrt(head_size),
                        nkv,
                        "BNSD",
                        block_size,
                        1,
                        1,
                        empty_mask,
                        0,
                        0,
                        0,
                        True,
                    )
                    _r136_p0_dump(
                        "eager_out",
                        layer_idx,
                        _R2XX_G.setdefault("_R109_STATS", {}).get("middle_calls", 0),
                        {"mid_start": int(mid_start), "mid_len": int(mid_len)},
                        {"out_h": out_h, "lse_h": lse_h},
                    )
            else:
                fr = type(self)._r112_mod.launch_lse_fast(
                    query_bnsd,
                    k_r91,
                    v_r91,
                    bt,
                    k_scale_flat,
                    k_offset_flat,
                    v_scale_flat,
                    v_offset_flat,
                    out_h,
                    lse_h,
                    nq,
                    1.0 / _m.sqrt(head_size),
                    nkv,
                    "BNSD",
                    block_size,
                    1,
                    1,
                    askv_list,
                    empty_mask,
                    0,
                    0,
                    0,
                    True,
                )
            _st["middle_calls"] += 1
            if not (int(fr[0]) == 0 and int(fr[2]) == 0):
                raise RuntimeError(f"[R112-MID] fast launch NOT ok: {tuple(int(x) for x in fr)}")
            _fst = globals().setdefault("_R112_FAST", {"calls": 0, "verify": []})
            _fst["calls"] += 1
            _ver = int(_os.environ.get("R112_VERIFY_LSE", "0"))
            if _ver > 0 and (_fst["calls"] % _ver) == 1:
                vd = type(self)._r112_mod.verify_lse(out_h, lse_h)
                _fst["verify"].append(
                    {"call": _fst["calls"], "lse_finite": int(vd["lse_finite"]), "lse_changed": int(vd["lse_changed"])}
                )
                if int(vd["lse_finite"]) < nq:
                    raise RuntimeError(
                        f"[R112-MID] periodic verify lse_finite={int(vd['lse_finite'])}<{nq} (fail-closed)"
                    )
            _st["middle_ok"] += 1
            return out_h[0, :, 0, :], lse_h[0, :, 0]

        result = type(self)._r91_mod.launch_lse(
            query_bnsd,
            k_r91,
            v_r91,
            bt,
            k_scale_flat,
            k_offset_flat,
            v_scale_flat,
            v_offset_flat,
            out_h,
            lse_h,
            nq,
            1.0 / _m.sqrt(head_size),
            nkv,
            "BNSD",
            block_size,
            1,
            1,
            askv_list,
            empty_mask,
            0,
            0,
            0,
            True,
        )
        if _st["middle_calls"] < 3:
            print(f"[R109-MID-CP3] post-launch result={result}", flush=True)
        _st["middle_calls"] += 1
        # R109: file-based counter (import-topology independent — vLLM re-imports
        # this module per layer/step in some paths, giving each instance fresh
        # globals; the filesystem is the only shared truth)
        if _os.environ.get("R112_NOSTATS", "0") != "1":
            try:
                with _odiag_o("r111_mid_calls.jsonl") as _f:
                    _f.write(_os.environ.get("R109_KTAG", "x") + "\n")
            except Exception:
                pass
        ok = (
            result["getws_ret"] == 0
            and result["launch_ret"] == 0
            and result["sync_ret"] == 0
            and result["lse_changed"] > 0
            and result["lse_finite"] > 0
        )
        if ok:
            _st["middle_ok"] += 1
            return out_h[0, :, 0, :], lse_h[0, :, 0]
        print(f"[R109-MID] candidate middle NOT ok: {result}", flush=True)
        return None
