# R147 OscarGraphAdapter — task-owned generalization extraction (P5 hard gate).
# Lifted VERBATIM from attention_v1.py R146/R147 blocks (see CHECKPOINTS backup):
#  - middle_decode(): static-shape carrier fill + per-(cap,layer) NPUGraph middle op
#  - sr_merge():     batched S+R FIA + vectorized 2-tier merge + V-unrot + writeback
# Contract (user gate 2026-09-13):
#  * NO model-name branches; NO hardcoded layer counts / headDim / NKV / Batch /
#    maxLen literals — every shape derives from vllm_config + AttentionMetadata at
#    runtime (ladder from max_model_len, BcapT from scheduler max_num_seqs, carrier
#    dims from num_heads/num_kv_heads/head_size/block_size).
#  * layer ordinal passed EXPLICITLY by the caller (framework-provided layer object);
#    graphs keyed by that ordinal.
#  * Any Bactive >= 1 (BcapT = max(max_num_seqs, Bp); bucket ladder arbitrary).
#  * Unsupported paths (no middle tier rows, empty ladder) return None — caller
#    falls back to the legacy per-row path (fail-closed, never silently wrong).
import math as _m144
import os as _os
from typing import Any

import torch
import torch_npu


class OscarGraphAdapter:
    """Per-backend-instance adapter. Shared carriers/staging live on the CLASS
    (content-only carriers, consumed by same-stream sequential layer replays)."""

    def __init__(self, impl):
        self.impl = impl

    # ---- middle tier: static carriers + per-(cap,layer) NPUGraph fill+op ----
    def middle_decode(
        self,
        *,
        layer,
        layer_ord,
        query,
        seq_lens,
        attn_metadata,
        batch_size,
        block_size,
        num_heads,
        num_kv_heads,
        head_size,
        stats,
        r144_bmid,
        Rk,
        graph_on,
        stride_on,
    ):
        """Returns dict(handled, carr, cb, cbl, cnb, cml, cap, Bp, BcapT) or None."""
        _impl = self.impl
        _bs = block_size
        _nq = num_heads
        _nkv = num_kv_heads
        _hd = head_size
        _st = stats
        _r144_bmid = r144_bmid
        _Rk144 = Rk
        _R145G = graph_on
        _R146S = stride_on
        _lord146 = layer_ord
        _r2, _lr2 = 0, 0
        _r145_handled = False
        if True:  # caller gates R145_TENSOR/R146_FILL
            _step = int(_os.environ.get("R145_CAP_STEP", "256"))
            _mml_top = int(getattr(_impl.vllm_config.model_config, "max_model_len", 4096))
            _ladder = list(range(_step, _mml_top + _step, _step))
            _cb = []
            _cbl = []
            _cnb = []
            _cml = []
            for _bb in range(batch_size):
                _sl = int(seq_lens[_bb])
                _pr = _impl._tt_pref.get(_impl._tt_rid(_bb), 0)
                _rc = _impl._tt_rcnt.get(_impl._tt_rid(_bb), 0)
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
                _cap = _ladder[-1]
            if len(_cb) >= 1 and _cap > 0:
                _Bp = len(_cb)
                _Bcap = max(int(_impl._tt_mseq), _Bp)
                _BcapT = _Bcap if _R146S else _Bcap + 1  # P1: no sentinel row
                _nbC = _cap // _bs
                _hoist147 = _os.environ.get("R147_HOIST", "0") == "1"
                _carr = _impl._r145_carriers.get(("r146", _cap))
                if _hoist147:
                    # R147 P1-B: carriers are CONTENT carriers (staged once per
                    # step, consumed sequentially by 36 same-stream layer replays)
                    # -- safe to share ONE per (cap, BcapT) across the per-layer
                    # backend instances. R146 allocated a full set PER INSTANCE
                    # (36x HBM: e.g. B16 3 buckets 68.1MB -> 2.45GB) and re-
                    # allocated on every BcapT change (capture churn). Keyed by
                    # (cap, BcapT) so shape changes get a fresh carrier and fresh
                    # graphs (graphs stay per-layer inside the shared dict).
                    _sh147 = getattr(type(_impl), "_r147_shared", None)
                    if _sh147 is None:
                        _sh147 = {}
                        type(_impl)._r147_shared = _sh147
                    _carr = _sh147.get(("r146", _cap, _BcapT))
                if _carr is None:
                    _kv_cap = _BcapT * _nbC
                    _d4 = _hd // 4
                    _mseq_i = int(_impl._tt_mseq)
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
                        type(_impl)._r147_shared[("r146", _cap, _BcapT)] = _carr
                    _impl._r145_carriers[("r146", _cap)] = _carr  # instance alias
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
                    _st147: dict[str, Any] | None = getattr(type(_impl), "_r147_stage", None)
                    if _st147 is None:
                        _st147 = {}
                        type(_impl)._r147_stage = _st147
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
                if _msx < int(_carr["btStage"].size(0)) and _vmode != "hoisted":  # sentinel/padding rows mirror row 0
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
                # ---- per-LAYER: eager qsrc/qrot + graph(fill+op) ----
                if _os.environ.get("R146_VANILLA", "0") == "1":
                    # staging present; middle = vanilla R145 math (per-row rot + C++ fill +
                    # op) into a vanilla-shaped carrier; NOTHING of my fill/op runs.
                    _carr3 = _impl._r145_carriers.get(_cap)
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
                        _impl._r145_carriers[_cap] = _carr3
                    for _ci, _bb in enumerate(_cb):
                        _qb = query[_bb : _bb + 1]
                        if _qb.dim() == 3:
                            _qb = _qb.unsqueeze(2)
                        if _Rk144 is not None:
                            torch.matmul(_qb, _Rk144, out=_carr3["qrot"][_ci : _ci + 1])
                        else:
                            _carr3["qrot"][_ci : _ci + 1].copy_(_qb)
                    type(_impl)._r112_mod.oscar_middle_prep_v3_fill(
                        attn_metadata.block_tables,
                        _cb,
                        _cbl,
                        _cnb,
                        _cml,
                        _impl.key_cache,
                        _impl.value_cache,
                        _impl.k_scale_cache,
                        _impl.k_zero_cache,
                        _impl.v_scale_cache,
                        _impl.v_zero_cache,
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
                    _rr3v = type(_impl)._r112_mod.oscar_middle_op_v3(
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
                    for _i, _bb in enumerate(_cb):
                        _r144_bmid[_bb] = (_carr3["out"][_i, :, 0, :].float(), _carr3["lse"][_i, :, 0].float())
                    _st["middle_calls"] += 1
                    _st["middle_ok"] += 1
                    _r145_handled = True
                    _r2, _lr2 = int(_rr3v[0]), int(_rr3v[2])
                if not _r145_handled:
                    torch.index_select(query, 0, _carr["bidx"], out=_carr["qsrc"])
                if _Rk144 is not None:
                    torch.matmul(_carr["qsrc"].unsqueeze(2), _Rk144, out=_carr["qrot"])
                else:
                    _carr["qrot"].copy_(_carr["qsrc"].unsqueeze(2))
                _kc = _impl.key_cache
                _vc = _impl.value_cache
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

                def _r146_fill_layer():
                    torch.index_select(_kc, 0, _idxF, out=_kfl)
                    torch.index_select(_vc, 0, _idxF, out=_vfl)
                    _kcfl.copy_(_btw.permute(0, 1, 3, 2, 4).reshape(_kcfl.shape))
                    _vcfl.copy_(_vbw.permute(0, 1, 3, 2, 4).reshape(_vcfl.shape))
                    # NOTE: uint8 permute-copy keeps packed INT2 bytes verbatim (no unpack).
                    torch.index_select(_impl.k_scale_cache, 0, _idxF, out=_carr["ksgS"])
                    _carr["kS"].copy_(_carr["ksgS"].view(_BcapT, _nbC * _bs, _nkv).permute(0, 2, 1))
                    torch.index_select(_impl.k_zero_cache, 0, _idxF, out=_carr["kzgS"])
                    _carr["kO"].copy_(_carr["kzgS"].view(_BcapT, _nbC * _bs, _nkv).permute(0, 2, 1))
                    _carr["kO"].mul_(_carr["kS"])
                    _carr["kO"].neg_()
                    torch.index_select(_impl.v_scale_cache, 0, _idxF, out=_carr["vsgS"])
                    _carr["vS"].copy_(_carr["vsgS"].view(_BcapT, _nbC * _bs, _nkv).permute(0, 2, 1))
                    torch.index_select(_impl.v_zero_cache, 0, _idxF, out=_carr["vzgS"])
                    _carr["vO"].copy_(_carr["vzgS"].view(_BcapT, _nbC * _bs, _nkv).permute(0, 2, 1))
                    _carr["vO"].mul_(_carr["vS"])
                    _carr["vO"].neg_()

                def _r146_op(_dz):
                    return type(_impl)._r112_mod.oscar_middle_op_v3(
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
                    _r3dbg = type(_impl)._r112_mod.oscar_middle_prep_v3_fill(
                        attn_metadata.block_tables,
                        _cb,
                        _cbl,
                        _cnb,
                        _cml,
                        _kc,
                        _vc,
                        _impl.k_scale_cache,
                        _impl.k_zero_cache,
                        _impl.v_scale_cache,
                        _impl.v_zero_cache,
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
                        _r144_bmid[_bb] = (_carr["out"][_i, :, 0, :].float(), _carr["lse"][_i, :, 0].float())
                    _st["middle_calls"] += 1
                    _st["middle_ok"] += 1
                    _r145_handled = True
                    # fall through to skip normal path this layer
                if not _r145_handled and _R145G and False:  # noqa: SIM223 - parked arm, deliberately unreachable
                    pass
                # R146_CMP=1: run BOTH paths this layer and diff out/lse per row.
                if _os.environ.get("R146_CMP", "0") == "1" and not _r145_handled:
                    _carr2 = _impl._r145_carriers.get(_cap)
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
                        _impl._r145_carriers[_cap] = _carr2
                    for _ci, _bb in enumerate(_cb):
                        _qb = query[_bb : _bb + 1]
                        if _qb.dim() == 3:
                            _qb = _qb.unsqueeze(2)
                        if _Rk144 is not None:
                            torch.matmul(_qb, _Rk144, out=_carr2["qrot"][_ci : _ci + 1])
                        else:
                            _carr2["qrot"][_ci : _ci + 1].copy_(_qb)
                    type(_impl)._r112_mod.oscar_middle_prep_v3_fill(
                        attn_metadata.block_tables,
                        _cb,
                        _cbl,
                        _cnb,
                        _cml,
                        _impl.key_cache,
                        _impl.value_cache,
                        _impl.k_scale_cache,
                        _impl.k_zero_cache,
                        _impl.v_scale_cache,
                        _impl.v_zero_cache,
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
                    _rr2 = type(_impl)._r112_mod.oscar_middle_op_v3(
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
                                f"L={_lord146} B={batch_size} Bp={_Bp} cap={_cap} ml={_cml} "
                                f"qrot={_qi} Kc={_ki} kS={_si} kO={_oi} askv={_ai} "
                                f"OUTdiff={_do} LSEdiff={_dl} rc2={int(_rr2[0])} rcm={int(_rrm[0])}\n"
                            )
                    except Exception:
                        pass
                    for _i, _bb in enumerate(_cb):
                        _r144_bmid[_bb] = (_carr2["out"][_i, :, 0, :].float(), _carr2["lse"][_i, :, 0].float())
                    _st["middle_calls"] += 1
                    _st["middle_ok"] += 1
                    _r145_handled = True
                _lord146 = _impl._oscar_layer_ord(layer)  # layer_idx attr is MISSING on
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
                    _rr146 = _r146_op(True)
                    _r2, _lr2 = int(_rr146[0]), int(_rr146[2])
                    # R147 P0-A guard: flaky-artifact class = eager dz=True call leaves
                    # LSE unwritten at fill value (-9999) with rc=0 (~1% under a
                    # replay-then-eager interleave; serving first-use eager runs
                    # BEFORE any replay so unexposed). Fail loudly ONCE per first
                    # use instead of feeding -9999 into the tier merge.
                    if _r2 == 0 and _os.environ.get("R147_LSE_GUARD", "1") == "1":
                        _mn147 = float(_carr["lse"][:_Bp].min().item())
                        if _mn147 < -9000.0:
                            raise RuntimeError(
                                f"[R147-LSE-GUARD] L={_lord146} Bp={_Bp} cap={_cap} "
                                f"eager first-use left LSE at fill value (min={_mn147}) "
                                f"— known flaky eager-side non-write; retry step"
                            )
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
                                        f"L={_lord146} Bp={_Bp} cap={_cap} ml={_cml} OUTdiff={_do} LSEdiff={_dl}\n"
                                    )
                            except Exception:
                                pass
                            _carr["out"][:_Bp].copy_(_oe146)
                            _carr["lse"][:_Bp].copy_(_le146)
                if not (_r2 == 0 and _lr2 == 0):
                    raise RuntimeError(f"[R146] middle NOT ok rc={_r2} lr={_lr2}")
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
                for _i, _bb in enumerate(_cb):
                    _r144_bmid[_bb] = (_carr["out"][_i, :, 0, :].float(), _carr["lse"][_i, :, 0].float())
                _st["middle_calls"] += 1
                _st["middle_ok"] += 1
                _r145_handled = True

        if _r145_handled:
            return {
                "handled": True,
                "carr": _carr,
                "cb": _cb,
                "cbl": _cbl,
                "cnb": _cnb,
                "cml": _cml,
                "cap": _cap,
                "Bp": _Bp,
                "BcapT": _BcapT,
            }
        return None

    # ---- S+R tiers + merge + unrot + writeback (batched, bit-exact vs legacy) ----
    def sr_merge(
        self, *, layer, layer_ord, query, seq_lens, output, batch_size, r144_bmid, carr, cb, cml, stats, Rk, Rv, handled
    ):
        """Returns True when it handled all rows (output written)."""
        _impl = self.impl
        _st = stats
        _r144_bmid = r144_bmid
        _carr = carr
        _cb = cb
        _cml = cml
        _Rk144 = Rk
        _Rv147 = Rv
        # ================= R147 P1-C: batched S+R + merge + unrot + writeback =====
        # (R147_SR_BATCH=1, default OFF; requires the R146 fill/carrier path active so
        #  middle results live in the shared carrier). Per layer: ONE padded FIA for
        #  all rows + per-row l_pr einsum (legacy op sequence verbatim, values read
        #  from the padded S+R carrier slices) + ONE vectorized merge in carrier
        #  space + one unrot matmul + one index_copy writeback (~18*B tiny ops/layer
        #  -> ~6+3B). Per-element math order identical to the legacy per-row formula;
        #  absent tiers selected by exact torch.where (no arithmetic on absent
        #  branch); FIA stays eager (host actual_seq_lengths). R147_SR_CMP=1
        #  recomputes the legacy per-row path per layer and logs max|diff|.
        _srB147 = _os.environ.get("R147_SR_BATCH", "0") == "1" and handled and len(_r144_bmid) > 0
        if _srB147:
            _Rv147 = None
            if _impl._oscar_R_v is not None and _impl._oscar_layer_ord(layer) < _impl._oscar_R_v.shape[0]:
                _Rv147 = _impl._oscar_R_v[_impl._oscar_layer_ord(layer)]
            _nqS = _impl.num_heads
            _nkvS = _impl.num_kv_heads
            _hdS = _impl.head_size
            _prefC = int(_impl._TT_PREFIX)
            _capS = int(_impl._TT_RECENT)
            _CPR = _prefC + _capS  # static S+R width bound
            _lordS = _impl._oscar_layer_ord(layer)
            _BpS = len(_cb)
            _BT = max(int(_carr["out"].size(0)), batch_size)
            _hkS, _hvS = _impl._tt_hp_bufs(_lordS, query.device, query.dtype)
            _shS = getattr(type(_impl), "_r147_shared", None)
            if _shS is None:
                _shS = {}
                type(_impl)._r147_shared = _shS
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
                }
                _shS[("sr", _BT, _CPR)] = _src
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
                _ridS = _impl._tt_rid(_bS)
                _slotS = _impl._tt_slot(_bS)
                _prS = _impl._tt_pref.get(_ridS, 0)
                _rcS = _impl._tt_rcnt.get(_ridS, 0)
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
                scale=_impl.scale,
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
                _ridS = _impl._tt_rid(_bS)
                _prS = _impl._tt_pref.get(_ridS, 0)
                _nS = _prs_list[_bS]
                _qrowS = _src["qrot"][_bS]  # [nq, hd] rotated (legacy q_rot)
                _kkS = _src["prk"][_bS, :_nS][:, _kvhS, :].permute(1, 0, 2).float()
                _scS = torch.einsum("hd,hnd->hn", _qrowS.float(), _kkS) * _impl.scale
                _src["lpr"][_bS].copy_(torch.logsumexp(_scS, dim=-1))
            for _bS in range(batch_size):
                if _bS in _cb:
                    continue
                _ridS = _impl._tt_rid(_bS)
                _prS = _impl._tt_pref.get(_ridS, 0)
                _nS = _prs_list[_bS]
                if _nS <= 0:
                    continue
                _qrowS = _src["qrot"][_bS]
                _kkS = _src["prk"][_bS, :_nS][:, _kvhS, :].permute(1, 0, 2).float()
                _scS = torch.einsum("hd,hnd->hn", _qrowS.float(), _kkS) * _impl.scale
                _src["lpr"][_bS].copy_(torch.logsumexp(_scS, dim=-1))
            # vectorized merge (identical per-element formula & order as legacy)
            _mS = torch.maximum(_src["lpr"], _src["lh"])
            _wprS = torch.exp(_src["lpr"] - _mS)
            _whS = torch.exp(_src["lh"] - _mS)
            _mgS = (_wprS.unsqueeze(-1) * _src["opr"] + _whS.unsqueeze(-1) * _src["oh"]) / (_wprS + _whS).unsqueeze(-1)
            _prOK = _src["prlens"] > 0
            _hOK = _src["hOKd"] > 0
            # NPU torch.where does not broadcast the cond dim; expand explicitly.
            _bothE = (_prOK & _hOK).view(-1, 1, 1).expand(-1, _nqS, _hdS)
            _hE = _hOK.view(-1, 1, 1).expand(-1, _nqS, _hdS)
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
                    _rid = _impl._tt_rid(b)
                    _slot = _impl._tt_slot(b)
                    pref = _impl._tt_pref.get(_rid, 0)
                    rcnt = _impl._tt_rcnt.get(_rid, 0)
                    n_pr = pref + rcnt
                    pr_k = torch.cat([_hkS[_slot, :pref], _hkS[_slot, pref : pref + rcnt]], dim=0)
                    pr_v = torch.cat([_hvS[_slot, :pref], _hvS[_slot, pref : pref + rcnt]], dim=0)
                    q_b = query[b : b + 1]
                    q_rotL = torch.matmul(q_b, _Rk144) if _Rk144 is not None else q_b
                    al, _ = torch_npu.npu_fused_infer_attention_score(
                        query=q_rotL.unsqueeze(2),
                        key=pr_k.permute(1, 0, 2).unsqueeze(0),
                        value=pr_v.permute(1, 0, 2).unsqueeze(0),
                        input_layout="BNSD",
                        actual_seq_lengths=[1],
                        actual_seq_lengths_kv=[n_pr],
                        num_heads=_impl.num_heads,
                        num_key_value_heads=_impl.num_kv_heads,
                        scale=_impl.scale,
                        sparse_mode=0,
                    )
                    o_prL = al[0, :, 0, :].float()
                    kk = pr_k[:, _kvhS, :].permute(1, 0, 2).float()
                    sc = torch.einsum("hd,hnd->hn", q_rotL[0].float(), kk) * _impl.scale
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

        return _srB147
