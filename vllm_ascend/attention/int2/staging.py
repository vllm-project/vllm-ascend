# SPDX-License-Identifier: Apache-2.0
"""Staging / chunked-prefill continuation mixin for the INT2 backend."""

from __future__ import annotations

import os
import os as _r144_ovl_os
from typing import Any

import torch

from vllm_ascend.attention.int2.ctx import (  # noqa: F401
    _odiag_o,
)
from vllm_ascend.attention.int2.helpers import (  # noqa: F401
    _oscar_dbg,
)


class Int2StagingMixin:
    # Host-state annotations only: the _r218_st_* staging pool and the
    # _r156_h3_* kernel handles are initialized by the slot-state mixin /
    # this module's loader, and the cache interfaces are owned by the
    # assembled backend. Bare annotations create no runtime attributes;
    # they pin types for mypy --check-untyped-defs.
    _TT_PREFIX: int
    _TT_RECENT: int
    _oscar_R_k: Any
    _r113_fcnt: Any
    _r156_h3_keep: Any
    _r156_h3_loaded: Any
    _r156_h3_mod: Any
    _r218_st_k: dict[Any, Any]
    _r218_st_v: dict[Any, Any]
    _r218_st_slot: dict[Any, Any]
    _r218_st_rid: dict[Any, Any]
    _r218_st_len: dict[Any, Any]
    _r218_st_clock: dict[Any, Any]
    _r218_st_tick: Any
    key_cache: Any
    value_cache: Any
    k_scale_cache: Any
    k_zero_cache: Any
    v_scale_cache: Any
    v_zero_cache: Any
    vllm_config: Any

    def _r156_h3_stage_all(self, update_stream, forward_context, params, num_tokens):
        """R156-H3: ONE fused launch staging every layer's middle carriers
        (byte-identical to the per-layer ATen staging; golden ALL-PASS).
        Returns True if it handled this step (updater then skips staging).
        B=1 fast path; B>1 returns False -> per-layer ATen path (correct,
        just slower). Fail-closed on kernel/binding problems."""
        import os as _os

        try:
            p0 = params[0]
            (
                marker,
                q,
                k,
                v,
                bt,
                ks_c,
                ko_c,
                vs_c,
                vo_c,
                out_h,
                lse_h,
                mask,
                nq,
                scale,
                nkv,
                layout,
                block_size,
                b,
                mid_start,
                rcnt,
                S_pad,
                impl,
                _nt,
                layer_idx,
                _lid0,
            ) = p0
        except Exception:
            return False
        if b != 0:
            return False  # B>1: per-b ptr tables not wired; ATen path
        for _p in params:
            if _p[17] != 0:
                return False
        md = None
        for key in forward_context.attn_metadata:
            md = forward_context.attn_metadata[key]
            break
        if md is None:
            return False
        sl = int(md.seq_lens_list[0])
        mid_len = sl - rcnt - mid_start
        if mid_len <= 0:
            return False  # neutralize branch handles it
        nb_now = (mid_len + block_size - 1) // block_size
        blk_lo = mid_start // block_size
        _dbg = _os.environ.get("R156_H3_DBG", "0") == "1"

        def _dbg_w(msg):
            if _dbg:
                try:
                    with open("/tmp/r156_h3_dbg.log", "a") as _f:
                        _f.write(f"{msg}\n")
                except Exception:
                    pass

        _dbg_w(f"stage_all enter L={len(params)} sl={sl} mid={mid_len}")
        if not impl._r156_h3_ensure(None):
            raise RuntimeError("[R156-H3] kernel binding missing (fail-closed)")
        mod = type(impl)._r156_h3_mod
        _dbg_w("ensure ok")
        # ptr table: STATIC addresses (caches+carriers) — build on CPU,
        # ONE H2D, cached until any address changes. bt_live is passed LIVE
        # as a kernel arg (content changes per step, address tracked in the
        # cache key).
        L = len(params)
        _cp = getattr(type(impl), "_r113_cpool", {})
        bt_live = md.block_tables
        if bt_live.dtype != torch.int32:
            return False
        # R156: bt_live may be a NON-CONTIGUOUS view (padded rows) — stride(0)
        # is the true row pitch; assuming shape[1] reads garbage page ids from
        # the padding => wild GM addresses => AIV exception (all-mode fault).
        if bt_live.dim() != 2 or bt_live.stride(1) != 1:
            return False
        bt_pitch = int(bt_live.stride(0))
        tab = getattr(type(impl), "_r156_h3_tab", None)
        _bt_ptr = bt_live.data_ptr()
        if (
            tab is None
            or tab[0] != L
            or tab[1] != _bt_ptr
            or tab[2] != bt.shape[1]
            or tab[3] != int(S_pad)
            or tab[4] != bt_pitch
        ):
            ptrs_cpu = torch.zeros(L, 15, dtype=torch.int64, pin_memory=True)
            keep = []
            addr_sig = []
            for i, _p in enumerate(params):
                (
                    _m,
                    _q,
                    _k,
                    _v,
                    _bt,
                    _ks,
                    _ko,
                    _vs,
                    _vo,
                    _oh,
                    _lh,
                    _mk,
                    _nq,
                    _sc,
                    _nkv,
                    _lay,
                    _bs2,
                    _b2,
                    _ms2,
                    _rc2,
                    _sp2,
                    _impl,
                    _nt2,
                    _li2,
                    _lid,
                ) = _p
                _c = _cp.get(_lid)
                if _c is None:
                    return False
                _hs_p = int(_k.shape[-1]) * 4
                _bp_full = getattr(type(_impl), "_r113_btpool_full", {})
                _btfull = _bp_full.get((_lid, int(_nq), _hs_p, int(_bt.shape[1])))
                if _btfull is None:
                    _btfull = _bt
                vals = [
                    _impl.key_cache,
                    _impl.value_cache,
                    _impl.k_scale_cache,
                    _impl.k_zero_cache,
                    _impl.v_scale_cache,
                    _impl.v_zero_cache,
                    _k,
                    _v,
                    _ks,
                    _ko,
                    _vs,
                    _vo,
                    _btfull,
                    _c["askv_dev"],
                    _c["asq_dev"],
                ]
                keep.extend(vals)
                for j, t in enumerate(vals):
                    _dp = t.data_ptr()
                    ptrs_cpu[i, j] = _dp
                    addr_sig.append(_dp)
                if i % 8 == 0:
                    _dbg_w(f"layer {i} ptrs set")
            _dbg_w("loop done, to-device next")
            # pinned + non_blocking H2D is the SUPPORTED async path; pageable
            # NB H2D hung on the update stream (r156h3_cb/b/e legs)
            ptrs = torch.empty(L, 15, dtype=torch.int64, device=k.device)
            ptrs.copy_(ptrs_cpu, non_blocking=True)
            _dbg_w("to-device done")
            type(impl)._r156_h3_tab = (L, _bt_ptr, int(bt.shape[1]), int(S_pad), bt_pitch, ptrs, keep, addr_sig)
            type(impl)._r156_h3_btfull = _btfull
            tab = type(impl)._r156_h3_tab
        (L2, _btp2, _btw2, _sp22, _btpitch2, ptrs, keep, addr_sig) = tab
        _btfull = getattr(type(impl), "_r156_h3_btfull", bt)
        _btw_pad = int(_btfull.shape[1])
        _dbg_w(f"table ok ptrs={ptrs.shape} bt={tuple(bt_live.shape)} nb={nb_now} blkLo={blk_lo} nkv={nkv}")
        _dbg_w(
            f"TRUTH bs={block_size} S_pad={S_pad} kc={tuple(impl.key_cache.shape)} "
            f"kcar={tuple(k.shape)} kscar={tuple(ks_c.shape)} "
            f"ksc={tuple(impl.k_scale_cache.shape)} "
            f"btcar={tuple(bt.shape)} askv={tuple(_cp[params[0][24]]['askv_dev'].shape)} "
            f"packed={impl.key_cache.shape[-1]} mid={mid_len} sl={sl}"
        )
        import struct as _st156

        _tb = _st156.pack(
            "<12I4Q",
            L,
            nb_now,
            blk_lo,
            block_size,
            int(nkv),
            int(impl.key_cache.shape[-1]),
            mid_len,
            0,
            _btw_pad,
            bt_pitch,
            int(S_pad),
            int(_os.environ.get("R156_H3_MODE", "0")),
            block_size * int(nkv) * int(impl.key_cache.shape[-1]),
            0,
            int(nkv) * int(S_pad),
            block_size * int(nkv),
        )
        # R156: vllm rebuilds md.block_tables EVERY STEP; allocator address
        # reuse means step i+1's H2D can overwrite the same address while the
        # step-i kernel still hasn't executed (wild pageIds -> MTE3 fault).
        # Give the kernel a PRIVATE, stream-ordered copy on the update stream.
        _btb = getattr(type(impl), "_r156_h3_btown", None)
        if _btb is None or tuple(_btb.shape) != tuple(bt_live.shape):
            _btb = type(impl)._r156_h3_btown = torch.empty_like(bt_live)
        # R156: dedicated launch stream, fenced by events against the
        # update stream (the update stream's event-entangled state with the
        # replaying ACLGraph is the remaining in-engine-only difference).
        _s3 = getattr(type(impl), "_r156_h3_stream", None)
        if _s3 is None:
            _s3 = type(impl)._r156_h3_stream = torch.npu.Stream()
            _s3.wait_stream(torch.npu.current_stream())
        _ev3a = torch.npu.Event()
        _ev3a.record(update_stream)
        _s3.wait_event(_ev3a)
        with torch.npu.stream(_s3):
            _btb.copy_(bt_live, non_blocking=True)
            _tb_cpu = torch.frombuffer(bytearray(_tb), dtype=torch.uint8)
            _tb_pin = _tb_cpu.pin_memory()
            _tb_dev = _tb_pin.to(ptrs.device, non_blocking=True)
            # keep BOTH the pinned staging buffer (the async H2D reads it) and
            # the device blob (the kernel reads it) alive for 4 steps — freeing
            # either lets the next step's identical allocation overwrite bytes
            # while the previous async copy/kernel is still in flight (the
            # delayed ~step-10 MTE3 fault).
            _kl = globals().setdefault("_R156_TB_KEEP", [])
            _kl.append((_tb_pin, _tb_dev))
            if len(_kl) > 4:
                _kl.pop(0)
            r = mod.stage36(
                ptrs,
                _btb,
                L,
                0,
                blk_lo,
                nb_now,
                mid_len,
                block_size,
                int(nkv),
                int(impl.key_cache.shape[-1]),
                _btw_pad,
                int(S_pad),
                bt_pitch,
                int(_os.environ.get("R156_H3_MODE", "0")),
                int(_s3.npu_stream),
                _tb_dev,
            )
        _ev3b = torch.npu.Event()
        _ev3b.record(_s3)
        update_stream.wait_event(_ev3b)
        _dbg_w(f"stage36 rc={r}")
        if int(r[1]) != 0:
            raise RuntimeError(f"[R156-H3] stage36 rc={r}")
        return True

    def _r156_h3_ensure(self, device):
        """R156-H3: load the fused stage36 direct-invoke kernel (golden
        ALL-PASS vs the ATen staging: k/v/ks/ko/vs/vo byte-identical, bt
        identity, askv/asq fills). One launch per step replaces the ~800
        enqueue staging loop."""
        import os as _os

        if type(self)._r156_h3_loaded:
            return type(self)._r156_h3_mod is not None
        type(self)._r156_h3_loaded = True
        try:
            import importlib.util as _ilu

            _so = _os.environ.get("R156_SO", "")
            _spec = _ilu.spec_from_file_location("r156_stage36_ops", _so)
            assert _spec is not None and _spec.loader is not None
            _m = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_m)
            type(self)._r156_h3_mod = _m
            type(self)._r156_h3_keep = []
            print("[R156-H3] stage36 kernel loaded", flush=True)
            return True
        except Exception as _e:
            print(f"[R156-H3] load FAILED: {_e}", flush=True)
            type(self)._r156_h3_mod = None
            return False

    def _r218_stage_assign(self, slot, rid, protect=None, dead=None):
        if os.environ.get("R218_STAGE", "1") == "0":
            return None
        ns = self._r218_stage_plan()
        if ns <= 0:
            return None
        si = self._r218_st_slot.get(slot)
        if si is not None and self._r218_st_rid.get(si) == rid:
            self._r218_st_tick[0] += 1
            self._r218_st_clock[si] = self._r218_st_tick[0]
            return si
        used = set(self._r218_st_rid.keys())
        # OCCUPIED stage indices (was .values() = rids: every slot looked
        # free -> si=0 always -> lookups missed)
        free = [i for i in range(ns) if i not in used]
        if free:
            si = free[0]
        else:
            # Victim order: (1) the least-recently-written stage whose request
            # already appears as a DECODE row of this call -- that request has
            # finished its prefill (continuation, if any, already read the
            # stage) so the stage is dead; (2) any stage whose request is NOT
            # in this call's row set (rows of this call include the very
            # continuations that still need their stage). If every stage is
            # live, fall back to the INT2 read (fail-open).
            cand = [i for i in self._r218_st_rid if dead and self._r218_st_rid[i] in dead] if dead else []
            if not cand:
                cand = [i for i in self._r218_st_rid if protect is None or self._r218_st_rid[i] not in protect]
            if not cand:
                return None
            si = min(cand, key=lambda i: self._r218_st_clock.get(i, 0))
        self._r218_st_slot[slot] = si
        self._r218_st_rid[si] = rid
        self._r218_st_len[si] = 0
        self._r218_st_tick[0] += 1
        self._r218_st_clock[si] = self._r218_st_tick[0]
        return si

    def _r218_stage_lookup(self, slot, rid):
        """Side-effect-free lookup: (stage_idx, staged_len) or (None, 0)."""
        si = self._r218_st_slot.get(slot)
        if si is None or self._r218_st_rid.get(si) != rid:
            return None, 0
        return si, int(self._r218_st_len.get(si, 0))

    def _r218_stage_plan(self):
        """Effective stage slot count under the memory budget (cached)."""
        eff = getattr(self, "_r218_st_ns_eff", None)
        if eff is not None:
            return eff
        ns = int(os.environ.get("R218_STAGE_SLOTS", "4") or 4)
        if os.environ.get("R218_STAGE", "1") == "0":
            ns = 0
        mml = int(getattr(getattr(self.vllm_config, "model_config", None), "max_model_len", 8192) or 8192)
        # The stage only holds the MIDDLE span [pref, total-rcnt): prefix and
        # recent come from the BF16 HP ring, so cap it at that span, not at the
        # whole prompt.
        cap = max(
            1,
            min(mml - self._TT_PREFIX - self._TT_RECENT + 8, int(os.environ.get("R218_STAGE_MAXLEN", "8200") or 8200)),
        )
        nk = getattr(self, "num_kv_heads", 8)
        hd = getattr(self, "head_size", 128)
        nlay = 36
        try:
            if self._oscar_R_k is not None:
                nlay = int(self._oscar_R_k.shape[0])
        except Exception:
            pass
        budget = float(os.environ.get("R218_STAGE_MAXGB", "3.0") or 3.0)
        while ns > 0:
            gb = ns * cap * nk * hd * 2 * 2 * nlay / float(1024**3)
            if gb <= budget:
                break
            ns -= 1
        # A stage smaller than one row of a prefill call is useless.
        self._r218_st_ns_eff = ns
        self._r218_st_cap_eff = cap
        if os.environ.get("R144_PF_PROBE", "0") == "2":
            try:
                with _odiag_o("r218_stage{}.log".format(_r144_ovl_os.environ.get("R218_STAGE_TAG", ""))) as _fst:
                    _gb = ns * cap * nk * hd * 2 * 2 * nlay / float(1024**3)
                    _fst.write(
                        f"P ns={ns} cap={cap} mml={mml} nk={nk} hd={hd} "
                        f"nlay={nlay} budget={budget:.2f}GB total={_gb:.3f}GB\n"
                    )
            except Exception:
                pass
        _oscar_dbg(f"[R218-STAGE] plan ns={ns} cap={cap} budget={budget}GB", flush=True)
        return ns

    def _r218_stage_bufs(self, lid, device, dtype):
        if lid not in self._r218_st_k:
            ns = self._r218_stage_plan()
            cap = getattr(self, "_r218_st_cap_eff", 1024)
            nk = getattr(self, "num_kv_heads", 8)
            hd = getattr(self, "head_size", 128)
            self._r218_st_k[lid] = torch.empty(max(1, ns), cap, nk, hd, dtype=dtype, device=device)
            self._r218_st_v[lid] = torch.empty(max(1, ns), cap, nk, hd, dtype=dtype, device=device)
            _oscar_dbg(f"[R218-STAGE] alloc lid={lid} shape=({ns},{cap},{nk},{hd}) {dtype}", flush=True)
        return self._r218_st_k[lid], self._r218_st_v[lid], self._r218_st_k[lid].shape[1]

    def _r113_flush_stage(self, update_stream, layer_idx):
        """R113: flush the previous replay's staged paged-KV write.

        Runs on the DEFAULT (compute) stream: the staging was written by replay
        t (earlier in the same stream), this flush is enqueued after it, and
        replay t+1 is enqueued after the flush — ordering guaranteed by stream
        order alone, no cross-stream events (an earlier update-stream design
        deadlocked the replay sync). The advanced-index write host-reads the
        slot tensor, which syncs the default stream — acceptable: the runner
        synchronizes before replay anyway. FIRST update is skipped: its
        staging holds capture-dummy data with possibly-invalid dummy slots.
        One-step lag is semantically free (new token serves the RECENT tier
        in-graph; middle reads >= _TT_RECENT-step-old tokens only).
        """
        _stg = getattr(type(self), "_r113_stage", None)
        if not _stg:
            return
        _s = _stg.get(layer_idx)
        if _s is None:
            return
        if not hasattr(type(self), "_r113_fcnt"):
            type(self)._r113_fcnt = {}
        _cnt = type(self)._r113_fcnt
        _cnt[layer_idx] = _cnt.get(layer_idx, 0) + 1
        if _cnt[layer_idx] <= 1:
            return
        k_cache = self.key_cache
        v_cache = self.value_cache
        num_blocks = k_cache.shape[0]
        block_size = self.vllm_config.cache_config.block_size
        k_flat = k_cache.view(num_blocks * block_size, *k_cache.shape[2:])
        v_flat = v_cache.view(num_blocks * block_size, *v_cache.shape[2:])
        ks_flat = self.k_scale_cache.view(num_blocks * block_size, -1)
        kz_flat = self.k_zero_cache.view(num_blocks * block_size, -1)
        vs_flat = self.v_scale_cache.view(num_blocks * block_size, -1)
        vz_flat = self.v_zero_cache.view(num_blocks * block_size, -1)
        slot = _s["slot"]
        k_flat[slot] = _s["k"]
        v_flat[slot] = _s["v"]
        ks_flat[slot] = _s["ks"]
        kz_flat[slot] = _s["kz"]
        vs_flat[slot] = _s["vs"]
        vz_flat[slot] = _s["vz"]

    def _r113_update_oscar_mid_batched(update_stream, key, param, handle, event, forward_context):
        """R161 P1-A: ONE batched replay updater per layer for OSCAR_MID_R113_BATCHED.

        Stages ALL B rows' carriers into the FLAT batched carrier (bit-identical
        bytes to the per-b path, laid out k_flat[b*nb_max+j]=row b block j),
        fills askv_dev[B]=mid_len[b] (0 for inactive/short rows -> the kernel
        attends 0 middle tokens -> o_h=0, l_h=-inf -> neutral in the merge),
        rewrites bt=offset-arange (the launch may mutate bt in place, R119 fix#5),
        records ONE event per layer (vs num_layers*B). Runs eagerly on the
        update_stream (legal indexed ops; no compiled per-b loop)."""
        (
            marker,
            q,
            k,
            v,
            bt,
            ks,
            ko,
            vs,
            vo,
            out_h,
            lse_h,
            mask,
            nq,
            scale,
            nkv,
            layout,
            block_size,
            Bcap,
            mid_start,
            rcnt,
            S_pad,
            impl,
            num_tokens,
            layer_idx,
            _lid,
        ) = param
        import os as _os

        Bcap = int(Bcap)
        _mk = getattr(type(impl), "_r163_markers", None)
        if _mk is None:
            _mk = type(impl)._r163_markers = set()
        if marker not in _mk:
            _mk.add(marker)
            with _odiag_o("r163_markers.log") as _f:
                _f.write(f"ACTIVE marker={marker} lid={_lid} Bcap={Bcap}\n")
        md = forward_context.attn_metadata[key]
        sl_list = md.seq_lens_list
        active = len(sl_list)
        _cp = getattr(type(impl), "_r113_cpool_batched", None)
        if _cp is None or (_lid, Bcap) not in _cp:
            raise RuntimeError("[R161-BATCHED] carriers missing at replay update (fail-closed)")
        _c = _cp[(_lid, Bcap)]
        nb_max = _c["nb_max"]
        blk_lo = int(mid_start) // int(block_size)
        kc = impl.key_cache
        vc = impl.value_cache
        k_flat, v_flat = _c["k"], _c["v"]
        ks_b, ko_b, vs_b, vo_b = _c["ks"], _c["ko"], _c["vs"], _c["vo"]
        _askv_dev, _asq_dev = _c["askv_dev"], _c["asq_dev"]
        _h1 = _os.environ.get("R156_H1", "0") == "1"
        _mega = getattr(impl, "_r156_scale_mega", None)
        # cached offset-arange for bt rewrite
        _btid = getattr(type(impl), "_r161_bt_off", None)
        if _btid is None:
            _btid = type(impl)._r161_bt_off = {}
        _bk = (Bcap, int(nb_max), str(bt.device))
        _off = _btid.get(_bk)
        if _off is None:
            _off = _btid[_bk] = (
                torch.arange(Bcap * int(nb_max), dtype=torch.int32, device=bt.device)
                .view(Bcap, int(nb_max))
                .contiguous()
            )
        # mid_len per row (0 for inactive / short -> neutral)
        mid_len_list = []
        for _b in range(Bcap):
            if _b < active:
                _ml = int(sl_list[_b]) - int(rcnt) - int(mid_start)
                if _ml < 0:
                    _ml = 0
            else:
                _ml = 0
            mid_len_list.append(_ml)
        with torch.npu.stream(update_stream):
            bt.copy_(_off)
            for _b in range(Bcap):
                _ml = mid_len_list[_b]
                if _ml <= 0:
                    continue
                _nb_now = (_ml + int(block_size) - 1) // int(block_size)
                _bt_gather = md.block_tables[_b, blk_lo : blk_lo + _nb_now]
                _slot0 = _b * nb_max
                k_flat[_slot0 : _slot0 + _nb_now].copy_(kc[_bt_gather].permute(0, 2, 1, 3))
                v_flat[_slot0 : _slot0 + _nb_now].copy_(vc[_bt_gather].permute(0, 2, 1, 3))
                _S_now = _nb_now * int(block_size)
                if _mega is not None and _h1:
                    _sc4 = _mega[:, _bt_gather]
                    _sc4f = _sc4.view(4, _S_now, int(nkv))
                    ks_b[_b, :, :_ml] = _sc4f[0, :_ml].t()
                    torch.neg(_sc4f[1, :_ml].t(), out=ko_b[_b, :, :_ml])
                    ko_b[_b, :, :_ml].mul_(ks_b[_b, :, :_ml])
                    vs_b[_b, :, :_ml] = _sc4f[2, :_ml].t()
                    torch.neg(_sc4f[3, :_ml].t(), out=vo_b[_b, :, :_ml])
                    vo_b[_b, :, :_ml].mul_(vs_b[_b, :, :_ml])
                else:
                    _ksb = impl.k_scale_cache[_bt_gather].reshape(_S_now, int(nkv))
                    _kzb = impl.k_zero_cache[_bt_gather].reshape(_S_now, int(nkv))
                    _vsb = impl.v_scale_cache[_bt_gather].reshape(_S_now, int(nkv))
                    _vzb = impl.v_zero_cache[_bt_gather].reshape(_S_now, int(nkv))
                    ks_b[_b, :, :_ml] = _ksb[:_ml].t().bfloat16()
                    ko_b[_b, :, :_ml] = (-_kzb[:_ml].t() * ks_b[_b, :, :_ml]).bfloat16()
                    vs_b[_b, :, :_ml] = _vsb[:_ml].t().bfloat16()
                    vo_b[_b, :, :_ml] = (-_vzb[:_ml].t() * vs_b[_b, :, :_ml]).bfloat16()
        # Re-issue the captured batched middle launch with the LIVE per-row
        # askv list (IntArray variant; graph_task_update replaces the baked
        # placeholder askv_cap). 0-askv rows attend 0 tokens -> o_h=0,
        # lse=-9999 (the cpp fill_) -> exp(-9999-m)=0 -> middle neutral.
        torch.npu.graph_task_update_begin(update_stream, handle)
        fr = type(impl)._r112_mod.launch_lse_fast(
            q,
            k,
            v,
            bt,
            ks,
            ko,
            vs,
            vo,
            out_h,
            lse_h,
            int(nq),
            float(scale),
            int(nkv),
            str(layout),
            int(block_size),
            1,
            1,
            mid_len_list,
            mask,
            0,
            0,
            0,
            True,
        )
        torch.npu.graph_task_update_end(update_stream)
        if not (int(fr[0]) == 0 and int(fr[2]) == 0):
            raise RuntimeError(f"[R161-BATCHED] replay launch NOT ok: {tuple(int(x) for x in fr)}")
        type(impl)._r163_stage_sr(impl, update_stream, md, _lid, Bcap, int(mid_start), int(rcnt), int(nkv))
        event.record(update_stream)
        if _os.environ.get("R155_T_NOFLUSH", "0") != "1":
            impl._r113_flush_stage(update_stream, layer_idx)

    def _r113_update_oscar_mid_v1cap(update_stream, key, param, handle, event, forward_context):
        """R163 V1CAP replay updater: staging + askv_dev CONTENT fill ONLY.

        The captured op task (V1 device-tensor variant, capacity-content at
        capture) re-executes on graph replay and reads per-row askv LIVE from
        GM — no task_update re-issue, no IntArray, no fills inside any graph
        region. Staging is byte-identical to the batched updater."""
        (
            marker,
            q,
            k,
            v,
            bt,
            ks,
            ko,
            vs,
            vo,
            out_h,
            lse_h,
            mask,
            nq,
            scale,
            nkv,
            layout,
            block_size,
            Bcap,
            mid_start,
            rcnt,
            S_pad,
            impl,
            num_tokens,
            layer_idx,
            _lid,
        ) = param
        import os as _os

        Bcap = int(Bcap)
        _mk = getattr(type(impl), "_r163_markers", None)
        if _mk is None:
            _mk = type(impl)._r163_markers = set()
        if marker not in _mk:
            _mk.add(marker)
            with _odiag_o("r163_markers.log") as _f:
                _f.write(f"ACTIVE marker={marker} lid={_lid} Bcap={Bcap}\n")
        md = forward_context.attn_metadata[key]
        sl_list = md.seq_lens_list
        active = len(sl_list)
        _cp = getattr(type(impl), "_r113_cpool_batched", None)
        if _cp is None or (_lid, Bcap) not in _cp:
            raise RuntimeError("[R163-V1CAP] carriers missing at replay update (fail-closed)")
        _c = _cp[(_lid, Bcap)]
        nb_max = _c["nb_max"]
        blk_lo = int(mid_start) // int(block_size)
        kc = impl.key_cache
        vc = impl.value_cache
        k_flat, v_flat = _c["k"], _c["v"]
        ks_b, ko_b, vs_b, vo_b = _c["ks"], _c["ko"], _c["vs"], _c["vo"]
        askv_dev, _asq_dev = _c["askv_dev"], _c["asq_dev"]
        _h1 = _os.environ.get("R156_H1", "0") == "1"
        _mega = getattr(impl, "_r156_scale_mega", None)
        _btid = getattr(type(impl), "_r161_bt_off", None)
        if _btid is None:
            _btid = type(impl)._r161_bt_off = {}
        _bk = (Bcap, int(nb_max), str(bt.device))
        _off = _btid.get(_bk)
        if _off is None:
            _off = _btid[_bk] = (
                torch.arange(Bcap * int(nb_max), dtype=torch.int32, device=bt.device)
                .view(Bcap, int(nb_max))
                .contiguous()
            )
        mid_len_list = []
        for _b in range(Bcap):
            if _b < active:
                _ml = int(sl_list[_b]) - int(rcnt) - int(mid_start)
                if _ml < 0:
                    _ml = 0
            else:
                _ml = 0
            mid_len_list.append(_ml)
        with torch.npu.stream(update_stream):
            bt.copy_(_off)
            for _b in range(Bcap):
                _ml = mid_len_list[_b]
                if _ml <= 0:
                    continue
                _nb_now = (_ml + int(block_size) - 1) // int(block_size)
                _bt_gather = md.block_tables[_b, blk_lo : blk_lo + _nb_now]
                _slot0 = _b * nb_max
                k_flat[_slot0 : _slot0 + _nb_now].copy_(kc[_bt_gather].permute(0, 2, 1, 3))
                v_flat[_slot0 : _slot0 + _nb_now].copy_(vc[_bt_gather].permute(0, 2, 1, 3))
                _S_now = _nb_now * int(block_size)
                if _mega is not None and _h1:
                    _sc4 = _mega[:, _bt_gather]
                    _sc4f = _sc4.view(4, _S_now, int(nkv))
                    ks_b[_b, :, :_ml] = _sc4f[0, :_ml].t()
                    torch.neg(_sc4f[1, :_ml].t(), out=ko_b[_b, :, :_ml])
                    ko_b[_b, :, :_ml].mul_(ks_b[_b, :, :_ml])
                    vs_b[_b, :, :_ml] = _sc4f[2, :_ml].t()
                    torch.neg(_sc4f[3, :_ml].t(), out=vo_b[_b, :, :_ml])
                    vo_b[_b, :, :_ml].mul_(vs_b[_b, :, :_ml])
                else:
                    _ksb = impl.k_scale_cache[_bt_gather].reshape(_S_now, int(nkv))
                    _kzb = impl.k_zero_cache[_bt_gather].reshape(_S_now, int(nkv))
                    _vsb = impl.v_scale_cache[_bt_gather].reshape(_S_now, int(nkv))
                    _vzb = impl.v_zero_cache[_bt_gather].reshape(_S_now, int(nkv))
                    ks_b[_b, :, :_ml] = _ksb[:_ml].t().bfloat16()
                    ko_b[_b, :, :_ml] = (-_kzb[:_ml].t() * ks_b[_b, :, :_ml]).bfloat16()
                    vs_b[_b, :, :_ml] = _vsb[:_ml].t().bfloat16()
                    vo_b[_b, :, :_ml] = (-_vzb[:_ml].t() * vs_b[_b, :, :_ml]).bfloat16()
            # askv content update ONLY (kernel reads live; stride baked at capture)
            askv_dev.copy_(torch.tensor(mid_len_list, dtype=torch.int64, device=askv_dev.device))
        type(impl)._r163_stage_sr(impl, update_stream, md, _lid, Bcap, int(mid_start), int(rcnt), int(nkv))
        event.record(update_stream)
        if _os.environ.get("R155_T_NOFLUSH", "0") != "1":
            impl._r113_flush_stage(update_stream, layer_idx)
