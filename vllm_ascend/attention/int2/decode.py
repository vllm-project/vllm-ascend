# SPDX-License-Identifier: Apache-2.0
"""Three-tier decode execution mixin (tier merge incl. empty-segment
handling). Numeric path device-bound."""

from __future__ import annotations

import os
import os as _r144_ovl_os
from typing import Any

import torch
import torch_npu

from vllm_ascend.attention.int2.ctx import (  # noqa: F401
    _odiag_o,
    _odiag_p,
)


class Int2DecodeMixin:
    # Host-state annotations only: the live attributes are owned by the
    # assembled backend / sibling mixins (bare annotations create no
    # runtime attributes, so the MRO at assembly is unaffected).
    _TT_RECENT: Any
    _dequantize_packed_int2: Any
    _ensure_tt: Any
    _oscar_R_k: Any
    _oscar_R_v: Any
    _oscar_layer_ord: Any
    _r164_bind_slots: Any
    _r198_pos: Any
    _r198_win_ridx: Any
    _tt_hp_bufs: Any
    _tt_pref: Any
    _tt_rcnt: Any
    _tt_rcnt_seen: Any
    _tt_resolve_slots: Any
    _tt_rid: Any
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

    def _forward_int2_decode_threetier(self, query, key_rot, value_rot, attn_metadata, output, layer):
        """Three-tier decode: write new rotated K/V to HP buffer, then gather
        HP prefix+recent + dequant INT2 middle, concat, single FIA call."""
        self._ensure_tt(query.device)
        # R144 P0: map batch-row b -> stable rid + reusable slot (gated, default OFF).
        if os.environ.get("R144_TT_REQID", "0") == "1":
            self._tt_resolve_slots(attn_metadata, self._oscar_layer_ord(layer), query.device, query.dtype)
        self._r164_bind_slots(attn_metadata, query.device, query.dtype, layer)
        layer_idx = getattr(layer, "layer_idx", 0)
        batch_size = len(attn_metadata.seq_lens_list)
        seq_lens = attn_metadata.seq_lens_list
        import sys as _sys

        if layer_idx == 0 and _r144_ovl_os.environ.get("R197_DBG", "0") == "1":  # R197 Fix-0
            print(
                f"[3T-DEC-DBG] ENTER batch_size={batch_size} seq_lens={seq_lens} "
                f"tt_pref={dict(self._tt_pref)} tt_rcnt={dict(self._tt_rcnt)}",
                file=_sys.stderr,
                flush=True,
            )

        # Write new decode token's rotated K/V to HP buffer
        for b in range(batch_size):
            sl = int(seq_lens[b])
            _rid = self._tt_rid(b)
            _slot = self._tt_slot(b)
            pref = self._tt_pref.get(_rid, 0)
            rcnt = self._tt_rcnt.get(_rid, 0)
            # OSCAR-faithful HP-recent window: the paged INT2 cache already holds
            # EVERY token (quantized+rotated) -- the HP buffer is a redundant
            # EXACT BF16 copy of prefix + the last `_TT_RECENT` tokens. So
            # "demote" needs no data move (sglang's unified-pool flush exists
            # because its HP/quant are distinct storage; here paged has all).
            # We just keep the HP-recent window sliding: while below capacity,
            # append; once full, slide left + append newest (drop oldest, whose
            # INT2 is already in the paged cache). This fixes the prior bug where
            # the ring filled and STOPPED -- dropping the newest tokens entirely
            # from attention and corrupting output.
            cap = self._TT_RECENT  # 512
            newk = key_rot[b : b + 1]  # [1, num_kv_heads, head_dim]  (batch-pos in input tensor)
            newv = value_rot[b : b + 1]
            _p198 = self._r198_pos(_rid, sl - 1, pref) if _r144_ovl_os.environ.get("R198_VRING", "0") == "1" else None
            if _p198 is not None:
                # R198 VRING: rotating append — ONE 1-token write per tensor,
                # no slide (the read side reassembles the window via rotated
                # indices; growing rows: pos == pref+rcnt as before).
                self._tt_write_hp(self._oscar_layer_ord(layer), _slot, _p198, (newk, newv))
                if rcnt < cap:
                    if _r144_ovl_os.environ.get("R147_RCNT_FIX", "0") == "1":
                        if self._tt_rcnt_seen.get(_rid) != sl:
                            self._tt_rcnt_seen[_rid] = sl
                            self._tt_rcnt[_rid] = rcnt + 1
                    else:
                        self._tt_rcnt[_rid] = rcnt + 1
                else:
                    self._tt_rcnt[_rid] = cap
            elif rcnt < cap:
                self._tt_write_hp(self._oscar_layer_ord(layer), _slot, pref + rcnt, (newk, newv))
                if _r144_ovl_os.environ.get("R147_RCNT_FIX", "0") == "1":
                    if self._tt_rcnt_seen.get(_rid) != sl:
                        self._tt_rcnt_seen[_rid] = sl
                        self._tt_rcnt[_rid] = rcnt + 1
                else:
                    self._tt_rcnt[_rid] = rcnt + 1
            else:
                # slide recent window [pref, pref+cap) left by one, append newest
                hk, hv = self._tt_hp_bufs(self._oscar_layer_ord(layer), query.device, query.dtype)
                hk[_slot, pref : pref + cap] = torch.cat([hk[_slot, pref + 1 : pref + cap], newk], dim=0)
                hv[_slot, pref : pref + cap] = torch.cat([hv[_slot, pref + 1 : pref + cap], newv], dim=0)
                # rcnt stays at cap (window full, sliding)

        # Now build the combined KV: HP prefix + INT2 middle + HP recent
        block_table = attn_metadata.block_tables
        block_size = self.vllm_config.cache_config.block_size
        _, _, _, packed_head = self.key_cache.shape

        # For batch_size == 1 (most common in tests), build a single contiguous KV
        total_kv_lens = []
        all_k = []
        all_v = []

        for b in range(batch_size):
            sl = int(seq_lens[b])
            _rid = self._tt_rid(b)
            _slot = self._tt_slot(b)
            pref = self._tt_pref.get(_rid, 0)
            rcnt = self._tt_rcnt.get(_rid, 0)
            mid_start = pref
            mid_end = sl - rcnt
            mid_len = mid_end - mid_start
            parts_k = []
            parts_v = []

            # HP prefix [0, pref)
            if pref > 0:
                hk, hv = self._tt_hp_bufs(self._oscar_layer_ord(layer), query.device, query.dtype)
                parts_k.append(hk[_slot, :pref])
                parts_v.append(hv[_slot, :pref])

            # INT2 middle [pref, sl-rcnt)
            if mid_len > 0:
                # OSCAR-faithful: dequant ONLY the middle blocks (prefix/recent
                # are exact BF16 from the HP buffer). softmax over
                # [prefix+middle+recent] is math-equal to the fused online-softmax.
                mid_blk_lo = mid_start // block_size
                mid_blk_hi = (mid_end + block_size - 1) // block_size
                bt = block_table[b, mid_blk_lo:mid_blk_hi]
                kb = self.key_cache[bt].contiguous()
                vb = self.value_cache[bt].contiguous()
                ks = self.k_scale_cache[bt].contiguous()
                kz = self.k_zero_cache[bt].contiguous()
                vs = self.v_scale_cache[bt].contiguous()
                vz = self.v_zero_cache[bt].contiguous()
                T = (mid_blk_hi - mid_blk_lo) * block_size
                kf = self._dequantize_packed_int2(
                    kb.view(T, self.num_kv_heads, packed_head),
                    ks.view(T, self.num_kv_heads),
                    kz.view(T, self.num_kv_heads),
                    query.dtype,
                )
                vf = self._dequantize_packed_int2(
                    vb.view(T, self.num_kv_heads, packed_head),
                    vs.view(T, self.num_kv_heads),
                    vz.view(T, self.num_kv_heads),
                    query.dtype,
                )
                off = mid_start - mid_blk_lo * block_size
                parts_k.append(kf[off : off + mid_len])
                parts_v.append(vf[off : off + mid_len])

            # HP recent [sl-rcnt, sl)
            if rcnt > 0:
                hk, hv = self._tt_hp_bufs(self._oscar_layer_ord(layer), query.device, query.dtype)
                _wi = (
                    self._r198_win_ridx(_rid, _slot, sl, rcnt, hk)
                    if _r144_ovl_os.environ.get("R198_VRING", "0") == "1"
                    else None
                )
                if _wi is not None:
                    # R198 VRING: rotated window gather (ring-only indices)
                    _nf = int(hk.shape[2]) * int(hk.shape[3])
                    parts_k.append(hk.view(-1, _nf).index_select(0, _wi).view(rcnt, hk.shape[2], hk.shape[3]))
                    parts_v.append(hv.view(-1, _nf).index_select(0, _wi).view(rcnt, hv.shape[2], hv.shape[3]))
                else:
                    parts_k.append(hk[_slot, pref : pref + rcnt])
                    parts_v.append(hv[_slot, pref : pref + rcnt])

            cat_k = (
                torch.cat(parts_k, dim=0)
                if parts_k
                else torch.empty(0, self.num_kv_heads, self.head_size, device=query.device, dtype=query.dtype)
            )
            cat_v = (
                torch.cat(parts_v, dim=0)
                if parts_v
                else torch.empty(0, self.num_kv_heads, self.head_size, device=query.device, dtype=query.dtype)
            )
            all_k.append(cat_k)
            all_v.append(cat_v)
            total_kv_lens.append(sl)

        # TND is packed-varlen: the fused op reads request i's KV from the
        # cumulative offset of actual_seq_lengths_kv[:i], so each request needs
        # its OWN rows. Allocating max(lens) and overwriting rows [0, n) per
        # request handed every request the LAST request's KV for B >= 2 (and
        # read past max_kv rows for sum(lens) > max_kv).
        kv_total = int(sum(total_kv_lens)) if total_kv_lens else 0
        key_tnd = query.new_zeros(kv_total, self.num_kv_heads, self.head_size)
        val_tnd = query.new_zeros(kv_total, self.num_kv_heads, self.head_size)
        _off = 0
        for b in range(batch_size):
            _n = total_kv_lens[b]
            key_tnd[_off : _off + _n] = all_k[b]
            val_tnd[_off : _off + _n] = all_v[b]
            _off += _n

        # Rotate Q into the same space
        query_tnd = query[:batch_size].contiguous()
        R_k = None
        R_v = None
        if self._oscar_R_k is not None and self._oscar_layer_ord(layer) < self._oscar_R_k.shape[0]:
            R_k = self._oscar_R_k[self._oscar_layer_ord(layer)]
            R_v = self._oscar_R_v[self._oscar_layer_ord(layer)]
            query_tnd = torch.matmul(query_tnd, R_k)

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        actual_seq_kvlen = total_kv_lens

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query_tnd,
            key=key_tnd,
            value=val_tnd,
            input_layout="TND",
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_kvlen,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            scale=self.scale,
            sparse_mode=3,
            atten_mask=attn_metadata.attn_mask,
        )

        if R_v is not None:
            attn_output = torch.matmul(attn_output, R_v.T)

        # ---------------- R111 A-path symmetric capture + S6 hash ----------------
        try:
            import os as _os2

            if _os2.environ.get("R111_HASH", "1") == "1":
                import hashlib as _hl2

                _h2 = _hl2.sha256(attn_output.detach().cpu().numpy().tobytes()).hexdigest()[:16]
                with _odiag_o("r111_a_hashes.jsonl") as _f2:
                    _f2.write(f'{{"p":"A","L":{layer_idx},"sl":{int(seq_lens[0])},"h":"{_h2}"}}\n')
            if _os2.environ.get("R111_CAPTURE_A", "0") == "1":
                _layers_a = set(int(x) for x in _os2.environ.get("R111_CAPT_LAYERS", "0").split(",") if x != "")
                _maxs_a = int(_os2.environ.get("R111_CAPT_STEPS", "2"))
                if layer_idx in _layers_a:
                    _dir_a = _odiag_p("direct_raw")
                    _os2.makedirs(_dir_a, exist_ok=True)
                    _step_a = 0
                    while _os2.path.exists(f"{_dir_a}/r111_acapt_L{layer_idx}_S{_step_a}.pt"):
                        _step_a += 1
                    if _step_a < _maxs_a:
                        b = 0
                        _rid0 = self._tt_rid(b)
                        sl0 = int(seq_lens[b])
                        pref0 = self._tt_pref.get(_rid0, 0)
                        rcnt0 = self._tt_rcnt.get(_rid0, 0)
                        import torch as _t2

                        _t2.save(
                            {
                                "path": "A",
                                "layer": layer_idx,
                                "step": _step_a,
                                "sl": sl0,
                                "pref": pref0,
                                "rcnt": rcnt0,
                                "mid_start": pref0,
                                "mid_end": sl0 - rcnt0,
                                "mid_len": sl0 - rcnt0 - pref0,
                                "cat_k": all_k[b].detach().cpu(),
                                "cat_v": all_v[b].detach().cpu(),
                                "q_rot": query_tnd[:1].float().cpu(),
                                "out_final": attn_output[:1].float().cpu(),
                                "R_v": None if R_v is None else R_v.detach().cpu(),
                                "scale": float(self.scale),
                                "nq": int(self.num_heads),
                                "nkv": int(self.num_kv_heads),
                                "hd": int(self.head_size),
                            },
                            f"{_dir_a}/r111_acapt_L{layer_idx}_S{_step_a}.pt",
                        )
                        print(f"[R111-CAPT-A] L{layer_idx} S{_step_a} sl={sl0}", file=_sys.stderr, flush=True)
        except Exception as _e2:
            print(f"[R111-CAPT-A] failed: {_e2}", file=_sys.stderr, flush=True)

        output[:batch_size] = attn_output
        return output
