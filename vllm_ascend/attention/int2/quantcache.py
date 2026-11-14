# SPDX-License-Identifier: Apache-2.0
"""INT2 quantize/cache-write mixin for the INT2 backend."""

from __future__ import annotations

import os
import os as _r144_ovl_os
from typing import Any

import torch

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.int2.ctx import (  # noqa: F401
    _OSCAR_CFG,
    _R2XX_G,
    AscendMetadata,
    AttentionLayer,
    _odiag_p,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm_ascend.attention.int2.helpers import (  # noqa: F401
    _r150_tr,
    _r166_tr,
)


class Int2QuantCacheMixin:
    # Host-state annotations only: the cache-interface attributes
    # (device/shape/num_kv_heads/...) and the rotation tensors are owned by
    # the assembled backend; _oscar_layer_ord lives on the slot-state mixin.
    # Bare annotations create no runtime attributes; they pin types for
    # mypy --check-untyped-defs.
    key_cache: Any
    value_cache: Any
    _oscar_R_k: Any
    _oscar_R_v: Any
    _oscar_layer_ord: Any
    _r112_mod: Any
    _r113_qdbg: Any
    _r113_stage: Any
    device: Any
    head_size: int
    is_kv_producer: bool
    num_kv_heads: int
    shape: Any
    vllm_config: Any

    def _prepare_int2_scales(self, layer, device):
        """Shard per-channel INT2 scales and cache as attributes."""
        if hasattr(layer, "_int2_scales_prepared"):
            return

        def _shard(raw):
            if raw.numel() == 1:
                return raw.to(device=device)
            expected = self.num_kv_heads * self.head_size
            if raw.numel() != expected:
                total_kv_heads = raw.numel() // self.head_size
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                kv_head_start = tp_rank * total_kv_heads // tp_size
                raw = raw.view(total_kv_heads, self.head_size)[
                    kv_head_start : kv_head_start + self.num_kv_heads
                ].contiguous()
            return raw.view(1, self.num_kv_heads, self.head_size).to(device=device)

        # Check if scales exist, create default ones if not
        if not hasattr(layer, "k_cache_scale"):
            # Create default scales (all ones) for INT2 quantization
            layer.k_cache_scale = torch.nn.Parameter(
                torch.ones(self.num_kv_heads, self.head_size, device=device), requires_grad=False
            )
        if not hasattr(layer, "v_cache_scale"):
            layer.v_cache_scale = torch.nn.Parameter(
                torch.ones(self.num_kv_heads, self.head_size, device=device), requires_grad=False
            )

        layer._int2_k_scale = _shard(layer.k_cache_scale.data)
        layer._int2_v_scale = _shard(layer.v_cache_scale.data)
        layer._int2_k_inv_scale = 1.0 / layer._int2_k_scale
        layer._int2_v_inv_scale = 1.0 / layer._int2_v_scale
        layer._int2_scales_prepared = True

    @staticmethod
    def _pack_int2_to_uint8(x_int8: torch.Tensor) -> torch.Tensor:
        """Pack int8 tensor with values in [-2, 1] into uint8.

        Last dim must be divisible by 4. Packs 4 consecutive values per byte:
            byte = (v0+2) | ((v1+2)<<2) | ((v2+2)<<4) | ((v3+2)<<6)
        """
        orig_shape = x_int8.shape
        head_size = orig_shape[-1]
        assert head_size % 4 == 0, f"head_size {head_size} not divisible by 4"

        x_reshaped = x_int8.view(*orig_shape[:-1], head_size // 4, 4)
        x_unsigned = (x_reshaped.to(torch.int16) + 2).to(torch.uint8) & 0x03

        packed = (
            x_unsigned[..., 0].to(torch.uint8)
            | (x_unsigned[..., 1].to(torch.uint8) << 2)
            | (x_unsigned[..., 2].to(torch.uint8) << 4)
            | (x_unsigned[..., 3].to(torch.uint8) << 6)
        )
        return packed

    @staticmethod
    def _unpack_uint8_to_int2(packed: torch.Tensor) -> torch.Tensor:
        """Unpack uint8 tensor back to int8 with values in [-2, 1]."""
        packed_shape = packed.shape
        head_size_packed = packed_shape[-1]
        head_size = head_size_packed * 4

        p = packed.to(torch.uint8)
        v0 = (p >> 0) & 0x03
        v1 = (p >> 2) & 0x03
        v2 = (p >> 4) & 0x03
        v3 = (p >> 6) & 0x03

        unpacked_unsigned = torch.stack([v0, v1, v2, v3], dim=-2)
        unpacked_unsigned = unpacked_unsigned.view(*packed_shape[:-1], head_size)
        unpacked = unpacked_unsigned.to(torch.int8) - 2
        return unpacked

    def _quantize_kv_to_int2(self, key, value, layer, num_actual_tokens, pre_rotated_kv=None):
        """Quantize K/V to packed INT2 with DYNAMIC per-token-head affine scales.

        Mirrors OSCAR exactly: rotate (K@R_k, V@R_v) -> clip outliers ->
        affine INT2 [0,3] with per-token-head scale+zero -> pack 4/byte.
        The static scale=1.0 path (q=clamp(round(x),-2,1)) destroyed signal;
        OSCAR uses dynamic scale=(max-min)/3, zero=-min/scale, q in [0,3].

        If ``pre_rotated_kv`` is given as ``(k_rot, v_rot)``, skips internal
        rotation (caller already applied R_k/R_v). Used by the three-tier path
        to avoid double-rotating.
        """
        from vllm_ascend.oscar.int2_quant import oscar_quantize_int2

        _r150_tr(
            "qfn_entry",
            T=int(num_actual_tokens),
            prerot=bool(pre_rotated_kv is not None),
            cppflag=os.environ.get("R113_QUANT_CPP", "0"),
            mod112=bool(getattr(type(self), "_r112_mod", None) is not None),
        )
        _r166_tr(
            "quant",
            T=int(num_actual_tokens),
            prerot=bool(pre_rotated_kv is not None),
            lid=int(self._oscar_layer_ord(layer)) if layer is not None else -1,
            cap=bool(_EXTRA_CTX.capturing),
        )

        actual_key = key[:num_actual_tokens].contiguous()
        actual_value = value[:num_actual_tokens].contiguous()

        # Apply Oscar rotation (K/V are stored rotated in the INT2 cache).
        # Skip if caller passes pre-rotated KV (e.g., three-tier path).
        if os.environ.get("R168_QIN", "0") == "1":
            # R168 P1: input-side checksums of the REAL legacy quantize path
            # (per-128-row segments; same prompt across runs => comparable).
            try:
                import json as _jq168

                _Tq = int(actual_key.shape[0])
                _rowq = actual_key.detach().float().abs().reshape(_Tq, -1).sum(dim=1)
                _segq = [round(float(_rowq[i * 128 : (i + 1) * 128].sum()), 2) for i in range(min(40, -(-_Tq // 128)))]
                with open("/tmp/r168_qin.jsonl", "a") as _fq168:
                    _fq168.write(
                        _jq168.dumps(
                            {
                                "T": _Tq,
                                "lid": int(self._oscar_layer_ord(layer)) if layer is not None else -1,
                                "seg": _segq,
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
        if pre_rotated_kv is not None:
            actual_key, actual_value = pre_rotated_kv
        elif getattr(self, "_oscar_R_k", None) is not None and getattr(self, "_oscar_R_v", None) is not None:
            if self._oscar_layer_ord(layer) < self._oscar_R_k.shape[0]:
                R_k = self._oscar_R_k[self._oscar_layer_ord(layer)]
                R_v = self._oscar_R_v[self._oscar_layer_ord(layer)]
                actual_key = torch.matmul(actual_key, R_k)
                actual_value = torch.matmul(actual_value, R_v)

        # R113 Fallback-1b: single-call C++ K/V quantize (bit-exact values; one
        # pybind crossing instead of ~58 dispatches). Pre-rotated path only.
        if (
            pre_rotated_kv is not None
            and _OSCAR_CFG.cpp_quantizer
            and getattr(type(self), "_r112_mod", None) is not None
        ):
            _r150_tr("cpp_quant_taken", T=int(num_actual_tokens), kshape=list(actual_key.shape))
            k_packed, k_scale, k_zero, v_packed, v_scale, v_zero = type(self)._r112_mod.oscar_quant_kv(
                actual_key, actual_value, 0.96, 0.92
            )
            if os.environ.get("R113_QUANT_SYNC", "0") == "1":
                # R113: device-level barrier required after C++-batched quant —
                # the pack write is consumed by the paged scatter across an
                # ordering boundary stream-level syncs do not cover (race class
                # proven: values bit-exact at all T; probe-recompute and device
                # sync both mask; stream guard/stream sync do not).
                torch.npu.synchronize()
            if os.environ.get("R113_QUANT_DBG", "0") == "1":
                if not hasattr(type(self), "_r113_qdbg"):
                    type(self)._r113_qdbg = 0
                type(self)._r113_qdbg += 1
                if type(self)._r113_qdbg <= 3:
                    import torch as _t

                    pk, ps, pz = oscar_quantize_int2(actual_key, clip_ratio=0.96)
                    qv_, qs_, qz_ = oscar_quantize_int2(actual_value, clip_ratio=0.92)
                    print(
                        f"R113QDBG n={type(self)._r113_qdbg} kin={tuple(actual_key.shape)} {actual_key.dtype} "
                        f"contig={actual_key.is_contiguous()} kpacked_eq={_t.equal(pk, k_packed)} "
                        f"kscale_eq={_t.equal(ps, k_scale)} kzero_eq={_t.equal(pz, k_zero)} "
                        f"vpacked_eq={_t.equal(qv_, v_packed)} vscale_eq={_t.equal(qs_, v_scale)} "
                        f"vzero_eq={_t.equal(qz_, v_zero)}",
                        flush=True,
                    )
                    if not _t.equal(pk, k_packed):
                        _bad = (pk != k_packed).nonzero()[:3]
                        print(
                            f"R113QDBG first_bad={_bad.tolist()} "
                            f"py={pk.flatten()[:8].tolist()} "
                            f"cpp={k_packed.flatten()[:8].tolist()}",
                            flush=True,
                        )
            # R218 live-route KV dump (default OFF). The block below the Python
            # path is DEAD under R113_QUANT_CPP=1 (this early return precedes it),
            # so the live cache is produced by the C++ op and was never snapshotted.
            # File-based on purpose: this runs in the V1 engine-core child, whose
            # stdout is NOT forwarded to the serve log (print() is silently lost).
            if os.environ.get("R218_QINDUMP", "0") == "1":
                _qst = globals().setdefault("_R218_QIN_N", [0])
                # R218_QINDUMP_LAYER: -1 = no filter (default); >=0 dumps only that
                # oscillator-layer ordinal, so per-ROW comparison is unambiguous
                # (each call is one row × one layer, T == that row's token count).
                _dlay = int(os.environ.get("R218_QINDUMP_LAYER", "-1"))
                _lay_i = int(self._oscar_layer_ord(layer)) if layer is not None else -1
                if _qst[0] < int(os.environ.get("R218_QINDUMP_MAX", "2")) and (_dlay < 0 or _lay_i == _dlay):
                    _qi = _qst[0]
                    _qst[0] += 1
                    try:
                        _dq = os.environ.get("R218_QINDUMP_DIR", _odiag_p("r218_qin"))
                        os.makedirs(_dq, exist_ok=True)
                        torch.save(
                            {
                                "idx": _qi,
                                "route": "cpp",
                                "layer": int(self._oscar_layer_ord(layer)) if layer is not None else -1,
                                "T": int(num_actual_tokens),
                                "pre_rotated": bool(pre_rotated_kv is not None),
                                "key_in": actual_key.detach().cpu().clone(),
                                "value_in": actual_value.detach().cpu().clone(),
                                "k_packed": k_packed.detach().cpu().clone(),
                                "k_scale": k_scale.detach().cpu().clone(),
                                "k_zero": k_zero.detach().cpu().clone(),
                                "v_packed": v_packed.detach().cpu().clone(),
                                "v_scale": v_scale.detach().cpu().clone(),
                                "v_zero": v_zero.detach().cpu().clone(),
                            },
                            f"{_dq}/qin_{_qi:04d}.pt",
                        )
                    except Exception:
                        pass
            return k_packed, k_scale, k_zero, v_packed, v_scale, v_zero

        k_packed, k_scale, k_zero = oscar_quantize_int2(actual_key, clip_ratio=0.96)
        v_packed, v_scale, v_zero = oscar_quantize_int2(actual_value, clip_ratio=0.92)
        # R218 quantizer-layout diagnostic (default OFF): keep the exact BF16
        # quantizer INPUT with its packed/scale/zero OUTPUT. The Python packer
        # groups head_dim as [4][32] while packing, but its dequant (and the
        # kernel/adjudicator lane convention) treats each byte as 4 CONTIGUOUS
        # dims — this snapshot decides which convention the live cache carries.
        if os.environ.get("R218_QINDUMP", "0") == "1":
            _qst = globals().setdefault("_R218_QIN_N", [0])
            if _qst[0] < int(os.environ.get("R218_QINDUMP_MAX", "2")):
                _qi = _qst[0]
                _qst[0] += 1
                try:
                    import torch as _tq

                    _dq = os.environ.get("R218_QINDUMP_DIR", _odiag_p("r218_qin"))
                    os.makedirs(_dq, exist_ok=True)
                    _tq.save(
                        {
                            "idx": _qi,
                            "route": "py",
                            "layer": int(self._oscar_layer_ord(layer)) if layer is not None else -1,
                            "T": int(num_actual_tokens),
                            "pre_rotated": bool(pre_rotated_kv is not None),
                            "key_in": actual_key.detach().cpu().clone(),
                            "value_in": actual_value.detach().cpu().clone(),
                            "k_packed": k_packed.detach().cpu().clone(),
                            "k_scale": k_scale.detach().cpu().clone(),
                            "k_zero": k_zero.detach().cpu().clone(),
                            "v_packed": v_packed.detach().cpu().clone(),
                            "v_scale": v_scale.detach().cpu().clone(),
                            "v_zero": v_zero.detach().cpu().clone(),
                        },
                        f"{_dq}/qin_{_qi:04d}.pt",
                    )
                except Exception:
                    pass
        return k_packed, k_scale, k_zero, v_packed, v_scale, v_zero

    def _dequantize_packed_int2(self, packed, scale, zero, target_dtype):
        """Unpack + dequant: x = (q - zero) * scale (dynamic per-token-head)."""
        from vllm_ascend.oscar.int2_quant import oscar_dequantize_int2

        return oscar_dequantize_int2(packed, scale, zero, target_dtype)

    def _ensure_int2_scale_buffers(self, k_cache, block_size, num_kv_heads, dtype, device):
        """Lazy-allocate per-token-head scale/zero buffers parallel to the
        data cache. Shape [num_blocks, block_size, num_kv_heads], one entry
        per cached token, matching the paged layout for gather in decode."""
        if getattr(self, "k_scale_cache", None) is None:
            num_blocks = k_cache.shape[0]
            z = (num_blocks, block_size, num_kv_heads)
            import os as _os156c

            if _os156c.environ.get("R156_H4", "0") == "1":
                # R156-H4: GLOBAL scale mega [L,4,nb,bs,nkv]; this impl's four
                # caches are views at its ordinal (creation order == layer
                # order at model load). Same H1C semantics per impl.
                _NL4 = int(getattr(self.vllm_config.model_config, "num_hidden_layers", 36))
                _gd = globals().setdefault("_R156_H4_SCALE", {})
                _gk = (str(device), _NL4, num_blocks, block_size, num_kv_heads)
                if _gk not in _gd:
                    _gd[_gk] = {"t": torch.zeros((_NL4, 4) + z, dtype=dtype, device=device), "n": 0}
                _g4 = _gd[_gk]
                _o4 = _g4["n"]
                _g4["n"] += 1
                self._r156_scale_mega = _g4["t"][_o4]
                self.k_scale_cache = self._r156_scale_mega[0]
                self.k_zero_cache = self._r156_scale_mega[1]
                self.v_scale_cache = self._r156_scale_mega[2]
                self.v_zero_cache = self._r156_scale_mega[3]
            elif _os156c.environ.get("R156_H1C", "0") == "1":
                # R156-H1c: ONE mega scale cache [4, nb, bs, nkv]; the four
                # per-name caches are VIEWS (identical bytes, identical
                # shapes for every existing consumer) so the replay staging
                # can gather all four page-blocks in ONE index launch.
                self._r156_scale_mega = torch.zeros((4,) + z, dtype=dtype, device=device)
                self.k_scale_cache = self._r156_scale_mega[0]
                self.k_zero_cache = self._r156_scale_mega[1]
                self.v_scale_cache = self._r156_scale_mega[2]
                self.v_zero_cache = self._r156_scale_mega[3]
            else:
                self.k_scale_cache = torch.zeros(z, dtype=dtype, device=device)
                self.k_zero_cache = torch.zeros(z, dtype=dtype, device=device)
                self.v_scale_cache = torch.zeros(z, dtype=dtype, device=device)
                self.v_zero_cache = torch.zeros(z, dtype=dtype, device=device)

    def _reshape_and_cache_int2(
        self,
        query,
        key_packed,
        k_scale,
        k_zero,
        value_packed,
        v_scale,
        v_zero,
        kv_cache,
        attn_metadata,
        output,
        layer_idx=None,
    ):
        """Write packed INT2 KV + per-token-head scales/zeros to paged cache."""
        if len(kv_cache) <= 1:
            return query, key_packed, value_packed, output
        if self.is_kv_producer:
            attn_metadata.reshape_cache_event = torch.npu.Event()
        k_cache, v_cache = self.key_cache, self.value_cache
        if k_cache is None:
            k_cache, v_cache = kv_cache[0], kv_cache[1]
            self.key_cache, self.value_cache = k_cache, v_cache
        slots = attn_metadata.slot_mapping
        _bt_r = getattr(attn_metadata, "block_tables", None)
        # R203_METAFIX: the R166_REFTRACE probe is default-OFF, but its
        # ARGUMENTS were evaluated unconditionally — block-table .tolist() +
        # slots[0/-1].item() = 3 device-to-host stream syncs per layer feeding
        # a probe that discards them (kprof r202b: part of the 12 aten::item
        # + long-aten::to gap population per pass-2 layer). Gate the whole
        # argument block behind the probe env; probe-ON legs are unchanged.
        if os.environ.get("R166_REFTRACE", "0") == "1" or os.environ.get("R203_METAFIX", "0") != "1":
            _bt_head = None
            if _bt_r is not None and getattr(_bt_r, "numel", lambda: 0)() and _bt_r.dim() >= 2:
                _bt_head = [int(x) for x in _bt_r[0, :10].tolist()]
            _r166_tr(
                "paged_write",
                T=int(key_packed.shape[0]),
                cap=bool(_EXTRA_CTX.capturing),
                s_head=int(slots[0].item()) if slots.numel() else -1,
                s_tail=int(slots[-1].item()) if slots.numel() else -1,
                bt=_bt_head,
                lid=int(layer_idx) if layer_idx is not None else -1,
                st=str(getattr(attn_metadata, "attn_state", "?")),
            )
        block_size = self.vllm_config.cache_config.block_size
        num_tokens = key_packed.shape[0]
        num_blocks = k_cache.shape[0]
        device = k_cache.device
        dtype = k_scale.dtype

        self._ensure_int2_scale_buffers(k_cache, block_size, k_cache.shape[2], dtype, device)

        k_flat = k_cache.view(num_blocks * block_size, *k_cache.shape[2:])
        v_flat = v_cache.view(num_blocks * block_size, *v_cache.shape[2:])
        # R113 G-A: indexed scatter (index_put_/index_copy_) FREEZES its index
        # args inside an aclgraph — replay writes to stale slots (probe2-proven
        # for put AND copy). Under capture we instead write the packed KV +
        # scales + slot into STATIC per-layer staging (plain copy_, static
        # addressing, content-refresh-safe); the update callback flushes the
        # previous step's staging EAGERLY on the update stream. Safety: the new
        # token belongs to the RECENT tier (HP buffer, in-graph slide), and the
        # middle span only reads tokens >= 512 steps old, so the one-step flush
        # lag never affects any read.
        if _EXTRA_CTX.capturing and layer_idx is not None and os.environ.get("R109_THREETIER_MIDDLE", "0") == "1":
            if not hasattr(type(self), "_r113_stage"):
                type(self)._r113_stage = {}
            _stg = type(self)._r113_stage
            _s = _stg.get(layer_idx)
            # first use, or a step whose token count differs from the one
            # that sized the staging tensors (split-prefill continuation, or
            # a SMALLER step after a larger one): reallocate — every field is
            # fully re-copied below, so nothing is lost. Size-exact also
            # keeps the flush consumer's indexed write in lockstep with the
            # staged slot tensor: an oversize staging would either raise in
            # copy_ (shape mismatch) or flush stale zero slots into row 0.
            if _s is None or _s["k"].shape[0] != num_tokens:
                _stg[layer_idx] = _s = {
                    "k": torch.zeros_like(key_packed[:num_tokens]),
                    "v": torch.zeros_like(value_packed[:num_tokens]),
                    "ks": torch.zeros_like(k_scale[:num_tokens]),
                    "kz": torch.zeros_like(k_zero[:num_tokens]),
                    "vs": torch.zeros_like(v_scale[:num_tokens]),
                    "vz": torch.zeros_like(v_zero[:num_tokens]),
                    "slot": torch.zeros_like(slots[:num_tokens]),
                    "flushed": False,
                }
            _s["k"].copy_(key_packed[:num_tokens])
            _s["v"].copy_(value_packed[:num_tokens])
            _s["ks"].copy_(k_scale[:num_tokens])
            _s["kz"].copy_(k_zero[:num_tokens])
            _s["vs"].copy_(v_scale[:num_tokens])
            _s["vz"].copy_(v_zero[:num_tokens])
            _s["slot"].copy_(slots[:num_tokens])
            _s["flushed"] = False
            return query, key_packed, value_packed, output
        if os.environ.get("R168_SCIN", "0") == "1":
            # R168 P1: scatter-input alignment probe. For each prefill scatter
            # call record T, boundary slots, slot deltas, and packed-row
            # checksums. Same tokens across calls MUST map to the same slots;
            # any drift = slot_mapping shift (the chunk-dependence writer).
            try:
                import json as _js168

                _Ts = int(num_tokens)
                _sl = slots[:_Ts].to(torch.long)
                _pk = key_packed[:_Ts].float().abs().reshape(_Ts, -1).sum(dim=1)
                _d = _sl[1:] - _sl[:-1]
                _dd = torch.bincount(_d[_d >= 0] // 128, minlength=8)[:8].tolist() if _Ts > 1 else [0]
                with open("/tmp/r168_scin.jsonl", "a") as _fs168:
                    _fs168.write(
                        _js168.dumps(
                            {
                                "T": _Ts,
                                "lid": int(layer_idx) if layer_idx is not None else -1,
                                "s0": int(_sl[0]),
                                "s1": int(_sl[min(1, _Ts - 1)]),
                                "sN": int(_sl[-1]),
                                "hist128": _dd,
                                "rows": [
                                    round(float(x), 2)
                                    for x in _pk[:4].tolist()
                                    + _pk[max(0, _Ts // 2) : _Ts // 2 + 2].tolist()
                                    + _pk[-4:].tolist()
                                ],
                            }
                        )
                        + "\n"
                    )
            except Exception as _es168:
                print(f"[R168-SCIN] fail: {_es168}", flush=True)
        if os.environ.get("R113_SCATTER_CPP", "0") == "1" and getattr(type(self), "_r112_mod", None) is not None:
            # R113 Fallback-1c: single-call C++ paged scatter (msprof: the 6
            # index_put writes cost ~19ms/step host; same kernels, one crossing)
            type(self)._r112_mod.oscar_scatter_int2(
                k_cache,
                v_cache,
                self.k_scale_cache,
                self.k_zero_cache,
                self.v_scale_cache,
                self.v_zero_cache,
                slots,
                int(num_tokens),
                key_packed,
                value_packed,
                k_scale,
                k_zero,
                v_scale,
                v_zero,
            )
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
            return query, key_packed, value_packed, output
        if _r144_ovl_os.environ.get("R166_STREAMFIX", "0") in ("paged", "both"):
            # explicit compute-friendly scatter (index_put_ advanced-assign may
            # route via the copy engine on some shapes; index_copy_ with a
            # device index must read indices on device => compute kernel)
            _sl64 = slots[:num_tokens].to(torch.long)
            k_flat.index_copy_(0, _sl64, key_packed)
            v_flat.index_copy_(0, _sl64, value_packed)
        else:
            k_flat[slots[:num_tokens]] = key_packed
            v_flat[slots[:num_tokens]] = value_packed

        ks_flat = self.k_scale_cache.view(num_blocks * block_size, -1)
        kz_flat = self.k_zero_cache.view(num_blocks * block_size, -1)
        vs_flat = self.v_scale_cache.view(num_blocks * block_size, -1)
        vz_flat = self.v_zero_cache.view(num_blocks * block_size, -1)
        ks_flat[slots[:num_tokens]] = k_scale
        kz_flat[slots[:num_tokens]] = k_zero
        vs_flat[slots[:num_tokens]] = v_scale
        vz_flat[slots[:num_tokens]] = v_zero
        # R166_WVERIFY (default OFF, causal one-shot): does the paged write's
        # CONTENT land? Post-write full-block checksum of the FIRST written
        # block (sync = perturbing; diagnostic legs only). Distinguishes
        # write-never-lands (dummy uniform content) vs lands-then-overwritten.
        # R166_WSYNC (default OFF, causal): full sync after each prefill paged
        # write — isolates the paged-write->(update-gather|replay-read) edge.
        if os.environ.get("R166_WSYNC", "0") == "1":
            torch.npu.synchronize()
        if os.environ.get("R166_WVERIFY", "0") == "1":
            try:
                _vn = globals().setdefault("_R166_WV_N", [0])
                if _vn[0] < 80:
                    _vn[0] += 1
                    torch.npu.synchronize()
                    _b0 = int(slots[0].item()) // int(block_size)
                    _sum = float(k_cache[_b0].to(torch.int16).sum().item())
                    _src = float(key_packed[:block_size].to(torch.int16).sum().item())
                    _r166_tr(
                        "wverify",
                        lid=int(layer_idx) if layer_idx is not None else -1,
                        block=_b0,
                        blk_sum=round(_sum, 1),
                        src_sum=round(_src, 1),
                    )
                    _R2XX_G.setdefault("_R166_WV_BLOCKS", {})[
                        (int(layer_idx) if layer_idx is not None else -1, _b0)
                    ] = round(_src, 1)
            except Exception:
                pass

        if self.is_kv_producer:
            attn_metadata.reshape_cache_event.record()
        return query, key_packed, value_packed, output

    def _gather_and_dequantize_int2_cache(
        self, attn_metadata: AscendMetadata, layer: AttentionLayer, target_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged INT2 KV blocks and dequantize to float."""
        num_block, block_size, _, packed_head = self.key_cache.shape
        batch_size = len(attn_metadata.seq_lens_list)
        max_blocks = attn_metadata.block_tables.shape[1]

        key_gathered = self.key_cache[attn_metadata.block_tables]
        value_gathered = self.value_cache[attn_metadata.block_tables]

        key_gathered = key_gathered.view(batch_size, max_blocks * block_size, self.num_kv_heads, packed_head)
        value_gathered = value_gathered.view(batch_size, max_blocks * block_size, self.num_kv_heads, packed_head)

        ks = self.k_scale_cache[attn_metadata.block_tables].view(batch_size, max_blocks * block_size, self.num_kv_heads)
        kz = self.k_zero_cache[attn_metadata.block_tables].view(batch_size, max_blocks * block_size, self.num_kv_heads)
        vs = self.v_scale_cache[attn_metadata.block_tables].view(batch_size, max_blocks * block_size, self.num_kv_heads)
        vz = self.v_zero_cache[attn_metadata.block_tables].view(batch_size, max_blocks * block_size, self.num_kv_heads)

        key_float = self._dequantize_packed_int2(key_gathered, ks, kz, target_dtype)
        value_float = self._dequantize_packed_int2(value_gathered, vs, vz, target_dtype)

        return key_float, value_float
