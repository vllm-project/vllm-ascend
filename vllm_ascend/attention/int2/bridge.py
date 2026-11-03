# SPDX-License-Identifier: Apache-2.0
"""Launcher-extension loading mixin for the INT2 backend (r91/r112 fast loader, r135 Tensor-ABI variant)."""

from __future__ import annotations

import sys as _r144_ovl_sys
from typing import Any

import torch

from vllm_ascend.attention.int2.ctx import (  # noqa: F401
    _R2XX_G,
    logger,
)
from vllm_ascend.attention.int2.helpers import (  # noqa: F401
    _r218_diag,
)


class Int2BridgeMixin:
    # Host-state annotations: the bridge handles/counters live on the CLASS
    # (shared across impls, initialized lazily by the loaders below); the
    # cache-interface and rotation names are owned by the assembled backend
    # / sibling mixins. Class-level defaults (None) because the loaders and
    # readers go through type(self).
    _r91_loaded: Any = None
    _r91_mod: Any = None
    _r91_entry_count: int = 0
    _r91_fallback_count: int = 0
    _r91_getws_count: int = 0
    _r91_launch_count: int = 0
    _r91_sync_count: int = 0
    _r91_output_consumed_count: int = 0
    _r112_loaded: Any = None
    _r112_mod: Any = None
    _r135_loaded: Any = None
    _r135_mod: Any = None
    _oscar_R_k: Any
    _oscar_R_v: Any
    _oscar_layer_ord: Any
    head_size: int
    k_scale_cache: Any
    k_zero_cache: Any
    key_cache: Any
    num_heads: int
    num_kv_heads: int
    v_scale_cache: Any
    v_zero_cache: Any
    value_cache: Any

    def _r91_ensure_loaded(self, device):
        """Build and load the R91 C++ extension (class-level, once)."""
        import os as _os

        if type(self)._r91_loaded:
            return type(self)._r91_mod is not None
        type(self)._r91_loaded = True
        try:
            from vllm_ascend.oscar import paths as _opaths91

            # Vendor root: same OPP chain resolution as the fast launcher
            # (_r112_ensure_fast); the R98 bridge links -lcust_opapi from the
            # first chain entry and builds its aclnn headers from there.
            _r157opp = _os.environ.get("R157_OPP")
            _chain91 = (_r157opp + ":") if _r157opp else ""
            if _os.environ.get("R144_OPP_PATH"):
                _chain91 += _os.environ["R144_OPP_PATH"]
            elif _os.environ.get("OSCAR_OPP_PATH"):
                _chain91 += _os.environ["OSCAR_OPP_PATH"]
            VENDOR = _chain91.split(":")[0] if _chain91 else ""
            if not VENDOR:
                logger.error("[R91] vendor OPP path not set (R144_OPP_PATH/OSCAR_OPP_PATH); middle tier disabled")
                return False
            _os.environ.setdefault("TORCH_EXTENSIONS_DIR", _opaths91.runner_tmp("torch_ext"))
            _os.makedirs(_os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
            _os.environ.setdefault("KERNEL_META", _opaths91.runner_tmp("kernel_meta"))
            _os.makedirs(_os.environ["KERNEL_META"], exist_ok=True)
            _os.environ["ASCEND_CUSTOM_OPP_PATH"] = _chain91
            # Pre-load API library via ctypes RTLD_GLOBAL
            import ctypes as _ct

            _ct.CDLL("/usr/local/Ascend/driver/lib64/driver/libascend_hal.so", mode=_ct.RTLD_GLOBAL)
            _ct.CDLL(VENDOR + "/op_api/lib/libcust_opapi.so", mode=_ct.RTLD_GLOBAL)
            import torch_npu as _tn
            from torch.utils import cpp_extension as _ce

            # R98 LSE bridge source: repo csrc by default (OSCAR_LSE_BRIDGE_SRC
            # overrides, e.g. for an out-of-tree frozen copy).
            LSE_SRC = _opaths91.lse_bridge_src()
            TN = _os.path.dirname(_tn.__file__ or "")
            _cann = _os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
            INC = [
                _os.path.join(TN, "include"),
                _os.path.join(TN, "include", "third_party", "op-plugin"),
                _cann + "/aarch64-linux/include",
                _cann + "/include",
                VENDOR + "/op_api/include",
            ]
            LD = [
                "-L" + VENDOR + "/op_api/lib",
                "-lcust_opapi",
                "-L" + _cann + "/lib64",
                "-lascendcl",
                "-L" + _os.path.join(TN, "lib"),
                "-ltorch_npu",
                "-Wl,-rpath," + VENDOR + "/op_api/lib",
                "-Wl,-rpath," + _cann + "/lib64",
            ]
            type(self)._r91_mod = _ce.load(
                name="int2_op_launch_lse_r108",
                sources=[LSE_SRC],
                extra_include_paths=INC,
                extra_ldflags=LD,
                verbose=False,
            )
            print(f"[R108-LSE] Extension loaded: {type(self)._r91_mod.r98_lse_version()}", flush=True)
            return type(self)._r91_mod is not None
        except Exception as e:
            print(f"[R96-R91] Load failed: {e}", flush=True)
            return False

    def _r112_ensure_fast(self, device):
        """R112: sync-free launcher (no npuSynchronizeDevice, no D2H, no .item(),
        cached workspace). Env-gated R112_LSE_FAST=1; FAIL-CLOSED on any error."""
        import os as _os

        if type(self)._r112_loaded:
            return type(self)._r112_mod is not None
        type(self)._r112_loaded = True
        try:
            from vllm_ascend.oscar import paths as _opaths

            # Vendor OPP resolution. R144_OPP_PATH carries the FULL multi-
            # vendor colon chain composed by the serve harness (candidate
            # first, then the R133V/R116V/R106V support vendors whose ops
            # must keep resolving) -- propagate it verbatim; do NOT collapse
            # to the first entry. Fail-closed when nothing is set (no silent
            # stale home-directory fallback).
            _r157opp = _os.environ.get("R157_OPP")
            _chain = (_r157opp + ":") if _r157opp else ""
            if _os.environ.get("R144_OPP_PATH"):
                _chain += _os.environ["R144_OPP_PATH"]
            elif _os.environ.get("OSCAR_OPP_PATH"):
                _chain += _os.environ["OSCAR_OPP_PATH"]
            if not _chain:
                raise FileNotFoundError("vendor OPP path not set (R144_OPP_PATH/OSCAR_OPP_PATH)")
            VENDOR = _chain.split(":")[0]
            _os.environ.setdefault("TORCH_EXTENSIONS_DIR", _opaths.runner_tmp("torch_ext"))
            _os.environ.setdefault("KERNEL_META", _opaths.runner_tmp("kernel_meta"))
            _os.environ["ASCEND_CUSTOM_OPP_PATH"] = _chain
            import ctypes as _ct

            _ct.CDLL("/usr/local/Ascend/driver/lib64/driver/libascend_hal.so", mode=_ct.RTLD_GLOBAL)
            _ct.CDLL(VENDOR + "/op_api/lib/libcust_opapi.so", mode=_ct.RTLD_GLOBAL)
            import torch_npu as _tn
            from torch.utils import cpp_extension as _ce

            TN = _os.path.dirname(_tn.__file__)
            _cann = _os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
            INC = [
                _os.path.join(TN, "include"),
                _os.path.join(TN, "include", "third_party", "op-plugin"),
                _cann + "/aarch64-linux/include",
                _cann + "/include",
                VENDOR + "/op_api/include",
            ]
            LD = [
                "-L" + VENDOR + "/op_api/lib",
                "-lcust_opapi",
                "-L" + _cann + "/lib64",
                "-lascendcl",
                "-L" + _os.path.join(TN, "lib"),
                "-ltorch_npu",
                "-Wl,-rpath," + VENDOR + "/op_api/lib",
                "-Wl,-rpath," + _cann + "/lib64",
            ]
            type(self)._r112_mod = _ce.load(
                name=_opaths.launcher_name(),
                sources=[_opaths.launcher_src()],
                extra_include_paths=INC,
                extra_ldflags=LD,
                verbose=False,
            )
            print(f"[R112-FAST] Extension loaded: {type(self)._r112_mod.r112fast_version()}", flush=True)
            return type(self)._r112_mod is not None
        except Exception as e:
            import os as _os2
            import traceback as _tb2

            _os2.write(2, f"[R112-FAST] Load failed: {e}\n{_tb2.format_exc()[:1500]}\n".encode())
            return False

    def _r135_ensure_v1(self, device):
        """R135: load the R134 device-tensor (V1) binding. Reuses the R133V
        vendor (oscar_r133_v1, patched op_host) when R133_GRAPH_V1=1, else R106
        (prod, byte-identical). Same op_api lib as _r91/_r112 (no second CDLL
        of a distinct vendor -> no symbol clash). The kernel reads
        actual_kv_len LIVE from GM (arch32 base.h:2370); the binding passes
        device int64 tensors (Tensor aclnn variant). R134 P0/P1/P2 proven N=36."""
        import os as _os

        if type(self)._r135_loaded:
            return type(self)._r135_mod is not None
        type(self)._r135_loaded = True
        try:
            R135BASE: str = _os.environ.get("R135_BASE", "")
            VENDOR: str = _os.environ.get("R106_VENDOR", "")
            if _os.environ.get("R133_GRAPH_V1", "0") == "1":
                _v133 = _os.environ.get("R133_VENDOR", "")
                if _v133:
                    VENDOR = _v133
            if not VENDOR:
                raise FileNotFoundError("vendor OPP path not set (R106_VENDOR/R133_VENDOR)")
            _os.environ["TORCH_EXTENSIONS_DIR"] = R135BASE + "/TASK_OWNED/torch_ext"
            _os.makedirs(_os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
            _os.environ["KERNEL_META"] = R135BASE + "/TASK_OWNED/kernel_meta"
            _os.makedirs(_os.environ["KERNEL_META"], exist_ok=True)
            _r157opp2 = _os.environ.get("R157_OPP")
            _os.environ["ASCEND_CUSTOM_OPP_PATH"] = ((_r157opp2 + ":") if _r157opp2 else "") + _os.environ.get(
                "R144_OPP_PATH", VENDOR
            )
            import ctypes as _ct

            _ct.CDLL("/usr/local/Ascend/driver/lib64/driver/libascend_hal.so", mode=_ct.RTLD_GLOBAL)
            # R157: dedicated R157_OPP first
            # str(): both operands are always strings; the str() keeps the
            # expression str-typed regardless of environ.get overloads
            _opp = str(_r157opp2 or _os.environ.get("R144_OPP_PATH", VENDOR)).split(":")[0]
            _ct.CDLL(_opp + "/op_api/lib/libcust_opapi.so", mode=_ct.RTLD_GLOBAL)
            import torch_npu as _tn
            from torch.utils import cpp_extension as _ce

            TN = _os.path.dirname(_tn.__file__)
            _cann = _os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
            INC = [
                _os.path.join(TN, "include"),
                _os.path.join(TN, "include", "third_party", "op-plugin"),
                _cann + "/aarch64-linux/include",
                _cann + "/include",
                VENDOR + "/op_api/include",
                R135BASE + "/TASK_OWNED",
            ]
            LD = [
                "-L" + VENDOR + "/op_api/lib",
                "-lcust_opapi",
                "-L" + _cann + "/lib64",
                "-lascendcl",
                "-L" + _os.path.join(TN, "lib"),
                "-ltorch_npu",
                "-Wl,-rpath," + VENDOR + "/op_api/lib",
                "-Wl,-rpath," + _cann + "/lib64",
            ]
            type(self)._r135_mod = _ce.load(
                name="int2_lse_v1_r135",
                sources=[R135BASE + "/TASK_OWNED/int2_op_launch_lse_v1_r134.cpp"],
                extra_include_paths=INC,
                extra_ldflags=LD,
                verbose=False,
            )
            print(f"[R135-V1] V1 binding loaded (R133_GRAPH_V1={_os.environ.get('R133_GRAPH_V1', '0')})", flush=True)
            return type(self)._r135_mod is not None
        except Exception as e:
            print(f"[R135-V1] Load failed: {e}", flush=True)
            return False

    def _r91_bridge_decode(self, query, key_rot, value_rot, attn_metadata, output, layer):
        """R91 native fused INT2 decode attention. Reads persistent INT2 cache."""
        import math as _m

        try:
            type(self)._r91_entry_count += 1
            num_block, block_size, num_kv_heads, packed_head = self.key_cache.shape
            head_size = self.head_size
            nq = self.num_heads
            nkv = self.num_kv_heads
            batch_size = len(attn_metadata.seq_lens_list)
            seq_lens = attn_metadata.seq_lens_list
            getattr(layer, "layer_idx", 0)
            device = query.device

            if not self._r91_ensure_loaded(device):
                type(self)._r91_fallback_count += 1
                return False

            block_table = attn_metadata.block_tables
            actual_seq_kvlen = seq_lens
            num_blocks_needed = [(s + block_size - 1) // block_size for s in actual_seq_kvlen]
            max_blocks_needed = max(num_blocks_needed) if num_blocks_needed else 0
            if max_blocks_needed == 0:
                return False

            bt = block_table[:batch_size, :max_blocks_needed].to(torch.int32).contiguous()

            # Gather packed INT2 blocks from persistent cache
            # key_cache: [num_blocks, block_size, nkv, packed_head] uint8
            k_packed = self.key_cache[bt[0]]  # [max_blocks, block_size, nkv, packed_head]
            v_packed = self.value_cache[bt[0]]

            # R96 FIX: permute to [num_blocks, nkv, block_size, packed_head]
            # (tiling function expects nkv before block_size)
            k_r91 = k_packed.permute(0, 2, 1, 3).contiguous().view(torch.int8)
            v_r91 = v_packed.permute(0, 2, 1, 3).contiguous().view(torch.int8)

            # Gather scale/zero from persistent cache
            k_scale_blk = self.k_scale_cache[bt[0]]  # [max_blocks, block_size, nkv]
            k_zero_blk = self.k_zero_cache[bt[0]]
            v_scale_blk = self.v_scale_cache[bt[0]]
            v_zero_blk = self.v_zero_cache[bt[0]]

            S = max_blocks_needed * block_size

            # Convert to R91 format: offset = -zero * scale, scale stays same
            # .contiguous() required: operator rejects uncontiguous antiquant scale/offset (errno 561000)
            k_scale_flat = k_scale_blk.reshape(S, nkv).t().unsqueeze(0).bfloat16().contiguous()  # [1, nkv, S]
            k_zero_flat = k_zero_blk.reshape(S, nkv).t().unsqueeze(0).bfloat16().contiguous()
            k_offset_flat = (-k_zero_flat * k_scale_flat).contiguous()
            v_scale_flat = v_scale_blk.reshape(S, nkv).t().unsqueeze(0).bfloat16().contiguous()
            v_zero_flat = v_zero_blk.reshape(S, nkv).t().unsqueeze(0).bfloat16().contiguous()
            v_offset_flat = (-v_zero_flat * v_scale_flat).contiguous()

            # R108: use REAL askv (maskless, NO pad_zero). The LSE sibling candidate
            # honors actual_seq_kv via maskless clip [askv:cap]; padding never enters
            # softmax/dequant. R107 G6 proved 0/0/0 + LSE 5e-7 with real (non-block-aligned)
            # askv. pad_zero is OBSOLETE for the corrected candidate (and causes NaN with
            # the LSE sibling dequant zero=-offset/scale when scale=0).
            askv_list = [int(actual_seq_kvlen[_si]) for _si in range(batch_size)]

            # Q rotation
            R_k = R_v = None
            if self._oscar_R_k is not None and self._oscar_layer_ord(layer) < self._oscar_R_k.shape[0]:
                R_k = self._oscar_R_k[self._oscar_layer_ord(layer)]
                R_v = self._oscar_R_v[self._oscar_layer_ord(layer)]

            query_bnsd = query[:batch_size]
            if query_bnsd.dim() == 3:
                query_bnsd = query_bnsd.unsqueeze(2)

            # Output must be 4D BNSD [batch, nq, 1, hd] to match operator contract
            out_bnsd = output[:batch_size]
            if out_bnsd.dim() == 3:
                out_bnsd = out_bnsd.unsqueeze(2)  # view, shares data with output

            # Rotate Q
            if R_k is not None:
                query_bnsd = torch.matmul(query_bnsd, R_k)
                _r218b = _R2XX_G.setdefault("_R218_ROT_SEEN", set())
                if "B" not in _r218b:
                    _r218b.add("B")
                    _r218_diag("FIRE-B")
                    print(
                        "[R218ROT-B] bnsd Q-rotate FIRES q={}".format(
                            tuple(query_bnsd.shape),
                        ),
                        file=_r144_ovl_sys.stderr,
                        flush=True,
                    )

            # R108: Allocate LSE output [B, nq, 1] float32. The LSE sibling op
            # OscarInt2FusedAttnInt2Lse writes per-head logsumexp here (S2 capture).
            lse_out = torch.zeros(batch_size, nq, 1, dtype=torch.float32, device=device)

            # R108: call the LSE sibling launcher (maskless: mask=empty→nullptr, pre=0, next=0).
            # Pass an EMPTY tensor for attenMask (launcher: numel()==0 → a_mask=nullptr → maskless).
            empty_mask = torch.zeros(0, dtype=torch.uint8, device=device)
            result = type(self)._r91_mod.launch_lse(
                query_bnsd,
                k_r91,
                v_r91,
                bt,
                k_scale_flat,
                k_offset_flat,
                v_scale_flat,
                v_offset_flat,
                out_bnsd,
                lse_out,
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

            getws = result["getws_ret"]
            launch = result["launch_ret"]
            sync = result["sync_ret"]
            changed = result["out_changed"]
            result["lse_changed"]
            result["lse_finite"]
            result["output_valid"]

            if getws == 0:
                type(self)._r91_getws_count += 1
            if launch == 0:
                type(self)._r91_launch_count += 1
            if sync == 0:
                type(self)._r91_sync_count += 1
            if getws == 0 and launch == 0 and sync == 0 and changed > 0:
                type(self)._r91_output_consumed_count += 1
                # OSCAR V-unrotation: output @ R_v.T (undo the V rotation)
                # The R91 operator computes attention in rotated V space;
                # the output must be un-rotated to original V space.
                if R_v is not None:
                    output[:batch_size] = torch.matmul(output[:batch_size], R_v.transpose(-1, -2))
                return True
            else:
                type(self)._r91_fallback_count += 1
                print(
                    f"[R96-R91] Bridge fallback: getws={getws} launch={launch} sync={sync} changed={changed}",
                    flush=True,
                )
                return False
        except Exception as e:
            type(self)._r91_fallback_count += 1
            print(f"[R96-R91] Bridge exception: {e}", flush=True)
            return False
