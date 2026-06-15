#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
"""JIT loader for the fused W4A4 MoE "mega" kernel.

Single-launch int4xint4 MoE expert path for Qwen3.x-MoE on Ascend 910B: in-kernel
block-diagonal Hadamard (cube GEMM) -> int4 quant + routing scatter -> int4 gate_up
-> SwiGLU + int4 requant -> int4 down -> unpermute/top-k combine, with cube/vector
overlap. Mirrors ``fast_hadamard.py`` (PR #8401): the PTO-ISA source co-located in
``ops/`` is compiled once with ``bisheng`` at first use and loaded via ctypes.

The cube matmuls use AscendC ``MatmulImpl<int4b_t>`` (CANN tikcfw); the vector stages
use PTO-ISA, so this kernel requires a PTO-ISA toolchain (``PTO_LIB_PATH``) in addition
to CANN -- see the contribution notes / PR description.
"""

import contextlib
import ctypes
import fcntl
import functools
import hashlib
import math
import os
import subprocess
import sys

import numpy as np
import torch

import vllm_ascend.envs as envs_ascend

_OPS_DIR = os.path.dirname(os.path.abspath(__file__))
_KERNEL_SRC = os.path.join(_OPS_DIR, "mega_moe_w4a4_qwen36_pto-isa.cpp")
_KERNEL_LIB = envs_ascend.VLLM_ASCEND_MEGA_MOE_SO or os.path.join(_OPS_DIR, "mega_moe_w4a4_jit.so")
# Every source the .so is built from — the staleness check must compare against
# all of them (the qwen36 shim only ``#include``s the real implementation).
_KERNEL_DEPS = [
    _KERNEL_SRC,
    os.path.join(_OPS_DIR, "mega_moe_w4a4_pto-isa.cpp"),
    os.path.join(_OPS_DIR, "int4_cvt.hpp"),
]

# Block-diagonal Hadamard size (must match the offline rotation baked into the
# gate_up weights). 64 is the validated production value for Qwen3.x-MoE.
HADAMARD_BLOCK_SIZE = 64

# Per-rank shapes the kernel is COMPILED for (constants in the .cpp). The scheme
# must fail fast if a model/TP produces different per-rank dims (see
# w4a4_dynamic.process_weights_after_loading).
KERNEL_H_DIM = 2048
KERNEL_I_DIM = 128

_PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", "/sources/pto-isa")
_CANN_HOME = os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/cann-8.5.1")
_CANN_INC = f"{_CANN_HOME}/include"
_CANN_ARCH_PKG = f"{_CANN_HOME}/aarch64-linux/pkg_inc"
_TIK_DIR = os.environ.get("ASCEND_TIKCFW", f"{_CANN_HOME}/aarch64-linux/tikcpp/tikcfw")

# Production build of the self-contained S3FP16 + cube-Hadamard variant. Each flag is a
# validated lever (see PR description / FINAL_REPORT); cube-Hadamard makes the kernel fully
# fused so no external Hadamard op is needed.
_BUILD_DEFINES = [
    f"-DMEGA_HADAMARD_N={HADAMARD_BLOCK_SIZE}",
    "-DMEGA_CUBE_HADAMARD",  # in-kernel block-diag Hadamard (Stage 1) -> self-contained
    "-DMEGA_HADAMARD_KERNEL_SKIP",  # quant-stage scale uses 1/sqrtN=1 (rotation done by the Hadamard stage)
    "-DMEGA_OVERLAP",
    "-DMEGA_OVERLAP_SAFESYNC",  # cube/vector overlap schedule
    # NOTE: the Sn build-flag digits are the kernel's internal stage index and do
    # NOT track the doc's 1-based numbering; they are named by function below.
    "-DMEGA_S5_SCATTER",  # atomic-add unpermute/combine (the combine stage)
    "-DMEGA_S1_FAST",  # multi-row batched quant stage
    "-DMEGA_S3_DBUF",
    "-DMEGA_S5_FP16",  # double-buffered SwiGLU / fp16 combine
    "-DMEGA_S3_ROWPART",
    "-DMEGA_S5_ROWPART",  # skew-immune row partition
    "-DMEGA_SCALE_FOLD",  # fold the int4 *7 constant into the per-row divisor
]


def _defines_hash() -> str:
    """Hash of the build flags so a change in -D defines forces a rebuild."""
    return hashlib.sha1(" ".join(_BUILD_DEFINES).encode()).hexdigest()


def _is_lib_fresh() -> bool:
    """Fresh iff the .so is newer than every kernel source AND was built with the
    current build defines (recorded in a ``.defines`` sidecar)."""
    try:
        so_mtime = os.path.getmtime(_KERNEL_LIB)
    except OSError:
        return False
    for dep in _KERNEL_DEPS:
        if os.path.exists(dep) and so_mtime < os.path.getmtime(dep):
            return False
    try:
        with open(_KERNEL_LIB + ".defines") as f:
            return f.read().strip() == _defines_hash()
    except OSError:
        return False


def compile_kernel(verbose: bool = False) -> str:
    """Compile the mega-MoE kernel .so with bisheng if missing/stale. Returns the .so path.

    Serialized with a file lock so concurrent TP worker processes don't race on
    the shared ``-o`` target (a half-written .so would fail to load)."""
    if _is_lib_fresh():
        return _KERNEL_LIB
    lock_path = _KERNEL_LIB + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        # Another worker may have built it while we waited for the lock.
        if _is_lib_fresh():
            return _KERNEL_LIB
        return _compile_kernel_locked(verbose)


def _compile_kernel_locked(verbose: bool) -> str:
    command = [
        "bisheng",
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        "-Wno-ignored-attributes",
        "--cce-aicore-arch=dav-c220",  # cube + vector
        "-isystem",
        f"{_PTO_LIB_PATH}/include",
        "-isystem",
        _TIK_DIR,
        "-isystem",
        f"{_TIK_DIR}/impl",
        "-isystem",
        f"{_TIK_DIR}/interface",
        "-isystem",
        f"{_TIK_DIR}/lib",
        "-isystem",
        _CANN_INC,
        "-isystem",
        _CANN_ARCH_PKG,
        "-isystem",
        f"{_CANN_ARCH_PKG}/runtime",
        "-isystem",
        f"{_CANN_ARCH_PKG}/profiling",
        *_BUILD_DEFINES,
        _KERNEL_SRC,
        "-o",
        _KERNEL_LIB,
    ]
    if verbose:
        print(f"[mega-moe-w4a4] building {_KERNEL_SRC} -> {_KERNEL_LIB}", file=sys.stderr)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mega-moe-w4a4 kernel build failed:\n{result.stderr[:3000]}")
    # Record the build-defines hash so a later flag change forces a rebuild.
    with open(_KERNEL_LIB + ".defines", "w") as f:
        f.write(_defines_hash())
    return _KERNEL_LIB


@functools.cache
def get_mega_moe_kernel():
    """Return the ctypes entry point for the fused W4A4 MoE kernel.

    Signature: ``call_mega_kernel_hybrid_qwen36(block_dim, stream, *20 ptrs, M_total,
    E, top_k, T_orig)`` where the 20 pointers are x, w13, w13_scale, w2, w2_scale,
    group_list, eri, sort_idx, topk_w, then 6 workspaces, y, tiling_gu, tiling_dn,
    b1 (Hadamard blocks), xrot_ws.
    """
    lib = ctypes.CDLL(compile_kernel())
    func = lib.call_mega_kernel_hybrid_qwen36
    func.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 20 + [ctypes.c_uint32] * 4
    func.restype = None
    return func


# ============================================================================
# Host-side orchestration for the fused kernel: weight repack (FRACTAL_NZ int4),
# scale packing (uint64 for the cube FIXPIPE dequant), per-shape workspaces, the
# routing permutation, and the launch. The kernel is the production hybrid build
# (AscendC int4 cube + PTO vector, in-kernel Stage-1 Hadamard, fp16 cube dequant
# folded into FIXPIPE for the gate_up/down matmuls), so the gate_up/down scratch
# buffers are fp16 and both weight scales are uint64-packed.
# ============================================================================

_BLOCK_DIM = 24  # 910B (A2) cube core count
_M_TILE = 32  # cube writes in 32-row chunks; pad workspaces to this
_M_TILE_GU = 128  # gate_up cube baseM (== singleCoreM; see _make_tiling)
_M_TILE_DN = 64  # down cube baseM


def _cached_void_p(t: torch.Tensor) -> "ctypes.c_void_p":
    """``c_void_p(t.data_ptr())`` memoized on the tensor (weights/scales are
    static after load, so this is paid once per layer instead of per token)."""
    p = t.data_ptr()
    cache = getattr(t, "_mega_ptr_cache", None)
    if cache is not None and cache[0] == p:
        return cache[1]
    cv = ctypes.c_void_p(p)
    with contextlib.suppress(AttributeError):
        t._mega_ptr_cache = (p, cv)
    return cv


_STREAM_CACHE: dict = {}


def _get_stream(device: torch.device):
    s = _STREAM_CACHE.get(device)
    if s is None:
        s = torch.npu.current_stream(device)
        _STREAM_CACHE[device] = s
    return s


def routing_prep(topk_ids: torch.Tensor, num_experts: int):
    """Sort-by-expert permutation, replacing ``npu_moe_init_routing_v2``.

    Returns ``(group_list, expanded_row_idx, sort_idx)``:
      * ``group_list`` [E] int64 — cumulative per-expert token counts.
      * ``expanded_row_idx`` [M] int32 — vendor convention; for original flat
        index ``i`` it is the expanded slot token ``i`` landed at (combine stage).
      * ``sort_idx`` [M] int32 — inverse: expanded slot -> original flat index
        (the quant stage reads ``x[sort_idx[m] // top_k]``).

    Done on fp32 so both argsorts stay on AICore (int sort falls back to AICpu).
    """
    flat = topk_ids.reshape(-1).to(torch.float32)
    sorted_ids, sort_idx = torch.sort(flat)
    eri = torch.argsort(sort_idx.to(torch.float32))
    expert_ids = torch.arange(num_experts, device=topk_ids.device, dtype=torch.float32)
    group_list = torch.searchsorted(sorted_ids, expert_ids, right=True).to(torch.int64)
    return group_list, eri.to(torch.int32), sort_idx.to(torch.int32)


def pack_nz_int4(w_kn: torch.Tensor) -> torch.Tensor:
    """``[E, K, N]`` int8 (one nibble per byte, range [-8, 7]) -> flat FRACTAL_NZ
    int4 bytes, the layout the AscendC ``CubeFormat::NZ`` int4 B matrix expects:
    ``[E, N/64, K/16, 16, 32]`` packed along the inner N0=64."""
    E, K, N = w_kn.shape
    assert K % 16 == 0 and N % 64 == 0, f"NZ pack needs K%16==0, N%64==0 (K={K}, N={N})"
    r = w_kn.reshape(E, K // 16, 16, N // 64, 64).permute(0, 3, 1, 2, 4).contiguous()
    lo = r[..., 0::2].to(torch.int32) & 0xF
    hi = r[..., 1::2].to(torch.int32) & 0xF
    return ((hi << 4) | lo).to(torch.int8).reshape(-1).contiguous()


# TCubeTiling: 50 int32 fields, in AscendC kernel_tiling.h order.
_TILING_FIELDS = [
    "usedCoreNum",
    "M",
    "N",
    "Ka",
    "Kb",
    "singleCoreM",
    "singleCoreN",
    "singleCoreK",
    "baseM",
    "baseN",
    "baseK",
    "depthA1",
    "depthB1",
    "stepM",
    "stepN",
    "isBias",
    "transLength",
    "iterateOrder",
    "shareMode",
    "shareL1Size",
    "shareL0CSize",
    "shareUbSize",
    "batchM",
    "batchN",
    "singleBatchM",
    "singleBatchN",
    "stepKa",
    "stepKb",
    "depthAL1CacheUB",
    "depthBL1CacheUB",
    "dbL0A",
    "dbL0B",
    "dbL0C",
    "ALayoutInfoB",
    "ALayoutInfoS",
    "ALayoutInfoN",
    "ALayoutInfoG",
    "ALayoutInfoD",
    "BLayoutInfoB",
    "BLayoutInfoS",
    "BLayoutInfoN",
    "BLayoutInfoG",
    "BLayoutInfoD",
    "CLayoutInfoB",
    "CLayoutInfoS1",
    "CLayoutInfoN",
    "CLayoutInfoG",
    "CLayoutInfoS2",
    "BatchNum",
    "mxTypePara",
]


def _make_tiling(m_tile: int, K: int, N: int, n_tile: int, base_k: int, base_m: int):
    """Build a 50-int32 TCubeTiling array. baseM MUST equal singleCoreM (=m_tile):
    the int4 ``MatmulImpl`` corrupts when baseM < singleCoreM (multiple baseM
    sub-iterations over the int4 L0A fractal)."""
    base_k = min(base_k, K)
    n_ka = (K + base_k - 1) // base_k
    t = dict.fromkeys(_TILING_FIELDS, 0)
    t["usedCoreNum"] = _BLOCK_DIM
    t["M"], t["N"], t["Ka"], t["Kb"] = m_tile, N, K, K
    t["singleCoreM"], t["singleCoreN"], t["singleCoreK"] = m_tile, n_tile, K
    t["baseM"], t["baseN"], t["baseK"] = max(16, base_m), n_tile, base_k
    t["depthA1"], t["depthB1"] = n_ka, n_ka
    t["stepM"], t["stepN"] = 1, 1
    t["stepKa"], t["stepKb"] = n_ka, n_ka
    t["dbL0A"], t["dbL0B"], t["dbL0C"] = 2, 2, 1
    return np.array([t[f] for f in _TILING_FIELDS], dtype=np.int32)


_TILING_CACHE: dict = {}


def _get_tilings(device: torch.device, H: int, i_dim: int, n_gu: int):
    """``(tiling_gu_voidp, tiling_dn_voidp)`` for the two grouped matmuls, cached
    per layer shape. gate_up: K=H, N=n_gu. down: K=i_dim, N=H (512-tiled)."""
    key = (device, H, i_dim, n_gu)
    e = _TILING_CACHE.get(key)
    if e is None:
        tg = _make_tiling(_M_TILE_GU, H, n_gu, n_gu, base_k=256, base_m=_M_TILE_GU)
        td = _make_tiling(_M_TILE_DN, i_dim, H, min(512, H), base_k=min(256, i_dim), base_m=_M_TILE_DN)
        tg_t = torch.from_numpy(tg).contiguous().to(device)
        td_t = torch.from_numpy(td).contiguous().to(device)
        e = (tg_t, td_t, ctypes.c_void_p(tg_t.data_ptr()), ctypes.c_void_p(td_t.data_ptr()))
        _TILING_CACHE[key] = e
    return e[2], e[3]


_B1_CACHE: dict = {}


def _get_b1(device: torch.device, H: int) -> "ctypes.c_void_p":
    """Stage-1 Hadamard weight: normalized 64x64 Walsh-Hadamard replicated H/64
    times, ``[H, 64]`` fp16. Generated at load (it's a fixed matrix, not stored
    in the checkpoint). The Sylvester matrix is symmetric, so it equals its own
    DN-transposed block."""
    key = (device, H)
    e = _B1_CACHE.get(key)
    if e is None:
        n = HADAMARD_BLOCK_SIZE
        rows = [[(-1) ** bin(i & j).count("1") for j in range(n)] for i in range(n)]
        hn = torch.tensor(rows, dtype=torch.float32) / math.sqrt(n)
        b1 = hn.repeat(H // n, 1).to(torch.float16).contiguous().to(device)
        e = (b1, ctypes.c_void_p(b1.data_ptr()))
        _B1_CACHE[key] = e
    return e[1]


_SCALE_U64_CACHE: dict = {}


def pack_scale_uint64(scale: torch.Tensor) -> "ctypes.c_void_p":
    """Per-channel fp32 scale -> uint64 (fp32 bits in the low 32) for the cube
    FIXPIPE ``SetQuantVector`` dequant. Cached + synchronized once per layer:
    the pack is an async op chain the cube reads on its first launch, so it must
    be materialized before any kernel can consume it (else cold launch reads
    zero -> all-zero output)."""
    p = scale.data_ptr()
    e = _SCALE_U64_CACHE.get(p)
    if e is None:
        f32 = scale.to(torch.float32).contiguous()
        u64 = (f32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF).contiguous()
        torch.npu.synchronize()
        e = (u64, ctypes.c_void_p(u64.data_ptr()))
        _SCALE_U64_CACHE[p] = e
    return e[1]


_WS_CACHE: dict = {}


def _get_workspaces(device: torch.device, M_total: int, T_orig: int, H: int, i_dim: int):
    """Grow-on-demand scratch, keyed by shape-independent ``(device, H, I)`` and
    grown to the max batch seen (a per-batch key would leak under decode's
    varying batch -> fragmentation -> OOM). Returns ``(y, ptr_tuple)`` where
    ``ptr_tuple`` = (xq, xs, gu, iq, is, d, y, xrot) c_void_p.

    Production hybrid build: gate_up (gu) and down (d) scratch are fp16 (the cube
    folds the per-channel dequant into FIXPIPE), and xq + xrot alias into d's
    buffer (both dead before the down matmul first writes d, separated by the
    FFTS barrier) to cut workspace peak."""
    key = (device, H, i_dim)
    m_pad = ((M_total + _M_TILE - 1) // _M_TILE) * _M_TILE
    entry = _WS_CACHE.get(key)
    if entry is None or m_pad > entry[2] or T_orig > entry[3]:
        m_cap = max(m_pad, entry[2] if entry else 0)
        t_cap = max(T_orig, entry[3] if entry else 0)
        h_act = H // 2  # int4-packed activations
        i_act = i_dim // 2
        # ``d`` (fp16, [rows, H]) also backs the aliased xq (offset 0, m_cap*h_act
        # bytes) + xrot (offset m_cap*h_act, t_cap*H*2 bytes). Worst case is top_k=1
        # where t_cap == m_cap, needing m_cap*h_act + m_cap*H*2 == 2.5*m_cap*H bytes;
        # 1.25*m_cap fp16 rows == 2.5*m_cap*H bytes covers it for every top_k.
        d = torch.empty(m_cap + m_cap // 4, H, dtype=torch.float16, device=device)
        xs = torch.empty(m_cap * 32, dtype=torch.float32, device=device)
        gu = torch.empty(m_cap, 2 * i_dim, dtype=torch.float16, device=device)
        iq = torch.empty(m_cap, i_act, dtype=torch.int8, device=device)
        is_ = torch.empty(m_cap * 32, dtype=torch.float32, device=device)
        y = torch.empty(t_cap, H, dtype=torch.float16, device=device)
        # xq + xrot alias d's bytes (see docstring).
        db = d.data_ptr()
        xq_ptr = ctypes.c_void_p(db)
        xrot_ptr = ctypes.c_void_p(db + m_cap * h_act)
        ptrs = (
            xq_ptr,
            ctypes.c_void_p(xs.data_ptr()),
            ctypes.c_void_p(gu.data_ptr()),
            ctypes.c_void_p(iq.data_ptr()),
            ctypes.c_void_p(is_.data_ptr()),
            ctypes.c_void_p(d.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            xrot_ptr,
        )
        entry = (y, ptrs, m_cap, t_cap, (xs, gu, iq, is_, d, y))  # last = GC anchor
        _WS_CACHE[key] = entry
    return entry[0], entry[1]


def mega_moe_forward(
    x: torch.Tensor,
    w13_nz: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_nz: torch.Tensor,
    w2_scale: torch.Tensor,
    group_list: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    sort_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    H: int,
    i_dim: int,
    n_gu: int,
) -> torch.Tensor:
    """One fused launch of the full expert path. ``x`` is the UN-permuted
    ``[T, H]`` fp16 input (the kernel scatters by ``expanded_row_idx``); weights
    are FRACTAL_NZ int4 with uint64-packed scales. Returns ``[T, H]`` fp16."""
    device = x.device
    T_orig = x.shape[0]
    E = int(group_list.shape[0])
    M_total = expanded_row_idx.shape[0]

    y, ws = _get_workspaces(device, M_total, T_orig, H, i_dim)
    y[:T_orig].zero_()  # the combine stage atomic-adds into pre-zeroed y

    func = get_mega_moe_kernel()
    tgu_p, tdn_p = _get_tilings(device, H, i_dim, n_gu)
    func(
        _BLOCK_DIM,
        ctypes.c_void_p(_get_stream(device).npu_stream),
        ctypes.c_void_p(x.data_ptr()),
        _cached_void_p(w13_nz),
        pack_scale_uint64(w13_scale),
        _cached_void_p(w2_nz),
        pack_scale_uint64(w2_scale),
        ctypes.c_void_p(group_list.data_ptr()),
        ctypes.c_void_p(expanded_row_idx.data_ptr()),
        ctypes.c_void_p(sort_idx.data_ptr()),
        ctypes.c_void_p(topk_weights.data_ptr()),
        ws[0],
        ws[1],
        ws[2],
        ws[3],
        ws[4],
        ws[5],
        ws[6],
        tgu_p,
        tdn_p,
        _get_b1(device, H),
        ws[7],
        M_total,
        E,
        top_k,
        T_orig,
    )
    return y[:T_orig]
