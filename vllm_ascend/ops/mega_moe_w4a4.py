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
import functools
import os
import subprocess
import sys

_OPS_DIR = os.path.dirname(os.path.abspath(__file__))
_KERNEL_SRC = os.path.join(_OPS_DIR, "mega_moe_w4a4_qwen36_pto-isa.cpp")
_KERNEL_LIB = os.environ.get("VLLM_ASCEND_MEGA_MOE_SO",
                             os.path.join(_OPS_DIR, "mega_moe_w4a4_jit.so"))

# Block-diagonal Hadamard size (must match the offline rotation baked into the
# gate_up weights). 64 is the validated production value for Qwen3.x-MoE.
HADAMARD_BLOCK_SIZE = 64

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
    "-DMEGA_CUBE_HADAMARD",          # in-kernel block-diag Hadamard (Stage 0) -> self-contained
    "-DMEGA_HADAMARD_KERNEL_SKIP",    # Stage-1 quant scale uses 1/sqrtN=1 (rotation done in Stage 0)
    "-DMEGA_OVERLAP", "-DMEGA_OVERLAP_SAFESYNC",   # cube/vector overlap schedule
    "-DMEGA_S5_SCATTER",             # atomic-add unpermute/combine in Stage 5
    "-DMEGA_S1_FAST",                # multi-row batched Stage 1 quant
    "-DMEGA_S3_DBUF", "-DMEGA_S5_FP16",            # double-buffered Stage 3 / fp16 Stage 5
    "-DMEGA_S3_ROWPART", "-DMEGA_S5_ROWPART",      # skew-immune row partition
    "-DMEGA_SCALE_FOLD",             # fold the int4 *7 constant into the per-row divisor
]


def _is_lib_fresh() -> bool:
    try:
        return os.path.getmtime(_KERNEL_LIB) >= os.path.getmtime(_KERNEL_SRC)
    except OSError:
        return False


def compile_kernel(verbose: bool = False) -> str:
    """Compile the mega-MoE kernel .so with bisheng if missing/stale. Returns the .so path."""
    if _is_lib_fresh():
        return _KERNEL_LIB
    command = [
        "bisheng",
        "-fPIC", "-shared", "-xcce", "-DMEMORY_BASE", "-O2",
        "-std=c++17", "-Wno-ignored-attributes",
        "--cce-aicore-arch=dav-c220",   # cube + vector
        "-isystem", f"{_PTO_LIB_PATH}/include",
        "-isystem", _TIK_DIR, "-isystem", f"{_TIK_DIR}/impl",
        "-isystem", f"{_TIK_DIR}/interface", "-isystem", f"{_TIK_DIR}/lib",
        "-isystem", _CANN_INC, "-isystem", _CANN_ARCH_PKG,
        "-isystem", f"{_CANN_ARCH_PKG}/runtime", "-isystem", f"{_CANN_ARCH_PKG}/profiling",
        *_BUILD_DEFINES,
        _KERNEL_SRC, "-o", _KERNEL_LIB,
    ]
    if verbose:
        print(f"[mega-moe-w4a4] building {_KERNEL_SRC} -> {_KERNEL_LIB}", file=sys.stderr)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mega-moe-w4a4 kernel build failed:\n{result.stderr[:3000]}")
    return _KERNEL_LIB


@functools.lru_cache(maxsize=None)
def get_mega_moe_kernel():
    """Return the ctypes entry point for the fused W4A4 MoE kernel.

    Signature: ``call_mega_kernel_hybrid_qwen36(block_dim, stream, *20 ptrs, M_total,
    E, top_k, T_orig)`` where the 20 pointers are x, w13, w13_scale, w2, w2_scale,
    group_list, eri, sort_idx, topk_w, then 6 workspaces, y, tiling_gu, tiling_dn,
    b1 (Hadamard blocks), xrot_ws.
    """
    import ctypes
    lib = ctypes.CDLL(compile_kernel())
    func = lib.call_mega_kernel_hybrid_qwen36
    func.argtypes = ([ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 20
                     + [ctypes.c_uint32] * 4)
    func.restype = None
    return func
