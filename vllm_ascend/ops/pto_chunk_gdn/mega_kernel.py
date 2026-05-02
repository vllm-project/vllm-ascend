#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Fused PTO GDN megakernel: all six GDN pipeline stages in one NPU launch.

Fuses cumsum → scaled_dot_kkt → solve_tril → wy_fast → chunk_h → chunk_o
into a single ``call_kernel`` dispatch, eliminating Python-level inter-stage
synchronization overhead.

GQA is supported: Q/K use ``Hg`` heads while V/gates use ``H ≥ Hg`` value heads
(``H % Hg == 0``).

Usage::

    from vllm_ascend.ops.pto_chunk_gdn.mega_kernel import run_mega_kernel

    o = run_mega_kernel(q, k, v, g, beta, cu_seqlens, stream=stream,
                        chunk_size=128, scale=head_dim**-0.5, key_heads=Hg)
"""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache

import torch

from vllm_ascend.ops.pto_chunk_gdn.compile import BLOCK_DIM, KERNELS_PTO, compile_mega_kernel


# ---------------------------------------------------------------------------
# Small utilities (self-contained, no external module dependency)
# ---------------------------------------------------------------------------

def _vp(t: torch.Tensor | None) -> ctypes.c_void_p:
    if t is None:
        return ctypes.c_void_p()
    return ctypes.c_void_p(t.data_ptr())


@lru_cache(maxsize=48)
def _precomputed_minus_identity(
    device_ty: str, device_index: int, chunk_size: int
) -> torch.Tensor:
    """``[C, C]`` fp16 buffer with diagonal ``-1``, cached per (device, C)."""
    idx = max(device_index, 0)
    dev = torch.device(device_ty, idx) if device_ty != "cpu" else torch.device("cpu")
    t = torch.zeros(chunk_size, chunk_size, device=dev, dtype=torch.float16)
    t.fill_diagonal_(-1)
    return t


@lru_cache(maxsize=48)
def _causal_masks(
    device_ty: str, device_index: int, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower-triangle and full causal masks, cached per (device, C)."""
    idx = max(device_index, 0)
    dev = torch.device(device_ty, idx) if device_ty != "cpu" else torch.device("cpu")
    m_lower = torch.tril(torch.ones(chunk_size, chunk_size, device=dev), diagonal=-1).float()
    m_full  = torch.tril(torch.ones(chunk_size, chunk_size, device=dev), diagonal=0).float()
    return m_lower, m_full


def _total_chunks(N_seq: int, cu_seqlens: torch.Tensor, chunk_size: int) -> int:
    """Total number of chunks across all sequences in the varlen batch."""
    cu = cu_seqlens.cpu().tolist()
    return sum(
        (cu[i + 1] - cu[i] + chunk_size - 1) // chunk_size
        for i in range(N_seq)
    )


# ---------------------------------------------------------------------------
# Kernel loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_mega_kernel(
    *,
    num_heads: int,
    key_heads: int,
    hidden_size: int,
    chunk_size: int,
) -> ctypes.CDLL:
    mtime = os.stat(os.path.join(KERNELS_PTO, "mega_kernel.cpp")).st_mtime_ns
    lib_path = compile_mega_kernel(
        num_heads=num_heads, key_heads=key_heads,
        hidden_size=hidden_size, chunk_size=chunk_size,
        cpp_mtime_ns=mtime,
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p]
        + [ctypes.c_void_p] * 28
        + [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_uint32]
    )
    lib.call_kernel.restype = None
    return lib


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_mega_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_in: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    stream,
    chunk_size: int = 128,
    scale: float = 1.0,
    block_dim: int | None = None,
    key_heads: int | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run all six GDN pipeline stages in a single fused NPU kernel launch.

    Args:
        q, k:   ``[B, T, Hg, D]`` fp16 query / key tensors.
        v:      ``[B, T, H, D]`` fp16 value tensor  (H ≥ Hg, H % Hg == 0).
        g_in:   ``[B, T, H]`` float32 pre-cumsum gate logits.
        beta:   ``[B, T, H]`` fp16 gate bias.
        cu_seqlens: ``int32`` cumulative sequence-length boundaries ``[0, …, T]``.
        stream: NPU stream handle (``torch.npu.current_stream()._as_parameter_``).
        chunk_size: Tile side length C (must be 128).
        scale:  Output scale (typically ``head_dim ** -0.5``).
        block_dim: AI-Core block count; auto-detected from device if None.
        key_heads: Q/K head count Hg (inferred from ``q`` if None).
        return_final_state: Also return ``[N_seq, H, D, D]`` final recurrent states.

    Returns:
        ``O * scale`` of shape ``[B, T, H, D]`` fp16, and optionally final
        recurrent state ``[N_seq, H, D, D]`` fp16.
    """
    dev = q.device
    kh = key_heads if key_heads is not None else q.shape[2]
    H, D = v.shape[2], q.shape[3]
    C = chunk_size
    T = q.shape[1]
    N_seq = int(cu_seqlens.numel()) - 1
    bd = block_dim or BLOCK_DIM

    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)

    dt, di = dev.type, dev.index if dev.index is not None else -1
    msk_lower, msk_full = _causal_masks(dt, di, C)
    minus_identity = _precomputed_minus_identity(dt, di, C)

    tc = _total_chunks(N_seq, cu_seqlens, C)
    num_matrices = tc * H

    g_sum      = torch.empty(1, T, H, device=dev, dtype=torch.float32)
    g_t        = torch.empty(H, T, device=dev, dtype=torch.float32)
    beta_t     = torch.empty(H, T, device=dev, dtype=torch.float16)
    A          = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
    A_inv_f32  = torch.zeros(1, T, H, C, device=dev, dtype=torch.float32)
    A_inv      = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
    w          = torch.empty_like(v)
    u          = torch.empty_like(v)
    s          = torch.zeros(tc * H, D, D, device=dev, dtype=torch.float16)
    v_new      = torch.empty_like(v)
    fs         = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)
    kkt_ws     = torch.zeros(bd * 2, C, C, device=dev, dtype=torch.float16)
    wy_ws_a1   = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    wy_ws_a2   = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    h_ws       = torch.zeros(bd * 4, D, D, device=dev, dtype=torch.float16)
    o_ws_qk    = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    o_ws_qs    = torch.zeros(bd, C, D, device=dev, dtype=torch.float16)
    o_ws_gated = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    o_out      = torch.empty_like(v)

    lib = _load_mega_kernel(num_heads=H, key_heads=kh, hidden_size=D, chunk_size=C)
    lib.call_kernel(
        bd, stream,
        _vp(q), _vp(k), _vp(v), _vp(g_in), _vp(beta),
        _vp(msk_lower), _vp(msk_full), _vp(minus_identity), _vp(cu_seqlens),
        _vp(o_out),
        _vp(g_sum), _vp(g_t), _vp(beta_t),
        _vp(A), _vp(A_inv_f32), _vp(A_inv),
        _vp(w), _vp(u), _vp(s), _vp(v_new), _vp(fs),
        _vp(kkt_ws), _vp(wy_ws_a1), _vp(wy_ws_a2), _vp(h_ws),
        _vp(o_ws_qk), _vp(o_ws_qs), _vp(o_ws_gated),
        N_seq, T, T, num_matrices,
    )

    o_scaled = o_out * scale
    if return_final_state:
        return o_scaled, fs.view(N_seq, H, D, D)
    return o_scaled
