# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2024, Tri Dao, Albert Gu.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Adapted from vLLM's Mamba SSD chunk scan kernel for Ascend NPU.
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

import vllm.model_executor.layers.mamba.ops.ssd_chunk_scan as _ssd_chunk_scan
from vllm.model_executor.layers.mamba.ops.triton_helpers import fast_exp
from vllm.triton_utils import tl, triton

_ORIGINAL_CHUNK_SCAN_FWD = _ssd_chunk_scan._chunk_scan_fwd
_ASCEND_MAX_CORE_DIM = 65535
_CHUNK_SCAN_MAX_PROGRAMS_PER_LAUNCH = 32768
_CHUNK_SCAN_BLOCK_SIZE_M = 128
_CHUNK_SCAN_BLOCK_SIZE_N = 64
_CHUNK_SCAN_BLOCK_SIZE_K = 64
_CHUNK_SCAN_NPU_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": _CHUNK_SCAN_BLOCK_SIZE_M,
            "BLOCK_SIZE_N": _CHUNK_SCAN_BLOCK_SIZE_N,
            "BLOCK_SIZE_K": _CHUNK_SCAN_BLOCK_SIZE_K,
        },
        num_stages=1,
        num_warps=4,
    )
]


def _chunk_scan_optional_kernel_args(reference, optional_tensor, shape):
    if optional_tensor is not None:
        return optional_tensor
    return reference.new_empty(shape)


def _chunk_scan_initial_states_kernel_arg(states, initial_states):
    if initial_states is not None:
        return initial_states
    return states


def _chunk_scan_initial_states_strides(initial_states):
    if initial_states is None:
        return (0, 0, 0, 0)
    return (
        initial_states.stride(0),
        initial_states.stride(1),
        initial_states.stride(2),
        initial_states.stride(3),
    )


def _chunk_scan_grid(chunk_size, headdim, nchunks, nheads):
    return lambda meta: (
        triton.cdiv(chunk_size, meta["BLOCK_SIZE_M"]) * triton.cdiv(headdim, meta["BLOCK_SIZE_N"]),
        nchunks,
        nheads,
    )


def _chunk_scan_launch_ranges(chunk_size, headdim, nchunks, nheads):
    programs_per_chunk = (
        triton.cdiv(chunk_size, _CHUNK_SCAN_BLOCK_SIZE_M) * triton.cdiv(headdim, _CHUNK_SCAN_BLOCK_SIZE_N) * nheads
    )
    chunks_per_launch = (
        min(
            _ASCEND_MAX_CORE_DIM,
            _CHUNK_SCAN_MAX_PROGRAMS_PER_LAUNCH,
        )
        // programs_per_chunk
    )
    if chunks_per_launch < 1:
        raise ValueError("chunk scan tile requires more programs than Ascend supports in one launch")
    for chunk_start in range(0, nchunks, chunks_per_launch):
        yield chunk_start, min(chunks_per_launch, nchunks - chunk_start)


@triton.autotune(
    # The upstream CUDA kernel autotunes a large configuration set. On
    # Triton-Ascend this makes the first chunked-prefill request compile many
    # kernel variants across TP ranks. Keep the NPU path fixed to a stable
    # tile for Mamba2's chunk_size=128, head_dim=64, dstate=128 profile.
    # BLOCK_SIZE_M=128 keeps the common 262,144-token request in one launch.
    # Longer requests are split along the chunk axis to stay below Ascend's
    # coreDim limit without adding a device synchronization. Keep each 3D grid
    # at 32,768 programs: grids close to the 65,535 hard limit can trigger an
    # MTE out-of-range fault on Ascend 910B with Triton-Ascend 3.2.x.
    configs=_CHUNK_SCAN_NPU_CONFIGS,
    key=["chunk_size", "hdim", "dstate", "IS_CAUSAL"],
)
@triton.jit
def _chunk_scan_fwd_kernel_npu(
    # Pointers to matrices
    cb_ptr,
    x_ptr,
    z_ptr,
    out_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    seq_idx_ptr,
    C_ptr,
    states_ptr,
    D_ptr,
    initstates_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimensions
    chunk_start: tl.int64,
    chunk_size: tl.constexpr,
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides
    stride_cb_chunk: tl.int64,
    stride_cb_head: tl.int64,
    stride_cb_csize_m: tl.int64,
    stride_cb_csize_k: tl.constexpr,
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    stride_z_seqlen: tl.int64,
    stride_z_head: tl.int64,
    stride_z_hdim: tl.constexpr,
    stride_out_seqlen: tl.int64,
    stride_out_head: tl.int64,
    stride_out_hdim: tl.constexpr,
    stride_dt_chunk: tl.int64,
    stride_dt_head: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    stride_seq_idx_chunk: tl.constexpr,
    stride_C_seqlen: tl.int64,
    stride_C_head: tl.int64,
    stride_C_dstate: tl.constexpr,
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    stride_init_states_batch: tl.int64,
    stride_init_states_head: tl.int64,
    stride_init_states_hdim: tl.int64,
    stride_init_states_dstate: tl.constexpr,
    stride_D_head: tl.constexpr,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
):
    pid_c = tl.program_id(axis=1).to(tl.int64) + chunk_start
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    cb_ptr += pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    x_ptr += chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += chunk_seqlen_start * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head

    # M-block offsets and prev states
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    seq_idx_ptr += pid_c * stride_seq_idx_chunk
    seq_idx = tl.load(seq_idx_ptr)
    seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_chunk, mask=pid_c >= 1, other=-1)
    starts_new_sequence = seq_idx != seq_idx_prev

    if HAS_INITSTATES:
        # Triton-Ascend rejects selecting between pointer values that originate
        # from different tensors. Keep the init-state path valid by loading
        # both candidate states and selecting the value, not the pointer.
        safe_prev_chunk = tl.where(pid_c >= 1, pid_c - 1, 0)
        state_prev_states_ptr = states_ptr + safe_prev_chunk * stride_states_chunk + pid_h * stride_states_head
        init_prev_states_ptr = initstates_ptr + seq_idx * stride_init_states_batch + pid_h * stride_init_states_head
    else:
        safe_prev_chunk = tl.where(pid_c >= 1, pid_c - 1, 0)
        prev_states_ptr = states_ptr + safe_prev_chunk * stride_states_chunk + pid_h * stride_states_head
        prev_states_hdim = stride_states_hdim
        prev_states_dstate = stride_states_dstate

    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)

    scale_m = fast_exp(dA_cs_m)
    if BLOCK_SIZE_DSTATE <= 128:
        C = tl.load(
            C_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate),
            other=0.0,
        )

        if HAS_INITSTATES:
            state_prev_states_ptrs = (
                state_prev_states_ptr
                + offs_n[None, :] * stride_states_hdim
                + offs_k_dstate[:, None] * stride_states_dstate
            )
            state_prev_states = tl.load(
                state_prev_states_ptrs,
                mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim),
                other=0.0,
            )
            state_prev_states = state_prev_states.to(C_ptr.dtype.element_ty)
            init_prev_states_ptrs = (
                init_prev_states_ptr
                + offs_n[None, :] * stride_init_states_hdim
                + offs_k_dstate[:, None] * stride_init_states_dstate
            )
            init_prev_states = tl.load(
                init_prev_states_ptrs,
                mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim),
                other=0.0,
            )
            init_prev_states = init_prev_states.to(C_ptr.dtype.element_ty)
            prev_states = tl.where(starts_new_sequence, init_prev_states, state_prev_states)
        else:
            if seq_idx != seq_idx_prev:
                prev_states = tl.zeros(
                    (BLOCK_SIZE_DSTATE, BLOCK_SIZE_N),
                    dtype=C_ptr.dtype.element_ty,
                )
            else:
                prev_states_ptrs = (
                    prev_states_ptr + offs_n[None, :] * prev_states_hdim + offs_k_dstate[:, None] * prev_states_dstate
                )
                prev_states = tl.load(
                    prev_states_ptrs,
                    mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim),
                    other=0.0,
                )
                prev_states = prev_states.to(C_ptr.dtype.element_ty)

        acc = tl.dot(C, prev_states) * scale_m[:, None]

    else:
        if HAS_INITSTATES:
            state_prev_states_ptrs = (
                state_prev_states_ptr
                + offs_n[None, :] * stride_states_hdim
                + offs_k_dstate[:, None] * stride_states_dstate
            )
            init_prev_states_ptrs = (
                init_prev_states_ptr
                + offs_n[None, :] * stride_init_states_hdim
                + offs_k_dstate[:, None] * stride_init_states_dstate
            )
        else:
            prev_states_ptrs = (
                prev_states_ptr + offs_n[None, :] * prev_states_hdim + offs_k_dstate[:, None] * prev_states_dstate
            )
        for k in range(0, dstate, BLOCK_SIZE_K):
            C = tl.load(
                C_ptrs,
                mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate - k),
                other=0.0,
            )
            if HAS_INITSTATES:
                state_prev_states = tl.load(
                    state_prev_states_ptrs,
                    mask=(offs_k_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim),
                    other=0.0,
                )
                state_prev_states = state_prev_states.to(C_ptr.dtype.element_ty)
                init_prev_states = tl.load(
                    init_prev_states_ptrs,
                    mask=(offs_k_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim),
                    other=0.0,
                )
                init_prev_states = init_prev_states.to(C_ptr.dtype.element_ty)
                prev_states = tl.where(starts_new_sequence, init_prev_states, state_prev_states)
                state_prev_states_ptrs += BLOCK_SIZE_K
                init_prev_states_ptrs += BLOCK_SIZE_K
            else:
                if seq_idx != seq_idx_prev:
                    prev_states = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=C_ptr.dtype.element_ty)
                else:
                    prev_states = tl.load(
                        prev_states_ptrs,
                        mask=(offs_k_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim),
                        other=0.0,
                    )
                    prev_states = prev_states.to(C_ptr.dtype.element_ty)
                prev_states_ptrs += BLOCK_SIZE_K
            acc += tl.dot(C, prev_states)
            C_ptrs += BLOCK_SIZE_K
        acc *= scale_m[:, None]

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit if not IS_CAUSAL else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(
            cb_ptrs,
            mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= fast_exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= dt_k
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)
        x = tl.load(
            x_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim),
            other=0.0,
        )
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(
            x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc += x_residual * D

    if HAS_Z:
        z_ptr += chunk_seqlen_start * stride_z_seqlen + pid_h * stride_z_head
        z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :])
        z = tl.load(
            z_ptrs,
            mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    out_ptr += chunk_seqlen_start * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
    tl.store(
        out_ptrs,
        acc,
        mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim),
    )


def _chunk_scan_fwd_no_initial_states_npu(
    cb,
    x,
    dt,
    dA_cumsum,
    C,
    states,
    cu_chunk_seqlens,
    out,
    seq_idx,
    D=None,
    z=None,
):
    assert seq_idx is not None, "this implementation requires seq_idx"

    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (seqlen, ngroups, dstate)
    assert cb.shape == (nchunks, ngroups, chunk_size, chunk_size)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if z is not None:
        assert z.shape == x.shape
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    assert states.shape == (nchunks, nheads, headdim, dstate)
    assert seq_idx.shape == (nchunks,)

    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)

    # Triton-Ascend may type-check pointers in constexpr-false branches. Keep
    # semantic flags false, but pass non-null tensors for optional pointers.
    # The initstates branch selects between initstates_ptr and states_ptr, so
    # the dummy pointer must share the states base to satisfy pointer-source
    # checks even when HAS_INITSTATES is false.
    z_ptr = _chunk_scan_optional_kernel_args(x, z, (1, 1, 1))
    D_ptr = _chunk_scan_optional_kernel_args(x, D, (1,))
    initstates_ptr = _chunk_scan_initial_states_kernel_arg(states, None)

    for chunk_start, launch_nchunks in _chunk_scan_launch_ranges(chunk_size, headdim, nchunks, nheads):
        grid = _chunk_scan_grid(chunk_size, headdim, launch_nchunks, nheads)
        _chunk_scan_fwd_kernel_npu[grid](
            cb_ptr=cb,
            x_ptr=x,
            z_ptr=z_ptr,
            out_ptr=out,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            seq_idx_ptr=seq_idx,
            C_ptr=C,
            states_ptr=states,
            D_ptr=D_ptr,
            initstates_ptr=initstates_ptr,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            chunk_start=chunk_start,
            chunk_size=chunk_size,
            hdim=headdim,
            dstate=dstate,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_cb_chunk=cb.stride(0),
            stride_cb_head=cb.stride(1),
            stride_cb_csize_m=cb.stride(2),
            stride_cb_csize_k=cb.stride(3),
            stride_x_seqlen=x.stride(0),
            stride_x_head=x.stride(1),
            stride_x_hdim=x.stride(2),
            stride_z_seqlen=z_strides[0],
            stride_z_head=z_strides[1],
            stride_z_hdim=z_strides[2],
            stride_out_seqlen=out.stride(0),
            stride_out_head=out.stride(1),
            stride_out_hdim=out.stride(2),
            stride_dt_chunk=dt.stride(1),
            stride_dt_head=dt.stride(0),
            stride_dt_csize=dt.stride(2),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_seq_idx_chunk=seq_idx.stride(0),
            stride_C_seqlen=C.stride(0),
            stride_C_head=C.stride(1),
            stride_C_dstate=C.stride(2),
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_hdim=states.stride(2),
            stride_states_dstate=states.stride(3),
            stride_init_states_batch=0,
            stride_init_states_head=0,
            stride_init_states_hdim=0,
            stride_init_states_dstate=0,
            stride_D_head=D.stride(0) if D is not None else 0,
            IS_CAUSAL=True,
            HAS_D=D is not None,
            D_HAS_HDIM=D.dim() == 2 if D is not None else True,
            HAS_Z=z is not None,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
            IS_TRITON_22=_ssd_chunk_scan.TRITON_22,
            HAS_INITSTATES=False,
        )
    return None


def _chunk_scan_fwd_npu(
    cb,
    x,
    dt,
    dA_cumsum,
    C,
    states,
    cu_chunk_seqlens,
    out,
    seq_idx,
    D=None,
    z=None,
    initial_states=None,
):
    if x.device.type != "npu":
        return _ORIGINAL_CHUNK_SCAN_FWD(
            cb,
            x,
            dt,
            dA_cumsum,
            C,
            states,
            cu_chunk_seqlens,
            out,
            seq_idx,
            D=D,
            z=z,
            initial_states=initial_states,
        )
    if initial_states is None:
        return _chunk_scan_fwd_no_initial_states_npu(
            cb,
            x,
            dt,
            dA_cumsum,
            C,
            states,
            cu_chunk_seqlens,
            out,
            seq_idx,
            D=D,
            z=z,
        )

    assert seq_idx is not None, "this implementation requires seq_idx"

    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (seqlen, ngroups, dstate)
    assert cb.shape == (nchunks, ngroups, chunk_size, chunk_size)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if z is not None:
        assert z.shape == x.shape
    if initial_states is not None:
        assert initial_states.shape[1:] == (nheads, headdim, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (nheads, nchunks, chunk_size)
    assert states.shape == (nchunks, nheads, headdim, dstate)
    assert seq_idx.shape == (nchunks,)

    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    initial_states_strides = _chunk_scan_initial_states_strides(initial_states)

    # Triton-Ascend may type-check pointers in constexpr-false branches. Keep
    # semantic flags false, but pass non-null tensors for optional pointers.
    z_ptr = _chunk_scan_optional_kernel_args(x, z, (1, 1, 1))
    D_ptr = _chunk_scan_optional_kernel_args(x, D, (1,))
    initstates_ptr = _chunk_scan_initial_states_kernel_arg(states, initial_states)

    has_initstates = initial_states is not None

    for chunk_start, launch_nchunks in _chunk_scan_launch_ranges(chunk_size, headdim, nchunks, nheads):
        grid = _chunk_scan_grid(chunk_size, headdim, launch_nchunks, nheads)
        _chunk_scan_fwd_kernel_npu[grid](
            cb_ptr=cb,
            x_ptr=x,
            z_ptr=z_ptr,
            out_ptr=out,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            seq_idx_ptr=seq_idx,
            C_ptr=C,
            states_ptr=states,
            D_ptr=D_ptr,
            initstates_ptr=initstates_ptr,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            chunk_start=chunk_start,
            chunk_size=chunk_size,
            hdim=headdim,
            dstate=dstate,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_cb_chunk=cb.stride(0),
            stride_cb_head=cb.stride(1),
            stride_cb_csize_m=cb.stride(2),
            stride_cb_csize_k=cb.stride(3),
            stride_x_seqlen=x.stride(0),
            stride_x_head=x.stride(1),
            stride_x_hdim=x.stride(2),
            stride_z_seqlen=z_strides[0],
            stride_z_head=z_strides[1],
            stride_z_hdim=z_strides[2],
            stride_out_seqlen=out.stride(0),
            stride_out_head=out.stride(1),
            stride_out_hdim=out.stride(2),
            stride_dt_chunk=dt.stride(1),
            stride_dt_head=dt.stride(0),
            stride_dt_csize=dt.stride(2),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_seq_idx_chunk=seq_idx.stride(0),
            stride_C_seqlen=C.stride(0),
            stride_C_head=C.stride(1),
            stride_C_dstate=C.stride(2),
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_hdim=states.stride(2),
            stride_states_dstate=states.stride(3),
            stride_init_states_batch=initial_states_strides[0],
            stride_init_states_head=initial_states_strides[1],
            stride_init_states_hdim=initial_states_strides[2],
            stride_init_states_dstate=initial_states_strides[3],
            stride_D_head=D.stride(0) if D is not None else 0,
            IS_CAUSAL=True,
            HAS_D=D is not None,
            D_HAS_HDIM=D.dim() == 2 if D is not None else True,
            HAS_Z=z is not None,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
            IS_TRITON_22=_ssd_chunk_scan.TRITON_22,
            HAS_INITSTATES=has_initstates,
        )
    return None
