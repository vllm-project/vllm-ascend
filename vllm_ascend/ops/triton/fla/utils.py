# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import contextlib
import functools
import math
import os
from collections.abc import Callable

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.fused_gdn_gating import _gdn_gating_rows
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

from .l2norm import _l2norm_rows


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_final_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = triton.cdiv(prepare_lens(cu_seqlens), chunk_size) + 1
    return torch.cumsum(indices, 0) - 1


def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)


def prepare_update_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size) + 1]).cumsum(-1)


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = torch.npu.device(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.jit(do_not_specialize=["inner_size", "row_stride"])
def _clear_ssm_states_kernel(
    states_ptr,
    has_initial_state_ptr,
    inner_size,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    col_block_idx = tl.program_id(axis=1)

    has_state = tl.load(has_initial_state_ptr + row_idx).to(tl.int1)
    if has_state:
        return

    cols = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < inner_size
    row_ptr = states_ptr + row_idx * row_stride + cols
    tl.store(row_ptr, tl.zeros((BLOCK_SIZE,), dtype=states_ptr.dtype.element_ty), mask=mask)


def clear_ssm_states(ssm_states: torch.Tensor, has_initial_state: torch.Tensor) -> None:
    """Zero out specific rows for the SSM states

    Args:
        ssm_states (torch.Tensor): input SSM states
        has_initial_state (torch.Tensor): indicates whether the row has initial states already
    """
    if ssm_states.numel() == 0:
        return

    if has_initial_state.device != ssm_states.device:
        has_initial_state = has_initial_state.to(ssm_states.device, non_blocking=True)
    if has_initial_state.dtype != torch.bool:
        has_initial_state = has_initial_state.to(torch.bool)

    has_initial_state = has_initial_state.reshape(-1).contiguous()
    num_rows = ssm_states.shape[0]
    if num_rows == 0:
        return
    if has_initial_state.numel() != num_rows:
        raise ValueError(
            f"clear_ssm_states: has_initial_state size mismatch: expected {num_rows}, got {has_initial_state.numel()}"
        )
    inner_size = ssm_states.numel() // num_rows
    if inner_size == 0:
        return

    block_size = 4096
    grid = (num_rows, triton.cdiv(inner_size, block_size))
    _clear_ssm_states_kernel[grid](
        ssm_states,
        has_initial_state,
        inner_size,
        ssm_states.stride(0),
        BLOCK_SIZE=block_size,
    )
# Pre-bound launcher cache: spec-key -> triton CompiledKernel.
#
# JITFunction.run() redoes per call: make_backend(target), three driver queries,
# argument binding, a string cache key (''.join(sig) + str(constexprs)) and a scan of
# used_global_vals - about 20us of the ~283us host cost of a Triton launch on this
# stack (measured: full path 31.99us vs pre-bound 11.93us in a warm loop). Calling the
# CompiledKernel directly skips all of it.
#
# Correctness: Triton 3.2 keys its cache on arg type strings plus the attrs descriptor.
# Probing this kernel over 5 shape variants (T = 3512/1464/127 tokens) yields exactly
# one specialisation, and the key shows 14 divisibility markers - one per pointer.
# Because every int arg is in do_not_specialize, ints do not affect the key at all;
# only pointer dtype and 16-byte alignment do. The guard below therefore covers exactly
# what Triton would have recomputed.
_prebound_kernels: dict = {}


def _prebound_key(ptrs, constexprs):
    """Spec key covering exactly what Triton keys on: pointer dtype + 16B alignment."""
    return (
        constexprs,
        tuple((t.dtype, (t.data_ptr() & 15) == 0) for t in ptrs),
    )


def preamble_fusion_enabled() -> bool:
    """Whether the GDN prefill preamble may be fused into one Triton launch.

    ``VLLM_ASCEND_GDN_FUSE_CLEAR_L2NORM=0`` restores the original per-op path
    (separate gating, Index, clear_ssm_states, cast and two l2norm launches).
    """
    return os.environ.get("VLLM_ASCEND_GDN_FUSE_CLEAR_L2NORM", "1") == "1"


def gating_gather_clear_l2norm_qk(
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    eps: float = 1e-6,
    out_dtype: torch.dtype | None = None,
    gbeta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole GDN prefill preamble - gating, the ssm gather, the clear, the dtype
    cast and both l2norms - in a single Triton launch.

    Replaces ``fused_gdn_gating_patch`` plus the ``Index -> clear_ssm_states -> cast``
    chain plus two ``l2norm_fwd`` calls, i.e. four launches by one. A Triton launch
    costs ~300us of host time on this stack while these kernels run in under 20us, so
    the launch count is what decides the cost. Collapsing to one took a GDN layer from
    1085us of host time to 356us, below its 661us of device time, which dropped the
    prep bubble from 582us to 163us.

    Returns ``(g, beta_output, initial_state, q_norm, k_norm)``. Callers must gate this
    on :func:`preamble_fusion_enabled`; it is only valid for a pure-prefill batch. The
    preconditions below hold for every GDN model we support, so a violation is a bug
    rather than a case to degrade silently.
    """
    batch, num_heads = a.shape
    if not (q.shape == k.shape and q.dtype == k.dtype):
        raise ValueError(f"gating_gather_clear_l2norm_qk: q/k must match, got {q.shape}/{q.dtype} vs {k.shape}/{k.dtype}. Set VLLM_ASCEND_GDN_FUSE_CLEAR_L2NORM=0 to use the unfused path.")
    if not (ssm_state.numel() > 0 and ssm_state.is_contiguous()):
        raise ValueError("gating_gather_clear_l2norm_qk: ssm_state must be non-empty and contiguous. Set VLLM_ASCEND_GDN_FUSE_CLEAR_L2NORM=0 to use the unfused path.")
    if not (a.is_contiguous() and b.is_contiguous() and a.shape == b.shape):
        raise ValueError(f"gating_gather_clear_l2norm_qk: a/b must be contiguous and same-shaped, got {a.shape}/{b.shape}. Set VLLM_ASCEND_GDN_FUSE_CLEAR_L2NORM=0 to use the unfused path.")

    # The builder already normalized these once per step, so only convert if a caller
    # hands us something unexpected; redoing it costs ~6 aten dispatches per layer.
    if state_indices.dtype != torch.int32 or not state_indices.is_contiguous() or state_indices.dim() != 1:
        state_indices = state_indices.reshape(-1).to(torch.int32).contiguous()
    if (
        has_initial_state.dtype != torch.bool
        or not has_initial_state.is_contiguous()
        or has_initial_state.dim() != 1
        or has_initial_state.device != ssm_state.device
    ):
        has_initial_state = has_initial_state.to(device=ssm_state.device, dtype=torch.bool).reshape(-1).contiguous()

    num_rows = state_indices.numel()
    inner_size = math.prod(ssm_state.shape[1:]) if ssm_state.shape[0] else 0
    if num_rows == 0 or inner_size == 0:
        raise ValueError(f"gating_gather_clear_l2norm_qk: empty state (rows={num_rows}, inner={inner_size}); the caller should not reach here for a prefill batch.")
    if has_initial_state.numel() != num_rows:
        raise ValueError(f"gating_gather_clear_l2norm_qk: has_initial_state size mismatch: expected {num_rows}, got {has_initial_state.numel()}")

    # reshape also normalizes a non-contiguous input by copying, which the kernel's
    # flat indexing (X + rindex + N * row_idx) relies on.
    q_shape_og, k_shape_og = q.shape, k.shape
    q2 = q.reshape(-1, q.shape[-1])
    k2 = k.reshape(-1, k.shape[-1])
    T, D = q2.shape[0], q2.shape[-1]
    assert q2.stride(-1) == 1 and k2.stride(-1) == 1

    max_fused_size = 65536 // q2.element_size()
    if D > min(max_fused_size, triton.next_power_of_2(D)):
        raise RuntimeError(f"gating_gather_clear_l2norm_qk: feature dim >= 64KB not supported, got {D}.")

    y_q = torch.empty_like(q2)
    y_k = torch.empty_like(k2)
    dst_dtype = out_dtype if out_dtype is not None else ssm_state.dtype
    initial_state = torch.empty((num_rows, *ssm_state.shape[1:]), dtype=dst_dtype, device=ssm_state.device)
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_out = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)

    num_core = get_vectorcore_num()
    block_size = 4096
    col_blocks = triton.cdiv(inner_size, block_size)
    n_gather = num_rows * col_blocks
    mblock = 69
    num_sub_blocks = triton.cdiv(triton.cdiv(T, num_core), mblock)
    # Mirrors the grid math in fused_gdn_gating_patch.
    blk_heads, blk_batches = 8, 64
    row_iter = triton.cdiv(triton.cdiv(batch, num_core), blk_batches)

    grid0 = n_gather + 3 * num_core
    ptrs = (ssm_state, state_indices, has_initial_state, initial_state, q2, y_q, k2, y_k,
            g, beta_out, A_log, a, b, dt_bias)
    # Only the real constexprs belong in the key; NUM_CHUNKS is a runtime arg in
    # do_not_specialize, so it does not affect the specialisation.
    constexprs = (D, mblock, num_core, block_size, blk_heads, blk_batches)
    key = _prebound_key(ptrs, constexprs)
    ck = _prebound_kernels.get(key)

    if ck is None:
        # First call for this spec: go through JITFunction.run so Triton compiles and
        # caches, then capture the CompiledKernel for subsequent launches.
        compiled = _gating_gather_clear_l2norm_qk_kernel[(grid0,)](
            ssm_state, state_indices, has_initial_state, initial_state, inner_size,
            ssm_state.stride(0), q2, y_q, k2, y_k, eps, T, g, beta_out, A_log, a, b,
            dt_bias, num_heads, batch, gbeta, threshold, row_iter, n_gather, col_blocks,
            N=D, MBLOCK=mblock, NUM_CHUNKS=num_sub_blocks, NUM_CORE=num_core,
            BLOCK_SIZE=block_size, BLK_HEADS=blk_heads, BLK_BATCHES=blk_batches,
        )
        # run() returns the CompiledKernel it used, which is safer than guessing from
        # the cache dict when several specialisations coexist.
        if (
            os.environ.get("VLLM_ASCEND_GDN_PREBIND_LAUNCH", "1") == "1"
            and compiled is not None
            and hasattr(compiled, "run")
            and hasattr(compiled, "packed_metadata")
        ):
            _prebound_kernels[key] = compiled
        return g, beta_out, initial_state, y_q.view(q_shape_og), y_k.view(k_shape_og)

    # Pre-bound: mirrors the tail of JITFunction.run without its preamble. Non-constexpr
    # values in declaration order; launch_metadata and the enter/exit hooks are None
    # because nothing installs them here.
    from triton.runtime import driver

    stream = driver.active.get_current_stream(driver.active.get_current_device())
    # Argument order is the kernel's declaration order with constexprs removed, so
    # NUM_CHUNKS lands last (it is declared after N/MBLOCK but is not a constexpr).
    ck.run(
        grid0, 1, 1, stream, ck.function, ck.packed_metadata, None, None, None,
        ssm_state, state_indices, has_initial_state, initial_state, inner_size,
        ssm_state.stride(0), q2, y_q, k2, y_k, eps, T, g, beta_out, A_log, a, b,
        dt_bias, num_heads, batch, gbeta, threshold, row_iter, n_gather, col_blocks,
        num_sub_blocks,
    )
    return g, beta_out, initial_state, y_q.view(q_shape_og), y_k.view(k_shape_og)


@triton.jit(
    do_not_specialize=[
        "inner_size",
        "src_row_stride",
        "eps",
        "M",
        "NUM_CHUNKS",
        "N_GATHER",
        "COL_BLOCKS",
        "NUM_HEADS",
        "NUM_BATCHES",
        "gbeta",
        "threshold",
        "ROW_ITER",
    ]
)
def _gating_gather_clear_l2norm_qk_kernel(
    src_ptr,
    idx_ptr,
    has_initial_state_ptr,
    dst_ptr,
    inner_size,
    src_row_stride,
    XQ,
    YQ,
    XK,
    YK,
    eps,
    M,
    G,
    BETA_OUT,
    A_LOG,
    A,
    B,
    DT_BIAS,
    NUM_HEADS,
    NUM_BATCHES,
    gbeta,
    threshold,
    ROW_ITER,
    N_GATHER,
    COL_BLOCKS,
    N: tl.constexpr,
    MBLOCK: tl.constexpr,
    NUM_CHUNKS,
    NUM_CORE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLK_HEADS: tl.constexpr,
    BLK_BATCHES: tl.constexpr,
):
    """gating + gather + clear + cast + l2norm(q) + l2norm(k) in one launch.

    Grid partition: ``[0, N_GATHER)`` gathers ssm rows, then ``NUM_CORE`` programs
    each for gating, q and k. The four jobs write disjoint tensors (g/beta_out,
    initial_state, y_q, y_k) and none reads another's output, so their order inside
    the launch is free.
    """
    pid = tl.program_id(0)
    if pid < N_GATHER:
        row_idx = pid // COL_BLOCKS
        col_block_idx = pid % COL_BLOCKS
        cols = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < inner_size
        dst = dst_ptr + row_idx.to(tl.int64) * inner_size + cols
        has_state = tl.load(has_initial_state_ptr + row_idx).to(tl.int1)
        if has_state:
            src_row = tl.load(idx_ptr + row_idx).to(tl.int64)
            vals = tl.load(src_ptr + src_row * src_row_stride + cols, mask=mask, other=0.0)
            tl.store(dst, vals.to(dst_ptr.dtype.element_ty), mask=mask)
        else:
            tl.store(dst, tl.zeros((BLOCK_SIZE,), dtype=dst_ptr.dtype.element_ty), mask=mask)
    elif pid < N_GATHER + NUM_CORE:
        # seq_len is 1 and i_s is 0 here, which collapses the gating row offset to
        # batch_off * NUM_HEADS + head_off.
        _gdn_gating_rows(
            G, BETA_OUT, A_LOG, A, B, DT_BIAS, 1, NUM_HEADS, NUM_BATCHES,
            gbeta, threshold, BLK_HEADS, BLK_BATCHES, ROW_ITER, pid - N_GATHER, 0,
        )
    elif pid < N_GATHER + 2 * NUM_CORE:
        _l2norm_rows(XQ, YQ, eps, M, N, MBLOCK, NUM_CHUNKS, pid - N_GATHER - NUM_CORE)
    else:
        _l2norm_rows(XK, YK, eps, M, N, MBLOCK, NUM_CHUNKS, pid - N_GATHER - 2 * NUM_CORE)
