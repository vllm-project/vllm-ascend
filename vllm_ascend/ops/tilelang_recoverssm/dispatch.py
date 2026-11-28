# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 RecoverSSM dispatch for the decode worker."""

import logging
from functools import cache, lru_cache, wraps

import torch

_ROWS = (1, 2, 4, 8, 9, 11, 12, 16)
_POSTPROCESS_ROWS = range(1, 17)
_LOG = logging.getLogger(__name__)


@cache
def _is_a3(device):
    return device.type == "npu" and torch.npu.get_device_name(device) == "Ascend910_9382"


@lru_cache(maxsize=64)
def _verify_kernel(
    rows, pages, tokens, state_stride, replay_stride, index_stride, gate_stride, b_token_stride, v_token_stride
):
    if (
        rows in (12, 16)
        and tokens == 4 * rows
        and pages in (2403, 2404)
        and (state_stride, replay_stride) == (458752, 24672)
        and index_stride in (1, 4)
        and (gate_stride, b_token_stride) in ((24, 24), (48, 48))
        and v_token_stride in (3072, 5120)
    ):
        from .window_verify import build_window

        _LOG.warning(
            "TILELANG_GDN_WINDOW_VERIFY rows=%d tokens=%d pages=%d strides=%s",
            rows,
            tokens,
            pages,
            (state_stride, replay_stride, index_stride, gate_stride, b_token_stride, v_token_stride),
        )
        return build_window(
            rows, pages, tokens, state_stride, replay_stride, index_stride, gate_stride, b_token_stride, v_token_stride
        )

    from .verify import build

    packed_replay = rows in (8, 12) and tokens == 4 * rows
    _LOG.warning(
        "TILELANG_GDN_VERIFY rows=%s tokens=%s pages=%s strides=%s packed_replay=%s",
        rows,
        tokens,
        pages,
        (state_stride, replay_stride, index_stride, gate_stride, b_token_stride, v_token_stride),
        packed_replay,
    )
    return build(
        128,
        rows=rows,
        pages=pages,
        tokens=tokens,
        state_stride=state_stride,
        replay_stride=replay_stride,
        index_stride=index_stride,
        gate_stride=gate_stride,
        b_token_stride=b_token_stride,
        v_token_stride=v_token_stride,
        rebase_pages=True,
        packed_replay=packed_replay,
    )


def wrap_verify(original):
    @wraps(original)
    def verify(A_log, a, b, dt_bias, q, k, v, **kw):
        h, replay = kw["checkpoint_state"], kw["replay_cache"]
        slots, starts = kw["state_indices"], kw["query_start_loc"]
        out = kw.get("out")
        if not _is_a3(q.device) or q.ndim != 4 or h.ndim != 4:
            return original(A_log, a, b, dt_bias, q, k, v, **kw)
        tokens, rows, pages = q.shape[1], slots.numel(), h.shape[0]
        if (
            rows not in _ROWS
            or pages < 1
            or slots.ndim != 1
            or slots.stride(0) < 1
            or any(t.device != q.device for t in (A_log, a, b, dt_bias, k, v, h, replay, slots, starts))
            or kw["spec_query_len"] != 4
            or kw.get("use_qk_l2norm_in_kernel", True)
            or kw.get("scale") not in (None, 128**-0.5)
            or kw.get("softplus_beta", 1.0) != 1
            or kw.get("softplus_threshold", 20.0) != 20
            or not (0 < tokens <= 4 * rows)
            or q.shape != (1, tokens, 8, 128)
            or k.shape != q.shape
            or v.shape != (1, tokens, 24, 128)
            or v.stride()[2:] != (128, 1)
            or h.shape[1:] != (24, 128, 128)
            or h.stride()[1:] != (16384, 128, 1)
            or replay.shape != (pages, 24, 4, 257)
            or replay.stride()[1:] != (1028, 257, 1)
            or h.stride(0) < 24 * 128 * 128
            or replay.stride(0) < 24 * 4 * 257
            or not q.is_contiguous()
            or not k.is_contiguous()
            or A_log.shape != (24,)
            or dt_bias.shape != (24,)
            or not A_log.is_contiguous()
            or not dt_bias.is_contiguous()
            or starts.shape != (rows + 1,)
            or not starts.is_contiguous()
            or starts.dtype != torch.int32
            or slots.dtype != torch.int32
            or A_log.dtype != torch.float32
            or replay.dtype != torch.float32
            or any(t.dtype != torch.bfloat16 for t in (q, k, v, h, a, b, dt_bias))
            or a.shape not in ((tokens, 24), (1, tokens, 24))
            or b.shape != a.shape
            or a.stride(-2) < 24
            or b.stride(-2) < 24
            or v.stride(1) < 24 * 128
            or a.stride(-1) != 1
            or b.stride(-1) != 1
            or (
                out is not None
                and (out.shape != v.shape or not out.is_contiguous() or out.dtype != v.dtype or out.device != v.device)
            )
        ):
            return original(A_log, a, b, dt_bias, q, k, v, **kw)
        a, b = a.view(tokens, 24), b.view(tokens, 24)
        kernel = _verify_kernel(
            rows, pages, tokens, h.stride(0), replay.stride(0), slots.stride(0), a.stride(0), b.stride(0), v.stride(1)
        )
        if out is None:
            out = torch.empty_like(v, memory_format=torch.contiguous_format)
        # The high-level Cython wrapper requires every input to be contiguous.
        # This compiled specialization already includes each physical stride.
        # Pass the view pointers on the current PyTorch NPU stream.
        kernel.adapter._forward_from_prebuild_lib(
            A_log,
            a,
            b,
            dt_bias,
            q,
            k,
            v,
            h,
            replay,
            starts,
            slots,
            out,
            stream=torch.npu.current_stream(v.device).npu_stream,
        )
        return out

    return verify


@lru_cache(maxsize=32)
def _commit_kernel(layers, rows, stride, align):
    # Page capacity is immaterial. Each page is rebased with its own INT64 stride.
    from .commit import build

    _LOG.warning("TILELANG_GDN_COMMIT L=%d B=%d index_stride=%d align=%s", layers, rows, stride, align)
    return build(layers, rows, 24, 1, steps=4, tile=128, align=align, index_stride=stride, packed=True, wide=True)


def wrap_commit(original):
    @wraps(original)
    def commit(
        state,
        replay,
        state_commit_cache_address_stride_table,
        state_indices,
        commit_lens,
        final_state_indices,
        boundary_state_indices,
        boundary_recovery_lens,
        *,
        spec_query_len,
        align_mode,
    ):
        table = state_commit_cache_address_stride_table
        plan = (commit_lens, final_state_indices, boundary_state_indices, boundary_recovery_lens)
        args = (state, replay, table, state_indices, *plan)
        rows = state_indices.numel()
        layers = table.shape[2] if table.ndim == 3 else 0
        matched = (
            (rows in _POSTPROCESS_ROWS and layers == 12)
            or (rows in _ROWS and layers in (10, 13))
            or (rows in (1, 2) and layers in (1, 2))
        )
        if (
            not _is_a3(state.device)
            or state.dtype != torch.bfloat16
            or spec_query_len != 4
            or not align_mode
            or state.ndim != 4
            or state.shape[0] < 1
            or state.stride(0) < 24 * 128 * 128
            or state.shape[1:] != (24, 128, 128)
            or state.stride()[1:] != (16384, 128, 1)
            or replay.dtype != torch.float32
            or replay.shape != (state.shape[0], 24, 4, 257)
            or replay.stride(0) < 24 * 4 * 257
            or replay.stride()[1:] != (1028, 257, 1)
            or table.ndim != 3
            or table.shape[:2] != (2, 2)
            or not matched
            or table.dtype != torch.int64
            or not table.is_contiguous()
            or state_indices.shape != (rows,)
            or state_indices.dtype != torch.int32
            or state_indices.stride(0) < 1
            or any(t.device != state.device for t in args[1:])
            or any(t.ndim != 1 or t.numel() < rows or t.dtype != torch.int32 or not t.is_contiguous() for t in plan)
        ):
            return original(*args, spec_query_len=spec_query_len, align_mode=align_mode)
        s = state_indices
        raw_s = s.as_strided((1 + (rows - 1) * s.stride(0),), (1,))
        k = _commit_kernel(layers, rows, s.stride(0), align_mode)
        k(state[0].view(-1), replay[0].view(-1), table.view(-1), raw_s, *(t[:rows] for t in plan))

    return commit


@lru_cache(maxsize=64)
def _compact_kernel(rows, pages, page_stride, index_stride, dtype):
    from .compact import build

    kernel = build(rows, pages, page_stride, 2560, index_stride, bulk=True, index_dtype=dtype)
    _LOG.warning(
        "TILELANG_PLE_COMPACT rows=%d pages=%d page_stride=%d index_stride=%d index_dtype=%s tile=2560",
        rows,
        pages,
        page_stride,
        index_stride,
        dtype,
    )
    return kernel


def compact_conv_state(
    conv_state_ref,
    conv_state_base_addrs,
    conv_state_block_strides,
    conv_state_dim_strides,
    conv_state_token_strides,
    state_indices,
    commit_lens,
    final_state_indices,
    boundary_state_indices,
    boundary_recovery_lens,
    *,
    conv_history_len,
    align_mode,
    dim_stride,
    null_block_id,
):
    """Compact PLE history using a single-layer table constructed from conv_state_ref.

    The caller must bind the base pointer and all three table strides to this
    same tensor. The fast path addresses its storage directly for graph replay.
    """
    shape = conv_state_ref.shape
    plan = (commit_lens, final_state_indices, boundary_state_indices, boundary_recovery_lens)
    rows = state_indices.numel()
    if (
        _is_a3(conv_state_ref.device)
        and conv_state_ref.ndim == 3
        and shape[0] > 0
        and conv_state_ref.dtype == torch.bfloat16
        and conv_state_base_addrs.numel() == 1
        and shape[1:] == (10240, 12)
        and conv_state_ref.stride()[1:] == (1, 10240)
        and conv_state_ref.stride(0) >= 122880
        and align_mode
        and null_block_id == 0
        and conv_history_len == 9
        and dim_stride is None
        and rows in _POSTPROCESS_ROWS
        and all(t.device == conv_state_ref.device for t in (state_indices, *plan))
        and state_indices.ndim == 1
        and state_indices.stride(0) > 0
        and state_indices.dtype in (torch.int32, torch.int64)
        and all(t.ndim == 1 and t.numel() >= rows and t.stride(0) == 1 and t.dtype == torch.int32 for t in plan)
    ):
        pages, page_stride = shape[0], conv_state_ref.stride(0)
        index_stride = state_indices.stride(0)
        dtype = "int32" if state_indices.dtype == torch.int32 else "int64"
        # A flat, contiguous address span preserves the original storage offset and padding,
        # and is accepted by the TileLang Cython adapter without copying cache or sampler rows.
        cache_span = (pages - 1) * page_stride + 122880
        cache = conv_state_ref.as_strided((cache_span,), (1,))
        slots = state_indices.as_strided(((rows - 1) * index_stride + 1,), (1,))
        _compact_kernel(rows, pages, page_stride, index_stride, dtype)(cache, slots, *(t[:rows] for t in plan))
        return
    from vllm_ascend.ops.triton.mamba.compact_conv_state import compact_conv_state as triton_compact

    return triton_compact(
        conv_state_ref,
        conv_state_base_addrs,
        conv_state_block_strides,
        conv_state_dim_strides,
        conv_state_token_strides,
        state_indices,
        *plan,
        conv_history_len=conv_history_len,
        align_mode=align_mode,
        dim_stride=dim_stride,
        null_block_id=null_block_id,
    )
