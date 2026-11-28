# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 Flash dfly ops."""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

RESIDUAL_BLOCK = 1024


@triton.jit
def _compact_c8_indices_kernel(
    selected, order, indices, lengths, active, candidates: tl.constexpr, block: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    live = col < candidates
    valid = live & (tl.load(selected + row * candidates + col, live, -1) >= 0)
    rank = tl.cumsum(valid.to(tl.int32), 0) - 1
    count = tl.sum(valid.to(tl.int32), 0)
    # The two store masks address disjoint output ranges.
    tl.store(order + row * candidates + rank, col.to(tl.int64), valid)
    tl.store(order + row * candidates + col, 0, live & (col >= count))
    native_index = tl.where(col < count, col, -1)
    # A fully padded row uses one safe physical KV row and is zeroed below.
    native_index = tl.where((count == 0) & (col == 0), 0, native_index)
    tl.store(indices + row * candidates + col, native_index, live)
    tl.store(lengths + row, tl.maximum(count, 1))
    tl.store(active + row, count > 0)


def compact_draft_c8_rows(rows, selected):
    """Pack valid candidate rows before calling the native sparse attention API."""
    batch, candidates, row_bytes = rows.shape
    queries = selected.shape[1] if selected.ndim == 3 else 1
    count = batch * queries
    order = torch.empty((count, candidates), device=rows.device, dtype=torch.int64)
    indices = torch.empty((count, candidates), device=rows.device, dtype=torch.int32)
    lengths = torch.empty(count, device=rows.device, dtype=torch.int32)
    active = torch.empty(count, device=rows.device, dtype=torch.bool)
    _compact_c8_indices_kernel[(count,)](
        selected.contiguous(), order, indices, lengths, active, candidates, triton.next_power_of_2(candidates)
    )
    source = rows[:, None].expand(batch, queries, candidates, row_bytes)
    gather = order.view(batch, queries, candidates, 1).expand(-1, -1, -1, row_bytes)
    packed = source.gather(2, gather).reshape(count, candidates, row_bytes)
    return packed, indices[:, None, None, :], lengths, active


@triton.jit
def _dual_rms_concat_kernel(
    a,
    b,
    wa,
    wb,
    out,
    a_stride: tl.constexpr,
    b_stride: tl.constexpr,
    a_col: tl.constexpr,
    b_col: tl.constexpr,
    width: tl.constexpr,
    eps_a: tl.constexpr,
    eps_b: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    side = tl.program_id(1)
    col = tl.arange(0, block)
    if side == 0:
        x = tl.load(a + row * a_stride + col * a_col, col < width, 0).to(tl.float32)
        weight = tl.load(wa + col, col < width, 0).to(tl.float32)
        inverse = tl.rsqrt(tl.sum(x * x, 0) / width + eps_a)
    else:
        x = tl.load(b + row * b_stride + col * b_col, col < width, 0).to(tl.float32)
        weight = tl.load(wb + col, col < width, 0).to(tl.float32)
        inverse = tl.rsqrt(tl.sum(x * x, 0) / width + eps_b)
    value = (x * inverse) * weight
    tl.store(out + row * (2 * width) + side * width + col, value, col < width)


@triton.jit
def _gated_residual_kernel(
    anchor,
    branch,
    gate,
    out,
    count: tl.constexpr,
    width: tl.constexpr,
    anchor_row: tl.constexpr,
    anchor_col: tl.constexpr,
    branch_row: tl.constexpr,
    branch_col: tl.constexpr,
    gate_stride: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    live = index < count
    row = index // width
    col = index % width
    x = tl.load(anchor + row * anchor_row + col * anchor_col, live, 0).to(tl.float32)
    y = tl.load(branch + row * branch_row + col * branch_col, live, 0).to(tl.float32)
    gain = tl.load(gate + col * gate_stride, live, 0).to(tl.float32)
    update = (gain * y).to(branch.dtype.element_ty).to(tl.float32)
    tl.store(out + index, x + update, live)


def dual_rms_concat(a, b, norm_a, norm_b):
    rows, width = a.shape
    output = torch.empty((rows, 2 * width), device=a.device, dtype=a.dtype)
    _dual_rms_concat_kernel[(rows, 2)](
        a,
        b,
        norm_a.weight,
        norm_b.weight,
        output,
        a.stride(0),
        b.stride(0),
        a.stride(1),
        b.stride(1),
        width,
        norm_a.variance_epsilon,
        norm_b.variance_epsilon,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return output


def gated_residual(anchor, branch, gate):
    output = torch.empty(anchor.shape, device=anchor.device, dtype=anchor.dtype)
    _gated_residual_kernel[(triton.cdiv(anchor.numel(), RESIDUAL_BLOCK),)](
        anchor,
        branch,
        gate,
        output,
        anchor.numel(),
        anchor.shape[-1],
        anchor.stride(0),
        anchor.stride(1),
        branch.stride(0),
        branch.stride(1),
        gate.stride(0),
        RESIDUAL_BLOCK,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _rounded_add_rms_kernel(
    hidden,
    residual,
    gamma,
    normalized,
    updated,
    width: tl.constexpr,
    hs0: tl.constexpr,
    hs1: tl.constexpr,
    rs0: tl.constexpr,
    rs1: tl.constexpr,
    epsilon: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    live = col < width
    h = tl.load(hidden + row * hs0 + col * hs1, live, 0).to(tl.float32)
    r = tl.load(residual + row * rs0 + col * rs1, live, 0).to(tl.float32)
    value = (h + r).to(hidden.dtype.element_ty).to(tl.float32)
    value = tl.where(live, value, 0.0)
    inverse = tl.rsqrt(tl.sum(value * value, 0) / width + epsilon)
    weight = tl.load(gamma + col, live, 0).to(tl.float32)
    tl.store(updated + row * width + col, value, live)
    tl.store(normalized + row * width + col, value * inverse * weight, live)


def rounded_add_rms_norm(hidden, residual, norm):
    rows, width = hidden.shape
    normalized = torch.empty((rows, width), device=hidden.device, dtype=hidden.dtype)
    updated = torch.empty_like(normalized)
    _rounded_add_rms_kernel[(rows,)](
        hidden,
        residual,
        norm.weight,
        normalized,
        updated,
        width,
        hidden.stride(0),
        hidden.stride(1),
        residual.stride(0),
        residual.stride(1),
        norm.variance_epsilon,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return normalized, updated


PREPARE_BLOCK = 256
HIDDEN_BLOCK = 1024


@triton.jit(do_not_specialize=["num_reqs", "padded_batch"])
def _prepare_query_kernel(
    ids,
    positions,
    padding,
    starts,
    lengths,
    hidden_out,
    active_out,
    sources,
    anchors,
    mrope,
    target_starts,
    target_positions,
    mapping,
    sampled,
    rejected,
    previous,
    hidden_in,
    delta,
    num_reqs,
    padded_batch,
    WIDTH: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    HIDDEN: tl.constexpr,
    HIDDEN_IN_STRIDE: tl.constexpr,
    HIDDEN_OUT_STRIDE: tl.constexpr,
    PREVIOUS_STRIDE: tl.constexpr,
    MROPE_DIMS: tl.constexpr,
    MROPE_STRIDE: tl.constexpr,
    MASK_ID: tl.constexpr,
    MAX_MODEL_LEN: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    real = req < num_reqs
    state = tl.load(mapping + req, real, 0).to(tl.int64)
    end = tl.load(target_starts + req + 1, real, 0)
    reject = tl.load(rejected + req, real, 0)
    source = tl.where(real, end - reject - 1, 0).to(tl.int64)
    anchor = tl.load(target_positions + source, real, -1).to(tl.int64) + 1
    count = tl.load(sampled + req, real, 0)
    active = real & (count > 0) & (anchor + WIDTH <= MAX_MODEL_LEN)
    bonus = tl.load(previous + state * PREVIOUS_STRIDE, active, 0)
    offsets = tl.arange(0, QUERY_BLOCK)
    valid = offsets < WIDTH
    slot = req * WIDTH + offsets
    query = tl.where(active, anchor + offsets, offsets)
    tl.store(ids + slot, tl.where(offsets == 0, bonus, MASK_ID), valid)
    tl.store(positions + slot, query, valid)
    tl.store(padding + slot, ~active, valid)
    tl.store(starts + req, req * WIDTH)
    if req == padded_batch - 1:
        tl.store(starts + padded_batch, padded_batch * WIDTH)
    tl.store(lengths + req, tl.where(active, anchor, 0) + WIDTH)
    tl.store(active_out + req, active)
    tl.store(sources + req, source)
    tl.store(anchors + req, anchor)
    if MROPE_DIMS:
        shift = tl.load(delta + state, real, 0).to(tl.int64)
        rope = tl.where(active, query + shift, 0)
        for dim in tl.static_range(MROPE_DIMS):
            tl.store(mrope + dim * MROPE_STRIDE + slot, rope, valid)
    for first in range(0, HIDDEN, HIDDEN_BLOCK):
        col = first + tl.arange(0, HIDDEN_BLOCK)
        value = tl.load(hidden_in + source * HIDDEN_IN_STRIDE + col, real & (col < HIDDEN), 0)
        tl.store(hidden_out + req * HIDDEN_OUT_STRIDE + col, value, col < HIDDEN)


@triton.jit(do_not_specialize=["num_reqs"])
def _prepare_c8_metadata_kernel(
    output,
    table_out,
    snapshot,
    table_in,
    active_ptr,
    sources,
    anchors,
    num_reqs,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    SNAPSHOT_STRIDE: tl.constexpr,
    SNAPSHOT_COL_STRIDE: tl.constexpr,
    TABLE_IN_STRIDE: tl.constexpr,
    TABLE_IN_COL_STRIDE: tl.constexpr,
    TABLE_OUT_STRIDE: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    query_offset = tl.program_id(1)
    real = req < num_reqs
    active = tl.load(active_ptr + req).to(tl.int1)
    source = tl.load(sources + req).to(tl.int64)
    anchor = tl.load(anchors + req)
    start = anchor - WINDOW
    # Each query writes its own complete index row. No fill/cat/copy staging.
    for first in range(0, OUT_WIDTH, BLOCK):
        col = first + tl.arange(0, BLOCK)
        history = tl.load(
            snapshot + source * SNAPSHOT_STRIDE + col * SNAPSHOT_COL_STRIDE,
            active & (col < TOPK),
            -1,
        )
        history = tl.where((history >= 0) & (history < start), history, -1)
        local_col = col - TOPK
        local = tl.maximum(start + local_col, -1)
        causal_col = local_col - WINDOW
        value = tl.where(
            col < TOPK,
            history,
            tl.where(local_col < WINDOW, local, tl.where(causal_col <= query_offset, anchor + causal_col, -1)),
        )
        value = tl.where(active, value, -1)
        tl.store(output + (req * WIDTH + query_offset) * OUT_WIDTH + col, value, col < OUT_WIDTH)
    if query_offset == 0:
        for first in range(0, TABLE_COLS, BLOCK):
            col = first + tl.arange(0, BLOCK)
            value = tl.load(
                table_in + req * TABLE_IN_STRIDE + col * TABLE_IN_COL_STRIDE,
                real & (col < TABLE_COLS),
                0,
            )
            tl.store(table_out + req * TABLE_OUT_STRIDE + col, value, col < TABLE_COLS)


def try_prepare_query(spec, inputs, num_reqs, padded_batch):
    """Use the fused integer/copy preparation when all layouts are supported."""
    if not HAS_TRITON or inputs is None or not num_reqs or spec.device.type != "npu":
        return False
    buffers = spec.input_buffers
    vectors = (
        inputs.query_start_loc,
        inputs.positions,
        inputs.idx_mapping,
        inputs.num_sampled,
        inputs.num_rejected,
        buffers.input_ids,
        buffers.positions,
        buffers.is_padding,
        buffers.query_start_loc,
        buffers.seq_lens,
        spec.active_requests,
        spec.source_rows,
        spec.anchors,
    )
    if any(t.ndim != 1 or not t.is_contiguous() or t.device != buffers.input_ids.device for t in vectors):
        return False
    hidden = inputs.last_hidden_states
    previous = inputs.last_sampled
    output = spec.target_hidden_states
    if (
        hidden.ndim != 2
        or output.ndim != 2
        or previous.ndim != 2
        or hidden.shape[1] != output.shape[1]
        or hidden.stride(1) != 1
        or output.stride(1) != 1
        or hidden.dtype != output.dtype
        or any(t.device != output.device for t in (hidden, previous))
    ):
        return False
    mrope = spec.mrope_positions
    delta = spec.model_state.rope_state.prefill_delta.gpu if mrope is not None else buffers.positions
    if mrope is not None and (
        mrope.ndim != 2
        or mrope.stride(1) != 1
        or delta.ndim != 1
        or not delta.is_contiguous()
        or delta.device != output.device
    ):
        return False
    width = spec.num_query_per_req
    _prepare_query_kernel[(padded_batch,)](
        buffers.input_ids,
        buffers.positions,
        buffers.is_padding,
        buffers.query_start_loc,
        buffers.seq_lens,
        output,
        spec.active_requests,
        spec.source_rows,
        spec.anchors,
        mrope if mrope is not None else buffers.positions,
        inputs.query_start_loc,
        inputs.positions,
        inputs.idx_mapping,
        inputs.num_sampled,
        inputs.num_rejected,
        previous,
        hidden,
        delta,
        num_reqs,
        padded_batch,
        width,
        triton.next_power_of_2(width),
        output.shape[1],
        hidden.stride(0),
        output.stride(0),
        previous.stride(0),
        mrope.shape[0] if mrope is not None else 0,
        mrope.stride(0) if mrope is not None else 0,
        spec.model.config.mask_token_id,
        spec.max_model_len,
        HIDDEN_BLOCK,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return True


def try_prepare_c8_metadata(snapshot, current, table, indices, active, sources, anchors, num_reqs, batch, window):
    if not HAS_TRITON or indices.device.type != "npu":
        return False
    if (
        indices.ndim != 3
        or not indices.is_contiguous()
        or snapshot.ndim != 2
        or current.ndim != 2
        or table.ndim != 2
        or not table.is_contiguous()
        or current.shape[0] < num_reqs
        or current.shape[1] < table.shape[1]
        or any(t.device != indices.device for t in (snapshot, current, table, active, sources, anchors))
        or any(t.ndim != 1 or not t.is_contiguous() for t in (active, sources, anchors))
        or indices.shape[-1] != snapshot.shape[1] + window + indices.shape[1]
    ):
        return False
    _prepare_c8_metadata_kernel[(batch, indices.shape[1])](
        indices,
        table,
        snapshot,
        current,
        active,
        sources,
        anchors,
        num_reqs,
        indices.shape[1],
        snapshot.shape[1],
        window,
        snapshot.stride(0),
        snapshot.stride(1),
        current.stride(0),
        current.stride(1),
        table.stride(0),
        table.shape[1],
        indices.shape[-1],
        PREPARE_BLOCK,
        num_warps=1,
    )
    return True
