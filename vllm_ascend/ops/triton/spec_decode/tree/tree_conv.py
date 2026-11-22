# SPDX-License-Identifier: Apache-2.0
"""Triton short convolution over each verified node's ancestors.

One program owns a request and a block of channels. It loads the committed
history before writing, since the sources can be the previous verification's
node activations. It then computes each node's convolution in tree order and
stores the current activations for the next step.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton

# Channel blocks per request. The fused kernel walks a request's nodes in one
# program, so the parallel dimension is the channels.
CHANNEL_BLOCKS = 8
MIN_CHANNELS_PER_PROGRAM = 128
MAX_CHANNELS_PER_PROGRAM = 1024
# Anchored columns the kernel keeps in registers. GDN short convs are
# ``width=4`` (three committed columns).
MAX_HISTORY_COLUMNS = 4
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _channels_per_program(conv_dim: int) -> int:
    """Power-of-two channel tile that keeps ``CHANNEL_BLOCKS`` requests busy."""
    wanted = max(MIN_CHANNELS_PER_PROGRAM, triton.cdiv(conv_dim, CHANNEL_BLOCKS))
    return min(MAX_CHANNELS_PER_PROGRAM, triton.next_power_of_2(wanted))


def _num_warps(channels: int) -> int:
    """Match the warp count to the channel tile size."""
    return max(1, min(8, channels // MIN_CHANNELS_PER_PROGRAM))


def _state_layout(conv_state: torch.Tensor) -> tuple[int, int, int]:
    """Return ``(dim_stride, column_stride, columns)`` of a conv-state page.

    ``is_conv_state_dim_first()`` decides whether a page is ``[dim, width]`` or
    ``[width, dim]``; both layouts keep one of the two axes contiguous and only
    the strides differ, so the kernel takes them as arguments.
    """
    if is_conv_state_dim_first():
        return conv_state.stride(1), conv_state.stride(2), conv_state.shape[2]
    return conv_state.stride(2), conv_state.stride(1), conv_state.shape[1]


def _supports_tensor_layout(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    window_rows: torch.Tensor,
    prefix_src: torch.Tensor,
) -> bool:
    """Check the dtypes and layouts used by the Tree convolution."""
    if x.dim() != 2 or conv_state.dim() != 3:
        return False
    if x.dtype != conv_state.dtype or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if weight.dim() != 2 or window_rows.dim() != 2 or prefix_src.dim() != 2:
        return False
    tokens, conv_dim = x.shape
    if tokens == 0 or conv_dim == 0 or conv_dim != conv_state.shape[1 if is_conv_state_dim_first() else 2]:
        return False
    width = window_rows.shape[1]
    if weight.shape != (conv_dim, width) or prefix_src.shape[1] != width - 1:
        return False
    if not 1 <= width - 1 <= MAX_HISTORY_COLUMNS:
        return False
    if window_rows.shape[0] < tokens or window_rows.dtype != torch.int32 or not window_rows.is_contiguous():
        return False
    if prefix_src.dtype != torch.int32 or not prefix_src.is_contiguous():
        return False
    if query_start_loc.dim() != 1 or not query_start_loc.is_contiguous():
        return False
    rows = query_start_loc.numel() - 1
    if rows < 1 or prefix_src.shape[0] < rows:
        return False
    if cache_indices.dim() != 2 or not cache_indices.is_contiguous() or cache_indices.shape[0] < rows:
        return False
    # The reserved node columns must hold every token, else a row would lose
    # tokens to the loop bound instead of being reported.
    _, _, columns = _state_layout(conv_state)
    nodes = columns - (width - 1)
    return nodes >= 1 and tokens <= rows * nodes


@triton.jit
def _tree_conv_kernel(
    x_ptr,
    out_ptr,
    state_ptr,
    cache_ptr,
    query_start_loc_ptr,
    window_ptr,
    prefix_ptr,
    weight_ptr,
    bias_ptr,
    stride_x_row: tl.int64,
    stride_x_dim: tl.int64,
    stride_out_row: tl.int64,
    stride_out_dim: tl.int64,
    stride_page: tl.int64,
    stride_dim: tl.int64,
    stride_column: tl.int64,
    stride_weight_dim: tl.int64,
    stride_weight_column: tl.int64,
    history_rows,
    cache_last,
    token_last,
    nodes,
    conv_dim: tl.constexpr,
    width: tl.constexpr,
    has_bias: tl.constexpr,
    with_activation: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Anchor one request's history and emit every node's short-conv output.

    ``prefix_ptr`` names, per request, the source column of each committed
    column (oldest first): the activations ending at the node the sampler
    accepted last step. ``window_ptr`` names ``width`` source rows per token;
    rows below ``history_rows`` resolve to a committed column, the rest to this
    step's activations. One program owns the request's page slice, so the anchor
    reads every source before it stores and the node walk resolves its window
    from those registers.
    """
    row = tl.program_id(0)
    lane = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    channel = lane < conv_dim
    start = tl.load(query_start_loc_ptr + row).to(tl.int64)
    end = tl.load(query_start_loc_ptr + row + 1).to(tl.int64)
    # A zero-length row is a full-graph padding request: it owns no page.
    live = channel & (end > start)
    # ``cache_indices`` is packed token-major, so a request's page sits at its
    # first token. Padding rows may carry a null page; clamp for the address.
    page = tl.maximum(tl.load(cache_ptr + tl.minimum(start, cache_last)).to(tl.int64), 0)
    base = state_ptr + page * stride_page + lane.to(tl.int64) * stride_dim

    # Anchor: stage every committed column in registers, then store. Two columns
    # may share a source, so no source may be overwritten before all are read.
    prefix = prefix_ptr + row * (width - 1)
    source_0 = tl.load(prefix).to(tl.int64)
    anchor_0 = tl.load(base + source_0 * stride_column, mask=live, other=0.0)
    if width > 2:
        source_1 = tl.load(prefix + 1).to(tl.int64)
        anchor_1 = tl.load(base + source_1 * stride_column, mask=live, other=0.0)
    if width > 3:
        source_2 = tl.load(prefix + 2).to(tl.int64)
        anchor_2 = tl.load(base + source_2 * stride_column, mask=live, other=0.0)
    if width > 4:
        source_3 = tl.load(prefix + 3).to(tl.int64)
        anchor_3 = tl.load(base + source_3 * stride_column, mask=live, other=0.0)
    tl.store(base, anchor_0, mask=live)
    if width > 2:
        tl.store(base + stride_column, anchor_1, mask=live)
    if width > 3:
        tl.store(base + 2 * stride_column, anchor_2, mask=live)
    if width > 4:
        tl.store(base + 3 * stride_column, anchor_3, mask=live)

    # One weight vector per tap, reused across the nodes of this request.
    weight_0 = tl.load(weight_ptr + lane.to(tl.int64) * stride_weight_dim, mask=channel, other=0.0).to(tl.float32)
    if width > 1:
        weight_1 = tl.load(
            weight_ptr + lane.to(tl.int64) * stride_weight_dim + stride_weight_column,
            mask=channel,
            other=0.0,
        ).to(tl.float32)
    if width > 2:
        weight_2 = tl.load(
            weight_ptr + lane.to(tl.int64) * stride_weight_dim + 2 * stride_weight_column,
            mask=channel,
            other=0.0,
        ).to(tl.float32)
    if width > 3:
        weight_3 = tl.load(
            weight_ptr + lane.to(tl.int64) * stride_weight_dim + 3 * stride_weight_column,
            mask=channel,
            other=0.0,
        ).to(tl.float32)
    if has_bias:
        bias = tl.load(bias_ptr + lane, mask=channel, other=0.0).to(tl.float32)

    history_base = row * (width - 1)
    # A row cannot hold more tokens than the batch, so ``tokens`` also bounds the
    # node walk; a full-graph batch keeps the whole reserved width.
    for node in range(0, nodes):
        row_live = node < end - start
        node_live = channel & row_live
        # ``node`` walks the request's reserved node columns, so a short row
        # leaves trailing iterations without a token; they read a clamped token
        # and mask every store, which keeps the walk independent of tree width.
        token = tl.minimum(start + node, token_last)
        # Match the native kernel: FP32 products and accumulation in tap order.
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for tap in tl.static_range(width):
            source = tl.load(window_ptr + token * width + tap).to(tl.int64)
            column = source - history_base
            committed = source < history_rows
            # A committed tap is one of the anchored columns, which are still in
            # registers: select it instead of reading the page this program just
            # stored to.
            if width == 2:
                staged = anchor_0
            elif width == 3:
                staged = tl.where(column == 0, anchor_0, anchor_1)
            elif width == 4:
                staged = tl.where(column == 0, anchor_0, tl.where(column == 1, anchor_1, anchor_2))
            else:
                staged = tl.where(
                    column == 0,
                    anchor_0,
                    tl.where(column == 1, anchor_1, tl.where(column == 2, anchor_2, anchor_3)),
                )
            activation = tl.load(
                x_ptr + (source - history_rows) * stride_x_row + lane.to(tl.int64) * stride_x_dim,
                mask=node_live & (source >= history_rows),
                other=0.0,
            )
            if tap == 0:
                weight = weight_0
            elif tap == 1:
                weight = weight_1
            elif tap == 2:
                weight = weight_2
            else:
                weight = weight_3
            acc += tl.where(committed, staged, activation).to(tl.float32) * weight
        if has_bias:
            acc += bias
        if with_activation:
            acc = acc * tl.sigmoid(acc)
        tl.store(
            out_ptr + token * stride_out_row + lane.to(tl.int64) * stride_out_dim,
            acc.to(out_ptr.dtype.element_ty),
            mask=node_live,
        )
        # Keep this node's activation at column ``width - 1 + node``: the next
        # step's anchor reads it as the accepted node's history. Columns past a
        # row's nodes keep the previous step's values and are never read.
        own = tl.load(
            x_ptr + token * stride_x_row + lane.to(tl.int64) * stride_x_dim,
            mask=node_live,
            other=0.0,
        )
        tl.store(base + (width - 1 + node) * stride_column, own, mask=node_live)


def tree_causal_conv1d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    window_rows: torch.Tensor,
    prefix_src: torch.Tensor,
    bias: torch.Tensor | None,
    activation: bool,
) -> torch.Tensor:
    """Compute Tree convolution and update the request's running state in place."""
    if x.device.type != "npu" or not _supports_tensor_layout(
        x, weight, conv_state, cache_indices, query_start_loc, window_rows, prefix_src
    ):
        raise ValueError("Unsupported GDN Tree convolution layout")
    tokens, conv_dim = x.shape
    width = window_rows.shape[1]
    rows = query_start_loc.numel() - 1
    dim_stride, column_stride, columns = _state_layout(conv_state)
    channels = _channels_per_program(conv_dim)
    cache = cache_indices.reshape(-1)
    out = torch.empty((tokens, conv_dim), dtype=x.dtype, device=x.device)
    _tree_conv_kernel[(rows, triton.cdiv(conv_dim, channels))](
        x,
        out,
        conv_state,
        cache,
        query_start_loc,
        window_rows,
        prefix_src,
        weight,
        bias if bias is not None else x,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        conv_state.stride(0),
        dim_stride,
        column_stride,
        weight.stride(0),
        weight.stride(1),
        rows * (width - 1),
        # Last addressable entry of the packed page table and of the token
        # tables: padding rows still evaluate an address but store nothing.
        cache.numel() - 1,
        tokens - 1,
        # Reserved node columns, bounded by the token count (a row cannot hold
        # more tokens than the whole batch).
        min(columns - (width - 1), tokens),
        conv_dim=conv_dim,
        width=width,
        has_bias=bias is not None,
        with_activation=bool(activation),
        BLOCK=channels,
        num_warps=_num_warps(channels),
    )
    return out
