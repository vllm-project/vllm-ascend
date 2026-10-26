# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Materialize a cached pure GDN decode layout into captured input buffers."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["actual_rows", "graph_rows", "token_count"])
def _cached_decode_kernel(
    state_source,
    query_source,
    accepted_source,
    token_source,
    state_output,
    query_output,
    accepted_output,
    token_output,
    mask_output,
    lengths_output,
    actual_rows,
    graph_rows,
    token_count,
    STATE_ROW_STRIDE: tl.constexpr,
    STATE_COL_STRIDE: tl.constexpr,
    QUERY_STRIDE: tl.constexpr,
    ACCEPTED_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program owns each small allocation, including its 32-byte tail.
    slots = tl.arange(0, BLOCK_SLOTS)
    for tile in range(tl.cdiv(graph_rows, BLOCK)):
        rows = tile * BLOCK + tl.arange(0, BLOCK)
        live = rows < actual_rows
        states = tl.load(
            state_source + rows[:, None].to(tl.int64) * STATE_ROW_STRIDE + slots[None, :] * STATE_COL_STRIDE,
            live[:, None] & (slots[None, :] < WIDTH),
            other=0,
        )
        tl.store(
            state_output + rows[:, None] * WIDTH + slots[None, :],
            states,
            (rows[:, None] < graph_rows) & (slots[None, :] < WIDTH),
        )
        accepted = tl.load(accepted_source + rows * ACCEPTED_STRIDE, live, other=1)
        tl.store(accepted_output + rows, accepted, rows < graph_rows)
        tl.store(mask_output + rows, live, rows < graph_rows)
    for tile in range(tl.cdiv(graph_rows + 1, BLOCK)):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        valid = offsets <= graph_rows
        source_offsets = tl.minimum(offsets, actual_rows)
        query = tl.load(query_source + source_offsets * QUERY_STRIDE, valid, other=0)
        previous_offsets = tl.minimum(tl.maximum(offsets - 1, 0), actual_rows)
        previous = tl.load(query_source + previous_offsets * QUERY_STRIDE, valid, other=0)
        lengths = tl.where(offsets == 0, query, query - previous)
        tl.store(query_output + offsets, query, valid)
        tl.store(lengths_output + offsets, lengths, valid)
    for tile in range(tl.cdiv(token_count, BLOCK)):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(token_source + offsets, offsets < token_count, other=0)
        tl.store(token_output + offsets, values, offsets < token_count)


def materialize_cached_gdn_decode(
    state_source: torch.Tensor,
    query_source: torch.Tensor,
    accepted_source: torch.Tensor,
    token_source: torch.Tensor,
    state_output: torch.Tensor,
    query_output: torch.Tensor,
    accepted_output: torch.Tensor,
    token_output: torch.Tensor,
    mask_output: torch.Tensor,
    lengths_output: torch.Tensor,
    actual_rows: int,
    graph_rows: int,
) -> None:
    """Refresh current physical states, accepted counts and query boundaries."""
    width = state_output.shape[1]
    outputs = (
        state_output,
        query_output,
        accepted_output,
        token_output,
        lengths_output,
    )
    assert all(t.device == state_output.device and t.dtype == torch.int32 and t.is_contiguous() for t in outputs)
    assert mask_output.device == state_output.device and mask_output.dtype == torch.bool and mask_output.is_contiguous()
    assert all(
        t.device == state_output.device and t.dtype == torch.int32
        for t in (state_source, query_source, accepted_source, token_source)
    )
    assert state_source.ndim == 2 and state_source.shape[1] >= width
    assert query_source.ndim == accepted_source.ndim == token_source.ndim == 1 and token_source.is_contiguous()
    assert 0 <= actual_rows <= graph_rows <= min(state_output.shape[0], accepted_output.numel(), mask_output.numel())
    assert actual_rows <= min(state_source.shape[0], accepted_source.numel())
    assert actual_rows + 1 <= query_source.numel()
    assert graph_rows + 1 <= min(query_output.numel(), lengths_output.numel())
    assert token_source.numel() <= token_output.numel()
    assert state_output.device.type == "npu"
    _cached_decode_kernel[(1,)](
        state_source,
        query_source,
        accepted_source,
        token_source,
        state_output,
        query_output,
        accepted_output,
        token_output,
        mask_output,
        lengths_output,
        actual_rows,
        graph_rows,
        token_source.numel(),
        STATE_ROW_STRIDE=state_source.stride(0),
        STATE_COL_STRIDE=state_source.stride(1),
        QUERY_STRIDE=query_source.stride(0),
        ACCEPTED_STRIDE=accepted_source.stride(0),
        WIDTH=width,
        BLOCK_SLOTS=triton.next_power_of_2(width),
        BLOCK=32,
        multibuffer=False,
        unit_flag=False,
    )
