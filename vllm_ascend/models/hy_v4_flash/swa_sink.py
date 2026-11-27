# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


# Sequence length and head strides may change with every prefill chunk.
@triton.jit(do_not_specialize=["tokens", "max_head_stride", "sum_head_stride", "context_head_stride"])
def _sink_correct_into(
    context,
    row_max,
    row_sum,
    sink,
    output,
    tokens,
    first_head: tl.constexpr,
    context_token_stride: tl.constexpr,
    context_head_stride,
    context_value_stride: tl.constexpr,
    max_head_stride,
    max_token_stride: tl.constexpr,
    sum_head_stride,
    sum_token_stride: tl.constexpr,
    sink_stride: tl.constexpr,
    output_heads: tl.constexpr,
    value_dim: tl.constexpr,
    tile_rows: tl.constexpr,
):
    token = tl.program_id(0) * tile_rows + tl.arange(0, tile_rows)
    head = tl.program_id(1)
    active = token < tokens
    channel = tl.arange(0, value_dim)
    maximum = tl.load(row_max + head * max_head_stride + token * max_token_stride, active, other=0)
    denominator = tl.load(row_sum + head * sum_head_stride + token * sum_token_stride, active, other=1)
    bias = tl.load(sink + (first_head + head) * sink_stride)
    # The learned sink has zero value and contributes only to the softmax denominator.
    weight = tl.sigmoid(maximum + tl.log(denominator) - bias)
    source = token * context_token_stride + head * context_head_stride
    value = tl.load(context + source[:, None] + channel[None, :] * context_value_stride, active[:, None], other=0).to(
        tl.float32
    )
    destination = (token * output_heads + first_head + head) * value_dim
    tl.store(output + destination[:, None] + channel[None, :], value * weight[:, None], active[:, None])


def _write_window_output(context, row_max, row_sum, sink, output, first_head):
    """Write one attention head group into a contiguous [token, head, value] output."""
    tokens, group_heads, value_dim = context.shape
    tile_rows = 16
    _sink_correct_into[(triton.cdiv(tokens, tile_rows), group_heads)](
        context,
        row_max,
        row_sum,
        sink,
        output,
        tokens,
        first_head,
        *context.stride(),
        row_max.stride(1),
        row_max.stride(2),
        row_sum.stride(1),
        row_sum.stride(2),
        sink.stride(0),
        output.size(1),
        value_dim,
        tile_rows,
        num_warps=1,
    )
