# SPDX-License-Identifier: Apache-2.0
"""FP32 QK and fused sink-softmax/PV without BF16 intermediates."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _sink_softmax_pv(
    scores,
    values,
    selected,
    sink,
    output,
    heads: tl.constexpr,
    candidates: tl.constexpr,
    value_width: tl.constexpr,
    value_stride: tl.constexpr,
    selected_stride: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INVALID_MASK: tl.constexpr,
):
    row_head = tl.program_id(0)
    row = row_head // heads
    head = row_head % heads
    k = tl.arange(0, BLOCK_K)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_k = k < candidates
    ids = tl.load(selected + row * selected_stride + k, valid_k, 1 if INVALID_MASK else -1)
    valid_ids = ids == 0 if INVALID_MASK else ids >= 0
    logits = tl.load(scores + row_head * candidates + k, valid_k, 0) * scale
    logits = tl.where(valid_k & valid_ids, logits, float("-inf"))
    sink_logit = tl.load(sink + head).to(tl.float32)
    maximum = tl.maximum(tl.max(logits, 0), sink_logit)
    p = tl.exp(logits - maximum)
    denominator = tl.sum(p, 0) + tl.exp(sink_logit - maximum)
    result = tl.zeros((BLOCK_D,), tl.float32)
    for start in range(tl.cdiv(candidates, 128)):
        kk = start * 128 + tl.arange(0, 128)
        index = tl.load(selected + row * selected_stride + kk, kk < candidates, 1 if INVALID_MASK else -1)
        valid_index = index == 0 if INVALID_MASK else index >= 0
        s = tl.load(scores + row_head * candidates + kk, kk < candidates, 0) * scale
        probability = tl.exp(tl.where((kk < candidates) & valid_index, s, float("-inf")) - maximum) / denominator
        value = tl.load(
            values + row * candidates * value_stride + kk[:, None] * value_stride + d[None, :],
            ((kk < candidates) & valid_index)[:, None] & (d[None, :] < value_width),
            0,
        ).to(tl.float32)
        result += tl.sum(probability[:, None] * value, 0)
    tl.store(output + row_head * value_width + d, result, d < value_width)


def fused_fp32_mla_context(query, selected, kv, sink, scale, *, invalid_mask=None, scores=None, output_dtype=None):
    rows, heads, _ = query.shape
    candidates = selected.shape[-1]
    values = kv.contiguous()
    if scores is None:
        scores = torch.bmm(query.float(), values.transpose(1, 2))
    scores = scores.contiguous()
    indices = selected if invalid_mask is None else invalid_mask.reshape(rows, candidates)
    output = torch.empty((rows, heads, 512), device=query.device, dtype=output_dtype or query.dtype)
    if rows == 0 or candidates == 0:
        return output.zero_()
    block_d = 64
    _sink_softmax_pv[(rows * heads, triton.cdiv(512, block_d))](
        scores,
        values,
        indices,
        sink,
        output,
        heads,
        candidates,
        512,
        values.shape[-1],
        indices.stride(0),
        scale,
        triton.next_power_of_2(candidates),
        block_d,
        invalid_mask is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
