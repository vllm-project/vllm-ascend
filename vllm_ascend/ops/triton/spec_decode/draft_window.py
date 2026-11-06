# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Fused ND KV gather and FP32 conversion for ordinary Draft attention."""

from vllm.triton_utils import tl, triton

GATHER_TOKEN_BLOCK_SIZE = 8


@triton.jit
def _gather_draft_kv_fp32(
    slots,
    keys,
    values,
    out_keys,
    out_values,
    KEY_PAGE_STRIDE: tl.constexpr,
    VALUE_PAGE_STRIDE: tl.constexpr,
    BLOCK_SPAN: tl.constexpr,
    TOKEN_SIZE: tl.constexpr,
    TOKEN_PADDING: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    CHANNEL_BLOCK: tl.constexpr,
):
    # Infer allocation extent from the launch grid. Only the final tile's
    # padding specializes this kernel, not each Full-prefill capacity.
    total_tokens = tl.num_programs(0) * TOKEN_BLOCK - TOKEN_PADDING
    token = tl.program_id(0) * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    channel = tl.arange(0, CHANNEL_BLOCK)
    slot = tl.load(slots + token, mask=token < total_tokens, other=0).to(tl.int64)
    page, offset = slot // BLOCK_SPAN, slot % BLOCK_SPAN
    # Compute each physical token address once, then broadcast its channels.
    # The flat version repeated the slot load and page arithmetic per channel.
    valid = (token[:, None] < total_tokens) & (channel[None, :] < TOKEN_SIZE)
    output_offset = token[:, None] * TOKEN_SIZE + channel[None, :]
    key_offset = (page * KEY_PAGE_STRIDE + offset * TOKEN_SIZE)[:, None] + channel[None, :]
    key = tl.load(keys + key_offset, mask=valid, other=0)
    tl.store(out_keys + output_offset, key.to(tl.float32), mask=valid)
    value_offset = (page * VALUE_PAGE_STRIDE + offset * TOKEN_SIZE)[:, None] + channel[None, :]
    value = tl.load(values + value_offset, mask=valid, other=0)
    tl.store(out_values + output_offset, value.to(tl.float32), mask=valid)
