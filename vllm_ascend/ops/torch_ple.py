# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""PyTorch small-operator implementation of Qwen4Exp PLE convolution."""

from __future__ import annotations

from typing import Literal

import torch
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

_TOKEN_CHUNK = 1024


def ple_conv_torch(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    outer_residual: torch.Tensor | None,
    *,
    mode: Literal["decode", "spec", "prefill"],
    dilation: int,
    query_start_loc: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    has_initial_states: torch.Tensor | None = None,
    spec_query_len: int = 1,
    token_indices: torch.Tensor | None = None,
) -> None:
    """Apply independent channel convolutions and update the PLE state bank.

    Metadata stays on device. Token chunks bound the FP32 accumulator size on
    long prefill requests. The state bank is updated after all output reads.
    """
    if mode not in ("decode", "spec", "prefill"):
        raise ValueError(f"Unsupported short-conv mode: {mode}")
    tokens = token_indices.numel() if token_indices is not None else inputs.shape[0]
    channels, kernel_size = conv_weights.shape
    state_len = (kernel_size - 1) * dilation
    spec_len = spec_query_len if mode == "spec" else 1
    state_width = state_len + spec_len - 1
    # Match the upstream ple_conv state layout contract before clamping indices.
    if conv_state.shape[1] != channels or conv_state.shape[2] < state_width:
        raise ValueError(
            "conv_state must have shape [slots, channels, window], with "
            f"channels={channels} and window >= {state_width}"
        )
    if mode != "decode" and query_start_loc is None:
        raise ValueError("query_start_loc is required for prefill and spec")
    if mode == "spec" and num_accepted_tokens is None:
        raise ValueError("num_accepted_tokens is required for spec")
    if mode == "prefill" and has_initial_states is None:
        raise ValueError("has_initial_states is required for prefill")
    if not tokens:
        return

    device = inputs.device
    weights = conv_weights.float()
    num_reqs = tokens if mode == "decode" else state_indices.numel()
    old_state = conv_state
    logical = torch.arange(tokens, device=device, dtype=torch.int64)
    if mode == "decode":
        request = logical
        q_start = logical
        position = torch.zeros_like(logical)
        in_range = torch.ones_like(logical, dtype=torch.bool)
    else:
        assert query_start_loc is not None
        request = torch.searchsorted(query_start_loc[1:], logical, right=True)
        request = request.clamp(max=num_reqs - 1)
        q_start = query_start_loc[request].to(torch.int64)
        position = logical - q_start
        in_range = logical < query_start_loc[num_reqs]

    slot = state_indices[request].to(torch.int64)
    state_ok = slot != NULL_BLOCK_ID
    safe_slot = slot.clamp(min=0)
    if mode == "spec":
        assert num_accepted_tokens is not None
        slot_offset = (num_accepted_tokens[request].to(torch.int64) - 1).clamp(0, spec_len - 1)
        read_state = state_ok
        output_ok = in_range
    else:
        slot_offset = torch.zeros_like(logical)
        initialized = (
            has_initial_states[request].bool() if has_initial_states is not None else torch.ones_like(state_ok)
        )
        read_state = state_ok & initialized
        output_ok = in_range & state_ok
    physical = token_indices[logical].to(torch.int64) if token_indices is not None else logical
    tap_offsets = torch.arange(kernel_size, device=device, dtype=torch.int64) * dilation
    history_pos = position[:, None] + tap_offsets[None, :]
    from_state = history_pos < state_len
    state_pos = (history_pos + slot_offset[:, None]).clamp(0, conv_state.shape[2] - 1)
    input_pos = (q_start[:, None] + history_pos - state_len).clamp(0, inputs.shape[0] - 1)
    if token_indices is not None:
        input_pos = token_indices[input_pos].to(torch.int64)
    state_by_time = old_state.permute(0, 2, 1)

    for chunk_start in range(0, tokens, _TOKEN_CHUNK):
        chunk_end = min(chunk_start + _TOKEN_CHUNK, tokens)
        chunk = slice(chunk_start, chunk_end)
        state_value = state_by_time[safe_slot[chunk, None], state_pos[chunk]]
        input_value = inputs[input_pos[chunk]]
        history_value = torch.where(read_state[chunk, None, None], state_value, 0)
        values = torch.where(from_state[chunk, :, None], history_value, input_value)
        values = torch.where(output_ok[chunk, None, None], values, 0)
        acc = torch.zeros((chunk_end - chunk_start, channels), device=device, dtype=torch.float32)
        for tap in range(kernel_size):
            acc.add_(values[:, tap].float() * weights[:, tap])

        conv = acc.to(residual.dtype).float()
        activated = (conv * torch.sigmoid(conv)).to(residual.dtype)
        activated = torch.where(output_ok[chunk, None], activated, 0)
        output_rows = physical[chunk]
        if outer_residual is None:
            result = activated
        elif residual.dtype == torch.bfloat16:
            # BF16 Triton-Ascend keeps the first residual sum in FP32.
            result = (residual[output_rows].float() + activated.float() + outer_residual[output_rows].float()).to(
                residual.dtype
            )
        else:
            first_sum = (residual[output_rows] + activated).to(residual.dtype)
            result = (first_sum.float() + outer_residual[output_rows].float()).to(residual.dtype)
        residual.index_copy_(0, output_rows, result)

    # Gather the old state once for all requests, then write it back at the end.
    request = torch.arange(num_reqs, device=device, dtype=torch.int64)
    slot = state_indices[:num_reqs].to(torch.int64)
    state_ok = slot != NULL_BLOCK_ID
    safe_slot = slot.clamp(min=0)
    previous = old_state.index_select(0, safe_slot)
    next_state = previous.clone()
    if mode == "decode":
        q_start = request
        length = torch.ones_like(request)
        shift = length
        slot_offset = torch.zeros_like(request)
    else:
        assert query_start_loc is not None
        q_start = query_start_loc[:-1].to(torch.int64)
        length = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int64)
        shift = torch.ones_like(length) if mode == "spec" else length
        if mode == "spec":
            assert num_accepted_tokens is not None
            slot_offset = (num_accepted_tokens.to(torch.int64) - 1).clamp(0, spec_len - 1)
        else:
            slot_offset = torch.zeros_like(length)
    initialized = (
        has_initial_states[:num_reqs].bool()
        if has_initial_states is not None and mode != "spec"
        else torch.ones_like(state_ok)
    )
    index = torch.arange(state_width, device=device)
    history_pos = shift[:, None] + index[None, :]
    from_state = history_pos < state_len
    state_pos = (history_pos + slot_offset[:, None]).clamp(0, conv_state.shape[2] - 1)
    old_value = previous.permute(0, 2, 1)[request[:, None], state_pos]
    old_value = torch.where(initialized[:, None, None], old_value, 0)
    input_pos = (q_start[:, None] + history_pos - state_len).clamp(0, inputs.shape[0] - 1)
    if token_indices is not None:
        input_pos = token_indices[input_pos].to(torch.int64)
    input_value = inputs[input_pos]
    value = torch.where(from_state[:, :, None], old_value, input_value)
    write_ok = state_ok[:, None]
    if mode == "prefill":
        write_ok = write_ok & (length[:, None] > 0)
    elif mode == "spec":
        write_ok = write_ok & (index[None, :] < state_len + length[:, None] - 1)
    next_state[:, :, :state_width] = torch.where(
        write_ok[:, :, None], value, previous[:, :, :state_width].permute(0, 2, 1)
    ).permute(0, 2, 1)
    if state_ok.any():
        conv_state.index_copy_(0, safe_slot[state_ok], next_state[state_ok])
