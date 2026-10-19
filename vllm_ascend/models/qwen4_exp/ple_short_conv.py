# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused upstream PLE adaptation for normal multi-token decode chunks."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.models.qwen4_exp.short_conv_attn import PleShortConvAttentionMetadata


def _short_conv_dilated_prefill_batched(
    self: Any,
    x_p: torch.Tensor,
    metadata: PleShortConvAttentionMetadata,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices_tensor_p: torch.Tensor,
    num_prefills: int,
    num_decode_tokens: int,
    num_prefill_tokens: int,
) -> torch.Tensor:
    # The upstream metadata already carries offsets relative to the prefill
    # subgroup, including a one-token prefill beside ordinary decode rows.
    del num_decode_tokens
    query_start_loc_p = metadata.query_start_loc_p
    if query_start_loc_p is None:
        raise ValueError("query_start_loc_p is required for prefill short-conv")
    has_initial_states_p = metadata.has_initial_states_p
    if has_initial_states_p is None:
        raise ValueError("has_initial_states_p is required for prefill short-conv")
    return self._short_conv_dilated_chunk_batched(
        x_p=x_p,
        conv_state=conv_state,
        conv_weights=conv_weights,
        query_start_loc_p=query_start_loc_p,
        state_indices_tensor_p=state_indices_tensor_p,
        has_initial_states_p=has_initial_states_p,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        max_len=metadata.max_prefill_query_len,
    )


def _short_conv_dilated_chunk_batched(
    self: Any,
    x_p: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    query_start_loc_p: torch.Tensor,
    state_indices_tensor_p: torch.Tensor,
    has_initial_states_p: torch.Tensor,
    num_prefills: int,
    num_prefill_tokens: int,
    max_len: int,
) -> torch.Tensor:
    """Commit every prefill/replay token while retaining causal conv history.

    Counts and packing width come from CPU metadata. Unlike the speculative
    path, a normal replay chunk never rolls history back by accepted tokens.
    """
    output = torch.empty_like(x_p)
    q_starts = query_start_loc_p.to(torch.int64)
    if state_indices_tensor_p.numel() < num_prefills:
        raise ValueError(
            f"state_indices_tensor_p size mismatch: got {state_indices_tensor_p.numel()}, need >= {num_prefills}."
        )
    if has_initial_states_p.numel() < num_prefills:
        raise ValueError(
            f"has_initial_states_p size mismatch: got {has_initial_states_p.numel()}, need >= {num_prefills}."
        )
    if num_prefills == 0 or x_p.numel() == 0:
        return output
    lengths = q_starts[1:] - q_starts[:-1]
    # Use the CPU-computed packing width from the metadata builder instead
    # of synchronizing on lengths.max().
    if max_len <= 0:
        return output

    hidden_size = x_p.shape[1]
    positions = torch.arange(num_prefill_tokens, device=x_p.device, dtype=torch.int64)
    req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    col_indices = positions - q_starts[req_indices]

    packed_tokens = x_p.new_zeros((num_prefills, max_len, hidden_size))
    packed_tokens[req_indices, col_indices] = x_p
    packed_tokens = packed_tokens.transpose(1, 2).contiguous()

    state_indices = state_indices_tensor_p[:num_prefills].to(device=conv_state.device, dtype=torch.int64)
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(valid_state, state_indices, torch.zeros_like(state_indices))
    has_initial = has_initial_states_p[:num_prefills].to(device=conv_state.device, dtype=torch.bool)
    if self.conv_state_len > 0:
        if conv_state.shape[0] == 0:
            state = conv_state.new_zeros(
                (num_prefills, hidden_size, self.conv_state_len),
                dtype=x_p.dtype,
            )
        else:
            state = conv_state.index_select(0, state_indices)[..., : self.conv_state_len].to(x_p.dtype)
        use_initial_mask = (valid_state & has_initial).view(num_prefills, 1, 1)
        initial_state = torch.where(
            use_initial_mask,
            state,
            torch.zeros_like(state),
        )
        history = torch.cat((initial_state, packed_tokens), dim=-1)
    else:
        history = packed_tokens

    conv_output = F.conv1d(
        history,
        conv_weights.unsqueeze(1).contiguous(),
        groups=history.size(1),
        dilation=self.short_conv_dilation,
    )
    conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

    token_positions = torch.arange(max_len, device=x_p.device, dtype=torch.int64)
    valid_tokens = token_positions.view(1, max_len) < lengths.view(num_prefills, 1)
    valid_output_mask = valid_tokens & valid_state.to(device=x_p.device).view(num_prefills, 1)
    conv_output.masked_fill_(~valid_output_mask.unsqueeze(-1), 0)
    output.copy_(conv_output[req_indices, col_indices])

    if self.conv_state_len > 0 and conv_state.shape[0] > 0:
        state_starts = lengths.to(device=history.device, dtype=torch.int64).view(num_prefills, 1, 1)
        state_offsets = torch.arange(self.conv_state_len, device=history.device, dtype=torch.int64).view(
            1, 1, self.conv_state_len
        )
        next_state = history.gather(
            dim=2,
            index=(state_starts + state_offsets).expand(-1, history.size(1), -1),
        )
        # Write back without a host synchronization. Valid, non-empty rows
        # receive their new state; padding and zero-length rows keep the
        # current cache value.
        existing_state = conv_state.index_select(0, state_indices)
        existing_base_state = existing_state[..., : self.conv_state_len]
        update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
        safe_next_state = torch.where(
            update_mask.view(num_prefills, 1, 1),
            next_state.to(conv_state.dtype),
            existing_base_state,
        )
        existing_state[..., : self.conv_state_len] = safe_next_state
        conv_state.index_copy_(0, state_indices, existing_state)
    return output


def _wrap_ple_short_conv_dispatch(original: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    def dispatch(
        self: Any,
        inputs: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> torch.Tensor:
        chunk = getattr(metadata, "normal_chunk", None)
        if chunk is not None:
            x = inputs[: metadata.num_actual_tokens]
            chunk_output = self._short_conv_dilated_chunk_batched(
                x_p=x.index_select(0, chunk.token_indices),
                conv_state=conv_state,
                conv_weights=conv_weights,
                query_start_loc_p=chunk.query_start_loc,
                state_indices_tensor_p=chunk.state_indices_tensor,
                has_initial_states_p=chunk.has_initial_states,
                num_prefills=chunk.num_reqs,
                num_prefill_tokens=chunk.num_tokens,
                max_len=chunk.max_query_len,
            )
            output = torch.empty_like(x)
            output.index_copy_(0, chunk.token_indices, chunk_output)
            if metadata.remaining_metadata is not None:
                indices = metadata.remaining_token_indices
                assert indices is not None
                remaining_output = dispatch(
                    self,
                    x.index_select(0, indices),
                    metadata.remaining_metadata,
                    conv_state,
                    conv_weights,
                )
                output.index_copy_(0, indices, remaining_output)
            return output

        # A one-token prefill can appear before a regular decode request.
        # Preserve the upstream decode/prefill execution order internally and
        # scatter the result back to its original token positions afterwards.
        indices = metadata.non_spec_token_indx
        if metadata.spec_sequence_masks is None and indices is not None:
            x = inputs[: metadata.num_actual_tokens]
            result = original(self, x.index_select(0, indices), metadata, conv_state, conv_weights)
            output = torch.zeros_like(x)
            output.index_copy_(0, indices, result)
            return output
        return original(self, inputs, metadata, conv_state, conv_weights)

    return dispatch


def patch_ple_chunk_execution(layer_cls: type) -> None:
    original_attr = "_ascend_original_short_conv_dispatch"
    if hasattr(layer_cls, original_attr):
        return
    original = layer_cls._short_conv_dilated_dispatch
    setattr(layer_cls, original_attr, original)
    layer_cls._short_conv_dilated_chunk_batched = _short_conv_dilated_chunk_batched
    layer_cls._short_conv_dilated_prefill_batched = _short_conv_dilated_prefill_batched
    layer_cls._short_conv_dilated_dispatch = _wrap_ple_short_conv_dispatch(original)
