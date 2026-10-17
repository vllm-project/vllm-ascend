# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM bounded-gate contracts for the AscendC KDA operators."""

import torch

from vllm_ascend.models.glm5next.ops.state_ops import gather_initial_states, scatter_states
from vllm_ascend.ops.kda import KDA_CHUNK_SIZE, run_chunk_kda, run_recurrent_kda
from vllm_ascend.ops.triton.kda.output_writeback import write_recurrent_output

KDA_MAX_RECURRENT_TOKENS = 8


def recurrent_kda(
    q,
    k,
    v,
    raw_gate,
    raw_beta,
    state,
    cu_seqlens,
    state_indices,
    a_log,
    dt_bias,
    lower_bound,
    num_accepted_tokens=None,
    output_buffer=None,
):
    """Update the selected VK state slots, including MTP rejection rollback."""
    num_seqs = cu_seqlens.numel() - 1
    state_indices = state_indices[:num_seqs]
    if num_accepted_tokens is not None:
        num_accepted_tokens = num_accepted_tokens[:num_seqs]
    output = run_recurrent_kda(
        q,
        k,
        v,
        raw_gate,
        raw_beta,
        state,
        cu_seqlens,
        state_indices,
        a_log,
        dt_bias,
        lower_bound=lower_bound,
        beta_is_preprocessed=False,
        num_accepted_tokens=num_accepted_tokens,
    )
    if output_buffer is not None:
        write_recurrent_output(output, output_buffer, cu_seqlens)
        return output_buffer
    valid = torch.arange(q.shape[1], device=q.device) < cu_seqlens[-1]
    return output.masked_fill(~valid[None, :, None, None], 0)


def chunk_kda(
    q,
    k,
    v,
    raw_gate,
    raw_beta,
    state,
    state_indices,
    has_initial_state,
    metadata,
    a_log,
    dt_bias,
    lower_bound,
):
    """Run prefill with CPU chunk descriptors prepared by the GDN builder."""
    if metadata.keep_meta is not None:
        state_indices = state_indices[metadata.keep_meta]
        has_initial_state = has_initial_state[metadata.keep_meta]
    initial_state = gather_initial_states(state, state_indices, has_initial_state).float().contiguous()
    cu_seqlens = metadata.cu_seqlens_host if metadata.cu_seqlens_kern is None else metadata.cu_seqlens_kern
    chunk_indices = metadata.chunk_indices_chunk64_host
    beta = raw_beta.float().sigmoid()
    lengths = [end - start for start, end in zip(cu_seqlens, cu_seqlens[1:])]
    padded_cu_seqlens = None
    if any(length % KDA_CHUNK_SIZE for length in lengths):
        # FLA's partial chunks can produce incorrect outputs. Neutral tokens
        # complete each chunk without updating or decaying its final state.
        parts: list[list[torch.Tensor]] = [[] for _ in range(5)]
        padded_cu_seqlens = [0]
        chunk_indices = []
        for sequence, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
            length = end - start
            padding = -length % KDA_CHUNK_SIZE
            padded_cu_seqlens.append(padded_cu_seqlens[-1] + length + padding)
            chunk_indices.extend(
                index for chunk in range((length + padding) // KDA_CHUNK_SIZE) for index in (sequence, chunk)
            )
            for index, tensor in enumerate((q, k, v, raw_gate, beta)):
                part = tensor[:, start:end]
                if padding:
                    shape = (tensor.shape[0], padding, *tensor.shape[2:])
                    dummy = tensor.new_full(shape, float("-inf") if index == 3 else 0)
                    part = torch.cat((part, dummy), dim=1)
                parts[index].append(part)
        q, k, v, raw_gate, beta = (torch.cat(part, dim=1) for part in parts)
        cu_seqlens = padded_cu_seqlens
    output, final_state = run_chunk_kda(
        q,
        k,
        v,
        raw_gate,
        beta,
        initial_state,
        cu_seqlens,
        chunk_indices,
        a_log,
        dt_bias,
        lower_bound=lower_bound,
    )
    if padded_cu_seqlens is not None:
        output = torch.cat(
            [output[:, start : start + length] for start, length in zip(padded_cu_seqlens, lengths)], dim=1
        )
    scatter_states(state, final_state.to(state.dtype), state_indices)
    return output
