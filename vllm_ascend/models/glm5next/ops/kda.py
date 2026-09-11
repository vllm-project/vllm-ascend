# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM bounded-gate contracts for the AscendC KDA operators."""

import torch

from vllm_ascend.models.glm5next.ops.state_ops import gather_initial_states, scatter_states
from vllm_ascend.ops.kda import run_chunk_kda, run_recurrent_kda

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
    output, final_state = run_chunk_kda(
        q,
        k,
        v,
        raw_gate,
        raw_beta.float().sigmoid(),
        initial_state,
        cu_seqlens,
        metadata.chunk_indices_chunk64_host,
        a_log,
        dt_bias,
        lower_bound=lower_bound,
    )
    scatter_states(state, final_state.to(state.dtype), state_indices)
    return output
