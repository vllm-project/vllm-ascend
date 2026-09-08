# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend launch policy for the upstream stride-aware recurrent GDN kernel."""

import torch
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd_kernel,
)
from vllm.triton_utils import triton

_MAX_VALUES_PER_PROGRAM = 32
_RECURRENT_NUM_WARPS = 1
# On A3, the upstream three-stage launch can corrupt intermediate checkpoints
# while its generated outputs and final state remain correct. One stage passed
# repeated dense/padded-state checks without changing the math or tolerances.
_RECURRENT_NUM_STAGES = 1


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference with the upstream state indexing and a safe pipeline depth.

    The caller supplies a state with dense inner dimensions and an optionally
    padded leading stride. Keep that view intact when normalizing Q/K/V inputs.
    """
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("Variable-length recurrent GDN requires flattened batch size 1.")
    if scale <= 0:
        raise ValueError("Recurrent GDN scale must be positive.")
    q, k, v, g, beta = (tensor.contiguous() for tensor in (q, k, v, g, beta))
    batch_size, total_tokens, num_key_heads, key_dim = k.shape
    num_value_heads, value_dim = v.shape[2:]
    num_sequences = batch_size if cu_seqlens is None else len(cu_seqlens) - 1
    block_key = triton.next_power_of_2(key_dim)
    block_value = min(triton.next_power_of_2(value_dim), _MAX_VALUES_PER_PROGRAM)
    num_value_blocks = triton.cdiv(value_dim, block_value)

    output = q.new_empty(1, *v.shape)
    final_state = (
        initial_state
        if inplace_final_state
        else q.new_empty(total_tokens, num_value_heads, value_dim, key_dim, dtype=initial_state.dtype)
    )
    if ssm_state_indices is None:
        indices_sequence_stride, indices_token_stride = 1, 1
    elif ssm_state_indices.ndim == 1:
        indices_sequence_stride, indices_token_stride = ssm_state_indices.stride(0), 1
    else:
        indices_sequence_stride, indices_token_stride = ssm_state_indices.stride()

    fused_recurrent_gated_delta_rule_fwd_kernel[(1, num_value_blocks, num_sequences * num_value_heads)](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=output,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=num_sequences,
        T=total_tokens,
        B=batch_size,
        H=num_key_heads,
        HV=num_value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_key,
        BV=block_value,
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=final_state.stride(0),
        stride_indices_seq=indices_sequence_stride,
        stride_indices_tok=indices_token_stride,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=False,
        num_warps=_RECURRENT_NUM_WARPS,
        num_stages=_RECURRENT_NUM_STAGES,
    )
    return output.squeeze(0), final_state
