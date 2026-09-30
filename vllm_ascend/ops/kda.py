# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fla_npu KDA execution; callers own projections and cache updates."""

from collections.abc import Sequence

import torch
# KDA operators from the flashserve/flash-linear-attention-npu ecosystem repository.
# https://github.com/flashserve/flash-linear-attention-npu
from fla_npu.ops.ascendc import chunk_kda_fwd, recurrent_kda
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

KDA_CHUNK_SIZE = 64


def run_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float | None,
    beta_is_preprocessed: bool = True,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    # Recurrent KDA consumes independent Q/K/V token/head strides directly.
    output, _ = recurrent_kda(
        q,
        k,
        v,
        raw_gate.contiguous(),
        beta.contiguous(),
        state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        A_log=a_log.reshape(-1).contiguous(),
        dt_bias=dt_bias.contiguous(),
        num_accepted_tokens=num_accepted_tokens,
        layout="BSND",
        scale=q.shape[-1] ** -0.5,
        output_final_state=False,
        inplace_final_state=True,
        state_v_first=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=not beta_is_preprocessed,
        allow_neg_eigval=False,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound if lower_bound is not None else -5.0,
    )
    return output


def run_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: Sequence[int],
    chunk_indices: Sequence[int],
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume preprocessed beta and return output plus the final VK state."""
    output, final_state, *_ = chunk_kda_fwd(
        l2norm_fwd(q.contiguous()),
        l2norm_fwd(k.contiguous()),
        v.contiguous(),
        raw_gate.contiguous(),
        beta.contiguous(),
        q.shape[-1] ** -0.5,
        KDA_CHUNK_SIZE,
        layout="BSND",
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound if lower_bound is not None else -5.0,
        use_gate_in_kernel=True,
        A_log=a_log.reshape(-1).contiguous(),
        dt_bias=dt_bias.contiguous(),
        disable_recompute=False,
        return_intermediate_states=False,
        state_v_first=True,
    )
    return output, final_state
