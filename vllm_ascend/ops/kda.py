# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared AscendC KDA execution; callers own projections and cache updates."""

from collections.abc import Sequence

import torch
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend.utils import is_950

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
    return torch.ops._C_ascend.recurrent_kda(
        q,
        k,
        v,
        raw_gate.contiguous(),
        beta.contiguous(),
        state,
        cu_seqlens,
        state_indices,
        a_log.reshape(-1).contiguous(),
        dt_bias.contiguous(),
        num_accepted_tokens=num_accepted_tokens,
        scale=q.shape[-1] ** -0.5,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=not beta_is_preprocessed,
        allow_neg_eigval=False,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound if lower_bound is not None else -5.0,
    )


def run_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor | Sequence[int],
    chunk_indices: torch.Tensor | Sequence[int] | None,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume preprocessed beta and return output plus the final VK state."""
    # The A5 Prepare/FwdH/Finalize chain accepts raw Q/K and consumes VK state
    # directly. Keep the legacy path for the other dtype/gate/empty-sequence
    # contracts; deciding from host metadata introduces no device sync.
    use_fused_norm = (
        q.dtype == torch.bfloat16
        and q.shape[-1] == v.shape[-1] == 128
        and q.shape[-2] == v.shape[-2]
        and raw_gate.dtype == torch.bfloat16
        and beta.dtype == torch.float32
        and lower_bound is not None
        and not isinstance(cu_seqlens, torch.Tensor)
        and all(end > begin for begin, end in zip(cu_seqlens, cu_seqlens[1:]))
        and is_950()
    )
    q, k = q.contiguous(), k.contiguous()
    if not use_fused_norm:
        q, k = l2norm_fwd(q), l2norm_fwd(k)
    output, final_state, *_ = torch.ops._C_ascend.chunk_kda_fwd(
        q,
        k,
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
        use_qk_l2norm_in_kernel=use_fused_norm,
    )
    return output, final_state
