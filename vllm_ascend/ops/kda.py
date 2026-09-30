# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared AscendC KDA execution; callers own projections and cache updates."""

from collections.abc import Sequence

import torch
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

KDA_CHUNK_SIZE = 64
KDA_FUSED_NORM_HEAD_DIM = 128
KDA_SUPPORTED_HEAD_DIMS = (64, 128)


def _host_descriptor(values: torch.Tensor | Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError(f"{name} must be prepared on CPU before KDA prefill")
        values = values.reshape(-1).tolist()
    return tuple(values)


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
    chunk_indices: torch.Tensor | Sequence[int],
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume preprocessed beta and return output plus the final VK state.

    BF16 heads of size 128 use the V2 kernel's Q/K normalization. Other
    supported inputs keep the separate normalization and fused L0 path.
    Metadata stays on the host to avoid synchronizing the NPU per layer.
    """
    if q.shape[-1] != v.shape[-1] or q.shape[-1] not in KDA_SUPPORTED_HEAD_DIMS:
        raise ValueError("chunk_kda_fwd requires K=V=64 or K=V=128")
    cu_seqlens = _host_descriptor(cu_seqlens, "cu_seqlens")
    chunk_indices = _host_descriptor(chunk_indices, "chunk_indices")
    use_fused_norm = (
        q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.shape[-1] == KDA_FUSED_NORM_HEAD_DIM
        and all(begin < end for begin, end in zip(cu_seqlens, cu_seqlens[1:]))
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
        use_beta_sigmoid_in_kernel=False,
    )
    return output, final_state
