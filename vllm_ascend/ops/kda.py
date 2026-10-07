# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fla_npu KDA execution; callers own projections and cache updates."""

from collections.abc import Sequence

import torch

# KDA operators from the flashserve/flash-linear-attention-npu ecosystem repository.
# https://github.com/flashserve/flash-linear-attention-npu
from fla_npu.ops.ascendc import chunk_kda_fwd, recurrent_kda
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend.device.device_config import is_950

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


def _pad_kda_sequence_tensor(
    tensor: torch.Tensor,
    cu_seqlens: Sequence[int],
    padded_cu_seqlens: Sequence[int],
    fill_value: float,
) -> torch.Tensor:
    padded = tensor.new_full((tensor.shape[0], padded_cu_seqlens[-1], *tensor.shape[2:]), fill_value)
    for start, end, padded_start in zip(cu_seqlens[:-1], cu_seqlens[1:], padded_cu_seqlens[:-1]):
        padded[:, padded_start : padded_start + end - start].copy_(tensor[:, start:end])
    return padded


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
    original_tokens = q.shape[1]
    original_cu_seqlens = None
    if is_950() and any((end - start) % KDA_CHUNK_SIZE for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])):
        # The A5 FLA tail output can be nondeterministic. Give every sequence
        # complete chunks without changing the recurrence: zero Q/K/V and beta
        # make no update, while raw_gate=-inf gives log decay zero for both the
        # bounded sigmoid and the unbounded softplus gate.
        padded_cu = [0]
        padded_chunks = []
        for sequence, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
            num_chunks = (end - start + KDA_CHUNK_SIZE - 1) // KDA_CHUNK_SIZE
            padded_cu.append(padded_cu[-1] + num_chunks * KDA_CHUNK_SIZE)
            for chunk in range(num_chunks):
                padded_chunks.extend((sequence, chunk))
        q, k, v, raw_gate, beta = (
            _pad_kda_sequence_tensor(tensor, cu_seqlens, padded_cu, fill_value)
            for tensor, fill_value in ((q, 0.0), (k, 0.0), (v, 0.0), (raw_gate, float("-inf")), (beta, 0.0))
        )
        original_cu_seqlens = cu_seqlens
        cu_seqlens = tuple(padded_cu)
        chunk_indices = tuple(padded_chunks)

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
    if original_cu_seqlens is not None:
        # Keep any physical token padding outside the host descriptors defined.
        unpadded = output.new_zeros((output.shape[0], original_tokens, *output.shape[2:]))
        for start, end, padded_start in zip(original_cu_seqlens[:-1], original_cu_seqlens[1:], cu_seqlens[:-1]):
            unpadded[:, start:end].copy_(output[:, padded_start : padded_start + end - start])
        output = unpadded
    return output, final_state
