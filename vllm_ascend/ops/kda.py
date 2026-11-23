# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared KDA execution; callers own projections and cache updates."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

# KDA operators from the flashserve/flash-linear-attention-npu ecosystem repository.
# https://github.com/flashserve/flash-linear-attention-npu
from fla_npu.ops.ascendc import chunk_kda_fwd, recurrent_kda

if TYPE_CHECKING:
    from vllm_ascend.ops.gdn_attn_builder import GDNChunkedPrefillMetadata

KDA_CHUNK_SIZE = 64
FLASH_KDA_HEAD_DIM = 128
FLASH_KDA_MAX_HEADS = 96


@dataclass(frozen=True)
class KimiK31FlashKDASchedule:
    """Original schedule tensors generated for this Kimi K3.1 prefill batch."""

    cu_seqlens: torch.Tensor
    metadata: torch.Tensor
    flash_kda: Callable[..., tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]


def is_kimi_k31(model_config) -> bool:
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) == "kimi_k31" or getattr(type(hf_config), "is_kimi_k31", False)


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
    prefill_metadata: "GDNChunkedPrefillMetadata",
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float | None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume raw beta for Kimi K3.1 FlashKDA and activated beta for fla_npu."""
    # Only the Kimi K3.1 builder can attach a CANNBot schedule to this batch.
    kimi_k31_schedule = getattr(prefill_metadata, "kimi_k31_flash_kda", None)
    if kimi_k31_schedule is not None:
        out, final_state, _ = kimi_k31_schedule.flash_kda(
            q.squeeze(0).contiguous(),
            k.squeeze(0).contiguous(),
            v.squeeze(0).contiguous(),
            raw_gate.squeeze(0).contiguous(),
            beta.squeeze(0).contiguous(),
            kimi_k31_schedule.metadata,
            initial_state=initial_state.float().contiguous(),
            A_log=a_log.reshape(-1).contiguous(),
            dt_bias=dt_bias.reshape(-1, k.shape[-1]).contiguous(),
            cu_seqlens=kimi_k31_schedule.cu_seqlens,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=lower_bound,
            layout="TND",
        )
        if final_state is None:
            raise RuntimeError("CANNBot FlashKDA did not return the requested final state")
        return out.unsqueeze(0), final_state

    # FLA's import-time capability queries can create an unwanted NPU context
    # during operator discovery, before the worker selects its device.
    # TODO: Restore the top-level import once supported vLLM FLA or Triton
    # Ascend versions make these queries safe before device selection.
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

    if use_qk_l2norm_in_kernel:
        q = q.contiguous()
        k = k.contiguous()
    else:
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

    cu_seqlens = prefill_metadata.cu_seqlens_kern
    if cu_seqlens is None:
        cu_seqlens = prefill_metadata.cu_seqlens_host
    output, final_state, *_ = chunk_kda_fwd(
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
        chunk_indices=prefill_metadata.chunk_indices_chunk64_host,
        safe_gate=lower_bound is not None,
        lower_bound=lower_bound if lower_bound is not None else -5.0,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=True,
        A_log=a_log.reshape(-1).contiguous(),
        dt_bias=dt_bias.contiguous(),
        disable_recompute=False,
        return_intermediate_states=False,
        state_v_first=True,
    )
    return output, final_state
