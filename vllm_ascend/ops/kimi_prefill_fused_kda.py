# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CANNBot FlashKDA TND prefill ABI used by Kimi K3's fused prefill path."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

import torch
from vllm.logger import logger


@dataclass(frozen=True)
class PrefillFusedKDAOperators:
    prefill: Callable
    metadata: Callable


def load_prefill_fused_kda_operator() -> PrefillFusedKDAOperators | None:
    """Resolve both exports at worker construction, without trial execution."""
    try:
        prefill = import_module("ops.flash_kda").flash_kda
        metadata = import_module("ops.flash_kda_metadata").flash_kda_metadata
    except (ImportError, AttributeError) as error:
        logger.warning_once("CANNBot KDA prefill is unavailable; using AscendC prefill: %s", error)
        return None
    if not callable(prefill) or not callable(metadata):
        raise TypeError("ops.flash_kda.flash_kda and ops.flash_kda_metadata.flash_kda_metadata must be callable")
    return PrefillFusedKDAOperators(prefill, metadata)


def run_prefill_fused_kda(
    operator: PrefillFusedKDAOperators,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    scale: float,
    lower_bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume raw Q/K/gate/beta and return output plus final VK state.

    This is the ``ops.flash_kda`` ABI from the reference ``27.0_kda`` branch.
    The kernel performs Q/K normalization and gate/beta activation internally.
    The caller gathers initial states and scatters final states around this
    call; runner-owned page-strided caches never reach the external operator.
    """
    q, k, v, gate, beta = (tensor.contiguous() for tensor in (q, k, v, gate, beta))
    metadata = operator.metadata(q, v, initial_state, layout_qkv="TND", cu_seqlens=cu_seqlens)
    output, final_state = operator.prefill(
        q,
        k,
        v,
        g=gate,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        A_log=a_log.reshape(-1).contiguous(),
        dt_bias=dt_bias.contiguous(),
        lower_bound=lower_bound,
        layout_qkv="TND",
        metadata=metadata,
    )
    if output.shape != v.shape or output.dtype != v.dtype or output.device != v.device:
        raise RuntimeError("FlashKDA returned an unexpected output shape, dtype or device")
    if (
        final_state.shape != initial_state.shape
        or final_state.dtype != initial_state.dtype
        or final_state.device != initial_state.device
    ):
        raise RuntimeError("FlashKDA returned an unexpected final state shape, dtype or device")
    return output, final_state
