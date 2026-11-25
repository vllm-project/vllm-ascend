# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 prefill wrapper for CANNBot KDA Prolog and FlashKDA-Dev."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

import torch
from vllm.logger import logger


@dataclass(frozen=True)
class PrefillFusedKDAOperators:
    prolog: Callable
    prolog_metadata: Callable
    prefill: Callable
    metadata: Callable


class PrefillKDAMetadataWorkspace:
    """Share one request schedule across the KDA layers in a forward step."""

    def __init__(self, device, **_kwargs):
        pass

    def new_context(self, query_start_loc, keep_meta=None):
        if keep_meta is not None:
            query_start_loc = torch.cat((query_start_loc[:1], query_start_loc[1:][keep_meta]))
        return PrefillKDAMetadataContext(query_start_loc.to(dtype=torch.int32).contiguous())


class PrefillKDAMetadataContext:
    def __init__(self, query_start_loc):
        self.query_start_loc = query_start_loc
        self.prolog_metadata = None
        self.metadata = None
        self.signature = None

    def build(self, operator, hidden_states, qkv_weight, conv_state, initial_state, cache_indices, initial_mode):
        signature = (
            tuple(
                (x.shape, x.stride(), x.dtype, x.device) for x in (hidden_states, qkv_weight, conv_state, initial_state)
            ),
            self.query_start_loc.data_ptr(),
        )
        if self.metadata is not None:
            if signature != self.signature:
                raise RuntimeError("KDA prefill metadata cannot be shared across different layouts")
            return self.metadata
        self.prolog_metadata = operator.prolog_metadata(
            hidden_states, qkv_weight, conv_state, cache_indices, self.query_start_loc, initial_mode
        )
        heads = qkv_weight.shape[1] // (3 * 128)
        stub = torch.empty((hidden_states.shape[0], heads, 128), dtype=torch.bfloat16, device=hidden_states.device)
        self.metadata = operator.metadata(stub, stub, initial_state, "TND", self.query_start_loc, template="auto")
        self.signature = signature
        return self.metadata


def create_prefill_kda_metadata_workspace(vllm_config, layer_names, device):
    layers = vllm_config.compilation_config.static_forward_context
    if not any(getattr(layers.get(name), "_prefill_fused_kda_operator", None) is not None for name in layer_names):
        return None
    return PrefillKDAMetadataWorkspace(device)


def load_prefill_fused_kda_operator() -> PrefillFusedKDAOperators | None:
    """Load the only supported Kimi K3 prefill operator chain."""
    try:
        prolog = import_module("ops.kda_prolog").kda_prolog
        prolog_metadata = import_module("ops.kda_prolog_metadata").kda_prolog_metadata
        prefill = import_module("ops.flash_kda_dev").flash_kda_dev
        metadata = import_module("ops.flash_kda_dev_metadata").flash_kda_dev_metadata
    except (ImportError, AttributeError) as error:
        logger.warning_once("CANNBot KDA Prolog prefill is unavailable: %s", error)
        return None
    if not all(callable(op) for op in (prolog, prolog_metadata, prefill, metadata)):
        raise TypeError("KDA Prolog and FlashKDA-Dev exports must be callable")
    return PrefillFusedKDAOperators(prolog, prolog_metadata, prefill, metadata)


def run_prefill_fused_kda(
    operator: PrefillFusedKDAOperators,
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    f_a_weight: torch.Tensor,
    f_b_weight: torch.Tensor,
    b_weight: torch.Tensor,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    initial_state_mode: torch.Tensor | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    scale: float,
    lower_bound: float,
    metadata_context: PrefillKDAMetadataContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse KDA prefill preprocessing and recurrent attention."""
    hidden_states = hidden_states.contiguous()
    cache_indices = cache_indices.to(dtype=torch.int32).contiguous()
    cu_seqlens = cu_seqlens.to(dtype=torch.int32).contiguous()
    if initial_state_mode is not None:
        initial_state_mode = initial_state_mode.to(dtype=torch.int32).contiguous()
    if metadata_context is not None:
        metadata = metadata_context.build(
            operator,
            hidden_states,
            qkv_weight,
            conv_state,
            initial_state,
            cache_indices,
            initial_state_mode,
        )
        prolog_metadata = metadata_context.prolog_metadata
    else:
        prolog_metadata = operator.prolog_metadata(
            hidden_states, qkv_weight, conv_state, cache_indices, cu_seqlens, initial_state_mode
        )
        stub = torch.empty(
            (hidden_states.shape[0], a_log.numel(), 128),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        metadata = operator.metadata(stub, stub, initial_state, "TND", cu_seqlens, template="auto")
    q, k, v, gate, beta, _ = operator.prolog(
        hidden_states,
        qkv_weight,
        f_a_weight,
        f_b_weight,
        b_weight,
        conv_weight,
        conv_state,
        a_log,
        dt_bias,
        prolog_metadata,
        cache_indices=cache_indices,
        cu_seqlens=cu_seqlens,
        initial_state_mode=initial_state_mode,
        lower_bound=lower_bound,
    )
    output, final_state = operator.prefill(
        q,
        k,
        v,
        gate,
        beta,
        scale,
        initial_state,
        a_log.reshape(-1).contiguous(),
        dt_bias.contiguous(),
        lower_bound,
        "TND",
        metadata,
    )
    if output.ndim == 3:
        output = output.unsqueeze(0)
    if output.shape != (1, hidden_states.shape[0], a_log.numel(), 128):
        raise RuntimeError(f"FlashKDA-Dev returned unexpected output shape {tuple(output.shape)}")
    if final_state.shape != initial_state.shape:
        raise RuntimeError("FlashKDA-Dev returned an unexpected recurrent state shape")
    return output, final_state
