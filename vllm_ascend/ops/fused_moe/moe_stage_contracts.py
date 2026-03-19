from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import torch

from vllm_ascend.ops.fused_moe.moe_stage_params import (
    MoEMlpKernelParams,
    MoEMlpParams,
    MoEQuantParams,
    MoERoutingParams,
)

TMoERoutingMetadata = TypeVar("TMoERoutingMetadata")


# prepare -> fused_experts
@dataclass(frozen=True, slots=True)
class MoEPrepareOutput:
    """Typed output from prepare stage."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    mc2_mask: torch.Tensor | None
    padded_hidden_states_shape: torch.Size | None
    pertoken_scale: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MoEWeights:
    """Dense and quantized weight payloads consumed by MoE execution."""

    w1: torch.Tensor | list[torch.Tensor]
    w2: torch.Tensor | list[torch.Tensor]
    w1_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None
    w1_scale: torch.Tensor | list[torch.Tensor] | None = None
    w2_scale: torch.Tensor | list[torch.Tensor] | None = None
    w1_scale_bias: torch.Tensor | None = None
    w2_scale_bias: torch.Tensor | None = None
    w1_offset: torch.Tensor | None = None
    w2_offset: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MoEFusedExpertsInput:
    """Top-level input for the routed experts pipeline."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    weights: MoEWeights
    routing: MoERoutingParams
    mlp: MoEMlpParams
    quant: MoEQuantParams


@dataclass(frozen=True, slots=True)
class MoETokenDispatchInput:
    """Input to token dispatch."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    routing: MoERoutingParams
    quant: MoEQuantParams


# dispatch carry-over metadata for combine
@dataclass(frozen=True, slots=True)
class MoEMC2RoutingMetadata:
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    expert_map: torch.Tensor | None
    ep_recv_counts: torch.Tensor
    tp_recv_counts: torch.Tensor
    assist_info_for_combine: torch.Tensor
    expand_scales: torch.Tensor | None
    dispatch_with_quant: bool


@dataclass(frozen=True, slots=True)
class MoEAllGatherRoutingMetadata:
    topk_weights: torch.Tensor
    expanded_row_idx: torch.Tensor
    restore_shape: torch.Size


@dataclass(frozen=True, slots=True)
class MoEAllToAllRoutingMetadata:
    input_splits: np.ndarray
    output_splits: np.ndarray
    topk_weights: torch.Tensor
    reversed_local_input_permutation_mapping: torch.Tensor
    reversed_global_input_permutation_mapping: torch.Tensor | None
    hidden_shape: torch.Size
    hidden_shape_before_permute: torch.Size


@dataclass(frozen=True, slots=True)
class MoETokenDispatchOutput(Generic[TMoERoutingMetadata]):
    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    routing_metadata: TMoERoutingMetadata
    dynamic_scale: torch.Tensor | None = None
    topk_scales: torch.Tensor | None = None


# dispatch -> mlp -> combine
@dataclass(frozen=True, slots=True)
class MoEMlpComputeInput:
    """Input to MLP compute."""

    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    dynamic_scale: torch.Tensor | None
    topk_scales: torch.Tensor | None
    weights: MoEWeights
    quant: MoEQuantParams
    mlp: MoEMlpParams
    kernel: MoEMlpKernelParams


@dataclass(frozen=True, slots=True)
class MoETokenCombineOutput:
    routed_out: torch.Tensor


__all__ = [
    "MoEPrepareOutput",
    "MoEWeights",
    "MoEFusedExpertsInput",
    "MoETokenDispatchInput",
    "MoEMC2RoutingMetadata",
    "MoEAllGatherRoutingMetadata",
    "MoEAllToAllRoutingMetadata",
    "MoETokenDispatchOutput",
    "MoEMlpComputeInput",
    "MoETokenCombineOutput",
    "TMoERoutingMetadata",
]
