from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from vllm_ascend.ops.fused_moe.moe_stage_params import (
    MoEMlpKernelParams,
    MoEMlpParams,
    MoEQuantParams,
    MoERoutingParams,
)
from vllm_ascend.ops.fused_moe.moe_stage_weights import MoEWeights

TMoERoutingMetadata = TypeVar("TMoERoutingMetadata")


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
class MoETokenDispatchOutput(Generic[TMoERoutingMetadata]):
    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    routing_metadata: TMoERoutingMetadata
    dynamic_scale: torch.Tensor | None = None
    topk_scales: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MoETokenCombineOutput:
    routed_out: torch.Tensor


@dataclass(frozen=True, slots=True)
class MoEPrepareOutput:
    """Typed output from prepare stage."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    mc2_mask: torch.Tensor | None
    padded_hidden_states_shape: torch.Size | None
    pertoken_scale: torch.Tensor | None = None


__all__ = [
    "MoEFusedExpertsInput",
    "MoEMlpComputeInput",
    "MoEPrepareOutput",
    "MoETokenCombineOutput",
    "MoETokenDispatchInput",
    "MoETokenDispatchOutput",
    "TMoERoutingMetadata",
]
