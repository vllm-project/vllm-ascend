#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import torch

from vllm_ascend.ops.fused_moe.moe_stage_params import MoEQuantParams, MoERoutingParams

TMoECombineMetadata = TypeVar("TMoECombineMetadata")


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
    quant: MoEQuantParams
    activation: str = "silu"
    need_trans: bool = False
    dynamic_eplb: bool = False


@dataclass(frozen=True, slots=True)
class MoETokenDispatchInput:
    """Input to token dispatch."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    routing: MoERoutingParams
    quant: MoEQuantParams


# dispatch carry-over state consumed by combine
@dataclass(frozen=True, slots=True)
class MoEMC2CombineMetadata:
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    expert_map: torch.Tensor | None
    ep_recv_counts: torch.Tensor
    tp_recv_counts: torch.Tensor
    assist_info_for_combine: torch.Tensor
    expand_scales: torch.Tensor | None
    dispatch_with_quant: bool
    mc2_mask: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MoEAllGatherCombineMetadata:
    topk_weights: torch.Tensor
    expanded_row_idx: torch.Tensor
    restore_shape: torch.Size


@dataclass(frozen=True, slots=True)
class MoEAllToAllCombineMetadata:
    input_splits: np.ndarray
    output_splits: np.ndarray
    topk_weights: torch.Tensor
    reversed_local_input_permutation_mapping: torch.Tensor
    reversed_global_input_permutation_mapping: torch.Tensor | None
    hidden_shape: torch.Size
    hidden_shape_before_permute: torch.Size


@dataclass(frozen=True, slots=True)
class MoETokenDispatchOutput(Generic[TMoECombineMetadata]):
    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    combine_metadata: TMoECombineMetadata
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
    fusion: bool
    activation: str = "silu"
    need_trans: bool = False
    dynamic_eplb: bool = False


__all__ = [
    "MoEPrepareOutput",
    "MoEWeights",
    "MoEFusedExpertsInput",
    "MoETokenDispatchInput",
    "MoEMC2CombineMetadata",
    "MoEAllGatherCombineMetadata",
    "MoEAllToAllCombineMetadata",
    "MoETokenDispatchOutput",
    "MoEMlpComputeInput",
    "TMoECombineMetadata",
]
