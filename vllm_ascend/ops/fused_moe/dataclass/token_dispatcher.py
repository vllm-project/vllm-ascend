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
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import torch

from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput

if TYPE_CHECKING:
    from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEFusedExpertsInput

TMoECombineMetadata = TypeVar("TMoECombineMetadata")


@dataclass(frozen=True, slots=True)
class MoEMC2CombineMetadata:
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    expert_map: torch.Tensor | None
    ep_recv_counts: torch.Tensor
    tp_recv_counts: torch.Tensor
    assist_info_for_combine: torch.Tensor
    expand_scales: torch.Tensor | None
    quant: MoEQuantParams
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
class MoETokenDispatchInput:
    """Input to token dispatch."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    routing: MoeRouterInput
    quant: MoEQuantParams


@dataclass(frozen=True, slots=True)
class MoETokenDispatchOutput(Generic[TMoECombineMetadata]):
    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    combine_metadata: TMoECombineMetadata
    dynamic_scale: torch.Tensor | None = None
    topk_scales: torch.Tensor | None = None


def build_token_dispatch_input(
    *,
    fused_experts_input: MoEFusedExpertsInput,
) -> MoETokenDispatchInput:
    return MoETokenDispatchInput(
        hidden_states=fused_experts_input.hidden_states,
        topk_weights=fused_experts_input.topk_weights,
        topk_ids=fused_experts_input.topk_ids,
        routing=fused_experts_input.routing,
        quant=fused_experts_input.quant,
    )


from vllm_ascend.ops.fused_moe.dataclass.fused_experts import (  # noqa: E402
    MoEFusedExpertsInput,
    MoEWeights,
    build_fused_experts_input,
)
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import (  # noqa: E402
    MoEMlpComputeInput,
    build_mlp_compute_input,
)
from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import MoEPrepareOutput  # noqa: E402

MoERoutingParams = MoeRouterInput


__all__ = [
    "MoEAllGatherCombineMetadata",
    "MoEAllToAllCombineMetadata",
    "MoEFusedExpertsInput",
    "MoEMC2CombineMetadata",
    "MoEMlpComputeInput",
    "MoEPrepareOutput",
    "MoEQuantParams",
    "MoERoutingParams",
    "MoeRouterInput",
    "MoETokenDispatchInput",
    "MoETokenDispatchOutput",
    "MoEWeights",
    "TMoECombineMetadata",
    "build_fused_experts_input",
    "build_mlp_compute_input",
    "build_token_dispatch_input",
]
