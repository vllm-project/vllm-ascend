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

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class MoeRouterInput:
    """Routing and dispatch side inputs for one MoE invocation.

    `pertoken_scale` is intentionally kept here even though it is not a pure
    routing concept. It is used by pre-quantized activation flows, currently
    the AllGather + EP W8A8 prepare path, where prepare emits per-token
    activation scales and dispatch needs to carry them forward so the MLP
    quant path can reuse those scales instead of requantizing activations.
    """

    expert_map: torch.Tensor | None
    global_redundant_expert_num: int
    mc2_mask: torch.Tensor | None
    apply_router_weight_on_input: bool
    log2phy: torch.Tensor | None = None
    pertoken_scale: torch.Tensor | None = None
