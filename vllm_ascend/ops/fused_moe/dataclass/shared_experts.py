# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RoutedMoEMilestones:
    """Routed-path milestones consumed by shared-expert work.

    The milestones describe execution phases instead of a communication
    backend.  Their meaning therefore stays stable for both routed MoE paths:

    * AllGather performs its distributed input gather in ``prepare()``, before
      ``routed_dispatch_start``.  Its token dispatch/combine are local routing
      ops, followed by a distributed reduce-scatter in ``finalize()``.
    * All2All performs local pad/split work in ``prepare()``.  Its forward and
      reverse EP exchanges start after ``routed_dispatch_start`` and
      ``routed_combine_start``, respectively.

    Shared Gate-Up starts at ``router_output_ready``.  It can consequently
    cover the routed input AllGather, or All2All preprocessing plus its forward
    exchange, without embedding communication-type branches in the shared
    expert implementation.
    """

    shared_input_ready: torch.npu.Event | None = None
    router_output_ready: torch.npu.Event | None = None
    routed_dispatch_start: torch.npu.Event | None = None
    routed_gmm2_start: torch.npu.Event | None = None
    routed_combine_start: torch.npu.Event | None = None
    routed_finalize_done: torch.npu.Event | None = None


@dataclass(frozen=True)
class PreparedSharedExpertInput:
    """Prepared shared-expert activation and its asynchronous readiness.

    ``is_gathered`` describes the tensor layout; ``ready_event`` only describes
    scheduling.  Keeping them independent prevents stream mechanics from being
    used as a proxy for activation semantics.
    """

    hidden_states: torch.Tensor
    is_gathered: bool = False
    ready_event: torch.npu.Event | None = None
