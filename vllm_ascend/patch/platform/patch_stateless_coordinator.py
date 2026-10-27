#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
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
# This file is a part of the vllm-ascend project.
#
# Why this patch exists:
#   vLLM (>= 0.26) elastic EP creates *standby* groups through
#   vllm/distributed/stateless_coordinator.py::StatelessGroupCoordinator.
#   For groups with use_device_communicator=True (the standby DP/EP/EPLB
#   groups created by elastic_ep/standby_state.py), the constructor does:
#
#       device_comm_cls = resolve_obj_by_qualname(
#           current_platform.get_device_communicator_cls())
#       assert device_comm_cls == CudaCommunicator
#       self.device_communicator = CudaCommunicator(...)
#
#   On Ascend NPU, get_device_communicator_cls() returns NPUCommunicator,
#   so the assert fails and elastic EP standby group creation aborts.
#
# Approach (module symbol swap):
#   The assert and the construction both reference the module-global name
#   `CudaCommunicator` inside stateless_coordinator.py. We replace that
#   module attribute with NPUCommunicator, so:
#     - the assert compares NPUCommunicator == NPUCommunicator (passes),
#     - the standby groups get an NPUCommunicator built with the stateless
#       constructor signature (global_ranks/global_world_size/
#       tcp_store_group), which NPUCommunicator accepts since this change.
#   No vLLM behavior changes for CUDA users (this module is only imported
#   by the vllm-ascend platform patch chain on Ascend).
#
# Timing — guaranteed to run before first standby group creation:
#
#   any vLLM entrypoint
#     → vllm.platforms.__getattr__("current_platform")
#     → vllm_ascend:register() → NPUPlatform()
#     → NPUPlatform.pre_register_and_update()
#     → adapt_patch(is_global_patch=True)
#     → imports vllm_ascend.patch.platform
#     → THIS PATCH RUNS
#     ...
#     later: elastic scaling → create_standby_groups()
#       → StatelessGroupCoordinator(use_device_communicator=True)
#       → constructs NPUCommunicator (this patch's swap)
#
# Future Plan:
#   Remove this patch once upstream vLLM instantiates the platform's own
#   device communicator class (from get_device_communicator_cls()) instead
#   of hardcoding CudaCommunicator in stateless_coordinator.py.

import vllm.distributed.stateless_coordinator as _stateless_coordinator

from vllm_ascend.distributed.device_communicators.npu_communicator import (
    NPUCommunicator,
)

_stateless_coordinator.CudaCommunicator = NPUCommunicator
