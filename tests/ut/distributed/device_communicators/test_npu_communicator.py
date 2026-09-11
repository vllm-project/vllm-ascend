#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from unittest.mock import MagicMock, patch

from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator


def test_npu_communicator_exposes_graph_capture_compatibility_fields() -> None:
    with (
        patch.object(DeviceCommunicatorBase, "__init__", return_value=None),
        patch("torch.npu.current_device", return_value=0),
    ):
        communicator = NPUCommunicator(cpu_group=MagicMock())

    assert communicator.ca_comm is None
    assert communicator.fi_pcie_ipc_ar_comm is None
