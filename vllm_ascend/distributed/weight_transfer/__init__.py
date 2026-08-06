#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from vllm.distributed.weight_transfer.factory import (
    WeightTransferEngineFactory,
    WeightTransferTrainerFactory,
)

from vllm_ascend.utils import vllm_version_is


def register_engine():
    """Register Ascend weight transfer engines as vLLM plugins."""
    WeightTransferEngineFactory.register_engine(
        "hccl",
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        "HCCLWeightTransferEngine",
    )
    WeightTransferEngineFactory.register_engine(
        "npu_ipc",
        "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
        "NPUIPCWeightTransferEngine",
    )
    # main2main compat: the stateful `IPCTrainerWeightTransferEngine` path was
    # introduced in vllm main after 0.26.0; on 0.26.0 the trainer still uses
    # the static `IPCWeightTransferEngine.trainer_send_weights(...)` path, so
    # the trainer-side factory has nothing to register.
    if not vllm_version_is("0.26.0"):
        WeightTransferTrainerFactory.register_engine(
            "npu_ipc",
            "vllm_ascend.distributed.weight_transfer.npu_ipc_engine",
            "NPUIPCTrainerWeightTransferEngine",
        )
