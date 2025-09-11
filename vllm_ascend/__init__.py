#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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


def register():
    """Register the NPU platform."""

    return "vllm_ascend.platform.NPUPlatform"


def register_model():
    from .models import register_model
    register_model()


def register_kvconnector():
    from vllm.distributed.kv_transfer.kv_connector.factory import \
        KVConnectorFactory
    if "P2pHcclConnector" in KVConnectorFactory._registry:
        return

    KVConnectorFactory.register_connector(
        "P2pHcclConnector",
        "vllm_ascend.distributed.kv_transfer.kv_connector.v1.p2p.p2p_hccl_connector",
        "P2pHcclConnector")
