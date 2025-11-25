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

from vllm.distributed.kv_transfer.kv_connector.factory import \
    KVConnectorFactory


def register_connector():
    KVConnectorFactory.register_connector(
        "LLMDataDistCMgrConnector",
        "vllm_ascend.distributed.llmdatadist_c_mgr_connector",
        "LLMDataDistCMgrConnector")

    KVConnectorFactory.register_connector(
        "MooncakeConnectorV1", "vllm_ascend.distributed.mooncake_connector",
        "MooncakeConnector")

    KVConnectorFactory.register_connector(
        "MooncakeConnectorStoreV1",
        "vllm_ascend.distributed.kvpool.ascend_store_connector",
        "AscendStoreConnector")

    KVConnectorFactory.register_connector(
        "AscendStoreConnector",
        "vllm_ascend.distributed.kvpool.ascend_store_connector",
        "AscendStoreConnector")

    KVConnectorFactory.register_connector(
        "MooncakeLayerwiseConnector",
        "vllm_ascend.distributed.mooncake_layerwise_connector",
        "MooncakeLayerwiseConnector")
