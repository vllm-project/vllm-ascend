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

from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory


def register_ec_connector():
    ECConnectorFactory.register_connector(
        "EMoonCakeStoreConnector",
        "vllm_ascend.distributed.ec_transfer.e_mooncake_store_connector",
        "EMoonCakeStoreConnector",
    )
