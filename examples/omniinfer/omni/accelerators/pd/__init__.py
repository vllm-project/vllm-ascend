# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os

from vllm.distributed.kv_transfer.kv_connector.factory import \
    KVConnectorFactory


def register():
    KVConnectorFactory.register_connector(
        "AscendHcclConnectorV1",
        "omni.accelerators.pd.omni_cache_connector_v1" if os.getenv("ENABLE_OMNI_CACHE", "0") == "1" 
                                                        else "omni.accelerators.pd.llmdatadist_connector_v1",
        "LLMDataDistConnector"
    )