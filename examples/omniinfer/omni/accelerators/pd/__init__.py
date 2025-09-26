# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.distributed.kv_transfer.kv_connector.factory import \
    KVConnectorFactory


def register():
    KVConnectorFactory.register_connector(
        "AscendHcclConnectorV1",
        "omni.accelerators.pd.llmdatadist_connector_v1",
        "LLMDataDistConnector"
    )