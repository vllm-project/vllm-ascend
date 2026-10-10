# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend overrides for vLLM encoder-cache transfer connectors."""

from vllm.distributed.ec_transfer.ec_connector.factory import (
    ECConnectorFactory,
)


def register_connector() -> None:
    """Replace upstream ECMooncakeConnector with the Ascend adaptation."""

    if "ECMooncakeConnector" not in ECConnectorFactory._registry:
        return

    # Preserve the factory's lazy-loading contract. Importing the upstream
    # connector here would also import mooncake.engine in every process that
    # loads general plugins, including model-inspection subprocesses.
    ECConnectorFactory._registry.pop("ECMooncakeConnector")
    ECConnectorFactory.register_connector(
        "ECMooncakeConnector",
        "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.connector",
        "AscendECMooncakeConnector",
    )
