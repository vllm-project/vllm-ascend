# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend overrides for vLLM encoder-cache transfer connectors."""

import importlib


def register_connector() -> None:
    """
    Replace upstream ECMooncakeConnector with the Ascend adaptation.
    """

    upstream_module = "vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector"
    try:
        importlib.import_module(upstream_module)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == upstream_module or upstream_module.startswith(f"{exc.name}.")):
            return
        raise

    from vllm.distributed.ec_transfer.ec_connector.factory import (
        ECConnectorFactory,
    )

    # Replace the upstream connector with the Ascend adapter.
    ECConnectorFactory._registry.pop("ECMooncakeConnector", None)
    ECConnectorFactory.register_connector(
        "ECMooncakeConnector",
        "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.connector",
        "AscendECMooncakeConnector",
    )
