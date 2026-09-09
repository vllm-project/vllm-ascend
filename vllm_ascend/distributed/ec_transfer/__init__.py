# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend overrides for vLLM encoder-cache transfer connectors."""

import importlib


def _replace_connector(
    name: str,
    upstream_module: str,
    ascend_module: str,
    ascend_class: str,
) -> None:
    try:
        importlib.import_module(upstream_module)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == upstream_module or upstream_module.startswith(f"{exc.name}.")):
            return
        raise

    from vllm.distributed.ec_transfer.ec_connector.factory import (
        ECConnectorFactory,
    )

    ECConnectorFactory._registry.pop(name, None)
    ECConnectorFactory.register_connector(
        name,
        ascend_module,
        ascend_class,
    )


def register_connector() -> None:
    """Register available Ascend encoder-cache transfer adaptations."""

    _replace_connector(
        "ECCPUConnector",
        "vllm.distributed.ec_transfer.ec_connector.cpu.connector",
        "vllm_ascend.distributed.ec_transfer.ec_connector.cpu.connector",
        "AscendECCPUConnector",
    )
    _replace_connector(
        "ECMooncakeConnector",
        "vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector",
        "vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.connector",
        "AscendECMooncakeConnector",
    )
