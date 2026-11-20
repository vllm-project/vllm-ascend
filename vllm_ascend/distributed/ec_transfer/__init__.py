# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend overrides for vLLM encoder-cache transfer connectors."""

import importlib

from vllm.distributed.ec_transfer.ec_connector.factory import (
    ECConnectorFactory,
)


def _replace_cpu_connector() -> bool:
    upstream_module = "vllm.distributed.ec_transfer.ec_connector.cpu.connector"
    try:
        importlib.import_module(upstream_module)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == upstream_module or upstream_module.startswith(f"{exc.name}.")):
            return False
        raise

    ECConnectorFactory._registry.pop("ECCPUConnector", None)
    ECConnectorFactory.register_connector(
        "ECCPUConnector",
        "vllm_ascend.distributed.ec_transfer.ec_connector.cpu.connector",
        "AscendECCPUConnector",
    )
    return True


def _replace_mooncake_connector() -> None:
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


def _replace_tiering_connector() -> None:
    if "ECTieringConnector" not in ECConnectorFactory._registry:
        return

    # Older vLLM versions do not provide tiering. Keep registration lazy so
    # importing plugins starts neither filesystem threads nor device workers.
    ECConnectorFactory._registry.pop("ECTieringConnector")
    ECConnectorFactory.register_connector(
        "ECTieringConnector",
        "vllm_ascend.distributed.ec_transfer.ec_connector.tiering.connector",
        "AscendECTieringConnector",
    )


def register_connector() -> None:
    """Register available Ascend encoder-cache transfer adaptations."""

    if not _replace_cpu_connector():
        return
    _replace_mooncake_connector()
    _replace_tiering_connector()
