# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend overrides for vLLM encoder-cache transfer connectors."""

import importlib


def register_connector() -> None:
    """Replace upstream ECCPUConnector with the Ascend worker adaptation.

    Scheduler-side cache allocation and metadata remain upstream-owned. The
    override is installed only when the upstream CPU connector exists; import
    errors inside an available upstream module are deliberately propagated.
    """
    upstream_module = "vllm.distributed.ec_transfer.ec_connector.cpu.connector"
    try:
        importlib.import_module(upstream_module)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == upstream_module or upstream_module.startswith(f"{exc.name}.")):
            return
        raise

    from vllm.distributed.ec_transfer.ec_connector.factory import (
        ECConnectorFactory,
    )

    # replace worker
    ECConnectorFactory._registry.pop("ECCPUConnector", None)
    ECConnectorFactory.register_connector(
        "ECCPUConnector",
        "vllm_ascend.distributed.ec_transfer.ec_connector.cpu.connector",
        "AscendECCPUConnector",
    )
