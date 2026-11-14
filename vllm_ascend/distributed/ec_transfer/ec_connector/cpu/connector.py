# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Role shell for the Ascend CPU encoder-cache offload connector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.cpu.connector import ECCPUConnector

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole


class AscendECCPUConnector(ECCPUConnector):
    """Adapt upstream ECCPU scheduling and transfer plumbing for Ascend."""

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole) -> None:
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None
        if "ec_enable_nixl" in ec_config.ec_connector_extra_config:
            raise ValueError(
                "Ascend ECCPUConnector does not support ec_enable_nixl; remove it from ec_connector_extra_config."
            )
        super().__init__(vllm_config, role)

    def _make_worker(self, vllm_config: VllmConfig):
        from vllm_ascend.distributed.ec_transfer.ec_connector.cpu.worker import (
            AscendECCPUWorker,
        )

        return AscendECCPUWorker(vllm_config)
