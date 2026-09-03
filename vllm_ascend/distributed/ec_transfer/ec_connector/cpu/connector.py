# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Role shell for the Ascend CPU encoder-cache offload connector."""

from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.cpu.connector import ECCPUConnector

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class AscendECCPUConnector(ECCPUConnector):
    """Adapt upstream ECCPU scheduling and transfer plumbing for Ascend."""

    def _make_worker(self, vllm_config: "VllmConfig"):
        from vllm_ascend.distributed.ec_transfer.ec_connector.cpu.worker import (
            AscendECCPUWorker,
        )

        return AscendECCPUWorker(vllm_config)
