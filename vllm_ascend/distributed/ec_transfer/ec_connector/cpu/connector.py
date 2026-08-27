# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Role shell for the Ascend CPU encoder-cache offload connector."""

from typing import TYPE_CHECKING

import torch
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    _get_encoder_cache_hidden_dim,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.connector import ECCPUConnector

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class AscendECCPUConnector(ECCPUConnector):
    """Adapt upstream ECCPU scheduling and transfer plumbing for Ascend."""

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        ec_config = vllm_config.ec_transfer_config
        if ec_config is None:
            raise ValueError("ec_transfer_config is required for ECCPUConnector")

        extra_config = ec_config.ec_connector_extra_config or {}
        raw_cpu_bytes = extra_config.get("ec_cpu_bytes")
        if raw_cpu_bytes is None:
            raise ValueError("ec_cpu_bytes must be a positive integer in ec_connector_extra_config")
        try:
            cpu_bytes = int(raw_cpu_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError("ec_cpu_bytes must be a positive integer in ec_connector_extra_config") from exc

        dtype = vllm_config.model_config.dtype
        element_size = torch.empty((), dtype=dtype).element_size()
        block_size = _get_encoder_cache_hidden_dim(vllm_config) * element_size
        if cpu_bytes < block_size:
            raise ValueError(
                f"ec_cpu_bytes must hold at least one encoder-cache block: "
                f"configured={cpu_bytes}, block_size={block_size}"
            )

        super().__init__(vllm_config, role)

    def _make_worker(self, vllm_config: "VllmConfig"):
        from vllm_ascend.distributed.ec_transfer.ec_connector.cpu.worker import (
            AscendECCPUWorker,
        )

        return AscendECCPUWorker(vllm_config)
