# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Role shell for the Ascend CPU encoder-cache offload connector."""

from typing import TYPE_CHECKING

import torch
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    _get_encoder_cache_hidden_dim,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.connector import ECCPUConnector

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput


class AscendECCPUConnector(ECCPUConnector):
    """Adapt upstream ECCPU scheduling and transfer plumbing for Ascend."""

    def __init__(
        self, vllm_config: "VllmConfig", role: ECConnectorRole
    ) -> None:
        ec_config = vllm_config.ec_transfer_config
        if ec_config is None:
            raise ValueError("ec_transfer_config is required for ECCPUConnector")

        extra_config = ec_config.ec_connector_extra_config or {}
        raw_cpu_bytes = extra_config.get("ec_cpu_bytes")
        try:
            cpu_bytes = int(raw_cpu_bytes)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "ec_cpu_bytes must be a positive integer in "
                "ec_connector_extra_config"
            ) from exc

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

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECCPUConnectorMetadata:
        metadata = super().build_connector_meta(scheduler_output)

        # Upstream's LIFO allocator returns the selected blocks in stack-pop
        # order. Treat the allocation as a set and publish a stable ascending
        # order for both save and load metadata. This keeps their byte layout
        # identical while exposing physically adjacent blocks as contiguous
        # runs that the Ascend worker can submit as one larger DMA descriptor.
        metadata.saves = {
            mm_hash: sorted(block_ids)
            for mm_hash, block_ids in metadata.saves.items()
        }
        metadata.loads = {
            mm_hash: sorted(block_ids)
            for mm_hash, block_ids in metadata.loads.items()
        }
        return metadata
