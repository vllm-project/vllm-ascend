# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""NPU worker for vLLM's Mooncake encoder-cache connector."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake.config import (
    MooncakeECConfig,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
    ProducerMemoryPool,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    MooncakeTransfer,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.worker import (
    ECMooncakeWorker,
)
from vllm.utils.math_utils import round_up

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ASCEND_DIRECT_MEMORY_ALIGNMENT,
    AscendConsumerMemoryPool,
    AscendContiguousAllocator,
    AscendProducerAllocator,
    AscendProducerMemoryPool,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    AscendMooncakeTransfer,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


_DEFAULT_BOUNCE_LIMIT = 128
_BOUNCE_ARENA_CONFIG_KEY = "ascend_mooncake_bounce_arena_size"


def _resolve_bounce_arena_size(vllm_config: VllmConfig) -> int:
    ec_config = vllm_config.ec_transfer_config
    assert ec_config is not None

    extra_config = ec_config.ec_connector_extra_config

    if _BOUNCE_ARENA_CONFIG_KEY not in extra_config:
        quantum_count = min(
            _DEFAULT_BOUNCE_LIMIT,
            vllm_config.scheduler_config.max_num_seqs,
        )
        return ASCEND_DIRECT_MEMORY_ALIGNMENT * quantum_count

    value = extra_config[_BOUNCE_ARENA_CONFIG_KEY]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{_BOUNCE_ARENA_CONFIG_KEY} must be an integer")
    if value < ASCEND_DIRECT_MEMORY_ALIGNMENT:
        raise ValueError(
            f"{_BOUNCE_ARENA_CONFIG_KEY} must be at least "
            f"{ASCEND_DIRECT_MEMORY_ALIGNMENT} bytes"
        )

    return round_up(value, ASCEND_DIRECT_MEMORY_ALIGNMENT)


class AscendECMooncakeWorker(ECMooncakeWorker):
    """Reuse upstream orchestration with Ascend-specific primitives."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None

        resolved_bounce_arena_size = _resolve_bounce_arena_size(vllm_config)

        self._bounce_arena_size = (
            resolved_bounce_arena_size if ec_config.is_ec_producer else 0
        )

        super().__init__(vllm_config)

    def _make_config(self, vllm_config: VllmConfig) -> MooncakeECConfig:
        config = super()._make_config(vllm_config)

        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None

        protocol = config.protocol
        if "mooncake_protocol" not in ec_config.ec_connector_extra_config:
            protocol = "ascend"
        elif protocol != "ascend":
            raise ValueError("Ascend ECMooncakeConnector requires mooncake_protocol='ascend'")

        buffer_device = config.buffer_device
        if buffer_device == "cuda":
            buffer_device = "npu"
        device_type, separator, device_index = buffer_device.partition(":")
        is_valid_npu_device = device_type == "npu" and (
            not separator or (device_index.isascii() and device_index.isdigit())
        )
        if not is_valid_npu_device:
            raise ValueError("Ascend ECMooncakeWorker requires ec_buffer_device='npu'")

        return replace(
            config,
            protocol=protocol,
            buffer_device=buffer_device,
        )

    def _make_transfer(self, hostname: str, protocol: str) -> AscendMooncakeTransfer:
        if protocol != "ascend":
            raise ValueError("Ascend ECMooncakeConnector requires mooncake_protocol='ascend'")
        device_index = torch.npu.current_device()
        return AscendMooncakeTransfer(hostname, device_index)

    def _make_consumer_memory(self, capacity: int, transfer: MooncakeTransfer) -> ConsumerMemoryPool:
        return AscendConsumerMemoryPool(
            capacity,
            transfer,
            allocator=AscendContiguousAllocator(capacity),
        )

    def _make_producer_memory(self, capacity: int, transfer: MooncakeTransfer) -> ProducerMemoryPool:
        return AscendProducerMemoryPool(
            capacity,
            transfer,
            allocator=AscendProducerAllocator(
                staging_capacity=capacity,
                bounce_capacity=self._bounce_arena_size,
            ),
        )

    def _record_source_ready_event(self, tensor: torch.Tensor) -> torch.Event | None:
        if tensor.device.type != "npu":
            return super()._record_source_ready_event(tensor)
        ready_event = torch.npu.Event()
        ready_event.record(torch.npu.current_stream(tensor.device))
        return ready_event
