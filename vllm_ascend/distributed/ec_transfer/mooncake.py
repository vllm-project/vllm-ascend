# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend data-plane adapter for vLLM's native ECMooncakeConnector."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.config import (
    _RESERVATION_TTL_SECONDS,
    MooncakeECConfig,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
    ContiguousAllocator,
    ProducerMemoryPool,
    StagedSources,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.reservation import (
    ConsumerReservationManager,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.scheduler import (
    ECMooncakeScheduler,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import MooncakeTransfer
from vllm.distributed.ec_transfer.ec_connector.mooncake.worker import (
    _MAX_CANCELLED_TRANSFER_IDS,
    ECMooncakeWorker,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector import (
    ECMooncakeConnector,
)
from vllm.utils.math_utils import round_up
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    global_te,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_REGISTRATION_ALIGNMENT = 2 * 1024 * 1024


class AscendMooncakeTransfer(MooncakeTransfer):
    """Reuse Mooncake registration ownership with Ascend's shared engine."""

    def __init__(self, hostname: str, protocol: str) -> None:
        if protocol != "ascend":
            raise ValueError("Ascend ECMooncakeConnector requires mooncake_protocol='ascend'.")
        super().__init__(hostname, protocol)

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            self._engine = global_te.get_transfer_engine(self._hostname, device_name=None)
        return self._engine

    def acquire_sources(self, tensors: list[torch.Tensor]) -> list[int]:
        # Source views need not satisfy the transport's registration alignment.
        raise RuntimeError("Ascend ECMooncakeConnector requires sources in the aligned staging pool.")


class _AscendContiguousAllocator(ContiguousAllocator):
    """Allocate a 2 MiB-aligned registered slab required by Ascend transport."""

    def prepare(self, device: torch.device, transfer: AscendMooncakeTransfer) -> None:
        if self.tensor is not None or self._disabled:
            return
        if device.type != "npu":
            raise ValueError("Ascend ECMooncakeConnector requires ec_buffer_device='npu'.")
        registered_size = round_up(self._capacity, _REGISTRATION_ALIGNMENT)
        storage = torch.empty(
            registered_size + _REGISTRATION_ALIGNMENT,
            dtype=torch.uint8,
            device=device,
        )
        offset = (-storage.data_ptr()) % _REGISTRATION_ALIGNMENT
        tensor = storage.narrow(0, offset, registered_size)
        if transfer.register_memory(tensor) != 0:
            raise RuntimeError("Mooncake EC NPU memory registration failed.")
        self.tensor = tensor
        self._free = [(0, self._capacity)]


class _AscendProducerMemoryPool(ProducerMemoryPool):
    def __init__(self, capacity: int, transfer: AscendMooncakeTransfer) -> None:
        super().__init__(capacity, transfer)
        self._allocator = _AscendContiguousAllocator(capacity)

    def stage(self, tensors: list[torch.Tensor]) -> StagedSources:
        staged = super().stage(tensors)
        if staged is None:
            raise RuntimeError("Ascend ECMooncakeConnector source batch exceeds ec_buffer_size.")
        return staged


class _AscendConsumerMemoryPool(ConsumerMemoryPool):
    def __init__(self, capacity: int, transfer: AscendMooncakeTransfer) -> None:
        super().__init__(capacity, transfer)
        self._allocator = _AscendContiguousAllocator(capacity)

    def _record_release_event(self) -> torch.npu.Event | None:
        pool = self.tensor
        if pool is None:
            return None
        event = torch.npu.Event()
        event.record(torch.npu.current_stream(pool.device))
        return event


class _AscendECMooncakeWorker(ECMooncakeWorker):
    """Reuse vLLM's worker state machine with Ascend memory and transport."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        config = MooncakeECConfig.from_vllm_config(vllm_config)
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None
        protocol = config.protocol if "mooncake_protocol" in ec_config.ec_connector_extra_config else "ascend"
        transfer = AscendMooncakeTransfer(get_ip(), protocol)
        consumer_memory = _AscendConsumerMemoryPool(config.pool_size, transfer)
        producer_memory = _AscendProducerMemoryPool(config.pool_size, transfer)
        reservations = ConsumerReservationManager(
            consumer_memory,
            _RESERVATION_TTL_SECONDS,
            _MAX_CANCELLED_TRANSFER_IDS,
        )
        # Upstream leaves pools unallocated and the dispatcher waiting for work.
        # Replace its lazy data plane before any request or service can use it.
        super().__init__(vllm_config)
        self._buffer_device = "npu" if config.buffer_device == "cuda" else config.buffer_device
        self._transfer = transfer
        self._consumer_memory = consumer_memory
        self._producer_memory = producer_memory
        self._reservations = reservations
        self._control_thread = threading.local()

    def _reserve_push_destination(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not getattr(self._control_thread, "device_set", False):
            torch.npu.set_device(torch.device(self._buffer_device))
            self._control_thread.device_set = True
        return super()._reserve_push_destination(payload)

    def _bind_push_source(self, tensor: torch.Tensor, mm_hash: str) -> None:
        if tensor.device.type != "npu":
            raise ValueError(f"EC source must be on NPU for mm_hash={mm_hash}")
        ready_event = torch.npu.Event()
        ready_event.record(torch.npu.current_stream(tensor.device))
        self._producer_pushes.bind_source(mm_hash, tensor, ready_event)


class AscendECMooncakeConnector(ECMooncakeConnector):
    """Reuse vLLM's scheduler and lifecycle, replacing only the NPU data plane."""

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole) -> None:
        ECConnectorBase.__init__(self, vllm_config=vllm_config, role=role)
        self._scheduler = ECMooncakeScheduler(vllm_config) if role == ECConnectorRole.SCHEDULER else None
        self._worker = _AscendECMooncakeWorker(vllm_config) if role == ECConnectorRole.WORKER else None
        if self._scheduler is None and self._worker is None:
            raise ValueError(f"Unknown EC connector role: {role}")
        self._closed = False
