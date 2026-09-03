# SPDX-License-Identifier: Apache-2.0
"""Mooncake shared Host-memory pool for sparse KV offload."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return (int(value) + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class HostPoolTopology:
    """Tensor-parallel ownership of one shared Host-memory pool."""

    tp_rank: int
    tp_size: int
    owner_rank: int = 0
    device_id: int = 0
    dp_rank: int = 0
    tp_group: Any = None

    def __post_init__(self) -> None:
        if self.tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {self.tp_size}")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(
                f"tp_rank out of range: rank={self.tp_rank}, "
                f"size={self.tp_size}"
            )
        if not 0 <= self.owner_rank < self.tp_size:
            raise ValueError(
                f"owner_rank out of range: owner={self.owner_rank}, "
                f"size={self.tp_size}"
            )


@dataclass
class HostMemoryRegion:
    """One shared allocation and its local NPU-addressable tensor view."""

    tensor: torch.Tensor
    handle: Any = None
    register_location: str | None = None
    release_callback: Callable[[Any], None] | None = None
    _released: bool = field(default=False, init=False)

    def release(self) -> None:
        if self._released:
            return
        if self.release_callback is not None:
            self.release_callback(self.handle)
        self.handle = None
        self._released = True


def _select_shared_segment_mode() -> tuple[bool, bool]:
    """Select a Mooncake mode that exposes an NPU-addressable Host VA."""
    try:
        from mooncake.shared_segment import shared_segment_supported
    except ImportError as exc:
        raise RuntimeError(
            "Mooncake shared_segment support is required for sparse KV "
            "offload with the Mooncake Host backend"
        ) from exc

    if shared_segment_supported(mmap=False):
        return False, False
    if shared_segment_supported(mmap=True, host_register=True):
        return True, True
    raise RuntimeError(
        "Mooncake shared_segment cannot expose an NPU-addressable address"
    )


def allocate_mooncake_host_region(
    *,
    size_bytes: int,
    alignment: int,
    topology: HostPoolTopology,
    name: str = "sparse_kv_offload_host_pool",
) -> HostMemoryRegion:
    """Create or map one shared segment for a Decode DP group's Host KV."""
    try:
        from mooncake.shared_segment import create_shared_segment
    except ImportError as exc:
        raise RuntimeError(
            "Mooncake shared_segment support is required for sparse KV "
            "offload with the Mooncake Host backend"
        ) from exc

    if size_bytes <= 0:
        raise ValueError(f"size_bytes must be positive, got {size_bytes}")
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    allocation_size_bytes = int(size_bytes) + int(alignment) - 1
    if topology.tp_size > 1 and topology.tp_group is None:
        raise RuntimeError(
            "create_shared_segment requires tp_group when tp_size > 1: "
            f"tp={topology.tp_rank}/{topology.tp_size}"
        )

    mmap, host_register = _select_shared_segment_mode()
    segment_name = f"{name}_dp{topology.dp_rank}"
    logger.info(
        "Creating sparse KV Mooncake Host pool name=%s tp=%s/%s owner=%s "
        "device=%s mmap=%s host_register=%s size_bytes=%s allocated=%s",
        segment_name,
        topology.tp_rank,
        topology.tp_size,
        topology.owner_rank,
        topology.device_id,
        mmap,
        host_register,
        size_bytes,
        allocation_size_bytes,
    )
    segment = create_shared_segment(
        segment_name,
        blocks={
            "pool": {
                "count": 1,
                "shape": (allocation_size_bytes,),
                "dtype": torch.int8,
            }
        },
        world_size=topology.tp_size,
        rank_id=topology.tp_rank,
        owner_rank=topology.owner_rank,
        device_id=topology.device_id,
        tp_group=topology.tp_group,
        mmap=mmap,
        host_register=host_register,
    )
    if host_register and os.getenv(
        "VLLM_ASCEND_SKIP_MIGRATEPAGES"
    ) is None:
        os.environ["VLLM_ASCEND_SKIP_MIGRATEPAGES"] = "1"

    raw = segment.tensors("pool")[0].reshape(-1)
    base_offset = _align_up(raw.data_ptr(), alignment) - raw.data_ptr()
    aligned = raw.narrow(0, base_offset, int(size_bytes))
    device = getattr(aligned, "device", None)
    if device is None or getattr(device, "type", None) == "cpu":
        raise RuntimeError(
            "Mooncake shared segment did not expose an NPU tensor: "
            f"device={device}, mmap={mmap}, host_register={host_register}"
        )
    return HostMemoryRegion(
        tensor=aligned,
        handle=segment,
        register_location=f"npu:{topology.device_id}",
    )


class MooncakeHostPool:
    """Aligned bump allocator over one Mooncake shared segment."""

    def __init__(
        self,
        region: HostMemoryRegion,
        topology: HostPoolTopology,
    ) -> None:
        if region.tensor.dtype != torch.int8:
            raise TypeError(
                "Mooncake Host pool must use an int8 byte tensor, got "
                f"{region.tensor.dtype}"
            )
        if not region.tensor.is_contiguous():
            raise ValueError("Mooncake Host pool allocation must be contiguous")
        self.region = region
        self.topology = topology
        self._offset = 0
        self._registered_engine: Any = None
        self._closed = False

    @classmethod
    def allocate(
        cls,
        *,
        size_bytes: int,
        alignment: int,
        topology: HostPoolTopology,
    ) -> MooncakeHostPool:
        region = allocate_mooncake_host_region(
            size_bytes=size_bytes,
            alignment=alignment,
            topology=topology,
        )
        try:
            return cls(region, topology)
        except Exception:
            region.release()
            raise

    @property
    def data_ptr(self) -> int:
        return int(self.region.tensor.data_ptr())

    @property
    def nbytes(self) -> int:
        return self.region.tensor.numel()

    @property
    def is_owner(self) -> bool:
        return self.topology.tp_rank == self.topology.owner_rank

    def allocate_tensors(
        self,
        sizes: list[int],
        alignment: int,
    ) -> list[torch.Tensor]:
        if self._closed:
            raise RuntimeError("cannot allocate from a closed Mooncake Host pool")
        tensors: list[torch.Tensor] = []
        for size in sizes:
            if size < 0:
                raise ValueError(f"allocation size must be non-negative, got {size}")
            start = _align_up(self.data_ptr + self._offset, alignment) - self.data_ptr
            end = start + int(size)
            if end > self.nbytes:
                raise MemoryError(
                    "Mooncake Host pool is exhausted: "
                    f"requested={size}, offset={start}, capacity={self.nbytes}"
                )
            tensors.append(self.region.tensor.narrow(0, start, int(size)))
            self._offset = end
        return tensors

    def register(self, engine: Any) -> None:
        if self._closed:
            raise RuntimeError("cannot register a closed Mooncake Host pool")
        if not self.is_owner:
            raise RuntimeError(
                "only the owner rank may register the Mooncake Host pool: "
                f"rank={self.topology.tp_rank}, owner={self.topology.owner_rank}"
            )
        if self._registered_engine is engine:
            return
        if self._registered_engine is not None:
            raise RuntimeError("Mooncake Host pool is already registered")
        location = self.region.register_location
        if location is None:
            result = engine.register_memory(self.data_ptr, self.nbytes)
        else:
            try:
                result = engine.register_memory(
                    self.data_ptr,
                    self.nbytes,
                    location=location,
                )
            except TypeError:
                result = engine.register_memory(
                    self.data_ptr,
                    self.nbytes,
                    location,
                )
        if result not in (0, None):
            raise RuntimeError(
                "Mooncake register_memory failed for sparse KV Host pool: "
                f"result={result}, ptr=0x{self.data_ptr:x}, size={self.nbytes}"
            )
        self._registered_engine = engine

    def unregister(self) -> None:
        if self._registered_engine is None:
            return
        unregister_memory = getattr(
            self._registered_engine,
            "unregister_memory",
            None,
        )
        if unregister_memory is None:
            raise RuntimeError(
                "Mooncake engine must unregister the Host pool before release"
            )
        try:
            result = unregister_memory(self.data_ptr)
        except TypeError:
            result = unregister_memory(self.data_ptr, self.nbytes)
        if result not in (0, None):
            raise RuntimeError(
                "Mooncake unregister_memory failed for sparse KV Host pool: "
                f"result={result}, ptr=0x{self.data_ptr:x}"
            )
        self._registered_engine = None

    def close(self) -> None:
        if self._closed:
            return
        self.unregister()
        self.region.release()
        self._closed = True
