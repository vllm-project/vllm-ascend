# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Memory geometry shared by pool workers and transport backends."""

from dataclasses import dataclass
from enum import Enum

# Device registration includes an aligned slab prefix. Host ranges must never
# be rounded down: bytes before a Host allocation do not belong to the pool.
DEVICE_BUFFER_ALIGNMENT = 2 * 1024 * 1024


class MemoryLocation(Enum):
    """Location of the local memory addressed by a pool transfer."""

    HOST = "host"
    DEVICE = "device"


@dataclass(frozen=True, kw_only=True)
class BufferDescriptor:
    address: int
    size_bytes: int
    memory_location: MemoryLocation = MemoryLocation.DEVICE
    alignment_bytes: int = 1

    def __post_init__(self) -> None:
        alignment = self.alignment_bytes
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("Pool buffer alignment must be a positive power of two")
        if self.address <= 0 or self.size_bytes <= 0:
            raise ValueError("Pool buffer address and size must be positive")
        if self.address % alignment:
            raise ValueError(f"Pool buffer address must be aligned to {alignment} bytes")


def buffer_arrays(buffers: list[BufferDescriptor]) -> tuple[list[int], list[int]]:
    """Lower registration descriptors only at the SDK boundary."""
    return [buffer.address for buffer in buffers], [buffer.size_bytes for buffer in buffers]
