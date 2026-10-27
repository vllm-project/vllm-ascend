"""Bind registered KVCache memory without constructing names or remote addresses."""

from .binding import bind_local_layout
from .types import LocalLayout, LocalMemorySegments, LocalPartitionLayout, ObjectPartitionCoordinate

__all__ = [
    "LocalLayout",
    "LocalMemorySegments",
    "LocalPartitionLayout",
    "ObjectPartitionCoordinate",
    "bind_local_layout",
]
