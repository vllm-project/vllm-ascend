"""Bind registered KVCache memory without constructing names or remote addresses."""

from .binding import bind_local_layout
from .types import LocalLayout, LocalMemorySegments, ObjectCoordinate, ObjectSourceSegments

__all__ = ["LocalLayout", "LocalMemorySegments", "ObjectSourceSegments", "ObjectCoordinate", "bind_local_layout"]
