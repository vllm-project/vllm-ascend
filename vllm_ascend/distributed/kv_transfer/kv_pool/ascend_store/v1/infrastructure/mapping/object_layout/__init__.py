"""Object sizes and segment positions; Backend-specific placement stays in Backend."""

from .default import build_default_object_layouts
from .types import RemoteObjectLayout, RemoteObjectLayoutFactory

__all__ = ["RemoteObjectLayout", "RemoteObjectLayoutFactory", "build_default_object_layouts"]
