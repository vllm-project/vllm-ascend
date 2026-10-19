"""Object sizes and segment positions; Backend-specific placement stays in Backend."""

from .default import build_default_object_layouts
from .types import ObjectLayout, ObjectLayoutFactory

__all__ = ["ObjectLayout", "ObjectLayoutFactory", "build_default_object_layouts"]
