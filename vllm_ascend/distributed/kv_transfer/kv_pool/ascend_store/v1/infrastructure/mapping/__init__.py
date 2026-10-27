"""Compose local layout, object naming and object placement into transfer views."""

from .bulk import BulkMapping
from .layerwise import LayerwiseMapping

__all__ = ["BulkMapping", "LayerwiseMapping"]
