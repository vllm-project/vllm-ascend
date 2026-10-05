"""Explicit registration binders for both Layerwise data planes."""

from typing import TypeAlias

from .gva import (
    GVALayerwiseGroupProjection,
    GVALayerwiseProjection,
    GVALayerwiseProjectionBinder,
    gva_layer_ranges,
    gva_local_keys,
    gva_lookup_keys,
)
from .key_range import (
    KeyRangeLayerwiseGroupProjection,
    KeyRangeLayerwiseProjection,
    KeyRangeLayerwiseProjectionBinder,
    key_range_layer_ranges,
    key_range_local_keys,
    key_range_lookup_keys,
)

LayerwiseProjection: TypeAlias = KeyRangeLayerwiseProjection | GVALayerwiseProjection
LayerwiseProjectionBinder: TypeAlias = KeyRangeLayerwiseProjectionBinder | GVALayerwiseProjectionBinder
LayerwiseGroupProjection: TypeAlias = KeyRangeLayerwiseGroupProjection | GVALayerwiseGroupProjection

__all__ = (
    "LayerwiseProjection",
    "GVALayerwiseGroupProjection",
    "GVALayerwiseProjectionBinder",
    "GVALayerwiseProjection",
    "KeyRangeLayerwiseGroupProjection",
    "KeyRangeLayerwiseProjectionBinder",
    "KeyRangeLayerwiseProjection",
    "LayerwiseGroupProjection",
    "LayerwiseProjectionBinder",
    "gva_layer_ranges",
    "gva_local_keys",
    "gva_lookup_keys",
    "key_range_layer_ranges",
    "key_range_local_keys",
    "key_range_lookup_keys",
)
