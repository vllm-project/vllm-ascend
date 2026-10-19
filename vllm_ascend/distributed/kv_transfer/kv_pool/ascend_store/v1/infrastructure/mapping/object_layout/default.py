"""Default objects contain exactly their assigned local payload, without gaps."""

from ..local_layout import LocalLayout
from .offsets import build_segment_offsets
from .types import ObjectLayout


def build_default_object_layouts(local: LocalLayout) -> tuple[ObjectLayout, ...]:
    return tuple(
        ObjectLayout(source.payload_bytes, offsets)
        for source, offsets in zip(local.objects, build_segment_offsets(local.objects), strict=True)
    )
