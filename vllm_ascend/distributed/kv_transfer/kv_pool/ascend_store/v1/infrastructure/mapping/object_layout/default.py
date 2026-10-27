"""Default objects contain exactly their assigned local payload, without gaps."""

from ..local_layout import LocalLayout
from .offsets import build_segment_offsets
from .types import RemoteObjectLayout


def build_default_object_layouts(local: LocalLayout) -> tuple[RemoteObjectLayout, ...]:
    return tuple(
        RemoteObjectLayout(source.payload_bytes, offsets)
        for source, offsets in zip(local.partitions, build_segment_offsets(local.partitions), strict=True)
    )
