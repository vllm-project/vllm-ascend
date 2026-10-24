"""Remote object geometry is static; concrete remote bases belong to Backend sessions."""

from dataclasses import dataclass
from typing import Protocol

from ..local_layout import LocalLayout
from ..types import UInt64Array


@dataclass(frozen=True, slots=True)
class ObjectLayout:
    """One local object's segment positions in a complete remote object, in bytes.

    Offsets are read-only. The final offset is the end of the payload, allowing
    layer boundary indexing.
    """

    size_bytes: int
    segment_offsets_bytes: UInt64Array


class ObjectLayoutFactory(Protocol):
    """Return one layout per local object, preserving object and segment order."""

    def __call__(self, local: LocalLayout) -> tuple[ObjectLayout, ...]: ...
