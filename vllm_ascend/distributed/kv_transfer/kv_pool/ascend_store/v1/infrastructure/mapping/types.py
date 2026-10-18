"""Mapping value contracts; addresses, lengths and strides are in bytes.

KeysByPrefix[p][r] pairs key-prefix index p with input block/hash row r.
BulkTransferArguments contains keys, local addresses and byte lengths.
LayerTransferArguments additionally contains byte offsets in complete objects.
Every argument list is key-aligned; each nested list follows payload segment order.
"""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

UInt64Array: TypeAlias = NDArray[np.uint64]
KeysByPrefix: TypeAlias = tuple[tuple[str, ...], ...]
BulkTransferArguments: TypeAlias = tuple[list[str], list[list[int]], list[list[int]]]
LayerTransferArguments: TypeAlias = tuple[list[str], list[list[int]], list[list[int]], list[list[int]]]


@dataclass(frozen=True, slots=True)
class LocalMemorySegments:
    """Read-only descriptions of contiguous LOCAL payload intervals for one block.

    For block ID b, segment j starts at block_zero_addresses[j] plus b times
    group_block_strides_bytes[j], and contains segment_lengths_bytes[j] bytes.
    The stride advances between blocks; it is not the payload length.
    Segments follow object serialization order, not necessarily address order.
    Strides and lengths may be singletons shared by all segment columns.
    This describes locations only: it neither owns memory nor proves readiness.
    """

    block_zero_addresses: UInt64Array
    group_block_strides_bytes: UInt64Array
    segment_lengths_bytes: UInt64Array
