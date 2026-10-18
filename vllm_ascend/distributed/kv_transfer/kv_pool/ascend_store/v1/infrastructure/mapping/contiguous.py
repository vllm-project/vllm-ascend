"""Bind each Entry's complete block payload as one contiguous local memory segment."""

import numpy as np

from .types import LocalMemorySegments, UInt64Array


def bind_segments(
    addresses: UInt64Array, entry_block_payload_bytes: UInt64Array, strides_bytes: UInt64Array
) -> LocalMemorySegments:
    if np.all(strides_bytes == strides_bytes[0]):
        strides_bytes = strides_bytes[:1]
    return LocalMemorySegments(addresses, strides_bytes, entry_block_payload_bytes)
