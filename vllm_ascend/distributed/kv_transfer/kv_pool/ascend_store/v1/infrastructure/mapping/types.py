"""Mapping value contracts; addresses, lengths and strides are in bytes.

KeysByPartition[p][r] pairs partition_index p with transfer_record_index r.
Partitions follow the bound LocalLayout; records retain input hash/block order.
The flat transfer_candidate_index is p * transfer_record_count + r.
BulkTransferArguments contains keys, local addresses and byte lengths.
LayerTransferArguments additionally contains byte offsets in complete objects.
Every argument list is key-aligned; each nested list follows payload segment order.
"""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

UInt64Array: TypeAlias = NDArray[np.uint64]
KeysByPartition: TypeAlias = tuple[tuple[str, ...], ...]
BulkTransferArguments: TypeAlias = tuple[list[str], list[list[int]], list[list[int]]]
LayerTransferArguments: TypeAlias = tuple[list[str], list[list[int]], list[list[int]], list[list[int]]]
