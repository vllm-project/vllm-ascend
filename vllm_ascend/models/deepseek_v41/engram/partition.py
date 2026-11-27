# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One integer partition rule for HIXL rows and MemFabric chunks."""


def partition_units(units, index, count):
    """Return the half-open ``[start, end)`` range assigned to one shard."""
    if (
        type(units) is not int
        or units < 1
        or type(index) is not int
        or type(count) is not int
        or count < 1
        or not 0 <= index < count
        or units < count
    ):
        raise ValueError("invalid Engram shard layout")
    return units * index // count, units * (index + 1) // count
