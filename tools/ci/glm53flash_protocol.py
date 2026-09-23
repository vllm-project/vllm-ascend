# SPDX-License-Identifier: Apache-2.0
"""CPU-only scenario contract shared by Flash collection and analysis."""

PRECISION_LENGTHS = (3, 4, 5, 127, 128, 129, 511, 512, 513, 2047, 2048, 2049, 2051, 2052, 2053, 2177)


def precision_lengths(block_size):
    """Return all required lengths, deduplicating overlapping block boundaries."""
    if type(block_size) is not int or block_size < 2:
        raise ValueError("block_size must be an integer >= 2")
    return sorted(set(PRECISION_LENGTHS) | {block_size - 1, block_size, block_size + 1})


def validate_precision_lengths(lengths, block_size):
    """Require the exact collector contract, not an arbitrary case count."""
    expected = precision_lengths(block_size)
    if not isinstance(lengths, list) or any(type(length) is not int for length in lengths) or lengths != expected:
        raise ValueError(f"Full boundary collection required: expected {expected}, got {lengths!r}")
