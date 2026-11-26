# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Tensor-independent pHash operations shared by schedulers and the index."""

PHASH_BITS = 64
PHASH_NUM_BANDS = 4


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def bands(phash: int, num_bands: int = PHASH_NUM_BANDS) -> list[tuple[int, int]]:
    """Recall all hashes within fewer than ``num_bands`` differing bits."""
    if isinstance(num_bands, bool) or not isinstance(num_bands, int) or num_bands <= 0 or PHASH_BITS % num_bands:
        raise ValueError(f"num_bands must be a positive integer divisor of {PHASH_BITS}")
    width = PHASH_BITS // num_bands
    mask = (1 << width) - 1
    return [(i, (phash >> (i * width)) & mask) for i in range(num_bands)]
