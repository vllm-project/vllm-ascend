# SPDX-License-Identifier: Apache-2.0
"""Physical tensor geometry for registered views and unallocated wire layouts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorLayout:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    itemsize: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "strides", tuple(self.strides))
        if not self.shape or len(self.shape) != len(self.strides):
            raise ValueError("shape and strides must have matching nonzero ranks")
        if any(type(v) is not int or v <= 0 for v in (*self.shape, *self.strides, self.itemsize)):
            raise ValueError("dimensions, element strides and itemsize must be positive integers")
        # Validate physical stride order; permit transposed/padded views.
        # Logical axis meaning remains the responsibility of the component schema.
        span = 1
        for size, stride in sorted(zip(self.shape, self.strides), key=lambda pair: pair[1]):
            if size > 1 and stride < span:
                raise ValueError("layout must have non-overlapping physical strides")
            span += (size - 1) * stride
