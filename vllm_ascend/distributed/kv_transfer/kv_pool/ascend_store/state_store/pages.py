"""Logical-token plans for replicated MLA/indexer buffers across TP layouts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageLayout:
    block_tokens: int
    kernel_tokens: int
    tokens_per_row: int
    row_bytes: int
    row_stride_bytes: int
    kernel_stride_bytes: int
    block_stride_bytes: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError("positive integer page dimensions are required")
        if self.block_tokens % self.kernel_tokens or self.kernel_tokens % self.tokens_per_row:
            raise ValueError("scheduler/kernel blocks must align with compression rows")
        rows = self.kernel_tokens // self.tokens_per_row
        kernel_span = (rows - 1) * self.row_stride_bytes + self.row_bytes
        block_span = (self.block_tokens // self.kernel_tokens - 1) * self.kernel_stride_bytes + kernel_span
        if (
            self.row_stride_bytes < self.row_bytes
            or self.kernel_stride_bytes < kernel_span
            or self.block_stride_bytes < block_span
        ):
            raise ValueError("overlapping physical page rows")
