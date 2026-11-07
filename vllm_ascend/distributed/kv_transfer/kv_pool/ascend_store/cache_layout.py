# SPDX-License-Identifier: Apache-2.0
"""Registered memory descriptors for one KV cache group."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GroupBufferLayout:
    base_addresses: tuple[int, ...]
    block_lengths: tuple[int, ...]
    block_strides: tuple[int, ...]
    segments: tuple[tuple[int, int], ...]
    layer_offsets: tuple[int, ...] | None = None
    num_layers: int = 1

    @classmethod
    def from_buffers(
        cls,
        base_addresses: list[int],
        block_lengths: list[int],
        block_strides: list[int] | None = None,
        segments: list[tuple[int, int]] | None = None,
        layer_offsets: list[int] | None = None,
        num_layers: int = 1,
    ) -> "GroupBufferLayout":
        # Broadcast lengths/strides once, rather than during every transfer.
        lengths = (
            tuple(block_lengths[index % len(block_lengths)] for index in range(len(base_addresses)))
            if block_lengths
            else ()
        )
        strides = (
            tuple(block_strides[index % len(block_lengths)] for index in range(len(base_addresses)))
            if block_strides and block_lengths
            else lengths
        )
        return cls(
            tuple(base_addresses),
            lengths,
            strides,
            tuple(segments) if segments else ((0, 1),) * len(base_addresses),
            tuple(layer_offsets) if layer_offsets is not None else None,
            num_layers,
        )

    def layer_range(self, layer_id: int) -> tuple[int, int]:
        if self.layer_offsets is not None:
            return self.layer_offsets[layer_id], self.layer_offsets[layer_id + 1]
        entries_per_layer = len(self.base_addresses) // self.num_layers if self.num_layers else 0
        start = layer_id * entries_per_layer
        return start, start + entries_per_layer

    def address_spans(
        self,
        start: int,
        end: int,
        block_id: int,
        block_size: int,
        entry_start: int = 0,
        entry_end: int | None = None,
    ) -> tuple[list[int], list[int]]:
        addresses: list[int] = []
        sizes: list[int] = []
        if not self.block_lengths:
            return addresses, sizes
        block_start = start // block_size * block_size
        start_in_block, end_in_block = start - block_start, end - block_start
        for index in range(entry_start, len(self.base_addresses) if entry_end is None else entry_end):
            segment_index, segment_count = self.segments[index]
            segment_tokens = block_size // segment_count
            segment_start = segment_index * segment_tokens
            copy_start = max(start_in_block, segment_start)
            copy_end = min(end_in_block, segment_start + segment_tokens)
            if copy_start >= copy_end:
                continue
            byte_start = self.block_lengths[index] * (copy_start - segment_start) // segment_tokens
            byte_end = self.block_lengths[index] * (copy_end - segment_start) // segment_tokens
            addresses.append(self.base_addresses[index] + block_id * self.block_strides[index] + byte_start)
            sizes.append(byte_end - byte_start)
        return addresses, sizes
