# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU packing and reference reconstruction of lossless StateTree weights.

The record layout is adapted from StateTree-Representation:
https://github.com/ammarali32/StateTree-Representation
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_ascend.ops.statetree.config import choose_tiles, record_sizes, validate_tiles


@dataclass(frozen=True)
class PackedWeight:
    """One contiguous packed allocation plus immutable matrix metadata."""

    payload: torch.Tensor
    n: int
    k: int
    n_tile: int
    k_tile: int
    escape_slots: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.n, self.k

    @property
    def device(self) -> torch.device:
        return self.payload.device

    @property
    def nbytes(self) -> int:
        return self.payload.numel() * self.payload.element_size()

    @property
    def _tile_shape(self) -> tuple[int, int]:
        return (self.n + self.n_tile - 1) // self.n_tile, self.k // self.k_tile

    def _views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_tiles, k_tiles = self._tile_shape
        unit_count = n_tiles * k_tiles * 2
        shapes = (
            (n_tiles, k_tiles, 2, self.k_tile // 16, self.n_tile // 2, 16),
            (n_tiles, k_tiles, 2, self.k_tile // 16, self.n_tile // 2, 8),
            (n_tiles, k_tiles, 2, self.k_tile // 64, self.n_tile // 2),
            (unit_count, 2, self.escape_slots),
        )
        dtypes = (torch.uint8, torch.uint8, torch.int16, torch.int32)
        views: list[torch.Tensor] = []
        offset = 0
        for size, shape, dtype in zip(
            record_sizes(self.n_tile, self.k_tile, self.escape_slots),
            shapes,
            dtypes,
        ):
            view = self.payload[:, offset : offset + size].view(dtype).view(shape)
            views.append(view)
            offset += size
        return views[0], views[1], views[2], views[3]

    @property
    def leaves(self) -> torch.Tensor:
        return self._views()[0]

    @property
    def states(self) -> torch.Tensor:
        return self._views()[1]

    @property
    def nodes(self) -> torch.Tensor:
        return self._views()[2]

    @property
    def escapes(self) -> torch.Tensor:
        return self._views()[3]

    def to(self, device: torch.device | str) -> PackedWeight:
        return PackedWeight(
            self.payload.to(device),
            self.n,
            self.k,
            self.n_tile,
            self.k_tile,
            self.escape_slots,
        )

    def validate(self) -> None:
        validate_tiles(16, self.n_tile, self.k_tile, self.escape_slots)
        if self.n <= 0 or self.k <= 0 or self.n % 16 or self.k % self.k_tile:
            raise ValueError("invalid StateTree matrix dimensions")
        n_tiles, k_tiles = self._tile_shape
        expected_shape = (
            n_tiles * k_tiles * 2,
            sum(record_sizes(self.n_tile, self.k_tile, self.escape_slots)),
        )
        if (
            tuple(self.payload.shape) != expected_shape
            or self.payload.dtype != torch.uint8
            or not self.payload.is_contiguous()
        ):
            raise ValueError("invalid StateTree packed payload")

    @classmethod
    def from_components(
        cls,
        leaves: torch.Tensor,
        states: torch.Tensor,
        nodes: torch.Tensor,
        escapes: torch.Tensor,
        n: int,
        k: int,
        n_tile: int,
        k_tile: int,
        escape_slots: int,
    ) -> PackedWeight:
        n_tiles = (n + n_tile - 1) // n_tile
        k_tiles = k // k_tile
        unit_count = n_tiles * k_tiles * 2
        sizes = record_sizes(n_tile, k_tile, escape_slots)
        payload = torch.empty((unit_count, sum(sizes)), dtype=torch.uint8)
        offset = 0
        for tensor, size in zip((leaves, states, nodes, escapes), sizes):
            payload[:, offset : offset + size].copy_(tensor.contiguous().view(torch.uint8).reshape(unit_count, size))
            offset += size
        packed = cls(payload, n, k, n_tile, k_tile, escape_slots)
        packed.validate()
        return packed


@torch.no_grad()
def pack(
    weight: torch.Tensor,
    n_tile: int | None = None,
    k_tile: int | None = None,
    device: torch.device | str = "npu",
) -> PackedWeight:
    """Losslessly encode a two-dimensional BF16 weight."""
    if weight.dtype != torch.bfloat16 or weight.ndim != 2 or weight.numel() == 0:
        raise ValueError("StateTree pack expects a nonempty two-dimensional BF16 weight")
    n, k = map(int, weight.shape)
    if n_tile is None and k_tile is None:
        n_tile, k_tile = choose_tiles(n, k)
    elif n_tile is None or k_tile is None:
        raise ValueError("N and K tiles must be specified together")
    validate_tiles(16, n_tile, k_tile, 8)
    if n % 16 or k % k_tile:
        raise ValueError("N must be aligned to 16 and K must be divisible by the K tile")

    n_tiles = (n + n_tile - 1) // n_tile
    k_tiles = k // k_tile
    leaves = torch.empty((n_tiles, k_tiles, 2, k_tile // 16, n_tile // 2, 16), dtype=torch.uint8)
    states = torch.empty((n_tiles, k_tiles, 2, k_tile // 16, n_tile // 2, 8), dtype=torch.uint8)
    nodes = torch.empty((n_tiles, k_tiles, 2, k_tile // 64, n_tile // 2), dtype=torch.int16)
    patches_by_chunk: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    max_count = 0

    def tiled(tensor: torch.Tensor, divisor: int) -> torch.Tensor:
        tile_count = tensor.shape[0] // n_tile
        tile = tensor.reshape(tile_count, 2, n_tile // 2, k_tiles, k_tile // divisor).permute(0, 3, 1, 2, 4)
        if divisor == 64:
            return tile.transpose(-1, -2).contiguous()
        return (
            tile.reshape(tile_count, k_tiles, 2, n_tile // 2, k_tile // 16, 16 // divisor)
            .transpose(-2, -3)
            .contiguous()
        )

    # Exact low BF16 byte + (sign, three-bit high-exponent delta). Pairs whose
    # exponent range exceeds three bits carry an exact patch. Work in bounded
    # chunks so large output heads do not create full-matrix int32 temporaries.
    for tile_start in range(0, n_tiles, 4):
        tile_end = min(n_tiles, tile_start + 4)
        rows_in_chunk = (tile_end - tile_start) * n_tile
        raw = (
            weight[tile_start * n_tile : tile_end * n_tile]
            .detach()
            .cpu()
            .contiguous()
            .view(torch.int16)
            .to(torch.int32)
            & 65535
        )
        if raw.shape[0] < rows_in_chunk:
            raw = torch.nn.functional.pad(raw, (0, 0, 0, rows_in_chunk - raw.shape[0]))
        high = ((raw >> 8) & 127).reshape(rows_in_chunk, k // 64, 64)
        exponent = high.amax(-1)
        delta = (exponent[..., None] - high).reshape(rows_in_chunk, k)
        state = (delta & 7) | ((raw >> 12) & 8)
        leaves[tile_start:tile_end] = tiled((raw & 255).to(torch.uint8), 1)
        states[tile_start:tile_end] = tiled(
            (state[:, 0::2] | (state[:, 1::2] << 4)).to(torch.uint8),
            2,
        )
        nodes[tile_start:tile_end] = tiled((exponent * 257).to(torch.int16), 64)

        exceptional = (delta[:, 0::2] > 7) | (delta[:, 1::2] > 7)
        mask = tiled(exceptional, 2).reshape(-1, n_tile * k_tile // 4)
        units, positions = mask.nonzero(as_tuple=True)
        counts = torch.bincount(units, minlength=mask.shape[0])
        max_count = max(max_count, int(counts.max()))
        starts = counts.cumsum(0) - counts
        slot = torch.arange(units.numel()) - starts[units]
        original = tiled(raw, 1).reshape(-1, n_tile * k_tile // 2)
        pairs = original[:, 0::2] | (original[:, 1::2] << 16)
        patches_by_chunk.append(
            (
                units + tile_start * k_tiles * 2,
                positions.to(torch.int32),
                slot,
                pairs[units, positions],
            )
        )

    escape_slots = max(8, 1 << (max(1, max_count) - 1).bit_length())
    validate_tiles(16, n_tile, k_tile, escape_slots)
    escapes = torch.full((n_tiles * k_tiles * 2, 2, escape_slots), -1, dtype=torch.int32)
    for units, positions, slot, values in patches_by_chunk:
        escapes[units, 0, slot] = positions
        escapes[units, 1, slot] = values
    return PackedWeight.from_components(
        leaves,
        states,
        nodes,
        escapes,
        n,
        k,
        n_tile,
        k_tile,
        escape_slots,
    ).to(device)


def _unpack_full_tiles(packed: PackedWeight) -> torch.Tensor:
    def untile(tensor: torch.Tensor, divisor: int) -> torch.Tensor:
        if divisor == 64:
            tile = tensor.cpu().transpose(-1, -2)
        else:
            tile = (
                tensor.cpu()
                .transpose(-2, -3)
                .reshape(
                    packed.n // packed.n_tile,
                    packed.k // packed.k_tile,
                    2,
                    packed.n_tile // 2,
                    packed.k_tile // divisor,
                )
            )
        return tile.permute(0, 2, 3, 1, 4).contiguous().reshape(packed.n, packed.k // divisor)

    leaves = packed.leaves.cpu().to(torch.int32)
    states = packed.states.cpu().to(torch.int32)
    delta = torch.stack((states & 15, states >> 4), -1).reshape(leaves.shape)
    exponent = (packed.nodes.cpu().to(torch.int32) & 255)[..., None, :, None]
    exponent = exponent.expand(
        packed.n // packed.n_tile,
        packed.k // packed.k_tile,
        2,
        packed.k_tile // 64,
        4,
        packed.n_tile // 2,
        16,
    ).reshape(leaves.shape)
    high = ((exponent - (delta & 7)) & 127) | ((delta & 8) << 4)
    bits = leaves | (high << 8)
    flat = bits.reshape(-1, packed.n_tile // 2 * packed.k_tile)
    pairs = flat[:, 0::2] | (flat[:, 1::2] << 16)
    escapes = packed.escapes.cpu()
    units, slots = (escapes[:, 0] >= 0).nonzero(as_tuple=True)
    pairs[units, escapes[units, 0, slots].long()] = escapes[units, 1, slots]
    bits = torch.stack((pairs & 65535, pairs >> 16), -1).to(torch.int16).reshape(leaves.shape)
    return untile(bits, 1).view(torch.bfloat16)


def unpack_reference(packed: PackedWeight) -> torch.Tensor:
    """Decode on CPU using bounded intermediates as an independent oracle."""
    packed.validate()
    result = torch.empty((packed.n, packed.k), dtype=torch.bfloat16)
    units_per_tile = packed.k // packed.k_tile * 2
    n_tiles = (packed.n + packed.n_tile - 1) // packed.n_tile
    for start in range(0, n_tiles, 4):
        end = min(n_tiles, start + 4)
        chunk = PackedWeight(
            packed.payload[start * units_per_tile : end * units_per_tile].cpu().contiguous(),
            (end - start) * packed.n_tile,
            packed.k,
            packed.n_tile,
            packed.k_tile,
            packed.escape_slots,
        )
        rows = min(packed.n, end * packed.n_tile) - start * packed.n_tile
        result[start * packed.n_tile : start * packed.n_tile + rows] = _unpack_full_tiles(chunk)[:rows]
    return result
