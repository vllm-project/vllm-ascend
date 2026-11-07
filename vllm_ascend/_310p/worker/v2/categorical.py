# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from torch.nn.functional import pad

from vllm_ascend.utils import global_stream, npu_stream_switch

_UNIFORM_BLOCK_SIZE = 256
_GENERATOR_OFFSET_PER_DRAW = 4
_VOCAB_TILE = 128
_TILED_MIN_ELEMENTS = 2**22
_MIN_UNIFORM = torch.finfo(torch.float32).tiny
_UINT64_MASK = (1 << 64) - 1
# Reserve a seed domain for the uniform blocks used by ordinary sampling.
_UNIFORM_SEED_DOMAIN = 0xD1B54A32D192ED03


def _block_seed(seed: int, block: int) -> int:
    """Mix request seed and block number without drawing host random values."""
    value = (seed + _UNIFORM_SEED_DOMAIN + block * 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def sample_from_uniform_310p(probs: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    vocab = probs.shape[-1]
    if probs.numel() < _TILED_MIN_ELEMENTS:
        prefix = probs.cumsum(-1)
        thresholds = (uniforms * prefix[:, -1:]).clamp_min_(_MIN_UNIFORM)
        tokens = torch.searchsorted(prefix, thresholds, out_int32=True)
        # A failed search (including a zero-mass row) returns the vocabulary size.
        return tokens.masked_fill_(tokens >= vocab, 0).flatten()

    batch, vocab = probs.shape
    tiles = pad(probs, (0, -vocab % _VOCAB_TILE)).view(batch, -1, _VOCAB_TILE)
    prefix = tiles.sum(-1).cumsum(-1)
    thresholds = (uniforms * prefix[:, -1:]).clamp_min_(_MIN_UNIFORM)
    tile = torch.searchsorted(prefix, thresholds)
    invalid_tile = tile >= tiles.shape[1]
    # Never let a failed search address a tile belonging to another request.
    tile.clamp_max_(tiles.shape[1] - 1)
    previous = prefix.gather(1, (tile - 1).clamp_min(0)).masked_fill_(tile == 0, 0)
    row_offsets = torch.arange(batch, device=probs.device, dtype=torch.int32) * tiles.shape[1]
    tile_rows = row_offsets + tile.flatten().to(torch.int32)
    local_prefix = tiles.flatten(0, 1).index_select(0, tile_rows).cumsum(-1)
    local_threshold = torch.minimum((thresholds - previous).clamp_min_(_MIN_UNIFORM), local_prefix[:, -1:])
    local_token = torch.searchsorted(local_prefix, local_threshold, out_int32=True)
    tokens = tile.to(torch.int32) * _VOCAB_TILE + local_token
    invalid = invalid_tile | (local_token >= _VOCAB_TILE) | (tokens >= vocab)
    return tokens.masked_fill_(invalid, 0).flatten()


class CategoricalSampler310:
    """Use NPU uniforms, amortizing RNG launches for independent request seeds."""

    def __init__(self) -> None:
        self._blocks: dict[tuple[int, int], torch.Tensor] = {}

    def _draw(self, batch: int, device: torch.device, generators: dict[int, torch.Generator]) -> torch.Tensor:
        if not generators:
            self._blocks.clear()
            return torch.rand((batch, 1), device=device, dtype=torch.float32)

        blocks: dict[tuple[int, int], torch.Tensor] = {}
        block_rows: dict[tuple[int, int], int] = {}
        offsets: dict[torch.Generator, int] = {}
        indices = [0] * batch
        for row, generator in generators.items():
            offset = offsets.get(generator)
            if offset is None:
                offset = generator.get_offset()
            offsets[generator] = offset + _GENERATOR_OFFSET_PER_DRAW
            block, position = divmod(offset // _GENERATOR_OFFSET_PER_DRAW, _UNIFORM_BLOCK_SIZE)
            seed = generator.initial_seed()
            key = (seed, block)
            if key not in blocks:
                values = self._blocks.get(key)
                if values is None:
                    rng = torch.Generator(device=device).manual_seed(_block_seed(seed, block))
                    values = torch.rand((_UNIFORM_BLOCK_SIZE,), device=device, dtype=torch.float32, generator=rng)
                block_rows[key] = len(blocks)
                blocks[key] = values
            indices[row] = block_rows[key] * _UNIFORM_BLOCK_SIZE + position

        values = next(iter(blocks.values())) if len(blocks) == 1 else torch.cat(tuple(blocks.values()))
        if len(generators) != batch:
            # Append independent default draws and gather in original row order.
            values = torch.cat((values, torch.rand((batch,), device=device, dtype=torch.float32)))
            start = len(blocks) * _UNIFORM_BLOCK_SIZE
            for row in range(batch):
                if row not in generators:
                    indices[row] = start + row
        index = torch.tensor(indices, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        uniforms = values.index_select(0, index).view(batch, 1)
        for generator, offset in offsets.items():
            generator.set_offset(offset)
        self._blocks = blocks
        return uniforms

    def __call__(self, probs: torch.Tensor, generators: dict[int, torch.Generator]) -> torch.Tensor:
        with npu_stream_switch(global_stream()):
            uniforms = self._draw(probs.shape[0], probs.device, generators)
        stream = torch.npu.current_stream()
        stream.wait_stream(global_stream())
        uniforms.record_stream(stream)
        return sample_from_uniform_310p(probs, uniforms).to(torch.int64)
