# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

_VOCAB_TILE_SIZE = 2048


@triton.jit(do_not_specialize=["num_rows"])
def _write_vocab_parallel_logits(
    gathered_ptr,
    output_ptr,
    num_rows,
    VOCAB_SIZE: tl.constexpr,
    SHARD_SIZE: tl.constexpr,
    TILES_PER_SHARD: tl.constexpr,
    TILES_PER_ROW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    lane = tl.arange(0, BLOCK_SIZE)
    for tile in range(tl.program_id(0), num_rows * TILES_PER_ROW, tl.num_programs(0)):
        row = (tile // TILES_PER_ROW).to(tl.int64)
        vocab_tile = tile % TILES_PER_ROW
        rank = vocab_tile // TILES_PER_SHARD
        shard_col = (vocab_tile % TILES_PER_SHARD) * BLOCK_SIZE + lane
        vocab_col = rank * SHARD_SIZE + shard_col
        valid = (shard_col < SHARD_SIZE) & (vocab_col < VOCAB_SIZE)
        # Each tile is contiguous within one rank's vocabulary shard.
        values = tl.load(gathered_ptr + (rank * num_rows + row) * SHARD_SIZE + shard_col, valid, other=0)
        tl.store(output_ptr + row * VOCAB_SIZE + vocab_col, values.to(output_ptr.dtype.element_ty), valid)


def pack_vocab_parallel_logits(
    gathered: torch.Tensor, num_rows: int, vocab_size: int, out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Pack rank-ordered shards, trimming padding and casting on the output store."""
    shard_size = gathered.shape[1]
    tiles_per_shard = triton.cdiv(shard_size, _VOCAB_TILE_SIZE)
    full_shards, tail = divmod(vocab_size, shard_size)
    tiles_per_row = full_shards * tiles_per_shard + triton.cdiv(tail, _VOCAB_TILE_SIZE)
    output = gathered.new_empty((num_rows, vocab_size), dtype=out_dtype)
    if num_rows == 0:
        return output
    grid = (min(num_rows * tiles_per_row, get_vectorcore_num()),)
    _write_vocab_parallel_logits[grid](
        gathered, output, num_rows, vocab_size, shard_size, tiles_per_shard, tiles_per_row, _VOCAB_TILE_SIZE
    )
    return output
