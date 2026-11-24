# SPDX-License-Identifier: Apache-2.0
"""Read the saved key and coordinates for a partial IndexPool group."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["first_token"])
def _read_index_history(
    cache,
    table,
    keys,
    positions,
    first_token,
    PAGE_ROWS: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    KEY_OFFSET: tl.constexpr,
    KEY_WORDS: tl.constexpr,
):
    row = tl.program_id(0)
    token = first_token + row
    page = tl.load(table + token // PAGE_ROWS).to(tl.int32)
    slot = page * PAGE_ROWS + token % PAGE_ROWS
    source = cache + slot * ROW_WORDS + KEY_OFFSET
    key = tl.arange(0, KEY_WORDS)
    tl.store(keys + row * KEY_WORDS + key, tl.load(source + key))
    coord = tl.arange(0, 8)
    valid_coord = coord < 6
    coord_words = tl.load(source + KEY_WORDS + coord, valid_coord, other=0)
    tl.store(positions + row * 6 + coord, coord_words, valid_coord)


def read_index_history(cache, table, block_size, first_token, count, key_offset, position_ndim):
    keys = torch.empty((count, 128), dtype=torch.bfloat16, device=cache.device)
    positions = torch.empty((count, 3), dtype=torch.int64, device=cache.device)
    _read_index_history[(count,)](
        cache.view(torch.int32),
        table,
        keys.view(torch.int32),
        positions.view(torch.int32),
        first_token,
        block_size,
        cache.size(1) // 4,
        key_offset // 4,
        64,
    )
    return keys, positions[:, 0] if position_ndim == 1 else positions.T
