# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.models.hy_v4_flash.reference import INDEX_DIM
from vllm.triton_utils import tl, triton

_GROUPS_PER_PROGRAM = 16


@triton.jit(do_not_specialize=["first_group", "num_groups"])
def _copy_pool_keys(
    cache,
    block_table,
    keys,
    first_group,
    num_groups,
    end_group,
    ROW_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    LANES_PER_ROW: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUPS: tl.constexpr,
    WRITE: tl.constexpr,
    DEVICE_START: tl.constexpr,
    DEVICE_END: tl.constexpr,
):
    row = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    lane = tl.arange(0, WIDTH)
    start = tl.load(first_group) if DEVICE_START else first_group
    group = row + start
    live = row < num_groups
    if DEVICE_END:
        live = live & (group < tl.load(end_group))
    token = group * RATIO
    block = tl.load(block_table + token // BLOCK_SIZE, mask=live, other=0).to(tl.int64)
    slot = block * BLOCK_SIZE + token % BLOCK_SIZE
    # Spread each key across the spare lanes of its group's token rows.
    address = (
        (slot[:, None] + lane[None, :] // LANES_PER_ROW) * ROW_WIDTH
        + ROW_WIDTH
        - LANES_PER_ROW
        + lane[None, :] % LANES_PER_ROW
    )
    key_address = row[:, None] * WIDTH + lane[None, :]
    mask = live[:, None]
    if WRITE:
        values = tl.load(keys + key_address, mask=mask, other=0)
        tl.store(cache + address, values, mask=mask)
    else:
        values = tl.load(cache + address, mask=mask, other=0).to(tl.float32)
        tl.store(keys + key_address, values, mask=row[:, None] < num_groups)


def copy_pool_keys(cache, block_table, keys, first_group, block_size, ratio, *, write, end_group=None):
    # The rounded E4M3 values have power-of-two row scales and fit exactly in BF16.
    cache = cache.view(torch.bfloat16)
    _copy_pool_keys[(triton.cdiv(keys.shape[0], _GROUPS_PER_PROGRAM),)](
        cache,
        block_table,
        keys,
        first_group,
        keys.shape[0],
        end_group,
        cache.shape[-1],
        block_size,
        ratio,
        INDEX_DIM // ratio,
        INDEX_DIM,
        _GROUPS_PER_PROGRAM,
        write,
        isinstance(first_group, torch.Tensor),
        end_group is not None,
        num_warps=1,
    )
