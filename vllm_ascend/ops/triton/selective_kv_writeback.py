# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["src_k", "src_v", "dst_k", "dst_v"])
def build_d2h_descriptors(
    slots,
    sources,
    destinations,
    lengths,
    count,
    src_k,
    src_v,
    dst_k,
    dst_v,
    ROWS: tl.constexpr,
    CAPACITY: tl.constexpr,
    K_BYTES: tl.constexpr,
    V_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tile * BLOCK + tl.arange(0, BLOCK)
    in_range = rows < ROWS
    slot = tl.load(slots + rows, in_range, other=-1).to(tl.int64)
    valid = (slot >= 0) & (slot < CAPACITY)
    slot = tl.where(valid, slot, 0)
    tl.store(sources + rows, src_k + rows.to(tl.int64) * K_BYTES, in_range)
    tl.store(sources + ROWS + rows, src_v + rows.to(tl.int64) * V_BYTES, in_range)
    tl.store(destinations + rows, dst_k + slot * K_BYTES, in_range)
    tl.store(destinations + ROWS + rows, dst_v + slot * V_BYTES, in_range)
    tl.store(lengths + rows, tl.where(valid, K_BYTES, 0), in_range)
    tl.store(lengths + ROWS + rows, tl.where(valid, V_BYTES, 0), in_range)
    if tile == 0:
        tl.store(count, 2 * ROWS)
