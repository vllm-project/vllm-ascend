# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit
def _fill_local_kv(
    k_src,
    v_src,
    k_dst,
    v_dst,
    topk,
    slots,
    stable_prefix,
    token_to_req,
    query_ends,
    req_ids,
    source_slots,
    TOPK: tl.constexpr,
    CAPACITY: tl.constexpr,
    K_WIDTH: tl.constexpr,
    V_WIDTH: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    NUM_REQS: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    req_id = tl.load(req_ids + row)
    req = tl.load(token_to_req + row)
    if (req_id != 0) & (req >= 0) & (req < NUM_REQS):
        start = tl.load(query_ends + req - 1, mask=req > 0, other=0)
        end = tl.load(query_ends + req)
        prefix = tl.load(stable_prefix + row)
        positions = tl.arange(0, TOPK_BLOCK)
        selected = tl.load(topk + row * TOPK + positions, positions < TOPK, other=-1)
        resident = tl.load(slots + row * TOPK + positions, positions < TOPK, other=-1)
        offsets = tl.arange(0, KV_BLOCK)
        # A query row can attend earlier fresh rows of the SAME request (MTP).
        # Activation rows use packed batch offsets, never request-local indices.
        for source in range(start, tl.minimum(end, NUM_ROWS)):
            physical_slot = tl.load(source_slots + source)
            token = prefix + source - start
            matches = (selected == token) & (resident >= 0) & (resident < CAPACITY)
            dst_slot = tl.min(tl.where(matches, resident, CAPACITY), 0)
            if physical_slot >= 0 and dst_slot < CAPACITY:
                k = tl.load(k_src + source * K_WIDTH + offsets, offsets < K_WIDTH, other=0)
                v = tl.load(v_src + source * V_WIDTH + offsets, offsets < V_WIDTH, other=0)
                tl.store(
                    k_dst + (row * CAPACITY + dst_slot) * K_WIDTH + offsets,
                    k,
                    offsets < K_WIDTH,
                )
                tl.store(
                    v_dst + (row * CAPACITY + dst_slot) * V_WIDTH + offsets,
                    v,
                    offsets < V_WIDTH,
                )


def fill_local_kv(
    k,
    v,
    resident_k,
    resident_v,
    topk,
    slots,
    stable_prefix,
    token_to_req,
    query_ends,
    req_ids,
    source_slots,
    capacity,
):
    rows, topk_size = topk.shape
    if rows == 0:
        return
    if token_to_req is None or query_ends is None:
        raise ValueError("Local KV fill requires request mapping and query boundaries")
    k_width, v_width = k.shape[-1], v.shape[-1]
    _fill_local_kv[(rows,)](
        k,
        v,
        resident_k,
        resident_v,
        topk,
        slots,
        stable_prefix,
        token_to_req,
        query_ends,
        req_ids,
        source_slots,
        TOPK=topk_size,
        CAPACITY=capacity,
        K_WIDTH=k_width,
        V_WIDTH=v_width,
        NUM_ROWS=rows,
        NUM_REQS=query_ends.numel(),
        TOPK_BLOCK=triton.next_power_of_2(topk_size),
        KV_BLOCK=triton.next_power_of_2(max(k_width, v_width)),
    )
