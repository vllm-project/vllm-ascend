# SPDX-License-Identifier: Apache-2.0
"""A3 MTP5 QLI with V25's six-row score pipeline and balanced tile ranges."""

from functools import lru_cache, partial

from qli_tilelang import _EXTRACT, _INIT_TOPK
from qli_tilelang_v1 import _MASK_TILE
from qli_tilelang_v13 import (
    _PACK_WEIGHT,
    _QK_FIXPIPE_PACKED,
    _STORE_WEIGHT,
    BATCH,
    CONTEXT,
    CORES,
    PAGE_SIZE,
    ROWS_PER_REQUEST,
)
from qli_tilelang_v17 import _KEY_EVENT_BASE, _key_panel_source
from qli_tilelang_v18 import _PRUNED_TOPK
from qli_tilelang_v25 import _QK_MMAD, _score_ring
from qli_tilelang_v25 import build_tilelang_call as build_v25_call

TILE_SIZE = 2048
TILES_PER_REQUEST = CONTEXT // TILE_SIZE
NATIVE_QUERY_GROUP = 4
NATIVE_GROUPS = (ROWS_PER_REQUEST + NATIVE_QUERY_GROUP - 1) // NATIVE_QUERY_GROUP
Q_HEADS = 32
HEAD_DIM = 128
PAGE_SLOTS = CONTEXT // PAGE_SIZE
_SCALE_EVENT = 6
_PACKED_KEY_PANEL = partial(_key_panel_source, request="request", first_tile="first_tile")
_PACK_WEIGHT_FOR_ROW = _PACK_WEIGHT.replace("packed[row * 32]", "packed[score_mask_values.GetValue(6) * 32]")
_STORE_WEIGHT_FOR_ROW = _STORE_WEIGHT.replace("+ row)", "+ score_mask_values.GetValue(6))")
_GATHER_SCALE = f"""
{{
    using namespace AscendC;
    constexpr uint32_t pages_per_tile = {TILE_SIZE // PAGE_SIZE};
    constexpr uint32_t page_size = {PAGE_SIZE};
    const uint32_t offset = score_mask_values.GetValue(6) * {PAGE_SLOTS}
        + score_mask_values.GetValue(7) * pages_per_tile;
    for (uint32_t page = 0; page < pages_per_tile; ++page) {{
        const uint64_t physical = table.GetValue(offset + page);
        SetFlag<HardEvent::S_MTE2>({_SCALE_EVENT});
        WaitFlag<HardEvent::S_MTE2>({_SCALE_EVENT});
        DataCopy(half_scale[page * page_size], kscale[physical * page_size], page_size);
    }}
    SetFlag<HardEvent::MTE2_V>({_SCALE_EVENT});
}}
"""


def _schedule(batch):
    quotient, remainder = divmod(batch * TILES_PER_REQUEST, CORES)
    if quotient == 0:
        raise ValueError("The packed schedule needs at least one tile per Cube")
    return quotient, remainder


def _small_batch_merge_source(batch):
    def bounds(total):
        quotient, remainder = divmod(total, CORES)
        return tuple(core * quotient + min(core, remainder) for core in range(CORES + 1))

    packed = bounds(batch * TILES_PER_REQUEST)
    native = bounds(batch * NATIVE_GROUPS * TILES_PER_REQUEST)
    slot_stride = batch + 2 * (CORES - 1)
    group_offsets = [0]
    list_slots = []
    for request in range(batch):
        request_start = request * TILES_PER_REQUEST
        for group in range(NATIVE_GROUPS):
            group_start = (request * NATIVE_GROUPS + group) * TILES_PER_REQUEST
            for native_core in range(CORES):
                start = max(0, native[native_core] - group_start) + request_start
                stop = min(TILES_PER_REQUEST, native[native_core + 1] - group_start) + request_start
                for packed_core in reversed(range(CORES)):
                    if max(start, packed[packed_core]) < min(stop, packed[packed_core + 1]):
                        slot = group * slot_stride + request + packed_core + native_core
                        list_slots.append(slot * NATIVE_QUERY_GROUP)
            group_offsets.append(len(list_slots))

    offsets_cpp = ", ".join(map(str, group_offsets))
    slots_cpp = ", ".join(map(str, list_slots))
    return f"""
{{
    using namespace AscendC;
    constexpr uint32_t k = {TILE_SIZE};
    constexpr uint32_t pairs = 2 * k;
    constexpr uint32_t copy_event = 7;
    constexpr uint16_t offsets[] = {{{offsets_cpp}}};
    constexpr uint16_t slots[] = {{{slots_cpp}}};
    const uint32_t qrow = GetBlockIdx();
    const uint32_t row = qrow % {ROWS_PER_REQUEST};
    const uint32_t group = qrow / {ROWS_PER_REQUEST} * {NATIVE_GROUPS} + row / {NATIVE_QUERY_GROUP};
    const uint32_t group_row = row % {NATIVE_QUERY_GROUP};
    const uint32_t begin = offsets[group];
    const uint32_t end = offsets[group + 1];
    DataCopy(topk, partial_topk[(slots[begin] + group_row) * pairs], pairs);
    for (uint32_t next = begin + 1; next < end; next += 3) {{
        const uint32_t count = end - next < 3 ? end - next : 3;
        SetFlag<HardEvent::V_MTE2>(copy_event);
        WaitFlag<HardEvent::V_MTE2>(copy_event);
        for (uint32_t i = 0; i < count; ++i) {{
            DataCopy(row_topk[i * pairs], partial_topk[(slots[next + i] + group_row) * pairs], pairs);
        }}
        SetFlag<HardEvent::MTE2_V>(copy_event);
        WaitFlag<HardEvent::MTE2_V>(copy_event);
        MrgSort4Info params;
        params.elementLengths[0] = k;
        params.elementLengths[1] = k;
        params.elementLengths[2] = k;
        params.elementLengths[3] = k;
        // At first exhaustion at least k pairs have been emitted, since
        // each valid run is k pairs. The output can hold up to all four.
        params.ifExhaustedSuspension = true;
        params.validBit = (1U << (count + 1U)) - 1U;
        params.repeatTimes = 1;
        MrgSortSrcList<float> runs(topk, row_topk[0], row_topk[pairs], row_topk[2 * pairs]);
        MrgSort<float>(merge_scratch, runs, params);
        PipeBarrier<PIPE_V>();
        Adds(topk.ReinterpretCast<int32_t>(), merge_scratch.ReinterpretCast<int32_t>(), 0, pairs);
        PipeBarrier<PIPE_V>();
    }}
}}
"""


@lru_cache(maxsize=128)
def _compile(batch, pages, four_way_merge=False, score_load_2d=False):
    import tilelang
    from tilelang import language as T

    quotient, remainder = _schedule(batch)
    first_run = (quotient + 1) * remainder
    native_quotient, native_remainder = divmod(batch * NATIVE_GROUPS * TILES_PER_REQUEST, CORES)
    native_first_run = (native_quotient + 1) * native_remainder
    tile_n = TILE_SIZE
    pair_count = 2 * tile_n
    packed_heads = ROWS_PER_REQUEST * Q_HEADS
    cube_n = PAGE_SIZE
    aiv_rows = ROWS_PER_REQUEST // 2
    score_slots = 2
    merge_workers = 2 * CORES
    merge_iterations = (batch * ROWS_PER_REQUEST + merge_workers - 1) // merge_workers
    small_batch_merge = _small_batch_merge_source(batch) if four_way_merge else ""
    ring_init, ring_tile, ring_drain = _score_ring(
        _QK_MMAD,
        _QK_FIXPIPE_PACKED,
        key_panel_source=_PACKED_KEY_PANEL,
        score_load_2d=score_load_2d,
    )
    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }

    @tilelang.jit(
        out_idx=[6],
        workspace_idx=[7, 8, 9],
        target="ascendc",
        platform="A3",
        pass_configs=pass_configs,
    )
    def stage_one():
        @T.macro
        def tile_owner(tile, owner):
            owner = T.if_then_else(  # noqa: F841
                tile < first_run, tile // (quotient + 1), remainder + (tile - first_run) // quotient
            )

        @T.macro
        def native_owner(tile, owner):
            owner = T.if_then_else(  # noqa: F841
                tile < native_first_run,
                tile // (native_quotient + 1),
                native_remainder + (tile - native_first_run) // native_quotient,
            )

        @T.macro
        def cube_tiles(query, key, table, weights_out, score_ws, cid, from_job, to_job):
            query_l1 = T.alloc_L1((packed_heads, HEAD_DIM), "int8")
            query_a = T.alloc_L0A((packed_heads, HEAD_DIM), "int8")
            key_l1 = T.alloc_L1((2, cube_n, HEAD_DIM), "int8")
            key_b = T.alloc_L0B((HEAD_DIM, cube_n), "int8")
            score_b = T.alloc_L0B((packed_heads, cube_n), "float16")
            qk_l0c = T.alloc_L0C((packed_heads, cube_n), "int32")
            qk_l1 = T.alloc_L1((2, packed_heads, cube_n), "float16")
            weight_l1 = T.alloc_L1((ROWS_PER_REQUEST, 16, Q_HEADS), "float16")
            weight_a = T.alloc_L0A((ROWS_PER_REQUEST, 16, Q_HEADS), "float16")
            score_l0c = T.alloc_L0C((16, cube_n), "float32")

            for req in T.serial(from_job // TILES_PER_REQUEST, T.ceildiv(to_job, TILES_PER_REQUEST)):
                request = T.alloc_var("int32", req)
                first_tile = T.alloc_var("int32", T.max(from_job - request * TILES_PER_REQUEST, 0))
                stop_tile = T.alloc_var("int32", T.min(to_job - request * TILES_PER_REQUEST, TILES_PER_REQUEST))
                tile_count = T.alloc_var("int32", stop_tile - first_tile)
                query_base = T.alloc_var("int32", request * packed_heads)
                T.copy(query[query_base : query_base + packed_heads, :], query_l1)
                T.copy(query_l1, query_a)
                T.wait_cross_flag(0)
                for row in T.serial(ROWS_PER_REQUEST):
                    T.copy(weights_out[cid, row, :, :], weight_l1[row, :, :])
                for row in T.serial(ROWS_PER_REQUEST):
                    T.copy(weight_l1[row, :, :], weight_a[row, :, :])
                T.barrier_all()

                T.set_flag("MTE1", "MTE2", _KEY_EVENT_BASE)
                T.set_flag("MTE1", "MTE2", _KEY_EVENT_BASE + 1)
                T._src_code(ring_init)
                for local_tile in T.serial(tile_count):
                    if local_tile >= score_slots:
                        T.wait_cross_flag(2)
                    score_slot = T.alloc_var("int32", local_tile % score_slots)
                    T._src_code(
                        ring_tile,
                        query_a.access_ptr("r"),
                        key.access_ptr("r"),
                        table.access_ptr("r"),
                        key_l1.access_ptr("rw"),
                        key_b.access_ptr("rw"),
                        qk_l0c.access_ptr("rw"),
                        qk_l1.access_ptr("rw"),
                        score_b.access_ptr("rw"),
                        score_l0c.access_ptr("rw"),
                        weight_a.access_ptr("r"),
                        score_ws.access_ptr("w"),
                        local_tile,
                        score_slot,
                        request,
                        first_tile,
                    )
                    T.set_cross_flag("FIX", 1)
                T.wait_flag("MTE1", "MTE2", _KEY_EVENT_BASE)
                T.wait_flag("MTE1", "MTE2", _KEY_EVENT_BASE + 1)
                T._src_code(ring_drain)
                for pending in T.serial(T.min(tile_count, score_slots)):
                    T.wait_cross_flag(2)
            T.sync_all()

        @T.macro
        def write_indices(topk, index_ub, out, qrow):
            T.barrier_all()
            T._src_code(_EXTRACT, topk.access_ptr("r"), index_ub.access_ptr("w"))
            T.barrier_all()
            T.copy(index_ub, out[qrow, 0, :])

        @T.macro
        def reset_topk(topk, row_topk, score_mask_values, lane):
            T._src_code(_INIT_TOPK, topk.access_ptr("w"))
            score_mask_values[3 + lane] = 0
            T.barrier_all()
            T.copy(topk, row_topk[lane, :])

        @T.macro
        def vector_tiles(
            weights, qscale, kscale, table, out, weights_out, score_ws, partial_topk, cid, vid, from_job, to_job
        ):
            weight_ub = T.alloc_ub((Q_HEADS,), "float16")
            qscale_ub = T.alloc_ub((Q_HEADS,), "float16")
            half_scale = T.alloc_ub((tile_n,), "float16")
            float_scale = T.alloc_ub((tile_n,), "float32")
            score_ub = T.alloc_ub((aiv_rows, tile_n), "float32")
            tile_pairs = T.alloc_ub((pair_count,), "float32")
            tmp_pairs = T.alloc_ub((pair_count,), "float32")
            row_topk = T.alloc_ub((aiv_rows, pair_count), "float32")
            topk = T.alloc_ub((pair_count,), "float32")
            merged = T.alloc_ub((2 * pair_count,), "float32")
            seq = T.alloc_ub((tile_n,), "int32")
            index_ub = T.alloc_ub((tile_n,), "int32")
            score_mask_values = T.alloc_ub((8,), "int32")
            T.tile.arith_progression(seq, 0, 1)

            for request in T.serial(from_job // TILES_PER_REQUEST, T.ceildiv(to_job, TILES_PER_REQUEST)):
                request_start = T.alloc_var("int32", request * TILES_PER_REQUEST)
                first = T.alloc_var("int32", T.max(from_job - request_start, 0))
                stop = T.alloc_var("int32", T.min(to_job - request_start, TILES_PER_REQUEST))
                native_start4 = T.alloc_var("int32", NATIVE_GROUPS * request_start)
                native_start2 = T.alloc_var("int32", native_start4 + TILES_PER_REQUEST)
                owner4 = T.alloc_var("int32")
                owner2 = T.alloc_var("int32")
                native_owner(native_start4 + first, owner4)
                native_owner(native_start2 + first, owner2)
                seam4 = T.alloc_var(
                    "int32", (owner4 + 1) * native_quotient + T.min(owner4 + 1, native_remainder) - native_start4
                )
                seam2 = T.alloc_var(
                    "int32", (owner2 + 1) * native_quotient + T.min(owner2 + 1, native_remainder) - native_start2
                )
                direct4 = T.alloc_var(
                    "int32",
                    T.if_then_else(first == 0 and stop == TILES_PER_REQUEST and seam4 >= TILES_PER_REQUEST, 1, 0),
                )
                direct2 = T.alloc_var(
                    "int32",
                    T.if_then_else(first == 0 and stop == TILES_PER_REQUEST and seam2 >= TILES_PER_REQUEST, 1, 0),
                )
                for lane in T.serial(aiv_rows):
                    row = T.alloc_var("int32", vid * aiv_rows + lane)
                    qrow = T.alloc_var("int32", request * ROWS_PER_REQUEST + row)
                    T.copy(weights[qrow, :], weight_ub)
                    T.copy(qscale[qrow, :], qscale_ub)
                    T.tile.mul(weight_ub, weight_ub, qscale_ub)
                    # Raw snippets consume the metadata, whose buffer name is
                    # stable across codegen scopes and unlike scalar row names.
                    score_mask_values[6] = row
                    T.barrier_all()
                    T._src_code(
                        _PACK_WEIGHT_FOR_ROW,
                        score_ub.access_ptr("w"),
                        weight_ub.access_ptr("r"),
                        score_mask_values.access_ptr("r"),
                    )
                    T.barrier_all()
                    T._src_code(
                        _STORE_WEIGHT_FOR_ROW,
                        score_ub.access_ptr("r"),
                        weights_out.access_ptr("w"),
                        score_mask_values.access_ptr("r"),
                    )
                    T.barrier_all()
                    reset_topk(topk, row_topk, score_mask_values, lane)
                T.pipe_barrier("MTE3")
                T.set_cross_flag("MTE3", 0)

                for local_tile in T.serial(stop - first):
                    tile = T.alloc_var("int32", first + local_tile)
                    score_slot = T.alloc_var("int32", local_tile % score_slots)
                    tile_offset = T.alloc_var("int32", tile * tile_n)
                    # The scheduled page-table range is in-bounds. Keep the
                    # data-dependent GM offsets inside the AscendC gather.
                    score_mask_values[6] = request
                    score_mask_values[7] = tile
                    T.barrier_all()
                    T._src_code(
                        _GATHER_SCALE,
                        half_scale.access_ptr("w"),
                        kscale.access_ptr("r"),
                        table.access_ptr("r"),
                        score_mask_values.access_ptr("r"),
                    )
                    T.wait_cross_flag(1)
                    for lane in T.serial(aiv_rows):
                        row = T.alloc_var("int32", vid * aiv_rows + lane)
                        T.copy(score_ws[cid, score_slot, row, :], score_ub[lane, :])
                    T.set_cross_flag("MTE2", 2)
                    T.wait_flag("MTE2", "V", _SCALE_EVENT)
                    T.tile.cast(float_scale, half_scale, "CAST_NONE", tile_n)
                    for lane in T.serial(aiv_rows):
                        row = T.alloc_var("int32", vid * aiv_rows + lane)
                        visible = T.alloc_var("int32", CONTEXT - ROWS_PER_REQUEST + row + 1)
                        score_mask_values[0] = T.min(tile_n, visible - tile_offset)
                        score_mask_values[1] = tile_offset
                        score_mask_values[2] = lane * tile_n
                        T.barrier_all()
                        T._src_code(
                            _MASK_TILE,
                            tile_pairs.access_ptr("w"),
                            score_ub.access_ptr("rw"),
                            float_scale.access_ptr("r"),
                            seq.access_ptr("r"),
                            score_mask_values.access_ptr("r"),
                        )
                        T._src_code(
                            _PRUNED_TOPK,
                            tile_pairs.access_ptr("rw"),
                            tmp_pairs.access_ptr("rw"),
                            row_topk.access_ptr("rw"),
                            merged.access_ptr("rw"),
                            score_mask_values.access_ptr("rw"),
                        )
                    T.barrier_all()
                    if tile + 1 < stop and (tile + 1 == seam4 or tile + 1 == seam2):
                        for lane in T.serial(aiv_rows):
                            row = T.alloc_var("int32", vid * aiv_rows + lane)
                            seam = T.alloc_var("int32", T.if_then_else(row < NATIVE_QUERY_GROUP, seam4, seam2))
                            if tile + 1 == seam:
                                owner = T.alloc_var("int32", T.if_then_else(row < NATIVE_QUERY_GROUP, owner4, owner2))
                                T.copy(
                                    row_topk[lane, :],
                                    partial_topk[
                                        row // NATIVE_QUERY_GROUP, request + cid + owner, row % NATIVE_QUERY_GROUP, :
                                    ],
                                )
                                T.barrier_all()
                                reset_topk(topk, row_topk, score_mask_values, lane)
                        if tile + 1 == seam4:
                            owner4 = owner4 + 1
                            seam4 = seam4 + native_quotient + T.if_then_else(owner4 < native_remainder, 1, 0)
                        if tile + 1 == seam2:
                            owner2 = owner2 + 1
                            seam2 = seam2 + native_quotient + T.if_then_else(owner2 < native_remainder, 1, 0)

                for lane in T.serial(aiv_rows):
                    row = T.alloc_var("int32", vid * aiv_rows + lane)
                    direct = T.alloc_var("int32", T.if_then_else(row < NATIVE_QUERY_GROUP, direct4, direct2))
                    if direct != 0:
                        qrow = T.alloc_var("int32", request * ROWS_PER_REQUEST + row)
                        T.copy(row_topk[lane, :], topk)
                        write_indices(topk, index_ub, out, qrow)
                    else:
                        owner = T.alloc_var("int32", T.if_then_else(row < NATIVE_QUERY_GROUP, owner4, owner2))
                        T.copy(
                            row_topk[lane, :],
                            partial_topk[row // NATIVE_QUERY_GROUP, request + cid + owner, row % NATIVE_QUERY_GROUP, :],
                        )

            T.barrier_all()
            T.sync_all()
            if four_way_merge:
                # Dedicated reduction output. Stage one's mask, scores and
                # pair scratch have ended their live ranges at the barrier.
                merge_scratch = T.alloc_ub((4 * pair_count,), "float32")
                qrow = T.alloc_var("int32", 2 * cid + vid)
                if qrow < batch * ROWS_PER_REQUEST:
                    T._src_code(
                        small_batch_merge,
                        partial_topk.access_ptr("r"),
                        topk.access_ptr("rw"),
                        row_topk.access_ptr("rw"),
                        merge_scratch.access_ptr("rw"),
                    )
                    write_indices(topk, index_ub, out, qrow)
            # Each AIV merges different rows, including the six rows of B1.
            for iteration in T.serial(0 if four_way_merge else merge_iterations):
                qrow = T.alloc_var("int32", iteration * merge_workers + 2 * cid + vid)
                if qrow < batch * ROWS_PER_REQUEST:
                    request = T.alloc_var("int32", qrow // ROWS_PER_REQUEST)
                    row = T.alloc_var("int32", qrow % ROWS_PER_REQUEST)
                    group = T.alloc_var("int32", row // NATIVE_QUERY_GROUP)
                    group_row = T.alloc_var("int32", row % NATIVE_QUERY_GROUP)
                    request_start = T.alloc_var("int32", request * TILES_PER_REQUEST)
                    native_start = T.alloc_var("int32", (NATIVE_GROUPS * request + group) * TILES_PER_REQUEST)
                    first_owner = T.alloc_var("int32")
                    last_owner = T.alloc_var("int32")
                    first_native = T.alloc_var("int32")
                    last_native = T.alloc_var("int32")
                    tile_owner(request_start, first_owner)
                    tile_owner(request_start + TILES_PER_REQUEST - 1, last_owner)
                    native_owner(native_start, first_native)
                    native_owner(native_start + TILES_PER_REQUEST - 1, last_native)
                    if first_owner < last_owner or first_native < last_native:
                        for native in T.serial(first_native, last_native + 1):
                            start = T.alloc_var(
                                "int32",
                                T.max(0, native * native_quotient + T.min(native, native_remainder) - native_start),
                            )
                            stop = T.alloc_var(
                                "int32",
                                T.min(
                                    TILES_PER_REQUEST,
                                    (native + 1) * native_quotient + T.min(native + 1, native_remainder) - native_start,
                                ),
                            )
                            first_packed = T.alloc_var("int32")
                            last_packed = T.alloc_var("int32")
                            tile_owner(request_start + start, first_packed)
                            tile_owner(request_start + stop - 1, last_packed)
                            T.copy(partial_topk[group, request + first_packed + native, group_row, :], row_topk[0, :])
                            # The newest tile has priority within one native owner.
                            for packed in T.serial(first_packed + 1, last_packed + 1):
                                T.copy(partial_topk[group, request + packed + native, group_row, :], tmp_pairs)
                                T.tile.merge_sort(merged, tmp_pairs, row_topk[0, :])
                                T.copy(merged[:pair_count], row_topk[0, :])
                            # Cross-owner native reduction gives the earlier owner priority.
                            if native == first_native:
                                T.copy(row_topk[0, :], topk)
                            else:
                                T.tile.merge_sort(merged, topk, row_topk[0, :])
                                T.copy(merged[:pair_count], topk)
                        write_indices(topk, index_ub, out, qrow)

        @T.prim_func
        def kernel(
            query: T.Tensor((batch * packed_heads, HEAD_DIM), "int8"),
            key: T.Tensor((pages * PAGE_SIZE, HEAD_DIM), "int8"),
            weights: T.Tensor((batch * ROWS_PER_REQUEST, Q_HEADS), "float16"),
            qscale: T.Tensor((batch * ROWS_PER_REQUEST, Q_HEADS), "float16"),
            kscale: T.Tensor((pages * PAGE_SIZE,), "float16"),
            table: T.Tensor((batch, PAGE_SLOTS), "int32"),
            out: T.Tensor((batch * ROWS_PER_REQUEST, 1, tile_n), "int32"),
            weights_out: T.Tensor((CORES, ROWS_PER_REQUEST, 16, Q_HEADS), "float16"),
            score_ws: T.Tensor((CORES, score_slots, ROWS_PER_REQUEST, tile_n), "float32"),
            # Nonempty intersections have strictly increasing request + packed +
            # native owner IDs within each group, so one additive slot is enough.
            partial_topk: T.Tensor((NATIVE_GROUPS, batch + 2 * (CORES - 1), NATIVE_QUERY_GROUP, pair_count), "float32"),
        ):
            with T.Kernel(CORES, is_npu=True) as (cid, vid):
                with T.Scope("C"):
                    first = T.alloc_var("int32", cid * quotient + T.min(cid, remainder))
                    stop = T.alloc_var("int32", (cid + 1) * quotient + T.min(cid + 1, remainder))
                    cube_tiles(query, key, table, weights_out, score_ws, cid, first, stop)
                with T.Scope("V"):
                    first = T.alloc_var("int32", cid * quotient + T.min(cid, remainder))
                    stop = T.alloc_var("int32", (cid + 1) * quotient + T.min(cid + 1, remainder))
                    vector_tiles(
                        weights, qscale, kscale, table, out, weights_out, score_ws, partial_topk, cid, vid, first, stop
                    )

        return kernel

    return stage_one()


def build_tilelang_call(
    query,
    key,
    weights,
    query_scale,
    key_scale,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    block_table,
    *,
    sparse_count=2048,
    sparse_mode=3,
    core_count=None,
    verified_lengths=None,
    score_load_2d=False,
):
    """Pack six rows for each verified 128K request on the 20-Cube A3."""
    import torch

    args = (query, key, weights, query_scale, key_scale, actual_seq_lengths_query, actual_seq_lengths_key, block_table)
    batch = block_table.shape[0] if block_table.ndim == 2 else 0
    rows = batch * ROWS_PER_REQUEST
    matching_layout = (
        batch > 0
        and sparse_count == TILE_SIZE
        and sparse_mode == 3
        and verified_lengths == (tuple(range(ROWS_PER_REQUEST, rows + 1, ROWS_PER_REQUEST)), (CONTEXT,) * batch)
        and query.shape == (rows, Q_HEADS, HEAD_DIM)
        and key.ndim == 4
        and key.shape[1:] == (PAGE_SIZE, 1, HEAD_DIM)
        and block_table.shape == (batch, PAGE_SLOTS)
        and actual_seq_lengths_query.numel() == actual_seq_lengths_key.numel() == batch
        and weights.numel() == query_scale.numel() == rows * Q_HEADS
        and key_scale.numel() == key.shape[0] * PAGE_SIZE
    )
    if matching_layout and core_count is None:
        import torch_npu

        properties = torch_npu.npu.get_device_properties(query.device)
        core_count = getattr(properties, "cube_core_num", None)
    if not matching_layout or core_count != CORES or batch == BATCH:
        options = {}
        if score_load_2d and matching_layout and core_count == CORES and batch == BATCH:
            options["score_ring"] = _score_ring(_QK_MMAD, _QK_FIXPIPE_PACKED, score_load_2d=True)
        run = build_v25_call(
            *args,
            sparse_count=sparse_count,
            sparse_mode=sparse_mode,
            core_count=core_count,
            verified_lengths=verified_lengths,
            **options,
        )
        if run.qli_dispatch == "packed":
            run.qli_dispatch = "tilelang-v29-load2d" if options else "tilelang-v25"
        return run

    expected = (
        torch.int8,
        torch.int8,
        torch.float16,
        torch.float16,
        torch.float16,
        torch.int32,
        torch.int32,
        torch.int32,
    )
    if any(tensor.dtype != dtype for tensor, dtype in zip(args, expected, strict=True)):
        raise ValueError("Unexpected GE QLI input dtype")
    pages = key.shape[0]
    kernel_args = (
        query.view(rows * Q_HEADS, HEAD_DIM),
        key.view(pages * PAGE_SIZE, HEAD_DIM),
        weights.view(rows, Q_HEADS),
        query_scale.view(rows, Q_HEADS),
        key_scale.view(pages * PAGE_SIZE),
        block_table,
    )
    four_way_merge = batch <= 2
    kernel = _compile(batch, pages, four_way_merge, score_load_2d)

    def run():
        return kernel(*kernel_args)

    if four_way_merge:
        run.qli_dispatch = "tilelang-v29-merge4"
    elif score_load_2d:
        run.qli_dispatch = "tilelang-v29-load2d"
    else:
        run.qli_dispatch = "tilelang-v29"
    return run
