# SPDX-License-Identifier: Apache-2.0
"""A3 QLI packed score schedule with early, slot-owned K prefetch.

Once MTE1 reads the current K panel into L0B, MTE2 can stage the next
panel from GM while Cube and FixPipe process the current panel.
"""

from functools import lru_cache
from textwrap import indent

from qli_tilelang import _EXTRACT, _INIT_TOPK, _SORT_TILE
from qli_tilelang_v1 import _MASK_TILE
from qli_tilelang_v13 import (
    _COMPUTE_SCORE_PACKED,
    _LOAD_SCORE_B_PACKED,
    _PACK_WEIGHT,
    _QK_FIXPIPE_PACKED,
    _STORE_SCORE_PACKED,
    _STORE_WEIGHT,
    BATCH,
    CONTEXT,
    CORES,
    EXPECTED_KEY_LENGTHS,
    EXPECTED_QUERY_ENDS,
    PAGE_SIZE,
    ROWS_PER_REQUEST,
)

# Match V10's manually owned event IDs, away from the compiler's events.
_KEY_EVENT_BASE = 4
_KEY_READY = f"""
{{
    using namespace AscendC;
    WaitFlag<HardEvent::MTE2_MTE1>({_KEY_EVENT_BASE} + (cube_panel & 1U));
}}
"""
_KEY_RELEASE = f"""
{{
    using namespace AscendC;
    SetFlag<HardEvent::MTE1_MTE2>({_KEY_EVENT_BASE} + (cube_panel & 1U));
}}
"""
_NEXT_KEY_READY = _KEY_READY.replace("(cube_panel & 1U)", "((cube_panel + 1) & 1U)")
_NEXT_KEY_RELEASE = _KEY_RELEASE.replace("(cube_panel & 1U)", "((cube_panel + 1) & 1U)")


def _key_panel_source(panel, guard="true", *, request=None, first_tile=None):
    if request is None:
        owner = "const uint32_t cube = GetBlockIdx();"
        request = "cube / 2"
        first_tile = "(cube & 1U) * 32"
    else:
        owner = ""
    return f"""
{{
    using namespace AscendC;
    if ({guard}) {{
        {owner}
        const uint32_t tile = {first_tile} + local_tile;
        const uint32_t panel = {panel};
        const uint32_t physical_page = table.GetValue(({request}) * 1024 + tile * 16 + panel);
        const uint32_t slot = panel & 1U;
        const uint32_t event = {_KEY_EVENT_BASE} + slot;
        WaitFlag<HardEvent::MTE1_MTE2>(event);
        tl::ascend::copy_gm_to_l1<int8_t, 128, 128>(
            key_l1[slot * 16384], key[physical_page * 16384], 128);
        SetFlag<HardEvent::MTE2_MTE1>(event);
    }}
}}
"""


def _ahead_key_source(panels_per_tile, key_bank_event, later_key_panel):
    return f"""
{{
    using namespace AscendC;
    const uint32_t next_panel = cube_panel + 1;
    if (next_panel < {panels_per_tile}) {{
        const uint32_t next_slot = next_panel & 1U;
        const uint32_t next_event = {_KEY_EVENT_BASE} + next_slot;
        WaitFlag<HardEvent::M_MTE1>({key_bank_event});
        WaitFlag<HardEvent::MTE2_MTE1>(next_event);
        tl::ascend::copy_l1_to_l0b<int8_t, 128, 128, true>(
            key_b[0], key_l1[next_slot * 16384], 128, 128);
        SetFlag<HardEvent::MTE1_MTE2>(next_event);
{indent(later_key_panel.strip(), "        ")}
    }}
}}
"""


@lru_cache(maxsize=16)
def _compile(
    pages, topk_source=None, *, coalesce_scores=False, overlap_key_copy=False, opaque_key_copy=False, score_ring=None
):
    import tilelang
    from tilelang import language as T

    tile_n = 2048
    pair_count = 2 * tile_n
    if topk_source is None:

        @T.macro
        def initialize_topk(topk, score_mask_values, lane):
            T._src_code(_INIT_TOPK, topk.access_ptr("w"))

        @T.macro
        def update_topk(tile_pairs, tmp_pairs, row_topk, merged, score_mask_values, lane):
            T._src_code(_SORT_TILE, tile_pairs.access_ptr("rw"), tmp_pairs.access_ptr("rw"))
            T.tile.merge_sort(merged, tile_pairs, row_topk[lane, :])
            T.copy(merged[:pair_count], row_topk[lane, :])
    else:

        @T.macro
        def initialize_topk(topk, score_mask_values, lane):
            T._src_code(_INIT_TOPK, topk.access_ptr("w"))
            score_mask_values[3 + lane] = 0

        @T.macro
        def update_topk(tile_pairs, tmp_pairs, row_topk, merged, score_mask_values, lane):
            T._src_code(
                topk_source,
                tile_pairs.access_ptr("rw"),
                tmp_pairs.access_ptr("rw"),
                row_topk.access_ptr("rw"),
                merged.access_ptr("rw"),
                score_mask_values.access_ptr("rw"),
            )

    q_heads = 32
    packed_heads = ROWS_PER_REQUEST * q_heads
    dim = 128
    cube_n = PAGE_SIZE
    use_score_ring = score_ring is not None
    ring_init, ring_tile, ring_drain = score_ring if use_score_ring else ("", "", "")
    qk_l1_shape = (2, packed_heads, cube_n) if use_score_ring else (packed_heads, cube_n)
    score_slots = 2
    aiv_rows = ROWS_PER_REQUEST // 2
    tiles_per_owner = CONTEXT // (2 * tile_n)
    page_slots = CONTEXT // PAGE_SIZE
    panel_event = 7
    key_bank_event = 4
    # An ahead QK can finish before FIX consumes the preceding score-store post.
    qk_fix_event = 5 if overlap_key_copy else panel_event
    panels_per_tile = tile_n // cube_n
    first_key_panel = _key_panel_source("0")
    following_key_panel = _key_panel_source("cube_panel + 1", f"cube_panel + 1 < {panels_per_tile}")
    second_key_panel = _key_panel_source("1", f"1 < {panels_per_tile}")
    later_key_panel = _key_panel_source("cube_panel + 2", f"cube_panel + 2 < {panels_per_tile}")
    ahead_key_panel = _ahead_key_source(panels_per_tile, key_bank_event, later_key_panel)
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
        def cube_tile_panels(
            query_a,
            key,
            table,
            key_l1,
            key_b,
            qk_l0c,
            qk_l1,
            score_b,
            score_l0c,
            weight_a,
            score_ws,
            local_tile,
            score_slot,
        ):
            if use_score_ring:
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
                )
            else:
                T._src_code(
                    first_key_panel, key_l1.access_ptr("w"), key.access_ptr("r"), table.access_ptr("r"), local_tile
                )
                if overlap_key_copy:
                    T._src_code(
                        second_key_panel, key_l1.access_ptr("w"), key.access_ptr("r"), table.access_ptr("r"), local_tile
                    )
                    # The preceding tile leaves a key-bank token after its final QK.
                    T.wait_flag("M", "MTE1", key_bank_event)
                    T.wait_flag("MTE2", "MTE1", _KEY_EVENT_BASE)
                    T.copy(key_l1[0, :, :], key_b, transpose=True)
                    T.set_flag("MTE1", "MTE2", _KEY_EVENT_BASE)
                    T.set_flag("MTE1", "M", key_bank_event)
                    T.wait_flag("MTE1", "M", key_bank_event)
                for cube_panel in T.serial(panels_per_tile):
                    if not overlap_key_copy:
                        T._src_code(_KEY_READY, cube_panel)
                        T.copy(key_l1[cube_panel % 2, :, :], key_b, transpose=True)
                        T._src_code(_KEY_RELEASE, cube_panel)
                        T._src_code(
                            following_key_panel,
                            key_l1.access_ptr("w"),
                            key.access_ptr("r"),
                            table.access_ptr("r"),
                            local_tile,
                            cube_panel,
                        )

                    T.mma(query_a, key_b, qk_l0c, init=True)
                    if overlap_key_copy:
                        T.set_flag("M", "MTE1", key_bank_event)
                    T.set_flag("M", "FIX", qk_fix_event)
                    T.wait_flag("M", "FIX", qk_fix_event)
                    T._src_code(_QK_FIXPIPE_PACKED, qk_l1.access_ptr("w"), qk_l0c.access_ptr("r"))
                    T.set_flag("FIX", "MTE1", panel_event)
                    if overlap_key_copy:
                        if opaque_key_copy:
                            # Preserve the manual events without a compiler-inserted branch barrier.
                            T._src_code(
                                ahead_key_panel,
                                key_b.access_ptr("w"),
                                key_l1.access_ptr("rw"),
                                key.access_ptr("r"),
                                table.access_ptr("r"),
                                local_tile,
                                cube_panel,
                            )
                        else:
                            if cube_panel + 1 < panels_per_tile:
                                # Use the independent key bank while FIX converts QK.
                                T.wait_flag("M", "MTE1", key_bank_event)
                                T._src_code(_NEXT_KEY_READY, cube_panel)
                                T.copy(key_l1[(cube_panel + 1) % 2, :, :], key_b, transpose=True)
                                T._src_code(_NEXT_KEY_RELEASE, cube_panel)
                                T._src_code(
                                    later_key_panel,
                                    key_l1.access_ptr("w"),
                                    key.access_ptr("r"),
                                    table.access_ptr("r"),
                                    local_tile,
                                    cube_panel,
                                )
                    T.wait_flag("FIX", "MTE1", panel_event)
                    T._src_code(_LOAD_SCORE_B_PACKED, score_b.access_ptr("w"), qk_l1.access_ptr("r"))
                    # This also makes the ahead K copy visible to the next QK.
                    T.set_flag("MTE1", "M", panel_event)
                    T.wait_flag("MTE1", "M", panel_event)
                    T.wait_flag("FIX", "M", panel_event)
                    T._src_code(
                        _COMPUTE_SCORE_PACKED,
                        score_l0c.access_ptr("w"),
                        weight_a.access_ptr("r"),
                        score_b.access_ptr("r"),
                    )
                    T.set_flag("M", "FIX", panel_event)
                    T.wait_flag("M", "FIX", panel_event)
                    T._src_code(
                        _STORE_SCORE_PACKED, score_ws.access_ptr("w"), score_l0c.access_ptr("r"), score_slot, cube_panel
                    )
                    T.set_flag("FIX", "M", panel_event)

        @T.macro
        def cube_tiles(query, key, table, weights_out, score_ws, cid):
            request = T.alloc_var("int32", cid // 2)
            query_base = T.alloc_var("int32", request * ROWS_PER_REQUEST * q_heads)
            query_l1 = T.alloc_L1((packed_heads, dim), "int8")
            query_a = T.alloc_L0A((packed_heads, dim), "int8")
            key_l1 = T.alloc_L1((2, cube_n, dim), "int8")
            # L0B is exactly 64 KiB with disjoint K and score operands.
            key_b = T.alloc_L0B((dim, cube_n), "int8")
            score_b = T.alloc_L0B((packed_heads, cube_n), "float16")
            qk_l0c = T.alloc_L0C((packed_heads, cube_n), "int32")
            qk_l1 = T.alloc_L1(qk_l1_shape, "float16")
            weight_l1 = T.alloc_L1((ROWS_PER_REQUEST, 16, q_heads), "float16")
            weight_a = T.alloc_L0A((ROWS_PER_REQUEST, 16, q_heads), "float16")
            score_l0c = T.alloc_L0C((16, cube_n), "float32")

            T.copy(query[query_base : query_base + packed_heads, :], query_l1)
            T.copy(query_l1, query_a)
            T.wait_cross_flag(0)
            for row in T.serial(ROWS_PER_REQUEST):
                T.copy(weights_out[cid, row, :, :], weight_l1[row, :, :])
            # M is one 16-row L0A fractal, so the six K blocks are contiguous.
            for row in T.serial(ROWS_PER_REQUEST):
                T.copy(weight_l1[row, :, :], weight_a[row, :, :])
            T.barrier_all()

            # A free token belongs to MTE2 until that slot is populated.
            T.set_flag("MTE1", "MTE2", _KEY_EVENT_BASE)
            T.set_flag("MTE1", "MTE2", _KEY_EVENT_BASE + 1)
            if use_score_ring:
                T._src_code(ring_init)
            else:
                # The next QK uses independent L0C storage while FIX drains scores.
                T.set_flag("FIX", "M", panel_event)
                if overlap_key_copy:
                    T.set_flag("M", "MTE1", key_bank_event)
            for local_tile in T.serial(tiles_per_owner):
                if local_tile >= score_slots:
                    T.wait_cross_flag(2)
                score_slot = T.alloc_var("int32", local_tile % score_slots)
                cube_tile_panels(
                    query_a,
                    key,
                    table,
                    key_l1,
                    key_b,
                    qk_l0c,
                    qk_l1,
                    score_b,
                    score_l0c,
                    weight_a,
                    score_ws,
                    local_tile,
                    score_slot,
                )
                T.set_cross_flag("FIX", 1)
            # A short tile returns each consumed slot to free. Any unused slot
            # still holds its original free token.
            T.wait_flag("MTE1", "MTE2", _KEY_EVENT_BASE)
            T.wait_flag("MTE1", "MTE2", _KEY_EVENT_BASE + 1)
            if use_score_ring:
                T._src_code(ring_drain)
            else:
                if overlap_key_copy:
                    T.wait_flag("M", "MTE1", key_bank_event)
                T.wait_flag("FIX", "M", panel_event)
            for slot in T.serial(score_slots):
                T.wait_cross_flag(2)
            T.sync_all()

        @T.macro
        def vector_tiles(weights, qscale, kscale, table, out, weights_out, score_ws, partial, cid, vid):
            request = T.alloc_var("int32", cid // 2)
            first = T.alloc_var("int32", cid % 2 * tiles_per_owner)
            # Live UB allocations total 184,480 B before compiler temporaries.
            weight_ub = T.alloc_ub((q_heads,), "float16")
            qscale_ub = T.alloc_ub((q_heads,), "float16")
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
            for lane in T.serial(aiv_rows):
                row = T.alloc_var("int32", vid * aiv_rows + lane)
                qrow = T.alloc_var("int32", request * ROWS_PER_REQUEST + row)
                T.copy(weights[qrow, :], weight_ub)
                T.copy(qscale[qrow, :], qscale_ub)
                T.tile.mul(weight_ub, weight_ub, qscale_ub)
                # Use the first KiB of the idle score buffer for the K block.
                # Its only nonzero weights are in its matching M row.
                T._src_code(_PACK_WEIGHT, score_ub.access_ptr("w"), weight_ub.access_ptr("r"), row)
                T.barrier_all()
                T._src_code(_STORE_WEIGHT, score_ub.access_ptr("r"), weights_out.access_ptr("w"), row)
                T.barrier_all()
                initialize_topk(topk, score_mask_values, lane)
                T.barrier_all()
                T.copy(topk, row_topk[lane, :])
            T.pipe_barrier("MTE3")
            T.set_cross_flag("MTE3", 0)

            for local_tile in T.serial(tiles_per_owner):
                score_slot = T.alloc_var("int32", local_tile % score_slots)
                tile_offset = T.alloc_var("int32", (first + local_tile) * tile_n)
                for page in T.serial(tile_n // PAGE_SIZE):
                    logical_page = T.alloc_var("int32", tile_offset // PAGE_SIZE + page)
                    scale_start = T.alloc_var("int32", table[request, logical_page] * PAGE_SIZE)
                    scale_panel_start = T.alloc_var("int32", page * PAGE_SIZE)
                    T.copy(
                        kscale[scale_start : scale_start + PAGE_SIZE],
                        half_scale[scale_panel_start : scale_panel_start + PAGE_SIZE],
                    )
                T.wait_cross_flag(1)
                if coalesce_scores:
                    first_score_row = T.alloc_var("int32", vid * aiv_rows)
                    T.copy(score_ws[cid, score_slot, first_score_row : first_score_row + aiv_rows, :], score_ub)
                else:
                    for lane in T.serial(aiv_rows):
                        row = T.alloc_var("int32", vid * aiv_rows + lane)
                        T.copy(score_ws[cid, score_slot, row, :], score_ub[lane, :])
                T.set_cross_flag("MTE2", 2)
                T.tile.cast(float_scale, half_scale, "CAST_NONE", tile_n)
                for lane in T.serial(aiv_rows):
                    row = T.alloc_var("int32", vid * aiv_rows + lane)
                    visible = T.alloc_var("int32", CONTEXT - ROWS_PER_REQUEST + row + 1)
                    valid_lanes = T.alloc_var("int32", T.min(tile_n, visible - tile_offset))
                    score_mask_values[0] = valid_lanes
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
                    update_topk(tile_pairs, tmp_pairs, row_topk, merged, score_mask_values, lane)
                T.barrier_all()

            # The earlier owner keeps its own lists in UB. The later owner
            # publishes its lists for the cross-core merge.
            if cid % 2 == 1:
                for lane in T.serial(aiv_rows):
                    row = T.alloc_var("int32", vid * aiv_rows + lane)
                    T.copy(row_topk[lane, :], partial[request, row, :])
            T.barrier_all()
            T.sync_all()
            if cid % 2 == 0:
                for lane in T.serial(aiv_rows):
                    row = T.alloc_var("int32", vid * aiv_rows + lane)
                    qrow = T.alloc_var("int32", request * ROWS_PER_REQUEST + row)
                    T.copy(partial[request, row, :], tmp_pairs)
                    T.tile.merge_sort(merged, tmp_pairs, row_topk[lane, :])
                    T.copy(merged[:pair_count], topk)
                    T.barrier_all()
                    T._src_code(_EXTRACT, topk.access_ptr("r"), index_ub.access_ptr("w"))
                    T.barrier_all()
                    T.copy(index_ub, out[qrow, 0, :])

        @T.prim_func
        def kernel(
            query: T.Tensor((BATCH * ROWS_PER_REQUEST * q_heads, dim), "int8"),
            key: T.Tensor((pages * PAGE_SIZE, dim), "int8"),
            weights: T.Tensor((BATCH * ROWS_PER_REQUEST, q_heads), "float16"),
            qscale: T.Tensor((BATCH * ROWS_PER_REQUEST, q_heads), "float16"),
            kscale: T.Tensor((pages * PAGE_SIZE,), "float16"),
            table: T.Tensor((BATCH, page_slots), "int32"),
            out: T.Tensor((BATCH * ROWS_PER_REQUEST, 1, tile_n), "int32"),
            weights_out: T.Tensor((CORES, ROWS_PER_REQUEST, 16, q_heads), "float16"),
            score_ws: T.Tensor((CORES, score_slots, ROWS_PER_REQUEST, tile_n), "float32"),
            partial: T.Tensor((BATCH, ROWS_PER_REQUEST, pair_count), "float32"),
        ):
            with T.Kernel(CORES, is_npu=True) as (cid, vid):
                with T.Scope("C"):
                    cube_tiles(query, key, table, weights_out, score_ws, cid)
                with T.Scope("V"):
                    vector_tiles(weights, qscale, kscale, table, out, weights_out, score_ws, partial, cid, vid)

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
    topk_source=None,
    coalesce_scores=False,
    overlap_key_copy=False,
    opaque_key_copy=False,
    score_ring=None,
):
    """Use the packed schedule with the caller's matching immutable lengths.

    The paired fixture supplies (cpu_query_ends, cpu_key_lengths) from the
    same lengths that initialized the device tensors. Without a matching
    record, dispatch to V1 without reading the device lengths on the host.
    """
    import torch
    from qli_tilelang_v1 import build_tilelang_call as build_v1

    args = (query, key, weights, query_scale, key_scale, actual_seq_lengths_query, actual_seq_lengths_key, block_table)
    matching_layout = (
        sparse_count == 2048
        and sparse_mode == 3
        and verified_lengths == (EXPECTED_QUERY_ENDS, EXPECTED_KEY_LENGTHS)
        and query.shape == (BATCH * ROWS_PER_REQUEST, 32, 128)
        and key.ndim == 4
        and key.shape[1:] == (PAGE_SIZE, 1, 128)
        and block_table.shape == (BATCH, CONTEXT // PAGE_SIZE)
        and actual_seq_lengths_query.numel() == actual_seq_lengths_key.numel() == BATCH
    )
    if matching_layout and core_count is None:
        import torch_npu

        properties = torch_npu.npu.get_device_properties(query.device)
        core_count = getattr(properties, "cube_core_num", None)
    if not matching_layout or core_count != CORES:
        run = build_v1(*args, sparse_count=sparse_count, sparse_mode=sparse_mode, core_count=core_count)
        run.qli_dispatch = "tilelang-v1"
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
    rows = BATCH * ROWS_PER_REQUEST
    kernel_args = (
        query.view(rows * 32, 128),
        key.view(pages * PAGE_SIZE, 128),
        weights.view(rows, 32),
        query_scale.view(rows, 32),
        key_scale.view(pages * PAGE_SIZE),
        block_table,
    )
    kernel = _compile(
        pages,
        topk_source,
        coalesce_scores=coalesce_scores,
        overlap_key_copy=overlap_key_copy,
        opaque_key_copy=opaque_key_copy,
        score_ring=score_ring,
    )

    def run():
        return kernel(*kernel_args)

    run.qli_dispatch = "packed"
    return run
