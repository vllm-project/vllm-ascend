# SPDX-License-Identifier: Apache-2.0
"""A3 QLI for B10 MTP5 at 128K with packed six-row Cube score reduction.

The fixed schedule gives two consecutive Cube cores to each request and
three rows to each AIV. A block-diagonal weight matrix reduces all six
rows in one score MMAD and keeps the V11 score workspace and TopK order.
"""

from functools import lru_cache

from qli_tilelang import _EXTRACT, _INIT_TOPK, _LOAD_SCORE_B, _SORT_TILE
from qli_tilelang_v1 import _MASK_TILE, _QK_FIXPIPE

BATCH = 10
ROWS_PER_REQUEST = 6
CONTEXT = 131072
PAGE_SIZE = 128
CORES = 2 * BATCH
EXPECTED_QUERY_ENDS = tuple(range(ROWS_PER_REQUEST, BATCH * ROWS_PER_REQUEST + 1, ROWS_PER_REQUEST))
EXPECTED_KEY_LENGTHS = (CONTEXT,) * BATCH

# FP16 NZ uses one 32-byte block per C0, so the L1 Z stride is M.
_QK_FIXPIPE_PACKED = _QK_FIXPIPE.replace("cube_m", "192").replace("cfg.dstStride = 128;", "cfg.dstStride = 192;")
_LOAD_SCORE_B_PACKED = (
    _LOAD_SCORE_B.replace("cfg.l1H = 8;", "cfg.l1H = 12;")
    .replace("cfg.mExtension = 32;", "cfg.mExtension = 192;")
    .replace("cfg.mStartPt = cube_row * 32;", "cfg.mStartPt = 0;")
)
_COMPUTE_SCORE_PACKED = """
{
    AscendC::MmadParams cfg;
    cfg.m = 16;
    cfg.n = 128;
    cfg.k = 192;
    cfg.cmatrixSource = false;
    cfg.cmatrixInitVal = true;
    cfg.unitFlag = 0;
    AscendC::Mmad(score_l0c, weight_a, score_b, cfg);
}
"""
_STORE_SCORE_PACKED = """
{
    using namespace AscendC;
    DataCopyCO12DstParams cfg;
    cfg.mSize = 6;
    cfg.nSize = 128;
    cfg.dstStride = 2048;
    cfg.srcStride = 16;
    cfg.quantPre = QuantMode_t::NoQuant;
    cfg.reluPre = 0;
    cfg.channelSplit = 0;
    cfg.nz2ndEn = true;
    SetFixpipeNz2ndFlag(1, 1, 1);
    const uint32_t workspace_row = (GetBlockIdx() * 2 + score_slot) * 6;
    DataCopy(score_ws[workspace_row * 2048 + cube_panel * 128], score_l0c, cfg);
}
"""
_PACK_WEIGHT = """
{
    using namespace AscendC;
    auto packed = score_ub.ReinterpretCast<half>();
    PipeBarrier<PIPE_V>();
    Duplicate(packed, static_cast<half>(0.0f), 16 * 32);
    PipeBarrier<PIPE_V>();
    DataCopy(packed[row * 32], weight_ub, 32);
}
"""
_STORE_WEIGHT = """
{
    // Mixed mode launches two AIVs per Cube. The workspace uses the Cube ID.
    const uint32_t cube = AscendC::GetBlockIdx() / 2U;
    const uint32_t weight_base = (cube * 6 + row) * 16 * 32;
    AscendC::DataCopy(weights_out[weight_base], score_ub.ReinterpretCast<half>(), 16 * 32);
}
"""


def _key_panel_source(panel, guard="true"):
    return f"""
{{
    if ({guard}) {{
        const uint32_t cube = AscendC::GetBlockIdx();
        const uint32_t tile = (cube & 1U) * 32 + local_tile;
        const uint32_t panel = {panel};
        const uint32_t physical_page = table.GetValue((cube / 2) * 1024 + tile * 16 + panel);
        tl::ascend::copy_gm_to_l1<int8_t, 128, 128>(
            key_l1[(panel & 1U) * 16384], key[physical_page * 16384], 128);
    }}
}}
"""


@lru_cache(maxsize=16)
def _compile(pages):
    import tilelang
    from tilelang import language as T

    q_heads = 32
    packed_heads = ROWS_PER_REQUEST * q_heads
    dim = 128
    cube_n = PAGE_SIZE
    tile_n = 2048
    pair_count = 2 * tile_n
    score_slots = 2
    aiv_rows = ROWS_PER_REQUEST // 2
    tiles_per_owner = CONTEXT // (2 * tile_n)
    page_slots = CONTEXT // PAGE_SIZE
    panel_event = 7
    first_key_panel = _key_panel_source("0")
    following_key_panel = _key_panel_source("cube_panel + 1", "cube_panel + 1 < 16")
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
            qk_l1 = T.alloc_L1((packed_heads, cube_n), "float16")
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

            # The next QK uses independent L0C storage while FIX drains scores.
            T.set_flag("FIX", "M", panel_event)
            for local_tile in T.serial(tiles_per_owner):
                if local_tile >= score_slots:
                    T.wait_cross_flag(2)
                score_slot = T.alloc_var("int32", local_tile % score_slots)
                T._src_code(
                    first_key_panel, key_l1.access_ptr("w"), key.access_ptr("r"), table.access_ptr("r"), local_tile
                )
                for cube_panel in T.serial(tile_n // cube_n):
                    T.set_flag("MTE2", "MTE1", panel_event)
                    T.wait_flag("MTE2", "MTE1", panel_event)
                    T.copy(key_l1[cube_panel % 2, :, :], key_b, transpose=True)

                    T.mma(query_a, key_b, qk_l0c, init=True)
                    T.set_flag("M", "FIX", panel_event)
                    T.wait_flag("M", "FIX", panel_event)
                    T._src_code(_QK_FIXPIPE_PACKED, qk_l1.access_ptr("w"), qk_l0c.access_ptr("r"))
                    T._src_code(
                        following_key_panel,
                        key_l1.access_ptr("w"),
                        key.access_ptr("r"),
                        table.access_ptr("r"),
                        local_tile,
                        cube_panel,
                    )
                    T.set_flag("FIX", "MTE1", panel_event)
                    T.wait_flag("FIX", "MTE1", panel_event)
                    T._src_code(_LOAD_SCORE_B_PACKED, score_b.access_ptr("w"), qk_l1.access_ptr("r"))
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
                T.set_cross_flag("FIX", 1)
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
                T._src_code(_INIT_TOPK, topk.access_ptr("w"))
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
                    T._src_code(_SORT_TILE, tile_pairs.access_ptr("rw"), tmp_pairs.access_ptr("rw"))
                    T.tile.merge_sort(merged, tile_pairs, row_topk[lane, :])
                    T.copy(merged[:pair_count], row_topk[lane, :])
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
        return build_v1(*args, sparse_count=sparse_count, sparse_mode=sparse_mode, core_count=core_count)

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
    kernel = _compile(pages)

    def run():
        return kernel(*kernel_args)

    return run
