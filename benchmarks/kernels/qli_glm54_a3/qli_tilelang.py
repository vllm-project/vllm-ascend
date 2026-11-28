# SPDX-License-Identifier: Apache-2.0
"""Ascend A3 INT8 QLI for GLM-5.3, using TND Q and PA_BSND KV.

The arithmetic, tile ownership and TopK merge order follow CANN QLIPreload
arch22 at 91de0d9959ab57be7fcbef8b0592d16c242c5955.
"""

from functools import lru_cache
from math import gcd

# The historical AscendC TileLang target has no INT32 L0C -> FP16 L1
# operation. Keep the intermediate on chip with the native FixPipe sequence.
_QK_FIXPIPE = """
{
    using namespace AscendC;
    AscendC::DataCopyCO12DstParams cfg;
    cfg.mSize = 128;
    cfg.nSize = 128;
    cfg.dstStride = 128;
    cfg.srcStride = 128;
    cfg.quantPre = QuantMode_t::DEQF16;
    cfg.reluPre = 1;
    cfg.channelSplit = 0;
    cfg.nz2ndEn = 0;
    AscendC::SetFixpipePreQuantFlag(0x3a800000);
    AscendC::DataCopy(qk_l1, qk_l0c, cfg);
}
"""

_LOAD_SCORE_B = """
{
    AscendC::LoadData3DParamsV2<half> cfg;
    cfg.l1H = 8;
    cfg.l1W = 16;
    cfg.channelSize = 128;
    cfg.padList[0] = 0;
    cfg.padList[1] = 0;
    cfg.padList[2] = 0;
    cfg.padList[3] = 255;
    cfg.mExtension = 32;
    cfg.kExtension = 128;
    cfg.mStartPt = cube_row * 32;
    cfg.kStartPt = 0;
    cfg.strideW = 1;
    cfg.strideH = 1;
    cfg.filterW = 1;
    cfg.filterSizeW = 0;
    cfg.filterH = 1;
    cfg.filterSizeH = 0;
    cfg.dilationFilterW = 1;
    cfg.dilationFilterH = 1;
    cfg.enTranspose = 1;
    cfg.fMatrixCtrl = 0;
    AscendC::LoadData<half, Catlass::Gemm::Tile::CONV3D_LOAD3DV2_DEFAULT_CONFIG>(score_b, qk_l1, cfg);
}
"""

_STORE_SCORE = """
{
    using namespace AscendC;
    AscendC::DataCopyCO12DstParams cfg;
    cfg.mSize = 1;
    cfg.nSize = 128;
    cfg.dstStride = 2048;
    cfg.srcStride = 16;
    cfg.quantPre = QuantMode_t::NoQuant;
    cfg.reluPre = 0;
    cfg.channelSplit = 0;
    cfg.nz2ndEn = true;
    AscendC::SetFixpipeNz2ndFlag(1, 32, 2048);
    AscendC::DataCopy(score_ws[(AscendC::GetBlockIdx() * 4 + cube_row) * 2048 + cube_panel * 128], score_l0c, cfg);
}
"""

_INIT_TOPK = """
{
    auto topk_bits = topk.ReinterpretCast<int32_t>();
    uint64_t value_mask[2] = {0x5555555555555555ULL, 0};
    uint64_t index_mask[2] = {0xaaaaaaaaaaaaaaaaULL, 0};
    AscendC::Duplicate(topk_bits, static_cast<int32_t>(0xff800000U), value_mask, 64, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate(topk_bits, -1, index_mask, 64, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
}
"""

_MASK_TILE = """
{
    const int32_t valid_lanes = score_mask_values.GetValue(0);
    const int32_t tile_offset = score_mask_values.GetValue(1);
    auto ids = tile_pairs[2048].ReinterpretCast<int32_t>();
    auto value_bits = tile_pairs.ReinterpretCast<int32_t>();
    AscendC::Mul(score_ub, score_ub, float_scale, valid_lanes);
    AscendC::Duplicate(value_bits, static_cast<int32_t>(0xff800000U), 2048);
    AscendC::Duplicate(ids, -1, 2048);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(tile_pairs, score_ub, 0.0f, valid_lanes);
    AscendC::Adds(ids, seq, tile_offset, valid_lanes);
    AscendC::PipeBarrier<PIPE_V>();
}
"""

# Keep indices in the integer lanes of the value/index pairs throughout the
# native Sort32 and MrgSort sequence. The newer tile is the first merge source.
_SORT_TILE = """
{
    AscendC::Sort32(tmp_pairs, tile_pairs, tile_pairs[2048].ReinterpretCast<uint32_t>(), 64);
    AscendC::PipeBarrier<PIPE_V>();
    for (int level = 0; level < 3; ++level) {
        auto source = level % 2 == 0 ? tmp_pairs : tile_pairs;
        auto dest = level % 2 == 0 ? tile_pairs : tmp_pairs;
        const int length = 32 << (2 * level);
        AscendC::MrgSort4Info config;
        for (int i = 0; i < 4; ++i) {
            config.elementLengths[i] = length;
        }
        config.ifExhaustedSuspension = false;
        config.validBit = 15;
        config.repeatTimes = 16 >> (2 * level);
        AscendC::MrgSortSrcList<float> list;
        list.src1 = source;
        list.src2 = source[2 * length];
        list.src3 = source[4 * length];
        list.src4 = source[6 * length];
        AscendC::MrgSort<float>(dest, list, config);
        AscendC::PipeBarrier<PIPE_V>();
    }
}
"""

_EXTRACT = """
{
    auto index_bits = index_ub.ReinterpretCast<uint32_t>();
    auto pairs = topk.ReinterpretCast<uint32_t>();
    AscendC::GatherMaskParams cfg;
    cfg.repeatTimes = 64;
    cfg.src0BlockStride = 1;
    cfg.src0RepeatStride = 8;
    cfg.src1RepeatStride = 0;
    uint64_t count = 0;
    AscendC::GatherMask(index_bits, pairs, static_cast<uint8_t>(2), false,
                        static_cast<uint32_t>(0), cfg, count);
}
"""


@lru_cache(maxsize=32)
def _compile(rows, batch, pages, page_size, page_slots, cores):
    import tilelang
    from tilelang import language as T

    q_heads = 32
    q_group = 4
    dim = 128
    cube_n = 128
    tile_n = 2048
    pair_count = 2 * tile_n
    key_copy = gcd(cube_n, page_size)
    scale_copy = gcd(tile_n, page_size)
    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }

    @tilelang.jit(
        out_idx=[8],
        workspace_idx=[9, 10, 11],
        target="ascendc",
        platform="A3",
        pass_configs=pass_configs,
    )
    def stage_one():
        @T.macro
        def cube_groups(qends, klens, weights_out, score_ws, query, key, table, cid, from_job, to_job):
            offset = T.alloc_var("int32")
            begin_q = T.alloc_var("int32")
            offset = 0
            begin_q = 0
            query_l1 = T.alloc_L1((q_group * q_heads, dim), "int8")
            key_l1 = T.alloc_L1((cube_n, dim), "int8")
            query_a = T.alloc_L0A((q_group * q_heads, dim), "int8")
            key_b = T.alloc_L0B((dim, cube_n), "int8")
            qk_l0c = T.alloc_L0C((q_group * q_heads, cube_n), "int32")
            qk_l1 = T.alloc_L1((q_group * q_heads, cube_n), "float16")
            weight_l1 = T.alloc_L1((q_group, 16, q_heads), "float16")
            weight_a = T.alloc_L0A((16, q_heads), "float16")
            score_b = T.alloc_L0B((q_heads, cube_n), "float16")
            score_l0c = T.alloc_L0C((16, cube_n), "float32")
            for request in T.serial(batch):
                qend = T.alloc_var("int32", qends[request])
                qlen = T.alloc_var("int32", qend - begin_q)
                klen = T.alloc_var("int32", klens[request])
                group_count = T.alloc_var("int32", T.ceildiv(qlen, q_group))
                for group in T.serial(group_count):
                    group_row = T.alloc_var("int32", group * q_group)
                    qcount = T.alloc_var("int32", T.min(q_group, qlen - group_row))
                    group_k = T.alloc_var("int32", T.min(klen, T.max(1, klen - qlen + group_row + q_group)))
                    jobs = T.alloc_var("int32", T.ceildiv(group_k, tile_n))
                    first = T.alloc_var("int32", T.max(from_job - offset, 0))
                    stop = T.alloc_var("int32", T.min(to_job - offset, jobs))
                    if first < stop:
                        for row in T.serial(qcount):
                            query_start = T.alloc_var("int32", (begin_q + group_row + row) * q_heads)
                            q_panel_start = T.alloc_var("int32", row * q_heads)
                            T.copy(
                                query[query_start : query_start + q_heads, :],
                                query_l1[q_panel_start : q_panel_start + q_heads, :],
                            )
                        T.copy(query_l1, query_a)
                        T.wait_cross_flag(0)
                        for weight_row in T.serial(qcount):
                            T.copy(weights_out[cid, weight_row, :, :], weight_l1[weight_row, :, :])
                        for tile in T.serial(first, stop):
                            n = T.alloc_var("int32", T.min(tile_n, group_k - tile * tile_n))
                            panel_count = T.alloc_var("int32", T.ceildiv(n, cube_n))
                            for cube_panel in T.serial(panel_count):
                                column = T.alloc_var("int32", cube_panel * cube_n)
                                for page in T.serial(cube_n // key_copy):
                                    logical = T.alloc_var("int32", tile * tile_n + column + page * key_copy)
                                    if logical < klen:
                                        key_start = T.alloc_var(
                                            "int32",
                                            table[request, logical // page_size] * page_size + logical % page_size,
                                        )
                                        k_panel_start = T.alloc_var("int32", page * key_copy)
                                        T.copy(
                                            key[key_start : key_start + key_copy, :],
                                            key_l1[k_panel_start : k_panel_start + key_copy, :],
                                        )
                                T.copy(key_l1, key_b, transpose=True)
                                T.mma(query_a, key_b, qk_l0c, init=True)
                                T.barrier_all()
                                T._src_code(_QK_FIXPIPE, qk_l1.access_ptr("w"), qk_l0c.access_ptr("r"))
                                T.barrier_all()
                                for cube_row in T.serial(qcount):
                                    T.copy(weight_l1[cube_row, :, :], weight_a)
                                    T._src_code(_LOAD_SCORE_B, score_b.access_ptr("w"), qk_l1.access_ptr("r"), cube_row)
                                    T.barrier_all()
                                    T.mma(weight_a, score_b, score_l0c, init=True)
                                    T.barrier_all()
                                    T._src_code(
                                        _STORE_SCORE,
                                        score_ws.access_ptr("w"),
                                        score_l0c.access_ptr("r"),
                                        cid,
                                        cube_row,
                                        cube_panel,
                                    )
                                    T.barrier_all()
                            T.set_cross_flag("FIX", 1)
                            T.wait_cross_flag(2)
                    offset = offset + jobs
                begin_q = qend

        @T.macro
        def vector_groups(
            qends,
            klens,
            weights_out,
            score_ws,
            partial,
            weights,
            qscale,
            kscale,
            table,
            out,
            cid,
            vid,
            from_job,
            to_job,
            quotient,
            remainder,
        ):
            # Apply per-head weights and key scales, then merge each core's top-k candidates.
            offset = T.alloc_var("int32")
            begin_q = T.alloc_var("int32")
            offset = 0
            begin_q = 0
            weight_ub = T.alloc_ub((q_heads,), "float16")
            qscale_ub = T.alloc_ub((q_heads,), "float16")
            weight_matrix = T.alloc_ub((16, q_heads), "float16")
            half_scale = T.alloc_ub((tile_n,), "float16")
            float_scale = T.alloc_ub((tile_n,), "float32")
            score_ub = T.alloc_ub((tile_n,), "float32")
            tile_pairs = T.alloc_ub((pair_count,), "float32")
            tmp_pairs = T.alloc_ub((pair_count,), "float32")
            row_topk = T.alloc_ub((2, pair_count), "float32")
            topk = T.alloc_ub((pair_count,), "float32")
            merged = T.alloc_ub((2 * pair_count,), "float32")
            seq = T.alloc_ub((tile_n,), "int32")
            index_ub = T.alloc_ub((tile_n,), "int32")
            score_mask_values = T.alloc_ub((8,), "int32")
            valid_lanes = T.alloc_var("int32")
            tile_offset = T.alloc_var("int32")
            T.tile.arith_progression(seq, 0, 1)
            for request in T.serial(batch):
                qend = T.alloc_var("int32", qends[request])
                qlen = T.alloc_var("int32", qend - begin_q)
                klen = T.alloc_var("int32", klens[request])
                group_count = T.alloc_var("int32", T.ceildiv(qlen, q_group))
                for group in T.serial(group_count):
                    group_row = T.alloc_var("int32", group * q_group)
                    qcount = T.alloc_var("int32", T.min(q_group, qlen - group_row))
                    group_k = T.alloc_var("int32", T.min(klen, T.max(1, klen - qlen + group_row + q_group)))
                    jobs = T.alloc_var("int32", T.ceildiv(group_k, tile_n))
                    first = T.alloc_var("int32", T.max(from_job - offset, 0))
                    stop = T.alloc_var("int32", T.min(to_job - offset, jobs))
                    if first < stop:
                        first_half = T.alloc_var("int32", T.ceildiv(qcount, 2))
                        owned = T.alloc_var("int32", T.if_then_else(vid == 0, first_half, qcount - first_half))
                        for lane in T.serial(owned):
                            row = T.alloc_var("int32", vid * first_half + lane)
                            qrow = T.alloc_var("int32", begin_q + group_row + row)
                            T.copy(weights[qrow, :], weight_ub)
                            T.copy(qscale[qrow, :], qscale_ub)
                            T.tile.mul(weight_ub, weight_ub, qscale_ub)
                            T.tile.broadcast(weight_matrix, weight_ub, axis=0)
                            T.copy(weight_matrix, weights_out[cid, row, :, :])
                            T.barrier_all()
                            T._src_code(_INIT_TOPK, topk.access_ptr("w"))
                            T.barrier_all()
                            T.copy(topk, row_topk[lane, :])
                        T.pipe_barrier("MTE3")
                        T.set_cross_flag("MTE3", 0)
                        for tile in T.serial(first, stop):
                            T.wait_cross_flag(1)
                            tile_offset = tile * tile_n
                            for page in T.serial(tile_n // scale_copy):
                                logical = T.alloc_var("int32", tile_offset + page * scale_copy)
                                if logical < klen:
                                    scale_start = T.alloc_var(
                                        "int32", table[request, logical // page_size] * page_size + logical % page_size
                                    )
                                    scale_panel_start = T.alloc_var("int32", page * scale_copy)
                                    T.copy(
                                        kscale[scale_start : scale_start + scale_copy],
                                        half_scale[scale_panel_start : scale_panel_start + scale_copy],
                                    )
                            T.tile.cast(float_scale, half_scale, "CAST_NONE", tile_n)
                            for lane in T.serial(owned):
                                row = T.alloc_var("int32", vid * first_half + lane)
                                visible = T.alloc_var("int32", klen - qlen + group_row + row + 1)
                                valid_lanes = T.min(tile_n, visible - tile_offset)
                                if valid_lanes > 0:
                                    T.copy(score_ws[cid, row, :], score_ub)
                                    score_mask_values[0] = valid_lanes
                                    score_mask_values[1] = tile_offset
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
                                    T.copy(merged[0:pair_count], row_topk[lane, :])
                            T.barrier_all()
                            T.set_cross_flag("MTE3", 2)
                        for lane in T.serial(owned):
                            row = T.alloc_var("int32", vid * first_half + lane)
                            qrow = T.alloc_var("int32", begin_q + group_row + row)
                            visible = T.alloc_var("int32", klen - qlen + group_row + row + 1)
                            row_jobs = T.alloc_var("int32", T.ceildiv(T.max(visible, 0), tile_n))
                            if first < row_jobs:
                                if first == 0 and stop >= row_jobs:
                                    T.copy(row_topk[lane, :], topk)
                                    T.barrier_all()
                                    T._src_code(_EXTRACT, topk.access_ptr("r"), index_ub.access_ptr("w"))
                                    T.barrier_all()
                                    T.copy(index_ub, out[qrow, 0, :])
                                else:
                                    seam = T.if_then_else(first == 0, 1, 0)
                                    T.copy(row_topk[lane, :], partial[cid, seam, row, :])
                    offset = offset + jobs
                begin_q = qend

            T.barrier_all()
            T.sync_all()
            offset = 0
            begin_q = 0
            first_run = T.alloc_var("int32", (quotient + 1) * remainder)
            for request in T.serial(batch):
                qend = T.alloc_var("int32", qends[request])
                qlen = T.alloc_var("int32", qend - begin_q)
                klen = T.alloc_var("int32", klens[request])
                group_count = T.alloc_var("int32", T.ceildiv(qlen, q_group))
                for group in T.serial(group_count):
                    group_row = T.alloc_var("int32", group * q_group)
                    group_k = T.alloc_var("int32", T.min(klen, T.max(1, klen - qlen + group_row + q_group)))
                    jobs = T.alloc_var("int32", T.ceildiv(group_k, tile_n))
                    qcount = T.alloc_var("int32", T.min(q_group, qlen - group_row))
                    owner = T.alloc_var(
                        "int32",
                        T.if_then_else(
                            offset < first_run,
                            offset // (quotient + 1),
                            remainder + (offset - first_run) // T.max(1, quotient),
                        ),
                    )
                    for row in T.serial(qcount):
                        qrow = T.alloc_var("int32", begin_q + group_row + row)
                        visible = T.alloc_var("int32", klen - qlen + group_row + row + 1)
                        if visible <= 0:
                            if qrow % (2 * cores) == 2 * cid + vid:
                                T.tile.fill(index_ub, -1)
                                T.copy(index_ub, out[qrow, 0, :])
                        else:
                            last = T.alloc_var("int32", offset + T.ceildiv(visible, tile_n) - 1)
                            end_core = T.alloc_var(
                                "int32",
                                T.if_then_else(
                                    last < first_run,
                                    last // (quotient + 1),
                                    remainder + (last - first_run) // T.max(1, quotient),
                                ),
                            )
                            if cid == owner and end_core > owner and row % 2 == vid:
                                T.copy(partial[cid, 1, row, :], topk)
                                for source in T.serial(cid + 1, end_core + 1):
                                    T.copy(partial[source, 0, row, :], tmp_pairs)
                                    T.tile.merge_sort(merged, topk, tmp_pairs)
                                    T.copy(merged[0:pair_count], topk)
                                T.barrier_all()
                                T._src_code(_EXTRACT, topk.access_ptr("r"), index_ub.access_ptr("w"))
                                T.barrier_all()
                                T.copy(index_ub, out[qrow, 0, :])
                    offset = offset + jobs
                begin_q = qend

        @T.macro
        def build_schedule(qends, klens, cid, quotient, remainder, first_job, last_job):
            total_jobs = T.alloc_var("int32")
            begin_q = T.alloc_var("int32")
            total_jobs = 0
            begin_q = 0
            for request in T.serial(batch):
                qend = T.alloc_var("int32", qends[request])
                qlen = T.alloc_var("int32", qend - begin_q)
                klen = T.alloc_var("int32", klens[request])
                group_count = T.alloc_var("int32", T.ceildiv(qlen, q_group))
                for group in T.serial(group_count):
                    group_k = T.alloc_var("int32", T.min(klen, T.max(1, klen - qlen + (group + 1) * q_group)))
                    total_jobs = total_jobs + T.ceildiv(group_k, tile_n)
                begin_q = qend
            quotient = total_jobs // cores
            remainder = total_jobs % cores
            # TileLang makes these macro bindings available to the calling kernel.
            first_job = cid * quotient + T.min(cid, remainder)  # noqa: F841
            last_job = (cid + 1) * quotient + T.min(cid + 1, remainder)  # noqa: F841

        @T.prim_func
        def kernel(
            query: T.Tensor((rows * q_heads, dim), "int8"),
            key: T.Tensor((pages * page_size, dim), "int8"),
            weights: T.Tensor((rows, q_heads), "float16"),
            qscale: T.Tensor((rows, q_heads), "float16"),
            kscale: T.Tensor((pages * page_size,), "float16"),
            qends: T.Tensor((batch,), "int32"),
            klens: T.Tensor((batch,), "int32"),
            table: T.Tensor((batch, page_slots), "int32"),
            out: T.Tensor((rows, 1, tile_n), "int32"),
            weights_out: T.Tensor((cores, q_group, 16, q_heads), "float16"),
            score_ws: T.Tensor((cores, q_group, tile_n), "float32"),
            partial: T.Tensor((cores, 2, q_group, pair_count), "float32"),
        ):
            with T.Kernel(cores, is_npu=True) as (cid, vid):
                with T.Scope("C"):
                    quotient = T.alloc_var("int32")
                    remainder = T.alloc_var("int32")
                    first_job = T.alloc_var("int32")
                    last_job = T.alloc_var("int32")
                    build_schedule(qends, klens, cid, quotient, remainder, first_job, last_job)
                    cube_groups(qends, klens, weights_out, score_ws, query, key, table, cid, first_job, last_job)
                    T.sync_all()
                with T.Scope("V"):
                    quotient = T.alloc_var("int32")
                    remainder = T.alloc_var("int32")
                    first_job = T.alloc_var("int32")
                    last_job = T.alloc_var("int32")
                    build_schedule(qends, klens, cid, quotient, remainder, first_job, last_job)
                    vector_groups(
                        qends,
                        klens,
                        weights_out,
                        score_ws,
                        partial,
                        weights,
                        qscale,
                        kscale,
                        table,
                        out,
                        cid,
                        vid,
                        first_job,
                        last_job,
                        quotient,
                        remainder,
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
):
    """Compile once and return the paired benchmark's no-argument QLI callable."""
    import torch
    import torch_npu

    if sparse_count != 2048 or sparse_mode != 3:
        raise ValueError("GLM-5.3 A3 QLI requires sparse_count=2048 and sparse_mode=3")
    rows, heads, dim = query.shape
    pages, page_size, k_heads, k_dim = key.shape
    batch, page_slots = block_table.shape
    if (heads, dim, k_heads, k_dim) != (32, 128, 1, 128):
        raise ValueError("Expected Q[T,32,128] and K[pages,page_size,1,128]")
    if not 16 <= page_size <= 1024 or page_size % 16:
        raise ValueError("Expected a 16-aligned page_size between 16 and 1024")
    if core_count is None:
        properties = torch_npu.npu.get_device_properties(query.device)
        core_count = getattr(properties, "cube_core_num", None)
    if core_count is None or core_count <= 0:
        raise ValueError("Pass the available Cube core count on torch_npu versions that do not report it")
    core_count = int(core_count)
    kernel = _compile(rows, batch, pages, page_size, page_slots, core_count)
    args = (
        query.view(rows * heads, dim),
        key.view(pages * page_size, dim),
        weights.view(rows, heads),
        query_scale.view(rows, heads),
        key_scale.view(-1),
        actual_seq_lengths_query.view(batch),
        actual_seq_lengths_key.view(batch),
        block_table,
    )
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
    if any(arg.dtype != dtype for arg, dtype in zip(args, expected)):
        raise ValueError("Unexpected GE QLI input dtype")

    def run():
        return kernel(*args)

    return run
