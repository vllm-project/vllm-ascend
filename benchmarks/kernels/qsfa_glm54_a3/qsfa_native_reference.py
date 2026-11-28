# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt.

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang port of CANN 9.2.1 arch22 QSFA MLA for the GLM-5.4 MTP5 fixture."""

from functools import lru_cache, partial

from qsfa_native_reference_cube import make_cube_lifecycle, make_cube_stages
from qsfa_native_reference_gather import make_gather_stage

_VECTOR_INIT = """
    int qsfa_next_vec1_task = 0;
    int qsfa_next_vec2_task = 0;
    int qsfa_vec_pingpong = 0;
    int qsfa_gather_last_request = -1;
    int qsfa_gather_causal_base = 0;
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::Duplicate(vec_max_default, -2.0e38f, 128);
    AscendC::Duplicate(vec_sum_default, 0.0f, 128);
"""

_VECTOR_DRAIN = """
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
"""


def _softmax_source(scale):
    return f"""
    {{
        const int task = qsfa_next_vec1_task++;
        const int core = AscendC::GetBlockIdx() / 2;
        const int lane = AscendC::GetBlockIdx() % 2;
        const int stats_out = (task % 2) * 128;
        const int stats_in = ((task + 1) % 2) * 128;
        const int gm_start = ((core * 2 + task % 2) * 128 + lane * 32) * 512;
        const AscendC::DataCopyExtParams valid_copy{{1, 256 * sizeof(int32_t), 0, 0, 0}};
        const AscendC::DataCopyPadExtParams<int32_t> valid_pad{{false, 0, 0, 0}};
        const int valid_start = (core * 4 + task % 4) * 256;
        AscendC::DataCopyPad(gather_valid[128], valid_gm[valid_start], valid_copy, valid_pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);

        for (int head = 0; head < 32; head += 16) {{
            const int ping = qsfa_vec_pingpong;
            auto qk = vec_in1[ping * 8192];
            auto p = vec_out1.ReinterpretCast<bfloat16_t>();
            auto scratch = vec_tmp1.ReinterpretCast<uint8_t>();
            const int gm_offset = gm_start + head * 512;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + ping);
            AscendC::DataCopy(qk, scores[gm_offset], 8192);
            if (head == 0) {{
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
            }}
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
            AscendC::Muls(qk, qk, static_cast<float>({scale!r}), 8192);
            for (int half = 0; half < 2; ++half) {{
                const int count = gather_valid.GetValue(128 + half * 128 + task % 4);
                if (count < 256) {{
                    const int start = half * 256 + count;
                    const int stop = (half + 1) * 256;
                    const int aligned = (start + 7) / 8 * 8;
                    if (start < aligned) {{
                        const uint64_t one_block = (0xffu << (start % 8)) & 0xffu;
                        uint64_t mask[1] = {{one_block * 0x0101010101010101ULL}};
                        for (int r = 0; r < 16; r += 8) {{
                            AscendC::Duplicate(qk[r * 512 + start / 8 * 8], -2.0e38f, mask, 1, 64, 0);
                        }}
                    }}
                    if (aligned < stop) {{
                        for (int r = 0; r < 16; ++r) {{
                            AscendC::Duplicate(qk[r * 512 + aligned], -2.0e38f, stop - aligned);
                        }}
                    }}
                }}
            }}
            AscendC::PipeBarrier<PIPE_V>();
            auto old_sum = task % 4 == 0 ? vec_sum_default : vec_sum[stats_in + head];
            auto old_max = task % 4 == 0 ? vec_max_default : vec_max[stats_in + head];
            const AscendC::SoftMaxShapeInfo shape{{16, 512, 16, 512}};
            const auto tiling = AscendC::SoftMaxFlashV2TilingFunc(
                shape, sizeof(float), sizeof(float), 32768, true, false);
            static constexpr AscendC::SoftmaxConfig cfg{{
                false, 0, 0, AscendC::SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC}};
            AscendC::SoftmaxFlashV2<float, true, true, false, false, cfg>(
                qk, vec_sum[stats_out + head], vec_max[stats_out + head], qk,
                vec_exp[stats_out + head], old_sum, old_max, scratch, tiling, shape);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
            AscendC::Cast(p, qk, AscendC::RoundMode::CAST_ROUND, 8192);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2 + ping);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
            AscendC::DataCopy(probs[gm_offset], p, 8192);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
            qsfa_vec_pingpong ^= 1;
        }}
    }}
    """


_VECTOR_OUTPUT = """
    {
        const int task = qsfa_next_vec2_task++;
        const int core = AscendC::GetBlockIdx() / 2;
        const int lane = AscendC::GetBlockIdx() % 2;
        const int row = qsfa_vec_row_begin + task / 4;
        const int stats = (task % 2) * 128;
        const int gm_now = ((core * 2 + task % 2) * 128 + lane * 32) * 512;
        const int gm_prev = ((core * 2 + (task + 1) % 2) * 128 + lane * 32) * 512;
        const AscendC::BinaryRepeatParams by_head{1, 1, 0, 64, 64, 1};

        for (int head = 0; head < 32; head += 16) {
            const int ping = qsfa_vec_pingpong;
            const int offset = head * 512;
            auto pv = vec_in1[ping * 8192];
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + ping);
            AscendC::DataCopy(pv, partial[gm_now + offset], 8192);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
            AscendC::DataCopy(vec_tmp1, pv, 8192);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2 + ping);

            if (task % 4 != 0) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(7);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(7);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
                AscendC::DataCopy(vec_in2, numerator[gm_prev + offset], 8192);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
                AscendC::Brcb(vec_tmp2, vec_exp[stats + head], 4, {1, 8});
                AscendC::PipeBarrier<PIPE_V>();
                for (int col = 0; col < 512; col += 64) {
                    AscendC::Mul(vec_in2[col], vec_in2[col], vec_tmp2, 64, 16, by_head);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(vec_tmp1, vec_tmp1, vec_in2, 8192);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
            }

            if (task % 4 == 3) {
                AscendC::Brcb(vec_tmp2, vec_sum[stats + head], 4, {1, 8});
                AscendC::PipeBarrier<PIPE_V>();
                for (int col = 0; col < 512; col += 64) {
                    AscendC::Div(vec_tmp1[col], vec_tmp1[col], vec_tmp2, 64, 16, by_head);
                }
                AscendC::PipeBarrier<PIPE_V>();
                auto result = vec_out1.ReinterpretCast<bfloat16_t>();
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
                AscendC::Cast(result, vec_tmp1, AscendC::RoundMode::CAST_RINT, 8192);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
                const int out_offset = (row * 64 + lane * 32 + head) * 512;
                const AscendC::DataCopyExtParams output_copy{16, 1024, 0, 0, 0};
                AscendC::DataCopyPad(output[out_offset], result, output_copy);
            } else {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
                AscendC::DataCopy(vec_out1, vec_tmp1, 8192);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
                AscendC::DataCopy(numerator[gm_now + offset], vec_out1, 8192);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
            qsfa_vec_pingpong ^= 1;
        }
    }
"""


@lru_cache(maxsize=8)
def _compile_glm54(
    scale,
    *,
    bulk_metadata=False,
    resident_query=False,
    blocked_kv=False,
    output_fused_dma_pv_prefetch=False,
):
    import tilelang
    from tilelang import language as T

    qk, pv = make_cube_stages(
        T,
        resident_query=resident_query,
        blocked_kv=blocked_kv,
        pv_l1_prefetch=output_fused_dma_pv_prefetch,
    )
    cube_init, cube_drain = make_cube_lifecycle(T, resident_query=resident_query)
    gather = make_gather_stage(
        T,
        bulk_metadata=bulk_metadata,
        blocked_kv=blocked_kv,
        output_fused_dma=output_fused_dma_pv_prefetch,
    )
    softmax_source = _softmax_source(scale)
    cores = 20
    rows = T.symbolic("query_rows", "int32")
    pages = T.symbolic("pages", "int32")
    batch = T.symbolic("batch", "int32")
    workspace_rows = 128  # Arch22 TND tiling reserves 128 rows per GM exchange slot.
    gather_out_words = 32 * (512 + 64) // 2 if output_fused_dma_pv_prefetch else 8192
    dtype = "bfloat16"
    passes = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    }

    @tilelang.jit(
        out_idx=[6],
        workspace_idx=[7, 8, 9, 10, 11, 12],
        target="ascendc",
        platform="A3",
        compile_flags=[
            "--cce-auto-sync=off",
            "-mllvm -cce-vf-remove-membar=false",
            "-mllvm -cce-aicore-hoist-movemask=false",
        ],
        pass_configs=passes,
    )
    def compile_kernel():
        @T.prim_func
        def main(
            query: T.Tensor((rows, 64, 576), dtype),
            packed: T.Tensor((pages, 128, 1, 656), "int8"),
            indices: T.Tensor((rows, 1, 2048), "int32"),
            page_table: T.Tensor((batch, 1024), "int32"),
            query_ends: T.Tensor((batch,), "int32"),
            kv_lengths: T.Tensor((batch,), "int32"),
            output: T.Tensor((rows, 64, 512), dtype),
            # AscendC owns the flat slot layout, with 36x512 or 4x36x128 C0 blocks.
            kv_nz: T.Tensor((cores, 4, 36, 512, 16), dtype),
            scores: T.Tensor((cores, 2, workspace_rows, 512), "float32"),
            probs: T.Tensor((cores, 2, workspace_rows, 512), dtype),
            partial: T.Tensor((cores, 2, workspace_rows, 512), "float32"),
            numerator: T.Tensor((cores, 2, workspace_rows, 512), "float32"),
            valid_gm: T.Tensor((cores, 4, 2, 128), "int32"),
        ):
            with T.Kernel(cores, is_npu=True) as (cid, vid):
                T.reads(
                    query[0:rows, 0:64, 0:576],
                    packed[0:pages, 0:128, 0:1, 0:656],
                    indices[0:rows, 0:1, 0:2048],
                    page_table[0:batch, 0:1024],
                    query_ends[0:batch],
                    kv_lengths[0:batch],
                    kv_nz[0:cores, 0:4, 0:36, 0:512, 0:16],
                    scores[0:cores, 0:2, 0:workspace_rows, 0:512],
                    probs[0:cores, 0:2, 0:workspace_rows, 0:512],
                    partial[0:cores, 0:2, 0:workspace_rows, 0:512],
                    numerator[0:cores, 0:2, 0:workspace_rows, 0:512],
                    valid_gm[0:cores, 0:4, 0:2, 0:128],
                )
                T.writes(
                    output[0:rows, 0:64, 0:512],
                    kv_nz[0:cores, 0:4, 0:36, 0:512, 0:16],
                    scores[0:cores, 0:2, 0:workspace_rows, 0:512],
                    probs[0:cores, 0:2, 0:workspace_rows, 0:512],
                    partial[0:cores, 0:2, 0:workspace_rows, 0:512],
                    numerator[0:cores, 0:2, 0:workspace_rows, 0:512],
                    valid_gm[0:cores, 0:4, 0:2, 0:128],
                )
                vec_in1 = T.alloc_ub((16384,), "float32")
                vec_in2 = T.alloc_ub((8192,), "float32")
                vec_out1 = T.alloc_ub((gather_out_words,), "float32")
                vec_out2 = T.meta_var(vec_out1) if output_fused_dma_pv_prefetch else T.alloc_ub((1024,), "float32")
                vec_tmp1 = T.alloc_ub((8192,), "float32")
                vec_tmp2 = T.alloc_ub((2048,), "float32")
                gather_valid = T.alloc_ub((2048,), "int32")
                vec_max = T.alloc_ub((256,), "float32")
                vec_sum = T.alloc_ub((256,), "float32")
                vec_exp = T.alloc_ub((256,), "float32")
                vec_max_default = T.alloc_ub((128,), "float32")
                vec_sum_default = T.alloc_ub((128,), "float32")

                rows_per_core = (rows + cores - 1) // cores
                row_begin = cid * rows_per_core
                task_count = T.max(0, T.min(rows_per_core, rows - row_begin)) * 4

                # One task per S2 panel. Consecutive native TND rows share
                # cross-core events and the final two drain ticks.
                with T.Scope("C"):
                    if cid < rows:
                        cube_init()
                        for _ in T.unroll(4):
                            T.set_cross_flag("MTE2", 3)
                        for tick in T.serial(task_count + 1):
                            if tick < task_count:
                                T.wait_cross_flag(6)
                                qk(query, kv_nz, scores, cid, row_begin + tick // 4, tick % 4, tick % 2, 512)
                                T.set_cross_flag("FIX", 7)
                            if tick > 0:
                                T.wait_cross_flag(8)
                                pv(kv_nz, probs, partial, cid, (tick - 1) % 4, (tick - 1) % 2, 512)
                                T.set_cross_flag("FIX", 9)
                                T.set_cross_flag("MTE2", 3)
                        cube_drain()

                with T.Scope("V"):
                    if cid < rows:
                        T._src_code(_VECTOR_INIT, vec_max_default.access_ptr("w"), vec_sum_default.access_ptr("w"))
                        qsfa_vec_row_begin = T.alloc_var("int32", row_begin)
                        for tick in T.serial(task_count + 2):
                            if tick < task_count:
                                T.wait_cross_flag(3)
                                gather(
                                    packed,
                                    indices,
                                    page_table,
                                    query_ends,
                                    kv_lengths,
                                    kv_nz,
                                    cid,
                                    vid,
                                    row_begin + tick // 4,
                                    tick % 4,
                                    tick % 4,
                                    512,
                                    vec_in1,
                                    vec_in2,
                                    vec_out1,
                                    vec_out2,
                                    vec_tmp1,
                                    vec_tmp2,
                                    gather_valid,
                                    valid_gm,
                                )
                                T.set_cross_flag("MTE3", 6)
                            if tick > 0 and tick < task_count + 1:
                                T.wait_cross_flag(7)
                                T._src_code(
                                    softmax_source,
                                    scores.access_ptr("r"),
                                    probs.access_ptr("w"),
                                    valid_gm.access_ptr("r"),
                                    gather_valid.access_ptr("w"),
                                    vec_in1.access_ptr("rw"),
                                    vec_out1.access_ptr("w"),
                                    vec_tmp1.access_ptr("rw"),
                                    vec_sum.access_ptr("rw"),
                                    vec_max.access_ptr("rw"),
                                    vec_exp.access_ptr("w"),
                                    vec_sum_default.access_ptr("r"),
                                    vec_max_default.access_ptr("r"),
                                )
                                T.set_cross_flag("MTE3", 8)
                            if tick > 1:
                                T.wait_cross_flag(9)
                                T._src_code(
                                    _VECTOR_OUTPUT,
                                    partial.access_ptr("r"),
                                    numerator.access_ptr("rw"),
                                    output.access_ptr("w"),
                                    vec_in1.access_ptr("rw"),
                                    vec_in2.access_ptr("rw"),
                                    vec_out1.access_ptr("w"),
                                    vec_tmp1.access_ptr("rw"),
                                    vec_tmp2.access_ptr("w"),
                                    vec_sum.access_ptr("r"),
                                    vec_exp.access_ptr("r"),
                                    qsfa_vec_row_begin,
                                )
                        for _ in T.unroll(4):
                            T.wait_cross_flag(3)
                        T._src_code(_VECTOR_DRAIN)

        return main

    return compile_kernel()


def _run_glm54_mtp5(
    query,
    packed,
    indices,
    page_table,
    query_ends,
    kv_lengths,
    *,
    scale_value,
    block_i=512,
    packed_gather=True,
    bulk_metadata=False,
    resident_query=False,
    blocked_kv=False,
    output_fused_dma_pv_prefetch=False,
):
    """Run the six-row-per-request, all-2048-indices-valid MTP5 benchmark fixture."""
    if block_i != 512 or not packed_gather:
        raise ValueError("The arch22 MTP5 fixture requires 512-column packed gather")
    batch = query_ends.shape[0]
    if not 1 <= batch <= 80:
        raise ValueError("The arch22 MTP5 fixture covers batches 1 through 80")
    rows = 6 * batch
    expected = (
        (query, (rows, 64, 576), "bfloat16"),
        (packed, (batch * 1024, 128, 1, 656), "int8"),
        (indices, (rows, 1, 2048), "int32"),
        (page_table, (batch, 1024), "int32"),
        (query_ends, (batch,), "int32"),
        (kv_lengths, (batch,), "int32"),
    )
    for tensor, shape, dtype in expected:
        if tuple(tensor.shape) != shape or str(tensor.dtype) != f"torch.{dtype}" or not tensor.is_contiguous():
            raise ValueError(f"Expected contiguous {dtype} tensor with shape {shape}")
    kernel = _compile_glm54(
        float(scale_value),
        bulk_metadata=bulk_metadata,
        resident_query=resident_query,
        blocked_kv=blocked_kv,
        output_fused_dma_pv_prefetch=output_fused_dma_pv_prefetch,
    )
    return kernel(query, packed, indices, page_table, query_ends, kv_lengths)


qsfa_glm54_mtp5 = partial(_run_glm54_mtp5)
qsfa_glm54_mtp5_bulk_metadata = partial(_run_glm54_mtp5, bulk_metadata=True)
qsfa_glm54_mtp5_resident_query = partial(_run_glm54_mtp5, resident_query=True)
qsfa_glm54_mtp5_bulk_metadata_resident_query = partial(_run_glm54_mtp5, bulk_metadata=True, resident_query=True)
qsfa_glm54_mtp5_blocked_kv = partial(_run_glm54_mtp5, blocked_kv=True)
qsfa_glm54_mtp5_blocked_output_fused_dma_pv_prefetch = partial(
    _run_glm54_mtp5,
    blocked_kv=True,
    output_fused_dma_pv_prefetch=True,
)
for entry_point in (
    qsfa_glm54_mtp5,
    qsfa_glm54_mtp5_bulk_metadata,
    qsfa_glm54_mtp5_resident_query,
    qsfa_glm54_mtp5_bulk_metadata_resident_query,
    qsfa_glm54_mtp5_blocked_kv,
    qsfa_glm54_mtp5_blocked_output_fused_dma_pv_prefetch,
):
    entry_point.uses_combined_query = True
