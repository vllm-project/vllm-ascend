/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ffn_apt.cpp
 * \brief A5 (ascend950, arch35) kernel entry for FFN operator.
 *        arch35 fused 路径，TilingKey 模板化：
 *        DTYPE(bf16/fp16) × ACT(gelu/silu/swiglu) × MODE(basic/streamK)。
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include <type_traits>
#include "lib/matmul_intf.h"
#include "tensor_api/tensor.h"
#include "3rd/mat_mul_v3/arch35/mat_mul_tiling_data.h"
#include "3rd/mat_mul_v3/arch35/mat_mul_pingpong_basic_cmct.h"
#include "3rd/mat_mul_v3/arch35/mat_mul_gelu_basic_cmct.h"
#include "ffn_arch35_tiling_key.h"
const uint64_t BLOCK_SIZE = 16;
#include "3rd/mat_mul_v3/arch35/mat_mul_streamk_basic_cmct.h"

namespace {

__aicore__ inline MatMulV3BasicTilingData MakeFfnMMTiling(const FFNTilingData *ffn, bool isUp)
{
    MatMulV3BasicTilingData t;
    if (isUp) {
        t.usedCoreNum = ffn->upUsedCoreNum;
        t.m = ffn->upM;
        t.n = ffn->upN;
        t.k = ffn->upK;
        t.mL1 = ffn->upML1;
        t.nL1 = ffn->upNL1;
        t.kL1 = ffn->upKL1;
        t.baseM = ffn->upBaseM;
        t.baseN = ffn->upBaseN;
        t.baseK = ffn->upBaseK;
        t.mTailCnt = ffn->upMTailCnt;
        t.nTailCnt = ffn->upNTailCnt;
        t.mBaseTailSplitCnt = ffn->upMBaseTailSplitCnt;
        t.nBaseTailSplitCnt = ffn->upNBaseTailSplitCnt;
        t.mTailMain = ffn->upMTailMain;
        t.nTailMain = ffn->upNTailMain;
        t.l1BufferNum = ffn->upL1BufferNum;
        t.l0cDB = ffn->upL0cDB;
        t.ubDB = ffn->upUbDB;
    } else {
        t.usedCoreNum = ffn->downUsedCoreNum;
        t.m = ffn->downM;
        t.n = ffn->downN;
        t.k = ffn->downK;
        t.mL1 = ffn->downML1;
        t.nL1 = ffn->downNL1;
        t.kL1 = ffn->downKL1;
        t.baseM = ffn->downBaseM;
        t.baseN = ffn->downBaseN;
        t.baseK = ffn->downBaseK;
        t.mTailCnt = ffn->downMTailCnt;
        t.nTailCnt = ffn->downNTailCnt;
        t.mBaseTailSplitCnt = ffn->downMBaseTailSplitCnt;
        t.nBaseTailSplitCnt = ffn->downNBaseTailSplitCnt;
        t.mTailMain = ffn->downMTailMain;
        t.nTailMain = ffn->downNTailMain;
        t.l1BufferNum = ffn->downL1BufferNum;
        t.l0cDB = ffn->downL0cDB;
        t.ubDB = ffn->downUbDB;
    }
    t.skSingleCoreK = isUp ? ffn->upSkSingleCoreK : ffn->downSkSingleCoreK;
    t.fullLoad = isUp ? ffn->upFullLoad : ffn->downFullLoad;
    t.mmadParam = 0;
    t.l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    t.sliceM = 0;
    t.srcNdStride = 0;
    t.innerBatch = 1;
    return t;
}

template <typename T, typename BiasT, bool TransB>
__aicore__ inline void RunFfnFusedGeluUp(GM_ADDR x, GM_ADDR weight1, GM_ADDR bias1, GM_ADDR hidden,
                                         const MatMulV3BasicTilingData &up)
{
    using namespace AscendC;
    if constexpr (TransB) {
        // linear 权重布局 [N, K]（out, in）：kernel 内 transB
        MatmulV3Advanced::MatMulGeluMixWithoutQueActKernel<
            T, T, T, BiasT,
            Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor,
            MatmulV3Advanced::OP_TYPE_GELU_ERF>(x, weight1, bias1, hidden, up);
    } else {
        // canonical 权重布局 [K, N]
        MatmulV3Advanced::MatMulGeluMixWithoutQueActKernel<
            T, T, T, BiasT,
            Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor,
            MatmulV3Advanced::OP_TYPE_GELU_ERF>(x, weight1, bias1, hidden, up);
    }
}

template <typename T, typename BiasT, bool TransB>
__aicore__ inline void RunFfnFusedSiluUp(GM_ADDR x, GM_ADDR weight1, GM_ADDR bias1, GM_ADDR hidden,
                                         const MatMulV3BasicTilingData &up)
{
    using namespace AscendC;
    if constexpr (TransB) {
        // linear 权重布局 [N, K]（out, in）：kernel 内 transB
        MatmulV3Advanced::MatMulSiluMixWithoutQueActKernel<
            T, T, T, BiasT,
            Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor>(
            x, weight1, bias1, hidden, up);
    } else {
        // canonical 权重布局 [K, N]
        MatmulV3Advanced::MatMulSiluMixWithoutQueActKernel<
            T, T, T, BiasT,
            Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor>(
            x, weight1, bias1, hidden, up);
    }
}

// swiglu 前级：gate 半块 matmul + bias，输出 fp32（供 up 半块 epilogue 读取）
template <typename T, typename BiasT, bool TransB>
__aicore__ inline void RunFfnRawGateUp(GM_ADDR x, GM_ADDR weight1, GM_ADDR bias1, GM_ADDR gate,
                                       const MatMulV3BasicTilingData &up)
{
    using namespace AscendC;
    if constexpr (TransB) {
        MatmulV3Advanced::MatMulRawMixWithoutQueActKernel<
            T, T, float, BiasT,
            Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor>(
            x, weight1, bias1, gate, up);
    } else {
        MatmulV3Advanced::MatMulRawMixWithoutQueActKernel<
            T, T, float, BiasT,
            Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor>(
            x, weight1, bias1, gate, up);
    }
}

// swiglu 后级：up 半块 matmul + bias，AIV epilogue 读 gate(fp32) 算 silu(gate)*up 并写 hidden(bf16)
template <typename T, typename BiasT, bool TransB>
__aicore__ inline void RunFfnSwigluUp(GM_ADDR x, GM_ADDR weight1, GM_ADDR bias1, GM_ADDR gate, GM_ADDR hidden,
                                      const MatMulV3BasicTilingData &up)
{
    using namespace AscendC;
    if constexpr (TransB) {
        MatmulV3Advanced::MatMulSwigluMixWithoutQueActKernel<
            T, T, T, BiasT,
            Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor>(
            x, weight1, bias1, gate, hidden, up);
    } else {
        MatmulV3Advanced::MatMulSwigluMixWithoutQueActKernel<
            T, T, T, BiasT,
            Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor>(
            x, weight1, bias1, gate, hidden, up);
    }
}

// swiglu 单 matmul：一次 up matmul（N=2H），epilogue 内 silu(gate)*up -> hidden
template <typename T, typename BiasT, bool TransB>
__aicore__ inline void RunFfnSwigluSingleUp(GM_ADDR x, GM_ADDR weight1, GM_ADDR bias1, GM_ADDR hidden,
                                            const MatMulV3BasicTilingData &up)
{
    using namespace AscendC;
    if constexpr (TransB) {
        MatmulV3Advanced::MatMulSwigluSingleMixWithoutQueActKernel<
            T, T, T, BiasT,
            Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor>(
            x, weight1, bias1, hidden, up);
    } else {
        static_assert(TransB, "swiglu single only supports transB (linear [N,K] weight)");
    }
}

template <typename T, typename BiasT, bool TransB, uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void RunFfnDownMM(GM_ADDR hidden, GM_ADDR weight2, GM_ADDR bias2, GM_ADDR y,
                                    const MatMulV3BasicTilingData &down)
{
    using namespace AscendC;
    if constexpr (TransB) {
        MatmulV3Advanced::MatMulActKernel<
            T, T, T, BiasT,
            Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor,
            FULL_LOAD_MODE>(
            hidden, weight2, bias2, y, nullptr, down, 1);
    } else {
        MatmulV3Advanced::MatMulActKernel<
            T, T, T, BiasT,
            Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor,
            FULL_LOAD_MODE>(
            hidden, weight2, bias2, y, nullptr, down, 1);
    }
}

// down L1 全载
template <typename T, typename BiasT, bool TransB>
__aicore__ inline void RunFfnDownMMFullLoad(GM_ADDR hidden, GM_ADDR weight2, GM_ADDR bias2, GM_ADDR y,
                                            const MatMulV3BasicTilingData &down)
{
    if (down.fullLoad == 1) {
        RunFfnDownMM<T, BiasT, TransB, 1>(hidden, weight2, bias2, y, down);
    } else if (down.fullLoad == 2) {
        RunFfnDownMM<T, BiasT, TransB, 2>(hidden, weight2, bias2, y, down);
    } else {
        RunFfnDownMM<T, BiasT, TransB, 0>(hidden, weight2, bias2, y, down);
    }
}

// StreamK split-K
template <typename T, typename BiasT, bool TransB>
__aicore__ inline void RunFfnDownStreamK(GM_ADDR hidden, GM_ADDR weight2, GM_ADDR bias2, GM_ADDR y, GM_ADDR ws,
                                         const MatMulV3BasicTilingData &down)
{
    using namespace AscendC;
    if constexpr (TransB) {
        MatMulStreamKActKernel<
            T, T, T, BiasT,
            Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor,
            Cmct::Gemm::MatMulL0C2Out::ON_THE_FLY>(hidden, weight2, bias2, y, ws, down, 1);
    } else {
        MatMulStreamKActKernel<
            T, T, T, BiasT,
            Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor,
            Cmct::Gemm::MatMulL0C2Out::ON_THE_FLY>(hidden, weight2, bias2, y, ws, down, 1);
    }
}

} // namespace

template <typename T, typename BiasT, uint8_t ACT, uint8_t MODE>
__aicore__ inline void FfnArch35KernelImpl(__gm__ uint8_t *x, __gm__ uint8_t *weight1, __gm__ uint8_t *weight2,
                                           __gm__ uint8_t *expertTokens, __gm__ uint8_t *bias1, __gm__ uint8_t *bias2,
                                           __gm__ uint8_t *scale, __gm__ uint8_t *offset, __gm__ uint8_t *deqScale1,
                                           __gm__ uint8_t *deqScale2, __gm__ uint8_t *antiquant_scale1,
                                           __gm__ uint8_t *antiquant_scale2, __gm__ uint8_t *antiquant_offset1,
                                           __gm__ uint8_t *antiquant_offset2, __gm__ uint8_t *y,
                                           __gm__ uint8_t *workSpace, __gm__ uint8_t *tiling)
{
    using namespace AscendC;
    GET_TILING_DATA(tiling_data, tiling);
    const FFNTilingData *__restrict ffn_tiling_data = &tiling_data;

    __gm__ uint8_t *user1 = GetUserWorkspace(workSpace);
    const MatMulV3BasicTilingData up = MakeFfnMMTiling(ffn_tiling_data, true);
    const MatMulV3BasicTilingData down = MakeFfnMMTiling(ffn_tiling_data, false);
    __gm__ T *hidden = reinterpret_cast<__gm__ T *>(user1 + ffn_tiling_data->hiddenOffset);
    constexpr bool isSilu = (ACT == FFN_TPL_ACT_SILU);
    constexpr bool isSwiglu = (ACT == FFN_TPL_ACT_SWIGLU);
    const bool hasBias = (ffn_tiling_data->hasBias != 0);
    if constexpr (isSwiglu) {
        // swiglu 单 matmul：up.n = 2H；旧三段：up.n = H
        const uint32_t hiddenCols = (ffn_tiling_data->swigluSingle != 0) ? (up.n / 2) : up.n;
        __gm__ float *gate = reinterpret_cast<__gm__ float *>(user1);
        const uint64_t w1HalfBytes = static_cast<uint64_t>(hiddenCols) * up.k * sizeof(T);
        const uint64_t b1HalfBytes = static_cast<uint64_t>(hiddenCols) * sizeof(BiasT);
        if (ffn_tiling_data->transB == 0) {
            // canonical [K,2H] 的右半列切片暂未实现（需要 B 列 stride）；tiling 侧也应已拒绝
            return;
        }
        if (ffn_tiling_data->swigluSingle != 0) {
            // 单 matmul：一次 up(N=2H)，B 装载交错 gate/up，epilogue 内 silu(gate)*up -> hidden
            RunFfnSwigluSingleUp<T, BiasT, true>(x, weight1, hasBias ? bias1 : nullptr,
                                                  reinterpret_cast<__gm__ uint8_t *>(hidden), up);
        } else {
            // 阶段1a：gate（左半 [H,K]）-> fp32 gate
            RunFfnRawGateUp<T, BiasT, true>(x, weight1, hasBias ? bias1 : nullptr,
                                             reinterpret_cast<__gm__ uint8_t *>(gate), up);
            SyncAll<false>();
            // 阶段1b：up（右半 [H,K]），epilogue 内 silu(gate)*up -> hidden
            RunFfnSwigluUp<T, BiasT, true>(x, weight1 + w1HalfBytes,
                                            hasBias ? (bias1 + b1HalfBytes) : nullptr,
                                            reinterpret_cast<__gm__ uint8_t *>(gate),
                                            reinterpret_cast<__gm__ uint8_t *>(hidden), up);
        }
        SyncAll<false>();
        // 阶段2：AIC 执行 down matmul + bias2 -> y
        if ASCEND_IS_AIC {
            RunFfnDownMMFullLoad<T, BiasT, true>(reinterpret_cast<__gm__ uint8_t *>(hidden), weight2,
                                                  hasBias ? bias2 : nullptr, y, down);
            PipeBarrier<PIPE_ALL>();
        }
        return;
    }
    // 阶段1：up matmul + bias1 + gelu/silu（AIC mmad + AIV epilogue）。
    if (ffn_tiling_data->transB != 0) {
        if constexpr (isSilu) {
            RunFfnFusedSiluUp<T, BiasT, true>(x, weight1, hasBias ? bias1 : nullptr,
                                               reinterpret_cast<__gm__ uint8_t *>(hidden), up);
        } else {
            RunFfnFusedGeluUp<T, BiasT, true>(x, weight1, hasBias ? bias1 : nullptr,
                                               reinterpret_cast<__gm__ uint8_t *>(hidden), up);
        }
    } else {
        if constexpr (isSilu) {
            RunFfnFusedSiluUp<T, BiasT, false>(x, weight1, hasBias ? bias1 : nullptr,
                                                reinterpret_cast<__gm__ uint8_t *>(hidden), up);
        } else {
            RunFfnFusedGeluUp<T, BiasT, false>(x, weight1, hasBias ? bias1 : nullptr,
                                                reinterpret_cast<__gm__ uint8_t *>(hidden), up);
        }
    }
    // 阶段2：全核（AIC+AIV）全局屏障
    SyncAll<false>();
    if constexpr (MODE == FFN_TPL_MODE_STREAMK) {
        __gm__ uint8_t *downWs = user1 + ffn_tiling_data->hiddenOffset +
                                 static_cast<size_t>(ffn_tiling_data->hiddenRows) *
                                     ffn_tiling_data->hiddenCols * sizeof(uint16_t) +
                                 128;
        if (ffn_tiling_data->transB != 0) {
            RunFfnDownStreamK<T, BiasT, true>(reinterpret_cast<__gm__ uint8_t *>(hidden), weight2,
                                               hasBias ? bias2 : nullptr, y, downWs, down);
        } else {
            RunFfnDownStreamK<T, BiasT, false>(reinterpret_cast<__gm__ uint8_t *>(hidden), weight2,
                                                hasBias ? bias2 : nullptr, y, downWs, down);
        }
        if ASCEND_IS_AIC {
            PipeBarrier<PIPE_ALL>();
        }
        return;
    }
    if ASCEND_IS_AIC {
        // 阶段3：down matmul + bias2（AIC-only）
        if (ffn_tiling_data->transB != 0) {
            RunFfnDownMMFullLoad<T, BiasT, true>(reinterpret_cast<__gm__ uint8_t *>(hidden), weight2,
                                                  hasBias ? bias2 : nullptr, y, down);
        } else {
            RunFfnDownMMFullLoad<T, BiasT, false>(reinterpret_cast<__gm__ uint8_t *>(hidden), weight2,
                                                   hasBias ? bias2 : nullptr, y, down);
        }
        PipeBarrier<PIPE_ALL>();
    }
}

// arch35 fused 入口：TilingKey 模板化。
template <uint8_t DTYPE, uint8_t ACT, uint8_t MODE>
__global__ __aicore__ void ffn(__gm__ uint8_t *x, __gm__ uint8_t *weight1, __gm__ uint8_t *weight2,
                               __gm__ uint8_t *expertTokens, __gm__ uint8_t *bias1, __gm__ uint8_t *bias2,
                               __gm__ uint8_t *scale, __gm__ uint8_t *offset, __gm__ uint8_t *deqScale1,
                               __gm__ uint8_t *deqScale2, __gm__ uint8_t *antiquant_scale1,
                               __gm__ uint8_t *antiquant_scale2, __gm__ uint8_t *antiquant_offset1,
                               __gm__ uint8_t *antiquant_offset2, __gm__ uint8_t *y,
                               __gm__ uint8_t *workSpace, __gm__ uint8_t *tiling)
{
    InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    using T = std::conditional_t<DTYPE == FFN_TPL_DTYPE_BF16, bfloat16_t, half>;
    const bool biasBf16 = (reinterpret_cast<__gm__ const FFNTilingData *>(tiling)->biasIsBf16 != 0);
    const bool biasFp16 = (reinterpret_cast<__gm__ const FFNTilingData *>(tiling)->biasIsFp16 != 0);
    if constexpr (DTYPE == FFN_TPL_DTYPE_FP16) {
        if (biasFp16) {
            FfnArch35KernelImpl<half, half, ACT, MODE>(x, weight1, weight2, expertTokens, bias1, bias2, scale,
                                                       offset, deqScale1, deqScale2, antiquant_scale1,
                                                       antiquant_scale2, antiquant_offset1, antiquant_offset2, y,
                                                       workSpace, tiling);
        } else {
            FfnArch35KernelImpl<half, float, ACT, MODE>(x, weight1, weight2, expertTokens, bias1, bias2, scale,
                                                        offset, deqScale1, deqScale2, antiquant_scale1,
                                                        antiquant_scale2, antiquant_offset1, antiquant_offset2, y,
                                                        workSpace, tiling);
        }
    } else if (biasBf16) {
        FfnArch35KernelImpl<bfloat16_t, bfloat16_t, ACT, MODE>(x, weight1, weight2, expertTokens, bias1, bias2, scale,
                                                               offset, deqScale1, deqScale2, antiquant_scale1,
                                                               antiquant_scale2, antiquant_offset1,
                                                               antiquant_offset2, y, workSpace, tiling);
    } else {
        FfnArch35KernelImpl<bfloat16_t, float, ACT, MODE>(x, weight1, weight2, expertTokens, bias1, bias2, scale,
                                                          offset, deqScale1, deqScale2, antiquant_scale1,
                                                          antiquant_scale2, antiquant_offset1, antiquant_offset2, y,
                                                          workSpace, tiling);
    }
}
