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
 * \file mat_mul_gelu_basic_cmct.h
 * \brief 参照 ops-nn fused_mat_mul 的 arch35/fused_mat_mul_gelu_basic_cmct.h：
 *        MatMulGeluMixWithoutQueActKernel，AIC matmul + bias，AIV FusionGelu epilogue。
 *        与官方差异：v1 目标 N>=16，不需要 small-N 的 GM workspace 路径，
 *        因此去掉官方 Params 中的 workspaceGmAddr/useGmWorkspace 两个字段。
 */
#pragma once

#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/epilogue/block_epilogue_elementwise.h"
#include "cmct/epilogue/fusion/fusion_silu.h"
#include "cmct/epilogue/fusion/fusion_swiglu.h"
#include "cmct/epilogue/fusion/fusion_swiglu_single.h"
#include "cmct/kernel/kernel_matmul_mix_without_que.h"
#include "cmct/tile/tile_copy.h"
#include "block_scheduler_aswt.h"

namespace MatmulV3Advanced {
using namespace Cmct;
using namespace Cmct::Gemm;

template <uint64_t OpType, class OutType, class InType>
struct GeluFusionOpSelector;

template <class OutType, class InType>
struct GeluFusionOpSelector<OP_TYPE_GELU_ERF, OutType, InType> {
    using type = Block::FusionGelu<OutType, InType, Block::GeluApproxiMate::ERF>;
};

template <class OutType, class InType>
struct GeluFusionOpSelector<OP_TYPE_GELU_TANH, OutType, InType> {
    using type = Block::FusionGelu<OutType, InType, Block::GeluApproxiMate::TANH>;
};

template <class OutType, class InType>
struct SiluFusionOpSelector {
    using type = Block::FusionSilu<OutType, InType>;
};

// 纯拷贝融合：up matmul 原始输出（fp32→bf16）不经激活直接写 GM（swiglu 前级）
template <typename DataTypeOut_, typename DataTypeIn_>
class FusionCopy {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    static constexpr bool kRegBaseAct = false;
    __aicore__ inline FusionCopy(){};

    struct Arguments {
        GM_ADDR inputGmAddr{nullptr};
    };

    struct Params {
        GM_ADDR inputGmAddr{nullptr};
    };

    int64_t stageSize_{0};

    template <class LocalTensor>
    __aicore__ inline void Init(Params const& params, LocalTensor ubTensor, int64_t ubCalcM, int64_t ubCalcN,
                                int64_t& ubOffset, int64_t& stageSize)
    {
        static constexpr int64_t stageNum = 1;
        int64_t lastUBSize = AscendC::TOTAL_UB_SIZE - ubOffset * sizeof(DataTypeIn);
        ASCENDC_ASSERT((lastUBSize > ubCalcN * sizeof(DataTypeIn)), {
            KERNEL_LOG(KERNEL_ERROR, , "ub size limit %ld, %ld!", lastUBSize, ubCalcN * sizeof(DataTypeIn));
        });
        stageSize_ = AscendC::Std::min(
            static_cast<int64_t>(lastUBSize / stageNum / sizeof(DataTypeIn) / ubCalcN * ubCalcN), ubCalcM * ubCalcN);
        ubOffset += 0;
        stageSize = stageSize_;
    }

    __aicore__ inline void operator()(const AscendC::LocalTensor<DataTypeIn>& srcLocal,
                                      AscendC::LocalTensor<DataTypeIn>& outputLocal, int64_t offset, int64_t curAivM,
                                      int64_t curAivN, int64_t strideN, int64_t stageSize)
    {
        TPipeSetWaitFlag<AscendC::HardEvent::MTE3_V>();
        AscendC::Adds(outputLocal, srcLocal, 0.0f, stageSize); // 与 gelu/silu 同类的 V 管“拷贝”
        AscendC::PipeBarrier<PIPE_V>();
    }

    __host_aicore__ static Params InitParams(Arguments const /* &args */, GM_ADDR /* workspaceGm */)
    {
        return {};
    }
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
          uint64_t GELU_OP_TYPE>
__aicore__ inline void MatMulGeluMixWithoutQueActKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                        const MatMulV3BasicTilingData& matMulTilingData,
                                                        int64_t batch = 0)
{
    static_assert(GELU_OP_TYPE == OP_TYPE_GELU_ERF || GELU_OP_TYPE == OP_TYPE_GELU_TANH, "unsupported gelu op type");
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    using AType = A_TYPE;
    using BType = B_TYPE;
    using OutType = C_TYPE;
    using MmadOutType = float;
    using BiasType = BIAS_TYPE;
    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;
    using BlockScheduler = BuiltInAswtScheduler<0>;
    using DispatchPolicy = MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, 0,
                                                      GELU_OP_TYPE>;
    using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, MmadOutType, LayoutC, BiasType, LayoutC,
                                              L1TileShape, L0TileShape, BlockScheduler, DispatchPolicy>;
    using FusionOp = typename GeluFusionOpSelector<GELU_OP_TYPE, OutType, MmadOutType>::type;
    using BlockEpilogue = Block::BlockEpilogueElementwise<L0TileShape, OutType, MmadOutType, FusionOp>;
    using ProblemShape = MatmulShape;
    using MatmulKernel = Kernel::KernelMatmulMixWithoutQue<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {{matMulTilingData.m, matMulTilingData.n, matMulTilingData.k, batch},
                     {aGM, bGM, yGM, biasGM},
                     {yGM, {nullptr}},
                     {&matMulTilingData}};
    MatmulKernel mm;
    mm(params);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT>
__aicore__ inline void MatMulSiluMixWithoutQueActKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                       const MatMulV3BasicTilingData& matMulTilingData,
                                                       int64_t batch = 0)
{
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    using AType = A_TYPE;
    using BType = B_TYPE;
    using OutType = C_TYPE;
    using MmadOutType = float;
    using BiasType = BIAS_TYPE;
    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;
    using BlockScheduler = BuiltInAswtScheduler<0>;
    using DispatchPolicy = MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, 0, OP_TYPE_SILU>;
    using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, MmadOutType, LayoutC, BiasType, LayoutC,
                                              L1TileShape, L0TileShape, BlockScheduler, DispatchPolicy>;
    using FusionOp = typename SiluFusionOpSelector<OutType, MmadOutType>::type;
    using BlockEpilogue = Block::BlockEpilogueElementwise<L0TileShape, OutType, MmadOutType, FusionOp>;
    using ProblemShape = MatmulShape;
    using MatmulKernel = Kernel::KernelMatmulMixWithoutQue<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {{matMulTilingData.m, matMulTilingData.n, matMulTilingData.k, batch},
                     {aGM, bGM, yGM, biasGM},
                     {yGM, {nullptr}},
                     {&matMulTilingData}};
    MatmulKernel mm;
    mm(params);
}

// swiglu 前级：up matmul + bias 原始输出（无激活），MIX 拆分避免 AIC-only 大 N 时 L0C 超限
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT>
__aicore__ inline void MatMulRawMixWithoutQueActKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                       const MatMulV3BasicTilingData& matMulTilingData,
                                                       int64_t batch = 0)
{
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    using AType = A_TYPE;
    using BType = B_TYPE;
    using OutType = C_TYPE;
    using MmadOutType = float;
    using BiasType = BIAS_TYPE;
    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;
    using BlockScheduler = BuiltInAswtScheduler<0>;
    // RAW fp32 gate：不能借 OP_TYPE_SILU（会把 fixpipe 的 UB 行宽按 16 补齐，
    // 而 FusionCopy<float> 按 8 对齐读，非 16 对齐 H 的尾块会逐行错位 32B）。
    // OP_TYPE_EMPTY 使 need16Align=false，fixpipe/ epilogue/GM 统一 8 对齐紧凑布局。
    using DispatchPolicy = MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, 0, OP_TYPE_EMPTY>;
    using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, MmadOutType, LayoutC, BiasType, LayoutC,
                                              L1TileShape, L0TileShape, BlockScheduler, DispatchPolicy>;
    using FusionOp = FusionCopy<MmadOutType, MmadOutType>;
    using BlockEpilogue = Block::BlockEpilogueElementwise<L0TileShape, OutType, MmadOutType, FusionOp>;
    using ProblemShape = MatmulShape;
    using MatmulKernel = Kernel::KernelMatmulMixWithoutQue<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {{matMulTilingData.m, matMulTilingData.n, matMulTilingData.k, batch},
                     {aGM, bGM, yGM, biasGM},
                     {yGM, {nullptr}},
                     {&matMulTilingData}};
    MatmulKernel mm;
    mm(params);
}

// swiglu 后级：up matmul + bias，AIV epilogue 从 GM 读 gate(fp32) 算 silu(gate)*up 并写 hidden(bf16)
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT>
__aicore__ inline void MatMulSwigluMixWithoutQueActKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR gateGM,
                                                          GM_ADDR yGM,
                                                          const MatMulV3BasicTilingData& matMulTilingData,
                                                          int64_t batch = 0)
{
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    using AType = A_TYPE;
    using BType = B_TYPE;
    using OutType = C_TYPE;
    using MmadOutType = float;
    using BiasType = BIAS_TYPE;
    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;
    using BlockScheduler = BuiltInAswtScheduler<0>;
    using DispatchPolicy = MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, 0, OP_TYPE_SILU>;
    using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, MmadOutType, LayoutC, BiasType, LayoutC,
                                              L1TileShape, L0TileShape, BlockScheduler, DispatchPolicy>;
    using FusionOp = Block::FusionSwiglu<MmadOutType, MmadOutType>;
    using BlockEpilogue = Block::BlockEpilogueElementwise<L0TileShape, OutType, MmadOutType, FusionOp>;
    using ProblemShape = MatmulShape;
    using MatmulKernel = Kernel::KernelMatmulMixWithoutQue<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {{matMulTilingData.m, matMulTilingData.n, matMulTilingData.k, batch},
                     {aGM, bGM, yGM, biasGM},
                     {yGM, {gateGM}},
                     {&matMulTilingData}};
    MatmulKernel mm;
    mm(params);
}

// swiglu 单 matmul：一次 up matmul（N=2H，B 装载按 16 行块交错 gate/up），
// AIV epilogue 在 UB 内算 silu(g_j)*u_j，输出 N=H。无 gate GM 往返。
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT>
__aicore__ inline void MatMulSwigluSingleMixWithoutQueActKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                                GM_ADDR yGM,
                                                                const MatMulV3BasicTilingData& matMulTilingData,
                                                                int64_t batch = 0)
{
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    using AType = A_TYPE;
    using BType = B_TYPE;
    using OutType = C_TYPE;
    using MmadOutType = float;
    using BiasType = BIAS_TYPE;
    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;
    using BlockScheduler = BuiltInAswtScheduler<0>;
    using DispatchPolicy = MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, 0, OP_TYPE_SWIGLU_SINGLE>;
    using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, MmadOutType, LayoutC, BiasType, LayoutC,
                                              L1TileShape, L0TileShape, BlockScheduler, DispatchPolicy>;
    using FusionOp = Block::FusionSwigluSingle<OutType, MmadOutType>;
    using BlockEpilogue = Block::BlockEpilogueElementwise<L0TileShape, OutType, MmadOutType, FusionOp>;
    using ProblemShape = MatmulShape;
    using MatmulKernel = Kernel::KernelMatmulMixWithoutQue<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {{matMulTilingData.m, matMulTilingData.n, matMulTilingData.k, batch},
                     {aGM, bGM, yGM, biasGM},
                     {yGM, {}},
                     {&matMulTilingData}};
    MatmulKernel mm;
    mm(params);
}
} // namespace MatmulV3Advanced
