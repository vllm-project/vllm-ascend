/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_ffn_combine.h
 * \brief
 */

#ifndef DISPATCH_FFN_COMBINE_H
#define DISPATCH_FFN_COMBINE_H

using namespace AscendC;

#include "kernel_operator.h"

#include "utils/moe_distribute_base.h"

#include "dispatch_ffn_combine_tiling.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "utils/select_helper.hpp"
#include "utils/const_args.hpp"
#include "dispatch_ffn_combine_kernel.hpp"
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"

using namespace Catlass;

namespace DispatchFFNCombineImpl {
#define TemplateMMA2AClass typename AType_, typename BType_, typename CType_, bool TB_, bool Nz_
#define TemplateMMA2ACFunc AType_, BType_, CType_, TB_, Nz_

using namespace AscendC;
template <TemplateMMA2AClass>
class DispatchFFNCombine {
public:
    __aicore__ inline DispatchFFNCombine() {};
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM, GM_ADDR scale2GM,
                                GM_ADDR probs, GM_ADDR outGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();


private:
    GM_ADDR xGM_;
    GM_ADDR weight1GM_;
    GM_ADDR weight2GM_;
    GM_ADDR expertIdGM_;
    GM_ADDR scale1GM_;
    GM_ADDR scale2GM_;
    GM_ADDR probs_;
    GM_ADDR outGM_;
    GM_ADDR workspaceGM_;

    GM_ADDR moeInitRoutingQuantV2Scale = nullptr;
    GM_ADDR moeInitRoutingQuantV2Offset = nullptr;
    GM_ADDR expertTokensBeforeCapacity = nullptr;


    TBuf<AscendC::TPosition::VECCALC> uBuf_;

    int32_t rank;
    int32_t rankSize;
    int32_t aivNum;
    
    int32_t m0;
    int32_t k0;
    int32_t n0;
    int32_t swizzlOffset;
    int32_t swizzlDirect;
    int32_t ubMoveNum;
    int32_t pValue;

    int32_t commNpuSplit;
    int32_t commDataSplit;
    int32_t lenPerLoop;

    int32_t m;
    int32_t k;
    int32_t n;
    int32_t topK;
    int32_t expertPerRank;
    int32_t maxOutputSize;
    int32_t EP;
    int32_t listLen;

    optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
    uint64_t initRoutingQuantTilingKey;

    // Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;

};


template <TemplateMMA2AClass>
__aicore__ inline void DispatchFFNCombine<TemplateMMA2ACFunc>::Init(GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM, GM_ADDR scale2GM,
                                                                    GM_ADDR probs, GM_ADDR outGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(DispatchFFNCombineTilingData);
    auto tiling = (__gm__ DispatchFFNCombineTilingData*)tilingGM;
    GET_TILING_DATA(tilingData, tilingGM);

    xGM_ = xGM;
    weight1GM_ = weight1GM;
    weight2GM_ = weight2GM;
    expertIdGM_ = expertIdGM;
    scale1GM_ = scale1GM;
    scale2GM_ = scale2GM;
    probs_ = probs;

    outGM_ = outGM;

    workspaceGM_ = workspaceGM;

    aivNum = tilingData.dispatchFFNCombineInfo.aivNum;

    m = tilingData.dispatchFFNCombineInfo.M;
    k = tilingData.dispatchFFNCombineInfo.K;
    n = tilingData.dispatchFFNCombineInfo.N;
    EP =  tilingData.dispatchFFNCombineInfo.worldSize;
    topK = tilingData.dispatchFFNCombineInfo.topK;
    expertPerRank = tilingData.dispatchFFNCombineInfo.expertPerRank;
    maxOutputSize = tilingData.dispatchFFNCombineInfo.maxOutputSize;
    listLen = tilingData.dispatchFFNCombineInfo.listLen;

    m0 = tilingData.cocTiling.m0;
    k0 = tilingData.cocTiling.k0;
    n0 = tilingData.cocTiling.n0;
    swizzlDirect = tilingData.cocTiling.swizzleDirect;
    swizzlOffset = tilingData.cocTiling.swizzleOffset;
    ubMoveNum = tilingData.cocTiling.ubMoveNum;
    pValue = tilingData.cocTiling.pValue;
    commNpuSplit = tilingData.cocTiling.commNpuSplit;
    commDataSplit = tilingData.cocTiling.commDataSplit;
    lenPerLoop = tilingData.cocTiling.lenPerLoop;
    moeInitRoutingQuantV2TilingData = tilingData.cocTiling.moeInitRoutingQuantV2TilingData;
    initRoutingQuantTilingKey = tilingData.cocTiling.initRoutingQuantTilingKey;

    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    __gm__ HcclOpResParamCustom *WinContext_{nullptr};
    WinContext_ = (__gm__ HcclOpResParamCustom *)contextGM0;

    rank = WinContext_->localUsrRankId;
    rankSize = WinContext_->rankSize;
}

template <TemplateMMA2AClass>
__aicore__ inline void DispatchFFNCombine<TemplateMMA2ACFunc>::Process()
{
    // Define ArchTag
    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;

    uint32_t k2 = n/2;
    uint32_t n2 = k;

    int64_t activeNum = 0;
    int64_t expertCapacity = 0;
    int64_t expertNum = expertPerRank * EP;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t quantMode = 1;

    using LayoutA = layout::RowMajor;
    using LayoutB = typename std::conditional<
        Nz_,
        layout::zN,
        typename std::conditional<TB_, layout::ColumnMajor, layout::RowMajor>::type
    >::type;

    LayoutB layoutB1 = LayoutBInitializer<LayoutB, BType_>::create(k, n);
    LayoutB layoutB2 = LayoutBInitializer<LayoutB, BType_>::create(k2, n2);
    using LayoutC = layout::RowMajor;
    using L1TileShape = GemmShape<128, 256, 512>;   // M, N, K

    constexpr uint32_t workspaceStages = 2;
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;

    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsyncFixpipe<
        preloadStages,
        l1Stages, l0AStages, l0BStages, l0CStages,
        enableUnitFlag, enableShuffleK
    >;

    using L0TileShape = GemmShape<128, 256, 128>;
    using AType = Gemm::GemmType<int8_t, layout::RowMajor>;
    using BType = Gemm::GemmType<int8_t, LayoutB>;
    using CType = Gemm::GemmType<float16_t, layout::RowMajor>;
    using D1Type = Gemm::GemmType<int8_t, layout::RowMajor>;

    using D2Type = typename std::conditional<
        std::is_same_v<CType_, bfloat16_t>, 
        Gemm::GemmType<bfloat16_t, layout::RowMajor>,
        Gemm::GemmType<CType_, layout::RowMajor>
        >::type;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    constexpr uint32_t ubStages = 2;

    using EpilogueDispatchPolicy1 = Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuant<ubStages>;

    using ScaleType = Gemm::GemmType<uint64_t, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using ElementMulType = Gemm::GemmType<float, layout::RowMajor>;
    using TileElemWiseMuls = Epilogue::Tile::TileElemWiseMuls<ArchTag, ElementMulType, 0>;

    using TileCopy1 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D1Type>;
    using BlockEpilogue1 = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy1, CType, PerTokenScaleType,
        D1Type, TileElemWiseMuls, TileCopy1>;

    using EpilogueDispatchPolicy2 = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using TileCopy2 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D2Type>;
    using BlockEpilogue2 = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType,PerTokenScaleType,
        D2Type, TileCopy2>;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
    using ElementGroupList = int64_t;
    using MatmulKernel = Gemm::Kernel::DispatchFFNCombineKernel<BlockMmad,
        BlockScheduler, ElementGroupList, BlockEpilogue1, BlockEpilogue2>;

    LayoutA layoutA1{static_cast<uint32_t>(m), static_cast<uint32_t>(k)};
    LayoutA layoutA2{static_cast<uint32_t>(m), static_cast<uint32_t>(k2)};
    layout::VectorLayout layoutScale1{static_cast<uint32_t>(n)};
    layout::VectorLayout layoutScale2{static_cast<uint32_t>(n2)};
    layout::RowMajor layoutD1{static_cast<uint32_t>(maxOutputSize), static_cast<uint32_t>(k2)};
    layout::RowMajor layoutD2{static_cast<uint32_t>(m*topK), static_cast<uint32_t>(n2)};
    // Prepare params

    GemmCoord problemShape{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};

    uint32_t epilogueCoreNum = aivNum / 2;
    uint32_t epilogueGranularity = expertPerRank - 1;

    typename MatmulKernel::Params params{
        problemShape, static_cast<uint32_t>(EP), static_cast<uint32_t>(listLen), static_cast<uint32_t>(expertPerRank), static_cast<uint32_t>(maxOutputSize),
        static_cast<uint32_t>(rank), static_cast<uint32_t>(rankSize),
        static_cast<uint32_t>(topK), initRoutingQuantTilingKey,
        epilogueCoreNum, epilogueGranularity,
        xGM_, layoutA1, layoutA2,
        weight1GM_, layoutB1,
        weight2GM_, layoutB2,
        scale1GM_, layoutScale1,
        scale2GM_, layoutScale2,
        outGM_, layoutD1, layoutD2,
        expertIdGM_, moeInitRoutingQuantV2Scale, moeInitRoutingQuantV2Offset,
        expertTokensBeforeCapacity, probs_,
        workspaceGM_, ubMoveNum, moeInitRoutingQuantV2TilingData};
    //Call kernel
    MatmulKernel kernel(params);
    kernel(params);
}

} // DispatchFFNCombineImpl
#endif // DISPATCH_FFN_COMBINE_H
