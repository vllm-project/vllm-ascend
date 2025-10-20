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
 * \file dispatch_gmm_combine.h
 * \brief
 */

#ifndef GMM_DISPATCH_COMBINE_H
#define GMM_DISPATCH_COMBINE_H

using namespace AscendC;

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "dispatch_gmm_combine_tiling.h"
#include "moe_distribute_base.h"

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
#include "catlass/status.hpp"

#include "dispatch_gmm_combine_kernel.hpp"
#include "utils/select_helper.hpp"



// #include "dispatch_policy_custom.hpp"
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"
using namespace Catlass;

namespace DispatchGmmCombineImpl {

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}
// MMA2A : DispatchGmmCombine
#define TemplateMMA2AClass typename AType_, typename BType_, typename CType_, bool TB_, bool Nz_
#define TemplateMMA2ACFunc AType_, BType_, CType_, TB_, Nz_

using namespace AscendC;
template <TemplateMMA2AClass>
class DispatchGmmCombine {
    constexpr static uint32_t BUFFER_NUM = 2U;                   // 多buf
    constexpr static uint32_t STATE_OFFSET = 512U;               // 状态空间偏移地址
    constexpr static uint32_t BLOCK_SIZE = 32U;
    constexpr static uint32_t B32_PER_BLOCK = 8U;
    constexpr static uint32_t B64_PER_BLOCK = 4U;
    constexpr static int32_t CUBE_MATRIX_SIZE_B16 = 256;                    // 16 * 16
    constexpr static int32_t L0AB_PINGPONG_BUFFER_SIZE = 32768;             // 32 KB
    constexpr static int32_t MAX_BLOCK_COUNT = 2;
    constexpr static int32_t FLAG_ZERO_IDX = 0;
    constexpr static int32_t FLAG_ONE_IDX = 1;
    constexpr static int32_t FLAG_VALUE = 1;
    constexpr static int32_t USED_UB_SIZE = 160 * 1024;
    constexpr static uint64_t STATE_WIN_OFFSET = 900 * 1024;
    constexpr static uint32_t UB_ALIGN = 32; // UB按32字节对齐
    constexpr static uint64_t WIN_STATE_OFFSET = 512 * 1024;
    constexpr static uint64_t MB_SIZE = 1024 * 1024UL;

public:
    __aicore__ inline DispatchGmmCombine() {};
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

    GM_ADDR dataAddrPerRank[16];
    GM_ADDR stateAddrPerRank[16];

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

    uint64_t buffSize;

    uint32_t dataState_{0};
    optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
    uint64_t initRoutingQuantTilingKey;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;

    __gm__ HcclOpResParam *WinContext_{nullptr};
};


template <TemplateMMA2AClass>
__aicore__ inline void DispatchGmmCombine<TemplateMMA2ACFunc>::Init(GM_ADDR xGM, GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR expertIdGM, GM_ADDR scale1GM, GM_ADDR scale2GM,
                                                                    GM_ADDR probs, GM_ADDR outGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(DispatchGmmCombineTilingData);
    auto tiling = (__gm__ DispatchGmmCombineTilingData*)tilingGM;
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

    aivNum = tilingData.dispatchGmmCombineInfo.aivNum;

    m = tilingData.dispatchGmmCombineInfo.M;
    k = tilingData.dispatchGmmCombineInfo.K;
    n = tilingData.dispatchGmmCombineInfo.N;
    EP =  tilingData.dispatchGmmCombineInfo.worldSize;
    topK = tilingData.dispatchGmmCombineInfo.topK;
    expertPerRank = tilingData.dispatchGmmCombineInfo.expertPerRank;
    maxOutputSize = tilingData.dispatchGmmCombineInfo.maxOutputSize;

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
    WinContext_ = (__gm__ HcclOpResParam *)contextGM0;

    rank = WinContext_->localUsrRankId;
    rankSize = WinContext_->rankSize;
    buffSize = WinContext_->winSize / MB_SIZE;

    for (int i = 0; i < rankSize; i++) {
        stateAddrPerRank[i] = (GM_ADDR)((i == rank) ? WinContext_->localWindowsIn :
                            ((HcclRankRelationResV2 *)(WinContext_->remoteRes[i].nextDevicePtr))->windowsIn);
    }
}

template <TemplateMMA2AClass>
__aicore__ inline void DispatchGmmCombine<TemplateMMA2ACFunc>::Process()
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

    // using EpilogueDispatchPolicy2 = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using EpilogueDispatchPolicy2 = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using TileCopy2 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D2Type>;
    using BlockEpilogue2 = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType,PerTokenScaleType,
        D2Type, TileCopy2>;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
    using ElementGroupList = int64_t;
    using MatmulKernel = Gemm::Kernel::DispatchGmmCombineKernel<BlockMmad,
        BlockScheduler, ElementGroupList, BlockEpilogue1, BlockEpilogue2>;

    LayoutA layoutA1{static_cast<uint32_t>(m), static_cast<uint32_t>(k)};
    LayoutA layoutA2{static_cast<uint32_t>(m), static_cast<uint32_t>(k2)};
    layout::VectorLayout layoutScale1{static_cast<uint32_t>(n)};
    layout::VectorLayout layoutScale2{static_cast<uint32_t>(n2)};
    layout::RowMajor layoutD1{static_cast<uint32_t>(maxOutputSize), static_cast<uint32_t>(k2)};
    layout::RowMajor layoutD2{static_cast<uint32_t>(m*topK), static_cast<uint32_t>(n2)};
    // Prepare params

    GemmCoord problemShape{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};

    typename MatmulKernel::Params params{
        problemShape, static_cast<uint32_t>(EP), static_cast<uint32_t>(expertPerRank), static_cast<uint32_t>(maxOutputSize),
        static_cast<uint32_t>(rank), static_cast<uint32_t>(rankSize),
        activeNum, expertCapacity, expertNum, dropPadMode, expertTokensCountOrCumsumFlag,
        expertTokensBeforeCapacityFlag, quantMode, static_cast<uint32_t>(topK), initRoutingQuantTilingKey,
        xGM_, layoutA1, layoutA2,
        weight1GM_, layoutB1,
        weight2GM_, layoutB2,
        scale1GM_, layoutScale1,
        scale2GM_, layoutScale2,
        outGM_, layoutD1, layoutD2,
        expertIdGM_, moeInitRoutingQuantV2Scale, moeInitRoutingQuantV2Offset,
        expertTokensBeforeCapacity, probs_,
        workspaceGM_, 
        stateAddrPerRank, ubMoveNum, moeInitRoutingQuantV2TilingData, buffSize};

    //Call kernel
    MatmulKernel kernel(params);
    kernel(params);
}

} // DispatchGmmCombineImpl
#endif // GMM_ALLTOALLV_H
