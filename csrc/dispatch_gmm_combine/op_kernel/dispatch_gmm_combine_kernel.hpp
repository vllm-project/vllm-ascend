/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_GROUPED_MATMUL_DEQUANT_FIXPIPE_HPP
#define CATLASS_GEMM_KERNEL_GROUPED_MATMUL_DEQUANT_FIXPIPE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

#include "utils/block_mmad_preload_async_fixpipe_quant.hpp"
#include "utils/block_mmad_preload_fixpipe.hpp"
#include "utils/copy_gm_to_l1_custom.hpp"
#include "utils/copy_l0c_to_gm_custom.hpp"
#include "utils/block_epilogue_pertoken_row.hpp"
#include "utils/block_epilogue_pertoken_quant.hpp"
#include "utils/block_epilogue_pertoken_swiglu.hpp"
#include "utils/sync_util.hpp"
#include "kernel_operator.h"

#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"

#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2.cpp"
#include "moe_init_routing_quant_v2/moe_v2_fullload_dynamic_quant.h"

#include "unpermute/moe_token_unpermute.h"

using namespace AscendC;
// inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg{Catlass::L2_OFFSET};

namespace Catlass::Gemm::Kernel {


template <
    class BlockMmad_,
    class BlockScheduler_,
    class ElementGroupList_,
    class BlockEpilogue1_,
    class BlockEpilogue2_
>
class DispatchGmmCombineKernel {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using ElementScale = uint64_t;
    using LayoutScale = typename layout::VectorLayout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = typename layout::VectorLayout;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue1 = BlockEpilogue1_;
    using BlockEpilogue2 = BlockEpilogue2_;
    using ElementD1 = typename BlockEpilogue1::ElementD;
    using LayoutD1 = typename BlockEpilogue1::LayoutD;
    using ElementD2 = typename BlockEpilogue2::ElementD;
    using LayoutD2 = typename BlockEpilogue2::LayoutD;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        LayoutA layoutA2;
        __gm__ ElementB *ptrB1;
        LayoutB layoutB1;
        __gm__ ElementB *ptrB2;
        LayoutB layoutB2;
        __gm__ ElementScale *ptrScale1;
        LayoutScale layoutScale1;
        __gm__ ElementScale *ptrScale2;
        LayoutScale layoutScale2;
        __gm__ ElementD2 *ptrOutput;
        LayoutD1 layoutD1;
        LayoutD2 layoutD2;
        GM_ADDR ptrWorkspace;
        int32_t EP;
        int32_t expertPerRank;
        uint32_t maxOutputSize;
        uint32_t rank;
        uint32_t rankSize;
        GM_ADDR* symmetricPtr;
        int32_t ubMoveNum;
        //--------------
        GM_ADDR expertIdx;
        GM_ADDR moeInitRoutingQuantV2Scale;
        GM_ADDR moeInitRoutingQuantV2Offset;
        GM_ADDR expandedX;
        GM_ADDR expandedRowIdx;
        GM_ADDR expertTokensCountOrCumsum;
        GM_ADDR expertTokensBeforeCapacity;
        GM_ADDR dynamicQuantScale;
        GM_ADDR probs;
        int64_t activeNum;
        int64_t expertCapacity;
        int64_t expertNum;
        int64_t dropPadMode;
        int64_t expertTokensCountOrCumsumFlag;
        bool expertTokensBeforeCapacityFlag;
        int64_t quantMode;
        int64_t topK;
        uint64_t initRoutingQuantTilingKey;
        optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
        uint64_t buffSize;
        //--------------

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord problemShape_,
            uint32_t EP_, uint32_t expertPerRank_, uint32_t maxOutputSize_,
            uint32_t rank_, uint32_t rankSize_,

            int64_t activeNum_, int64_t expertCapacity_, int64_t expertNum_,
            int64_t dropPadMode_, int64_t expertTokensCountOrCumsumFlag_,
            bool expertTokensBeforeCapacityFlag_, int64_t quantMode_, int64_t topK_,
            uint64_t initRoutingQuantTilingKey_,
            GM_ADDR ptrA_, LayoutA layoutA_, LayoutA layoutA2_,
            GM_ADDR ptrB1_, LayoutB layoutB1_,
            GM_ADDR ptrB2_, LayoutB layoutB2_,
            GM_ADDR ptrScale1_, LayoutScale layoutScale1_,
            GM_ADDR ptrScale2_, LayoutScale layoutScale2_,
            GM_ADDR ptrOutput_, LayoutD2 layoutD1_, LayoutD2 layoutD2_,

            GM_ADDR expertIdx_, GM_ADDR moeInitRoutingQuantV2Scale_,
            GM_ADDR moeInitRoutingQuantV2Offset_,
            GM_ADDR expertTokensBeforeCapacity_, GM_ADDR probs_,
            GM_ADDR ptrWorkspace_, GM_ADDR* symmetricPtr_, int32_t ubMoveNum_,
            optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData_,
            uint64_t buffSize_

        ) : problemShape(problemShape_),
            EP(EP_), expertPerRank(expertPerRank_), maxOutputSize(maxOutputSize_),
            rank(rank_), rankSize(rankSize_),
            activeNum(activeNum_), expertCapacity(expertCapacity_), expertNum(expertNum_),
            dropPadMode(dropPadMode_), expertTokensCountOrCumsumFlag(expertTokensCountOrCumsumFlag_),
            expertTokensBeforeCapacityFlag(expertTokensBeforeCapacityFlag_), quantMode(quantMode_), topK(topK_),
            initRoutingQuantTilingKey(initRoutingQuantTilingKey_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_), layoutA2(layoutA2_),
            ptrB1(reinterpret_cast<__gm__ ElementB *>(ptrB1_)), layoutB1(layoutB1_),
            ptrB2(reinterpret_cast<__gm__ ElementB *>(ptrB2_)), layoutB2(layoutB2_),
            ptrScale1(reinterpret_cast<__gm__ ElementScale *>(ptrScale1_)), layoutScale1(layoutScale1_),
            ptrScale2(reinterpret_cast<__gm__ ElementScale *>(ptrScale2_)), layoutScale2(layoutScale2_),
            ptrOutput(reinterpret_cast<__gm__ ElementD2 *>(ptrOutput_)), layoutD1(layoutD1_), layoutD2(layoutD2_),
            expertIdx(expertIdx_), moeInitRoutingQuantV2Scale(moeInitRoutingQuantV2Scale_),
            moeInitRoutingQuantV2Offset(moeInitRoutingQuantV2Offset_), 
            expertTokensBeforeCapacity(expertTokensBeforeCapacity_), probs(probs_),
            ptrWorkspace(ptrWorkspace_), symmetricPtr(symmetricPtr_), ubMoveNum(ubMoveNum_),
            moeInitRoutingQuantV2TilingData(moeInitRoutingQuantV2TilingData_),
            buffSize(buffSize_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    DispatchGmmCombineKernel(Params const &params)
    {
        if ASCEND_IS_AIC {
            coreIdx = AscendC::GetBlockIdx();
            coreNum = AscendC::GetBlockNum();
        }

        if ASCEND_IS_AIV {
            coreIdx = get_block_idx() + get_subblockid() * get_block_num();
            coreNum = get_block_num() * get_subblockdim();
        }

        initBuffer(params);
    }

    CATLASS_DEVICE
    ~DispatchGmmCombineKernel()
    {
    }


    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        GMM1(params);

        AscendC::CrossCoreWaitFlag<0x2>(2);
        // AscendC::SyncAll<true>();

        GMM2(params);
    }


    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        Dispatch(params);
        AscendC::SyncAll<true>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(2);

        Combine(params);
    }

private:
    CATLASS_DEVICE void initBuffer(Params const &params) {
        workspaceInfo = WorkspaceInfo(params);
        peermemInfo = PeermemInfo(params);

        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrcumsumMM));
        cumsumSend.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrcumsumSend));

        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(workspaceInfo.ptrA));
        gmS.SetGlobalBuffer(params.ptrScale1);
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(workspaceInfo.ptrC));

        gmPermutedToken.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD1 *>(workspaceInfo.ptrPermutedToken));
        gmS2.SetGlobalBuffer(params.ptrScale2);
        gmC2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(workspaceInfo.ptrC2));

        gmPerTokenScale1.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPerTokenScale *>(workspaceInfo.ptrPerTokenScale));
        gmPerTokenScale2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPerTokenScale *>(workspaceInfo.ptrPerTokenScale2));

        tokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(peermemInfo.ptrPeerTokenPerExpert));
    }

    template<typename T>
    CATLASS_DEVICE void CopyGMToGM(
        AscendC::GlobalTensor<T> dst,
        AscendC::GlobalTensor<T> src,
        int32_t elemNum,
        int32_t ubMoveNum
    )
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

        using TType = Gemm::GemmType<T, layout::RowMajor>;
        using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
        using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
        CopyGmToUb copyGmToUb;
        CopyUbToGm copyUbToGm;
        constexpr int32_t BufferNum = 2;
        int tmpBufferSize = 32 * 1024 / sizeof(T);   // 32 KB
        AscendC::LocalTensor<T> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<T>(0);
        tmpBuffer1.SetSize(tmpBufferSize);
        int tmpBufferOffset = 96 * 1024; // half of UB
        AscendC::LocalTensor<T> tmpBuffer2 = resource.ubBuf.template GetBufferByByte<T>(tmpBufferOffset);
        tmpBuffer2.SetSize(tmpBufferSize);

        // [ReduceScatter] 2. Pre Interface Sync
        int pingpongId = 0;
        auto processCount = CeilDiv(elemNum, ubMoveNum);
        for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
            uint32_t curProcessNum = (processIndex == processCount - 1) ? elemNum - ubMoveNum * (processCount - 1) : ubMoveNum;
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
            auto processOffset = processIndex * ubMoveNum;

            auto inputOffset = processOffset;
            auto outputOffset = processOffset;
            // [ReduceScatter] 2. Pre Interface Sync
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            // [ReduceScatter] 3. Start shmem_mte_get_mem_nbi
            copyGmToUb(buf, src[inputOffset], layout::RowMajor{ 1, curProcessNum}, layout::RowMajor{1, curProcessNum});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            copyUbToGm(dst[outputOffset], buf, layout::RowMajor{ 1, curProcessNum}, layout::RowMajor{1, curProcessNum});

            // [ReduceScatter] 4. Post Interface Sync
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            pingpongId = (pingpongId + 1) % BufferNum;
        }
        // [ReduceScatter] 4. Post Interface Sync

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    CATLASS_DEVICE
    void GetCumsumForMMAIV(AscendC::GlobalTensor<int32_t> & tokenPerExpert, AscendC::GlobalTensor<int32_t> & result, uint32_t expertPerRank, uint32_t rankId, uint32_t EP)
    {
        int32_t expertPerRankAligned = (expertPerRank + 8 - 1) / 8 * 8;
        AscendC::LocalTensor<int32_t> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> tmpResult = resource.ubBuf.template GetBufferByByte<int32_t>(EP * expertPerRank * sizeof(int32_t));
        #define U16(x) static_cast<uint16_t>(x)

        AscendC::DataCopyPad(
            tmpBuffer1,
            tokenPerExpert[rankId * expertPerRank],
            {U16(EP), U16(expertPerRank * sizeof(int32_t)), U16((EP - 1) * expertPerRank * sizeof(int32_t)), 0},
            {}
        );
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t i = 1; i < EP; ++i) {
            AscendC::Add(tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[(i - 1) * expertPerRankAligned], expertPerRank);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopyPad(
            result,
            tmpBuffer1,
            {U16(EP), U16((expertPerRank) * sizeof(int32_t)), 0, 0}
        );
    }

    CATLASS_DEVICE
    void GetCumsumForSendRecv(AscendC::GlobalTensor<int32_t> & tokenPerExpert, AscendC::GlobalTensor<int32_t> & resultSend,
        uint32_t expertPerRank, uint32_t rankId, uint32_t EP)
    {
        layout::RowMajor commMatLayout{EP, EP * expertPerRank};
        for(uint32_t i = coreIdx; i < EP; i += coreNum) {
            int32_t sum = 0;
            for(uint32_t j = 0; j < rankId; ++j) {
                int32_t srcIdx = commMatLayout.GetOffset(MakeCoord(i, j * expertPerRank));
                for(uint32_t k = 0; k < expertPerRank; ++k) {
                    sum += tokenPerExpert(srcIdx + k);
                }
            }
            resultSend(i) = sum;
        }
    }

    CATLASS_DEVICE
    void GMM1(Params const &params){
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;
        uint32_t startCoreIdx = 0;
        uint32_t syncGroupIdx = 0;
        AscendC::CrossCoreWaitFlag<0x2>(0); // 等待aiv计算cumsumformm
        int64_t preCurrentmSum = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM >= params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            } 
            AscendC::GlobalTensor<ElementB> gmB1;
            gmB1.SetGlobalBuffer(params.ptrB1);
            if (currentM <= L1TileShape::M) {
                gmB1.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};
            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB1 = params.layoutB1;
            LayoutScale layoutScale = params.layoutScale1;
            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                for(;syncGroupIdx <= groupIdx; syncGroupIdx++) {
                    AscendC::CrossCoreWaitFlag<0x2>(0);
                }
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB1.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetS = groupIdx * params.problemShape.n() + blockCoord.n() * L1TileShape::N;   // 每个expert一组scale
                if (currentM > 0) {
                    blockMmad(
                        gmA[gmGroupOffsetA + gmOffsetA], layoutA,
                        gmB1[gmGroupOffsetB + gmOffsetB], layoutB1,
                        gmC[gmGroupOffsetC + gmOffsetC], layoutC,
                        gmS[gmOffsetS], layoutScale,
                        actualBlockShape
                    );
                }
            }
            preCurrentmSum += currentM;
            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();
            startCoreIdx = (startCoreIdx  + coreLoops) % coreNum;
        }
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(1);
    }

    CATLASS_DEVICE
    void GMM2(Params const &params) {
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);
    
        uint32_t n2 = params.problemShape.k();
        uint32_t k2 = params.problemShape.n() / 2;

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;

        uint32_t startCoreIdx = 0;

        AscendC::PipeBarrier<PIPE_ALL>();

        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;
        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM > params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            } 
            AscendC::GlobalTensor<ElementB> gmB2;
            gmB2.SetGlobalBuffer(params.ptrB2);
            if (currentM <= L1TileShape::M) {
                gmB2.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            GemmCoord inGroupProblemShape{currentM, n2, k2}; // M N K

            LayoutA layoutA = params.layoutA2.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB2 = params.layoutB2;
            LayoutScale layoutScale = params.layoutScale2;
            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                if (loopIdx + coreNum >= coreLoops) {
                    syncLoopIdx = groupIdx;
                }
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};

                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB2.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetS = groupIdx * n2 + blockCoord.n() * L1TileShape::N;   // 每个expert一组scale
                if (currentM > 0) {
                    blockMmad(
                        gmPermutedToken[gmGroupOffsetA + gmOffsetA], layoutA,
                        gmB2[gmGroupOffsetB + gmOffsetB], layoutB2,
                        gmC2[gmGroupOffsetC + gmOffsetC], layoutC,
                        gmS2[gmOffsetS], layoutScale,
                        actualBlockShape, syncLoopIdx, 3
                    );
                }
            }
            preCurrentmSum += currentM;
            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;

        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
        blockMmad.Finalize(params.expertPerRank - 1, 3);
    }


    CATLASS_DEVICE
    void Dispatch(Params const &params) {
        __gm__ void* srcPeermemPtr = params.symmetricPtr[params.rank];
        GM_ADDR localTokenPerExpert = peermemInfo.ptrPeerTokenPerExpert + params.rank * params.EP * params.expertPerRank * sizeof(int32_t);     // 把通信矩阵全部放到peermem
        //---initRouting------
        moe_init_routing_quant_v2<ElementD2>(reinterpret_cast<GM_ADDR> (params.ptrA), params.expertIdx, 
        params.moeInitRoutingQuantV2Scale, params.moeInitRoutingQuantV2Offset, peermemInfo.ptrA, 
        workspaceInfo.expandedRowIdx, localTokenPerExpert, params.expertTokensBeforeCapacity, 
        peermemInfo.ptrPeerPerTokenScale, params.ptrWorkspace, &params.moeInitRoutingQuantV2TilingData, params.initRoutingQuantTilingKey);
        AscendC::SyncAll<true>();

        int64_t localTokenPerExpertOffset = params.problemShape.m() * params.topK * params.problemShape.k() * sizeof(ElementA) +
                                            params.problemShape.m() * params.topK * sizeof(ElementPerTokenScale) +
                                            params.rank * params.EP * params.expertPerRank * sizeof(int32_t);

        // 本卡peermem->其他卡peermem，远端写
        if (coreIdx < params.EP && coreIdx != params.rank) {
            AscendC::GlobalTensor<int32_t> srcAddress;
            srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(localTokenPerExpert));
            AscendC::GlobalTensor<int32_t> dstAddress;
            __gm__ void* dstPeermemPtr = params.symmetricPtr[coreIdx] + localTokenPerExpertOffset;
            dstAddress.SetGlobalBuffer((__gm__ int32_t * )dstPeermemPtr);
            CopyGMToGM(dstAddress, srcAddress, params.EP * params.expertPerRank, params.ubMoveNum);
        }
        // 此处不需要syncall，因为远端写，写通信矩阵和写flag是一个核；  AscendC::SyncAll<true>();
        CrossRankSync(params.symmetricPtr, params.rank, params.rankSize, params.buffSize);

        GetCumsumForMMAIV(tokenPerExpert, cumsumMM, params.expertPerRank, params.rank, params.EP);

        AscendC::SyncAll<true>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
        //-------end-----------
        uint32_t curGroupOffset = 0;
        int32_t prevSumBeforeRank = 0;
        int32_t groupIdxDeq = 0;
        // TODO: 处理不了rank数大于卡数的情况
        if (coreIdx < params.EP) {
            for (int32_t i = 0; i < params.rank * params.expertPerRank; i++) {
                prevSumBeforeRank += tokenPerExpert(coreIdx * params.EP * params.expertPerRank + i);
            }
            m_prevSumBeforeRank[coreIdx] = prevSumBeforeRank;
        }
        int prevSum = prevSumBeforeRank;
        uint32_t prevGroupSum1 = 0;
        for (int32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            // 第i个core从第i个rank的peermem读数据
            groupIdxDeq = groupIdx - 2;
            for(int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
                uint32_t rowStart = (dstEpIdx == 0 ? 0 : cumsumMM((dstEpIdx - 1) * params.expertPerRank + groupIdx)) + prevGroupSum1;
                if (rowStart < params.maxOutputSize) {
                    uint32_t rows = tokenPerExpert(dstEpIdx * params.EP * params.expertPerRank + params.rank * params.expertPerRank + groupIdx);
                    if (rowStart + rows > params.maxOutputSize) {
                        rows = params.maxOutputSize - rowStart;
                    }
                    uint32_t rowSrc = prevSum;
                    prevSum += rows;
                    __gm__ void* peermemPtr = params.symmetricPtr[dstEpIdx];
                    AscendC::GlobalTensor<ElementA> gmRemoteA;
                    gmRemoteA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(peermemPtr));
                    AscendC::GlobalTensor<ElementPerTokenScale> gmRemotePerTokenScale;
                    gmRemotePerTokenScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPerTokenScale*>
                        (reinterpret_cast<GM_ADDR>(peermemPtr) + params.problemShape.m() * params.topK * params.problemShape.k() * sizeof(ElementA)));
                    MatrixCoord offsetA{rowStart, 0};
                    MatrixCoord shapeA{rows, params.problemShape.k()};
                    MatrixCoord offsetPeer{rowSrc, 0};
                    int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
                    int64_t gmOffsetPeer = params.layoutA.GetOffset(offsetPeer);
                    // 通信Data
                    CopyGMToGM(gmA[gmOffsetA], gmRemoteA[gmOffsetPeer], rows * params.problemShape.k(), params.ubMoveNum);
                    // 通信scale
                    CopyGMToGM(gmPerTokenScale1[rowStart], gmRemotePerTokenScale[rowSrc], rows, rows);
                }
            }

            prevGroupSum1 += cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0);   // V通知C当前轮的通信已完成
        }
        prevGroupSum1 = min(prevGroupSum1, params.maxOutputSize);

        // serial dequant, V等C完成计算
        AscendC::CrossCoreWaitFlag<0x2>(1);
        AscendC::SyncAll<true>();
        BlockEpilogue1 blockEpilogue(resource);
        uint32_t rowStartThisCore = 0;
        MatrixCoord offsetC{rowStartThisCore, 0};
        MatrixCoord shapeC{prevGroupSum1, params.problemShape.n()};
        LayoutC layoutC{prevGroupSum1, params.problemShape.n()};
        int64_t gmOffsetC = layoutC.GetOffset(offsetC);
        int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
        blockEpilogue(gmC[gmOffsetC], shapeC, gmPerTokenScale1[rowStartThisCore], gmPermutedToken[gmOffsetD], gmPerTokenScale2[rowStartThisCore]);
        blockEpilogue.Finalize();
    }

    CATLASS_DEVICE
    void Combine(Params const &params) {
        int32_t prevSumBeforeRank = 0;
        if (coreIdx < params.EP) {
            // for (int32_t i = 0; i < params.rank * params.expertPerRank; i++) {
            //     prevSumBeforeRank += tokenPerExpert(coreIdx * params.EP * params.expertPerRank + i);
            // }
            prevSumBeforeRank = m_prevSumBeforeRank[coreIdx];
        }

        int prevSum = prevSumBeforeRank;
        uint32_t n2 = params.problemShape.k();
        uint32_t k2 = params.problemShape.n() / 2;

        // TODO 计算tokenperexpert的cumsum
        typename BlockEpilogue2::Params epilogueParams{
            static_cast<int32_t>(params.EP),
            static_cast<int32_t>(params.expertPerRank),
            reinterpret_cast<__gm__ int32_t *>(params.ptrWorkspace)
        };
        BlockEpilogue2 blockEpilogue(resource, epilogueParams);
        int32_t prevGroupSum2 = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            AscendC::CrossCoreWaitFlag<0x2>(groupIdx / 8 + 3);
            AscendC::SyncAll<true>();

            int64_t ptrDOffset = params.problemShape.m() * params.topK * params.problemShape.k() * sizeof(ElementA) +
                                params.problemShape.m() * params.topK * sizeof(ElementPerTokenScale) +
                                params.EP * params.EP * params.expertPerRank * sizeof(int32_t);

            for(int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
                __gm__ void* dstPeermemPtr = params.symmetricPtr[dstEpIdx] + ptrDOffset;
                AscendC::GlobalTensor<ElementD2> gmRemotePeer;
                gmRemotePeer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD2*>(dstPeermemPtr));
                uint32_t srcRowOffset = (dstEpIdx == 0 ? 0 : cumsumMM((dstEpIdx - 1) * params.expertPerRank + groupIdx)) + prevGroupSum2;
                if (srcRowOffset < params.maxOutputSize) {
                    uint32_t dataRows = tokenPerExpert(dstEpIdx * params.EP * params.expertPerRank + groupIdx + params.rank * params.expertPerRank);
                    if (srcRowOffset + dataRows > params.maxOutputSize) {
                        dataRows = params.maxOutputSize - srcRowOffset;
                    }
                    uint32_t dstRowOffset = prevSum;
                    prevSum += dataRows;
                    MatrixCoord offsetC{srcRowOffset, 0};
                    MatrixCoord offsetPeer{dstRowOffset, 0};
                    MatrixCoord shapeC{dataRows, n2};
                    int64_t gmOffsetC = params.layoutD2.GetOffset(offsetC);
                    int64_t gmOffsetPeer = params.layoutD2.GetOffset(offsetPeer);
                    if constexpr (std::is_same_v<ElementA, int8_t>) {
                        blockEpilogue(gmC2[gmOffsetC], shapeC, gmPerTokenScale2[srcRowOffset], gmRemotePeer[gmOffsetPeer]);
                    } else {
                        blockEpilogue(gmC2[gmOffsetC], shapeC, gmRemotePeer[gmOffsetPeer]);
                    }
                }
            }
            prevGroupSum2 += cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);

        }
        blockEpilogue.Finalize();
        CrossRankSync(params.symmetricPtr, params.rank, params.EP, params.buffSize);
        MoeTokenUnpermuteTilingData tilingData;
        MoeTokenUnpermuteTiling(params.problemShape.m() * params.topK, n2, params.topK, tilingData, coreNum);
        KernelMoeTokenUnpermute<ElementD2, int32_t, float, true> kernelMoeTokenUnpermuteOp;

        kernelMoeTokenUnpermuteOp.Init(peermemInfo.ptrD, workspaceInfo.expandedRowIdx, params.probs, reinterpret_cast<GM_ADDR>(params.ptrOutput), &tilingData);
        kernelMoeTokenUnpermuteOp.Process();
    }

private:
  struct WorkspaceInfo {
        GM_ADDR ptrA;
        GM_ADDR ptrPerTokenScale;
        GM_ADDR ptrcumsumMM;
        GM_ADDR ptrcumsumSend;
        GM_ADDR ptrC;
        GM_ADDR ptrC2;
        GM_ADDR ptrPermutedToken;
        GM_ADDR ptrPerTokenScale2;
        GM_ADDR expandedRowIdx;
        GM_ADDR ptrTokenPerExpert;

        CATLASS_DEVICE
        WorkspaceInfo(){}

        CATLASS_DEVICE
        WorkspaceInfo(const Params & params) {
            uint32_t k2 = params.problemShape.n() / 2;
            uint32_t n2 = params.problemShape.k();
            int64_t workspaceOffset = 0;
            expandedRowIdx = params.ptrWorkspace;

            workspaceOffset += AlignUp(params.problemShape.m(), 256) * params.topK * sizeof(int32_t);
            ptrcumsumMM = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);
            ptrcumsumSend = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);
            ptrPerTokenScale = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            ptrPerTokenScale2 = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            ptrTokenPerExpert =  params.ptrWorkspace + workspaceOffset;

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);
            ptrC = params.ptrWorkspace + workspaceOffset;
            ptrC2 = ptrC;

            workspaceOffset += max(params.maxOutputSize * params.problemShape.n() * sizeof(ElementC),
                                    params.maxOutputSize * n2 * sizeof(ElementC));
            ptrA = params.ptrWorkspace + workspaceOffset;
            ptrPermutedToken = ptrA;

            workspaceOffset += max(params.maxOutputSize * params.problemShape.k() * sizeof(ElementA),
                    params.maxOutputSize * k2 * sizeof(ElementA));
        }
    };

    struct PeermemInfo {
        GM_ADDR ptrA;
        GM_ADDR ptrPeerPerTokenScale;
        GM_ADDR ptrPeerTokenPerExpert;
        GM_ADDR ptrD;

        CATLASS_DEVICE
        PeermemInfo(){}

        CATLASS_DEVICE
        PeermemInfo(const Params & params) {
            ptrA = params.symmetricPtr[params.rank];
            ptrPeerPerTokenScale = params.symmetricPtr[params.rank] + params.problemShape.m() * params.topK * params.problemShape.k() * sizeof(ElementA);
            ptrPeerTokenPerExpert = ptrPeerPerTokenScale + params.problemShape.m() * params.topK * sizeof(ElementPerTokenScale);
            ptrD = ptrPeerTokenPerExpert + params.EP * params.EP * params.expertPerRank * sizeof(int32_t);
        }
    };

    Arch::Resource<ArchTag> resource;

    uint32_t coreIdx;
    uint32_t coreNum;

    Params params;
    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;

    int64_t m_prevSumBeforeRank[32];

    AscendC::GlobalTensor<int32_t> cumsumMM;
    AscendC::GlobalTensor<int32_t> cumsumSend;

    AscendC::GlobalTensor<ElementA> gmA;
    AscendC::GlobalTensor<ElementC> gmC;
    AscendC::GlobalTensor<ElementScale> gmS;

    AscendC::GlobalTensor<ElementD1> gmPermutedToken;
    AscendC::GlobalTensor<ElementScale> gmS2;
    AscendC::GlobalTensor<ElementC> gmC2;

    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale1;
    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale2;

    AscendC::GlobalTensor<int32_t> tokenPerExpert;

};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_GROUPED_MATMUL_M_PER_TOKEN_DEQUANT_MULTISTAGE_WORKSPACE_HPP
