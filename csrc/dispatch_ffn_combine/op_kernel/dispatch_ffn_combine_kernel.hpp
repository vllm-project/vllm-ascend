/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DISPATCH_FFN_COMBINE_KERNEL_HPP
#define DISPATCH_FFN_COMBINE_KERNEL_HPP

#include "kernel_operator.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

#ifndef HCCL_COMM
    #include "block_mmad_preload_async_fixpipe_quant.hpp"
    #include "copy_gm_to_l1_custom.hpp"
    #include "copy_l0c_to_gm_custom.hpp"
    #include "block_epilogue_pertoken_row.hpp"
    #include "block_epilogue_pertoken_v2.hpp"
    #include "block_epilogue_pertoken_swiglu.hpp"
    #include "hccl_shmem.hpp"
    #include "const_args.hpp"
    #include "layout3d.hpp"
    #include "tiling/moe_init_routing_quant_v2_tiling.h"
    #include "moe_init_routing_quant_v2/moe_init_routing_quant_v2.cpp"
    #include "moe_init_routing_quant_v2/moe_v2_fullload_dynamic_quant.h"
    #include "moe_token_unpermute.h"
    #include "get_tensor_addr.hpp"
    inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg{Catlass::L2_OFFSET};
#else
    #include "utils/block_mmad_preload_async_fixpipe_quant.hpp"
    #include "utils/copy_gm_to_l1_custom.hpp"
    #include "utils/copy_l0c_to_gm_custom.hpp"
    #include "utils/block_epilogue_pertoken_row.hpp"
    #include "utils/block_epilogue_pertoken_v2.hpp"
    #include "utils/block_epilogue_pertoken_swiglu.hpp"
    #include "utils/hccl_shmem.hpp"
    #include "utils/const_args.hpp"
    #include "utils/layout3d.hpp"
    #include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"
    #include "moe_init_routing_quant_v2/moe_init_routing_quant_v2.cpp"
    #include "moe_init_routing_quant_v2/moe_v2_fullload_dynamic_quant.h"
    #include "unpermute/moe_token_unpermute.h"
    #include "utils/get_tensor_addr.hpp"
#endif

using namespace AscendC;

namespace Catlass::Gemm::Kernel {

constexpr uint16_t SYNCFLAGC2V = 9;
constexpr uint16_t SYNCFLAGV2C = 10;

template <
    class BlockMmad_,
    class BlockScheduler_,
    class ElementGroupList_,
    class BlockEpilogue1_,
    class BlockEpilogue2_
>
class DispatchFFNCombineKernel {
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
        GM_ADDR ptrB1;
        LayoutB layoutB1;
        GM_ADDR ptrB2;
        LayoutB layoutB2;
        GM_ADDR ptrScale1;
        LayoutScale layoutScale1;
        GM_ADDR ptrScale2;
        LayoutScale layoutScale2;
        __gm__ ElementD2 *ptrOutput;
        LayoutD1 layoutD1;
        LayoutD2 layoutD2;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrExpertTokenNums;
        int32_t EP;
        int32_t listLen;
        int32_t expertPerRank;
        uint32_t maxOutputSize;
        uint32_t rank;
        uint32_t rankSize;
        int32_t ubMoveNum;
        GM_ADDR symmetricPtr;
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
        int64_t topK;
        uint64_t initRoutingQuantTilingKey;
        uint32_t epilogueCoreNum;
        uint32_t epilogueGranularity;
        optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
        //--------------

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            GemmCoord problemShape_,
            uint32_t EP_, uint32_t listLen_, uint32_t expertPerRank_, uint32_t maxOutputSize_,
            uint32_t rank_, uint32_t rankSize_, int64_t topK_,
            uint64_t initRoutingQuantTilingKey_, uint32_t epilogueCoreNum_, uint32_t epilogueGranularity_,
            GM_ADDR ptrA_, LayoutA layoutA_, LayoutA layoutA2_,
            GM_ADDR ptrB1_, LayoutB layoutB1_,
            GM_ADDR ptrB2_, LayoutB layoutB2_,
            GM_ADDR ptrScale1_, LayoutScale layoutScale1_,
            GM_ADDR ptrScale2_, LayoutScale layoutScale2_,
            GM_ADDR ptrOutput_, LayoutD2 layoutD1_, LayoutD2 layoutD2_,
            GM_ADDR expertIdx_, GM_ADDR moeInitRoutingQuantV2Scale_,
            GM_ADDR moeInitRoutingQuantV2Offset_,
            GM_ADDR expertTokensBeforeCapacity_, GM_ADDR probs_,
            GM_ADDR ptrWorkspace_, GM_ADDR gmExpertTokenNums_, int32_t ubMoveNum_,
            optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData_
        ) : problemShape(problemShape_),
            EP(EP_), listLen(listLen_), expertPerRank(expertPerRank_), maxOutputSize(maxOutputSize_),
            rank(rank_), rankSize(rankSize_), topK(topK_),
            initRoutingQuantTilingKey(initRoutingQuantTilingKey_),
            epilogueCoreNum(epilogueCoreNum_), epilogueGranularity(epilogueGranularity_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_), layoutA2(layoutA2_),
            ptrB1(ptrB1_), layoutB1(layoutB1_),
            ptrB2(ptrB2_), layoutB2(layoutB2_),
            ptrScale1(ptrScale1_), layoutScale1(layoutScale1_),
            ptrScale2(ptrScale2_), layoutScale2(layoutScale2_),
            ptrOutput(reinterpret_cast<__gm__ ElementD2 *>(ptrOutput_)), layoutD1(layoutD1_), layoutD2(layoutD2_),
            expertIdx(expertIdx_), moeInitRoutingQuantV2Scale(moeInitRoutingQuantV2Scale_),
            moeInitRoutingQuantV2Offset(moeInitRoutingQuantV2Offset_), 
            expertTokensBeforeCapacity(expertTokensBeforeCapacity_), probs(probs_),
            ptrWorkspace(ptrWorkspace_), ptrExpertTokenNums(gmExpertTokenNums_), ubMoveNum(ubMoveNum_),
            moeInitRoutingQuantV2TilingData(moeInitRoutingQuantV2TilingData_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    DispatchFFNCombineKernel(Params const &params)
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
    ~DispatchFFNCombineKernel()
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
        AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C);
        GMM2(params);
    }


    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        DispatchAndCombine(params);
    }

private:
    CATLASS_DEVICE void initBuffer(Params const &params) {
        #ifndef HCCL_COMM
            shmem.initShmem(params.symmetricPtr, params.rank, params.rankSize);
        #endif
        workspaceInfo = WorkspaceInfo(params);
        peermemInfo = PeermemInfo(params, shmem);
        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrcumsumMM));
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(workspaceInfo.ptrA));
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(workspaceInfo.ptrC));
        gmPermutedToken.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD1 *>(workspaceInfo.ptrPermutedToken));
        gmC2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(workspaceInfo.ptrC2));
        gmPerTokenScale1.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPerTokenScale *>(workspaceInfo.ptrPerTokenScale));
        gmPerTokenScale2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPerTokenScale *>(workspaceInfo.ptrPerTokenScale2));
        tokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(shmem() + peermemInfo.offsetPeerTokenPerExpert));
        tokenPerExpertLayout = Layout3D(AlignUp(params.EP * params.expertPerRank, ALIGN_128), params.expertPerRank);
        preSumBeforeRank.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrSumBeforeRank));
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

    // Move tokens and scales together, then write them to different positions respectively
    template<typename T>
    CATLASS_DEVICE void CopyGMToGMPerToken(
        AscendC::GlobalTensor<T> dst,
        AscendC::GlobalTensor<float> dstScale,
        AscendC::GlobalTensor<T> src,
        int32_t rows,
        int32_t hiddenSize
    )
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
 
        constexpr int32_t BufferNum = 2;
        AscendC::LocalTensor<T> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<T>(0);
        constexpr int tmpBufferOffset = 96 * 1024; // half of UB
        AscendC::LocalTensor<T> tmpBuffer2 = resource.ubBuf.template GetBufferByByte<T>(tmpBufferOffset);
        uint32_t copyInNum = hiddenSize + ALIGN_512;
        // [ReduceScatter] 2. Pre Interface Sync
        int pingpongId = 0;
        for (uint32_t processIndex = 0; processIndex < rows; ++processIndex) {
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
            AscendC::LocalTensor<float> bufScale = buf[hiddenSize].template ReinterpretCast<float>();
            auto inputOffset = processIndex * copyInNum;
            auto outputOffset = processIndex * hiddenSize;
            // [ReduceScatter] 2. Pre Interface Sync
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            // [ReduceScatter] 3. Start shmem_mte_get_mem_nbi
            AscendC::DataCopy(buf, src[inputOffset], copyInNum);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::DataCopy(dst[outputOffset], buf, hiddenSize);
            AscendC::DataCopyPad(dstScale[processIndex], bufScale, {1, 4, 0, 0, 0});
 
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
            {U16(EP), U16(expertPerRank * sizeof(int32_t)), U16((AlignUp(EP * expertPerRank, 128) - expertPerRank) * sizeof(int32_t)), 0},
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
    void GMM1(Params const &params){
        icache_preload(8);
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);
        float aivFinishGroups = 0.0f;
        __gm__ float* aivFinishPtr = workspaceInfo.ptrSoftFlagBase + params.EP * FLAGSTRIDE;

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;
        uint32_t startCoreIdx = 0;
        uint32_t syncGroupIdx = 0;
        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;

        uint16_t syncgmmIdx = 0;
        AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT); // Wait for AIV to finish cumsum for matmul
        syncgmmIdx++;
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM >= params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            } 
            AscendC::GlobalTensor<ElementB> gmB1;
            AscendC::GlobalTensor<ElementScale> gmS;
            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            gmB1.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(GetTensorAddr<int8_t>(arrayGroupIdx, params.ptrB1)));
            gmS.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(GetTensorAddr<int64_t>(arrayGroupIdx, params.ptrScale1)));
            AscendC::PipeBarrier<PIPE_ALL>();
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
                    AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                    syncgmmIdx ++;
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
                int64_t gmOffsetS = blockCoord.n() * L1TileShape::N + (params.listLen == 1 ? groupIdx * params.problemShape.n() : 0);
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
 
            if ((groupIdx + 1) == params.epilogueGranularity  && (groupIdx < params.expertPerRank - 1)) {
                syncLoopIdx ++;
                if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                    blockMmad.SynchronizeBlock();
                }
                // Synchronization signal: GMM1 notifies SwiGLU [1]
                blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V);
            }

            preCurrentmSum += currentM;
            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();
            startCoreIdx = (startCoreIdx  + coreLoops) % coreNum;
        }

        for(;syncGroupIdx < params.expertPerRank; syncGroupIdx++) {
            AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmmIdx ++;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
        // Synchronization signal: GMM1 notifies SwiGLU [2]
        blockMmad.Finalize(syncLoopIdx + 1, SYNCFLAGC2V);
    }

    CATLASS_DEVICE
    void GMM2(Params const &params) {
        icache_preload(8);
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
        uint32_t lastDequantExpertNum = params.expertPerRank;

        if (params.epilogueGranularity < params.expertPerRank) {
            lastDequantExpertNum = params.expertPerRank - params.epilogueGranularity;
        }

        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM > params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            } 
            AscendC::GlobalTensor<ElementB> gmB2;
            AscendC::GlobalTensor<ElementScale> gmS2;
            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            gmB2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(GetTensorAddr<int8_t>(arrayGroupIdx, params.ptrB2)));
            gmS2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(GetTensorAddr<int64_t>(arrayGroupIdx, params.ptrScale2)));
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
            if (params.expertPerRank > lastDequantExpertNum && groupIdx + 1 == params.expertPerRank - lastDequantExpertNum) {
                AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C);
            }

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
                int64_t gmOffsetS = blockCoord.n() * L1TileShape::N + (params.listLen == 1 ? groupIdx * n2 : 0);   // One scale group per expert
                if (currentM > 0) {
                    blockMmad(
                            gmPermutedToken[gmGroupOffsetA + gmOffsetA], layoutA,
                            gmB2[gmGroupOffsetB + gmOffsetB], layoutB2,
                            gmC2[gmGroupOffsetC + gmOffsetC], layoutC,
                            gmS2[gmOffsetS], layoutScale,
                            actualBlockShape, syncLoopIdx, 0
                        );
                }
            }
            preCurrentmSum += currentM;
            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
    }


    CATLASS_DEVICE 
    void InitArithProgress(Params const &params) {
        AscendC::LocalTensor<float> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<float>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::Duplicate(tmpBuffer1, 0.0f, (params.EP + 1) * FLAGSTRIDE);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::GlobalTensor<float> flagGlobalBase;
        flagGlobalBase.SetGlobalBuffer(workspaceInfo.ptrSoftFlagBase);
        AscendC::DataCopy(flagGlobalBase, tmpBuffer1, (params.EP + 1) * FLAGSTRIDE);
    }


    CATLASS_DEVICE
    void CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(Params const &params, int64_t localTokenPerExpertOffset){
        uint32_t numPerCore = AlignUp(params.EP * params.expertPerRank, 128);
        AscendC::LocalTensor<int32_t> tmpBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> prevSumBuf = tmpBuffer[numPerCore];

        for(int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
            if (dstEpIdx == params.rank) {
                continue;
            }
            AscendC::GlobalTensor<int32_t> srcAddress;
            srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(shmem() + localTokenPerExpertOffset));
            AscendC::GlobalTensor<int32_t> dstAddress;
            __gm__ void* dstPeermemPtr = shmem(localTokenPerExpertOffset, coreIdx);
            dstAddress.SetGlobalBuffer((__gm__ int32_t * )dstPeermemPtr);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            using TType = Gemm::GemmType<int32_t, layout::RowMajor>;
            using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
            using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
            CopyGmToUb copyGmToUb;
            CopyUbToGm copyUbToGm;
            
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            
            copyGmToUb(tmpBuffer, srcAddress[0], 
                layout::RowMajor{ 1, numPerCore}, 
                layout::RowMajor{1, numPerCore});

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            copyUbToGm(dstAddress[0], tmpBuffer, 
                layout::RowMajor{ 1, numPerCore}, 
                layout::RowMajor{1, numPerCore});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        for(int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
            if (dstEpIdx != params.rank) {
                int32_t intPer512 = CACHE_LINE / sizeof(int);
                for(int32_t checkIdx = 0; checkIdx < AlignUp(params.EP * params.expertPerRank, 128); checkIdx += intPer512) {
                    __gm__ int32_t* sync_check = reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetPeerTokenPerExpert) + tokenPerExpertLayout(dstEpIdx, 0, checkIdx);
                    gm_signal_wait_until_ne(sync_check, 0);
                }
                AscendC::DataCopy(tmpBuffer, tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Adds(tmpBuffer, tmpBuffer, -0x800000, numPerCore);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], tmpBuffer, numPerCore);
            } else {
                AscendC::DataCopy(tmpBuffer, tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            int32_t prevSum = 0;
            int32_t j = 0;
            for (int32_t i = 0; i < (params.rank + 1) * params.expertPerRank; i++) {
                if (i >= params.rank * params.expertPerRank) {
                    prevSumBuf(j) = prevSum;
                    j++;
                }
                prevSum += tmpBuffer(i);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(preSumBeforeRank[dstEpIdx * params.expertPerRank], prevSumBuf,
            AscendC::DataCopyParams{1, static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)), 0, 0});
        }

        AscendC::SyncAll<true>();
    }

    CATLASS_DEVICE
    void ResetTokenPerExpert(int32_t num)
    {
        if (coreIdx != coreNum - 1) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::LocalTensor<int32_t> tmp = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::Duplicate(tmp, 0, num);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(tokenPerExpert, tmp, num);
    }

    CATLASS_DEVICE
    void UpdateAicFlags(const Params &params)
    {
        float flagBase = 1.0f * params.expertPerRank;
        __gm__ float* aicFinishPtr = workspaceInfo.ptrSoftFlagBase + params.EP * FLAGSTRIDE;
        float flag = 0.0f;
        float lastflag = -1.0f;
        AscendC::LocalTensor<float> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<float>(0);
        __gm__ float* flagPtr = workspaceInfo.ptrSoftFlagBase;
        AscendC::GlobalTensor<float> flagGM;
        flagGM.SetGlobalBuffer(flagPtr);
        int32_t flagBufferSize = max(4, params.EP) * FLAGSTRIDE;
        AscendC::LocalTensor<float> dstValueBuffer = resource.ubBuf.template GetBufferByByte<float>(flagBufferSize);
        AscendC::LocalTensor<float> sharedTmpBuffer = resource.ubBuf.template GetBufferByByte<float>((flagBufferSize + 64));
        uint64_t mask[1] = {0};
        uint32_t repeatNum = (flagBufferSize / (4 * FLAGSTRIDE));
        for (int32_t i = 0; i < 4; i ++) {
            if (i < params.EP) {
                mask[0] |= 1ull * (1ull << (i * 16));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        while (flag < flagBase) {
            flag = flagBase;
            AscendC::DataCopy(tmpBuffer1, flagGM, params.EP * FLAGSTRIDE);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::ReduceMin<float>(dstValueBuffer, tmpBuffer1, sharedTmpBuffer, mask, repeatNum, 8, false);

            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            flag = min(flag, dstValueBuffer.GetValue(0));

            if (flag > lastflag) {
                *aicFinishPtr = flag;
                gm_dcci(aicFinishPtr);
                lastflag = flag;
            }
        }
    }


    CATLASS_DEVICE
    void CombineSetFlag() {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }


    CATLASS_DEVICE
    void DispatchAndCombine(Params const &params) {
        icache_preload(8);
        int64_t localTokenPerExpertOffset = peermemInfo.offsetPeerTokenPerExpert + tokenPerExpertLayout(params.rank, 0, 0) * sizeof(int32_t);
        GM_ADDR localTokenPerExpert = shmem() + localTokenPerExpertOffset;     // Place the entire communication matrix in peermem
        uint32_t expandedRowIdxOffset = AlignUp(params.problemShape.m(), 256) * params.topK * sizeof(int32_t);

        //---initRouting------
        moe_init_routing_quant_v2<ElementD2>(reinterpret_cast<GM_ADDR> (params.ptrA), params.expertIdx, 
        params.moeInitRoutingQuantV2Scale, params.moeInitRoutingQuantV2Offset, shmem() + peermemInfo.offsetA, 
        workspaceInfo.expandedRowIdx, localTokenPerExpert, params.expertTokensBeforeCapacity, 
        shmem() + peermemInfo.offsetPeerPerTokenScale, 
        params.ptrWorkspace + expandedRowIdxOffset, 
        &params.moeInitRoutingQuantV2TilingData, params.initRoutingQuantTilingKey);

        AscendC::SyncAll<true>();

        CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(params, localTokenPerExpertOffset);

        if (coreIdx == 0) {
            GetCumsumForMMAIV(tokenPerExpert, cumsumMM, params.expertPerRank, params.rank, params.EP);
        }
        
        uint32_t curGroupOffset = 0;
        int32_t prevSumBeforeRank = 0;
        int32_t prevSum = 0;
        if (coreIdx < params.EP) {
            prevSum = preSumBeforeRank(coreIdx * params.expertPerRank);
        }
        AscendC::SyncAll<true>();
        
        AscendC::GlobalTensor<int32_t> ExpertTokenNums;
        ExpertTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrExpertTokenNums));
        AscendC::GlobalTensor<int32_t> LcalCumsumMM;
        LcalCumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrcumsumMM + (params.EP - 1) * params.expertPerRank * sizeof(int32_t)));
        CopyGMToGM(ExpertTokenNums, LcalCumsumMM, params.expertPerRank, params.ubMoveNum);
        AscendC::SyncAll<true>();
        uint16_t syncgmm1Idx = 0;
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
        syncgmm1Idx++;

        uint32_t prevGroupSum1 = 0, dequantSum1 = 0, dequantSum2 = 0;
        uint32_t dequantSum = 0;

        icache_preload(8);
        for (int32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            // The ith core reads data from the ith rank's peermem
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            for(int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
                uint32_t rowStart = (dstEpIdx == 0 ? 0 : cumsumMM((dstEpIdx - 1) * params.expertPerRank + groupIdx)) + prevGroupSum1;
                if (rowStart < params.maxOutputSize) {
                    uint32_t rows = tokenPerExpert(tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
                    if (rowStart + rows > params.maxOutputSize) {
                        rows = params.maxOutputSize - rowStart;
                    }
                    uint32_t rowSrc = prevSum;
                    prevSum += rows;
                    GM_ADDR otherRankPtr = shmem(0, dstEpIdx);
                    AscendC::GlobalTensor<ElementA> gmRemoteA;
                    gmRemoteA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(otherRankPtr + peermemInfo.offsetA));

                    MatrixCoord offsetA{rowStart, 0};
                    MatrixCoord offsetPeer{rowSrc, 0};
                    int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
                    int64_t gmOffsetPeer = rowSrc * (params.problemShape.k() + ALIGN_512);
                    // Communication data
                    CopyGMToGMPerToken(gmA[gmOffsetA], gmPerTokenScale1[rowStart], gmRemoteA[gmOffsetPeer],  rows, params.problemShape.k());
                }

            }
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmm1Idx ++;

            prevGroupSum1 += currentM;

            // Token count and truncation logic for the first SwiGLU operation
            if (groupIdx + 1 <= params.epilogueGranularity) {
                if (dequantSum1 + currentM <= params.maxOutputSize) {
                    dequantSum1 += currentM;
                } else if (dequantSum1 < params.maxOutputSize) {
                    dequantSum1 = params.maxOutputSize;
                }
            }

            // Token count and truncation logic for the second SwiGLU operation
            if (groupIdx + 1 > params.epilogueGranularity && dequantSum1 < params.maxOutputSize) {
                if (dequantSum1 + dequantSum2 + currentM <= params.maxOutputSize) {
                    dequantSum2 += currentM;
                } else if (dequantSum1 + dequantSum2 < params.maxOutputSize) {
                    dequantSum2 += params.maxOutputSize - dequantSum1 - dequantSum2;
                }
            }
        }

        uint32_t n2 = params.problemShape.k();

        typename BlockEpilogue2::Params epilogueParams{
            static_cast<int32_t>(params.EP),
            static_cast<int32_t>(params.expertPerRank),
            static_cast<int32_t>(params.rank),
            reinterpret_cast<__gm__ int32_t *>(shmem() + peermemInfo.offsetPeerTokenPerExpert),
            params.layoutD2,
            static_cast<int32_t>(n2),
            static_cast<int32_t>(L1TileShape::N),
            shmem,
            static_cast<int32_t>(peermemInfo.offsetD)
        };

        uint32_t n = params.problemShape.n();
        BlockEpilogue2 blockEpilogue2(resource, epilogueParams);
        BlockEpilogue1 blockEpilogue1(resource, n);

        // Synchronous wait: SwiGLU waits for GMM1 [1]
        AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V);
        AscendC::SyncAll<true>();
        if (dequantSum1 > 0) { 
            uint32_t rowStartThisCore = 0;
            MatrixCoord offsetC{0U, 0};
            MatrixCoord shapeC{dequantSum1, params.problemShape.n()};
            LayoutC layoutC{dequantSum1, params.problemShape.n()};
            int64_t gmOffsetC = layoutC.GetOffset(offsetC);
            int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
            blockEpilogue1(gmC[gmOffsetC], shapeC, gmPerTokenScale1[rowStartThisCore], gmPermutedToken[gmOffsetD], gmPerTokenScale2[rowStartThisCore], params.epilogueCoreNum);
        }
        AscendC::SyncAll<true>();
        // Synchronization signal: SwiGLU notifies GMM2 [1]
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C);
        
        if ((params.epilogueGranularity < params.expertPerRank && params.epilogueGranularity > 0)) {
            // Synchronous wait: SwiGLU waits for GMM1 [2]
            AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V);
            AscendC::SyncAll<true>();
            if (dequantSum2 > 0) {
                uint32_t rowStartThisCore = dequantSum1;
                MatrixCoord offsetC{rowStartThisCore, 0};
                uint32_t dequantLen = dequantSum2;
                MatrixCoord shapeC{dequantLen, params.problemShape.n()};
                LayoutC layoutC{dequantLen, params.problemShape.n()};
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
                blockEpilogue1(gmC[gmOffsetC], shapeC, gmPerTokenScale1[rowStartThisCore], gmPermutedToken[gmOffsetD], gmPerTokenScale2[rowStartThisCore], coreNum);
            }
            AscendC::SyncAll<true>();
            // Synchronization signal: SwiGLU notifies GMM2 [2]
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C);
        }

        blockEpilogue1.Finalize();


        CombineSetFlag();

        CombineV2(params, blockEpilogue2);

        AscendC::SyncAll<true>();
        #ifndef __CROSSRANKSYNCANDALLGATHERV1__
        ResetTokenPerExpert(params.EP * AlignUp(params.EP * params.expertPerRank, 128));
        #endif
        shmem.InitStatusTargetSum();
        if (get_subblockid() == 0) {
            AscendC::LocalTensor<int32_t> ctrBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
            shmem.CrossRankSyncV2Set(ctrBuffer);
        } else {
            uint32_t uboffset = 0;
            uint32_t aicCoreNum = coreNum / 2;
            uint32_t aicCoreIdx = get_block_idx();
            uint32_t sendRankNum_ = params.EP / aicCoreNum;
            uint32_t remainderRankNum = params.EP % aicCoreNum;
            if (aicCoreIdx < remainderRankNum) {
                sendRankNum_++;
            }
            AscendC::LocalTensor<float> statusTensor = resource.ubBuf.template GetBufferByByte<float>(uboffset);
            uboffset += sendRankNum_ * UB_ALIGN;
            AscendC::LocalTensor<float> gatherMaskOutTensor = resource.ubBuf.template GetBufferByByte<float>(uboffset);
            uboffset += params.EP * sizeof(float);
            AscendC::LocalTensor<uint32_t> gatherTmpTensor = resource.ubBuf.template GetBufferByByte<uint32_t>(uboffset);
            uboffset += sizeof(uint32_t);
            AscendC::LocalTensor<float> statusSumOutTensor = resource.ubBuf.template GetBufferByByte<float>(uboffset);
            uboffset += sizeof(float);
            shmem.CrossRankSyncV2Wait(statusTensor, gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
            MoeTokenUnpermuteTilingData tilingData;
            MoeTokenUnpermuteTiling(params.problemShape.m() * params.topK, n2, params.topK, tilingData, coreNum / 2);
            KernelMoeTokenUnpermute<ElementD2, int32_t, float, true> kernelMoeTokenUnpermuteOp;
            kernelMoeTokenUnpermuteOp.Init(shmem() + peermemInfo.offsetD, workspaceInfo.expandedRowIdx, params.probs, reinterpret_cast<GM_ADDR>(params.ptrOutput), &tilingData);
            kernelMoeTokenUnpermuteOp.Process();
        }
 
    }

    CATLASS_DEVICE
    void CombineV2(Params const &params, BlockEpilogue2 & blockEpilogue) {
        BlockScheduler blockScheduler;
        int32_t syncLoopIdx = 0;
        uint32_t startCoreIdx = 0;
        uint32_t aicCoreNum = coreNum / 2;
        uint32_t aicCoreIdx = get_block_idx();
        uint32_t aivSubCoreIdx = get_subblockid();
        uint32_t preSrcExpertSum = 0;
        uint32_t n2 = params.problemShape.k();
        uint32_t k2 = params.problemShape.n() / 2;
        icache_preload(8);
        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentExpertM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            if (preSrcExpertSum >= params.maxOutputSize) {
                currentExpertM = 0;
            } else if (preSrcExpertSum + currentExpertM > params.maxOutputSize) {
                currentExpertM = params.maxOutputSize - preSrcExpertSum;
            }
            GemmCoord inGroupProblemShape{currentExpertM, n2, k2}; // M N K
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            uint32_t startLoopIdx = ((aicCoreIdx < startCoreIdx) ? (aicCoreIdx + aicCoreNum) : aicCoreIdx) - startCoreIdx;

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicCoreNum) {
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                int32_t m0 = 16;
                //  Block count, the shape of each block is (m0, actualBlockShape.n())
                int32_t m_rows = (actualBlockShape.m() + m0 - 1) / m0;
                int32_t aiv_m_rows = m_rows / 2;
                if (aivSubCoreIdx == 1 && aiv_m_rows * 2 < m_rows) {
                    aiv_m_rows += 1;
                }
                uint32_t m_offset = blockCoord.m() * L1TileShape::M;//blockOffset
                if(aivSubCoreIdx == 1) {
                    m_offset += (m_rows / 2) * m0;
                }


                for (;syncLoopIdx <= groupIdx; syncLoopIdx ++) {
                    int32_t flag_id = syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
                    AscendC::CrossCoreWaitFlag<0x2>(flag_id);
                }

                for (int32_t cur_row = 0; cur_row < aiv_m_rows; cur_row ++) {
                    GemmCoord realTileCoord{m_offset, blockCoord.n() * L1TileShape::N, 1};
                    uint32_t actualm = m0;
                    if(aivSubCoreIdx == 1 && cur_row == aiv_m_rows - 1){
                        actualm = actualBlockShape.m() - (m_rows / 2) * m0 - cur_row * m0;
                    }
                    GemmCoord realTileShape{actualm, actualBlockShape.n(), 1};
                    blockEpilogue(gmC2, gmPerTokenScale2, realTileCoord, realTileShape, groupIdx, preSrcExpertSum, preSumBeforeRank);
                    m_offset += m0;
                }
            }
            preSrcExpertSum += currentExpertM;
            startCoreIdx = (startCoreIdx + coreLoops) % aicCoreNum;
        }
        blockEpilogue.Finalize();
    }


private:
  struct WorkspaceInfo {
        GM_ADDR ptrA;
        GM_ADDR ptrPerTokenScale;
        GM_ADDR ptrcumsumMM;
        GM_ADDR ptrC;
        GM_ADDR ptrC2;
        GM_ADDR ptrPermutedToken;
        GM_ADDR ptrPerTokenScale2;
        GM_ADDR expandedRowIdx;
        GM_ADDR ptrTokenPerExpert;
        GM_ADDR ptrSumBeforeRank;
        __gm__ float* ptrSoftFlagBase;


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

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);
            ptrPerTokenScale = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            ptrPerTokenScale2 = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            ptrTokenPerExpert =  params.ptrWorkspace + workspaceOffset;

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);
            ptrC = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * params.problemShape.n() * sizeof(ElementC);
            ptrC2 = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * n2 * sizeof(ElementC);
            ptrA = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * params.problemShape.k() * sizeof(ElementA);
            ptrPermutedToken = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * k2 * sizeof(ElementA);
            ptrSumBeforeRank = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.EP * sizeof(int32_t) * FLAGSTRIDE;
            ptrSoftFlagBase = reinterpret_cast<__gm__ float*>(params.ptrWorkspace + workspaceOffset);
        }
    };

    struct PeermemInfo {
        int64_t offsetA;
        int64_t offsetPeerPerTokenScale;
        int64_t offsetPeerTokenPerExpert;
        int64_t offsetD;

        CATLASS_DEVICE
        PeermemInfo(){}

        CATLASS_DEVICE
        PeermemInfo(const Params & params, const HcclShmem & shmem) {
            offsetA = 0;    // Occupies one third of BUFFSIZE
            offsetPeerPerTokenScale = offsetA + AlignUp(shmem.SegmentSize() / 3, 512); // Occupies 1 MB
            offsetD = offsetPeerPerTokenScale + MB_SIZE;    // Occupies the remaining space
            offsetPeerTokenPerExpert = shmem.SegmentSize() - 2 * MB_SIZE;     // Occupies the final 2 MB
        }
    };

    Arch::Resource<ArchTag> resource;

    uint32_t coreIdx;
    uint32_t coreNum;

    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;

    AscendC::GlobalTensor<ElementA> gmA;
    AscendC::GlobalTensor<ElementC> gmC;

    AscendC::GlobalTensor<ElementD1> gmPermutedToken;
    AscendC::GlobalTensor<ElementC> gmC2;

    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale1;
    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale2;

    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    AscendC::GlobalTensor<int32_t> cumsumMM;
    AscendC::GlobalTensor<int32_t> preSumBeforeRank;
    Layout3D tokenPerExpertLayout;
    HcclShmem shmem;
};

} // namespace Catlass::Gemm::Kernel

#endif // DISPATCH_FFN_COMBINE_KERNEL_HPP
