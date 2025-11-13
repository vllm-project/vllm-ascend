/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_GROUPGEMM_HPP
#define CATLASS_GEMM_KERNEL_GROUPGEMM_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/helper.hpp"

namespace Catlass::Gemm::Kernel{

    namespace detail {

        template <class T>
        CATLASS_DEVICE
        void UnpackListParam(T *const dst, GM_ADDR src, uint32_t len)
        {
            for (uint32_t i = 0; i * sizeof(uint64_t) < len * sizeof(T); ++i) {
                reinterpret_cast<uint64_t *>(dst)[i] = reinterpret_cast<__gm__ uint64_t *>(src)[i];
            }
        }
        
    }  // namespace detail


template<
    class ArchTag_,
    class Element_,
    class Layout_,
    uint32_t COMPUTE_LENGTH
>
struct PaddingMatrix {
public:
    using ArchTag = ArchTag_;
    using Element = Element_;
    using Layout = Layout_;
    using CopyGm2Ub = Catlass::Epilogue::Tile::CopyGm2Ub<
        ArchTag, Gemm::GemmType<Element, Catlass::layout::RowMajor>>;
    using CopyUb2Gm = Catlass::Epilogue::Tile::CopyUb2Gm<
        ArchTag, Gemm::GemmType<Element, Catlass::layout::RowMajor>>;
    using ComputeLayout = Catlass::layout::RowMajor; 

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    CATLASS_DEVICE
    PaddingMatrix(Arch::Resource<ArchTag> &resource){
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) { // 
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element));
            bufferOffset += COMPUTE_LENGTH;
        }
    }

    CATLASS_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::RowMajor const &layout){ 
        return ComputeLayout(layout.shape(0), layout.shape(1), layout.stride(0)); 
    }

    CATLASS_DEVICE
    ComputeLayout GetPaddingComputeLayout(layout::ColumnMajor const &layout){
        return ComputeLayout(layout.shape(1), layout.shape(0), layout.stride(1)); 
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<Element> const &dst,
                    AscendC::GlobalTensor<Element> const &src,
                    Layout layoutDst, Layout layoutSrc
    ){
        ComputeLayout computeLayoutSrc = GetPaddingComputeLayout(layoutSrc);
        ComputeLayout computeLayoutDst = GetPaddingComputeLayout(layoutDst);

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum(); 
        uint32_t aivId = AscendC::GetBlockIdx(); 

        // Each line is a tile.
        uint32_t tilesNum = computeLayoutSrc.shape(0); 
        uint32_t tileLen = computeLayoutSrc.shape(1); 
        uint32_t paddingStride = computeLayoutDst.stride(0); 

        uint32_t tilesPerAiv = tilesNum / aivNum; 
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++; 
        }
        uint32_t mIdx = aivId * tilesPerAiv;
        if (aivId >= tileRemain) {
            mIdx += tileRemain;
        }
        MatrixCoord blockOffset(mIdx, 0);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]); 
        uint32_t coreLoops{ 0 };
        if (paddingStride > COMPUTE_LENGTH) { 
            // Handle the same tile on multiple loops.
            uint32_t loopsPerTile = CeilDiv(tileLen, COMPUTE_LENGTH); 
            coreLoops = tilesPerAiv * loopsPerTile;  
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx / loopsPerTile;
                uint32_t inTileLoopIdx = loopIdx % loopsPerTile;
                MatrixCoord loopOffset(tileIdx, inTileLoopIdx * COMPUTE_LENGTH);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + loopOffset);
                uint32_t actualDataNum = COMPUTE_LENGTH;
                if (tileLen - inTileLoopIdx * COMPUTE_LENGTH < COMPUTE_LENGTH) { 
                    actualDataNum = tileLen - inTileLoopIdx * COMPUTE_LENGTH;
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout dstLayout = computeLayoutDst.GetTileLayout(MatrixCoord(1, actualDataNum)); 
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(1, actualDataNum));
                ComputeLayout &ubLayout = dstLayout;
                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + loopOffset);
                copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex], dstLayout, ubLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                bufferIndex = 1 - bufferIndex;
            }
        } else {
            // Handle multiple tile each loop.
            uint32_t tilesPerLoop = COMPUTE_LENGTH / paddingStride; 
            coreLoops = CeilDiv(tilesPerAiv, tilesPerLoop);
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) { 
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                MatrixCoord tileOffset(tileIdx, 0);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + tileOffset);
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) { 
                    actualTilesNum = tilesPerAiv - tileIdx;
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout dstLayout = computeLayoutDst.GetTileLayout(MatrixCoord(actualTilesNum, tileLen)); 
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(actualTilesNum, tileLen));
                ComputeLayout &ubLayout = dstLayout;
                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + tileOffset);
                copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex], dstLayout, ubLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                bufferIndex = 1 - bufferIndex;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]); 
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
    }

    CATLASS_DEVICE
    ~PaddingMatrix() {}
private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<Element> inputBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(Element) <= ArchTag::UB_SIZE, "Excedding the UB space!");
};


template<
    class BlockGemm_,
    class BlockEpilogue_ ,
    class BlockScheduler_ = void
>
class KernelGroupGemm{
public:
    using BlockGemm = BlockGemm_;
    using ArchTag = typename BlockGemm::ArchTag;
    using L1TileShape = typename BlockGemm::L1TileShape;
    using ElementA = typename BlockGemm::ElementA;
    using LayoutA = typename BlockGemm::LayoutA;
    using LayoutWA = typename BlockGemm::LayoutA;
    using ElementB = typename BlockGemm::ElementB;
    using LayoutB = typename BlockGemm::LayoutB;
    using LayoutWB = typename BlockGemm::LayoutB;
    using ElementC = typename BlockGemm::ElementC;
    using LayoutC = typename BlockGemm::LayoutC;
    using ElementAccumulator = typename BlockGemm::ElementAccumulator;
    
    using BlockEpilogue = BlockEpilogue_;
    using EpilogueParams = typename BlockEpilogue::Params;
    using ElementCompute =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using ElementScalar = ElementCompute; 
    static constexpr uint32_t MAX_TENSOR_COUNT = 32;

    const uint32_t maxMPerBlock = L1TileShape::M;
    const uint32_t maxNPerBlock = L1TileShape::N;
    const uint32_t cSize = maxMPerBlock * maxNPerBlock * sizeof(ElementAccumulator);
    const uint32_t l0CBlockNum = ArchTag::L0C_SIZE / cSize;

    static constexpr uint32_t STAGES = BlockGemm::STAGES; 
    using BlockScheduler = BlockScheduler_;
    
    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA); 
    using PaddingA = PaddingMatrix<ArchTag, ElementA, LayoutA, COMPUTE_LENGTH_A>;
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using PaddingB = PaddingMatrix<ArchTag, ElementB, LayoutB, COMPUTE_LENGTH_B>;

    struct Params{
        // Data members
        uint32_t problemCount;
        GM_ADDR ptrProblemShape;
        GM_ADDR alpha;
        GM_ADDR beta;
        GM_ADDR ptrA;
        GM_ADDR ptrLayoutA;
        GM_ADDR ptrB;
        GM_ADDR ptrLayoutB;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrLayoutWorkspace;
        GM_ADDR ptrWA;
        GM_ADDR ptrLayoutWA;
        GM_ADDR ptrWB;
        GM_ADDR ptrLayoutWB;
        GM_ADDR ptrX;
        GM_ADDR ptrD;
        

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            uint32_t problemCount_, GM_ADDR ptrProblemShape_,
            GM_ADDR alpha_, GM_ADDR beta_,
            GM_ADDR ptrA_, GM_ADDR ptrLayoutA_,
            GM_ADDR ptrB_, GM_ADDR ptrLayoutB_,
            GM_ADDR ptrWorkspace_, GM_ADDR ptrLayoutWorkspace_,
            GM_ADDR ptrWA_,GM_ADDR ptrLayoutWA_, 
            GM_ADDR ptrWB_,GM_ADDR ptrLayoutWB_, GM_ADDR ptrX_, GM_ADDR ptrD_)
                :problemCount(problemCount_), ptrProblemShape(ptrProblemShape_),
                alpha(alpha_), beta(beta_),
                ptrA(ptrA_), ptrLayoutA(ptrLayoutA_),
                ptrB(ptrB_), ptrLayoutB(ptrLayoutB_),
                ptrWorkspace(ptrWorkspace_), ptrLayoutWorkspace(ptrLayoutWorkspace_), 
                ptrWA(ptrWA_), ptrLayoutWA(ptrLayoutWA_), 
                ptrWB(ptrWB_), ptrLayoutWB(ptrLayoutWB_),ptrX(ptrX_), ptrD(ptrD_){}
    };

    struct Arguments{
        uint32_t problemCount;
        GM_ADDR ptrProblemShape;
        GM_ADDR alpha;
        GM_ADDR beta;
        GM_ADDR ptrA;
        GM_ADDR ptrLayoutA;
        GM_ADDR ptrB;
        GM_ADDR ptrLayoutB;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrLayoutWorkspace;
        GM_ADDR ptrWA;
        GM_ADDR ptrLayoutWA; 
        GM_ADDR ptrWB;
        GM_ADDR ptrLayoutWB;
        GM_ADDR ptrX;
        GM_ADDR ptrD;
    };

    static bool CanImplement(const Arguments &args){
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace){
        Params params{args.problemCount, args.ptrProblemShape, args.alpha, args.beta, args.ptrA, args.ptrLayoutA, args.ptrB, args.ptrLayoutB,
                    args.ptrWorkspace, args.ptrLayoutWorkspace, args.ptrWA, args.ptrLayoutWA, args.ptrWB, args.ptrLayoutWB, 
                    args.ptrX, args.ptrD};
        return params;
    }

    CATLASS_DEVICE
    KernelGroupGemm(){}

    CATLASS_DEVICE
    ~KernelGroupGemm(){}

    CATLASS_DEVICE
    size_t GetWorkspaceLen(layout::RowMajor layout)
    {
        return layout.shape(0) * layout.stride(0);
    }

    CATLASS_DEVICE
    size_t GetWorkspaceLen(layout::ColumnMajor layout)
    {
        return layout.shape(1) * layout.stride(1);
    }

    template<int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params &params){}

    template<>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params &params)
    {
        GemmCoord problemShapeList[MAX_TENSOR_COUNT];
        LayoutA layoutAList[MAX_TENSOR_COUNT];
        LayoutB layoutBList[MAX_TENSOR_COUNT];
        LayoutC layoutWorkspaceList[MAX_TENSOR_COUNT];
        LayoutA layoutWAList[MAX_TENSOR_COUNT];
        LayoutB layoutWBList[MAX_TENSOR_COUNT];
        // Get matmul information from parameters
        detail::UnpackListParam(problemShapeList, params.ptrProblemShape, params.problemCount);
        detail::UnpackListParam(layoutAList, params.ptrLayoutA, params.problemCount);
        detail::UnpackListParam(layoutBList, params.ptrLayoutB, params.problemCount);
        detail::UnpackListParam(layoutWorkspaceList, params.ptrLayoutWorkspace, params.problemCount);
        detail::UnpackListParam(layoutWAList, params.ptrLayoutWA, params.problemCount);
        detail::UnpackListParam(layoutWBList, params.ptrLayoutWB, params.problemCount);
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint64_t inGroupOffsetA = 0;
        uint64_t inGroupOffsetB = 0;
        uint64_t inGroupOffsetWorkspace = 0;
        uint32_t startCoreIdx = 0;
        uint32_t startLoopIdx;
        Arch::Resource<ArchTag> resource;
        BlockGemm blockGemm(resource);
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx)
        {
            GemmCoord problemShape = problemShapeList[groupIdx];
            LayoutA layoutA = layoutAList[groupIdx];
            LayoutB layoutB = layoutBList[groupIdx];
            LayoutC layoutWorkspace = layoutWorkspaceList[groupIdx];
            LayoutA layoutWA = layoutWAList[groupIdx];
            LayoutB layoutWB = layoutWBList[groupIdx];
            Arch::CrossCoreWaitFlag(flagAivFinishPadding);
            AscendC::GlobalTensor<ElementA> gmA;
            gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrWA);
            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrWB);
            AscendC::GlobalTensor<ElementC> gmC;
            gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrWorkspace);
            uint32_t M = problemShape.m();
            uint32_t N = problemShape.n();
            uint32_t K = problemShape.k();
            #pragma unroll
            for (uint32_t i = 0; i < l0CBlockNum; i++)
            {
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
            }
            uint32_t mLoops = CeilDiv(M, maxMPerBlock);
            uint32_t nLoops = CeilDiv(N, maxNPerBlock);
            uint32_t coreLoops = mLoops * nLoops;
            // Determine the starting loopIdx of the current core under the current groupIdx
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            uint32_t singleIdx = 0;
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) 
            {
                uint32_t mGmBlockIdx = loopIdx / nLoops;
                uint32_t nGmBlockIdx = loopIdx % nLoops;
                uint32_t mGmActual = (mGmBlockIdx == mLoops - 1) ? (M - mGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t nGmActual = (nGmBlockIdx == nLoops - 1) ? (N - nGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                bool isFirstBlock = (loopIdx == startLoopIdx);
                bool hasNextBlock = false;
                GemmCoord nextActualShape;
                uint32_t mNextGmBlockIdx = 0; uint32_t nNextGmBlockIdx = 0;
                if (loopIdx + AscendC::GetBlockNum() < coreLoops)
                {
                    hasNextBlock = true;
                    uint32_t nextLoopIdx = loopIdx + AscendC::GetBlockNum();
                    mNextGmBlockIdx = nextLoopIdx / nLoops;
                    nNextGmBlockIdx = nextLoopIdx % nLoops;
                    uint32_t mNextGmActual = (mNextGmBlockIdx == mLoops - 1) ? (M - mNextGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                    uint32_t nNextGmActual = (nNextGmBlockIdx == nLoops - 1) ? (N - nNextGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                    nextActualShape = MakeCoord(mNextGmActual, nNextGmActual, K); 
                }
                GemmCoord actualShape{mGmActual, nGmActual, K};
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)singleIdx);
                MatrixCoord gmTileAOffset{mGmBlockIdx * maxMPerBlock, 0}; 
                auto gmTileA = gmA[inGroupOffsetA + layoutWA.GetOffset(gmTileAOffset)];
                MatrixCoord gmTileBOffset{0, nGmBlockIdx * maxNPerBlock}; 
                auto gmTileB = gmB[inGroupOffsetB + layoutWB.GetOffset(gmTileBOffset)];
                MatrixCoord gmTileCOffset{mGmBlockIdx * maxMPerBlock, nGmBlockIdx * maxNPerBlock}; 
                auto gmTileC = gmC[inGroupOffsetWorkspace + layoutWorkspace.GetOffset(gmTileCOffset)];
                MatrixCoord gmTileNextAOffset{mNextGmBlockIdx * maxMPerBlock, 0}; 
                auto gmTileNextA = gmA[inGroupOffsetA + layoutWA.GetOffset(gmTileNextAOffset)];
                MatrixCoord gmTileNextBOffset{0, nNextGmBlockIdx * maxNPerBlock};
                auto gmTileNextB = gmB[inGroupOffsetB + layoutWB.GetOffset(gmTileNextBOffset)];
                blockGemm(
                    gmTileA, layoutWA, 
                    gmTileB, layoutWB,
                    gmTileC, layoutWorkspace,
                    gmTileNextA, gmTileNextB,
                    actualShape, nextActualShape, isFirstBlock, hasNextBlock, singleIdx
                );
                Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagAicFinishStore);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>((int32_t)singleIdx);
                singleIdx = (singleIdx + 1) % l0CBlockNum;
            }
            inGroupOffsetA += GetWorkspaceLen(layoutWA);
            inGroupOffsetB += GetWorkspaceLen(layoutWB);
            inGroupOffsetWorkspace += problemShape.m() * problemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
            #pragma unroll
            for (uint32_t i = 0; i < l0CBlockNum; i++)
            {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((int32_t)i);
            }
        }
    } 

    template<>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params &params){
        GemmCoord problemShapeList[MAX_TENSOR_COUNT];
        LayoutA layoutAList[MAX_TENSOR_COUNT];
        LayoutB layoutBList[MAX_TENSOR_COUNT];
        LayoutC layoutWorkspaceList[MAX_TENSOR_COUNT];
        ElementScalar alphaList[MAX_TENSOR_COUNT];
        ElementScalar betaList[MAX_TENSOR_COUNT];
        LayoutA layoutWAList[MAX_TENSOR_COUNT];
        LayoutB layoutWBList[MAX_TENSOR_COUNT];
        detail::UnpackListParam(problemShapeList, params.ptrProblemShape, params.problemCount);
        detail::UnpackListParam(layoutAList, params.ptrLayoutA, params.problemCount);
        detail::UnpackListParam(layoutBList, params.ptrLayoutB, params.problemCount);
        detail::UnpackListParam(layoutWorkspaceList, params.ptrLayoutWorkspace, params.problemCount);
        detail::UnpackListParam(alphaList, params.alpha, params.problemCount);
        detail::UnpackListParam(betaList, params.beta, params.problemCount);
        detail::UnpackListParam(layoutWAList, params.ptrLayoutWA, params.problemCount);
        detail::UnpackListParam(layoutWBList, params.ptrLayoutWB, params.problemCount);
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint64_t inGroupOffsetA = 0;
        uint64_t inGroupOffsetWA = 0;
        uint64_t inGroupOffsetB = 0;
        uint64_t inGroupOffsetWB = 0;
        uint64_t inGroupOffsetWorkspace = 0;
        uint32_t startCoreIdx = 0;
        uint32_t startLoopIdx;
        GemmCoord blockShape = L1TileShape::ToCoord();
        Arch::Resource<ArchTag> resource;
        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementA> gmWA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrA));
        gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrWA));
        AscendC::GlobalTensor<ElementB> gmB;
        AscendC::GlobalTensor<ElementB> gmWB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrB));
        gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(params.ptrWB));
        PaddingA paddingA(resource);
        PaddingB paddingB(resource);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrWorkspace);
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx)
        {
            GemmCoord problemShape = problemShapeList[groupIdx];
            LayoutA layoutA = layoutAList[groupIdx];
            LayoutB layoutB = layoutBList[groupIdx];
            LayoutC layoutWorkspace = layoutWorkspaceList[groupIdx];
            ElementScalar alpha_ = alphaList[groupIdx];
            ElementScalar beta_ = betaList[groupIdx];
            LayoutA layoutWA = layoutWAList[groupIdx];
            LayoutB layoutWB = layoutWBList[groupIdx];
            paddingA(gmWA[inGroupOffsetWA], gmA[inGroupOffsetA], layoutWA, layoutA); 
            paddingB(gmWB[inGroupOffsetWB], gmB[inGroupOffsetB], layoutWB, layoutB);
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
            EpilogueParams epilogueParams{alpha_, beta_, params.ptrX, layoutWorkspace, params.ptrD, layoutWorkspace};
            BlockEpilogue blockEpilogue(resource, blockShape, epilogueParams);
            uint32_t M = problemShape.m();
            uint32_t N = problemShape.n();
            uint32_t K = problemShape.k();
            uint32_t mLoops = CeilDiv(M, maxMPerBlock);
            uint32_t nLoops = CeilDiv(N, maxNPerBlock);
            uint32_t coreLoops = mLoops * nLoops;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum())
            {
                uint32_t mGmBlockIdx = loopIdx / nLoops;
                uint32_t nGmBlockIdx = loopIdx % nLoops;
                uint32_t mGmActual = (mGmBlockIdx == mLoops - 1) ? (M - mGmBlockIdx * maxMPerBlock) : maxMPerBlock;
                uint32_t nGmActual = (nGmBlockIdx == nLoops - 1) ? (N - nGmBlockIdx * maxNPerBlock) : maxNPerBlock;
                GemmCoord actualShape{mGmActual, nGmActual, K};
                GemmCoord blockCoord{mGmBlockIdx, nGmBlockIdx, 0};
                Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagAicFinishStore); 
                blockEpilogue(actualShape, blockCoord, gmC, layoutWorkspace, inGroupOffsetWorkspace);
            }
            inGroupOffsetA += problemShape.m() * problemShape.k();
            inGroupOffsetWA += GetWorkspaceLen(layoutWA);
            inGroupOffsetB += problemShape.k() * problemShape.n();
            inGroupOffsetWB += GetWorkspaceLen(layoutWB);
            inGroupOffsetWorkspace += problemShape.m() * problemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        
    }
private:
    static constexpr Arch::FlagID FLAG_AIC_FINISH_STORE = 0;
    static constexpr Arch::FlagID RV_FLAG_AIC_FINISH_STORE = 1;
    Arch::CrossCoreFlagWithReverse<> flagAicFinishStore{FLAG_AIC_FINISH_STORE, RV_FLAG_AIC_FINISH_STORE};
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
};
}

#endif // CATLASS_GEMM_KERNEL_GROUPGEMM_HPP