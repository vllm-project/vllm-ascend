/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_SLICE_K_WITH_PROLOGUE_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_SLICE_K_WITH_PROLOGUE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_traits.hpp"

namespace Catlass::Gemm::Block {

template <
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmad <
    MmadAtlasA2PingpongSliceKWithPrologue<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2PingpongSliceKWithPrologue<ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;

    using PrologueA = typename TileCopy_::PrologueA;
    using PrologueB = typename TileCopy_::PrologueB;
    using PrologueC = typename TileCopy_::PrologueB; // NOTE: SHOULD BE Epilogue

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;

    static constexpr bool HAS_PROLOGUE_A = !std::is_same_v<PrologueA, void>;
    static constexpr bool HAS_PROLOGUE_B = !std::is_same_v<PrologueB, void>;

    // Check L1TileShape
    static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES) <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);
    static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");
    
    // 32B (256b) aligned
    static_assert(Gemm::helper::TileShapeAlignChecker<L1TileShape, L0TileShape, ElementA, ElementB>::_ALIGN == 256, 
        "Tile shape must be 32B aligned.");
    
    struct Params {
        typename Tile::PrologueTraits<PrologueA>::Params prologueAParams{};
        typename Tile::PrologueTraits<PrologueB>::Params prologueBParams{};
        uint32_t mScalar;
        uint32_t nScalar;
        uint32_t splitkLength;
    };

    /// Construct
    CATLASS_DEVICE
    BlockMmad(Arch::Resource<ArchTag> &resource, Params const &params_ = {}, 
        uint32_t l1BufAddrStart = 0) : params(params_)
    {
        // PREVIOUSLY:
        // Initilize PrologueA & PrologueB instantly
        // prologueA(resource, params_.prologueAParams), prologueB(resource, params_.prologueBParams)
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_SIZE * STAGES;
        // Init set/wait flags
        flag0[0].id = flagID0;
        flag0[1].id = flagID1;
        flag1[0].id = flagID2;
        flag1[1].id = flagID3;

        // AIC Type
        if constexpr (g_coreType == AscendC::AIC) {
            // Init buffers
            for (uint32_t i = 0; i < STAGES; i++) {
                // Assign L1/L0A/L0B space for each stages
                l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_SIZE * i);
                l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_SIZE * i);
                l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
                l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);

                // Assign event ID for each stages
                l1AEventList[i] = i;
                l1BEventList[i] = i + STAGES;
                l0AEventList[i] = i;
                l0BEventList[i] = i + STAGES;

                // The event id that needs to be set before the loop
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);

            // CV 同步
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flag0[0]);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flag0[1]);
        }
        // AIV Coretype
        if constexpr (g_coreType == AscendC::AIV) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmad() {
        if constexpr (g_coreType == AscendC::AIC) {
            for (uint32_t i = 0; i < STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }

        if constexpr (g_coreType == AscendC::AIV) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);

            Catlass::Arch::CrossCoreWaitFlag(flag0[0]);
            Catlass::Arch::CrossCoreWaitFlag(flag0[1]);
        }
        
    }

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, 
        AscendC::GlobalTensor<ElementB> const &gmB, 
        AscendC::GlobalTensor<ElementC> const &gmC,
        GemmCoord actualBlockShape,
        GemmCoord problemShape
    )
    {
        MNScalerMatmulImpl(gmA, gmB, gmC, {}, actualBlockShape, problemShape);
    }

    /// Prologue
    // template <class T = PrologueA, class U = PrologueB>
    CATLASS_DEVICE
    void Prologue(
        AscendC::GlobalTensor<typename PrologueA::ElementDst> const &gmDstA, 
        AscendC::GlobalTensor<typename PrologueA::ElementSrc> const &gmSrcA, typename PrologueA::LayoutSrc const &layoutSrcA,
        AscendC::GlobalTensor<typename PrologueB::ElementDst> const &gmDstB, 
        AscendC::GlobalTensor<typename PrologueB::ElementSrc> const &gmSrcB, typename PrologueB::LayoutSrc const &layoutSrcB,
        AscendC::GlobalTensor<half> const &gmDstC, 
        AscendC::GlobalTensor<ElementC> const &gmSrcC, LayoutC const &layoutSrcC,
        GemmCoord const &blockCoord, GemmCoord const &nextBlockCoord, 
        GemmCoord const &actualBlockShape, GemmCoord const &nextActualBlockShape,
        GemmCoord const &problemShape,
        bool isFirstBlock, bool hasNextBlock,
        uint32_t &bufferIndex
    )
    {
        PrologueImpl(
            gmDstA, gmSrcA, layoutSrcA,
            gmDstB, gmSrcB, layoutSrcB,
            gmDstC, gmSrcC, layoutSrcC,
            blockCoord, nextBlockCoord,
            actualBlockShape, nextActualBlockShape, problemShape,
            isFirstBlock, hasNextBlock, bufferIndex
        );
    }

protected:
    CATLASS_DEVICE
    void PrologueImpl(
        AscendC::GlobalTensor<typename PrologueA::ElementDst> const & gmDstA,
        AscendC::GlobalTensor<typename PrologueA::ElementSrc> const & gmSrcA,
        typename PrologueA::LayoutSrc const & layoutSrcA,
        AscendC::GlobalTensor<typename PrologueB::ElementDst> const & gmDstB,
        AscendC::GlobalTensor<typename PrologueB::ElementSrc> const & gmSrcB,
        typename PrologueB::LayoutSrc const & layoutSrcB,
        AscendC::GlobalTensor<half> const &gmDstC,
        AscendC::GlobalTensor<ElementC> const &gmSrcC,
        LayoutC const &layoutSrcC,
        GemmCoord blockCoord,
        GemmCoord nextBlockCoord,
        GemmCoord actualBlockShape,
        GemmCoord nextActualBlockShape,
        GemmCoord problemShape,
        bool const isFirstBlock, 
        bool const hasNextBlock,
        uint32_t &bufferIndex
    )
    {
        // 当前任务块 Current block
        // 计算基础偏移量
        // layoutSrcA: params.layoutA, layoutSrcB: params.layoutB, layoutSrcC: params.layoutC
        int64_t gmOffsetA, gmOffsetB, gmOffsetC, gmOffsetWA, gmOffsetWB, gmOffsetWC;
        int64_t gmOffsetNextA, gmOffsetNextB, gmOffsetWADelta, gmOffsetWBDelta;

        gmOffsetA = layoutSrcA.GetOffset(MakeCoord(blockCoord.m() * (L1TileShape::M * params.mScalar), 0U));
        gmOffsetNextA = layoutSrcA.GetOffset(MakeCoord(nextBlockCoord.m() * (L1TileShape::M * params.mScalar), 0U));
        gmOffsetWA = (AscendC::GetBlockIdx() / AIVPERCORE) * 
                    (params.mScalar * L1TileShape::M) * params.splitkLength * STAGES;
        gmOffsetWADelta = (params.mScalar * L1TileShape::M) * params.splitkLength;

        gmOffsetB = layoutSrcB.GetOffset(MakeCoord(0U, blockCoord.n() * (L1TileShape::N * params.nScalar)));
        gmOffsetNextB = layoutSrcB.GetOffset(MakeCoord(0U, nextBlockCoord.n() * (L1TileShape::N * params.nScalar)));
        gmOffsetWB = (AscendC::GetBlockIdx() / AIVPERCORE) * 
                    (params.nScalar * L1TileShape::N) * params.splitkLength * STAGES;
        gmOffsetWBDelta = params.splitkLength * (params.nScalar * L1TileShape::N);

        MatrixCoord offsetC{blockCoord.m() * (L1TileShape::M * params.mScalar), 
                        blockCoord.n() * (L1TileShape::N * params.nScalar)};        
        gmOffsetC = layoutSrcC.GetOffset(offsetC);
        gmOffsetWC = (AscendC::GetBlockIdx() / AIVPERCORE) * 
                (params.mScalar * L1TileShape::M) * (params.nScalar * L1TileShape::N);

        uint32_t srcAStride = layoutSrcA.stride(
            std::is_same_v<LayoutA, Catlass::layout::RowMajor> ? 0:1);
        uint32_t srcBStride = layoutSrcB.stride(
            std::is_same_v<LayoutB, Catlass::layout::RowMajor> ? 0:1);
        uint32_t mShape_ = actualBlockShape.m();
        uint32_t nShape_ = actualBlockShape.n();

        // Begin resource
        Arch::Resource<ArchTag> resource;

        // SPLIT-K MainLoop
        uint32_t kLoop = (problemShape.k() + params.splitkLength - 1) / params.splitkLength;
        uint32_t kResidueLenght = problemShape.k() % params.splitkLength;
        for (uint32_t ldk = 0; ldk < kLoop; ldk++) {
            uint32_t kActual = (problemShape.k() < (ldk + 1) * params.splitkLength)
                                       ? kResidueLenght : params.splitkLength;
            uint32_t kActual_ = kActual;

            if (ldk == 0 && isFirstBlock) { // 首块
                Catlass::Arch::CrossCoreWaitFlag(flag0[crossCoreBufferIndexAIV]);

                uint32_t kActualAligned_ = RoundUp<256, uint32_t>(kActual_);
                LayoutA layoutWA(actualBlockShape.m(), kActual_, srcAStride);
                LayoutA layoutDstA(actualBlockShape.m(), kActualAligned_);
                LayoutB layoutWB(kActual_, actualBlockShape.n(), srcBStride);
                LayoutB layoutDstB(kActualAligned_, actualBlockShape.n());
                
                PrologueA prologueA(resource, params.prologueAParams);
                PrologueB prologueB(resource, params.prologueBParams);
                prologueA(
                    gmDstA[gmOffsetWA + crossCoreBufferIndexAIV * gmOffsetWADelta],
                    layoutDstA,
                    gmSrcA[gmOffsetA],
                    layoutWA,
                    bufferIndex
                );
                prologueB(
                    gmDstB[gmOffsetWB + crossCoreBufferIndexAIV * gmOffsetWBDelta],
                    layoutDstB,
                    gmSrcB[gmOffsetB],
                    layoutWB,
                    bufferIndex
                );
            }
            if (ldk < kLoop - 1) { // 后续块
                kActual_ = (problemShape.k() < (ldk + 2) * params.splitkLength)
                                       ? kResidueLenght : params.splitkLength;

                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag1[crossCoreBufferIndexAIV]);
                Catlass::Arch::CrossCoreWaitFlag(flag0[1 - crossCoreBufferIndexAIV]);         
                gmOffsetA += layoutSrcA.GetOffset(MakeCoord(0U, kActual));
                gmOffsetB += layoutSrcB.GetOffset(MakeCoord(kActual, 0U));

                uint32_t kActualAligned_ = RoundUp<256, uint32_t>(kActual_);
                LayoutA layoutWA(actualBlockShape.m(), kActual_, srcAStride);
                LayoutA layoutDstA(actualBlockShape.m(), kActualAligned_);
                LayoutB layoutWB(kActual_, actualBlockShape.n(), srcBStride);
                LayoutB layoutDstB(kActualAligned_, actualBlockShape.n());

                PrologueA prologueA(resource, params.prologueAParams);
                PrologueB prologueB(resource, params.prologueBParams);
                prologueA(
                    gmDstA[gmOffsetWA + (1 - crossCoreBufferIndexAIV) * gmOffsetWADelta],
                    layoutDstA,
                    gmSrcA[gmOffsetA],
                    layoutWA,
                    bufferIndex
                );
                prologueB(
                    gmDstB[gmOffsetWB + (1 - crossCoreBufferIndexAIV) * gmOffsetWBDelta],
                    layoutDstB,
                    gmSrcB[gmOffsetB],
                    layoutWB,
                    bufferIndex
                );
            }
            if ((ldk == kLoop - 1) && hasNextBlock) { // SliceK尾部块
                kActual_ = (problemShape.k() < params.splitkLength) ? kResidueLenght : params.splitkLength;
                
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag1[crossCoreBufferIndexAIV]);
                Catlass::Arch::CrossCoreWaitFlag(flag0[1 - crossCoreBufferIndexAIV]);

                uint32_t kActualAligned_ = RoundUp<256, uint32_t>(kActual_);
                LayoutA layoutWA(nextActualBlockShape.m(), kActual_, srcAStride);
                LayoutA layoutDstA(nextActualBlockShape.m(), kActualAligned_);
                LayoutB layoutWB(kActual_, nextActualBlockShape.n(), srcBStride);
                LayoutB layoutDstB(kActualAligned_, nextActualBlockShape.n());

                PrologueA prologueA(resource, params.prologueAParams);
                PrologueB prologueB(resource, params.prologueBParams);
                prologueA(
                    gmDstA[gmOffsetWA + (1 - crossCoreBufferIndexAIV) * gmOffsetWADelta],
                    layoutDstA,
                    gmSrcA[gmOffsetNextA],
                    layoutWA,
                    bufferIndex
                );
                prologueB(
                    gmDstB[gmOffsetWB + (1 - crossCoreBufferIndexAIV) * gmOffsetWBDelta],
                    layoutDstB,
                    gmSrcB[gmOffsetNextB],
                    layoutWB,
                    bufferIndex
                );
            }
           
            if (ldk == kLoop - 1) { // 尾部块
                if (!hasNextBlock) {
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flag1[crossCoreBufferIndexAIV]);
                }
                Catlass::Arch::CrossCoreWaitFlag(flag4);

                Catlass::layout::RowMajor layoutBlockC(
                    actualBlockShape.m(), actualBlockShape.n(), params.nScalar * L1TileShape::N);
                Catlass::layout::RowMajor layoutBlockDstC(
                    problemShape.m(), problemShape.n());

                PrologueC epCast;
                epCast.EpCastFp32ToFp16(
                    gmDstC[gmOffsetC],
                    layoutBlockDstC,
                    gmSrcC[gmOffsetWC],
                    layoutBlockC
                );
            }
                crossCoreBufferIndexAIV = 1 - crossCoreBufferIndexAIV;
            } // end: for-loop (ldk)

    }

    CATLASS_DEVICE
    void MNScalerMatmulImpl(
        AscendC::GlobalTensor<ElementA> const &gmWA, 
        AscendC::GlobalTensor<ElementB> const &gmWB, 
        AscendC::GlobalTensor<ElementC> const &gmWC,
        [[maybe_unused]] GemmCoord blockCoord,
        GemmCoord actualBlockShape,
        GemmCoord problemShape
    )
    {
        constexpr bool isLayoutAColumnMajor = std::is_same_v<LayoutA, Catlass::layout::ColumnMajor>;
        constexpr bool isLayoutBColumnMajor = std::is_same_v<LayoutB, Catlass::layout::ColumnMajor>;

        // uint32_t splitKLen = params.splitkLength;
        uint32_t kLoop = CeilDiv(problemShape.k(), params.splitkLength);
        uint32_t mLoop = CeilDiv<L1TileShape::M>(actualBlockShape.m());
        uint32_t nLoop = CeilDiv<L1TileShape::N>(actualBlockShape.n());

        uint32_t nextMIdx, nextNIdx;
        // SPLIT-K MainLoop
        for (uint32_t ldk = 0; ldk < kLoop; ldk++) {           // K-loop
            bool isFirstKSlice = (ldk == 0);
            uint32_t kActual = (problemShape.k() < (ldk + 1) * params.splitkLength) ? 
                            problemShape.k() % params.splitkLength : params.splitkLength;
            uint32_t kActualAligned = RoundUp<256>(kActual);

            uint32_t srcStrideA = isLayoutAColumnMajor ? actualBlockShape.m(): kActualAligned;
            uint32_t srcStrideB = isLayoutBColumnMajor ? kActualAligned: actualBlockShape.n();

            // CV 同步
            Catlass::Arch::CrossCoreWaitFlag(flag1[crossCoreBufferIndexAIC]);
            
            // M-N-loop
            for (uint32_t mIdx=0; mIdx < mLoop; mIdx++) {       // M-loop
                for (uint32_t nIdx = 0; nIdx < nLoop; nIdx++) { // N-loop
                    bool isFirstBlock = (mIdx == 0 && nIdx == 0);
                    bool hasNextBlock = !(mIdx == mLoop-1 && nIdx == nLoop-1);

                    // 当前偏移及实际尺寸计算(No padding)
                    int64_t gmOffsetWA_ = AscendC::GetBlockIdx() * (L1TileShape::M * params.mScalar) * 
                                params.splitkLength * STAGES;
                    int64_t gmOffsetWB_ = AscendC::GetBlockIdx() * (L1TileShape::N * params.nScalar) *  
                                params.splitkLength * STAGES;
                    gmOffsetWA_ += crossCoreBufferIndexAIC ? params.splitkLength * 
                                (L1TileShape::M * params.mScalar) : 0;
                    gmOffsetWB_ += crossCoreBufferIndexAIC ? params.splitkLength * 
                                (L1TileShape::N * params.nScalar) : 0;
                    int64_t gmOffsetWA = isLayoutAColumnMajor ? gmOffsetWA_ + mIdx * L1TileShape::M :
                                gmOffsetWA_ + mIdx * L1TileShape::M * kActualAligned;
                    int64_t gmOffsetWB = isLayoutBColumnMajor ? gmOffsetWB_ + nIdx * L1TileShape::N * 
                                kActualAligned : gmOffsetWB_ + nIdx * L1TileShape::N;       
                    int64_t gmOffsetWC = AscendC::GetBlockIdx() * (L1TileShape::M * params.mScalar) * 
                                (L1TileShape::N * params.nScalar) + mIdx * L1TileShape::M * 
                                (L1TileShape::N * params.nScalar) + nIdx * L1TileShape::N;

                    uint32_t mActual = (mIdx == mLoop-1 && actualBlockShape.m() % L1TileShape::M != 0) ? 
                                actualBlockShape.m() % L1TileShape::M: L1TileShape::M;
                    uint32_t nActual = (nIdx == nLoop-1 && actualBlockShape.n() % L1TileShape::N != 0) ? 
                                actualBlockShape.n() % L1TileShape::N: L1TileShape::N;
                    LayoutA layoutWA(mActual, kActual, srcStrideA);
                    LayoutB layoutWB(kActual, nActual, srcStrideB);
                    LayoutC layoutWC(mActual, nActual, params.nScalar * L1TileShape::N);
                    GemmCoord actualSmallBlockShape(mActual, nActual, kActual);
                    
                    // Next实际尺寸(No padding)
                    int64_t gmOffsetWANext, gmOffsetWBNext, gmOffsetWCNext;
                    LayoutA layoutWANext;
                    LayoutB layoutWBNext;
                    LayoutC layoutWCNext;
                    GemmCoord nextSmallBlockShape;  
                    if (hasNextBlock) {
                        uint32_t nextNIdx = (nIdx + 1) % nLoop;
                        uint32_t nextMIdx = mIdx +  static_cast<uint32_t>(nextNIdx == 0);

                        gmOffsetWANext = isLayoutAColumnMajor ? gmOffsetWA_ + nextMIdx * L1TileShape::M :
                                gmOffsetWA_ + nextMIdx * L1TileShape::M * kActualAligned;
                        gmOffsetWBNext = isLayoutBColumnMajor ? gmOffsetWB_ + nextNIdx * L1TileShape::N * 
                                kActualAligned : gmOffsetWB_ + nextNIdx * L1TileShape::N;
                        gmOffsetWCNext = AscendC::GetBlockIdx() * (L1TileShape::M * params.mScalar) * 
                                (L1TileShape::N * params.nScalar) + nextMIdx * L1TileShape::M * 
                                (L1TileShape::N * params.nScalar) + nextNIdx * L1TileShape::N;
                        
                        uint32_t mActualNext = (nextMIdx == mLoop-1 && actualBlockShape.m() % L1TileShape::M != 0) ? 
                                actualBlockShape.m() % L1TileShape::M: L1TileShape::M;
                        uint32_t nActualNext = (nextNIdx == nLoop-1 && actualBlockShape.n() % L1TileShape::N != 0) ? 
                                actualBlockShape.n() % L1TileShape::N: L1TileShape::N;
                        layoutWANext = LayoutA(mActualNext, kActual, srcStrideA);
                        layoutWBNext = LayoutB(kActual, nActualNext, srcStrideB);
                        layoutWCNext = LayoutC(mActualNext, nActualNext, params.nScalar * L1TileShape::N);
                        nextSmallBlockShape = GemmCoord(mActualNext, nActualNext, kActual);  
                    }
                    
                    // 完成一个128 * 256的小结果矩阵基本块的运算
                    smallBlockMmadImpl(gmWA[gmOffsetWA],
                        layoutWA,
                        gmWB[gmOffsetWB],
                        layoutWB,
                        gmWC[gmOffsetWC],
                        layoutWC,
                        gmWA[gmOffsetWANext],
                        layoutWANext,
                        gmWB[gmOffsetWBNext],
                        layoutWBNext,
                        actualSmallBlockShape,
                        nextSmallBlockShape,
                        isFirstKSlice,
                        isFirstBlock,
                        hasNextBlock);
                }                          // End of 'N-loop'
            }                              // End of 'M-loop'

            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flag0[crossCoreBufferIndexAIC]);
            crossCoreBufferIndexAIC = 1 - crossCoreBufferIndexAIC;
            if (ldk == kLoop - 1) {
                // cast 256 * 512的fp32大结果基本块为fp16
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flag4);
            }
        }                                  // End of 'K-loop'
    }

    CATLASS_DEVICE
    void smallBlockMmadImpl(
        AscendC::GlobalTensor<ElementA> const &gmWA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmWB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementA> const &gmNextWA, LayoutA const &layoutNextA,
        AscendC::GlobalTensor<ElementB> const &gmNextWB, LayoutB const &layoutNextB,
        GemmCoord const &actualShape, GemmCoord const &nextActualShape,
        bool isFirstKSlice, bool isFirstBlock, bool hasNextBlock
    )
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mRound, nRound));

        uint32_t kActual = min(actualShape.k(), L1TileShape::K);

        // load first matrix A tile from GM to L1
        if (isFirstBlock) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActual));
            copyGmToL1A(l1ATensorList[l1ListId], gmWA, layoutAInL1, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

            // load first matrix B tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActual, actualShape.n()));
            copyGmToL1B(l1BTensorList[l1ListId], gmWB, layoutBInL1, layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
        }

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }

        uint32_t mPartLoop = CeilDiv<L0TileShape::M>(mRound);
        uint32_t nPartLoop = CeilDiv<L0TileShape::N>(nRound);

        // main loop
        uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            uint32_t kActualNext{0};
            // preload next tile from GM to L1
            if (kLoopIdx < kTileCount - 1) {
                uint32_t kLoopIdxNext = kLoopIdx + 1;
                kActualNext = (kLoopIdxNext < kTileCount - 1) ?
                    L1TileShape::K : (actualShape.k() - kLoopIdxNext * L1TileShape::K);

                // Get L1 tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];
                // Get GM tile for next stage
                MatrixCoord gmTileAOffset{0, kLoopIdxNext * L1TileShape::K};
                MatrixCoord gmTileBOffset{kLoopIdxNext * L1TileShape::K, 0};
                auto gmTileA = gmWA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTileB = gmWB[layoutB.GetOffset(gmTileBOffset)];

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActualNext));
                copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActualNext, actualShape.n()));
                copyGmToL1B(l1BTensor, gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            } else if (hasNextBlock) {
                uint32_t kLoopIdxNext = 0;
                kActualNext = (kLoopIdxNext < kTileCount - 1) ? L1TileShape::K : (nextActualShape.k());

                // Get L1 tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];
                // Get GM tile for next stage
                MatrixCoord gmTileAOffset{0, kLoopIdxNext * L1TileShape::K};
                MatrixCoord gmTileBOffset{kLoopIdxNext * L1TileShape::K, 0};
                auto gmTileA = gmNextWA[layoutNextA.GetOffset(gmTileAOffset)];
                auto gmTileB = gmNextWB[layoutNextB.GetOffset(gmTileBOffset)];

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                auto layoutTileA = layoutNextA.GetTileLayout(MakeCoord(nextActualShape.m(), kActualNext));
                copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileB = layoutNextB.GetTileLayout(MakeCoord(kActualNext, nextActualShape.n()));
                copyGmToL1B(l1BTensor, gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }

            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];

            // Get the loop nums on L0
            uint32_t kPartLoop = CeilDiv<L0TileShape::K>(kActual);

            for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
                uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ?
                    L0TileShape::M : (mRound - mPartIdx * L0TileShape::M);

                for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                    uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ?
                        L0TileShape::K : (kActual - kPartIdx * L0TileShape::K);

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0AListId];
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                    // Locate the current tile of matrix A on L1
                    MatrixCoord l1AOffset{mPartIdx * L0TileShape::M, kPartIdx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    if ((mPartIdx == 0) && (kPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                    }

                    // Load current tile from L1 to L0A
                    copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, layoutAInL1);

                    if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    }

                    for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                        uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                            L0TileShape::N : (nRound - nPartIdx * L0TileShape::N);

                        // Locate the current tile on L0B
                        auto l0BTile = l0BTensorList[l0BListId];
                        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                        // Locate the current tile of matrix B on L1
                        MatrixCoord l1BOffset{kPartIdx * L0TileShape::K, nPartIdx * L0TileShape::N};
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffset)];

                        // Wait for mmad finished
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                        if ((kPartIdx == 0) && (nPartIdx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                        }

                        // Load current tile from L1 to L0B
                        copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, layoutBInL1);

                        // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                        }
                        // Notify to do mmad
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // Locate the current tile on L0C
                        MatrixCoord l0COffset{mPartIdx * L0TileShape::M, nPartIdx * L0TileShape::N};
                        auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0COffset)];

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                        bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        // If the unit flag is enabled, the unit flag is set according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
                                (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                unitFlag = 0b11;
                            } else {
                                unitFlag = 0b10;
                            }
                        }
                        // Perform calculation operations
                        tileMmad(l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC, unitFlag);

                        // Notify to move the next L0B tile
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
                }
            }
            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }

        // copy block out
        LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());

        if (!isFirstKSlice) {  // 切K后非第一块 开启原子加
            AscendC::SetAtomicAdd<float>();
        }
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C, 0b11);
        }
        if (!isFirstKSlice) {
            AscendC::SetAtomicNone();
        }
    }
protected:
    /// Data members
    Params params;

    // static constexpr uint32_t STAGES = 2;
    static constexpr uint32_t AIVPERCORE = 2;

    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    // The id of current stage
    uint32_t l1ListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    // CV Flag
    static constexpr Arch::FlagID flagID0 = 0;
    static constexpr Arch::FlagID flagID1 = 1;
    static constexpr Arch::FlagID flagID2 = 2;
    static constexpr Arch::FlagID flagID3 = 3;
    static constexpr Arch::FlagID flagID4 = 4;

    Arch::CrossCoreFlag flag0[STAGES];
    Arch::CrossCoreFlag flag1[STAGES];
    Arch::CrossCoreFlag flag4{flagID4};

    uint32_t crossCoreBufferIndexAIC{0};
    uint32_t crossCoreBufferIndexAIV{0};

    // Tile action
    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;

    // Buffer index (for prologue)
    uint32_t bufferIndex{0};
    
    Tile::PrologueTraits<PrologueA> prologueA;
    Tile::PrologueTraits<PrologueB> prologueB;
};

} // namespace Catlass::Gemm::Block
#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_PINGPONG_SLICE_K_WITH_PROLOGUE_HPP
