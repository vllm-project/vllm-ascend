/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_MMAD_QK_HPP
#define CATLASS_GEMM_BLOCK_MMAD_QK_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

////////////////////////////////////////////////////////////////////

namespace Catlass::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <
    bool PAGED_CACHE_FLAG_,
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmad<
    MmadAtlasA2FAIQK<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2FAIQK<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
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
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t EMBED_SPLIT_SIZE = 128;
    static constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;
    static constexpr uint32_t KV_BASE_BLOCK = 512;
    static constexpr uint32_t KV_SPLIT_SIZE = 128;

    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    CATLASS_DEVICE
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        // Allocate L1 memory space
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart);
        for (uint32_t i = 0; i < STAGES; i++) {
            l1BTensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1A_SIZE + L1B_SIZE * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    }

    CATLASS_DEVICE
    ~BlockMmad() {}

    CATLASS_DEVICE
    void loadQGM(AscendC::GlobalTensor<ElementA> gA, LayoutA layoutA, uint32_t rowNum, uint32_t &singleGroupHeads, uint32_t &qHeads){
        uint32_t embed = layoutA.shape(1);
        uint32_t rowNumRound = RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);
        uint32_t tokenNumPerGroup = rowNum / singleGroupHeads;
        auto layoutSingleANd = layoutA.GetTileLayout(MakeCoord(singleGroupHeads, embed));
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);
        copyGmToL1A(
            l1ATensor, gA,
            layoutAInL1, layoutSingleANd,
            tokenNumPerGroup, qHeads * embed, tokenNumPerGroup, BLOCK_SIZE, rowNumRound);
        // AscendC::Nd2NzParams intriParams;
        // intriParams.nValue = singleGroupHeads;
        // intriParams.dValue = embed;
        // intriParams.srcDValue = embed;
        // intriParams.dstNzNStride = tokenNumPerGroup;
        // intriParams.dstNzC0Stride = rowNumRound;
        // intriParams.ndNum = tokenNumPerGroup;
        // intriParams.dstNzMatrixStride = 16;
        // AscendC::DataCopy(l1ATensor, gA, intriParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
    }
    
    CATLASS_DEVICE
    void getBlockShape(GemmCoord &actualShape, uint32_t &nowNIdx, uint32_t &kIdx,
                       uint32_t &nLoop, uint32_t &kLoop, uint32_t &kvSeqlen, uint32_t &embed, bool firstBlock, uint32_t maskTailS = 0)
    {
        uint32_t nSplitSize = KV_SPLIT_SIZE;
        uint32_t embedSplitSize = EMBED_SPLIT_SIZE;
        if (nowNIdx == nLoop - 1) {
            nSplitSize = kvSeqlen - nowNIdx * KV_SPLIT_SIZE;
        }
        if (firstBlock && maskTailS != 0) {
            nSplitSize = nSplitSize - maskTailS;
        }
        // }
        if (kIdx == kLoop - 1) {
            embedSplitSize = embed - kIdx * EMBED_SPLIT_SIZE;
        }
        actualShape[1] = nSplitSize;
        actualShape[2] = embedSplitSize;
    }

    CATLASS_DEVICE
    void getKVOffset(AscendC::GlobalTensor<int32_t> &gBlockTable, uint32_t &kOffset, uint32_t &nowNIdx, uint32_t &kIdx,
                     uint32_t &nLoop, uint32_t &kLoop, uint32_t &strideKV, uint32_t &blockSize, uint32_t maskTailS = 0)
    {
        if (nowNIdx >= nLoop || kIdx >= kLoop) {
            kOffset = 0;
        }
        if constexpr (PAGED_CACHE_FLAG_) {
            uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);
            kOffset = blockTableId * blockSize * strideKV + maskTailS * strideKV + kIdx * EMBED_SPLIT_SIZE;
        } else {
            kOffset = nowNIdx * KV_SPLIT_SIZE * strideKV + kIdx * EMBED_SPLIT_SIZE;
        }
    }


    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementA> gA,
                    AscendC::GlobalTensor<ElementB> gB,
                    AscendC::GlobalTensor<ElementC> gC,
                    AscendC::GlobalTensor<int32_t> gBlockTable,
                    LayoutA layoutA, LayoutB layoutB, GemmCoord actualOriShape,
                    uint32_t &nIdx, uint32_t &nLoop, uint32_t &blockSize, uint32_t kvSeqlen, uint32_t strideKV)
    {
        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[2];
        uint32_t kLoop = CeilDiv<L1TileShape::K>(embed);
        uint32_t nkBlockLoop = nLoop * kLoop;
        GemmCoord actualShape{rowNum, 0, 0};
        GemmCoord actualNextShape{rowNum, 0, 0};
        uint32_t nkBlockNextIdx = nIdx * kLoop + 1;
        uint32_t gBOffset = 0;
        uint32_t gBNextOffset = 0;
        uint32_t stackTile = 0;
        for (uint32_t blockStackIdx = 0; (blockStackIdx < UNIT_BLOCK_STACK_NUM) && ((nIdx + blockStackIdx) < nLoop);
            ++blockStackIdx) {
            for (uint32_t kIdx = 0; kIdx < kLoop; kIdx++) {
                uint32_t nowNIdx = nIdx + blockStackIdx;
                uint32_t nLoopNextIdx = nkBlockNextIdx / kLoop;
                uint32_t kLoopNextIdx = nkBlockNextIdx % kLoop;
                uint32_t gCOffset = blockStackIdx / 2 * 2 * KV_SPLIT_SIZE;
                getBlockShape(actualShape, nowNIdx, kIdx, nLoop, kLoop, kvSeqlen, embed, nowNIdx == nIdx);
                getBlockShape(actualNextShape, nLoopNextIdx, kLoopNextIdx, nLoop, kLoop, kvSeqlen, embed, nLoopNextIdx == nIdx);
                getKVOffset(gBlockTable, gBOffset, nowNIdx, kIdx, nLoop, kLoop, strideKV, blockSize);
                getKVOffset(gBlockTable, gBNextOffset, nLoopNextIdx, kLoopNextIdx, nLoop, kLoop, strideKV, blockSize);
                bool firstItr = ((blockStackIdx % 2) == 0) && (kIdx == 0);
                bool endItr = (((blockStackIdx % 2) == 1) || (nowNIdx == nLoop - 1)) && (kIdx == kLoop - 1);
                bool firstQtr = blockStackIdx == 0;
                bool endQItr = ((nowNIdx == nLoop - 1) || (blockStackIdx == UNIT_BLOCK_STACK_NUM - 1)) && (kIdx == kLoop - 1);
                int cc = 1;
                bool initMmad = kIdx == 0;
                stackTile += actualShape[1];
                LayoutC layOutSTemp(rowNum, stackTile, 512);
                // AscendC::printf("firstItr:%d\n", firstItr);
                // AscendC::printf("endItr:%d\n", endItr);
                // AscendC::printf("initMmad:%d\n", initMmad);
                // AscendC::printf("stackTile:%d\n", stackTile);
                // AscendC::printf("blockStackIdx:%d\n", blockStackIdx);
                // AscendC::printf("gBOffset:%d\n", gBOffset);
                // AscendC::printf("gBNextOffset:%d\n", gBNextOffset);
                // AscendC::printf("actualShape[0]:%d\n", actualShape[0]);
                // AscendC::printf("actualShape[1]:%d\n", actualShape[1]);
                // AscendC::printf("actualShape[2]:%d\n", actualShape[2]);
                // AscendC::printf("actualShape.m():%d\n", actualShape.m());
                // AscendC::printf("actualShape.n():%d\n", actualShape.n());
                // AscendC::printf("actualShape.k():%d\n", actualShape.k());
                // AscendC::printf("actualNextShape[0]:%d\n", actualNextShape[0]);
                // AscendC::printf("actualNextShape[1]:%d\n", actualNextShape[1]);
                // AscendC::printf("actualNextShape[2]:%d\n", actualNextShape[2]);
                computeQK(gA, gB[gBOffset], gC[gCOffset], gB[gBNextOffset], layoutA, layoutB, layOutSTemp,
                          actualShape, actualNextShape, blockStackIdx, nkBlockNextIdx, nkBlockLoop, firstItr, endItr, initMmad, firstQtr, endQItr);
                ++nkBlockNextIdx;
                if (endItr) {
                    stackTile = 0;
                }
            }
        }
    }

    CATLASS_DEVICE void computeQK(
        AscendC::GlobalTensor<ElementA> const &gA,
        AscendC::GlobalTensor<ElementB> const &gB,
        AscendC::GlobalTensor<ElementC> const &gC,
        AscendC::GlobalTensor<ElementB> const &gmNextBlockB,
        LayoutA layoutA, LayoutB layoutB, LayoutC layoutC,
        GemmCoord actualShape, GemmCoord actualNextShape, uint32_t nowIdx, uint32_t &nkblockIdx,
        uint32_t &nkblockLoop, bool firstItr, bool endItr, bool initMmad, bool firstQItr, bool endQItr)
    {
        uint32_t mActual = actualShape.m();
        uint32_t kActual = actualShape.k();
        uint32_t nActual = actualShape.n();
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mActual);
        uint32_t kRound = RoundUp<L1AAlignHelper::M_ALIGNED>(kActual);
        uint32_t nRound = RoundUp<L1AAlignHelper::M_ALIGNED>(nActual);
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(mRound, kActual); // embed
        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kActual);
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);
        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kActual, nRound);
        uint32_t locPingPongFlag = nowIdx % 2;
        uint32_t l1KvPingPongFlag = nkblockIdx % 2;
        uint32_t l0ABPingPongFlag = nkblockIdx % 2;
        if (nkblockIdx == 1) {
            auto layoutBTile = layoutB.GetTileLayout(MakeCoord(kActual, nActual));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
            copyGmToL1B(l1BTensor[l1KvPingPongFlag], gB, layoutBInL1, layoutBTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
        }
        if (nkblockIdx != nkblockLoop) {
            uint32_t nNextActual = actualNextShape.n();
            uint32_t kNextActual = actualNextShape.k();
            LayoutBInL1 layoutBNextInL1 = LayoutBInL1::template MakeLayout<ElementB>(kNextActual, nNextActual);
            auto layoutNextBTile = layoutB.GetTileLayout(MakeCoord(kNextActual, nNextActual));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1 - l1KvPingPongFlag);
            copyGmToL1B(l1BTensor[1 - l1KvPingPongFlag], gmNextBlockB, layoutBNextInL1, layoutNextBTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1 - l1KvPingPongFlag);
        }
        if(firstQItr) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
            copyL1ToL0A(l0ATensor[0], l1ATensor, layoutAInL0, layoutAInL1);
        }
        
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
        copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTensor[l1KvPingPongFlag], layoutBInL0, layoutBInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag);
        uint8_t unitFlag = 0b00;
        if constexpr (!ENABLE_UNIT_FLAG_) {
            if (firstItr) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
            }
        } else {
            unitFlag = 0b11;
        }
        tileMmad(
            l0CTensor[locPingPongFlag * mRound * 128], l0ATensor[0], l0BTensor[l0ABPingPongFlag],
            mRound, nActual, kActual, initMmad, unitFlag);
        // AscendC::PipeBarrier<PIPE_M>();
        if (endQItr) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mActual, (uint32_t)256));
        if (endItr) {
            if constexpr (!ENABLE_UNIT_FLAG_) {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l1KvPingPongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l1KvPingPongFlag);
                copyL0CToGm(gC, l0CTensor, layoutC, layoutInL0C);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
            } else {
                copyL0CToGm(gC, l0CTensor, layoutC, layoutInL0C, unitFlag);
            }
        }
    }
 
protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor;
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_MMAD_QK_HPP