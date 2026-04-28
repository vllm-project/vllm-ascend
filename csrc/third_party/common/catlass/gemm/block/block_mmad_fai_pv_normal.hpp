/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_MMAD_PV_HPP
#define CATLASS_GEMM_BLOCK_MMAD_PV_HPP

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
    MmadAtlasA2FAIPV<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>,
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
    using DispatchPolicy = MmadAtlasA2FAIPV<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
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
    static constexpr uint32_t LOAB_BLOCK = 1;

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    /// Construct
    CATLASS_DEVICE
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        // Allocate L1 memory space
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart);
        for (uint32_t i = 0; i < STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
            l1BTensor[i] =
            resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1A_SIZE * 2 * 2 + L1B_SIZE * i);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmad() {}
    
    CATLASS_DEVICE
    void getBlockShape(GemmCoord &actualShape, uint32_t &nowNIdx, uint32_t &kIdx,
                       uint32_t &nLoop, uint32_t &kLoop, uint32_t &kvSeqlen, uint32_t &embed, bool firstBlock, uint32_t maskTailS = 0)
    {
        uint32_t nSplitSize = KV_SPLIT_SIZE * LOAB_BLOCK;
        uint32_t embedSplitSize = EMBED_SPLIT_SIZE;
        if (nowNIdx + LOAB_BLOCK > nLoop - 1) {
            nSplitSize = kvSeqlen - nowNIdx * KV_SPLIT_SIZE;
        }
        if (firstBlock && maskTailS != 0) {
            nSplitSize = nSplitSize - maskTailS;
        }
        if (kIdx == kLoop - 1) {
            embedSplitSize = embed - kIdx * EMBED_SPLIT_SIZE;
        }
        actualShape[1] = embedSplitSize;
        actualShape[2] = nSplitSize;
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
                    uint32_t &nIdx, uint32_t &nLoop, uint32_t &blockSize, uint32_t kvSeqlen, uint32_t strideKV, Arch::CrossCoreFlag softmaxFlag)
    {
        // Arch::CrossCoreWaitFlag(softmaxFlag);
        uint32_t embed = actualOriShape[1];
        uint32_t kLoop = CeilDiv<L1TileShape::K>(embed);
        uint32_t rowNum = layoutA.shape(0);
        uint32_t blockN = layoutA.shape(1);
        GemmCoord actualShape{rowNum, 0, 0};
        GemmCoord actualNextShape{rowNum, 0, 0};
        uint32_t nkBlockLoop = (nLoop + LOAB_BLOCK - 1) / LOAB_BLOCK * kLoop; // gap
        uint32_t nkBlockNextIdx = (nIdx + LOAB_BLOCK - 1) / LOAB_BLOCK * kLoop + 1; // gap
        uint32_t gBOffset = 0;
        uint32_t gBNextOffset = 0;
        for (uint32_t kIdx = 0; kIdx < kLoop; kIdx++) {
            for (uint32_t blockStackIdx = 0; (blockStackIdx < UNIT_BLOCK_STACK_NUM) && ((nIdx + blockStackIdx) < nLoop);
                 blockStackIdx += LOAB_BLOCK) {
                uint32_t nowNIdx = nIdx + blockStackIdx;
                uint32_t kLoopNextIdx = (nkBlockNextIdx % (kLoop * UNIT_BLOCK_STACK_NUM)) / (UNIT_BLOCK_STACK_NUM / LOAB_BLOCK);
                uint32_t nLoopNextIdx = (nkBlockNextIdx % (kLoop * UNIT_BLOCK_STACK_NUM)) % (UNIT_BLOCK_STACK_NUM / LOAB_BLOCK) + nkBlockNextIdx / (kLoop * UNIT_BLOCK_STACK_NUM) * UNIT_BLOCK_STACK_NUM;
                getBlockShape(actualShape, nowNIdx, kIdx, nLoop, kLoop, kvSeqlen, embed, nowNIdx == nIdx);
                getBlockShape(actualNextShape, nLoopNextIdx, kLoopNextIdx, nLoop, kLoop, kvSeqlen, embed, nowNIdx == nIdx);
                getKVOffset(gBlockTable, gBOffset, nowNIdx, kIdx, nLoop, kLoop, strideKV, blockSize);
                getKVOffset(gBlockTable, gBNextOffset, nLoopNextIdx, kLoopNextIdx, nLoop, kLoop, strideKV, blockSize);
                bool firstItr = blockStackIdx == 0;
                bool endItr = (blockStackIdx + LOAB_BLOCK > UNIT_BLOCK_STACK_NUM - 1) || (nowNIdx + LOAB_BLOCK > nLoop - 1);
                bool initMmad = blockStackIdx == 0;
                bool pvCVItr = firstItr && kIdx == 0;
                LayoutC layoutOTmpTemp(rowNum, embed, embed);
                computePV(gA, gB[gBOffset], gC, gB[gBNextOffset], layoutA, layoutB, layoutOTmpTemp,
                          actualShape, actualNextShape, blockStackIdx, nkBlockNextIdx, nkBlockLoop, firstItr, endItr, initMmad, pvCVItr, softmaxFlag);
                ++nkBlockNextIdx;
            }
        }
    }

    CATLASS_DEVICE
    void computePV(
        AscendC::GlobalTensor<ElementA> const &gA,
        AscendC::GlobalTensor<ElementB> const &gB,
        AscendC::GlobalTensor<ElementC> const &gC,
        AscendC::GlobalTensor<ElementB> const &gmNextBlockB,
        LayoutA layoutA, LayoutB layoutB, LayoutC layoutC,
        GemmCoord actualShape, GemmCoord actualNextShape, uint32_t nowIdx, uint32_t &nkblockIdx,
        uint32_t &nkblockLoop, bool firstItr, bool endItr, bool initMmad, bool pvCVItr, Arch::CrossCoreFlag softmaxFlag, bool preloadFlag = false)
    {
        uint32_t MActual = actualShape.m();
        uint32_t kActual = actualShape.k();
        uint32_t nActual = actualShape.n();
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(MActual);
        uint32_t kRound = RoundUp<L1AAlignHelper::M_ALIGNED>(kActual);
        uint32_t nRound = RoundUp<L1AAlignHelper::M_ALIGNED>(nActual);
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(mRound, (uint32_t)512);
        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kActual);
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);
        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kActual, nActual);
        uint32_t l1KvPingPongFlag = nkblockIdx % 2;
        uint32_t l0ABPingPongFlag = nkblockIdx % 2;
        if (nkblockIdx == 1 || preloadFlag) {
            auto layoutBTile = layoutB.GetTileLayout(MakeCoord(kActual, nActual));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag + 2);
            copyGmToL1B(l1BTensor[l1KvPingPongFlag], gB, layoutBInL1, layoutBTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag + 2);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag + 2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
        copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTensor[l1KvPingPongFlag], layoutBInL0, layoutBInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag + 2);

        if (pvCVItr) {
            Arch::CrossCoreWaitFlag(softmaxFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(5);
            auto layoutATile = layoutA.GetTileLayout(MakeCoord(MActual, (uint32_t)512));
            copyGmToL1A(l1ATensor, gA, layoutAInL1, layoutATile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(4);
        }

        if (nkblockIdx != nkblockLoop) {
            uint32_t nNextActual = actualNextShape.n();
            uint32_t kNextActual = actualNextShape.k();
            LayoutBInL1 layoutBNextInL1 = LayoutBInL1::template MakeLayout<ElementB>(kNextActual, nNextActual);
            auto layoutNextBTile = layoutB.GetTileLayout(MakeCoord(kNextActual, nNextActual));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1 - l1KvPingPongFlag + 2);
            copyGmToL1B(l1BTensor[1 - l1KvPingPongFlag], gmNextBlockB, layoutBNextInL1, layoutNextBTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1 - l1KvPingPongFlag + 2);
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
        copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATensor[mRound * 128 * nowIdx], layoutAInL0, layoutAInL1);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag);
        uint8_t unitFlag = 0b00;
        if constexpr (!ENABLE_UNIT_FLAG_) {
            if (firstItr) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
            }
        } else {
            if (endItr) {
                unitFlag = 0b11;
            } else {
                unitFlag = 0b10;
            }
        }
        tileMmad(
            l0CTensor[0], l0ATensor[l0ABPingPongFlag], l0BTensor[l0ABPingPongFlag],
            mRound, nActual, kActual, initMmad, unitFlag);
        // AscendC::PipeBarrier<PIPE_M>();
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(MActual, nActual));
        if (endItr) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(5);
            if constexpr (!ENABLE_UNIT_FLAG_) {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
                copyL0CToGm(gC, l0CTensor[0], layoutC, layoutInL0C);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
            } else {
                copyL0CToGm(gC, l0CTensor[0], layoutC, layoutInL0C, 0b11);
            }
        }
    }
 
protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor;
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_MMAD_PV_HPP