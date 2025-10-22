/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_MLA_PV_TP1_SPEC_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_MLA_PV_TP1_SPEC_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"

////////////////////////////////////////////////////////////////////

namespace Catlass::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_>
struct BlockMmad<
    MmadAtlasA2MLAPVTp1Spec,
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
    using DispatchPolicy = MmadAtlasA2MLAPV;
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
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;
    static constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;
    static constexpr uint32_t EMBED_SPLIT_LOOP = 4;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t EMBED_SPLIT_SIZE = 128;
    static constexpr uint32_t L1_PV_ADDR_START = 311296; // reserved for Q(no db) and K(db)
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::N * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * EMBED_SPLIT_SIZE * sizeof(ElementB);

    /// Construct
    CATLASS_DEVICE
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = L1_PV_ADDR_START)
    {
        // Allocate L1 memory space
        for (uint32_t i = 0; i < STAGES; i++) {
            l1ATensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + L1B_SIZE * STAGES + L1A_SIZE * i);
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1B_SIZE * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmad() {}

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementA> gB,
                    AscendC::GlobalTensor<int32_t> gblockTable, AscendC::GlobalTensor<ElementC> gC, LayoutA layoutA,
                    LayoutB layoutB, LayoutC layoutC, GemmCoord actualShape, uint32_t &nIdx, uint32_t &nLoop,
                    uint32_t &blockSize, uint32_t kvSeqlen, Arch::CrossCoreFlag softmaxReady)
    {
        uint32_t rowNum = actualShape.m();
        uint32_t stackSeqTile = actualShape.k();
        uint32_t seqTile = blockSize;
        uint32_t embed = actualShape.n();
        uint32_t embedSplitSize = EMBED_SPLIT_SIZE;
        uint32_t embedSplitLoopV = EMBED_SPLIT_LOOP;
        uint32_t rowNumRound = RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);
        uint32_t embedSplitSizeRound = RoundUp<L1BAlignHelper::N_ALIGNED>(embedSplitSize);
        uint32_t stackSeqTileRound = layoutA.stride(0);
        uint32_t seqTileRound = RoundUp<BLOCK_SIZE>(seqTile);

        for (uint32_t embedSplitIdx = 0; embedSplitIdx < embedSplitLoopV; embedSplitIdx++) {
            uint32_t L0CPingPongFlag = embedSplitIdx % 2;
            for (uint32_t blockStackIdx = 0;
                 (blockStackIdx < UNIT_BLOCK_STACK_NUM) && ((nIdx + blockStackIdx) < (nLoop + UNIT_BLOCK_STACK_NUM));
                 blockStackIdx++) {
                uint32_t nIdxActual = nIdx + blockStackIdx;
                if (nIdxActual == (nLoop + UNIT_BLOCK_STACK_NUM - 1)) {
                    seqTile = (kvSeqlen - (nIdxActual - UNIT_BLOCK_STACK_NUM) * blockSize);
                } else {
                    seqTile = blockSize;
                }
                seqTileRound = RoundUp<BLOCK_SIZE>(seqTile);
                uint32_t L1ABPingPongFlag = nIdxActual % 2;
                uint32_t L0ABPingPongFlag = nIdxActual % 2;
                uint32_t blockTableId = gblockTable.GetValue(nIdxActual - UNIT_BLOCK_STACK_NUM);
                uint64_t kvOffset = (uint64_t)blockTableId * blockSize * embed + embedSplitIdx * embedSplitSize;

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1ABPingPongFlag + 4);
                auto layoutUnitBSplitN = layoutB.GetTileLayout(MakeCoord(seqTile, embedSplitSize));
                // copy V to L1
                LayoutBInL1 layoutUnitBSplitNInL1 = LayoutBInL1::template MakeLayout<ElementB>(seqTile, embedSplitSize);
                copyGmToL1B(l1BTensor[L1ABPingPongFlag], gB[kvOffset], layoutUnitBSplitNInL1, layoutUnitBSplitN);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1ABPingPongFlag);

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1ABPingPongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0ABPingPongFlag + 2);
                LayoutBInL0 layoutUnitBSplitNInL0 = LayoutBInL0::template MakeLayout<ElementB>(seqTile, embedSplitSize);
                // copy V from L1 to L0B
                copyL1ToL0B(l0BTensor[L0ABPingPongFlag], l1BTensor[L1ABPingPongFlag], layoutUnitBSplitNInL0,
                            layoutUnitBSplitNInL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1ABPingPongFlag + 4);

                if (embedSplitIdx == 0 && blockStackIdx == 0) {
                    Arch::CrossCoreWaitFlag(softmaxReady);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1ABPingPongFlag + 6);
                auto layoutASplitK = layoutA.GetTileLayout(MakeCoord(rowNum, seqTile));
                LayoutAInL1 layoutASplitKInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, seqTile);
                // copy P to L1
                copyGmToL1A(l1ATensor[L1ABPingPongFlag], gA[blockStackIdx * blockSize], layoutASplitKInL1,
                            layoutASplitK);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID7);
                // copy P to l0a
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0ABPingPongFlag);
                LayoutAInL0 layoutASplitKInL0 = LayoutAInL0::template MakeLayout<ElementA>(rowNum, seqTile);
                copyL1ToL0A(l0ATensor[L0ABPingPongFlag], l1ATensor[L1ABPingPongFlag], layoutASplitKInL0,
                            layoutASplitKInL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1ABPingPongFlag + 6);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0ABPingPongFlag);
                // mmad
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0ABPingPongFlag);
                if (blockStackIdx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0CPingPongFlag);
                }
                tileMmad(l0CTensor[L0CPingPongFlag], l0ATensor[L0ABPingPongFlag], l0BTensor[L0ABPingPongFlag],
                         rowNumRound, embedSplitSize, seqTile, blockStackIdx == 0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0ABPingPongFlag + 2);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0ABPingPongFlag);
            }
            // copy Otmp to gm
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(L0CPingPongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(L0CPingPongFlag);
            auto blockShape = MakeCoord(rowNum, embedSplitSize);
            auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
            auto layoutCSplitN = layoutC.GetTileLayout(MakeCoord(rowNumRound, embedSplitSizeRound));
            copyL0CToGm(gC[embedSplitIdx * embedSplitSizeRound], l0CTensor[L0CPingPongFlag], layoutCSplitN,
                        layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0CPingPongFlag);
        }
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
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

} // namespace Catlass::Gemm::block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_MLA_PV_TP1_SPEC_HPP