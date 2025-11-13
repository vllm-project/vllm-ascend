/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_MLA_QK_TP1_SPEC_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_MLA_QK_TP1_SPEC_HPP

#include "catlass/catlass.hpp"
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
    MmadAtlasA2MLAQKTp1Spec,
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
    using DispatchPolicy = MmadAtlasA2MLAQKTp1Spec;
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
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;
    static constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t EMBED_SPLIT_SIZE = 128;
    static constexpr uint32_t EMBED_ROPE = 64;
    static constexpr uint32_t GM_L1_EMBED_SPLIT_SIZE = 256;
    static constexpr uint32_t EMBED_SPLIT_LOOP = 5;
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * GM_L1_EMBED_SPLIT_SIZE * sizeof(ElementB);
    static constexpr uint32_t L1BROPE_SIZE = L1TileShape::N * EMBED_ROPE * sizeof(ElementB);
    static constexpr uint32_t L1B_ROPE_START = L1TileShape::N * GM_L1_EMBED_SPLIT_SIZE;

    /// Construct
    CATLASS_DEVICE
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        // Allocate L1 memory space
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart);
        for (uint32_t i = 0; i < STAGES; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1A_SIZE + L1B_SIZE * i);
            l1BRopeTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1A_SIZE +
                                                                                 L1B_SIZE * STAGES + L1BROPE_SIZE * i);
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
    void operator()(AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementA> gARope,
                    AscendC::GlobalTensor<ElementB> gB, AscendC::GlobalTensor<ElementB> gBRope,
                    AscendC::GlobalTensor<int32_t> gblockTable, AscendC::GlobalTensor<ElementC> gC, LayoutA layoutA,
                    LayoutA layoutARope, LayoutB layoutB, LayoutB layoutBRope, LayoutC layoutC, GemmCoord actualShape,
                    uint32_t &nIdx, uint32_t &nLoop, uint32_t &blockSize, uint32_t kvSeqlen)
    {
        uint32_t rowNum = actualShape.m();
        uint32_t stackSeqTile = actualShape.n();
        uint32_t seqTile = blockSize;
        uint32_t embed = layoutA.shape(1);
        uint32_t embedRope = layoutARope.shape(1);
        uint32_t embedCat = actualShape.k();
        uint32_t embedSplitGm2L1 = GM_L1_EMBED_SPLIT_SIZE;
        uint32_t stackSeqTileRound = layoutC.shape(1);
        uint32_t rowNumRound = layoutC.shape(0);
        uint32_t seqTileRound = RoundUp<BLOCK_SIZE>(seqTile);

        if (nIdx == 0) {
            // copy Q to L1
            LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);
            copyGmToL1A(l1ATensor, gA, layoutAInL1, layoutA);

            // copy QRope to L1
            LayoutAInL1 layoutARopeInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embedRope);
            copyGmToL1A(l1ATensor[rowNumRound * embed], gARope, layoutARopeInL1, layoutARope);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        }

        for (uint32_t blockStackIdx = 0; (blockStackIdx < UNIT_BLOCK_STACK_NUM) && ((nIdx + blockStackIdx) < nLoop);
             blockStackIdx++) {
            uint32_t nIdxActual = nIdx + blockStackIdx;
            uint32_t L0CPingPongFlag = nIdxActual % 2;
            uint32_t L1BRopePingPongFlag = nIdxActual % 2;
            if (nIdxActual == (nLoop - 1)) {
                seqTile = (kvSeqlen - nIdxActual * blockSize);
                seqTileRound = RoundUp<BLOCK_SIZE>(seqTile);
            }
            uint32_t blockTableId = gblockTable.GetValue(nIdxActual);
            uint64_t kvOffset = (uint64_t)blockTableId * blockSize * embed;
            uint64_t kvOffsetRope = (uint64_t)blockTableId * blockSize * embedRope;
            uint64_t l1bSplitOffset = 0;
            uint32_t embedSplitSize = EMBED_SPLIT_SIZE;
            uint32_t embedSplitLoopK = EMBED_SPLIT_LOOP;
            for (uint32_t embedSplitIdx = 0; embedSplitIdx < embedSplitLoopK; embedSplitIdx++) {
                uint32_t L0ABPingPongFlag = (blockStackIdx + embedSplitIdx) % 2;
                uint32_t innerSplitIdxL12L0 = embedSplitIdx % 2;
                uint32_t L1BPingPongFlag = embedSplitIdx / 2;
                if (embedSplitIdx == 4) {
                    embedSplitSize = embedRope;
                }
                // copy Q to l0a
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0ABPingPongFlag);
                LayoutAInL1 layoutACatSplitKInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embedSplitSize);
                LayoutAInL0 layoutACatSplitKInL0 = LayoutAInL0::template MakeLayout<ElementA>(rowNum, embedSplitSize);
                copyL1ToL0A(l0ATensor[L0ABPingPongFlag], l1ATensor[embedSplitIdx * rowNumRound * EMBED_SPLIT_SIZE],
                            layoutACatSplitKInL0, layoutACatSplitKInL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0ABPingPongFlag);
                
                if (embedSplitIdx == 0 || embedSplitIdx == 2) {
                    // copy K to l1b
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1BPingPongFlag);
                    auto layoutUnitBSplitK = layoutB.GetTileLayout(MakeCoord(embedSplitGm2L1, seqTile));
                    LayoutBInL1 layoutUnitBSplitKInL1 =
                        LayoutBInL1::template MakeLayout<ElementB>(embedSplitGm2L1, seqTile);
                    copyGmToL1B(l1BTensor[L1BPingPongFlag], gB[kvOffset + embedSplitIdx * embedSplitSize],
                                layoutUnitBSplitKInL1, layoutUnitBSplitK);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1BPingPongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1BPingPongFlag);
                } else if (embedSplitIdx == 4) {
                    // copy KRope to L1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1BRopePingPongFlag + 2);
                    auto layoutUnitBRope = layoutBRope.GetTileLayout(MakeCoord(embedRope, seqTile));
                    LayoutBInL1 layoutBRopeInL1 = LayoutBInL1::template MakeLayout<ElementB>(embedRope, seqTile);
                    copyGmToL1B(l1BRopeTensor[L1BRopePingPongFlag], gBRope[kvOffsetRope], layoutBRopeInL1,
                                layoutUnitBRope);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1BRopePingPongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1BRopePingPongFlag);
                }
                // copy K to l0b
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0ABPingPongFlag + 2);
                LayoutBInL1 layoutBCatSplitKInL1 = LayoutBInL1::template MakeLayout<ElementB>(embedSplitSize, seqTile);
                LayoutBInL0 layoutBCatSplitKInL0 = LayoutBInL0::template MakeLayout<ElementB>(embedSplitSize, seqTile);
                if (embedSplitIdx != 4) {
                    copyL1ToL0B(l0BTensor[L0ABPingPongFlag],
                                l1BTensor[L1BPingPongFlag][innerSplitIdxL12L0 * EMBED_SPLIT_SIZE * seqTileRound],
                                layoutBCatSplitKInL0, layoutBCatSplitKInL1);
                } else {
                    copyL1ToL0B(l0BTensor[L0ABPingPongFlag], l1BRopeTensor[L1BRopePingPongFlag], layoutBCatSplitKInL0,
                                layoutBCatSplitKInL1);
                }

                if (embedSplitIdx == 1 || embedSplitIdx == 3) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1BPingPongFlag);
                } else if (embedSplitIdx == 4) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1BRopePingPongFlag + 2);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0ABPingPongFlag + 2);

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0ABPingPongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0ABPingPongFlag + 2);
                if (embedSplitIdx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0CPingPongFlag);
                }
                // mmad
                tileMmad(l0CTensor[L0CPingPongFlag], l0ATensor[L0ABPingPongFlag], l0BTensor[L0ABPingPongFlag],
                         rowNumRound, seqTile, embedSplitSize, embedSplitIdx == 0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0ABPingPongFlag);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0ABPingPongFlag + 2);
            }
            // copy S to gm
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(L0CPingPongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(L0CPingPongFlag);
            auto blockShape = MakeCoord(rowNum, seqTile);
            auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
            auto layoutCSplitN = layoutC.GetTileLayout(MakeCoord(rowNumRound, seqTileRound));
            // copy L0C to gm
            copyL0CToGm(gC[blockStackIdx * blockSize], l0CTensor[L0CPingPongFlag], layoutCSplitN, layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0CPingPongFlag);
        }
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor;
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BRopeTensor[STAGES];
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

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_MLA_QK_TP1_SPEC_HPP