/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_MLA_QK_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_MLA_QK_HPP

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
    MmadAtlasA2MLAQK,
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
    using DispatchPolicy = MmadAtlasA2MLAQK;
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
    static constexpr uint32_t EMBED_SPLIT_LOOP = 5;

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    /// Construct
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
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmad() {}

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> gA,
        AscendC::GlobalTensor<ElementA> gARope,
        AscendC::GlobalTensor<ElementB> gB,
        AscendC::GlobalTensor<ElementB> gBRope,
        AscendC::GlobalTensor<ElementC> gC,
        LayoutA layoutA, LayoutA layoutARope, LayoutB layoutB, LayoutB layoutBRope, LayoutC layoutC,
        GemmCoord actualShape, MatrixCoord qShapeSingleNd,
        uint32_t &qHeads, uint32_t &nIdx)
    {
        uint32_t rowNum = actualShape.m();
        uint32_t kSeqTile = actualShape.n();
        uint32_t embed = layoutB.shape(0);
        uint32_t embedRope = layoutBRope.shape(0);
        uint32_t embedCat = actualShape.k();
        uint32_t embedSplitSize = EMBED_SPLIT_SIZE;
        uint32_t embedSplitLoopK = EMBED_SPLIT_LOOP;
        uint32_t curHeadNum = qShapeSingleNd.row();
        uint32_t tokenNumPerHead = rowNum / curHeadNum;
        uint32_t kSeqTileRound = RoundUp<L1BAlignHelper::N_ALIGNED>(kSeqTile);
        uint32_t rowNumRound = RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);
        uint32_t l1KvPingPongFlag = nIdx % 2;

        if (nIdx == 0) {
            // copy Q to L1
            auto layoutASingleNd = layoutA.GetTileLayout(MakeCoord(curHeadNum, embed));
            LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);
            copyGmToL1A(
                l1ATensor, gA,
                layoutAInL1, layoutASingleNd,
                tokenNumPerHead, qHeads * embed, tokenNumPerHead, BLOCK_SIZE, rowNumRound);

            // copy QRope to L1
            auto layoutARopeSingleNd = layoutARope.GetTileLayout(MakeCoord(curHeadNum, embedRope));
            LayoutAInL1 layoutARopeInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embedRope);
            copyGmToL1A(
                l1ATensor[rowNumRound * embed], gARope,
                layoutARopeInL1, layoutARopeSingleNd,
                tokenNumPerHead, qHeads * embedRope, tokenNumPerHead, BLOCK_SIZE, rowNumRound);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
        // copy K to L1
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(embed, kSeqTile);
        copyGmToL1B(l1BTensor[l1KvPingPongFlag], gB, layoutBInL1, layoutB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag + 2);
        // copy KRope to L1
        LayoutBInL1 layoutBRopeInL1 = LayoutBInL1::template MakeLayout<ElementB>(embedRope, kSeqTile);
        copyGmToL1B(l1BTensor[l1KvPingPongFlag][kSeqTileRound * embed], gBRope, layoutBRopeInL1, layoutBRope);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag + 2);

        for (uint32_t embedSplitIdx = 0; embedSplitIdx < embedSplitLoopK; embedSplitIdx++) {
            uint32_t l0ABPingPongFlag = embedSplitIdx % 2;
            if (embedSplitIdx == embedSplitLoopK - 1) {
                embedSplitSize = embedRope;
            }
            // copy Q from L1 to l0a
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
            LayoutAInL1 layoutACatInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embedCat);
            LayoutAInL0 layoutACatInL0 = LayoutAInL0::template MakeLayout<ElementA>(rowNum, embedSplitSize);
            copyL1ToL0A(
                l0ATensor[l0ABPingPongFlag], l1ATensor[embedSplitIdx * rowNumRound * EMBED_SPLIT_SIZE],
                layoutACatInL0, layoutACatInL1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag);

            if (embedSplitIdx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
            } else if (embedSplitIdx == embedSplitLoopK - 1) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag + 2);
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
            // copy K from L1 to l0b
            LayoutBInL1 layoutBCatInL1 = LayoutBInL1::template MakeLayout<ElementB>(embedCat, kSeqTile);
            LayoutBInL0 layoutBCatInL0 = LayoutBInL0::template MakeLayout<ElementB>(embedSplitSize, kSeqTile);
            copyL1ToL0B(l0BTensor[l0ABPingPongFlag],
                        l1BTensor[l1KvPingPongFlag][embedSplitIdx * kSeqTileRound * EMBED_SPLIT_SIZE],
                        layoutBCatInL0, layoutBCatInL1);
            if (embedSplitIdx == embedSplitLoopK - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag + 2);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag + 2);
            // mmad
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag + 2);
            if (embedSplitIdx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l1KvPingPongFlag);
            }
            // mmad
            tileMmad(
                l0CTensor[l1KvPingPongFlag], l0ATensor[l0ABPingPongFlag], l0BTensor[l0ABPingPongFlag],
                rowNumRound, kSeqTile, embedSplitSize, embedSplitIdx == 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
        }
        // copy S to gm
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l1KvPingPongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l1KvPingPongFlag);
        auto blockShape = MakeCoord(rowNum, kSeqTile);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
        // copy L0C to gm
        copyL0CToGm(gC, l0CTensor[l1KvPingPongFlag], layoutC, layoutInL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l1KvPingPongFlag);
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

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_MLA_QK_HPP