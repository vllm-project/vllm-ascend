/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_MMAD_PV_HPP
#define BLOCK_MMAD_PV_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/coord.hpp"
#include "../../../attn_infra/arch/cross_core_sync.hpp"
#include "../../../attn_infra/gemm/dispatch_policy.hpp"
#include "../../../attn_infra/gemm/helper.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_mmad.hpp"

////////////////////////////////////////////////////////////////////

namespace NpuArch::Gemm::Block {
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
    MmadAtlasA2SFAIPV<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>,
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
    using DispatchPolicy = MmadAtlasA2SFAIPV<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
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

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    /// Construct
    __aicore__ inline
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        // Allocate L1 memory space
        l1BTensor = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1A_SIZE * 2);
        for (uint32_t i = 0; i < STAGES; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + L1A_SIZE * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    /// Destructor
    __aicore__ inline
    ~BlockMmad() {}

    __aicore__ inline
    void getBlockShape(
        GemmCoord &actualShape, uint32_t &nowNIdx, uint32_t &nLoop, uint32_t &stackSeqTile, uint32_t &blockSize)
    {
        uint32_t nSplitSize = blockSize;
        if (nowNIdx == nLoop - 1) {
            nSplitSize = stackSeqTile - nowNIdx * blockSize;
        }
        actualShape[2] = nSplitSize;
    }

    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementA> gA,
                    AscendC::GlobalTensor<ElementB> gB,
                    AscendC::GlobalTensor<ElementC> gC,
                    AscendC::GlobalTensor<int32_t> gBlockTable,
                    AscendC::GlobalTensor<int32_t> gSelectIdx,
                    LayoutA layoutA, LayoutB layoutB, LayoutC layoutC,GemmCoord actualOriShape,
                    uint32_t nIdx, uint32_t nLoop, uint32_t &blockSize, uint32_t kvSeqlen, uint32_t strideKV,
                    uint32_t blockStackNum, Arch::CrossCoreFlag softmaxFlag, 
                    uint32_t &y, uint32_t selectNum, uint32_t kvYBlockNum)
    {
        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        uint32_t stackSeqTile = actualOriShape[2];
        GemmCoord actualShape{rowNum, embed, 0};

        // load V
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(stackSeqTile, embed);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        uint32_t nL1Loop = CeilDiv<L1TileShape::N>(stackSeqTile);

        for (uint32_t blockStackIdx = 0; blockStackIdx < nL1Loop; ++blockStackIdx) {
            uint32_t nowNIdx = nIdx + blockStackIdx;
            getBlockShape(actualShape, blockStackIdx, nL1Loop, stackSeqTile, blockSize);
            uint32_t kActual = actualShape.k(); 
            uint32_t nActual = actualShape.n();

            uint32_t processSize = 0;
            uint32_t nBlockOffset = nowNIdx * blockSize;
            uint32_t currentSelectYIdx = nBlockOffset / y;
            uint32_t currentYoffset = nBlockOffset % y;
            uint32_t currentYIdx = gSelectIdx.GetValue(currentSelectYIdx);
            uint32_t offsetInKV = currentYIdx * y + currentYoffset;

            while (processSize < kActual && currentSelectYIdx < selectNum && currentYIdx < kvYBlockNum && offsetInKV < kvSeqlen) {
                uint32_t yAcutal = (currentSelectYIdx == selectNum - 1 && currentYIdx == kvYBlockNum - 1 && kvSeqlen % y != 0) ? 
                                    (kvSeqlen - y * currentYIdx) : y;
                uint32_t remainingInYBlock = yAcutal - currentYoffset;
                uint32_t remainingInNBlock = kActual - processSize;

                uint32_t actualYSize = min(remainingInNBlock, remainingInYBlock);
                if (actualYSize == 0) {
                    break;
                }

                auto layoutBTile = layoutB.GetTileLayout(MakeCoord(actualYSize, nActual));
                MatrixCoord l1BTileCoord{blockStackIdx * blockSize + processSize, 0};
                auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];
           
                copyGmToL1B(l1BTile, gB[offsetInKV * strideKV], layoutBInL1, layoutBTile);
                
                processSize += actualYSize;
                currentYoffset += actualYSize;
                offsetInKV += actualYSize;

                if (currentYoffset >= yAcutal) {
                    currentSelectYIdx++;
                    if (currentSelectYIdx >= selectNum) {
                        break;
                    }
                    currentYoffset = 0;
                    currentYIdx = gSelectIdx.GetValue(currentSelectYIdx);
                    offsetInKV = currentYIdx * y;
                }
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

        Arch::CrossCoreWaitFlag(softmaxFlag);

        uint32_t mL1Loop = CeilDiv<L1TileShape::M>(rowNum);
        uint32_t kL1Loop = CeilDiv<L1TileShape::K>(stackSeqTile);
        for (uint32_t mL1Idx = 0; mL1Idx < mL1Loop; mL1Idx++) {
            uint32_t mL1Actual = (mL1Idx < mL1Loop - 1) ? L1TileShape::M : (rowNum - mL1Idx * L1TileShape::M);
            uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mL1Actual);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
            for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; kL1Idx++) {
                uint32_t kL1Actual = (kL1Idx < kL1Loop - 1) ? L1TileShape::K : (stackSeqTile - kL1Idx * L1TileShape::K);

                // load P
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1PPingPongFlag);
                MatrixCoord gmATileCoord{mL1Idx * L1TileShape::M, kL1Idx * L1TileShape::K};
                auto gmTileA = gA[layoutA.GetOffset(gmATileCoord)];
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(mL1Actual, kL1Actual));
                LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(mL1Actual, kL1Actual);
                copyGmToL1A(l1ATensor[l1PPingPongFlag], gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1PPingPongFlag);

                uint32_t kL0Loop = CeilDiv<L0TileShape::K>(kL1Actual);
                for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    uint32_t kL0Actual =
                        (kL0Idx < kL0Loop - 1) ? L0TileShape::K : (kL1Actual - kL0Idx * L0TileShape::K);

                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL1Actual, kL0Actual);
                    MatrixCoord l1ATileCoord{0, kL0Idx * L0TileShape::K};
                    auto l1ATile = l1ATensor[l1PPingPongFlag][layoutAInL1.GetOffset(l1ATileCoord)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    if (kL0Idx == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1PPingPongFlag);
                    }
                    copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATile, layoutAInL0, layoutAInL1);
                    if (kL0Idx == kL0Loop - 1) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1PPingPongFlag);
                    }

                    LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, embed);
                    MatrixCoord l1BTileCoord{kL1Idx * L1TileShape::K + kL0Idx * L0TileShape::K, 0};
                    auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
                    copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTile, layoutBInL0, layoutBInL1);

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    bool initMmad = kL1Idx == 0 && kL0Idx == 0;
                    tileMmad(l0CTensor[l0CPingPongFlag],
                        l0ATensor[l0ABPingPongFlag],
                        l0BTensor[l0ABPingPongFlag],
                        mRound,
                        embed,
                        kL0Actual,
                        initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
                    l0ABPingPongFlag = 1 - l0ABPingPongFlag;
                }
                l1PPingPongFlag = 1 - l1PPingPongFlag;
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            MatrixCoord gmCTileCoord{mL1Idx * L0TileShape::M, 0};
            LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL1Actual, embed));
            auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL1Actual, embed));
            copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPongFlag], layoutCTile, layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
            l0CPingPongFlag = 1 - l0CPingPongFlag;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor;
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;

    uint32_t l1PPingPongFlag = 0;
    uint32_t l0CPingPongFlag = 0;
    uint32_t l0ABPingPongFlag = 0;
};

////////////////////////////////////////////////////////////////////

}  // namespace NpuArch::Gemm::Block

#endif  // GEMM_BLOCK_MMAD_SFAI_PV_HPP

