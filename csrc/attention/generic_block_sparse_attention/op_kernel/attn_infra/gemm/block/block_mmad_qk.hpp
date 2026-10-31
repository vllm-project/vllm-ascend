/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_MMAD_QK_HPP
#define BLOCK_MMAD_QK_HPP

#include "../../../attn_infra/gbsa_base_defs.hpp"
#include "../../../attn_infra/arch/gbsa_resource.hpp"
#include "../../../attn_infra/gbsa_coord.hpp"
#include "../../../attn_infra/gemm/gbsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/gbsa_helper.hpp"
#include "../../../attn_infra/gbsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/gbsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/gbsa_tile_mmad.hpp"
#include "../../../gbsa_sparse_kv_geometry.hpp"
#include "../../../gbsa_sparse_pair_copy.hpp"

////////////////////////////////////////////////////////////////////

namespace NpuArch::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <bool PAGED_CACHE_FLAG_, bool ENABLE_UNIT_FLAG_, bool HAS_RESIDUAL_BLOCK_, class L1TileShape_,
          class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_, class TileCopy_,
          class TileMmad_>
struct BlockMmad<MmadAtlasA2SFAIQK<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_, HAS_RESIDUAL_BLOCK_>, L1TileShape_,
                 L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA2SFAIQK<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_, HAS_RESIDUAL_BLOCK_>;
    static constexpr bool HAS_RESIDUAL_BLOCK = DispatchPolicy::HAS_RESIDUAL_BLOCK;
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

    static constexpr uint32_t BLOCK_SIZE = 16;

    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    __aicore__ inline BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0, uint32_t kDyn = 128)
    {
        uint32_t l1ADynSize = L1TileShape::M * kDyn * sizeof(ElementA);
        uint32_t l1BDynSize = L1TileShape::N * kDyn * sizeof(ElementB);

        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart);

        for (uint32_t i = 0; i < STAGES; i++) {
            l1BTensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + l1ADynSize + l1BDynSize * i);

            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    __aicore__ inline ~BlockMmad() {}

    __aicore__ inline void loadQGM(AscendC::GlobalTensor<ElementA> gA, LayoutA layoutA, uint32_t rowNum,
                                   uint32_t& singleGroupHeads, uint64_t qStride)
    {
        uint32_t embed = layoutA.shape(1);
        uint32_t rowNumRound = RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);
        uint32_t tokenNumPerGroup = rowNum / singleGroupHeads;

        auto layoutSingleANd = layoutA.GetTileLayout(MakeCoord(singleGroupHeads, embed));

        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);

        copyGmToL1A(l1ATensor, gA, layoutAInL1, layoutSingleANd, tokenNumPerGroup, qStride, tokenNumPerGroup,
                    BLOCK_SIZE, rowNumRound);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
    }

    __aicore__ inline void getBlockShape(GemmCoord& actualShape, uint32_t nL1Idx, uint32_t nL1Loop,
                                         uint32_t stackSeqTile)
    {
        uint32_t nSplitSize = L1TileShape::N;
        if (nL1Idx == nL1Loop - 1) {
            nSplitSize = stackSeqTile - nL1Idx * L1TileShape::N;
        }
        actualShape[1] = nSplitSize;
    }

    __aicore__ inline void LoadKvL1(uint32_t stage, uint32_t nIdx, uint32_t nL1Idx, uint32_t nL1Loop,
                                    uint32_t stackSeqTile, uint32_t embed, AscendC::GlobalTensor<ElementB> gB,
                                    LayoutB layoutB, AscendC::GlobalTensor<int32_t> gBlockTable,
                                    AscendC::GlobalTensor<int32_t> gSelectIdx, uint32_t blockSize, uint32_t strideKV,
                                    uint32_t y, uint32_t selectNum, uint32_t kvYBlockNum, uint32_t kvSeqlen,
                                    uint32_t pageSize, uint64_t pageStride, uint32_t localExplicitBlockCount,
                                    uint32_t residualBlockId, uint64_t peerPageStride, GbsaSparsePairCopy::KvCopyInfo *kvCopyInfo = nullptr)
    {
        GemmCoord actualShape{0, 0, embed};
        getBlockShape(actualShape, nL1Idx, nL1Loop, stackSeqTile);
        uint32_t kActual = embed;
        uint32_t nActual = actualShape.n();
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);

        uint32_t processSize = 0;
        uint32_t nBlockOffset = nIdx * blockSize + nL1Idx * L1TileShape::N;
        const bool yFitsPage = y != 0U && pageSize == 128U && (pageSize % y) == 0U;
        const uint32_t firstSlot = nBlockOffset / y;
        const __gm__ int32_t *selectIdx = gSelectIdx.GetPhyAddr();
        const __gm__ int32_t *blockPages = nullptr;
        uint32_t cachedLogicalPage = 0xFFFFFFFFU;
        uint32_t cachedPhysicalPage = 0U;
        if constexpr (PAGED_CACHE_FLAG_) {
            blockPages = gBlockTable.GetPhyAddr();
        }
        uint32_t fastBlockCount = 0U;
        if ((L1TileShape::N == 128U || L1TileShape::N == 256U) && strideKV < 65536U && PAGED_CACHE_FLAG_ && yFitsPage &&
            (nBlockOffset % y) == 0U && firstSlot < localExplicitBlockCount) {
            const uint32_t explicitAvail = localExplicitBlockCount - firstSlot;
            const uint32_t byTokens = nActual / y;
            fastBlockCount = byTokens < explicitAvail ? byTokens : explicitAvail;
            if (fastBlockCount != 0U &&
                (static_cast<uint64_t>(selectIdx[firstSlot + fastBlockCount - 1U]) + 1U) * y > kvSeqlen) {
                fastBlockCount = 0U;
            }
        }
        uint32_t currentSelectYIdx = firstSlot;
        uint32_t currentYoffset = nBlockOffset % y;
        uint32_t currentYIdx = 0U;
        uint64_t offsetInKV = 0;

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(stage);

        if (fastBlockCount != 0U) {
            constexpr uint32_t copyTokens = GbsaSparsePairCopy::KV_COPY_SUB_TILE_TOKENS;
            const uint32_t copySubtiles = (fastBlockCount * y + copyTokens - 1U) / copyTokens;
            for (uint32_t sub = 0U; sub < copySubtiles; ++sub) {
                const uint32_t tokenOffset = sub * copyTokens;
                const uint32_t blocksLeft = fastBlockCount - tokenOffset / y;
                const uint32_t subBlocks = min(copyTokens / y, blocksLeft);
                const uint32_t copyIndex = nL1Idx * (L1TileShape::N / copyTokens) + sub;
                auto *plan = kvCopyInfo == nullptr ? nullptr : kvCopyInfo + copyIndex;
                GbsaSparsePairCopy::Load<true>(l1BTensor[stage], gB, layoutBInL1, gSelectIdx, gBlockTable,
                    firstSlot + tokenOffset / y, subBlocks, tokenOffset, kActual, strideKV, pageStride,
                    peerPageStride, y, kvYBlockNum, kvSeqlen, plan, tokenOffset);
            }
            processSize = fastBlockCount * y;
            currentSelectYIdx = firstSlot + fastBlockCount;
            currentYoffset = 0U;
        }
        if (processSize < nActual && currentSelectYIdx < selectNum) {
            currentYIdx = GbsaSparseGeometry::ReadSparseBlockId<HAS_RESIDUAL_BLOCK>(
                selectIdx, currentSelectYIdx, localExplicitBlockCount, residualBlockId);
            offsetInKV = static_cast<uint64_t>(currentYIdx) * y + currentYoffset;
            while (processSize < nActual && currentSelectYIdx < selectNum && currentYIdx < kvYBlockNum &&
                   offsetInKV < kvSeqlen) {
                uint32_t yAcutal =
                    (currentSelectYIdx == selectNum - 1 && currentYIdx == kvYBlockNum - 1 && kvSeqlen % y != 0) ?
                        (kvSeqlen - y * currentYIdx) :
                        y;
                uint32_t remainingInYBlock = yAcutal - currentYoffset;
                uint32_t remainingInNBlock = nActual - processSize;

                uint32_t actualYSize = min(remainingInNBlock, remainingInYBlock);
                if (actualYSize == 0) {
                    break;
                }

                auto layoutBTile = layoutB.GetTileLayout(MakeCoord(kActual, actualYSize));

                uint64_t sourceOffset = offsetInKV * strideKV;
                if constexpr (PAGED_CACHE_FLAG_) {
                    const uint32_t pageIdx = static_cast<uint32_t>(offsetInKV / pageSize);
                    if (pageIdx != cachedLogicalPage) {
                        cachedLogicalPage = pageIdx;
                        cachedPhysicalPage = static_cast<uint32_t>(blockPages[pageIdx]);
                    }
                    sourceOffset = static_cast<uint64_t>(cachedPhysicalPage) * pageStride +
                                   (offsetInKV % pageSize) * strideKV;
                }
                copyGmToL1B(l1BTensor[stage][layoutBInL1.GetOffset(MatrixCoord{0, processSize})], gB[sourceOffset],
                            layoutBInL1, layoutBTile);

                processSize += actualYSize;
                currentYoffset += actualYSize;
                offsetInKV += actualYSize;

                if (currentYoffset >= yAcutal) {
                    currentSelectYIdx++;
                    if (currentSelectYIdx >= selectNum) {
                        break;
                    }
                    currentYoffset = 0;
                    currentYIdx = GbsaSparseGeometry::ReadSparseBlockId<HAS_RESIDUAL_BLOCK>(
                        selectIdx, currentSelectYIdx, localExplicitBlockCount, residualBlockId);
                    offsetInKV = static_cast<uint64_t>(currentYIdx) * y;
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(stage);
    }

    __aicore__ inline void MmadQkFromL1(uint32_t stage, uint32_t nL1Idx, uint32_t nL1Loop, uint32_t stackSeqTile,
                                        uint32_t rowNum, uint32_t embed, AscendC::GlobalTensor<ElementC> gC,
                                        LayoutAInL1 layoutAInL1, LayoutC layoutC)
    {
        GemmCoord actualShape{rowNum, 0, embed};
        getBlockShape(actualShape, nL1Idx, nL1Loop, stackSeqTile);
        uint32_t mActual = actualShape.m();
        uint32_t kActual = actualShape.k();
        uint32_t nActual = actualShape.n();
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mActual);
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);

        uint32_t mL0Loop = CeilDiv<L0TileShape::M>(mActual);
        uint32_t kL0Loop = CeilDiv<L0TileShape::K>(kActual);

        for (uint32_t mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
            uint32_t mL0Actual = (mL0Idx < mL0Loop - 1) ? L0TileShape::M : (mActual - mL0Idx * L0TileShape::M);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);

            for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                uint32_t kL0Actual = (kL0Idx < kL0Loop - 1) ? L0TileShape::K : (kActual - kL0Idx * L0TileShape::K);

                LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL0Actual, kL0Actual);
                MatrixCoord l1ATileCoord{mL0Idx * L0TileShape::M, kL0Idx * L0TileShape::K};
                auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1ATileCoord)];

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATile, layoutAInL0, layoutAInL1);

                LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, nActual);
                MatrixCoord l1BTileCoord{kL0Idx * L0TileShape::K, 0};
                auto l1BTile = l1BTensor[stage][layoutBInL1.GetOffset(l1BTileCoord)];

                if ((mL0Idx == 0) && (kL0Idx == 0)) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(stage);
                }

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
                copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTile, layoutBInL0, layoutBInL1);

                if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1)) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(stage);
                }

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                bool initMmad = kL0Idx == 0;
                tileMmad(l0CTensor[l0CPingPongFlag], l0ATensor[l0ABPingPongFlag], l0BTensor[l0ABPingPongFlag], mRound,
                         nActual, kL0Actual, initMmad);

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);

                l0ABPingPongFlag = 1 - l0ABPingPongFlag;
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

            MatrixCoord gmCTileCoord{mL0Idx * L0TileShape::M, nL1Idx * L1TileShape::N};
            LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL0Actual, nActual));
            auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL0Actual, nActual));

            copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPongFlag], layoutCTile, layoutInL0C);

            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
            l0CPingPongFlag = 1 - l0CPingPongFlag;
        }
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementB> gB,
                                      AscendC::GlobalTensor<ElementC> gC, AscendC::GlobalTensor<int32_t> gBlockTable,
                                      AscendC::GlobalTensor<int32_t> gSelectIdx, LayoutA layoutA, LayoutB layoutB,
                                      LayoutC layoutC, GemmCoord actualOriShape, uint32_t nIdx, uint32_t nLoop,
                                      uint32_t &blockSize, uint32_t strideKV, uint32_t &y, uint32_t selectNum,
                                      uint32_t kvYBlockNum, uint32_t &kvSeqlen, uint32_t pageSize, uint64_t pageStride,
                                      uint32_t localExplicitBlockCount, uint32_t residualBlockId,
                                      uint64_t peerPageStride, GbsaSparsePairCopy::KvCopyInfo *kvCopyInfo = nullptr)
    {
        uint32_t rowNum = actualOriShape[0];
        uint32_t stackSeqTile = actualOriShape[1];
        uint32_t embed = actualOriShape[2];

        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);

        uint32_t nL1Loop = CeilDiv<L1TileShape::N>(stackSeqTile);
        if (nL1Loop == 0U) {
            return;
        }

        LoadKvL1(l1KvPingPongFlag, nIdx, 0U, nL1Loop, stackSeqTile, embed, gB, layoutB, gBlockTable, gSelectIdx,
                 blockSize, strideKV, y, selectNum, kvYBlockNum, kvSeqlen, pageSize, pageStride,
                 localExplicitBlockCount, residualBlockId, peerPageStride, kvCopyInfo);
        for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; ++nL1Idx) {
            uint32_t stage = l1KvPingPongFlag;
            uint32_t nextStage = 1U - stage;
            if (nL1Idx + 1U < nL1Loop) {
                LoadKvL1(nextStage, nIdx, nL1Idx + 1U, nL1Loop, stackSeqTile, embed, gB, layoutB, gBlockTable,
                         gSelectIdx, blockSize, strideKV, y, selectNum, kvYBlockNum, kvSeqlen, pageSize, pageStride,
                         localExplicitBlockCount, residualBlockId, peerPageStride, kvCopyInfo);
            }
            MmadQkFromL1(stage, nL1Idx, nL1Loop, stackSeqTile, rowNum, embed, gC, layoutAInL1, layoutC);
            l1KvPingPongFlag = nextStage;
        }
    }

protected:
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

    uint32_t l1KvPingPongFlag = 0;
    uint32_t l0CPingPongFlag = 0;
    uint32_t l0ABPingPongFlag = 0;
};

////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Block

#endif // GEMM_BLOCK_MMAD_SFAI_QK_HPP
