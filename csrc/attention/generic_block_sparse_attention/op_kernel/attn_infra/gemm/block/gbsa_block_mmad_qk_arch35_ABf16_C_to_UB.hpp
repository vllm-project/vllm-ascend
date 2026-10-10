/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief matmul implementation for single q&k^t base tile
 * This implementation is designed for the following senario:
 * A full q base tile is loaded to L1 from GM at the very beginning,
 * and it remains persistent until each k base tile is dealt
 * A full q*k^t base tile is loaded to UB from l0C, no workspace transit
 */
#ifndef GEMM_BLOCK_QK_ARCH35_ABF16_C2UB_HPP
#define GEMM_BLOCK_QK_ARCH35_ABF16_C2UB_HPP

#include "../../../attn_infra/gbsa_base_defs.hpp"
#include "../../../attn_infra/arch/gbsa_resource.hpp"
#include "../../../attn_infra/arch/gbsa_cross_core_sync.hpp"
#include "../../../attn_infra/gbsa_coord.hpp"
#include "../../../attn_infra/gemm/gbsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/gbsa_helper.hpp"
#include "../../../attn_infra/gbsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/gbsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/gbsa_tile_mmad.hpp"
#include "../../../tla/gbsa_tla_layout.hpp"
#include "../../../tla/gbsa_tla_tensor.hpp"
#include "../../../arch35/gbsa_layout_utils.hpp"

////////////////////////////////////////////////////////////////////

namespace NpuArch::Gemm::Block {
////////////////////////////////////////////////////////////////////

struct Mm1L1TileHelper {
    uint32_t mm1L1TileM;
    uint32_t mm1L1TileN;
    uint32_t mm1L1TileKLeft;
    uint32_t mm1L1TileKRight;
    uint32_t qL1BufNum;
    uint32_t kL1BufNum;

    __aicore__ inline Mm1L1TileHelper() {}

    __aicore__ inline Mm1L1TileHelper(uint32_t m, uint32_t n, uint32_t kl, uint32_t kr, uint32_t pbn, uint32_t vbn)
        : mm1L1TileM(m),
          mm1L1TileN(n),
          mm1L1TileKLeft(kl),
          mm1L1TileKRight(kr),
          qL1BufNum(pbn),
          kL1BufNum(vbn)
    {}
};

template <bool PAGED_CACHE_FLAG_, bool HAS_RESIDUAL_BLOCK_, bool PRECOMPUTE_SPARSE_, class L1TileShape_, class L0TileShape_,
          class ElementA_, class ElementB_,
          class ElementC_, class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadAtlasA5BsaQK<PAGED_CACHE_FLAG_, HAS_RESIDUAL_BLOCK_, PRECOMPUTE_SPARSE_>, L1TileShape_, L0TileShape_,
                    ElementA_, ElementB_,
                    ElementC_, ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA5BsaQK<PAGED_CACHE_FLAG_, HAS_RESIDUAL_BLOCK_, PRECOMPUTE_SPARSE_>;
    static constexpr bool PAGED_CACHE_FLAG = DispatchPolicy::PAGED_CACHE_FLAG;
    static constexpr bool HAS_RESIDUAL_BLOCK = DispatchPolicy::HAS_RESIDUAL_BLOCK;
    static constexpr bool PRECOMPUTE_SPARSE = DispatchPolicy::PRECOMPUTE_SPARSE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGES;
    static constexpr uint32_t L0C_HALF_BUF_SIZE = ArchTag::L0C_SIZE / 2;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_HALF_BUF_SIZE / L0_STAGES;

    static constexpr uint32_t MAX_L1_STAGES = 3; // 编译期常量，为静态L1Tensor数组开辟准备。取一个buffer份数的极大值
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16; // 核间同步mode4，AIC侧需要两个flagId分别对应两个AIV

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag> &resource, Mm1L1TileHelper &mm1L1TileHelper)
    {
        l1ATileM = mm1L1TileHelper.mm1L1TileM;
        l1BTileN = mm1L1TileHelper.mm1L1TileN;
        l1ATileK = mm1L1TileHelper.mm1L1TileKLeft;
        l1BTileK = mm1L1TileHelper.mm1L1TileKRight;
        l1ABufNum = mm1L1TileHelper.qL1BufNum;
        l1BBufNum = mm1L1TileHelper.kL1BufNum;

        for (uint32_t i = 0; i < l1ABufNum; i++) {
            l1ATensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementA>(l1ATileM * l1ATileK * sizeof(ElementA) * i);
        }
        for (uint32_t i = 0; i < l1BBufNum; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1ATileM * l1ATileK * sizeof(ElementA) * l1ABufNum + l1BTileK * l1BTileN * sizeof(ElementB) * i);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    /// Destructor
    __aicore__ inline ~BlockMmadTla() {}

    template <class TensorA>
    __aicore__ inline void loadQGM(TensorA &gATensor, GemmCoord actualOriShape)
    {
        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorA>;
        CopyGmToL1A copyGmToL1A;
        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, embed);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[0], l1ALayoutTla, Arch::PositionL1{});
        auto l1ATensorTlaTile = GetTile(l1ATensorTla, tla::MakeCoord(0, 0), tla::MakeShape(rowNum, embed));
        auto gATensorTlaTile = GetTile(gATensor, tla::MakeCoord(0, 0), tla::MakeShape(rowNum, embed));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        copyGmToL1A(l1ATensorTlaTile, gATensorTlaTile);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    __aicore__ inline uint32_t GetCurLoopCounter(uint32_t outterLoopItr, uint32_t curLoopNum, uint32_t curLoopItr)
    {
        return outterLoopItr * curLoopNum + curLoopItr;
    }

    template <class TensorB, class TensorL1B>
    __aicore__ inline void SparseKL1TileNLoad(TensorB &gBTensor, TensorL1B &l1BTensorTla,
                                             AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                                             uint32_t gatheredKvSTileIdx, uint32_t kvSBaseTile, uint32_t blockShapeY,
                                             uint32_t l1BTileNAct, uint32_t l1BTileKAct,
                                             uint32_t kvSBaseTileInnerOffset, uint32_t embedInnerOffset,
                                             AscendC::GlobalTensor<int32_t> gBlockTable, uint32_t blockSize,
                                             uint64_t pageStride, uint64_t tokenStride,
                                             uint32_t localExplicitBlockCount, uint32_t residualBlockId,
                                             AscendC::GlobalTensor<uint64_t> kvOffsetInfo, uint32_t kvOffsetGroupWords)
    {
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        CopyGmToL1B copyGmToL1B;
        if constexpr (PRECOMPUTE_SPARSE) {
            constexpr uint32_t recordWords = PAGED_CACHE_FLAG ? 3U : 2U;
            auto groupInfo = kvOffsetInfo[kvSBaseTileInnerOffset / l1BTileN * kvOffsetGroupWords];
            // Refresh once after Ready; the remaining D slices and PV reuse this table.
            if (embedInnerOffset == 0U) {
                AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                    AscendC::DcciDst::CACHELINE_OUT>(groupInfo);
            }
            const uint32_t blockCount = groupInfo.GetValue(0);
            if (embedInnerOffset == 0U) {
                for (uint32_t word = 8U; word < 8U + blockCount * recordWords; word += 8U) {
                    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                        AscendC::DcciDst::CACHELINE_OUT>(groupInfo[word]);
                }
            }
            uint32_t dealtLenAccum = 0;
            for (uint32_t i = 0; i < blockCount; ++i) {
                const uint32_t word = 8U + i * recordWords;
                const uint64_t sourceOffset = groupInfo.GetValue(word);
                const uint64_t lengthOrMeta = groupInfo.GetValue(word + (PAGED_CACHE_FLAG ? 2U : 1U));
                const uint32_t curDealtLen = PAGED_CACHE_FLAG ? static_cast<uint32_t>(lengthOrMeta) :
                    static_cast<uint32_t>(lengthOrMeta) >> 1U;
                uint32_t ndNum = 1;
                uint32_t sourceStride = 0;
                if constexpr (!PAGED_CACHE_FLAG) {
                    ndNum = 1U + static_cast<uint32_t>(lengthOrMeta & 1U);
                    sourceStride = static_cast<uint32_t>(lengthOrMeta >> 32U);
                }
                auto l1BTensorTlaTile = GetTile(l1BTensorTla, tla::MakeCoord(0, dealtLenAccum),
                                               tla::MakeShape(l1BTileKAct, curDealtLen));
                auto gBSourceTensor = tla::MakeTensor(gBTensor.data()[sourceOffset + embedInnerOffset], gBTensor.layout(),
                                                     Arch::PositionGM{});
                auto gBTensorTlaTile = GetTile(gBSourceTensor, tla::MakeCoord(0, 0),
                                              tla::MakeShape(l1BTileKAct, curDealtLen));
                copyGmToL1B(l1BTensorTlaTile, gBTensorTlaTile, ndNum, sourceStride,
                             curDealtLen * tla::get<1, 0>(l1BTensorTla.stride()));
                dealtLenAccum += ndNum * curDealtLen;
            }
            return;
        }
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile + kvSBaseTileInnerOffset;
        uint32_t yBlockInnerStartOffset = baseTileStartOffset % blockShapeY;
        uint32_t dealtLenAccum = 0;
        // Tile length has already been clipped to validKvTokens. Thus every
        // sparse index and token visited here is valid, including the last block.
        for (uint32_t gatheredYBlockIdx = baseTileStartOffset / blockShapeY;
             dealtLenAccum < l1BTileNAct; ++gatheredYBlockIdx) {
            const uint32_t curYBlockSize = min(blockShapeY - yBlockInnerStartOffset,
                                               l1BTileNAct - dealtLenAccum);
            const int32_t logicalBlockId = GbsaKernelArch35::GetSparseBlockId<HAS_RESIDUAL_BLOCK>(
                gSparseBlockIdx, gatheredYBlockIdx, localExplicitBlockCount, residualBlockId);
            const uint64_t oriStartOffset =
                static_cast<uint64_t>(logicalBlockId) * blockShapeY + yBlockInnerStartOffset;
            // Only PA can split one sparse block across several physical pages.
            for (uint32_t copied = 0; copied < curYBlockSize;) {
                uint32_t curDealtLen = curYBlockSize - copied;
                uint64_t sourceToken = oriStartOffset + copied;
                uint64_t pageOffset = 0;
                if constexpr (PAGED_CACHE_FLAG) {
                    const uint32_t pageInnerOffset = sourceToken % blockSize;
                    curDealtLen = min(curDealtLen, blockSize - pageInnerOffset);
                    const uint32_t physicalPage = gBlockTable.GetValue(sourceToken / blockSize);
                    pageOffset = static_cast<uint64_t>(physicalPage) * pageStride;
                    sourceToken = pageInnerOffset;
                }
                auto l1BTensorTlaTile =
                    GetTile(l1BTensorTla, tla::MakeCoord(0, dealtLenAccum + copied),
                            tla::MakeShape(l1BTileKAct, curDealtLen));
                // GM is already rebased to the batch/head or physical block, with zero tensor coordinates.
                const uint64_t sourceOffset = pageOffset + sourceToken * tokenStride + embedInnerOffset;
                auto gBSourceTensor = tla::MakeTensor(gBTensor.data()[sourceOffset], gBTensor.layout(),
                                                     Arch::PositionGM{});
                auto gBTensorTlaTile =
                    GetTile(gBSourceTensor, tla::MakeCoord(0, 0), tla::MakeShape(l1BTileKAct, curDealtLen));
                copyGmToL1B(l1BTensorTlaTile, gBTensorTlaTile);
                copied += curDealtLen;
            }
            dealtLenAccum += curYBlockSize;
            yBlockInnerStartOffset = 0;
        }
    }

    template <class TensorB, class TensorC>
    __aicore__ inline void operator()(TensorB &gBTensor, TensorC &ubCTensor,
                                      AscendC::GlobalTensor<int32_t> gSparseBlockIdx, GemmCoord actualOriShape,
                                      uint32_t gatheredKvSTileIdx, uint32_t kvSBaseTile, uint32_t blockShapeY,
                                      uint64_t prefixSumL0AStages, uint64_t prefixSumL0BStages,
                                      Arch::CrossCoreFlag mm1ToSmFlag, uint64_t tokenStride,
                                      AscendC::GlobalTensor<int32_t> gBlockTable = {},
                                      uint32_t blockSize = 0, uint64_t pageStride = 0,
                                      uint32_t localExplicitBlockCount = 0, uint32_t residualBlockId = 0,
                                      AscendC::GlobalTensor<uint64_t> kvOffsetInfo = {}, uint32_t kvOffsetGroupWords = 0)
    {
        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[2];
        uint32_t curBaseTileSize = actualOriShape[1];

        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, embed);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[0], l1ALayoutTla, Arch::PositionL1{});

        uint32_t nL1LoopNum = CeilDiv(curBaseTileSize, l1BTileN);
        uint32_t kL1LoopNum = CeilDiv(embed, l1BTileK);
        uint32_t mL0LoopNum = CeilDiv(rowNum, L0_TILE_M);
        uint32_t totalNL0LoopNum = CeilDiv(kvSBaseTile, L0_TILE_N);
        uint32_t totalKL0LoopNum = CeilDiv(embed, L0_TILE_K);
        uint32_t fullNL0LoopNum = CeilDiv(l1BTileN, L0_TILE_N);
        uint32_t fullKL0LoopNum = CeilDiv(l1BTileK, L0_TILE_K);

        bool l1ToL0ANoRepeatFlag = (mL0LoopNum == 1) && (totalKL0LoopNum <= L0_STAGES);

        // while splitting the base tile S to 2 AIVs,
        // the order of the elements in each column is expected to be preserved,
        // which means a column in l0C cannot be chunked and processed by dualMode FixPipe seperately.
        // therefore, FixPipe won't launch until each portion(chunked only by columns, based on nbuffer strategy)
        // of the base tile is ready on l0C
        for (uint32_t nL1Itr = 0; nL1Itr < nL1LoopNum; nL1Itr++) {
            uint32_t l1TileNAct = (nL1Itr == nL1LoopNum - 1) ? (curBaseTileSize - nL1Itr * l1BTileN) : l1BTileN;
            uint32_t nL0LoopNum = CeilDiv(l1TileNAct, L0_TILE_N);
            for (uint32_t kL1Itr = 0; kL1Itr < kL1LoopNum; ++kL1Itr) {
                uint32_t l1TileKAct =
                    (kL1Itr == kL1LoopNum - 1) ? (embed - kL1Itr * l1BTileK) : l1BTileK;
                uint32_t kLoopCounterL1 =
                    (gatheredKvSTileIdx * nL1LoopNum + nL1Itr) * kL1LoopNum + kL1Itr;
                uint32_t l1BBufId = kLoopCounterL1 % l1BBufNum;
                uint32_t l1BEventId = l1BBufId + 1;
                auto l1BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL1B>(l1TileKAct, l1TileNAct);
                auto l1BTensorTla = tla::MakeTensor(l1BTensor[l1BBufId], l1BLayoutTla, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
                SparseKL1TileNLoad(
                    gBTensor, l1BTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSBaseTile, blockShapeY,
                    l1TileNAct, l1TileKAct, nL1Itr * l1BTileN, kL1Itr * l1BTileK, gBlockTable, blockSize,
                    pageStride, tokenStride, localExplicitBlockCount, residualBlockId, kvOffsetInfo, kvOffsetGroupWords);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);

                uint32_t kL0LoopNum = CeilDiv(l1TileKAct, L0_TILE_K);
                for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
                    uint32_t l0TileNAct =
                        (nL0Itr == nL0LoopNum - 1) ? (l1TileNAct - nL0Itr * L0_TILE_N) : L0_TILE_N;
                    uint32_t globalNL0Itr = nL1Itr * fullNL0LoopNum + nL0Itr;
                    uint32_t l0CLoopCounter =
                        gatheredKvSTileIdx * totalNL0LoopNum + globalNL0Itr;
                    uint32_t l0CBufId = l0CLoopCounter % L0_STAGES;
                    uint32_t l0CEventId = l0CBufId;
                    auto l0CLayoutTla = tla::MakeLayoutL0C(rowNum, l0TileNAct);
                    auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CBufId], l0CLayoutTla, Arch::PositionL0C{});
                    for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                        uint32_t l0TileMAct =
                            (mL0Itr == mL0LoopNum - 1) ? (rowNum - mL0Itr * L0_TILE_M) : L0_TILE_M;
                        auto l0CTensorTlaTile = GetTile(l0CTensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, 0),
                                                        tla::MakeShape(l0TileMAct, l0TileNAct));
                        for (uint32_t kL0Itr = 0; kL0Itr < kL0LoopNum; kL0Itr++) {
                            uint32_t l0TileKAct = (kL0Itr == kL0LoopNum - 1) ?
                                                     (l1TileKAct - kL0Itr * L0_TILE_K) : L0_TILE_K;
                            uint32_t globalKL0Itr =
                                kL1Itr * fullKL0LoopNum + kL0Itr;
                            uint32_t l0ALoopCounter =
                                prefixSumL0AStages + mL0Itr * totalKL0LoopNum + globalKL0Itr;
                            uint32_t l0BLoopCounter = prefixSumL0BStages +
                                globalNL0Itr * totalKL0LoopNum + globalKL0Itr;
                            uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                            uint32_t l0BBufId = l0BLoopCounter % L0_STAGES;
                            uint32_t l0AEventId = l0ABufId;
                            uint32_t l0BEventId = l0BBufId + 2;

                            auto l1BTensorTlaTile =
                                GetTile(l1BTensorTla, tla::MakeCoord(kL0Itr * L0_TILE_K, nL0Itr * L0_TILE_N),
                                        tla::MakeShape(l0TileKAct, l0TileNAct));
                            auto l0BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL0B>(l0TileKAct, l0TileNAct);
                            auto l0BTensorTla =
                                tla::MakeTensor(l0BTensor[l0BBufId], l0BLayoutTla, Arch::PositionL0B{});
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                            copyL1ToL0B(l0BTensorTla, l1BTensorTlaTile);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                            if (nL0Itr == nL0LoopNum - 1U && mL0Itr == mL0LoopNum - 1U &&
                                kL0Itr == kL0LoopNum - 1U) {
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
                            }

                            auto l1ATensorTlaTile = GetTile(
                                l1ATensorTla,
                                tla::MakeCoord(mL0Itr * L0_TILE_M, kL1Itr * l1BTileK + kL0Itr * L0_TILE_K),
                                tla::MakeShape(l0TileMAct, l0TileKAct));
                            auto l0ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL0A>(l0TileMAct, l0TileKAct);
                            auto l0ATensorTla =
                                tla::MakeTensor(l0ATensor[l0ABufId], l0ALayoutTla, Arch::PositionL0A{});
                            const bool firstN = (nL1Itr == 0U) && (nL0Itr == 0U);
                            const bool lastN = (nL1Itr == nL1LoopNum - 1U) && (nL0Itr == nL0LoopNum - 1U);
                            if (!l1ToL0ANoRepeatFlag || firstN) {
                                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                                copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                            }

                            if (!l1ToL0ANoRepeatFlag || firstN) {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                            }
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                            if (mL0Itr == 0U && kL1Itr == 0U && kL0Itr == 0U) {
                                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                            }
                            bool initMmad = (kL1Itr == 0U) && (kL0Itr == 0U);
                            uint32_t l0TileMAligned = RoundUp(l0TileMAct, 16);
                            tileMmad(l0CTensorTlaTile, l0ATensorTla, l0BTensorTla, l0TileMAligned, l0TileNAct,
                                     l0TileKAct, initMmad);
                            if (!l1ToL0ANoRepeatFlag || lastN) {
                                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                        }
                    }

                    if (kL1Itr == kL1LoopNum - 1U) {
                        // All L1 N tiles share one S buffer and one release token.
                        // Acquire it only before the first Fixpipe of the whole KV tile.
                        if (globalNL0Itr == 0U) {
                            WaitCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
                        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
                        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
                        if constexpr (AscendC::IsSameType<ElementC, float>::value) {
                            uint32_t nFixPAligned8 = RoundUp(l0TileNAct, 8);
                            auto ubCTensorTlaTile =
                                GetTile(ubCTensor, tla::MakeCoord(0, globalNL0Itr * L0_TILE_N),
                                        tla::MakeShape(rowNum, nFixPAligned8));
                            CopyL0CToDst copyL0CToDst;
                            copyL0CToDst(ubCTensorTlaTile, l0CTensorTla);
                        } else {
                            uint32_t mPerSubCore = RoundUp(rowNum, 2) / 2;
                            uint32_t nFixPAligned16 = RoundUp(l0TileNAct, 16);
                            auto ubCTensorTlaTile =
                                GetTile(ubCTensor, tla::MakeCoord(0, globalNL0Itr * L0_TILE_N),
                                        tla::MakeShape(mPerSubCore, nFixPAligned16));
                            auto l0CTensorTlaTileSub0 =
                                GetTile(l0CTensorTla, tla::MakeCoord(0, 0),
                                        tla::MakeShape(mPerSubCore, l0TileNAct));
                            auto l0CTensorTlaTileSub1 =
                                GetTile(l0CTensorTla, tla::MakeCoord(mPerSubCore, 0),
                                        tla::MakeShape(mPerSubCore, l0TileNAct));
                            CopyL0CToDst copyL0CToDstSub0;
                            CopyL0CToDst copyL0CToDstSub1;
                            copyL0CToDstSub0(ubCTensorTlaTile, l0CTensorTlaTileSub0, false);
                            copyL0CToDstSub1(ubCTensorTlaTile, l0CTensorTlaTileSub1, true);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                    }
                }
            }
        }
        // crossCoreSync after all fixPipe move
        SetCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[L0_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[L0_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0_STAGES];

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t l1ATileM;
    uint32_t l1BTileN;
    uint32_t l1ATileK;
    uint32_t l1BTileK;
    uint32_t l1ABufNum;
    uint32_t l1BBufNum;

    uint32_t l1PPingPongFlag = 0;
    uint32_t l0CPingPongFlag = 0;
    uint32_t l0ABPingPongFlag = 0;
};
////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Block
#endif
