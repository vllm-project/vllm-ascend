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
 * @brief matmul implementation for single p&v base tile
 * This implementation is designed for the following senario:
 * A full p base tile is loaded to L1 from UB, no workspace transit
 * A full v base tile is loaded to L1 from GM，relavent instructions launched before p base tile crossCore wait
 * A full p*v base tile is stored from L0C to UB, or to GM for row-streamed rescaleO.
 */
#ifndef GEMM_BLOCK_PV_ARCH35_ABF16_C2UB_HPP
#define GEMM_BLOCK_PV_ARCH35_ABF16_C2UB_HPP

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

struct Mm2L1TileHelper {
    uint32_t mm2L1TileM;
    uint32_t mm2L1TileN;
    uint32_t mm2L1TileKLeft;
    uint32_t mm2L1TileKRight;
    uint32_t pL1BufNum;
    uint32_t vL1BufNum;

    __aicore__ inline Mm2L1TileHelper() {}

    __aicore__ inline Mm2L1TileHelper(uint32_t m, uint32_t n, uint32_t kl, uint32_t kr, uint32_t pbn, uint32_t vbn)
        : mm2L1TileM(m),
          mm2L1TileN(n),
          mm2L1TileKLeft(kl),
          mm2L1TileKRight(kr),
          pL1BufNum(pbn),
          vL1BufNum(vbn)
    {}
};

template <bool PAGED_CACHE_FLAG_, bool HAS_RESIDUAL_BLOCK_, bool PRECOMPUTE_SPARSE_, class L1TileShape_, class L0TileShape_,
          class ElementA_, class ElementB_,
          class ElementC_, class ElementBias_, class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadAtlasA5BsaPV<PAGED_CACHE_FLAG_, HAS_RESIDUAL_BLOCK_, PRECOMPUTE_SPARSE_>, L1TileShape_, L0TileShape_,
                    ElementA_, ElementB_,
                    ElementC_, ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA5BsaPV<PAGED_CACHE_FLAG_, HAS_RESIDUAL_BLOCK_, PRECOMPUTE_SPARSE_>;
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

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart,
                                   Mm2L1TileHelper &mm2L1TileHelper)
    {
        l1ATileM = mm2L1TileHelper.mm2L1TileM;
        l1BTileN = mm2L1TileHelper.mm2L1TileN;
        l1ATileK = mm2L1TileHelper.mm2L1TileKLeft;
        l1BTileK = mm2L1TileHelper.mm2L1TileKRight;
        l1ABufNum = mm2L1TileHelper.pL1BufNum;
        l1BBufNum = mm2L1TileHelper.vL1BufNum;
        for (uint32_t i = 0; i < l1ABufNum; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + l1ATileM * l1ATileK *
                                                                                                  sizeof(ElementA) * i);
        }
        for (uint32_t i = 0; i < l1BBufNum; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BufAddrStart + l1ATileM * l1ATileK * sizeof(ElementA) * l1ABufNum +
                l1BTileK * l1BTileN * sizeof(ElementB) * i);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_HALF_BUF_SIZE +
                                                                                        L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    /// Destructor
    __aicore__ inline ~BlockMmadTla() {}

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
    __aicore__ inline void SparseVBaseTileL1FullLoad(TensorB &gBTensor, TensorL1B &l1BTensorTla,
                                                     AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                                                     uint32_t gatheredKvSTileIdx, uint32_t kvSBaseTile,
                                                     uint32_t blockShapeY, uint32_t curBaseTileSize, uint32_t embed,
                                                     AscendC::GlobalTensor<int32_t> gBlockTable, uint32_t blockSize,
                                                     uint64_t pageStride, uint64_t tokenStride,
                                                     uint32_t localExplicitBlockCount, uint32_t residualBlockId,
                                                     AscendC::GlobalTensor<uint64_t> kvOffsetInfo,
                                                     uint32_t kvOffsetGroupWords, uint32_t kvOffsetGroupN)
    {
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        CopyGmToL1B copyGmToL1B;
        if constexpr (PRECOMPUTE_SPARSE) {
            constexpr uint32_t recordWords = PAGED_CACHE_FLAG ? 3U : 2U;
            uint32_t dealtLenAccum = 0;
            const uint32_t groupNum = CeilDiv(curBaseTileSize, kvOffsetGroupN);
            for (uint32_t groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
                auto groupInfo = kvOffsetInfo[groupIdx * kvOffsetGroupWords];
                const uint32_t blockCount = groupInfo.GetValue(0);
                for (uint32_t i = 0; i < blockCount; ++i) {
                    const uint32_t word = 8U + i * recordWords;
                    const uint64_t sourceOffset = groupInfo.GetValue(word + (PAGED_CACHE_FLAG ? 1U : 0U));
                    const uint64_t lengthOrMeta = groupInfo.GetValue(word + (PAGED_CACHE_FLAG ? 2U : 1U));
                    const uint32_t curDealtLen = PAGED_CACHE_FLAG ? static_cast<uint32_t>(lengthOrMeta) :
                        static_cast<uint32_t>(lengthOrMeta) >> 1U;
                    uint32_t ndNum = 1;
                    uint32_t sourceStride = 0;
                    if constexpr (!PAGED_CACHE_FLAG) {
                        ndNum = 1U + static_cast<uint32_t>(lengthOrMeta & 1U);
                        sourceStride = static_cast<uint32_t>(lengthOrMeta >> 32U);
                    }
                    auto l1BTensorTlaTile = GetTile(l1BTensorTla, tla::MakeCoord(dealtLenAccum, 0),
                                                   tla::MakeShape(curDealtLen, embed));
                    auto gBSourceTensor = tla::MakeTensor(gBTensor.data()[sourceOffset], gBTensor.layout(),
                                                         Arch::PositionGM{});
                    auto gBTensorTlaTile = GetTile(gBSourceTensor, tla::MakeCoord(0, 0),
                                                  tla::MakeShape(curDealtLen, embed));
                    copyGmToL1B(l1BTensorTlaTile, gBTensorTlaTile, ndNum, sourceStride,
                                 curDealtLen * tla::get<0, 0>(l1BTensorTla.stride()));
                    dealtLenAccum += ndNum * curDealtLen;
                }
            }
            return;
        }
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t yBlockInnerStartOffset = baseTileStartOffset % blockShapeY;
        uint32_t dealtLenAccum = 0;
        // The caller clips this tile to validKvTokens, so every visited token is valid.
        for (uint32_t gatheredYBlockIdx = baseTileStartOffset / blockShapeY;
             dealtLenAccum < curBaseTileSize; ++gatheredYBlockIdx) {
            const uint32_t curYBlockSize = min(blockShapeY - yBlockInnerStartOffset,
                                               curBaseTileSize - dealtLenAccum);
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
                    GetTile(l1BTensorTla, tla::MakeCoord(dealtLenAccum + copied, 0),
                            tla::MakeShape(curDealtLen, embed));
                // GM is already rebased to the batch/head or physical block, with zero tensor coordinates.
                const uint64_t sourceOffset = pageOffset + sourceToken * tokenStride;
                auto gBSourceTensor = tla::MakeTensor(gBTensor.data()[sourceOffset], gBTensor.layout(),
                                                     Arch::PositionGM{});
                auto gBTensorTlaTile =
                    GetTile(gBSourceTensor, tla::MakeCoord(0, 0), tla::MakeShape(curDealtLen, embed));
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
                                      Arch::CrossCoreFlag smToMm2Flag, Arch::CrossCoreFlag mm2ToReFlag,
                                      uint64_t tokenStride,
                                      AscendC::GlobalTensor<int32_t> gBlockTable = {},
                                      uint32_t blockSize = 0, uint64_t pageStride = 0,
                                      uint32_t localExplicitBlockCount = 0, uint32_t residualBlockId = 0,
                                      AscendC::GlobalTensor<uint64_t> kvOffsetInfo = {},
                                      uint32_t kvOffsetGroupWords = 0, uint32_t kvOffsetGroupN = 0)
    {
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;

        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        uint32_t curBaseTileSize = actualOriShape[2];

        uint32_t l1BBufId = gatheredKvSTileIdx % l1BBufNum;
        uint32_t l1BEventId = l1BBufId + 3;
        uint32_t l1ABufId = gatheredKvSTileIdx % l1ABufNum;

        auto l1BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL1B>(curBaseTileSize, embed);
        auto l1BTensorTla = tla::MakeTensor(l1BTensor[l1BBufId], l1BLayoutTla, Arch::PositionL1{});
        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, curBaseTileSize);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[l1ABufId], l1ALayoutTla, Arch::PositionL1{});
        // load V full base tile to L1 before crossCoreSync
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
        SparseVBaseTileL1FullLoad(
            gBTensor, l1BTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSBaseTile, blockShapeY,
            curBaseTileSize, embed, gBlockTable, blockSize, pageStride, tokenStride,
            localExplicitBlockCount, residualBlockId, kvOffsetInfo, kvOffsetGroupWords, kvOffsetGroupN);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
        // fwd crossCoreSync from online sm to mm2
        WaitCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
        // P full base tile already on L1
        uint32_t mL0LoopNum = CeilDiv(rowNum, L0_TILE_M);
        uint32_t nL0LoopNum = CeilDiv(embed, L0_TILE_N);
        uint32_t kL0LoopNum = CeilDiv(curBaseTileSize, L0_TILE_K);
        // while splitting the base tile OTmp to 2 AIVs,
        // the order of the elements in each column is expected to be preserved,
        // which means a column in l0C cannot be chunked and processed by dualMode FixPipe seperately.
        // therefore, FixPipe won't launch until each portion(chunked only by columns, based on nbuffer strategy)
        // of the base tile is ready on l0C
        for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
            uint32_t l0TileNAct = (nL0Itr == nL0LoopNum - 1) ? (embed - nL0Itr * L0_TILE_N) : L0_TILE_N;
            uint32_t nLoopCounter = GetCurLoopCounter(gatheredKvSTileIdx, nL0LoopNum, nL0Itr);
            // l0C nbuffer chunked only in n loop
            uint32_t l0CLoopCounter = nLoopCounter;
            uint32_t l0CBufId = l0CLoopCounter % L0_STAGES;
            // uint32_t l0CEventId = l0CBufId;
            uint32_t l0CEventId = l0CBufId + 2;
            auto l0CLayoutTla = tla::MakeLayoutL0C(rowNum, l0TileNAct);
            auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CBufId], l0CLayoutTla, Arch::PositionL0C{});
            for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (rowNum - mL0Itr * L0_TILE_M) : L0_TILE_M;
                // uint32_t mLoopCounter = GetCurLoopCounter(gatheredKvSTileIdx, mL0LoopNum, mL0Itr);
                // different m chunks will be concated in the same piece of l0C buffer
                auto l0CTensorTlaTile = GetTile(l0CTensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, 0),
                                                tla::MakeShape(l0TileMAct, l0TileNAct));
                for (uint32_t kL0Itr = 0; kL0Itr < kL0LoopNum; kL0Itr++) {
                    uint32_t l0ALoopCounter = prefixSumL0AStages + GetCurLoopCounter(mL0Itr, kL0LoopNum, kL0Itr);
                    uint32_t l0BLoopCounter = prefixSumL0BStages + GetCurLoopCounter(nL0Itr, kL0LoopNum, kL0Itr);
                    uint32_t l0TileKAct =
                        (kL0Itr == kL0LoopNum - 1) ? (curBaseTileSize - kL0Itr * L0_TILE_K) : L0_TILE_K;
                    uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                    // l0ABufId = (mLoopCounter % 2 == 0) ? (1 - l0ABufId) : l0ABufId;
                    uint32_t l0BBufId = l0BLoopCounter % L0_STAGES;
                    uint32_t l0AEventId = l0ABufId;
                    uint32_t l0BEventId = l0BBufId + 2;
                    // when L0B buffers wouldn't be reused across the k loop
                    // redundant L0B load caused by m loop can be avoided
                    auto l1BTensorTlaTile =
                        GetTile(l1BTensorTla, tla::MakeCoord(kL0Itr * L0_TILE_K, nL0Itr * L0_TILE_N),
                                tla::MakeShape(l0TileKAct, l0TileNAct));
                    auto l0BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL0B>(l0TileKAct, l0TileNAct);
                    auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayoutTla, Arch::PositionL0B{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    copyL1ToL0B(l0BTensorTla, l1BTensorTlaTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

                    if ((mL0Itr == mL0LoopNum - 1) && (nL0Itr == nL0LoopNum - 1) && (kL0Itr == kL0LoopNum - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
                    }

                    auto l1ATensorTlaTile =
                        GetTile(l1ATensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K),
                                tla::MakeShape(l0TileMAct, l0TileKAct));
                    auto l0ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL0A>(l0TileMAct, l0TileKAct);
                    auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayoutTla, Arch::PositionL0A{});

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    // reverse crossCoreSync for P
                    if ((mL0Itr == mL0LoopNum - 1) && (nL0Itr == nL0LoopNum - 1) && (kL0Itr == kL0LoopNum - 1)) {
                        SetCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
                    }

                    bool initMmad = (kL0Itr == 0);
                    uint32_t l0TileMAligned = RoundUp(l0TileMAct, 16);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

                    if (mL0Itr == 0 && kL0Itr == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                    }
                    tileMmad(l0CTensorTlaTile, l0ATensorTla, l0BTensorTla, l0TileMAligned, l0TileNAct, l0TileKAct,
                             initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                }
            }
            // fixpipe
            if (nL0Itr == 0) {
                // Reverse crossCoreSync: wait until the destination UB/GM tile is fully released.
                WaitCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            // UB SPLIT_M rounds M to an even value; GM retains the full M. N is aligned to 32B.
            uint32_t nFixPAligned8 = RoundUp(l0TileNAct, 8);
            auto ubCTensorTlaTile =
                GetTile(ubCTensor, tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(rowNum, nFixPAligned8));
            copyL0CToDst(ubCTensorTlaTile, l0CTensorTla);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }
        // crossCoreSync after all fixPipe move
        SetCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
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
};
////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Block
#endif
