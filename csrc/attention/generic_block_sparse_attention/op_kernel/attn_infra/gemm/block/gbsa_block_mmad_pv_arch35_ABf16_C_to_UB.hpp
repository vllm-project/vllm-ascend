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
 * This implementation is designed for the following scenario:
 * A full p base tile is loaded to L1 from UB, no workspace transit
 * V keeps a full base tile in L1; non-quantized groups can be consumed while later groups are loading.
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

template <bool PAGED_CACHE_FLAG_, bool HAS_RESIDUAL_BLOCK_, bool PRECOMPUTE_SPARSE_, class L1TileShape_,
          class L0TileShape_, class ElementA_, class ElementB_, class ElementC_, class ElementBias_, class TileCopy_,
          class TileMmad_>
struct BlockMmadTla<MmadAtlasA5BsaPV<PAGED_CACHE_FLAG_, HAS_RESIDUAL_BLOCK_, PRECOMPUTE_SPARSE_>, L1TileShape_,
                    L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_, TileMmad_> {
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

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart,
                                   Mm2L1TileHelper& mm2L1TileHelper)
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
    __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag& crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs separately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag& crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs separately
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
    __aicore__ inline void SparseVGroupL1Load(TensorB& gBTensor, TensorL1B& l1BTensorTla,
                                              AscendC::GlobalTensor<uint64_t> groupInfo, uint32_t rowOffset,
                                              uint32_t embed)
    {
        constexpr uint32_t recordWords = PAGED_CACHE_FLAG ? 4U : 2U;
        constexpr uint32_t elementsPerC0 = BYTE_PER_C0 / sizeof(ElementB);
        const auto& srcData = gBTensor.data();
        const auto& dstStorage = l1BTensorTla.data();
        // Rebind the existing L1 view with an explicit logical position.
        // Subtract the L1 base so the address is relative in CPU debug as well.
        const AscendC::LocalTensor<ElementB> l1Base(AscendC::TPosition::A1, 0, ArchTag::L1_SIZE / sizeof(ElementB));
        const uint32_t dstAddr =
            static_cast<uint32_t>((uint64_t)dstStorage.GetPhyAddr() - (uint64_t)l1Base.GetPhyAddr());
        const AscendC::LocalTensor<ElementB> dstData(AscendC::TPosition::A1, dstAddr, dstStorage.GetSize());
        const uint32_t dstRowStride = tla::get<0, 0>(l1BTensorTla.stride());
        const auto dstBase = l1BTensorTla.layout()(l1BTensorTla.coord());
        AscendC::Nd2NzParams copyParams;
        copyParams.dValue = embed;
        copyParams.srcDValue = tla::get<0>(gBTensor.stride());
        copyParams.dstNzC0Stride = tla::get<1, 1>(l1BTensorTla.stride()) / elementsPerC0;
        copyParams.dstNzNStride = dstRowStride / elementsPerC0;
        const __gm__ uint64_t* records = groupInfo.GetPhyAddr();
        const uint32_t blockCount = records[0];
        for (uint32_t i = 0; i < blockCount; ++i) {
            const uint32_t word = 8U + i * recordWords;
            const uint64_t sourceOffset = records[word + (PAGED_CACHE_FLAG ? 1U : 0U)];
            const uint64_t lengthOrMeta = records[word + (PAGED_CACHE_FLAG ? 3U : 1U)];
            const uint32_t curDealtLen = static_cast<uint32_t>(lengthOrMeta) >> 1U;
            const uint32_t ndNum = 1U + static_cast<uint32_t>(lengthOrMeta & 1U);
            copyParams.ndNum = ndNum;
            copyParams.nValue = curDealtLen;
            copyParams.srcNdMatrixStride = static_cast<uint32_t>(lengthOrMeta >> 32U);
            copyParams.dstNzMatrixStride = curDealtLen * dstRowStride;
            AscendC::DataCopy(dstData[dstBase + rowOffset * dstRowStride], srcData[sourceOffset], copyParams);
            rowOffset += ndNum * curDealtLen;
        }
    }

    template <class TensorB, class TensorL1B>
    __aicore__ inline void SparseVBaseTileL1FullLoad(
        TensorB& gBTensor, TensorL1B& l1BTensorTla, AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
        uint32_t gatheredKvSTileIdx, uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t curBaseTileSize,
        uint32_t embed, AscendC::GlobalTensor<int32_t> gBlockTable, uint32_t blockSize, uint64_t pageStride,
        uint64_t tokenStride, uint32_t localExplicitBlockCount, uint32_t residualBlockId,
        AscendC::GlobalTensor<uint64_t> kvOffsetInfo, uint32_t kvOffsetGroupWords, uint32_t kvOffsetGroupN)
    {
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        CopyGmToL1B copyGmToL1B;
        if constexpr (PRECOMPUTE_SPARSE) {
            const uint32_t groupNum = CeilDiv(curBaseTileSize, kvOffsetGroupN);
            for (uint32_t groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
                SparseVGroupL1Load(gBTensor, l1BTensorTla, kvOffsetInfo[groupIdx * kvOffsetGroupWords],
                                   groupIdx * kvOffsetGroupN, embed);
            }
            return;
        }
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t yBlockInnerStartOffset = baseTileStartOffset % blockShapeY;
        uint32_t dealtLenAccum = 0;
        // The caller clips this tile to validKvTokens, so every visited token is valid.
        for (uint32_t gatheredYBlockIdx = baseTileStartOffset / blockShapeY; dealtLenAccum < curBaseTileSize;
             ++gatheredYBlockIdx) {
            const uint32_t curYBlockSize = min(blockShapeY - yBlockInnerStartOffset, curBaseTileSize - dealtLenAccum);
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
                auto l1BTensorTlaTile = GetTile(l1BTensorTla, tla::MakeCoord(dealtLenAccum + copied, 0),
                                                tla::MakeShape(curDealtLen, embed));
                // GM is already rebased to the batch/head or physical block, with zero tensor coordinates.
                const uint64_t sourceOffset = pageOffset + sourceToken * tokenStride;
                auto gBSourceTensor =
                    tla::MakeTensor(gBTensor.data()[sourceOffset], gBTensor.layout(), Arch::PositionGM{});
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
    __aicore__ inline void operator()(
        TensorB& gBTensor, TensorC& ubCTensor, AscendC::GlobalTensor<int32_t> gSparseBlockIdx, GemmCoord actualOriShape,
        uint32_t gatheredKvSTileIdx, uint32_t kvSBaseTile, uint32_t blockShapeY, uint64_t prefixSumL0AStages,
        uint64_t prefixSumL0BStages, Arch::CrossCoreFlag smToMm2Flag, Arch::CrossCoreFlag mm2ToReFlag,
        uint64_t tokenStride, AscendC::GlobalTensor<int32_t> gBlockTable = {}, uint32_t blockSize = 0,
        uint64_t pageStride = 0, uint32_t localExplicitBlockCount = 0, uint32_t residualBlockId = 0,
        AscendC::GlobalTensor<uint64_t> kvOffsetInfo = {}, uint32_t kvOffsetGroupWords = 0, uint32_t kvOffsetGroupN = 0)
    {
        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        uint32_t curBaseTileSize = actualOriShape[2];

        // Non-quantized tiling allocates one or two V buffers; P may use three.
        uint32_t l1BBufId =
            PRECOMPUTE_SPARSE ? (gatheredKvSTileIdx & (l1BBufNum - 1U)) : (gatheredKvSTileIdx % l1BBufNum);
        uint32_t l1BEventId = l1BBufId + 3;
        uint32_t l1ABufId = gatheredKvSTileIdx % l1ABufNum;

        auto l1BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL1B>(curBaseTileSize, embed);
        auto l1BTensorTla = tla::MakeTensor(l1BTensor[l1BBufId], l1BLayoutTla, Arch::PositionL1{});
        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, curBaseTileSize);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[l1ABufId], l1ALayoutTla, Arch::PositionL1{});
        // Retain the full V layout. Only readiness is split along the KV axis.
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
        const uint32_t vGroupNum = PRECOMPUTE_SPARSE ? CeilDiv(curBaseTileSize, kvOffsetGroupN) : 1U;
        const uint32_t vReadyBase =
            PRECOMPUTE_SPARSE ? EVENT_ID3 + l1BBufId * CeilDiv(l1BTileK, kvOffsetGroupN) : l1BEventId;
        if constexpr (PRECOMPUTE_SPARSE) {
            // Current tiling has at most four groups across the V buffers.
            // QK uses IDs 0..2; IDs 3..6 are reused only after this V buffer is released.
            for (uint32_t groupIdx = 0; groupIdx < vGroupNum; ++groupIdx) {
                SparseVGroupL1Load(gBTensor, l1BTensorTla, kvOffsetInfo[groupIdx * kvOffsetGroupWords],
                                   groupIdx * kvOffsetGroupN, embed);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(vReadyBase + groupIdx);
            }
            const uint32_t freeFlag =
                GbsaKernelArch35::KV_OFFSET_FLAG_BASE + gatheredKvSTileIdx % GbsaKernelArch35::KV_OFFSET_STAGES;
            AscendC::CrossCoreSetFlag<4, PIPE_S>(freeFlag);
            AscendC::CrossCoreSetFlag<4, PIPE_S>(freeFlag + V0_V1_FLAG_ID_OFFSET);
        } else {
            SparseVBaseTileL1FullLoad(gBTensor, l1BTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSBaseTile,
                                      blockShapeY, curBaseTileSize, embed, gBlockTable, blockSize, pageStride,
                                      tokenStride, localExplicitBlockCount, residualBlockId, kvOffsetInfo,
                                      kvOffsetGroupWords, kvOffsetGroupN);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
        }
        // fwd crossCoreSync from online sm to mm2
        WaitCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
        if constexpr (PRECOMPUTE_SPARSE) {
            // Select geometry once. The inner N loop then has one or two
            // compile-time iterations, with fixed ownership of both PV L0Cs.
            if (embed <= L0_TILE_N) {
                ComputePV<1U>(l1ATensorTla, l1BTensorTla, ubCTensor, actualOriShape, gatheredKvSTileIdx,
                              prefixSumL0AStages, prefixSumL0BStages, smToMm2Flag, mm2ToReFlag, l1BEventId, vReadyBase,
                              kvOffsetGroupN);
            } else {
                ComputePV<2U>(l1ATensorTla, l1BTensorTla, ubCTensor, actualOriShape, gatheredKvSTileIdx,
                              prefixSumL0AStages, prefixSumL0BStages, smToMm2Flag, mm2ToReFlag, l1BEventId, vReadyBase,
                              kvOffsetGroupN);
            }
        } else {
            ComputePV<0U>(l1ATensorTla, l1BTensorTla, ubCTensor, actualOriShape, gatheredKvSTileIdx, prefixSumL0AStages,
                          prefixSumL0BStages, smToMm2Flag, mm2ToReFlag, l1BEventId, vReadyBase, kvOffsetGroupN);
        }
    }

    template <uint32_t N_SPLITS, class TensorL1A, class TensorL1B, class TensorC>
    __aicore__ inline void ComputePV(TensorL1A& l1ATensorTla, TensorL1B& l1BTensorTla, TensorC& ubCTensor,
                                     GemmCoord actualOriShape, uint32_t gatheredKvSTileIdx, uint64_t prefixSumL0AStages,
                                     uint64_t prefixSumL0BStages, Arch::CrossCoreFlag smToMm2Flag,
                                     Arch::CrossCoreFlag mm2ToReFlag, uint32_t l1BEventId, uint32_t vReadyBase,
                                     uint32_t kvOffsetGroupN)
    {
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
        CopyL0CToDst copyL0CToDst;
        const uint32_t rowNum = actualOriShape[0];
        const uint32_t embed = actualOriShape[1];
        const uint32_t curBaseTileSize = actualOriShape[2];
        // P full base tile already on L1
        // The host bounds non-quant groupSize by L0_TILE_M (128).
        const uint32_t mL0LoopNum = PRECOMPUTE_SPARSE ? 1U : CeilDiv(rowNum, L0_TILE_M);
        const uint32_t nL0LoopNum = N_SPLITS != 0U ? N_SPLITS : CeilDiv(embed, L0_TILE_N);
        uint32_t kL0LoopNum = CeilDiv(curBaseTileSize, L0_TILE_K);
        // D <= 256: retain both D slices in the existing PV L0C buffers.
        // Their strides depend only on M; the actual N tail is selected below.
        const uint32_t l0TileNTail = embed - (nL0LoopNum - 1U) * L0_TILE_N;
        const uint32_t l0CBaseId = gatheredKvSTileIdx * nL0LoopNum;
        auto l0CLayoutTla = tla::MakeLayoutL0C(rowNum, L0_TILE_N);
        const uint32_t l0ABaseId = prefixSumL0AStages % L0_STAGES;
        const uint32_t l0BBaseId = prefixSumL0BStages % L0_STAGES;
        uint32_t vGroupIdx = 0;
        for (uint32_t kL0Itr = 0; kL0Itr < kL0LoopNum; kL0Itr++) {
            const uint32_t l0TileKAct =
                (kL0Itr == kL0LoopNum - 1U) ? (curBaseTileSize - kL0Itr * L0_TILE_K) : L0_TILE_K;
            if constexpr (PRECOMPUTE_SPARSE) {
                if (kL0Itr * L0_TILE_K == vGroupIdx * kvOffsetGroupN) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(vReadyBase + vGroupIdx);
                    ++vGroupIdx;
                }
            }
#pragma unroll
            for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
                const uint32_t l0TileNAct = nL0Itr == nL0LoopNum - 1U ? l0TileNTail : L0_TILE_N;
                const uint32_t l0CBufId = (l0CBaseId + nL0Itr) % L0_STAGES;
                const uint32_t l0CEventId = l0CBufId + 2;
                auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CBufId], l0CLayoutTla, Arch::PositionL0C{});
                for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                    uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (rowNum - mL0Itr * L0_TILE_M) : L0_TILE_M;
                    // Different M slices occupy disjoint rows in the same L0C buffer.
                    auto l0CTensorTlaTile = GetTile(l0CTensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, 0),
                                                    tla::MakeShape(l0TileMAct, l0TileNAct));
                    // Ping-pong follows MMAD issue order, not the old N-major order.
                    const uint32_t l0LoopCounter = (kL0Itr * nL0LoopNum + nL0Itr) * mL0LoopNum + mL0Itr;
                    // Both D slices use the same P tile. Advance L0A only along K.
                    uint32_t l0ALoopCounter = l0ABaseId + (N_SPLITS == 2U ? kL0Itr : l0LoopCounter);
                    uint32_t l0BLoopCounter = l0BBaseId + l0LoopCounter;
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

                    auto l0ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL0A>(l0TileMAct, l0TileKAct);
                    auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayoutTla, Arch::PositionL0A{});

                    if (N_SPLITS != 2U || nL0Itr == 0U) {
                        auto l1ATensorTlaTile =
                            GetTile(l1ATensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K),
                                    tla::MakeShape(l0TileMAct, l0TileKAct));
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                        copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    }
                    // Release L1 P after its last copy. Both MMADs retain the L0A copy.
                    if ((mL0Itr == mL0LoopNum - 1) && (N_SPLITS == 2U ? nL0Itr == 0U : nL0Itr == nL0LoopNum - 1U) &&
                        (kL0Itr == kL0LoopNum - 1)) {
                        SetCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
                    }

                    bool initMmad = (kL0Itr == 0);
                    uint32_t l0TileMAligned = RoundUp(l0TileMAct, 16);
                    if (N_SPLITS != 2U || nL0Itr == 0U) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                    }

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

                    if (mL0Itr == 0 && kL0Itr == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                    }
                    tileMmad(l0CTensorTlaTile, l0ATensorTla, l0BTensorTla, l0TileMAligned, l0TileNAct, l0TileKAct,
                             initMmad);
                    if (N_SPLITS != 2U || nL0Itr == nL0LoopNum - 1U) {
                        // The second MMAD must finish reading P before reusing L0A.
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                }
            }
        }
// Each D slice contains every M row before dual-AIV Fixpipe splits it.
#pragma unroll
        for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
            const uint32_t l0TileNAct = nL0Itr == nL0LoopNum - 1U ? l0TileNTail : L0_TILE_N;
            const uint32_t l0CBufId = (l0CBaseId + nL0Itr) % L0_STAGES;
            const uint32_t l0CEventId = l0CBufId + 2;
            auto l0CTensorTla =
                tla::MakeTensor(l0CTensor[l0CBufId], tla::MakeLayoutL0C(rowNum, l0TileNAct), Arch::PositionL0C{});
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
