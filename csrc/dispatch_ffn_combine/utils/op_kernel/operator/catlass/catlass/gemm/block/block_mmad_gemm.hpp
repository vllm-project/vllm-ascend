/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_BLOCK_BLOCK_MMAD_GEMM_HPP
#define CATLASS_BLOCK_BLOCK_MMAD_GEMM_HPP

#include "catlass/catlass.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/arch/resource.hpp"

namespace Catlass::Gemm::Block{
template<
    bool ENABLE_UNIT_FLAG_,
    bool ENABLE_SHUFFLE_K_,
    bool ENABLE_ABBA_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockGemm<
    Gemm::GemmAtlasA2<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_, ENABLE_ABBA_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
>{
public:
    using DispatchPolicy = Gemm::GemmAtlasA2<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_, ENABLE_ABBA_>;
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
    using LayoutAInL1 = typename CopyGmToL1A::LayoutDst;
    using LayoutBInL1 = typename CopyGmToL1B::LayoutDst;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;
    static constexpr bool ENABLE_ABBA = DispatchPolicy::ENABLE_ABBA;
    const uint32_t L1Size = ArchTag::L1_SIZE;
    const uint32_t L1ASize = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    const uint32_t L1BSize = L1TileShape::K * L1TileShape::N * sizeof(ElementB);
    const uint32_t cSize = L1TileShape::M * L1TileShape::N * sizeof(ElementAccumulator);
    const uint32_t BlockCnt = L1TileShape::M * L1TileShape::N;
    const uint32_t L0ASize = ArchTag::L0A_SIZE;
    const uint32_t L0BSize = ArchTag::L0B_SIZE;
    const uint32_t L0CSize = ArchTag::L0C_SIZE;
    const uint32_t L0A_PINGPONG_BUF_LEN = (L0ASize / STAGES);
    const uint32_t L0B_PINGPONG_BUF_LEN = (L0BSize / STAGES);
    const uint32_t l0CBlockNum = ArchTag::L0C_SIZE / cSize;

    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");
    
    CATLASS_DEVICE
    BlockGemm(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1ASize * STAGES;
        for(uint32_t i = 0; i < STAGES; i++){
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1ASize * i);
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1BSize * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_LEN * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_LEN * i);
            // Assign event ID for each stages
            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES;
            l0AEventList[i] = i;
            l0BEventList[i] = i + STAGES;
            // The event id that needs to be set before the loop
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
    }
    // destroy function
    CATLASS_DEVICE
    ~BlockGemm(){
        for (uint32_t i = 0; i < STAGES; i++) 
        {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
        AscendC::GlobalTensor<ElementB> const &gmNextBlockB,
        GemmCoord const &actualShape, GemmCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, uint32_t singleIdx)
    { 
        uint32_t K = actualShape.k();
        uint32_t maxKPerBlock = L1TileShape::K;
        uint32_t kLoops = CeilDiv(K, maxKPerBlock);
        uint32_t startTileIdx{0};
        if (ENABLE_SHUFFLE_K) 
        {
            startTileIdx = AscendC::GetBlockIdx();
        }
        uint32_t firstTileIdx = startTileIdx % kLoops; 
        uint32_t lastTileIdx = (startTileIdx + kLoops - 1) % kLoops; 
        uint32_t kGmActual = (firstTileIdx == kLoops - 1) ? (K - firstTileIdx * maxKPerBlock) : maxKPerBlock;
        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        for (uint32_t kIdx = 0; kIdx < kLoops; kIdx++)
        {
            uint32_t shuffleKIdx = (startTileIdx + kIdx) % kLoops;
            if (shuffleKIdx == firstTileIdx && isFirstBlock)
            { 
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kGmActual)); 
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kGmActual, actualShape.n()));
                MatrixCoord gmTileAOffset{0, shuffleKIdx * maxKPerBlock}; 
                auto gmTileA = gmA[layoutA.GetOffset(gmTileAOffset)];
                MatrixCoord gmTileBOffset{shuffleKIdx * maxKPerBlock, 0}; 
                auto gmTileB = gmB[layoutB.GetOffset(gmTileBOffset)];
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]); 
                copyGmToL1A(l1ATensor[l1ListId], gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                copyGmToL1B(l1BTensor[l1ListId], gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            }
            l1ListIdNext = 1 - l1ListId;
            uint32_t kGmActualNext = 0; 
            if (shuffleKIdx != lastTileIdx)
            { 
                uint32_t shuffleKIdxNext = (startTileIdx + kIdx + 1) % kLoops;
                kGmActualNext = (shuffleKIdxNext == kLoops - 1) ? (K - shuffleKIdxNext * maxKPerBlock) : maxKPerBlock; 
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kGmActualNext)); 
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kGmActualNext, actualShape.n()));
                MatrixCoord gmTileAOffset{0, shuffleKIdxNext * maxKPerBlock}; 
                auto gmTileA = gmA[layoutA.GetOffset(gmTileAOffset)];
                MatrixCoord gmTileBOffset{shuffleKIdxNext * maxKPerBlock, 0}; 
                auto gmTileB = gmB[layoutB.GetOffset(gmTileBOffset)];
                if (ENABLE_ABBA)
                {
                    if (shuffleKIdxNext % 2 == 1)
                    { 
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                        copyGmToL1B(l1BTensor[l1ListIdNext], gmTileB, layoutBInL1, layoutTileB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                        copyGmToL1A(l1ATensor[l1ListIdNext], gmTileA, layoutAInL1, layoutTileA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                    } else {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                        copyGmToL1A(l1ATensor[l1ListIdNext], gmTileA, layoutAInL1, layoutTileA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                        copyGmToL1B(l1BTensor[l1ListIdNext], gmTileB, layoutBInL1, layoutTileB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                    }
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                    copyGmToL1A(l1ATensor[l1ListIdNext], gmTileA, layoutAInL1, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                    copyGmToL1B(l1BTensor[l1ListIdNext], gmTileB, layoutBInL1, layoutTileB);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                }
            }
            if (shuffleKIdx == lastTileIdx && hasNextBlock)
            { 
                kGmActualNext = (firstTileIdx == kLoops - 1) ? (K - firstTileIdx * maxKPerBlock) : maxKPerBlock;
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShapeNext.m(), kGmActualNext)); 
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kGmActualNext, actualShapeNext.n()));
                MatrixCoord gmTileAOffset{0, firstTileIdx * maxKPerBlock}; 
                auto gmNextTileA = gmNextBlockA[layoutA.GetOffset(gmTileAOffset)];
                MatrixCoord gmTileBOffset{firstTileIdx * maxKPerBlock, 0}; 
                auto gmNextTileB = gmNextBlockB[layoutB.GetOffset(gmTileBOffset)];
                if (ENABLE_ABBA)
                {
                    if (shuffleKIdx % 2 == 0)
                    { 
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                        copyGmToL1B(l1BTensor[l1ListIdNext], gmNextTileB, layoutBInL1, layoutTileB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                        copyGmToL1A(l1ATensor[l1ListIdNext], gmNextTileA, layoutAInL1, layoutTileA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                    } else {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                        copyGmToL1A(l1ATensor[l1ListIdNext], gmNextTileA, layoutAInL1, layoutTileA);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                        copyGmToL1B(l1BTensor[l1ListIdNext], gmNextTileB, layoutBInL1, layoutTileB);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                    }
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                    copyGmToL1A(l1ATensor[l1ListIdNext], gmNextTileA, layoutAInL1, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                    copyGmToL1B(l1BTensor[l1ListIdNext], gmNextTileB, layoutBInL1, layoutTileB);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
                }
            }

            uint32_t kL0TileSize = L0TileShape::K;
            uint32_t kL0Loops = CeilDiv(kGmActual, kL0TileSize);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            auto l1ATile = l1ATensor[l1ListId];
            auto l1BTile = l1BTensor[l1ListId];
            uint32_t mActual{0};
            uint32_t nActual{0};
            for (uint32_t kL0Idx = 0; kL0Idx < kL0Loops; kL0Idx++)
            {
                uint32_t kL0Actual = (kL0Idx == kL0Loops - 1) ? (kGmActual - kL0Idx * kL0TileSize) : kL0TileSize;
                LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(L1TileShape::M, kL0Actual);
                LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, L1TileShape::N);
                uint32_t l1TileAOffset = layoutAInL1.GetOffset(MatrixCoord(0, kL0Idx * kL0TileSize));
                uint32_t l1TileBOffset = layoutBInL1.GetOffset(MatrixCoord(kL0Idx * kL0TileSize, 0));
                auto l1TileA = l1ATile[l1TileAOffset];
                auto l1TileB = l1BTile[l1TileBOffset];
                auto l0TileA = l0ATensor[l0ListId];
                auto l0TileB = l0BTensor[l0ListId];
                mActual = L1TileShape::M;
                nActual = L1TileShape::N;
                if (ENABLE_ABBA)
                {
                    if (shuffleKIdx % 2 == 0)
                    {
                        if (kL0Idx % 2 == 0) { 
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                            copyL1ToL0B(l0TileB, l1TileB, layoutBInL0, layoutBInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                            copyL1ToL0A(l0TileA, l1TileA, layoutAInL0, layoutAInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                        } else {
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                            copyL1ToL0A(l0TileA, l1TileA, layoutAInL0, layoutAInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                            copyL1ToL0B(l0TileB, l1TileB, layoutBInL0, layoutBInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                        }
                    } else {
                        if (kL0Idx % 2 == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                            copyL1ToL0A(l0TileA, l1TileA, layoutAInL0, layoutAInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                            copyL1ToL0B(l0TileB, l1TileB, layoutBInL0, layoutBInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                        } else {
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                            copyL1ToL0B(l0TileB, l1TileB, layoutBInL0, layoutBInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                            copyL1ToL0A(l0TileA, l1TileA, layoutAInL0, layoutAInL1);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                        }
                    }
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                    copyL1ToL0A(l0TileA, l1TileA, layoutAInL0, layoutAInL1);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                    copyL1ToL0B(l0TileB, l1TileB, layoutBInL0, layoutBInL1);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                }
                if (kL0Idx == kL0Loops - 1)
                { 
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                    l1ListId = l1ListIdNext;
                    kGmActual = kGmActualNext;
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0ListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0ListId]);
                tileMmad(l0CTensor[(singleIdx % l0CBlockNum) * BlockCnt], l0TileA, l0TileB, mActual, nActual, kL0Actual, (kIdx == 0) && (kL0Idx == 0));
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ListId]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0ListId]);
                l0ListId = 1 - l0ListId;
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((int32_t)(singleIdx % l0CBlockNum));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((int32_t)(singleIdx % l0CBlockNum));
        auto layoutInL0X = LayoutCInL0::MakeLayoutInL0C(MakeCoord(L1TileShape::M, L1TileShape::N)); 
        LayoutC layoutBlock = layoutC.GetTileLayout(MakeCoord(actualShape.m(), actualShape.n()));
        copyL0CToGm(gmC, l0CTensor[(singleIdx % l0CBlockNum) * BlockCnt], layoutBlock, layoutInL0X);
    }
private:
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;
    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    // The id of current stage
    uint32_t l1ListId{0};
    uint32_t l0ListId{0};
    uint32_t l1ListIdNext{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};
}

#endif // CATLASS_BLOCK_BLOCK_MMAD_GEMM_HPP