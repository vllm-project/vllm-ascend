/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_AIC_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_AIC_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemv_coord.hpp"

#include "catlass/gemv/helper.hpp"

namespace Catlass::Gemv::Block {

template <
    bool ENABLE_UNIT_FLAG_,
    bool ENABLE_SHUFFLE_K_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class XType_,
    class YType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockGemv<
    Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    XType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileMmad_
> {
public:
    // Type Aliases
    using DispatchPolicy = Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementX = typename XType_::Element;
    using LayoutX = typename XType_::Layout;
    using ElementY = typename YType_::Element;
    using LayoutY = typename YType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;
    using LayoutXInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutAInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutXInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutAInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutYInL0 = layout::zN;

    using L1AAlignHelper = Gemv::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1XAlignHelper = Gemv::helper::L1AlignHelper<ElementX, LayoutX>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1A_SIZE = 16 * L1TileShape::N * sizeof(ElementX);
    static constexpr uint32_t L1B_SIZE = L1TileShape::M * L1TileShape::N * sizeof(ElementA);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::M * L0TileShape::N;
    static constexpr uint32_t L0C_TILE_NUM = L0C_SIZE / L0C_TILE_SIZE / sizeof(ElementAccumulator);
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;

    // Check L1TileShape
    static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES) <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

    static constexpr uint32_t L0A_TILE_SIZE = L1XAlignHelper::M_ALIGNED * L0TileShape::N * sizeof(ElementX);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::M * L0TileShape::N * sizeof(ElementA);
    static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");

    /// Construct
    CATLASS_DEVICE
    BlockGemv(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0) 
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_SIZE * STAGES;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementX>(l1AOffset + L1A_SIZE * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BOffset + L1B_SIZE * i);
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementX>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementA>(L0B_PINGPONG_BUF_SIZE * i);

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

    /// Destructor
    CATLASS_DEVICE
    ~BlockGemv() 
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
    }

    /// Perform a block-scoped vector-matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementX> const& gmBlockX, LayoutX const& layoutX,
        AscendC::GlobalTensor<ElementA> const& gmBlockA, LayoutA const& layoutA,
        AscendC::GlobalTensor<ElementY> const& gmBlockY, LayoutY const& layoutY,
        AscendC::GlobalTensor<ElementX> const& gmNextBlockX,
        AscendC::GlobalTensor<ElementA> const& gmNextBlockA,
        GemvCoord const& actualShape, GemvCoord const& actualShapeNext,
        bool isFirstBlock, bool hasNextBlock, uint32_t singleIdx) {
        auto layoutXInL1 = LayoutXInL1::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, L1TileShape::N);
        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::N);
        auto layoutInL0C = LayoutYInL0::MakeLayoutInL0C(MatrixCoord(L1XAlignHelper::M_ALIGNED, actualShape.m()));

        uint32_t nTileCount = CeilDiv<L1TileShape::N>(actualShape.n());

        // Optimize points：ShuffleK
        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K_) {
            startTileIdx = AscendC::GetBlockIdx();
        }
        uint32_t firstTileIdx = startTileIdx % nTileCount;
        uint32_t lastTileIdx = (startTileIdx + nTileCount - 1) % nTileCount;

        uint32_t nActual = (firstTileIdx < nTileCount - 1) ? L1TileShape::N : (actualShape.n() - firstTileIdx * L1TileShape::N);
        uint32_t nRound = RoundUp<L1AAlignHelper::N_ALIGNED>(nActual);

        // main loop
        for (uint32_t nLoopIdx = 0; nLoopIdx < nTileCount; nLoopIdx++) {
            uint32_t shuffleKIdx = (startTileIdx + nLoopIdx) % nTileCount;
            if (shuffleKIdx == firstTileIdx && isFirstBlock) {
                MatrixCoord gmTileAOffset{0, shuffleKIdx * L1TileShape::N};
                uint32_t gmTilexOffset{shuffleKIdx * L1TileShape::N};

                auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTilex = gmBlockX[gmTilexOffset];

                // load first vector x tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                auto layoutTilex = layoutX.GetTileLayout(MakeCoord(nRound));
                copyGmToL1A(l1ATensorList[l1ListId], gmTilex, layoutXInL1, layoutTilex);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

                // load first matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), nRound));
                copyGmToL1B(l1BTensorList[l1ListId], gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            }

            uint32_t l1ListIdNext = (l1ListId + 1) % STAGES;
            uint32_t nActualNext{0};
            uint32_t nRoundNext{0};

            // preload next tile from GM to L1
            if (shuffleKIdx != lastTileIdx) {
                uint32_t shuffleKIdxNext = (startTileIdx + nLoopIdx + 1) % nTileCount;
                nActualNext = (shuffleKIdxNext < nTileCount - 1) ? L1TileShape::N : (actualShape.n() - shuffleKIdxNext * L1TileShape::N);
                nRoundNext = RoundUp<L1AAlignHelper::N_ALIGNED>(nActualNext);

                // Get L1 tensor
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];

                // Get GM tile
                MatrixCoord gmTileAOffset{0, shuffleKIdxNext * L1TileShape::N};
                uint32_t gmTilexOffset{shuffleKIdxNext * L1TileShape::N};

                auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTilex = gmBlockX[gmTilexOffset];

                // load vector x tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                auto layoutTilex = layoutX.GetTileLayout(MakeCoord(nRoundNext));

                copyGmToL1A(l1ATensor, gmTilex, layoutXInL1, layoutTilex);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load Matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), nRoundNext));

                copyGmToL1B(l1BTensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }
            if (shuffleKIdx == lastTileIdx && hasNextBlock) {
                // Get L1 tensor
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];

                // Get GM tensor for next stage
                nActualNext = (firstTileIdx < nTileCount - 1) ? L1TileShape::N : (actualShapeNext.n() - firstTileIdx * L1TileShape::N);  
                nRoundNext = RoundUp<L1AAlignHelper::N_ALIGNED>(nActualNext);

                // Get GM tile
                MatrixCoord gmTileAOffset{0, firstTileIdx * L1TileShape::N};
                uint32_t gmTilexOffset{firstTileIdx * L1TileShape::N};

                auto gmTileA = gmNextBlockA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTilex = gmNextBlockX[gmTilexOffset];

                // load vector x tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);

                auto layoutTilex = layoutX.GetTileLayout(MakeCoord(nRoundNext));

                copyGmToL1A(l1ATensor, gmTilex, layoutXInL1, layoutTilex);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load Matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShapeNext.m(), nRoundNext));

                copyGmToL1B(l1BTensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }

            // get L1 Tensor for current stage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

            uint32_t nRound = RoundUp<L1AAlignHelper::N_ALIGNED>(nActual);
            uint32_t nPartLoop = CeilDiv<L0TileShape::N>(nActual);

            for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ? L0TileShape::N : (nActual - nPartIdx * L0TileShape::N);

                // Locate the current tile on L0A
                auto l0ATile = l0ATensorList[l0AListId];
                LayoutXInL0 layoutxInL0 = LayoutXInL0::template MakeLayout<ElementX>(L1XAlignHelper::M_ALIGNED, nPartActual);

                MatrixCoord l1xOffset{0, nPartIdx * L0TileShape::N};
                auto l1ATile = l1ATensor[layoutXInL1.GetOffset(l1xOffset)];

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                // Load current tile from L1 to L0A
                copyL1ToL0A(l0ATile, l1ATile, layoutxInL0, layoutXInL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);

                // Locate the current tile on L0B
                auto l0BTile = l0BTensorList[l0BListId];
                LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(L0TileShape::M, nPartActual);

                MatrixCoord l1AOffset{0, nPartIdx * L0TileShape::N};
                auto l1BTile = l1BTensor[layoutAInL1.GetOffset(l1AOffset)];

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                // Load current tile from L1 to L0B
                copyL1ToL0B(l0BTile, l1BTile, layoutAInL0, layoutAInL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);

                auto l0CTile = l0CTensor[(singleIdx % L0C_TILE_NUM) * L0C_TILE_SIZE];

                // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                bool initC = ((nLoopIdx == 0) && (nPartIdx == 0));

                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0BListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0AListId]);
                tileMmad(l0CTile, l0ATile, l0BTile, L1XAlignHelper::M_ALIGNED, L0TileShape::M, nPartActual, initC);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);

                l0AListId = (l0AListId + 1) % STAGES;
                l0BListId = (l0BListId + 1) % STAGES;
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);

            l1ListId = l1ListIdNext;

            nActual = nActualNext;
        }

        auto l0CTile = l0CTensor[(singleIdx % L0C_TILE_NUM) * L0C_TILE_SIZE];

        // copy block out
        LayoutY layoutBlock = layoutY.GetTileLayout(MakeCoord(uint32_t(1), actualShape.m()));

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((int32_t)(singleIdx % L0C_TILE_NUM));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((int32_t)(singleIdx % L0C_TILE_NUM));

        copyL0CToGm(gmBlockY, l0CTile, layoutBlock, layoutInL0C);
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementX> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementA> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementX> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementA> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    // Multi-stage event id list
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    // The id of current stage
    uint32_t l1ListId{0};

    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

}  // namespace Catlass::Gemv::Block

#endif  // CATLASS_GEMV_BLOCK_BLOCK_AIC_HPP