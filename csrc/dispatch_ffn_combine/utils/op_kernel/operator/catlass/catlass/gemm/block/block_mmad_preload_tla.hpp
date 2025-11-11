/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_PRELOAD_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_PRELOAD_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
namespace Catlass::Gemm::Block {

template <
    bool ENABLE_UNIT_FLAG_,
    bool ENABLE_SHUFFLE_K_,
    class L1TileShape_,
    class L0TileShape_,
    class TensorA_,
    class TensorB_,
    class TensorC_,
    class TensorBias_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmadTla <
    MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>,
    L1TileShape_,
    L0TileShape_,
    TensorA_,
    TensorB_,
    TensorC_,
    TensorBias_,
    TileCopy_,
    TileMmad_
> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using TensorA = TensorA_;
    using TensorB = TensorB_;
    using TensorC = TensorC_;
    using ElementA = typename TensorA::Element;
    using LayoutA = typename TensorA::Layout;
    using ElementB = typename TensorB::Element;
    using LayoutB = typename TensorB::Layout;
    using ElementC = typename TensorC::Element;
    using LayoutC = typename TensorC::Layout;

    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator = typename CopyL0CToGm::ElementSrc;

    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;

    using L1AAlignHelper = typename TileCopy_::L1AAlignHelper;
    using L1BAlignHelper = typename TileCopy_::L1BAlignHelper;

    static_assert(tla::is_tuple<L1TileShape>::value && tla::is_static<L1TileShape>::value,
        "L1TileShape must be tla::tuple and static!");
    static_assert(tla::is_tuple<L0TileShape>::value && tla::is_static<L0TileShape>::value,
        "L0TileShape must be tla::tuple and static!");

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    // L1 tile size
    static constexpr uint32_t L1A_TILE_SIZE = L1_TILE_M * L1_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1_TILE_N * L1_TILE_K * sizeof(ElementB);
    // L0 tile size
    static constexpr uint32_t L0A_TILE_SIZE = L0_TILE_M * L0_TILE_K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0_TILE_K * L0_TILE_N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1_TILE_M * L1_TILE_N * sizeof(ElementAccumulator);

    // Check LayoutC
    static_assert(tla::detail::isRowMajor<LayoutC>::value, "LayoutC only support RowMajor yet!");

    // Check L1TileShape
    static_assert((L1A_TILE_SIZE + L1B_TILE_SIZE) * STAGES <= ArchTag::L1_SIZE,
        "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static_assert(L0A_TILE_SIZE * STAGES <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert(L0B_TILE_SIZE * STAGES <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");

    static_assert(L1_TILE_M == L0_TILE_M && L1_TILE_N == L0_TILE_N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");

    static constexpr auto L1A_LAYOUT = tla::MakeLayout<ElementA, LayoutAInL1>(L1_TILE_M, L1_TILE_K);
    static constexpr auto L1B_LAYOUT = tla::MakeLayout<ElementB, LayoutBInL1>(L1_TILE_K, L1_TILE_N);

    /// Construct
    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * STAGES;
        // Init buffers
        for (uint32_t i = 0; i < STAGES; i++) {
            // Assign L1/L0A/L0B space for each stages
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);

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
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmadTla()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        TensorA &tensorA, TensorB &tensorB, TensorC &tensorC, TensorA &tensorNextA, TensorB &tensorNextB,
        GemmCoord const &actualShape, GemmCoord const &actualShapeNext, bool isFirstBlock, bool hasNextBlock)
    {
        uint32_t mBlockActual = actualShape.m();
        uint32_t kBlockActual = actualShape.k();
        uint32_t nBlockActual = actualShape.n();
        uint32_t mNextBlockActual = actualShapeNext.m();
        uint32_t kNextBlockActual = actualShapeNext.k();
        uint32_t nNextBlockActual = actualShapeNext.n();

        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mBlockActual);
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nBlockActual);

        auto layoutInL0C = tla::MakeLayoutL0C(mRound, nRound);
        auto tensorL0C = tla::MakeTensor(l0CTensor, layoutInL0C, Arch::PositionL0C{});

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx();
        }

        uint32_t kTileCount = CeilDiv<L1_TILE_K>(kBlockActual);
        uint32_t firstTileIdx = startTileIdx % kTileCount;
        uint32_t lastTileIdx = (startTileIdx + kTileCount - 1) % kTileCount;
        uint32_t kActual =
            (firstTileIdx < kTileCount - 1) ? L1_TILE_K : (kBlockActual - firstTileIdx * L1_TILE_K);

        if (isFirstBlock) {
            // load first matrix A tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            auto tensorL1A = tla::MakeTensor(l1ATensorList[l1ListId], L1A_LAYOUT, Arch::PositionL1{});
            auto tensorTileA = GetTile(
                tensorA,
                tla::MakeCoord(0, firstTileIdx * L1_TILE_K),
                tla::MakeShape(mBlockActual, kActual)
            );
            copyGmToL1A(tensorL1A, tensorTileA, tla::MakeShape(mBlockActual, kActual));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

            // load first matrix B tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            auto tensorL1B = tla::MakeTensor(l1BTensorList[l1ListId], L1B_LAYOUT, Arch::PositionL1{});
            auto tensorTileB = GetTile(
                tensorB,
                tla::MakeCoord(firstTileIdx * L1_TILE_K, 0),
                tla::MakeShape(kActual, nBlockActual)
            );
            copyGmToL1B(tensorL1B, tensorTileB, tla::MakeShape(kActual, nBlockActual));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
        }

        uint32_t mPartLoop = CeilDiv<L0_TILE_M>(mRound);
        uint32_t nPartLoop = CeilDiv<L0_TILE_N>(nRound);

        // main loop
        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            uint32_t shuffleKIdx = (startTileIdx + kLoopIdx) % kTileCount;
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            uint32_t kActualNext{0};

            if (shuffleKIdx != lastTileIdx) {
                uint32_t shuffleKIdxNext = (startTileIdx + kLoopIdx + 1) % kTileCount;
                kActualNext = (shuffleKIdxNext < kTileCount - 1) ?
                    L1_TILE_K : (kBlockActual - shuffleKIdxNext * L1_TILE_K);

                auto tensorL1A = tla::MakeTensor(l1ATensorList[l1ListIdNext], L1A_LAYOUT, Arch::PositionL1{});
                auto tensorL1B = tla::MakeTensor(l1BTensorList[l1ListIdNext], L1B_LAYOUT, Arch::PositionL1{});
                auto tensorTileA = GetTile(
                    tensorA,
                    tla::MakeCoord(0, shuffleKIdxNext * L1_TILE_K),
                    tla::MakeShape(mBlockActual, kActualNext)
                );
                auto tensorTileB = GetTile(
                    tensorB,
                    tla::MakeCoord(shuffleKIdxNext * L1_TILE_K, 0),
                    tla::MakeShape(kActualNext, nBlockActual)
                );

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                copyGmToL1A(tensorL1A, tensorTileA, tla::MakeShape(mBlockActual, kActualNext));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                copyGmToL1B(tensorL1B, tensorTileB, tla::MakeShape(kActualNext, nBlockActual));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }

            // preload next tile from GM to L1
            if (shuffleKIdx == lastTileIdx && hasNextBlock) {
                kActualNext = (firstTileIdx < kTileCount - 1) ?
                    L1_TILE_K : (kNextBlockActual - firstTileIdx * L1_TILE_K);

                // Get L1 tensor for next stage
                auto tensorL1A = tla::MakeTensor(l1ATensorList[l1ListIdNext], L1A_LAYOUT, Arch::PositionL1{});
                auto tensorL1B = tla::MakeTensor(l1BTensorList[l1ListIdNext], L1B_LAYOUT, Arch::PositionL1{});
                // Get GM tile for next stage
                auto tensorTileA = GetTile(
                    tensorNextA,
                    tla::MakeCoord(0, firstTileIdx * L1_TILE_K),
                    tla::MakeShape(mNextBlockActual, kActualNext)
                );
                auto tensorTileB = GetTile(
                    tensorNextB,
                    tla::MakeCoord(firstTileIdx * L1_TILE_K, 0),
                    tla::MakeShape(kActualNext, nNextBlockActual)
                );

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                copyGmToL1A(tensorL1A, tensorTileA, tla::MakeShape(mNextBlockActual, kActualNext));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                copyGmToL1B(tensorL1B, tensorTileB, tla::MakeShape(kActualNext, nNextBlockActual));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }

            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];
            auto tensorL1A = tla::MakeTensor(l1ATensor, L1A_LAYOUT, Arch::PositionL1{});
            auto tensorL1B = tla::MakeTensor(l1BTensor, L1B_LAYOUT, Arch::PositionL1{});
            // Get the loop nums on L0
            uint32_t kPartLoop = CeilDiv<L0_TILE_K>(kActual);

            for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
                uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ?
                    L0_TILE_M : (mRound - mPartIdx * L0_TILE_M);

                for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                    uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ?
                        L0_TILE_K : (kActual - kPartIdx * L0_TILE_K);

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0AListId];
                    LayoutAInL0 layoutAInL0 = tla::MakeLayout<ElementA, LayoutAInL0>(mPartActual, kPartActual);
                    auto tensorL0A = tla::MakeTensor(l0ATile, layoutAInL0, Arch::PositionL0A{});
                    // Locate the current tile of matrix A on L1
                    auto tensorTileL1A = GetTile(tensorL1A, tla::MakeCoord(mPartIdx * L0_TILE_M, kPartIdx * L0_TILE_K),
                        tla::MakeShape(mPartActual, kPartActual));

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    if ((mPartIdx == 0) && (kPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                    }

                    // Load current tile from L1 to L0A
                    copyL1ToL0A(tensorL0A, tensorTileL1A);

                    if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    }

                    for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                        uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                            L0_TILE_N : (nRound - nPartIdx * L0_TILE_N);

                        // Locate the current tile on L0B
                        auto l0BTile = l0BTensorList[l0BListId];
                        LayoutBInL0 layoutBInL0 = tla::MakeLayout<ElementB, LayoutBInL0>(kPartActual, nPartActual);
                        auto tensorL0B = tla::MakeTensor(l0BTile, layoutBInL0, Arch::PositionL0B{});
                        // Locate the current tile of matrix B on L1
                        auto tensorTileL1B = GetTile(tensorL1B,
                                                     tla::MakeCoord(kPartIdx * L0_TILE_K, nPartIdx * L0_TILE_N),
                                                     tla::MakeShape(kPartActual, nPartActual));

                        // Wait for mmad finished
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                        if ((kPartIdx == 0) && (nPartIdx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                        }

                        // Load current tile from L1 to L0B
                        copyL1ToL0B(tensorL0B, tensorTileL1B);

                        // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                        }
                        // Notify to do mmad
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // Locate the current tile on L0C
                        auto tensorTileL0C = GetTile(tensorL0C,
                                                     tla::MakeCoord(mPartIdx * L0_TILE_M, nPartIdx * L0_TILE_N),
                                                     tla::MakeShape(mPartActual, nPartActual));

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                        bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        // If the unit flag is enabled, the unit flag is set according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
                                (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                unitFlag = 0b11;
                            } else {
                                unitFlag = 0b10;
                            }
                        }
                        // Perform calculation operations
                        tileMmad(tensorTileL0C, tensorL0A, tensorL0B,
                                 mPartActual, nPartActual, kPartActual, initC, unitFlag);

                        // Notify to move the next L0B tile
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
                }
            }
            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }

        // copy block out
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            copyL0CToGm(tensorC, tensorL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            copyL0CToGm(tensorC, tensorL0C, 0b11);
        }
    }

protected:
    // Multi-stage tensors list
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
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

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_BLOCK_MMAD_PRELOAD_TLA_HPP
