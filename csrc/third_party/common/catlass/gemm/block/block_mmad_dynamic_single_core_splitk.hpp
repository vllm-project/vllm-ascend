/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_MMAD_DYNAMIC_SINGLE_CORE_SPLITK_HPP
#define CATLASS_GEMM_BLOCK_MMAD_DYNAMIC_SINGLE_CORE_SPLITK_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"

namespace Catlass::Gemm::Block {

template <
    uint32_t L1A_STAGES_,
    uint32_t L1B_STAGES_,
    uint32_t L0C_STAGES_,
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
    MmadAtlasA2DynamicSingleCoreSplitk<L1A_STAGES_, L1B_STAGES_, L0C_STAGES_, ENABLE_UNIT_FLAG_>,
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
    using DispatchPolicy = MmadAtlasA2DynamicSingleCoreSplitk<L1A_STAGES_, L1B_STAGES_, L0C_STAGES_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
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
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr uint32_t L1A_STAGES = DispatchPolicy::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0AB_STAGES = DispatchPolicy::L0AB_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;

    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0AB_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0AB_STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = ArchTag::L0C_SIZE / L0C_STAGES;

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    /// Construct
    CATLASS_DEVICE
    BlockMmad(
        GemmCoord const &l1TileShape_,
        GemmCoord const &l0TileShape_,
        Arch::Resource<ArchTag> &resource,
        uint32_t l1BufAddrStart = 0
    )
        : l1TileShape(l1TileShape_)
        , l0TileShape(l0TileShape_)
    {
        if ASCEND_IS_AIV { return; }

        uint32_t l1ASize = l1TileShape.m() * l1TileShape.k() * sizeof(ElementA);
        uint32_t l1BSize = l1TileShape.k() * l1TileShape.n() * sizeof(ElementB);

        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + l1ASize * L1A_STAGES;
        // Init buffers
        for (uint32_t i = 0; i < L1A_STAGES; i++) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + l1ASize * i);
            l1AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + l1BSize * i);
            l1BEventList[i] = i + L1A_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }

        for (uint32_t i = 0; i < L0AB_STAGES; i++) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0AEventList[i] = i;
            l0BEventList[i] = i + L0AB_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; i++) {
            l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
            l0CEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockMmad()
    {
        if ASCEND_IS_AIV { return; }

        for (uint32_t i = 0; i < L1A_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }

        for (uint32_t i = 0; i < L0AB_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmBlockA,
        LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmBlockB,
        LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmBlockC,
        LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
        AscendC::GlobalTensor<ElementB> const &gmNextBlockB,
        GemmCoord const &actualShape,
        GemmCoord const &actualShapeNext,
        bool needLoadNextA,
        bool needLoadNextB,
        bool atomicAdd
    )
    {
        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(l1TileShape.m(), l1TileShape.k());
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(l1TileShape.k(), l1TileShape.n());

        bool isFirstBlock = !preLoadedA && !preLoadedB;
        bool isLastBlock = !needLoadNextA && !needLoadNextB;
        uint32_t l1AListIdNext = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
        uint32_t l1BListIdNext = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;

        if (isFirstBlock) {
            // Load matrix A tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), actualShape.k()));
            copyGmToL1A(l1ATensorList[l1AListId], gmBlockA, layoutAInL1, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);

            // Load matrix B tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
            auto layoutTileB = layoutB.GetTileLayout(MakeCoord(actualShape.k(), actualShape.n()));
            copyGmToL1B(l1BTensorList[l1BListId], gmBlockB, layoutBInL1, layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
        }

        if (needLoadNextA && (L1A_STAGES > 1)) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShapeNext.m(), actualShapeNext.k()));
            copyGmToL1A(l1ATensorList[l1AListIdNext], gmNextBlockA, layoutAInL1, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListIdNext]);
        }
        if (needLoadNextB && (L1B_STAGES > 1)) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
            auto layoutTileB = layoutB.GetTileLayout(MakeCoord(actualShapeNext.k(), actualShapeNext.n()));
            copyGmToL1B(l1BTensorList[l1BListIdNext], gmNextBlockB, layoutBInL1, layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListIdNext]);
        }

        // Get L1 tensor for current stage
        auto l1ATensor = l1ATensorList[l1AListId];
        auto l1BTensor = l1BTensorList[l1BListId];

        // Atomic set once in mmad level
        if (atomicAdd) {
            AscendC::SetAtomicAdd<ElementAccumulator>();
        } else {
            AscendC::SetAtomicNone();
        }

        uint32_t l1mLoops = CeilDiv(actualShape.m(), l0TileShape.m());
        uint32_t l1nLoops = CeilDiv(actualShape.n(), l0TileShape.n());
        for (uint32_t l1mLoopIdx = 0; l1mLoopIdx < l1mLoops; l1mLoopIdx++) {
            for (uint32_t l1nLoopIdx = 0; l1nLoopIdx < l1nLoops; l1nLoopIdx++) {
                // layoutAInL0
                uint32_t mActual = (l1mLoopIdx < l1mLoops - 1) ? l0TileShape.m()
                                                               : (actualShape.m() - l1mLoopIdx * l0TileShape.m());
                uint32_t nActual = (l1nLoopIdx < l1nLoops - 1) ? l0TileShape.n()
                                                               : (actualShape.n() - l1nLoopIdx * l0TileShape.n());
                uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(mActual);
                uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(nActual);
                auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mRound, nRound));

                uint32_t l0ABufId = 0;
                uint32_t l0BBufId = 0;

                // wait/set conditions in block/l1mLoops/l1nLoops level
                bool mnLoopStart = (l1mLoopIdx == 0 && l1nLoopIdx == 0);
                bool mnLoopEnd = (l1mLoopIdx == l1mLoops - 1) && (l1nLoopIdx == l1nLoops - 1);
                bool waitFlagA = (isFirstBlock || preLoadedA) && mnLoopStart;
                bool setFlagA = (needLoadNextA || isLastBlock) && mnLoopEnd;
                bool waitFlagB = (isFirstBlock || preLoadedB) && mnLoopStart;
                bool setFlagB = (needLoadNextB || isLastBlock) && mnLoopEnd;

                if constexpr (!ENABLE_UNIT_FLAG) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
                }

                // Get the loop nums on L0
                uint32_t l1kLoops = CeilDiv(actualShape.k(), l0TileShape.k());
                for (uint32_t l1kLoopIdx = 0; l1kLoopIdx < l1kLoops; l1kLoopIdx++) {
                    uint32_t kActual = (l1kLoopIdx < l1kLoops - 1) ? l0TileShape.k() : (actualShape.k() - l1kLoopIdx * l0TileShape.k());

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0ABufId];
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mRound, kActual);
                    // Locate the current tile of matrix A on L1
                    MatrixCoord l1ACoord{l1mLoopIdx * l0TileShape.m(), l1kLoopIdx * l0TileShape.k()};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1ACoord)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ABufId]);
                    if (waitFlagA && (l1kLoopIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
                    }

                    copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, layoutAInL1);

                    if (setFlagA && (l1kLoopIdx == l1kLoops - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                    }

                    // Locate the current tile on L0B
                    auto l0BTile = l0BTensorList[l0BBufId];
                    LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kActual, nRound);
                    // Locate the current tile of matrix B on L1
                    MatrixCoord l1BCoord{l1kLoopIdx * l0TileShape.k(), l1nLoopIdx * l0TileShape.n()};
                    auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BCoord)];

                    // Wait for mmad finished
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BBufId]);
                    // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                    if (waitFlagB && (l1kLoopIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
                    }

                    // Load current tile from L1 to L0B
                    copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, layoutBInL1);

                    // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                    if (setFlagB && (l1kLoopIdx == l1kLoops - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
                    }
                    // Notify to do mmad
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);

                    // Locate the current tile on L0C
                    auto l0CTile = l0CTensorList[l0CListId];

                    // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                    // Wait for loading L0B
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0CEventList[l0CListId]);

                    // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                    bool initC = (l1kLoopIdx == 0);
                    // If the unit flag is enabled, the unit flag is set according to the calculation progress
                    uint8_t unitFlag = 0b00;
                    if constexpr (ENABLE_UNIT_FLAG) {
                        if (l1kLoopIdx == l1kLoops - 1) {
                            unitFlag = 0b11;
                        } else {
                            unitFlag = 0b10;
                        }
                    }
                    // Perform calculation operations
                    tileMmad(l0CTile, l0ATile, l0BTile, mRound, nRound, kActual, initC, unitFlag);

                    // Notify to move the next L0B tile
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BBufId]);
                    l0BBufId = (l0BBufId + 1 < L0AB_STAGES) ? (l0BBufId + 1) : 0;

                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ABufId]);
                    l0ABufId = (l0ABufId + 1 < L0AB_STAGES) ? (l0ABufId + 1) : 0;
                }
                // copy block out
                LayoutC layoutBlock = layoutC.GetTileLayout(MakeCoord(mActual, nActual));
                MatrixCoord gmCoordC{l1mLoopIdx * l0TileShape.m(), l1nLoopIdx * l0TileShape.n()};
                auto gmTileC = gmBlockC[layoutC.GetOffset(gmCoordC)];
                if constexpr (!ENABLE_UNIT_FLAG) {
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                    copyL0CToGm(gmTileC, l0CTensorList[l0CListId], layoutBlock, layoutInL0C);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
                } else {
                    copyL0CToGm(gmTileC, l0CTensorList[l0CListId], layoutBlock, layoutInL0C, 0b11);
                }
                l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;
            }
        }

        // if L1A_STAGES/L1B_STAGES == 1, load next block after compute
        if (needLoadNextA && (L1A_STAGES == 1)) {
            // Load first matrix A tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShapeNext.m(), actualShapeNext.k()));
            copyGmToL1A(l1ATensorList[l1AListIdNext], gmNextBlockA, layoutAInL1, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListIdNext]);
        }
        if (needLoadNextB && (L1B_STAGES == 1)) {
            // Load first matrix B tile from GM to L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
            auto layoutTileB = layoutB.GetTileLayout(MakeCoord(actualShapeNext.k(), actualShapeNext.n()));
            copyGmToL1B(l1BTensorList[l1BListIdNext], gmNextBlockB, layoutBInL1, layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListIdNext]);
        }

        preLoadedA = needLoadNextA;
        preLoadedB = needLoadNextB;
        l1AListId = needLoadNextA ? l1AListIdNext : l1AListId;
        l1BListId = needLoadNextB ? l1BListIdNext : l1BListId;
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[L0AB_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];

    int32_t l1AEventList[L1A_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    int32_t l0AEventList[L0AB_STAGES];
    int32_t l0BEventList[L0AB_STAGES];
    int32_t l0CEventList[L0C_STAGES];

    bool preLoadedA{false};
    bool preLoadedB{false};
    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l0CListId{0};
    GemmCoord l1TileShape;
    GemmCoord l0TileShape;

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

} // namespace Catlass::Gemm::Block

#endif // CATLASS_GEMM_BLOCK_MMAD_DYNAMIC_SINGLE_CORE_SPLITK_HPP
