/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_BLOCK_BLOCK_CONV2D_PINGPONG_HPP
#define CATLASS_CONV_BLOCK_BLOCK_CONV2D_PINGPONG_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/conv_coord.hpp"
#include "catlass/conv/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"

namespace Catlass::Conv::Block {

template <
    uint32_t L1A_STAGES_,
    uint32_t L1B_STAGES_,
    uint32_t L0A_STAGES_,
    uint32_t L0B_STAGES_,
    uint32_t L0C_STAGES_,
    bool ENABLE_UNIT_FLAG_,
    class FmapL1TileShape_,
    class FilterL1TileShape_,
    class L0TileShape_,
    class FmapType_,
    class FilterType_,
    class OutputType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockConv2d <
    ConvAtlasA2Pingpong<
        L1A_STAGES_,
        L1B_STAGES_,
        L0A_STAGES_,
        L0B_STAGES_,
        L0C_STAGES_,
        ENABLE_UNIT_FLAG_
    >,
    FmapL1TileShape_,
    FilterL1TileShape_,
    L0TileShape_,
    FmapType_,
    FilterType_,
    OutputType_,
    BiasType_,
    TileCopy_,
    TileMmad_
> {
public:
    // Type Aliases
    using DispatchPolicy = ConvAtlasA2Pingpong<
        L1A_STAGES_,
        L1B_STAGES_,
        L0A_STAGES_,
        L0B_STAGES_,
        L0C_STAGES_,
        ENABLE_UNIT_FLAG_
    >;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using FmapL1TileShape = FmapL1TileShape_;
    using FilterL1TileShape = FilterL1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementFmap = typename FmapType_::Element;
    using LayoutFmap = typename FmapType_::Layout;
    using ElementFilter = typename FilterType_::Element;
    using LayoutFilter = typename FilterType_::Layout;
    using ElementOutput = typename OutputType_::Element;
    using LayoutOutput = typename OutputType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementFmap, ElementFilter>::ElementAccumulator;
    using LayoutFmapInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutFilterInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutFmapInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutFilterInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutOutputInL0 = layout::zN;

    using L1FmapAlignHelper = Gemm::helper::L1AlignHelper<ElementFmap, LayoutFmap>;
    using L1FilterAlignHelper = Gemm::helper::L1AlignHelper<ElementFilter, LayoutFilter>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t MAX_STAGES = 2;
    static constexpr uint32_t L1A_STAGES =
        (DispatchPolicy::L1A_STAGES < MAX_STAGES) ? DispatchPolicy::L1A_STAGES : MAX_STAGES;
    static constexpr uint32_t L1B_STAGES =
        (DispatchPolicy::L1B_STAGES < MAX_STAGES) ? DispatchPolicy::L1B_STAGES : MAX_STAGES;
    static constexpr uint32_t L0A_STAGES =
        (DispatchPolicy::L0A_STAGES < MAX_STAGES) ? DispatchPolicy::L0A_STAGES : MAX_STAGES;
    static constexpr uint32_t L0B_STAGES =
        (DispatchPolicy::L0B_STAGES < MAX_STAGES) ? DispatchPolicy::L0B_STAGES : MAX_STAGES;
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / L0A_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / L0B_STAGES;

    // Check LayoutOutput
    static_assert(std::is_same_v<LayoutOutput, layout::NC1HWC0>, "LayoutOutput only support NC1HWC0 yet!");

    // Check L0TileShape
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementFilter);
    static_assert((L0B_TILE_SIZE * L0B_STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");
   
    static constexpr uint32_t ELE_NUM_A_PER_C0 = BYTE_PER_C0 / sizeof(ElementFmap);
    static constexpr uint32_t ELE_NUM_B_PER_C0 = BYTE_PER_C0 / sizeof(ElementFilter);

    // Check big and small L1TileShape::Cin1 
    static_assert(
        (FilterL1TileShape::Cin1 >= FmapL1TileShape::Cin1) &&
        (FilterL1TileShape::Cin1 % FmapL1TileShape::Cin1 == 0),
        "FilterL1TileShape::Cin1 must be a multiple of FmapL1TileShape::Cin1");

    static constexpr uint32_t K_FMAP_PER_FILTER = FilterL1TileShape::Cin1 / FmapL1TileShape::Cin1;

    /// Construct
    CATLASS_DEVICE
    BlockConv2d(Arch::Resource<ArchTag> &resource, const Conv2dFilterParams &filterParams_, uint32_t l1BufAddrStart = 0)
        : filterParams(filterParams_), copyL1ToL0A(filterParams_)
    {
        hiBlock = (FmapL1TileShape::Ho - 1) * filterParams.strideH() +
            (filterParams.kh() - 1) * filterParams.dilationH() + 1;
        wiBlock = (FmapL1TileShape::Wo - 1) * filterParams.strideW() +
            (filterParams.kw() - 1) * filterParams.dilationW() + 1;
        l1A_size = FmapL1TileShape::Cin1 * hiBlock * wiBlock * BYTE_PER_C0;
        l1B_size = FilterL1TileShape::Cin1 * filterParams.kh() * filterParams.kw() * 
            FilterL1TileShape::Cout * BYTE_PER_C0;

        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + l1A_size * L1A_STAGES;

        // Init buffers
        for (uint32_t i = 0; i < MAX_STAGES; i++) {
            if (i < L1A_STAGES) {
                l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementFmap>(l1AOffset + l1A_size * i);
                l1AEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            }
            if (i < L1B_STAGES) {
                l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementFilter>(l1BOffset + l1B_size * i);
                l1BEventList[i] = i + L1A_STAGES;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
            if (i < L0A_STAGES) {
                l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementFmap>(L0A_PINGPONG_BUF_SIZE * i);
                l0AEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            if (i < L0B_STAGES) {
                l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementFilter>(L0B_PINGPONG_BUF_SIZE * i);
                l0BEventList[i] = i + L0A_STAGES;
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            }
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockConv2d() {
        for (uint32_t i = 0; i < MAX_STAGES; i++) {
            if (i < L1A_STAGES) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            }
            if (i < L1B_STAGES) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            }
            if (i < L0A_STAGES) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            }
            if (i < L0B_STAGES) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
            } 
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Perform a block-scoped matrix multiply-accumulate
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementFmap> const &gmFmap, LayoutFmap const &layoutFmap,
        AscendC::GlobalTensor<ElementFilter> const &gmFilter, LayoutFilter const &layoutFilter,
        AscendC::GlobalTensor<ElementOutput> const &gmOutput, LayoutOutput const &layoutOutput,
        Conv2dCoord const &actualShape, uint8_t* blockPadList)
    {
        uint8_t blockPadLeft = blockPadList[0];
        uint8_t blockPadRight = blockPadList[1];
        uint8_t blockPadTop = blockPadList[2];
        uint8_t blockPadBottom = blockPadList[3];
        uint32_t wiActual = actualShape.w();
        uint32_t hiActual = actualShape.h();
        int32_t wiActualOrg = wiActual + blockPadLeft + blockPadRight;
        int32_t hiActualOrg = hiActual + blockPadTop + blockPadBottom;

        uint32_t hoActual = (hiActualOrg - 1 - 
            (filterParams.kh() - 1) * filterParams.dilationH()) / filterParams.strideH() + 1;
        uint32_t woActual = (wiActualOrg - 1 -
            (filterParams.kw() - 1) * filterParams.dilationW()) / filterParams.strideW() + 1;
        uint32_t howoActual = hoActual * woActual;

        uint32_t howoRound = RoundUp<L1FmapAlignHelper::HOWO_ALIGNED>(howoActual);
        uint32_t coutRound = RoundUp<L1FilterAlignHelper::COUT_ALIGNED>(actualShape.cout());
        
        uint32_t nL0 = Min(RoundUp<C0_NUM_PER_FRACTAL>(L0TileShape::N), coutRound);
        uint32_t nPartLoop = CeilDiv(coutRound, nL0);
        uint32_t cin1L0Tile = Max(L0TileShape::K / (filterParams.kh() * filterParams.kw() * ELE_NUM_A_PER_C0), 1U);
        
        auto layoutFmapInL1 = LayoutFmapInL1::template MakeLayout<ElementFmap>(
            (uint32_t)1, FmapL1TileShape::Cin1, hiActual, wiActual, ELE_NUM_A_PER_C0);
        auto layoutFilterInL1 = LayoutFilterInL1::template MakeLayout<ElementFilter>(
            FilterL1TileShape::Cin1, filterParams.kh(), filterParams.kw(), coutRound, ELE_NUM_B_PER_C0);
        auto layoutInL0C = LayoutOutputInL0::MakeLayoutInL0C(MakeCoord(howoRound, coutRound));
    
        uint32_t cin1Actual = min(actualShape.cin1(), FmapL1TileShape::Cin1);
        uint32_t cin1FilterActual = min(actualShape.cin1(), FilterL1TileShape::Cin1);
       
        // load first Fmap tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
        auto layoutTileFmap = layoutFmap.GetTileLayout(
            MakeCoord((uint32_t)1, cin1Actual, hiActual, wiActual, ELE_NUM_A_PER_C0));
        copyGmToL1A(l1ATensorList[l1AListId], gmFmap, layoutFmapInL1, layoutTileFmap);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);

        // load first Filter tile from GM to L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
        auto layoutTileFilter = layoutFilter.GetTileLayout(
            MakeCoord(cin1FilterActual, (uint32_t)filterParams.kh(), (uint32_t)filterParams.kw(),
                      actualShape.cout(), ELE_NUM_B_PER_C0));
        copyGmToL1B(l1BTensorList[l1BListId], gmFilter, layoutFilterInL1, layoutTileFilter);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }

        // main loop
        uint32_t cin1TileCnt =
            CeilDiv<FmapL1TileShape::Cin1>(actualShape.cin1());
        uint32_t cin1FilterTileCnt =
            CeilDiv<FilterL1TileShape::Cin1>(actualShape.cin1());
        for (uint32_t cin1LoopIdx = 0; cin1LoopIdx < cin1TileCnt; cin1LoopIdx++) {
            uint32_t  cin1FmapIdx = cin1LoopIdx % K_FMAP_PER_FILTER;

            uint32_t l1AListIdNext = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
            uint32_t l1BListIdNext = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;

            uint32_t cin1ActualNext{0};
            uint32_t cin1FilterActualNext{0};

            // preload next tile A from GM to L1
            if (cin1LoopIdx < cin1TileCnt - 1) {
                uint32_t cin1LoopIdxNext = cin1LoopIdx + 1;
                cin1ActualNext = (cin1LoopIdxNext < cin1TileCnt - 1) ?
                    FmapL1TileShape::Cin1 : (actualShape.cin1() - cin1LoopIdxNext * FmapL1TileShape::Cin1);
                
                // Get L1 tensor A for next stage
                auto l1ATensor = l1ATensorList[l1AListIdNext];
                // Get GM tile A for next stage
                Conv2dFmapCoord gmTileFmapOffset{0, cin1LoopIdxNext * FmapL1TileShape::Cin1, 0, 0, 0};
                auto gmTileFmap = gmFmap[layoutFmap.GetOffset(gmTileFmapOffset)];

                // load next Fmap tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
                layoutTileFmap = layoutFmap.GetTileLayout(
                    MakeCoord((uint32_t)1, cin1ActualNext, hiActual, wiActual, ELE_NUM_A_PER_C0));
                copyGmToL1A(l1ATensor, gmTileFmap, layoutFmapInL1, layoutTileFmap);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListIdNext]);

                // preload next tile B from GM to L1
                if (cin1FmapIdx == K_FMAP_PER_FILTER - 1) {
                    uint32_t cin1FilterIdxNext = cin1LoopIdxNext / K_FMAP_PER_FILTER;
                    cin1FilterActualNext = (cin1FilterIdxNext < cin1FilterTileCnt - 1) ?
                        FilterL1TileShape::Cin1 : (actualShape.cin1() - cin1FilterIdxNext * FilterL1TileShape::Cin1);
                
                    // Get L1 tensor B for next stage
                    auto l1BTensor = l1BTensorList[l1BListIdNext];
                    // Get GM tile B for next stage
                    Conv2dFilterCoord gmTileFilterOffset{cin1FilterIdxNext * FilterL1TileShape::Cin1, 0, 0, 0, 0};
                    auto gmTileFilter = gmFilter[layoutFilter.GetOffset(gmTileFilterOffset)];

                    // load next Filter tile from GM to L1
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
                    auto layoutTileFilter = layoutFilter.GetTileLayout(
                        MakeCoord(cin1FilterActualNext, (uint32_t)filterParams.kh(), (uint32_t)filterParams.kw(),
                                actualShape.cout(), ELE_NUM_B_PER_C0));
                    copyGmToL1B(l1BTensor, gmTileFilter, layoutFilterInL1, layoutTileFilter);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListIdNext]);
                }
            }

            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[l1AListId];
            auto l1BTensor = l1BTensorList[l1BListId];

            // Get the loop nums on L0
            uint32_t kPartLoop = CeilDiv(cin1Actual, cin1L0Tile);

            for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                uint32_t cin1PartActual = (kPartIdx < kPartLoop - 1) ?
                    cin1L0Tile : (cin1Actual - kPartIdx * cin1L0Tile);
                uint32_t kPartActual = 
                    cin1PartActual * filterParams.kh() * filterParams.kw() * ELE_NUM_A_PER_C0;

                // Locate the current tile on L0A
                auto l0ATile = l0ATensorList[l0AListId];
                LayoutFmapInL0 layoutFmapInL0 = LayoutFmapInL0::template MakeLayout<ElementFmap>(howoRound, kPartActual);
                // Locate the current tile of matrix A on L1
                Conv2dFmapCoord l1AOffset{0, kPartIdx * cin1L0Tile, 0, 0, 0};
                auto l1ATile = l1ATensor[layoutFmapInL1.GetOffset(l1AOffset)];

                // Wait for mmad finished
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                if (kPartIdx == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
                }

                // Load current tile from L1 to L0A
                copyL1ToL0A(l0ATile, l1ATile, layoutFmapInL0, layoutFmapInL1, blockPadList);
                
                if (kPartIdx == kPartLoop - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                }

                for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                    uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                        nL0 : (coutRound - nPartIdx * nL0);

                    // Locate the current tile on L0B
                    auto l0BTile = l0BTensorList[l0BListId];
                    LayoutFilterInL0 layoutFilterInL0 = LayoutFilterInL0::template MakeLayout<ElementFilter>(kPartActual, nPartActual);
                    // Load current tile of matrix B on L1
                    Conv2dFilterCoord l1BOffset{
                        cin1FmapIdx * FmapL1TileShape::Cin1 + kPartIdx * cin1L0Tile,
                        0, 0, nPartIdx * nL0, 0};
                    auto l1BTile = l1BTensor[layoutFilterInL1.GetOffset(l1BOffset)];

                    // Wait for mmad finished
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    if ((cin1FmapIdx == 0) && (kPartIdx == 0) && (nPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
                    }

                    // Load current tile from L1 to L0B
                    copyL1ToL0B(l0BTile, l1BTile, layoutFilterInL0, layoutFilterInL1);
                    
                    if (((cin1FmapIdx == K_FMAP_PER_FILTER - 1) || (cin1LoopIdx == cin1TileCnt - 1)) &&
                        (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
                    }

                    // Notify to do mmad
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    // Locate the current tile on L0C
                    MatrixCoord l0COffset{0, nPartIdx * nL0};
                    auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0COffset)];

                    // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                    // Wait for loading L0B
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                    bool initC = (cin1LoopIdx == 0) && (kPartIdx == 0);
                    // If the unit flag is enabled, the unit flag is set according to the calculation progress
                    uint8_t unitFlag = 0b00;
                    if constexpr (ENABLE_UNIT_FLAG) {
                        if ((cin1LoopIdx == cin1TileCnt - 1) && (kPartIdx == kPartLoop - 1)) {
                            unitFlag = 0b11;
                        } else {
                            unitFlag = 0b10;
                        }
                    }
                    // Perfrom calculation operations
                    tileMmad(
                        l0CTile, l0ATile, l0BTile,
                        howoRound, nPartActual, kPartActual,
                        initC, unitFlag);

                    // Notify to move the next L0A, L0B tile
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
            }
            l1AListId = l1AListIdNext;
            cin1Actual = cin1ActualNext;
            if (cin1FmapIdx == K_FMAP_PER_FILTER - 1) {
                l1BListId = l1BListIdNext;
                cin1FilterActual = cin1FilterActualNext;
            }
        }

        // copy block out
        uint32_t cout1Actual = coutRound / ELE_NUM_A_PER_C0;
        LayoutOutput layoutBlock = layoutOutput.GetTileLayout(
            MakeCoord((uint32_t)1, cout1Actual, hoActual, woActual, ELE_NUM_A_PER_C0));

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            copyL0CToGm(gmOutput, l0CTensor, layoutBlock);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            copyL0CToGm(gmOutput, l0CTensor, layoutBlock, 0b11);
        }
    }

protected:
    Conv2dFilterParams filterParams;
    uint32_t hiBlock, wiBlock;
    uint32_t l1A_size, l1B_size;

    // Multi-stage tensors list
    AscendC::LocalTensor<ElementFmap> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementFilter> l1BTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementFmap> l0ATensorList[L0A_STAGES];
    AscendC::LocalTensor<ElementFilter> l0BTensorList[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    // Multi-stage event id list
    int32_t l1AEventList[L1A_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    int32_t l0BEventList[L0B_STAGES];

    // The id of current stage
    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

} // namespace Catlass::Conv::Block

#endif // CATLASS_CONV_BLOCK_BLOCK_CONV2D_PINGPONG_HPP
