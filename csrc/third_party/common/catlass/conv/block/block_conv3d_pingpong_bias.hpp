/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_BLOCK_BLOCK_CONV3D_PINGPONG_BIAS_HPP
#define CATLASS_CONV_BLOCK_BLOCK_CONV3D_PINGPONG_BIAS_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/conv_coord.hpp"
#include "catlass/conv/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace Catlass::Conv::Block {

template <
    uint32_t L1A_STAGES_,
    uint32_t L1B_STAGES_,
    uint32_t L0A_STAGES_,
    uint32_t L0B_STAGES_,
    uint32_t L0C_STAGES_,
    bool ENABLE_UNIT_FLAG_,
    class CoreTileShape_,
    class FmapL1TileShape_,
    class FilterL1TileShape_,
    class L0TileShape_,
    class FmapType_,
    class FilterType_,
    class OutType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockConv <
    ConvAtlasA2Pingpong<
        L1A_STAGES_,
        L1B_STAGES_,
        L0A_STAGES_,
        L0B_STAGES_,
        L0C_STAGES_,
        ENABLE_UNIT_FLAG_
    >,
    CoreTileShape_,
    FmapL1TileShape_,
    FilterL1TileShape_,
    L0TileShape_,
    FmapType_,
    FilterType_,
    OutType_,
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
    using CoreTileShape = CoreTileShape_;
    using FmapL1TileShape = FmapL1TileShape_;
    using FilterL1TileShape = FilterL1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementFmap = typename FmapType_::Element;
    using LayoutFmap = typename FmapType_::Layout;
    using ElementFilter = typename FilterType_::Element;
    using LayoutFilter = typename FilterType_::Layout;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementFmap, ElementFilter>::ElementAccumulator;
    using ElementOut = typename OutType_::Element;
    using LayoutOut = typename OutType_::Layout;
    using ElementBias = typename BiasType_::Element;
    using LayoutBias = typename BiasType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyGmToL1Bias = typename TileCopy_::CopyGmToL1Bias;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using CopyL1ToBT = typename TileCopy_::CopyL1ToBT;

    using LayoutFmapInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutFilterInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t L1A_STAGES = DispatchPolicy::L1A_STAGES;
    static constexpr uint32_t L1B_STAGES = DispatchPolicy::L1B_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0A_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0B_STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = ArchTag::L0C_SIZE / L0C_STAGES;
    static constexpr uint32_t C0_SIZE = 32;
    static constexpr uint32_t BLOCK_L0_M = 16;
    static constexpr uint32_t BLOCK_L0_N = 16;
    static constexpr uint32_t RIGHT_MOVE_8 = 8;
    static constexpr uint32_t PAD_SIZE = 4;
    static constexpr uint32_t PAD_IDX_T = 2;
    static constexpr uint32_t PAD_IDX_B = 3;
    static constexpr uint32_t PAD_IDX_L = 0;
    static constexpr uint32_t PAD_IDX_R = 1;
    static constexpr uint32_t BLOCK_SIZE = 512;

    // Check L0TileShape
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::mL0 * L0TileShape::kL0 * sizeof(ElementFmap);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::kL0 * L0TileShape::nL0 * sizeof(ElementFilter);
    static constexpr uint32_t L0C_TILE_SIZE = L0TileShape::mL0 * L0TileShape::nL0 * sizeof(ElementAccumulator);
    static constexpr uint32_t BIAS_BUF_SIZE = L0TileShape::nL0 * sizeof(ElementAccumulator);
    static_assert((L0A_TILE_SIZE * L0A_STAGES) <= ArchTag::L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert((L0B_TILE_SIZE * L0B_STAGES) <= ArchTag::L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L0C_TILE_SIZE * L0C_STAGES <= ArchTag::L0C_SIZE, "L0TileShape exceeding the L0C space!");
    static_assert(BIAS_BUF_SIZE <= ArchTag::BIAS_SIZE, "BIAS_BUF_SIZE exceeding the BT space! Reduce L0TileShape::nL0");

    // Check PingPong
    static_assert(L0C_STAGES == 1, "L0C PingPong must be 1!");
    static_assert(L0A_STAGES == 2, "L0A PingPong must be 2!");
    static_assert(L0B_STAGES == 2, "L0B PingPong must be 2!");

    ///// Construct 进行initBuffer
    CATLASS_DEVICE
    BlockConv(Arch::Resource<ArchTag> &resource, Conv3dParams const &conv3dParams_, uint32_t l1BufAddrStart = 0)
        : conv3dParams(conv3dParams_)
    {
        copyL1ToL0A = CopyL1ToL0A::MakeCopyL1ToL0A(conv3dParams.sW(), conv3dParams.sH(),
                                                   conv3dParams.kw(), conv3dParams.kh(),
                                                   conv3dParams.dW(), conv3dParams.dH());
        uint64_t bl1Spacesize = FilterL1TileShape::Kd * FilterL1TileShape::Ci1 *
                                conv3dParams.khkwcin0() * FilterL1TileShape::nBL1 * sizeof(ElementFilter);
        uint64_t hoAL1Max = FmapL1TileShape::mAL1  / conv3dParams.wo() + 2;
        uint64_t hiAL1Max = (hoAL1Max - 1) * conv3dParams.sH() + conv3dParams.dilatedKernelH();
        hiAL1Max = hiAL1Max > conv3dParams.hi() ? conv3dParams.hi() : hiAL1Max;
        uint64_t al1Spacesize = FmapL1TileShape::Kd * FmapL1TileShape::Ci1 * hiAL1Max *
                                conv3dParams.wicin0()  * sizeof(ElementFmap);

        for (uint32_t i = 0; i < L0A_STAGES; i++) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementFmap>(L0A_PINGPONG_BUF_SIZE * i);
            l0AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; i++) {
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementFilter>(L0B_PINGPONG_BUF_SIZE * i);
            l0BEventList[i] = i + L0A_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; i++) {
            l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
            l0CEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }

        uint32_t l1AOffset = l1BufAddrStart;
        for (uint32_t i = 0; i < L1A_STAGES; i++) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementFmap>(l1AOffset + al1Spacesize * i);
            l1AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        uint32_t l1BOffset = l1BufAddrStart + al1Spacesize * L1A_STAGES;
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementFilter>(l1BOffset + bl1Spacesize * i);
            l1BEventList[i] = i + L1A_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        uint32_t l1BiasOffset = l1BufAddrStart + al1Spacesize * L1A_STAGES + bl1Spacesize * L1B_STAGES;
        l1BiasTensor = resource.l1Buf.template GetBufferByByte<ElementBias>(l1BiasOffset);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
        l0BiasTensor = resource.btBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
    }

    /// Destructor
    CATLASS_DEVICE
    ~BlockConv()
    {
        for (uint32_t i = 0; i < L0A_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
        for (uint32_t i = 0; i < L1A_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
        }
        for (uint32_t i = 0; i < L1B_STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
    }

    // Perform a block-scoped conv3d
    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementFmap> const &fmapGm, LayoutFmap const &layoutFmap,
        AscendC::GlobalTensor<ElementFilter> const &filterGm, LayoutFilter const &layoutFilter,
        AscendC::GlobalTensor<ElementOut> const &outGm, LayoutFmap const &layoutOut,
        AscendC::GlobalTensor<ElementBias> const &biasGm,
        Conv3d6HdCoord const &actualBlockShape, Conv3d6HdCoord const &actualIdxStartFmap)
    {
        // Initialization of the loop parameter in the K direction
        InitKLoopParams();
        // Loop parameters in the M/Cout/D direction
        InitOutLoopParams(actualBlockShape);
        
        // The starting position of the input
        iterParams.diStartPos = actualIdxStartFmap.d();
        iterParams.hwStartPos = actualIdxStartFmap.hw();
        
        // Start the batch iterate
        for (uint32_t batchIter = 0; batchIter < actualBlockShape.n(); ++batchIter) {
            auto gmBatchFmap = fmapGm[batchIter * conv3dParams.fmapOneBatchSize()];
            auto gmBatchOut = outGm[batchIter * conv3dParams.outputOneBatchSize()];
            while (true) {
                // The parameters used need to be reinitialized in the first iteration
                if (iterParams.isFirstIterate) {
                    iterParams.nBL0Iter = 0;
                    iterParams.mAL0Iter = 0;
                    iterParams.mAL1Iter = 0;
                    iterParams.nBL1Iter = 0;
                    iterParams.dOutIter = 0;
                    iterParams.loadAL1Flag = true;
                    iterParams.loadBL1Flag = true;
                    iterParams.loadAL0Flag = true;
                    iterParams.loadBL0Flag = true;
                    iterParams.isFirstIterate = false;
                    if (L0TileShape::mL0 % conv3dParams.wo() == 0) {
                        iterParams.mL0IsDivisibleByWo = true;
                    }
                } else if (IterateNFirst() == false) {
                    break;
                }
                // Refresh the cycle round
                iterParams.l12l0LoopM = iterParams.mAL1Iter == iterParams.maxMAL1Iter
                    ? CeilDiv(iterParams.mAL1Tail, L0TileShape::mL0)
                    : iterParams.mAL1DivmL0;
                iterParams.maxML0Iter = iterParams.l12l0LoopM - 1;
                iterParams.l12l0LoopN = iterParams.nBL1Iter == iterParams.maxNBL1Iter
                    ? CeilDiv(iterParams.nBL1Tail, L0TileShape::nL0)
                    : iterParams.nBL1DivnL0;
                iterParams.maxNL0Iter = iterParams.l12l0LoopN - 1;
                // Start the K-axis iterate
                uint32_t n = (iterParams.nBL1Iter == iterParams.maxNBL1Iter && iterParams.nBL0Iter == iterParams.maxNL0Iter) ? iterParams.nL0Tail : L0TileShape::nL0;
                uint32_t m = (iterParams.mAL1Iter == iterParams.maxMAL1Iter && iterParams.mAL0Iter == iterParams.maxML0Iter) ? iterParams.mAL0Tail : L0TileShape::mL0;
                
                tileParams.l0CurrentM = CeilDiv(m, BLOCK_L0_M) * BLOCK_L0_M;
                tileParams.l0CurrentN = CeilDiv(n, BLOCK_L0_N) * BLOCK_L0_N;

                uint32_t biasGmOffset = iterParams.nBL1Iter * FilterL1TileShape::nBL1 + iterParams.nBL0Iter * L0TileShape::nL0;

                auto layoutTileBias = layout::VectorLayout(actualBlockShape.c1() * conv3dParams.n0());
                auto layoutBiasInL1 = layout::VectorLayout(tileParams.l0CurrentN);
                auto l0BiasTile = l0BiasTensor;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);
                copyGmToL1Bias(l1BiasTensor, biasGm[biasGmOffset], layoutBiasInL1, layoutTileBias);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1A_STAGES + L1B_STAGES);
                auto layoutBiasInL0 = layout::VectorLayout(tileParams.l0CurrentN);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                copyL1ToBT(l0BiasTile, l1BiasTensor, layoutBiasInL0, layoutBiasInL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0A_STAGES + L0B_STAGES);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0A_STAGES + L0B_STAGES);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1A_STAGES + L1B_STAGES);

                if constexpr (!ENABLE_UNIT_FLAG) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                }

                if (iterParams.preloadABL1DbFlag) {
                    ReduceKPreloadDbAll(gmBatchFmap, layoutFmap, filterGm, layoutFilter);
                } else if (iterParams.preloadAL1DbFlag) {
                    ReduceKPreloadDbFmap(gmBatchFmap, layoutFmap, filterGm, layoutFilter);
                } else {
                    ReduceK(gmBatchFmap, layoutFmap, filterGm, layoutFilter);
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0A_STAGES + L0B_STAGES);
                iterParams.kIter = 0;
                auto layoutCInL0 = LayoutCInL0::MakeLayoutInL0C(MakeCoord(m, CeilDiv(n, conv3dParams.cout0()) * conv3dParams.cout0()));
                LayoutFmap layoutOutGm = layoutOut.GetTileLayout(MakeCoord((uint32_t)1, conv3dParams.dout(), conv3dParams.cout1(), conv3dParams.ho(), conv3dParams.wo(), conv3dParams.cout0()));
                uint32_t cout1L1Idx = (FilterL1TileShape::nBL1 * iterParams.nBL1Iter + L0TileShape::nL0 * iterParams.nBL0Iter) / conv3dParams.cout0();
                uint32_t howoIdx = FmapL1TileShape::mAL1 * iterParams.mAL1Iter + L0TileShape::mL0 * iterParams.mAL0Iter;
                Conv3d6HdCoord gmTileOutOffset{0, iterParams.dOutIter, cout1L1Idx, howoIdx};
                auto gmTileOut = gmBatchOut[layoutOut.GetOffset(gmTileOutOffset)];
                if constexpr (!ENABLE_UNIT_FLAG) {
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                    copyL0CToGm(gmTileOut, l0CTensorList[l0CListId], layoutOutGm, layoutCInL0);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
                } else {
                    copyL0CToGm(gmTileOut, l0CTensorList[l0CListId], layoutOutGm, layoutCInL0, 0b11);
                }
            }
            iterParams.isFirstIterate = true;
            iterParams.nBL0Iter = 0;
            iterParams.nBL1Iter = 0;
        }
    }
protected:
    struct IterParams {
        uint8_t isFirstIterate = true;
        uint8_t loadAL1Flag = true;
        uint8_t loadBL1Flag = true;
        uint8_t loadAL0Flag = true;
        uint8_t loadBL0Flag = true;
        uint8_t kAL1fullload = false;
        uint8_t kBL1fullload = false;
        uint8_t biasFullLoadFlag = false;
        uint8_t preloadAL1DbFlag = false;
        uint8_t preloadABL1DbFlag = false;
        uint8_t mL0IsDivisibleByWo = false;

        uint8_t isGroupOptDimTail = false;
        
        uint32_t kAL1Iter = 0;
        uint32_t kBL1Iter = 0;
        uint32_t mAL1Iter = 0;
        uint32_t nBL1Iter = 0;
        uint32_t dOutIter = 0;
        uint32_t kIter = 0;
        uint32_t kAL0Iter = 0;
        uint32_t kBL0Iter = 0;
        uint32_t mAL0Iter = 0;
        uint32_t nBL0Iter = 0;
        uint32_t groupOptIter = 0;

        uint32_t maxKAL1Iter = 0;
        uint32_t maxMAL1Iter = 0;
        uint32_t maxNBL1Iter = 0;
        uint32_t maxKBL1Iter = 0;
        uint32_t maxNL0Iter = 0;
        uint32_t maxML0Iter = 0;
        uint32_t maxKL0Iter = 0;
        uint32_t maxDOutIter = 0;
        uint32_t maxGroupOptIter = 0;

        uint32_t ddr2l1LoopM = 0;
        uint32_t ddr2l1LoopN = 0;
        uint32_t l12l0LoopN = 0;
        uint32_t ddr2l1LoopD = 0;
        uint32_t l12l0LoopM = 0;
        uint32_t ddr2l0LoopK = 0;
        
        uint32_t kL0Tail = 0;
        uint32_t kAL1Tail = 0;
        uint32_t kBL1Tail = 0;
        uint32_t mAL1Tail = 0;
        uint32_t mAL0Tail = 0;
        uint32_t nL0Tail = 0;
        uint32_t nBL1Tail = 0;
        uint32_t multiKAL1 = 1;
        uint32_t multiKBL1 = 1;

        uint32_t hwStartPos = 0;
        uint32_t diStartPos = 0;

        uint32_t orgCoAlignK0 = 0;
        uint32_t orgCoAlignN0 = 0;
        uint32_t nBL1TailAlign = 0;
        uint32_t mAL1DivmL0 = 0;
        uint32_t nBL1DivnL0 = 0;

        bool aL1IsFullPad = false;

        CATLASS_DEVICE
        IterParams() = default;
    };

    struct TileParams {
        uint32_t l0CurrentM = 0;
        uint32_t l0CurrentN = 0;
        uint32_t l0NumN = 0;

        CATLASS_DEVICE
        TileParams() = default;
    };

    Conv3dParams conv3dParams;
    IterParams iterParams;
    TileParams tileParams;

    AscendC::LocalTensor<ElementFmap> l1ATensorList[L1A_STAGES];
    AscendC::LocalTensor<ElementFilter> l1BTensorList[L1B_STAGES];
    AscendC::LocalTensor<ElementFmap> l0ATensorList[L0A_STAGES];
    AscendC::LocalTensor<ElementFilter> l0BTensorList[L0B_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES];
    AscendC::LocalTensor<ElementBias> l1BiasTensor;
    AscendC::LocalTensor<ElementAccumulator> l0BiasTensor;

    LayoutFmapInL1 layoutFmapInL1;
    LayoutFilterInL1 layoutFilterInL1;

    // Multi-stage event id list
    int32_t l1AEventList[L1A_STAGES];
    int32_t l1BEventList[L1B_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    int32_t l0CEventList[L0C_STAGES];

    // The id of current stage
    uint32_t l1AListId{0};
    uint32_t l1BListId{0};
    uint32_t l0CListId{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyGmToL1Bias copyGmToL1Bias;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL1ToBT copyL1ToBT;
    CopyL0CToGm copyL0CToGm;

    CATLASS_DEVICE
    void InitKLoopParams()
    {
        // Initialization of the loop parameter in the K direction
        iterParams.ddr2l0LoopK = CeilDiv(conv3dParams.alignCinKhKwKd(), L0TileShape::kL0);
        iterParams.maxKL0Iter = iterParams.ddr2l0LoopK - 1;
        iterParams.kL0Tail = conv3dParams.alignCinKhKwKd() % L0TileShape::kL0;
        iterParams.kL0Tail = iterParams.kL0Tail == 0 ? L0TileShape::kL0 : iterParams.kL0Tail;

        // The k-axis loop iteration parameters of the B-matrix
        iterParams.maxKBL1Iter = CeilDiv(conv3dParams.kdcin1(), FilterL1TileShape::Kd * FilterL1TileShape::Ci1) - 1;
        iterParams.multiKBL1 = CeilDiv(FilterL1TileShape::Kd * FilterL1TileShape::Ci1 * conv3dParams.khkwcin0(), L0TileShape::kL0);
        iterParams.kBL1fullload = conv3dParams.kdcin1() == FilterL1TileShape::Kd * FilterL1TileShape::Ci1;
        uint32_t kBL1TailCheck = conv3dParams.alignCinKhKwKd() % (FilterL1TileShape::Kd * FilterL1TileShape::Ci1 * conv3dParams.khkwcin0());
        iterParams.kBL1Tail = kBL1TailCheck == 0 ? FilterL1TileShape::Kd * FilterL1TileShape::Ci1 * conv3dParams.khkwcin0() : kBL1TailCheck;

        // The k-axis loop iteration parameters of matrix A
        uint32_t kAL1 = FmapL1TileShape::Kd * FmapL1TileShape::Ci1 * conv3dParams.khkwcin0();
        iterParams.maxKAL1Iter = CeilDiv(conv3dParams.kdcin1(), FmapL1TileShape::Kd * FmapL1TileShape::Ci1) - 1;
        iterParams.multiKAL1 = CeilDiv(kAL1, L0TileShape::kL0);
        iterParams.kAL1fullload = conv3dParams.kdcin1() == FmapL1TileShape::Kd * FmapL1TileShape::Ci1;
        uint32_t kAL1TailCheck = conv3dParams.alignCinKhKwKd() % kAL1;
        iterParams.kAL1Tail = kAL1TailCheck == 0 ? kAL1 : kAL1TailCheck;
        uint8_t L0ASet2dFlag = conv3dParams.padhead() != 0 || conv3dParams.padtail() != 0 ||
                conv3dParams.padtop() >= conv3dParams.kh() || conv3dParams.padbottom() >= conv3dParams.kh();
        bool kAL1TailCase = iterParams.kAL1Tail != kAL1;
        iterParams.preloadAL1DbFlag = L1A_STAGES == 2 && !iterParams.kAL1fullload && !L0ASet2dFlag && !kAL1TailCase;
        iterParams.preloadABL1DbFlag = L1A_STAGES == 2 && L1B_STAGES == 2 && !iterParams.kAL1fullload &&
                                         !iterParams.kBL1fullload && !L0ASet2dFlag && !kAL1TailCase;
    }

    CATLASS_DEVICE
    void InitOutLoopParams(Conv3d6HdCoord const &actualBlockShape)
    {
        // Loop parameters in the M direction
        iterParams.mAL1Tail = actualBlockShape.hw() % FmapL1TileShape::mAL1;
        iterParams.mAL1Tail = iterParams.mAL1Tail == 0 ? FmapL1TileShape::mAL1 : iterParams.mAL1Tail;
        iterParams.mAL1DivmL0 = CeilDiv(FmapL1TileShape::mAL1, L0TileShape::mL0);
        iterParams.ddr2l1LoopM = CeilDiv(actualBlockShape.hw(), FmapL1TileShape::mAL1);
        iterParams.maxMAL1Iter = iterParams.ddr2l1LoopM - 1;
        iterParams.mAL0Tail = iterParams.mAL1Tail % L0TileShape::mL0;
        iterParams.mAL0Tail = iterParams.mAL0Tail == 0 ? L0TileShape::mL0 : iterParams.mAL0Tail;
        iterParams.l12l0LoopM = iterParams.mAL1DivmL0;
        iterParams.maxML0Iter = iterParams.l12l0LoopM - 1;

        // Loop parameters in the Cout direction
        iterParams.maxNBL1Iter = CeilDiv(actualBlockShape.c1() * conv3dParams.cout0(), FilterL1TileShape::nBL1) - 1;
        iterParams.nBL1Tail = (actualBlockShape.c1() * conv3dParams.cout0()) % FilterL1TileShape::nBL1;
        iterParams.nBL1Tail = iterParams.nBL1Tail == 0 ? FilterL1TileShape::nBL1 : iterParams.nBL1Tail;
        iterParams.nBL1DivnL0 = CeilDiv(FilterL1TileShape::nBL1, L0TileShape::nL0);
        iterParams.nBL1TailAlign = CeilDiv(iterParams.nBL1Tail, BLOCK_L0_N) * BLOCK_L0_N;
        iterParams.nL0Tail = iterParams.nBL1Tail % L0TileShape::nL0;
        iterParams.nL0Tail = iterParams.nL0Tail == 0 ? L0TileShape::nL0 : iterParams.nL0Tail;
        iterParams.ddr2l1LoopN = iterParams.maxNBL1Iter + 1;
        iterParams.l12l0LoopN = iterParams.nBL1DivnL0;
        iterParams.maxNL0Iter = iterParams.l12l0LoopN - 1;

        // Loop parameter in the D direction
        iterParams.ddr2l1LoopD = actualBlockShape.d();
    }

    CATLASS_DEVICE
    bool IterateNFirst()
    {
        // From N to M
        iterParams.nBL0Iter++;
        if (iterParams.nBL0Iter == iterParams.l12l0LoopN) {
            iterParams.nBL0Iter = 0;
            iterParams.mAL0Iter++;
        }
        if (iterParams.mAL0Iter == iterParams.l12l0LoopM) {
            iterParams.mAL0Iter = 0;
            iterParams.nBL1Iter++;
            iterParams.loadBL1Flag = true;
        }
        if (iterParams.nBL1Iter == iterParams.ddr2l1LoopN) {
            iterParams.nBL1Iter = 0;
            iterParams.mAL1Iter++;
            iterParams.loadAL1Flag = true;
        }
        if (iterParams.mAL1Iter == iterParams.ddr2l1LoopM) {
            iterParams.mAL1Iter = 0;
            iterParams.dOutIter++;
        }
        if (iterParams.dOutIter == iterParams.ddr2l1LoopD) {
            return false;
        }
        return true;
    }

    CATLASS_DEVICE
    void ReduceK(AscendC::GlobalTensor<ElementFmap> const &gmBatchFmap, LayoutFmap const &layoutFmap,
                 AscendC::GlobalTensor<ElementFilter> const &filterGm, LayoutFilter const &layoutFilter)
    {
        iterParams.kIter = 0;
        uint16_t isOdd = 0;
        while (iterParams.kIter < iterParams.ddr2l0LoopK) {
            if (iterParams.loadAL1Flag || (!iterParams.kAL1fullload && iterParams.kIter % iterParams.multiKAL1 == 0)) {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
                LoadAL1Process(gmBatchFmap, iterParams.kIter / iterParams.multiKAL1, layoutFmap, l1AListId);
                iterParams.loadAL1Flag = false;
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
            }
            if (iterParams.loadBL1Flag || (!iterParams.kBL1fullload && iterParams.kIter % iterParams.multiKBL1 == 0)) {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
                LoadBL1Process(filterGm, iterParams.kIter / iterParams.multiKBL1, layoutFilter, l1BListId);
                iterParams.loadBL1Flag = false;
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
            }
            ReduceKL0AL0BPingPong(isOdd, l1AListId, l1BListId);
            iterParams.kIter++;
            isOdd = iterParams.kIter & 0x1;
        }
    }

    CATLASS_DEVICE
    void ReduceKPreloadDbFmap(AscendC::GlobalTensor<ElementFmap> const &gmBatchFmap, LayoutFmap const &layoutFmap,
                             AscendC::GlobalTensor<ElementFilter> const &filterGm, LayoutFilter const &layoutFilter)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
        LoadAL1Process(gmBatchFmap, 0, layoutFmap, l1AListId);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);

        if (iterParams.loadBL1Flag || !(iterParams.kBL1fullload)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
            LoadBL1Process(filterGm, 0, layoutFilter, l1BListId);
            iterParams.loadBL1Flag = false;
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
        }

        ReduceKL0AL0BPingPong(event_t::EVENT_ID0, l1AListId, l1BListId);

        uint32_t l1AListIdNext = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
        LoadAL1Process(gmBatchFmap, 1, layoutFmap, l1AListIdNext);
        iterParams.loadAL1Flag = false;
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListIdNext]);

        iterParams.kIter = 1;
        uint16_t isOdd = 1;
        uint64_t maxKAL1PreloadIter = iterParams.ddr2l0LoopK - iterParams.multiKAL1;
        while (iterParams.kIter < iterParams.ddr2l0LoopK) {
            if (iterParams.kIter == maxKAL1PreloadIter) {
                l1AListId = l1AListIdNext;
                l1AListIdNext = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
            } else if (iterParams.kIter < maxKAL1PreloadIter && (!iterParams.kAL1fullload && iterParams.kIter % iterParams.multiKAL1 == 0)) {
                l1AListId = l1AListIdNext;
                l1AListIdNext = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
                LoadAL1Process(gmBatchFmap, (iterParams.kIter / iterParams.multiKAL1) + 1, layoutFmap, l1AListIdNext);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListIdNext]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
            }

            if (iterParams.loadBL1Flag || (!iterParams.kBL1fullload && iterParams.kIter % iterParams.multiKBL1 == 0)) {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
                LoadBL1Process(filterGm, iterParams.kIter / iterParams.multiKBL1, layoutFilter, l1BListId);
                iterParams.loadBL1Flag = false;
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
            }
            ReduceKL0AL0BPingPong(isOdd, l1AListId, l1BListId);
            iterParams.kIter++;
            isOdd = iterParams.kIter & 0x1;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
    }

    CATLASS_DEVICE
    void ReduceKPreloadDbAll(AscendC::GlobalTensor<ElementFmap> const &gmBatchFmap, LayoutFmap const &layoutFmap,
                             AscendC::GlobalTensor<ElementFilter> const &filterGm, LayoutFilter const &layoutFilter)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
        LoadAL1Process(gmBatchFmap, 0, layoutFmap, l1AListId);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
        LoadBL1Process(filterGm, 0, layoutFilter, l1BListId);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);

        uint32_t l1AListIdNext = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
        uint32_t l1BListIdNext = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
        LoadAL1Process(gmBatchFmap, 1, layoutFmap, l1AListIdNext);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListIdNext]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
        LoadBL1Process(filterGm, 1, layoutFilter, l1BListIdNext);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListIdNext]);

        ReduceKL0AL0BPingPong(event_t::EVENT_ID0, l1AListId, l1BListId);

        iterParams.kIter = 1;
        uint16_t isOdd = 1;
        uint64_t maxKAL1PreloadIter = iterParams.ddr2l0LoopK - iterParams.multiKAL1;
        uint64_t maxKBL1PreloadIter = iterParams.ddr2l0LoopK - iterParams.multiKBL1;
        while (iterParams.kIter < iterParams.ddr2l0LoopK) {
            if (iterParams.kIter == maxKAL1PreloadIter) {
                l1AListId = l1AListIdNext;
                l1AListIdNext = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
            } else if (iterParams.kIter < maxKAL1PreloadIter && (!iterParams.kAL1fullload && iterParams.kIter % iterParams.multiKAL1 == 0)) {
                l1AListId = l1AListIdNext;
                l1AListIdNext = (l1AListId + 1 < L1A_STAGES) ? (l1AListId + 1) : 0;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListIdNext]);
                LoadAL1Process(gmBatchFmap, (iterParams.kIter / iterParams.multiKAL1) + 1, layoutFmap, l1AListIdNext);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListIdNext]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1AListId]);
            }

            if (iterParams.kIter == maxKBL1PreloadIter) {
                l1BListId = l1BListIdNext;
                l1BListIdNext = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
            } else if (iterParams.kIter < maxKBL1PreloadIter && (!iterParams.kBL1fullload && iterParams.kIter % iterParams.multiKBL1 == 0)) {
                l1BListId = l1BListIdNext;
                l1BListIdNext = (l1BListId + 1 < L1B_STAGES) ? (l1BListId + 1) : 0;
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListIdNext]);
                LoadBL1Process(filterGm, (iterParams.kIter / iterParams.multiKBL1) + 1, layoutFilter, l1BListIdNext);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListIdNext]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1BListId]);
            }
            ReduceKL0AL0BPingPong(isOdd, l1AListId, l1BListId);
            iterParams.kIter++;
            isOdd = iterParams.kIter & 0x1;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1AListId]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1BListId]);
    }

    CATLASS_DEVICE
    void LoadAL1Process(AscendC::GlobalTensor<ElementFmap> const &gmBatchFmap, uint32_t kAL1Iter, LayoutFmap const &layoutFmap, uint32_t l1ListId)
    {
        iterParams.kAL1Iter = kAL1Iter;
        uint32_t currentML1 = iterParams.mAL1Iter == iterParams.maxMAL1Iter ? iterParams.mAL1Tail : FmapL1TileShape::mAL1;
        uint32_t currentM = iterParams.hwStartPos + iterParams.mAL1Iter * FmapL1TileShape::mAL1;
        uint32_t hoStartIdx = currentM / conv3dParams.wo();
        uint32_t hoEndIdx = CeilDiv(currentM + currentML1, conv3dParams.wo());
        uint32_t orgHiLoadL1 = ((hoEndIdx - hoStartIdx) - 1) * conv3dParams.sH() + conv3dParams.dilatedKernelH();
        uint32_t tmpCurCoreHiStartIdx = (iterParams.hwStartPos / conv3dParams.wo()) * conv3dParams.sH();
        uint32_t curCoreHiStartIdx = tmpCurCoreHiStartIdx <= conv3dParams.padtop() ? 0 : tmpCurCoreHiStartIdx - conv3dParams.padtop();
        uint32_t currentCin1LoadL1 = iterParams.kAL1Iter * FmapL1TileShape::Kd * FmapL1TileShape::Ci1;
        uint32_t kAL1Tmp = iterParams.kAL1Iter == iterParams.maxKAL1Iter ? iterParams.kAL1Tail : FmapL1TileShape::Kd * FmapL1TileShape::Ci1 * conv3dParams.khkwcin0();
        uint32_t orgCin1LoadL1 = kAL1Tmp / conv3dParams.kh() / conv3dParams.kw() / conv3dParams.cin0();
        uint32_t kdL1Idx = (currentCin1LoadL1 / conv3dParams.cin1()) % conv3dParams.kd();
        uint32_t cin1L1Idx = currentCin1LoadL1 % conv3dParams.cin1();
        uint32_t padTopL1 = 0;
        uint32_t padBottomL1 = 0;
        iterParams.aL1IsFullPad = false;
        bool set2dFlagDHead = false;
        bool set2dFlagDTail = false;
        uint32_t hiLoadL1 = orgHiLoadL1;
        uint32_t cin1LoadL1 = orgCin1LoadL1;

        uint32_t hiStartIdxWithPad = hoStartIdx * conv3dParams.sH();
        uint32_t hiEndIdxWithPad = hiStartIdxWithPad + hiLoadL1;
        uint32_t hiIdx = hiStartIdxWithPad - conv3dParams.padtop() - curCoreHiStartIdx;
        uint32_t hiWithPad = conv3dParams.hi() + conv3dParams.padtop();
        if (hiEndIdxWithPad <= conv3dParams.padtop()) {
            iterParams.aL1IsFullPad = true;
        } else if (hiStartIdxWithPad < conv3dParams.padtop()) {
            hiIdx = 0;
            hiLoadL1 = hiLoadL1 + hiStartIdxWithPad - conv3dParams.padtop();
            padTopL1 = conv3dParams.padtop() - hiStartIdxWithPad;
            if (hiEndIdxWithPad >= hiWithPad) {
                hiLoadL1 = conv3dParams.hi() - hiIdx;
                padBottomL1 = hiEndIdxWithPad - hiWithPad;
            }
        } else if (hiStartIdxWithPad >= hiWithPad) {
            iterParams.aL1IsFullPad = true;
        } else if (hiEndIdxWithPad > hiWithPad) {
            hiLoadL1 = hiWithPad - hiStartIdxWithPad;
            padBottomL1 = hiEndIdxWithPad - hiWithPad;
        }

        uint32_t diStartWithPad = iterParams.diStartPos + iterParams.dOutIter * conv3dParams.sD() + kdL1Idx * conv3dParams.dD();
        uint32_t diEndWithPad = cin1LoadL1 <= conv3dParams.cin1()
                                    ? diStartWithPad + 1
                                    : diStartWithPad + (cin1LoadL1 / conv3dParams.cin1() - 1) * conv3dParams.dD() + 1;
        uint32_t diIdx = iterParams.diStartPos <= conv3dParams.padhead() ? diStartWithPad - conv3dParams.padhead()
                                                            : diStartWithPad - iterParams.diStartPos;
        uint32_t diWithPad = conv3dParams.di() + conv3dParams.padhead();
        uint32_t cin1LoadL1PadHead = 0;
        uint32_t cin1LoadL1PadTail = 0;
        if (diEndWithPad <= conv3dParams.padhead()) {
            iterParams.aL1IsFullPad = true;
        } else if (diStartWithPad < conv3dParams.padhead()) {
            set2dFlagDHead = true;
            uint32_t kdTmp = CeilDiv((conv3dParams.padhead() - diStartWithPad), conv3dParams.dD());
            cin1LoadL1PadHead = kdTmp * conv3dParams.cin1();
            diIdx = conv3dParams.dD() == 1 ? 0 : kdTmp * conv3dParams.dD() - conv3dParams.padhead() + diStartWithPad;
            cin1LoadL1 -= cin1LoadL1PadHead;

            if (diEndWithPad > diWithPad) {
                set2dFlagDTail = true;
                kdTmp = CeilDiv((conv3dParams.di() - diIdx), conv3dParams.dD());
                cin1LoadL1PadTail = cin1LoadL1 - kdTmp * conv3dParams.cin1();
                cin1LoadL1 = kdTmp * conv3dParams.cin1();
            }
        } else if (diStartWithPad >= diWithPad) {
            iterParams.aL1IsFullPad = true;
        } else if (diEndWithPad > diWithPad) {
            set2dFlagDTail = true;
            uint32_t kdTmp = CeilDiv((diWithPad - diStartWithPad), conv3dParams.dD());
            cin1LoadL1PadTail = cin1LoadL1 - kdTmp * conv3dParams.cin1();
            cin1LoadL1 = kdTmp * conv3dParams.cin1();
        }
        if (!iterParams.aL1IsFullPad) {
            uint8_t padList[PAD_SIZE] = {0};
            padList[PAD_IDX_L] = conv3dParams.padleft();
            padList[PAD_IDX_R] = conv3dParams.padright();
            padList[PAD_IDX_T] = padTopL1;
            padList[PAD_IDX_B] = padBottomL1;
            SetFmatrix(hiLoadL1, conv3dParams.wi(), padList, AscendC::FmatrixMode::FMATRIX_LEFT);

            uint64_t aL1Offset = 0;
            if (set2dFlagDHead) {
                AscendC::InitConstValueParams<ElementFmap> initConstValueParams;
                initConstValueParams.repeatTimes = cin1LoadL1PadHead / conv3dParams.cin1();
                initConstValueParams.blockNum = conv3dParams.cin1() * hiLoadL1 * conv3dParams.wi();
                initConstValueParams.dstGap = 0;
                initConstValueParams.initValue = 0;
                InitConstValue(l1ATensorList[l1ListId], initConstValueParams);
                aL1Offset += cin1LoadL1PadHead * hiLoadL1 * conv3dParams.wicin0();
                set2dFlagDHead = false;
            }

            Conv3d6HdCoord gmTileFmapOffset{0, diIdx, cin1L1Idx, hiIdx * conv3dParams.wi()};
            auto layoutTileFmap = layoutFmap.GetTileLayout(MakeCoord((uint32_t)1, conv3dParams.dD(), conv3dParams.cin1(), conv3dParams.hi(), conv3dParams.wi(), conv3dParams.cin0()));
            auto gmTileFmap = gmBatchFmap[layoutTileFmap.GetOffset(gmTileFmapOffset)];
            layoutFmapInL1 = LayoutFmapInL1::MakeLayout(1, 1, cin1LoadL1, hiLoadL1, conv3dParams.wi() ,conv3dParams.cin0());

            copyGmToL1A(l1ATensorList[l1ListId][aL1Offset], gmTileFmap, layoutFmapInL1, layoutTileFmap);

            if (set2dFlagDTail) {
                aL1Offset += cin1LoadL1 * hiLoadL1 * conv3dParams.wi()* conv3dParams.cin0();
                AscendC::InitConstValueParams<ElementFmap> initConstValueParams;
                initConstValueParams.repeatTimes = cin1LoadL1PadTail / conv3dParams.cin1();
                initConstValueParams.blockNum = conv3dParams.cin1() * hiLoadL1 * conv3dParams.wi();
                initConstValueParams.dstGap = 0;
                initConstValueParams.initValue = 0;
                InitConstValue(l1ATensorList[l1ListId][aL1Offset], initConstValueParams);
                set2dFlagDTail = false;
            }
        }
        layoutFmapInL1 = LayoutFmapInL1::MakeLayout(1, 1, orgCin1LoadL1, orgHiLoadL1, conv3dParams.wi() ,conv3dParams.cin0());
    }

    CATLASS_DEVICE
    void LoadBL1Process(AscendC::GlobalTensor<ElementFilter> const &filterGm, uint32_t kBL1Iter, LayoutFilter const &layoutFilter, uint32_t l1ListId)
    {
        iterParams.kBL1Iter = kBL1Iter;
        uint32_t currentNBL1 = ((iterParams.nBL1Iter != iterParams.maxNBL1Iter) || (FilterL1TileShape::nBL1 >= conv3dParams.alignCout())) ? FilterL1TileShape::nBL1 : iterParams.nBL1TailAlign;
        uint32_t currentKBL1 = iterParams.kBL1Iter == iterParams.maxKBL1Iter ? iterParams.kBL1Tail : FilterL1TileShape::Kd * FilterL1TileShape::Ci1 * conv3dParams.khkwcin0();
        Conv3dFracZ3dCoord gmTileFilterOffset{iterParams.kBL1Iter * FilterL1TileShape::Kd * FilterL1TileShape::Ci1 * conv3dParams.khkw(), iterParams.nBL1Iter * FilterL1TileShape::nBL1};
        auto layoutTileFilter = layoutFilter;
        auto gmTileFiler = filterGm[layoutTileFilter.GetOffset(gmTileFilterOffset)];
        layoutFilterInL1 = LayoutFilterInL1::template MakeLayout<ElementFilter>(currentKBL1, currentNBL1);
        copyGmToL1B(l1BTensorList[l1ListId], gmTileFiler, layoutFilterInL1, layoutTileFilter);
    }

    CATLASS_DEVICE
    void ReduceKL0AL0BPingPong(const uint16_t& l0abFlag, uint32_t l1aListId, uint32_t l1bListId) {
        auto l0ATile = l0ATensorList[l0abFlag];
        auto l0BTile = l0BTensorList[l0abFlag];
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0abFlag]);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0abFlag]);
        iterParams.kAL0Iter = iterParams.kIter % iterParams.multiKAL1;
        uint32_t currentKL0 = iterParams.kIter == iterParams.maxKL0Iter ? iterParams.kL0Tail : L0TileShape::kL0;
        if (iterParams.aL1IsFullPad) {
            uint32_t al0Set2dSpacesize_ = tileParams.l0CurrentM * currentKL0 * sizeof(ElementFmap) / BLOCK_SIZE;
            AscendC::InitConstValueParams<ElementFmap> initConstValueParams(1, (uint16_t)al0Set2dSpacesize_, 0, 0);
            InitConstValue(l0ATensorList[l0abFlag], initConstValueParams);
        } else {
            uint32_t kStartPt = iterParams.kAL0Iter * L0TileShape::kL0;
            uint32_t mStartPt = iterParams.mL0IsDivisibleByWo ?
                iterParams.mAL0Iter * L0TileShape::mL0 + iterParams.hwStartPos % conv3dParams.wo() :
                iterParams.mAL0Iter * L0TileShape::mL0 + (iterParams.hwStartPos + iterParams.mAL1Iter * FmapL1TileShape::mAL1) % conv3dParams.wo();
            LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementFilter>(tileParams.l0CurrentM, currentKL0);
            copyL1ToL0A(l0ATile, l1ATensorList[l1aListId], layoutAInL0, layoutFmapInL1, kStartPt, mStartPt);
        }
        iterParams.kBL0Iter = iterParams.kIter % iterParams.multiKBL1;
        uint32_t tilingNBSrc_ = (iterParams.nBL1Iter != iterParams.maxNBL1Iter) ? FilterL1TileShape::nBL1 : iterParams.nBL1TailAlign;
        MatrixCoord l1TileFilterOffset{iterParams.kBL0Iter * L0TileShape::kL0, iterParams.nBL0Iter * L0TileShape::nL0};
        auto l1BTile = l1BTensorList[l1bListId][layoutFilterInL1.GetOffset(l1TileFilterOffset)];
        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementFilter>(currentKL0, tileParams.l0CurrentN);
        copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, layoutFilterInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0abFlag]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0abFlag]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventList[l0abFlag]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventList[l0abFlag]);
        auto l0CTile = l0CTensorList[l0CListId];
        uint8_t unitFlag = 0b00;
        if constexpr (ENABLE_UNIT_FLAG) {
            if (iterParams.kIter == iterParams.ddr2l0LoopK - 1) {
                unitFlag = 0b11;
            } else {
                unitFlag = 0b10;
            }
        }
        if (iterParams.kIter == 0) {
            tileMmad(l0CTile, l0ATile, l0BTile, l0BiasTensor, tileParams.l0CurrentM, tileParams.l0CurrentN, currentKL0,
                     true, unitFlag);
        } else {
            tileMmad(l0CTile, l0ATile, l0BTile, tileParams.l0CurrentM, tileParams.l0CurrentN, currentKL0, false, unitFlag);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0abFlag]);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0abFlag]);
    }
};
}  // namespace Catlass::Conv::Block

#endif // CATLASS_CONV_BLOCK_BLOCK_CONV3D_PINGPONG_BIAS_HPP