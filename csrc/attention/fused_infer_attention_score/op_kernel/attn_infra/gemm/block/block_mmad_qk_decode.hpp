/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_MMAD_QK_DECODE_HPP
#define GEMM_BLOCK_MMAD_QK_DECODE_HPP

#include "../../../attn_infra/fused_base_defs.hpp"
#include "../../../attn_infra/fused_coord.hpp"
#include "../../../attn_infra/gemm/fused_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/fused_helper.hpp"
#include "../../../attn_infra/fused_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/fused_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/fused_tile_mmad.hpp"
#include "../../../attn_infra/arch/fused_resource.hpp"

////////////////////////////////////////////////////////////////////

namespace NpuArch::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <
    bool PAGED_CACHE_FLAG_,
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
    MmadAtlasA2FAIQKDecode<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>,
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
    using DispatchPolicy = MmadAtlasA2FAIQKDecode<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using ElementB = typename BType_::Element;
    using ElementC = typename CType_::Element;
    using LayoutA = typename AType_::Layout;
    using LayoutB = typename BType_::Layout;
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
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;
    static constexpr uint32_t COORD_DIM0 = 0;
    static constexpr uint32_t COORD_DIM1 = 1;
    static constexpr uint32_t COORD_DIM2 = 2;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t EMBED_SPLIT_SIZE = 128;
    static constexpr uint32_t UNIT_BLOCK_STACK_NUM = 4;
    static constexpr uint32_t KV_BASE_BLOCK = 512;
    static constexpr uint32_t KV_SPLIT_SIZE = 128;

    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    __aicore__ inline
    BlockMmad() {}

    __aicore__ inline
    void init(Arch::Resource<ArchTag> &resource, uint32_t nDynamic, uint32_t kDynamic, uint32_t l1BufAddrStartQK = 0)
    {
        // Allocate L1 memory space
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStartQK);
        for (uint32_t i = 0; i < STAGES; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStartQK +
                L1TileShape::M * kDynamic * sizeof(ElementA) + nDynamic * kDynamic * sizeof(ElementB) * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
        l1NDynamic = nDynamic;
        l1KDynamic = kDynamic;
    }

    __aicore__ inline
    ~BlockMmad() {}

    __aicore__ inline
    void loadQGM(
        AscendC::GlobalTensor<ElementA> gA,
        LayoutA layoutA,
        uint32_t rowNum, uint32_t &singleGroupHeads, uint32_t &qHeads,
        uint32_t kvNBlockSizeParam)
    {
        uint32_t embed = layoutA.shape(1);
        uint32_t rowNumRound = RoundUp(rowNum, L1AAlignHelper::M_ALIGNED);
        uint32_t tokenNumPerGroup = rowNum / singleGroupHeads;
        auto layoutSingleANd = layoutA.GetTileLayout(MakeCoord(singleGroupHeads, embed));
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);
        copyGmToL1A(
            l1ATensor, gA,
            layoutAInL1, layoutSingleANd,
            tokenNumPerGroup, qHeads * embed, tokenNumPerGroup, BLOCK_SIZE, rowNumRound);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID3);

        l1ATotalRowNum = rowNum;
        l1ARowNumPerKvHead = rowNum / kvNBlockSizeParam;
        l1AEmbedDim = embed;
        l1AKvNBlockSize = kvNBlockSizeParam;
    }
    
    __aicore__ inline
    void getBlockShape(GemmCoord &actShape, uint32_t nL1Index, uint32_t nL1Loop, uint32_t stackSeqTile)
    {
        uint32_t nSplitSize = l1NDynamic;
        if (nL1Index == nL1Loop - 1U) {
            nSplitSize = stackSeqTile - nL1Index * l1NDynamic;
        }
        actShape[COORD_DIM1] = nSplitSize;
    }

    __aicore__ inline
    void getKVOffset(AscendC::GlobalTensor<int32_t> &gBlockTable, uint64_t &kOffset, uint32_t nowNIdx, uint32_t nL1Idx,
        uint32_t strideKV, uint32_t blockSize, uint64_t keyBnStride)
    {
        if constexpr (PAGED_CACHE_FLAG_) {
            uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);
            if (keyBnStride != 0) {
                kOffset = blockTableId * keyBnStride + nL1Idx * l1NDynamic * strideKV;
            } else {
                kOffset = blockTableId * blockSize * strideKV + nL1Idx * l1NDynamic * strideKV;
            }
        } else {
            kOffset = nowNIdx * blockSize * strideKV + nL1Idx * l1NDynamic * strideKV;
        }
    }

    __aicore__ inline
    void operator()(AscendC::GlobalTensor<ElementA> gA,
                    AscendC::GlobalTensor<ElementB> gB,
                    AscendC::GlobalTensor<ElementC> gC,
                    AscendC::GlobalTensor<int32_t> gBlockTable,
                    LayoutA layoutA, LayoutB layoutB, LayoutC layoutC, GemmCoord actualOriShape,
                    uint32_t nIdx, uint32_t nLoop, uint32_t blockSize, uint32_t strideKV,
                    uint32_t kvNIncreIdx, uint64_t keyBnStride)
    {
        uint32_t rowNum = actualOriShape[COORD_DIM0];
        uint32_t stackSeqTile = actualOriShape[COORD_DIM1];
        uint32_t embed = actualOriShape[COORD_DIM2];

        GemmCoord actualShape{rowNum, 0, embed};
        uint64_t gBOffset = 0;

        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(l1ATotalRowNum, l1AEmbedDim);

        uint32_t kvNRowOffset = kvNIncreIdx * l1ARowNumPerKvHead;

        uint32_t tileNNumPerBaseBlock = blockSize / l1NDynamic;
        uint32_t nL1Loop = NpuArch::Detail::Alignment::CeilDiv(stackSeqTile, l1NDynamic);
        for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; ++nL1Idx) {
            uint32_t nowNIdx = nIdx + nL1Idx / tileNNumPerBaseBlock;
            getBlockShape(actualShape, nL1Idx, nL1Loop, stackSeqTile);
            getKVOffset(gBlockTable, gBOffset, nowNIdx, nL1Idx % tileNNumPerBaseBlock,
                        strideKV, blockSize, keyBnStride);
            uint32_t mActual = actualShape.m();
            uint32_t kActual = actualShape.k();
            uint32_t nActual = actualShape.n();
            LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(kActual, nActual);

            auto layoutBTile = layoutB.GetTileLayout(MakeCoord(kActual, nActual));
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
            copyGmToL1B(l1BTensor[l1KvPingPongFlag], gB[gBOffset], layoutBInL1, layoutBTile);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);

            uint32_t mL0Loop = NpuArch::Detail::Alignment::CeilDiv(mActual, L0TileShape::M);
            uint32_t kL0Loop = NpuArch::Detail::Alignment::CeilDiv(kActual, L0TileShape::K);
            for (uint32_t mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
                uint32_t mL0Actual = (mL0Idx < mL0Loop - 1U) ? L0TileShape::M : (mActual - mL0Idx * L0TileShape::M);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    uint32_t kNumL0Actual = (kL0Idx < kL0Loop - 1U) ?
                        L0TileShape::K : (kActual - kL0Idx * L0TileShape::K);
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL0Actual, kNumL0Actual);
                    MatrixCoord l1ATileCoord{mL0Idx * L0TileShape::M + kvNRowOffset, kL0Idx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1ATileCoord)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    copyL1ToL0A(l0ATensor[l0ABPingPongFlag], l1ATile, layoutAInL0, layoutAInL1);

                    LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kNumL0Actual, nActual);
                    MatrixCoord l1BTileCoord{kL0Idx * L0TileShape::K, 0};
                    auto l1BTile = l1BTensor[l1KvPingPongFlag][layoutBInL1.GetOffset(l1BTileCoord)];
                    if ((mL0Idx == 0U) && (kL0Idx == 0U)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                    copyL1ToL0B(l0BTensor[l0ABPingPongFlag], l1BTile, layoutBInL0, layoutBInL1);
                    if ((mL0Idx == mL0Loop - 1U) && (kL0Idx == kL0Loop - 1U)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    bool initMmad = (kL0Idx == 0U);
                    uint32_t mL0Align = (mL0Actual + BLOCK_SIZE - 1U) / BLOCK_SIZE * BLOCK_SIZE;
                    tileMmad(l0CTensor[l0CPingPongFlag],
                        l0ATensor[l0ABPingPongFlag],
                        l0BTensor[l0ABPingPongFlag],
                        mL0Align,
                        nActual,
                        kNumL0Actual,
                        initMmad);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2U);
                    l0ABPingPongFlag = 1U - l0ABPingPongFlag;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                MatrixCoord gmCTileCoord{mL0Idx * L0TileShape::M, nL1Idx * l1NDynamic};
                LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL0Actual, nActual));
                auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL0Actual, nActual));
                copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPongFlag], layoutCTile, layoutInL0C);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
                l0CPingPongFlag = 1U - l0CPingPongFlag;
            }
            l1KvPingPongFlag = 1U - l1KvPingPongFlag;
        }
    }
protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementA> l1ATensor;
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
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

    uint32_t l1MDynamic = 0;
    uint32_t l1NDynamic = 0;
    uint32_t l1KDynamic = 0;

    uint32_t l1ATotalRowNum = 0;     
    uint32_t l1ARowNumPerKvHead = 0;  
    uint32_t l1AEmbedDim = 0;  
    uint32_t l1AKvNBlockSize = 1;     
};

////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Block

#endif // GEMM_BLOCK_MMAD_QK_DECODE_HPP