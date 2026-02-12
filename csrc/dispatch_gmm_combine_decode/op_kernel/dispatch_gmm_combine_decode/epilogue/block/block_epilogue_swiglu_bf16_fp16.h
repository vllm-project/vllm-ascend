/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"

#include "../tile/tile_stride_muls.h"
#include "../tile/tile_stride_binary.h"

namespace Catlass::Epilogue::Block {

template <uint32_t UB_STAGES_, uint32_t EXEC_FLAG_,
          class CType_, class ScaleType_, class LayoutScale_, class LayoutPerTokenScale_,
          class DType_, class TileRowBroadcastMul_, class TileBroadcastOneBlk_, class TileOneBlkColumnBroadcastMul_,
          class TileCopy_, class EpilogueTileSwizzle_>
class BlockEpilogue<EpilogueAtlasA2Swiglu<UB_STAGES_, EXEC_FLAG_>,
                    CType_, Gemm::GemmType<ScaleType_, LayoutScale_>, Gemm::GemmType<float, LayoutPerTokenScale_>,
                    DType_, TileRowBroadcastMul_, TileBroadcastOneBlk_, TileOneBlkColumnBroadcastMul_,
                    TileCopy_, EpilogueTileSwizzle_>
{
public:
    using DispatchPolicy = EpilogueAtlasA2Swiglu<UB_STAGES_, EXEC_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementRawScale = ScaleType_;
    using ElementFp32Scale = float;
    using LayoutScale = LayoutScale_;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    // Check data infos
    static_assert(std::is_same_v<ElementC, float> && std::is_same_v<ElementD, float>,
                  "The element type template parameters of BlockEpilogue are wrong");
    static_assert(std::is_same_v<LayoutC, layout::RowMajor> && std::is_same_v<LayoutScale, layout::VectorLayout> &&
                      std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> &&
                      std::is_same_v<LayoutD, layout::RowMajor>,
                  "The layout template parameters of BlockEpilogue are wrong");

    // Tile compute ops
    using TileRowBroadcastMul = TileRowBroadcastMul_;
    using TileBroadcastOneBlk = TileBroadcastOneBlk_;
    using TileOneBlkColumnBroadcastMul = TileOneBlkColumnBroadcastMul_;

    // Tile copy
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyGmToUbScale = typename TileCopy_::CopyGmToUbX;
    using CopyGmToUbPerTokenScale = typename TileCopy_::CopyGmToUbY;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    using EpilogueTileSwizzle = EpilogueTileSwizzle_;

    using TileShape = typename TileRowBroadcastMul::TileShape;
    static_assert(TileShape::ROW * sizeof(float) % BYTE_PER_BLK == 0,
                  "The per token scale granularity for word calculation must be 32 bytes aligned.");
    static_assert(TileShape::COLUMN % 2 == 0, "The n-axis needs to be divided into two parts.");

    static_assert(TileShape::ROW == TileBroadcastOneBlk::COMPUTE_LENGTH &&
                      std::is_same_v<TileShape, typename TileOneBlkColumnBroadcastMul::TileShape>,
                  "TileShape must be consistent for all tile compute ops");

    static_assert(UB_STAGES <= 2, "UB stages too large, event id is not enough.");

    static_assert((UB_STAGES * (TileShape::COUNT * sizeof(ElementC) + TileShape::COUNT * sizeof(ElementD)) +
                   TileShape::ROW * BYTE_PER_BLK) <= ArchTag::UB_SIZE,
                  "TileShape is too large to fit in UB");

    struct Params {
        __gm__ ElementRawScale *ptrScale{nullptr};
        LayoutScale layoutScale{};
        __gm__ ElementPerTokenScale *ptrPerTokenScale{nullptr};
        LayoutPerTokenScale layoutPerTokenScale{};
        __gm__ ElementD *ptrD{nullptr};
        LayoutD layoutD{};

        CATLASS_DEVICE
        Params() {};

        CATLASS_DEVICE
        Params(__gm__ ElementRawScale *ptrScale_, LayoutScale const &layoutScale_,
               __gm__ ElementPerTokenScale *ptrPerTokenScale_, LayoutPerTokenScale const &layoutPerTokenScale_,
               __gm__ ElementD *ptrD_, LayoutD const &layoutD_)
            : ptrScale(ptrScale_),
              layoutScale(layoutScale_),
              ptrPerTokenScale(ptrPerTokenScale_),
              layoutPerTokenScale(layoutPerTokenScale_),
              ptrD(ptrD_),
              layoutD(layoutD_)
        {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}) : params(params)
    {
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;
        int32_t eventMTE2V = 0;
        int32_t eventMTE3V = 0;
        int32_t eventVMTE3 = 0;
        int32_t eventMTE3MTE2 = 0;
        int32_t eventMTE2MTE3 = 0;
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementD);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;
            eventUbMTE3MTE2List[i] = eventMTE3MTE2++;
            eventUbMTE2MTE3List[i] = eventMTE2MTE3++;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbMTE3MTE2List[i]);
        }
        ubDenominatorMxN = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbMTE3MTE2List[i]);
        }
    }

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }

    CATLASS_DEVICE
    void operator()(GemmCoord const &blockShapeMNK, GemmCoord const &blockCoordMNK,
                    GemmCoord const &actualBlockShapeMNK, AscendC::GlobalTensor<ElementC> const &gmBlockC,
                    LayoutC const &layoutBlockC, Callback &&callback = Callback{})
    {
        if (0 == actualBlockShapeMNK.k()) {
            return;
        }
        callback();
        ubListId = 0;
        // Calculate the offset of the current block
        MatrixCoord blockShape = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoord * blockShape;
        bool isLeft = blockOffset.column() < (params.layoutD.shape(1) >> 1);
        AscendC::GlobalTensor<ElementRawScale> gmScale;
        gmScale.SetGlobalBuffer(params.ptrScale);
        AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale;
        gmPerTokenScale.SetGlobalBuffer(params.ptrPerTokenScale);
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = 0;  // for 1C1V
        uint32_t subblockNum = 1;  // for 1C1V
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffsetInBlock = tileCoord * tileShape;
            auto tileOffset = blockOffset + tileOffsetInBlock;

            auto gmTileC = gmBlockC[layoutBlockC.GetOffset(tileOffsetInBlock)];
            auto layoutGmTileC = layoutBlockC.GetTileLayout(actualTileShape);

            auto &ubC = ubCList[ubListId];
            LayoutC layoutUbC{actualTileShape, ubTileStride};

            auto &ubD = ubDList[ubListId];
            LayoutD layoutUbD{actualTileShape, ubTileStride};
            auto gmTileD = gmD[params.layoutD.GetOffset(tileOffset)];
            auto layoutGmTileD = params.layoutD.GetTileLayout(actualTileShape);

            if (isLeft) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
                copyGmToUbC(ubC, gmTileC, layoutUbC, layoutGmTileC);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
                AscendC::Muls(ubDenominatorMxN, ubC, -1.0f, TileShape::COUNT);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(ubDenominatorMxN, ubDenominatorMxN, TileShape::COUNT);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(ubDenominatorMxN, ubDenominatorMxN, 1.0f, TileShape::COUNT);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
                AscendC::Div(ubD, ubC, ubDenominatorMxN, TileShape::COUNT);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
                copyUbToGmD(gmTileD, ubD, layoutGmTileD, layoutUbD);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbMTE3MTE2List[ubListId]);
                copyGmToUbC(ubC, gmTileC, layoutUbC, layoutGmTileC);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventUbMTE2MTE3List[ubListId]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventUbMTE2MTE3List[ubListId]);
                copyUbToGmD(gmTileD, ubC, layoutGmTileD, layoutUbD);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventUbMTE3MTE2List[ubListId]);
            }

            ubListId = (ubListId + 1 < UB_STAGES) ? (ubListId + 1) : 0;
        }
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];

    int32_t eventUbCVMTE2List[UB_STAGES];
    int32_t eventUbCMTE2VList[UB_STAGES];
    int32_t eventUbDMTE3VList[UB_STAGES];
    int32_t eventUbDVMTE3List[UB_STAGES];
    int32_t eventUbMTE3MTE2List[UB_STAGES];
    int32_t eventUbMTE2MTE3List[UB_STAGES];

    uint32_t ubListId{0};

    AscendC::LocalTensor<float> ubDenominatorMxN;

    TileRowBroadcastMul tileRowBroadcastMul;
    TileBroadcastOneBlk tileBroadcastOneBlk;
    TileOneBlkColumnBroadcastMul tileOneBlkColumnBroadcastMul;

    CopyGmToUbC copyGmToUbC;
    CopyGmToUbScale copyGmToUbScale;
    CopyGmToUbPerTokenScale copyGmToUbPerTokenScale;
    CopyUbToGmD copyUbToGmD;
};

}  // namespace Catlass::Epilogue::Block
