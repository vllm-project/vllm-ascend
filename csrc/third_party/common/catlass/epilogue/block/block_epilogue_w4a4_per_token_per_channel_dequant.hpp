/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_W4A4_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_W4A4_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"

namespace Catlass::Epilogue::Block {

template <
    class CType_,
    class PerTokenScaleType_,
    class DType_,
    class TileBroadcastOneBlk_,
    class TileOneBlkColumnBroadcastMul_,
    class TileCopy_,
    class EpilogueTileSwizzle_
>
class BlockEpilogue <
    EpilogueAtlasA2W4A4PerTokenPerChannelDequant,
    CType_,
    PerTokenScaleType_,
    DType_,
    TileBroadcastOneBlk_,
    TileOneBlkColumnBroadcastMul_,
    TileCopy_,
    EpilogueTileSwizzle_
> {
public:
    using DispatchPolicy = EpilogueAtlasA2W4A4PerTokenPerChannelDequant;
    using ArchTag = typename DispatchPolicy::ArchTag;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = typename PerTokenScaleType_::Element;
    using LayoutPerTokenScale = typename PerTokenScaleType_::Layout;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    // Tile compute ops
    using TileBroadcastOneBlk = TileBroadcastOneBlk_;
    using TileOneBlkColumnBroadcastMul = TileOneBlkColumnBroadcastMul_;

    // Tile copy
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyGmToUbPerTokenScale = typename TileCopy_::CopyGmToUbPerTokenScale;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    using EpilogueTileSwizzle = EpilogueTileSwizzle_;
    using TileShape = typename TileOneBlkColumnBroadcastMul::TileShape;

    // Check
    static_assert(
        std::is_same_v<ElementC, half> && std::is_same_v<ElementD, bfloat16_t> &&
            std::is_same_v<ElementPerTokenScale, float>,
        "Accumulator type (`ElementC`) should be half, output type (`ElementD`) should be bfloat16_t while perTokenScale type should be float"
    );
    static_assert(
        std::is_same_v<LayoutC, layout::RowMajor> && 
            std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> && 
            std::is_same_v<LayoutD, layout::RowMajor>,
        "The layout template parameters of BlockEpilogue are wrong"
    );

    static_assert(
        TileShape::ROW == TileBroadcastOneBlk::COMPUTE_LENGTH,
        "TileShape must be consistent for all tile compute ops"
    );

    static_assert(
        TileShape::ROW % 16 == 0,
        "Epilogue TileShape::ROW must be divisible by 16"
    );

    static_assert(
        /*ubC*/ TileShape::COUNT * sizeof(ElementC)             
        + /*ubPerTokenScale*/ TileShape::ROW * sizeof(ElementPerTokenScale) 
        + /*ubD*/ TileShape::COUNT * sizeof(ElementD)
        + /*ubCFp32 & ubPerTokenScaleBrcb*/ 2 * TileShape::COUNT * sizeof(float) <= ArchTag::UB_SIZE,
        "TileShape is too large to fit in UB"
    );

    struct Params {
        __gm__ ElementPerTokenScale *ptrPerTokenScale{nullptr};
        LayoutPerTokenScale layoutPerTokenScale{};
        __gm__ ElementD *ptrD{nullptr};
        LayoutD layoutD{};

        CATLASS_DEVICE
        Params() {};

        CATLASS_DEVICE
        Params(
            __gm__ ElementPerTokenScale *ptrPerTokenScale_, LayoutPerTokenScale const &layoutPerTokenScale_,
            __gm__ ElementD *ptrD_, LayoutD const &layoutD_
        ) : ptrPerTokenScale(ptrPerTokenScale_), layoutPerTokenScale(layoutPerTokenScale_),
            ptrD(ptrD_), layoutD(layoutD_) {}
    };

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}) : params(params)
    {
        size_t ubCOffset = 0;
        size_t ubPerTokenScaleOffset = ubCOffset + TileShape::COUNT * sizeof(ElementC);
        size_t ubDOffset = ubPerTokenScaleOffset + TileShape::ROW * sizeof(ElementPerTokenScale);
        size_t ubOffset1 = ubDOffset + TileShape::COUNT * sizeof(ElementD);           // for `ubCFp32` & `ubDFp32` 
        size_t ubOffset2 = ubOffset1 + TileShape::COUNT  * sizeof(float);             // for `ubPerTokenScaleBrcb`  
        // for `ubPerTokenScaleBrcb`

        // Load from GM
        ubC = resource.ubBuf.template GetBufferByByte<ElementC>(ubCOffset);
        ubPerTokenScale = resource.ubBuf.template GetBufferByByte<ElementPerTokenScale>(ubPerTokenScaleOffset);
        ubD = resource.ubBuf.template GetBufferByByte<ElementD>(ubDOffset);

        // Caculation (with perToken scale)
        ubCFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset1);
        ubPerTokenScaleBrcb = resource.ubBuf.template GetBufferByByte<float>(ubOffset2);
        ubDFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset1);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
    }

    CATLASS_DEVICE
    void UpdateParams(Params const &params_)
    {
        params = params_;
    }

    CATLASS_DEVICE
    void operator() (
        GemmCoord const &blockShapeMNK,
        GemmCoord const &blockCoordMNK,
        GemmCoord const &actualBlockShapeMNK,
        AscendC::GlobalTensor<ElementC> const &gmBlockC,
        LayoutC const &layoutBlockC, Callback &&callback = Callback{}
    )
    {
        if (actualBlockShapeMNK.k() == 0) {
            return;
        }
        callback();

        // Calculate the offset of the current block
        MatrixCoord blockShape = blockShapeMNK.GetCoordMN();
        MatrixCoord blockCoord = blockCoordMNK.GetCoordMN();
        MatrixCoord actualBlockShape = actualBlockShapeMNK.GetCoordMN();
        MatrixCoord blockOffset = blockCoord * blockShape;

        AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale;
        gmPerTokenScale.SetGlobalBuffer(params.ptrPerTokenScale);
        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer(params.ptrD);

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffsetInBlock = tileCoord * tileShape;
            auto tileOffset = blockOffset + tileOffsetInBlock;

            // for GMc
            auto gmTileC = gmBlockC[layoutBlockC.GetOffset(tileOffsetInBlock)];
            auto layoutGmTileC = layoutBlockC.GetTileLayout(actualTileShape);

            LayoutC layoutUbC{actualTileShape, ubTileStride};

            auto calcNum = actualTileShape.row() * TileShape::COLUMN;
            
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutGmTileC);

            auto perTokenScaleTileOffset = tileOffset.template GetCoordByAxis<0>();
            auto perTokenScaleTileShape = actualTileShape.template GetCoordByAxis<0>();
            
            auto gmTilePerTokenScale = gmPerTokenScale[params.layoutPerTokenScale.GetOffset(perTokenScaleTileOffset)];
            auto layoutGmTilePerTokenScale = params.layoutPerTokenScale.GetTileLayout(perTokenScaleTileShape);

            auto layoutUbPerTokenScale = LayoutPerTokenScale::template MakeLayoutInUb<ElementPerTokenScale>(
                perTokenScaleTileShape);

            copyGmToUbPerTokenScale(ubPerTokenScale, gmTilePerTokenScale, layoutUbPerTokenScale,
                layoutGmTilePerTokenScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventID);
            
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventID);
            AscendC::Cast(/*float*/ ubCFp32, /*half*/ ubC, AscendC::RoundMode::CAST_NONE, calcNum);
            AscendC::PipeBarrier<PIPE_V>();

            tileBroadcastOneBlk(ubPerTokenScaleBrcb, ubPerTokenScale);
            AscendC::PipeBarrier<PIPE_V>();

            tileOneBlkColumnBroadcastMul(ubDFp32, ubCFp32, ubPerTokenScaleBrcb);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(ubD, ubDFp32, AscendC::RoundMode::CAST_RINT, calcNum);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventID);

            auto resTileShape = MatrixCoord{actualTileShape.row(), actualTileShape.column()};
            LayoutD layoutUbD{resTileShape, ubTileStride};

            auto gmTileD = gmD[params.layoutD.GetOffset(tileOffset)];
            auto layoutGmTileD = params.layoutD.GetTileLayout(resTileShape);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventID);
            copyUbToGmD(gmTileD, ubD, layoutGmTileD, layoutUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
        }
    }

private:
    Params params;

    AscendC::LocalTensor<ElementC> ubC;
    AscendC::LocalTensor<ElementPerTokenScale> ubPerTokenScale;
    AscendC::LocalTensor<ElementD> ubD;
    AscendC::LocalTensor<float> ubCFp32;
    AscendC::LocalTensor<float> ubPerTokenScaleBrcb;
    AscendC::LocalTensor<float> ubDFp32;
    // [gmTileC] -> ubC --(Cast:float)--> ubCFp32 --(Muls)--> ubDFp32 --(Cast:bf16)--> ubD -> [gmTileD]

    int32_t eventID{0};

    TileBroadcastOneBlk tileBroadcastOneBlk;
    TileOneBlkColumnBroadcastMul tileOneBlkColumnBroadcastMul;

    CopyGmToUbC copyGmToUbC;
    CopyGmToUbPerTokenScale copyGmToUbPerTokenScale;
    CopyUbToGmD copyUbToGmD;
};

}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_EPILOGUE_W4A4_HPP
