/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Tile {

template <class ArchTag_, class ElementCompute_, class TileShape_, class DstTileShape_, class SrcTileShape_>
struct TileStrideMuls {
    using ArchTag = ArchTag_;
    using ElementCompute = ElementCompute_;
    using TileShape = TileShape_;
    using DstTileShape = DstTileShape_;
    using SrcTileShape = SrcTileShape_;

    static_assert(DstTileShape::ROW == SrcTileShape::ROW && DstTileShape::ROW == TileShape::ROW, "Error");

    CATLASS_DEVICE
    TileStrideMuls() {}

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementCompute> const &ubDst,
                    AscendC::LocalTensor<ElementCompute> const &ubSrc, ElementCompute scalar)
    {
        constexpr uint32_t maxRepeatTimes = 255;
        constexpr uint32_t eleNumPerBlk = BYTE_PER_BLK / sizeof(ElementCompute);

        constexpr uint32_t dstBlkNumPerColumn = DstTileShape::COLUMN / eleNumPerBlk;
        constexpr uint32_t srcBlkNumPerColumn = SrcTileShape::COLUMN / eleNumPerBlk;
        AscendC::UnaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.srcBlkStride = 1;
        repeatParams.dstRepStride = dstBlkNumPerColumn;
        repeatParams.srcRepStride = srcBlkNumPerColumn;

        constexpr uint32_t rowNumPerCompute = maxRepeatTimes;
        constexpr uint32_t colNumPerCompute = BYTE_PER_VECTOR_FRACTAL / sizeof(ElementCompute);
        for (uint32_t rowOffset = 0; rowOffset < TileShape::ROW; rowOffset += rowNumPerCompute) {
            uint32_t residueM = TileShape::ROW - rowOffset;
            uint8_t repeatTimes = static_cast<uint8_t>((residueM > rowNumPerCompute) ? rowNumPerCompute : residueM);
            for (uint32_t colOffset = 0; colOffset < TileShape::COLUMN; colOffset += colNumPerCompute) {
                uint32_t residueN = TileShape::COLUMN - colOffset;
                uint64_t mask = (residueN > colNumPerCompute) ? colNumPerCompute : residueN;
                AscendC::Muls(ubDst[rowOffset * DstTileShape::COLUMN + colOffset],
                              ubSrc[rowOffset * SrcTileShape::COLUMN + colOffset], scalar, mask, repeatTimes,
                              repeatParams);
            }
        }
    }
};

}  // namespace Catlass::Epilogue::Tile
