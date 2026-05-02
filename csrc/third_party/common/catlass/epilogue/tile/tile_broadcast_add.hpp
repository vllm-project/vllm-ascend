/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_BROADCAST_ADD_HPP
#define CATLASS_EPILOGUE_TILE_TILE_BROADCAST_ADD_HPP

#include "catlass/catlass.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Epilogue::Tile {

/// BroadcastAdd computes the elementwise Addition of a tensor of shape (m, n) and a tensor
/// of shape (m, n) after broadcasting. There are two broadcast modes: row-broadcast and
/// column-broadcast.

/// @brief Computes the elementwise Addition of a tensor with shape (m, n) and a tensor with
/// original shape (1, n) broadcast to (m, n).
/// @tparam ArchTag_ is the architecture tag.
/// @tparam ComputeType_ includes the element type and layout information.
/// @tparam TileShape_ is the shape (m, n).
template <
    class ArchTag_,
    class ComputeType_,
    class TileShape_
>
struct TileRowBroadcastAdd {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    using TileShape = TileShape_;

    CATLASS_DEVICE
    TileRowBroadcastAdd() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementCompute> const &ubOut,
        AscendC::LocalTensor<ElementCompute> const &ubIn0,
        AscendC::LocalTensor<ElementCompute> const &ubIn1,
        MatrixCoord const &actualTileShape
    )
    {
        constexpr uint32_t maxRepeatTimes = 255;
        constexpr uint32_t eleNumPerBlk = BYTE_PER_BLK / sizeof(ElementCompute);

        constexpr uint32_t blkNumPerColumn = TileShape::COLUMN / eleNumPerBlk;  // 保证整除
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = blkNumPerColumn;
        repeatParams.src0RepStride = blkNumPerColumn;
        repeatParams.src1RepStride = 0;

        constexpr uint32_t rowNumPerCompute = maxRepeatTimes;
        constexpr uint32_t colNumPerCompute = BYTE_PER_VECTOR_FRACTAL / sizeof(ElementCompute);
        for (uint32_t rowOffset = 0; rowOffset < actualTileShape.row(); rowOffset += rowNumPerCompute) {
            uint32_t residueM = actualTileShape.row() - rowOffset;
            uint8_t repeatTimes = static_cast<uint8_t>((residueM > rowNumPerCompute) ? rowNumPerCompute : residueM);
            for (uint32_t colOffset = 0; colOffset < TileShape::COLUMN; colOffset += colNumPerCompute) {
                uint32_t residueN = TileShape::COLUMN - colOffset;
                uint64_t mask = (residueN > colNumPerCompute) ? colNumPerCompute : residueN;
                AscendC::Add(
                    ubOut[rowOffset * TileShape::COLUMN + colOffset],
                    ubIn0[rowOffset * TileShape::COLUMN + colOffset],
                    ubIn1[colOffset],
                    mask, repeatTimes, repeatParams
                );
            }
        }
    }
};

} // namespace Catlass::Epilogue::Tile

#endif
