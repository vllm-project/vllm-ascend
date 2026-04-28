/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_BROADCAST_INPLACE_BY_COLUMN_HPP
#define CATLASS_EPILOGUE_TILE_TILE_BROADCAST_INPLACE_BY_COLUMN_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Tile {

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Compute data type
    class ComputeType_,
    /// Length of the compute buffer
    class TileShape_
>
struct TileBroadcastInplaceByColumn {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    using TileShape = TileShape_;

    CATLASS_DEVICE
    TileBroadcastInplaceByColumn() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementCompute> const &ubInOut
    )
    {
        constexpr uint32_t eleNumPerBlk = BYTE_PER_BLK / sizeof(ElementCompute);
        constexpr uint32_t blkNumPerRow = TileShape::COLUMN / eleNumPerBlk;

        constexpr uint64_t defaultMask = BYTE_PER_VECTOR_FRACTAL / sizeof(ElementCompute);
        constexpr uint64_t tailMask = (TileShape::ROW % BLK_NUM_PER_VECTOR_FRACTAL) * eleNumPerBlk;

        constexpr uint8_t repeatTimes = 1;

        AscendC::CopyRepeatParams repeatParams;
        repeatParams.dstStride = blkNumPerRow;
        repeatParams.srcStride = blkNumPerRow;
        repeatParams.dstRepeatSize = 1;
        repeatParams.srcRepeatSize = 1;

        for (uint32_t rowOffset = 0; rowOffset < TileShape::ROW; rowOffset += BLK_NUM_PER_VECTOR_FRACTAL) {
            uint64_t mask = ((TileShape::ROW - rowOffset) >= BLK_NUM_PER_VECTOR_FRACTAL) ? defaultMask : tailMask;
            for (uint32_t colOffset = eleNumPerBlk; colOffset < TileShape::COLUMN; colOffset += eleNumPerBlk) {
                AscendC::Copy(
                    ubInOut[rowOffset * TileShape::COLUMN + colOffset],
                    ubInOut[rowOffset * TileShape::COLUMN],
                    mask, 1, repeatParams
                );
            }
        }
    }
};

} // namespace Catlass::Epilogue::Tile

#endif
