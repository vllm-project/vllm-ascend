/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_BROADCAST_INPLACE_BY_ROW_HPP
#define CATLASS_EPILOGUE_TILE_TILE_BROADCAST_INPLACE_BY_ROW_HPP

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
struct TileBroadcastInplaceByRow {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    using TileShape = TileShape_;

    CATLASS_DEVICE
    TileBroadcastInplaceByRow() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementCompute> const &ubInOut
    )
    {
        constexpr uint32_t eleNumPerVectorFractal = BYTE_PER_VECTOR_FRACTAL / sizeof(ElementCompute);

        constexpr uint64_t mask = eleNumPerVectorFractal;
        constexpr uint8_t repeatTimes = TileShape::COLUMN / eleNumPerVectorFractal;

        AscendC::CopyRepeatParams repeatParams;
        repeatParams.dstStride = 1;
        repeatParams.srcStride = 1;
        repeatParams.dstRepeatSize = BLK_NUM_PER_VECTOR_FRACTAL;
        repeatParams.srcRepeatSize = BLK_NUM_PER_VECTOR_FRACTAL;

        for (uint32_t rowOffset = 1; rowOffset < TileShape::ROW; ++rowOffset) {
            AscendC::Copy(ubInOut[rowOffset * TileShape::COLUMN], ubInOut, mask, repeatTimes, repeatParams);
        }
    }
};

} // namespace Catlass::Epilogue::Tile

#endif
