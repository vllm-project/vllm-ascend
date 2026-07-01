/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_TILE_TILE_SWIZZLE_HPP
#define EPILOGUE_TILE_TILE_SWIZZLE_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/detail/alignment.hpp"
#include "../../../attn_infra/matrix_coord.hpp"

namespace NpuArch::Epilogue::Tile
{

struct EpilogueIdentityTileSwizzle {
    MatrixCoord blockShape;
    MatrixCoord tileShape;
    MatrixCoord loopsMN;

    __aicore__ inline
    EpilogueIdentityTileSwizzle() = default;

    __aicore__ inline
    EpilogueIdentityTileSwizzle(MatrixCoord const &blockShape, MatrixCoord const &tileShape) :
        blockShape(blockShape),
        tileShape(tileShape)
    {
        loopsMN = NpuArch::Detail::Alignment::CeilDiv(blockShape, tileShape);
    }

    __aicore__ inline
    uint32_t GetLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    __aicore__ inline
    MatrixCoord GetTileCoord(uint32_t loopIdx) const
    {
        return MatrixCoord{ loopIdx / loopsMN.column(), loopIdx % loopsMN.column() };
    }

    __aicore__ inline
    MatrixCoord GetActualTileShape(MatrixCoord const &tileCoord) const
    {
        return MatrixCoord::Min(tileShape, blockShape - tileCoord * tileShape);
    }
};

struct EpilogueHorizontalTileSwizzle {
    MatrixCoord blockShape;
    MatrixCoord tileShape;
    MatrixCoord loopsMN;

    __aicore__ inline
    EpilogueHorizontalTileSwizzle() = default;

    __aicore__ inline
    EpilogueHorizontalTileSwizzle(MatrixCoord const &blockShape, MatrixCoord const &tileShape) :
        blockShape(blockShape),
        tileShape(tileShape)
    {
        loopsMN = NpuArch::Detail::Alignment::CeilDiv(blockShape, tileShape);
    }

    __aicore__ inline
    uint32_t GetLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    __aicore__ inline
    MatrixCoord GetTileCoord(uint32_t loopIdx) const
    {
        return MatrixCoord{ loopIdx % loopsMN.row(), loopIdx / loopsMN.row() };
    }

    __aicore__ inline
    MatrixCoord GetActualTileShape(MatrixCoord const &tileCoord) const
    {
        return MatrixCoord::Min(tileShape, blockShape - tileCoord * tileShape);
    }
};

}

#endif  // EPILOGUE_TILE_TILE_SWIZZLE_HPP