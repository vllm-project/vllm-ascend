/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_BLOCK_BLOCK_SWIZZLE_HPP
#define CATLASS_CONV_BLOCK_BLOCK_SWIZZLE_HPP

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/conv_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Conv::Block {
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Block swizzling function for conv3d
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct Conv3dIdentityBlockSwizzle {
    /// Data members
    Conv3d6HdCoord outShape;
    Conv3d6HdCoord coreTileShape;
    MatrixCoord loopsMN;
    Conv3d6HdCoord  loops;
    uint64_t nStart, doStart, co1Start, howoStart;

    // Methods
    CATLASS_DEVICE
    Conv3dIdentityBlockSwizzle() {}

    CATLASS_DEVICE
    Conv3dIdentityBlockSwizzle(Conv3d6HdCoord const &outShape_, Conv3d6HdCoord const &loops_)
        : outShape(outShape_), loops(loops_)
    {
        loops = Conv3d6HdCoord{min(outShape.n(), loops.n()), min(outShape.d(), loops.d()),
                               min(outShape.c1(), loops.c1()), min(outShape.hw(), loops.hw())};
        coreTileShape = Conv3d6HdCoord{CeilDiv(outShape.n(), loops.n()), CeilDiv(outShape.d(), loops.d()),
                                       CeilDiv(outShape.c1(), loops.c1()), CeilDiv(outShape.hw(), loops.hw())};
        loopsMN = MatrixCoord{loops.hw(), loops.n() * loops.d() * loops.c1()};
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    CATLASS_DEVICE
    Conv3d6HdCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        if constexpr (SwizzleDirection == 0) {
            uint32_t tileBlockLoop = CeilDiv(loopsMN.row(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.column());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.column());

            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMN.row() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMN.column() - nIdx - 1;
            }

            uint32_t howoIdx = mIdx;

            uint32_t noIdx = nIdx / (loops[1]*loops[2]);
            uint32_t doIdx = nIdx / loops[2] % loops[1];
            uint32_t c1Idx = nIdx % loops[2];
            return Conv3d6HdCoord{noIdx, doIdx, c1Idx, howoIdx};
        } else if constexpr (SwizzleDirection ==1) {
            uint32_t tileBlockLoop = CeilDiv(loopsMN.column(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.row());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.row());

            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMN.column() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMN.row() - mIdx - 1;
            }

            uint32_t howoIdx = mIdx;

            uint32_t noIdx = nIdx / (loops[1]*loops[2]);
            uint32_t doIdx = nIdx / loops[2] % loops[1];
            uint32_t c1Idx = nIdx % loops[2];
            return Conv3d6HdCoord{noIdx, doIdx, c1Idx, howoIdx};
        }
    }

    CATLASS_DEVICE
    Conv3d6HdCoord GetDimStartIdx(Conv3d6HdCoord blockCoord)
    {
        uint32_t nStart = blockCoord.n() * coreTileShape.n();
        uint32_t doStart = blockCoord.d() * coreTileShape.d();
        uint32_t c1Start = blockCoord.c1() * coreTileShape.c1();
        uint32_t howoStart = blockCoord.hw() * coreTileShape.hw();
        return Conv3d6HdCoord{nStart, doStart, c1Start, howoStart};
    }

    CATLASS_DEVICE
    Conv3d6HdCoord GetActualBlockShape(Conv3d6HdCoord blockCoord, Conv3d6HdCoord dimStartIdx)
    {
        uint32_t nActual = (blockCoord.n() == loops.n() - 1) ?
            (outShape[0] - dimStartIdx.n()) : coreTileShape.n();
        
        uint32_t doActual = (blockCoord.d() == loops.d() - 1) ?
            (outShape[1] - dimStartIdx.d()) : coreTileShape.d();
        
        uint32_t c1Actual = (blockCoord.c1() == loops.c1() - 1) ?
            (outShape[2] - dimStartIdx.c1()) : coreTileShape.c1();
        
        uint32_t hwActual = (blockCoord.hw() == loops.hw() - 1) ?
            (outShape[3] - dimStartIdx.hw()) : coreTileShape.hw();
        return Conv3d6HdCoord{nActual, doActual, c1Actual, hwActual};
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Block swizzling function for Conv2ds
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct Conv2dIdentityBlockSwizzle {
    /// Data members
    Conv2dCoord problemShape; // {Batch, Ho, Wo, Cout, Cin1}
    Conv2dHoWoCoCoord tiles;
    Conv2dHoWoCoCoord loops;

    /// Methods
    CATLASS_DEVICE
    Conv2dIdentityBlockSwizzle() {}

    CATLASS_DEVICE
    Conv2dIdentityBlockSwizzle(Conv2dCoord const &problemShape_, Conv2dHoWoCoCoord const &tiles_)
        : problemShape(problemShape_), tiles(tiles_) {
        loops = CeilDiv(Conv2dHoWoCoCoord(problemShape.GetHoWoCoCoord()), tiles);      
    }

    CATLASS_DEVICE
    Conv2dIdentityBlockSwizzle(Conv2dCoord const &problemShape_, Conv2dHoWoCoCoord const &tiles_,
        Conv2dHoWoCoCoord const &loops_)
        : problemShape(problemShape_), tiles(tiles_), loops(loops_) {}

    CATLASS_DEVICE
    void Update(Conv2dCoord const &problemShape_, Conv2dHoWoCoCoord const &tiles_) {
        problemShape = problemShape_;
        tiles = tiles_;
        loops = CeilDiv(Conv2dHoWoCoCoord(problemShape.GetHoWoCoCoord()), tiles);
    }

    CATLASS_DEVICE
    void Update(Conv2dCoord const &problemShape_, Conv2dHoWoCoCoord const &tiles_, Conv2dHoWoCoCoord const &loops_) {
        problemShape = problemShape_;
        tiles = tiles_;
        loops = loops_;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const {
        return loops.ho() * loops.wo() * loops.cout(); 
    }

    CATLASS_DEVICE
    uint32_t GetLoops() const {
        return problemShape.batch() * this->GetCoreLoops(); 
    }

    CATLASS_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx) {
        return taskIdx / (GetCoreLoops());
    }

    CATLASS_DEVICE
    Conv2dCoord GetBlockCoord(uint32_t taskIdx) {
        uint32_t outerIdx = this->GetBatchIdx(taskIdx);
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        if constexpr (SwizzleDirection == 0) { // HoWoCo
            uint32_t tileBlockLoop = CeilDiv(loops.howo(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loops.cout());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loops.cout());

            uint32_t nHoWo = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nHoWo = loops.howo() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t howoIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nHoWo;
            uint32_t coutIdx = inTileBlockIdx / nHoWo;
            if (tileBlockIdx % 2 == 1) {
                coutIdx = loops.cout() - coutIdx - 1;
            }
            uint32_t hoIdx = howoIdx / loops.wo();
            uint32_t woIdx = howoIdx % loops.wo();
            return Conv2dCoord{outerIdx, hoIdx, woIdx, coutIdx, 0};
        } else if constexpr (SwizzleDirection == 1) { // CoHoWo
            uint32_t tileBlockLoop = CeilDiv(loops.cout(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loops.howo());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loops.howo());

            uint32_t nCout = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCout = loops.cout() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t howoIdx = inTileBlockIdx / nCout;
            uint32_t coutIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCout;
            if (tileBlockIdx % 2 == 1) {
                howoIdx = loops.howo() - howoIdx - 1;
            }
            uint32_t hoIdx = howoIdx / loops.wo();
            uint32_t woIdx = howoIdx % loops.wo();
            return Conv2dCoord{outerIdx, hoIdx, woIdx, coutIdx, 0};
        }
    }

    CATLASS_DEVICE
    Conv2dCoord GetActualBlockShape(Conv2dCoord blockCoord) {
        uint32_t hoActual = (blockCoord.h() == loops.ho() - 1) ? 
            (problemShape.h() - blockCoord.h() * tiles.ho()) : tiles.ho();
        uint32_t woActual = (blockCoord.w() == loops.wo() - 1) ? 
            (problemShape.w() - blockCoord.w() * tiles.wo()) : tiles.wo();
        uint32_t coutActual = (blockCoord.cout() == loops.cout() - 1) ? 
            (problemShape.cout() - blockCoord.cout() * tiles.cout()) : tiles.cout();
        uint32_t cin1Actual = problemShape.cin1();
        return Conv2dCoord{1, hoActual, woActual, coutActual, cin1Actual};
    }
};
}  // namespace Catlass::Conv::Block

#endif  // CATLASS_CONV_BLOCK_BLOCK_SWIZZLE_HPP
