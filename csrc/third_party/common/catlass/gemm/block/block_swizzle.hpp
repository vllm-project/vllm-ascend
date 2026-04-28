/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_HPP

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Block {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Block swizzling function for Gemms
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct GemmIdentityBlockSwizzle {
    /// Data members

    GemmCoord problemShape;
    MatrixCoord tileMN;
    MatrixCoord loopsMN;

    /// Methods

    CATLASS_DEVICE
    GemmIdentityBlockSwizzle() {}

    CATLASS_DEVICE
    GemmIdentityBlockSwizzle(GemmCoord const &problemShape_, MatrixCoord const &tileMN_)
        : problemShape(problemShape_), tileMN(tileMN_)
    {
        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }

    CATLASS_DEVICE
    GemmIdentityBlockSwizzle(GemmCoord const &problemShape_, MatrixCoord const &tileMN_,
        MatrixCoord const &loopsMN_)
        : problemShape(problemShape_), tileMN(tileMN_), loopsMN(loopsMN_) {}

    CATLASS_DEVICE
    void Update(GemmCoord const &problemShape_, MatrixCoord const &tileMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;

        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }

    CATLASS_DEVICE
    void Update(GemmCoord const &problemShape_, MatrixCoord const &tileMN_, MatrixCoord const &loopsMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;
        loopsMN = loopsMN_;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    CATLASS_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx)
    {
        return taskIdx / (GetCoreLoops());
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        if constexpr (SwizzleDirection == 0) { // Zn
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
            return GemmCoord{mIdx, nIdx, 0};
        } else if constexpr (SwizzleDirection == 1) { // Nz
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
            return GemmCoord{mIdx, nIdx, 0};
        }
    }

    CATLASS_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord blockCoord)
    {
        uint32_t mActual = (blockCoord.m() == (loopsMN.row() - 1)) ?
            (problemShape.m() - blockCoord.m() * tileMN.row()) : tileMN.row();
        uint32_t nActual = (blockCoord.n() == (loopsMN.column() - 1)) ?
            (problemShape.n() - blockCoord.n() * tileMN.column()) : tileMN.column();
        uint32_t kActual = problemShape.k();
        return GemmCoord{mActual, nActual, kActual};
    }
};

struct DynamicGemmIdentityBlockSwizzle : public GemmIdentityBlockSwizzle<> {

    uint32_t swizzleOffset{1};
    uint32_t swizzleDirection{0};

    CATLASS_DEVICE
    DynamicGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, MatrixCoord const &tileMN_, 
        uint32_t swizzleOffset_, uint32_t swizzleDirection_) : 
        swizzleOffset(swizzleOffset_), swizzleDirection(swizzleDirection_), 
        GemmIdentityBlockSwizzle<>(problemShape_, tileMN_) {}

    CATLASS_DEVICE
    DynamicGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, MatrixCoord const &tileMN_) : 
        GemmIdentityBlockSwizzle<>(problemShape_, tileMN_) {}

    CATLASS_DEVICE
    DynamicGemmIdentityBlockSwizzle() {}


    CATLASS_DEVICE
    void SetSwizzleParams(uint32_t swizzleOffset_, uint32_t swizzleDirection_)
    {
        swizzleOffset = swizzleOffset_;
        swizzleDirection = swizzleDirection_;
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        if (swizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMN.row(), swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * loopsMN.column());
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * loopsMN.column());

            uint32_t nRow = swizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMN.row() - swizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMN.column() - nIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, 0};
        } else { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMN.column(), swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * loopsMN.row());
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * loopsMN.row());

            uint32_t nCol = swizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMN.column() - swizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMN.row() - mIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, 0};
        }
    }
};

/// Block swizzling function for Splitk Gemms
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct SplitkGemmIdentityBlockSwizzle {
    /// Data members

    GemmCoord problemShape;
    GemmCoord tileShape;
    GemmCoord loopsMNK;
    uint32_t splitkFactor = 1;  // splite k dim into virtual cores

    /// Methods

    CATLASS_DEVICE
    SplitkGemmIdentityBlockSwizzle() {}

    CATLASS_DEVICE
    SplitkGemmIdentityBlockSwizzle(
        GemmCoord const &problemShape_, GemmCoord const &tileShape_, uint32_t splitkFactor_ = 1
    ) : problemShape(problemShape_), tileShape(tileShape_), splitkFactor(splitkFactor_)
    {
        loopsMNK = CeilDiv(problemShape, tileShape);
    }

    CATLASS_DEVICE
    uint32_t GetKIdxBySplitkSliceIdx(uint32_t splitkSliceIdx) const
    {
        if (splitkSliceIdx < loopsMNK.k() % splitkFactor) {
            return (loopsMNK.k() / splitkFactor + 1) * splitkSliceIdx;
        } else {
            return splitkSliceIdx * (loopsMNK.k() / splitkFactor) + loopsMNK.k() % splitkFactor;
        }
    }

    CATLASS_DEVICE
    uint32_t GetSplitkSliceIdx(uint32_t taskIdx) const
    {
        uint32_t mnLoops = loopsMNK.m() * loopsMNK.n();
        return taskIdx % GetCoreLoops() / mnLoops;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return loopsMNK.m() * loopsMNK.n() * splitkFactor;
    }

    CATLASS_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx)
    {
        return taskIdx / GetCoreLoops();
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t splitkSliceIdx = GetSplitkSliceIdx(taskIdx);
        uint32_t kIdx = GetKIdxBySplitkSliceIdx(splitkSliceIdx);

        uint32_t innerIdx = taskIdx % (loopsMNK.m() * loopsMNK.n());
        if constexpr (SwizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.n());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.n());

            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMNK.n() - nIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        } else if constexpr (SwizzleDirection == 1) { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.m());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.m());

            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        }
    }

    CATLASS_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord blockCoord, uint32_t splitkSliceIdx)
    {
        uint32_t splitkSliceLen;
        if (splitkSliceIdx < loopsMNK.k() % splitkFactor) {
            splitkSliceLen = (loopsMNK.k() / splitkFactor + 1) * tileShape.k();
        } else {
            splitkSliceLen = (loopsMNK.k() / splitkFactor) * tileShape.k();
        }
        uint32_t mActual = (blockCoord.m() == (loopsMNK.m() - 1)) ?
            (problemShape.m() - blockCoord.m() * tileShape.m()) : tileShape.m();
        uint32_t nActual = (blockCoord.n() == (loopsMNK.n() - 1)) ?
            (problemShape.n() - blockCoord.n() * tileShape.n()) : tileShape.n();
        uint32_t kActual = (splitkSliceIdx == (splitkFactor - 1)) ?
            (problemShape.k() - blockCoord.k() * tileShape.k()) : splitkSliceLen;
        return GemmCoord{mActual, nActual, kActual};
    }
};


/// Block swizzleing function for single core splitk matmul
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct SingleCoreSplitkGemmIdentityBlockSwizzle {
    /// Data members
    static_assert(SwizzleDirection == 0 || SwizzleDirection == 1, "SwizzleDirection only support 0/1");

    GemmCoord problemShape;
    GemmCoord tileMNK;
    GemmCoord loopsMNK;
    uint32_t mnLoopRangeStart;
    uint32_t mnLoopRangeLen;

    /// Methods
    CATLASS_DEVICE
    SingleCoreSplitkGemmIdentityBlockSwizzle() {}

    /// Methods
    CATLASS_DEVICE
    SingleCoreSplitkGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, GemmCoord const &l1TileShape)
        : problemShape(problemShape_), tileMNK(l1TileShape)
    {
        loopsMNK = CeilDiv(problemShape, tileMNK);
        GetMNLoopRange(AscendC::GetBlockNum(), AscendC::GetBlockIdx());
    }

    CATLASS_DEVICE
    uint32_t GetSingleCoreLoops() const
    {
        return loopsMNK.k() * mnLoopRangeLen;
    }

    CATLASS_DEVICE
    void GetMNLoopRange(uint32_t coreNum, uint32_t coreIdx)
    {
        uint32_t mnLoops = loopsMNK.m() * loopsMNK.n();
        uint32_t bPerCore = CeilDiv(mnLoops, coreNum);
        // the first tmpNum cores process bPerCore blocks
        // and the last (coreNum - tmpNum) cores process (bPerCore - 1) blocks
        // total mnLoops = bPerCore * tmpNum + (bPerCore - 1) * (coreNum - tmpNum)
        uint32_t tmpNum = mnLoops - coreNum * (bPerCore - 1);

        mnLoopRangeStart = (coreIdx < tmpNum) ? (coreIdx * bPerCore)
            : (tmpNum * bPerCore + (coreIdx - tmpNum) * (bPerCore - 1));
        mnLoopRangeLen = (coreIdx < tmpNum) ? bPerCore : (bPerCore - 1);
    }

    CATLASS_DEVICE
    bool IsAtomicAdd(uint32_t loopIdx)
    {
        uint32_t kIdx = (loopIdx / mnLoopRangeLen);
        return kIdx != 0;
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t loopIdx)
    {
        uint32_t kIdx = (loopIdx / mnLoopRangeLen);
        // mnLoopIdx
        uint32_t innerIdx = mnLoopRangeStart + loopIdx % mnLoopRangeLen;
        if constexpr (SwizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.n());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.n());
            uint32_t tailOffset = loopsMNK.m() - (tileBlockLoop - 1) * SwizzleOffset;
            uint32_t nRow = SwizzleOffset;

            if (tileBlockLoop >= 2 && tileBlockIdx >= tileBlockLoop - 2 && tailOffset <= SwizzleOffset / 2) {
                tileBlockLoop = (loopsMNK.m() - tailOffset) / SwizzleOffset;
                tileBlockIdx = tileBlockLoop - 1;
                nRow = SwizzleOffset + tailOffset;
                inTileBlockIdx = innerIdx - SwizzleOffset * loopsMNK.n() * tileBlockIdx;
            } else if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - SwizzleOffset * tileBlockIdx;
            }

            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMNK.n() - nIdx - 1;
            }

            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            if (nIdx % 2 == 1) {
                mIdx = tileBlockIdx * SwizzleOffset + (nRow - inTileBlockIdx % nRow - 1);
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        } else if constexpr (SwizzleDirection == 1) { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.m());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.m());
            uint32_t tailOffset = loopsMNK.n() - (tileBlockLoop - 1) * SwizzleOffset;
            uint32_t nCol = SwizzleOffset;

            if (tileBlockLoop >= 2 && tileBlockIdx >= tileBlockLoop - 2 && tailOffset < SwizzleOffset / 2) {
                tileBlockLoop = (loopsMNK.n() - tailOffset) / SwizzleOffset;
                tileBlockIdx = tileBlockLoop - 1;
                nCol = SwizzleOffset + tailOffset;
                inTileBlockIdx = innerIdx - SwizzleOffset * loopsMNK.m() * tileBlockIdx;
            } else if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - SwizzleOffset * tileBlockIdx;
            }

            uint32_t mIdx = inTileBlockIdx / nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }

            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (mIdx % 2 == 1) {
                nIdx = tileBlockIdx * SwizzleOffset + (nCol - inTileBlockIdx % nCol - 1);
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        }
    }

    CATLASS_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord &blockCoord)
    {
        uint32_t mActual = (blockCoord.m() == (loopsMNK.m() - 1)) ?
            (problemShape.m() - blockCoord.m() * tileMNK.m()) : tileMNK.m();
        uint32_t nActual = (blockCoord.n() == (loopsMNK.n() - 1)) ?
            (problemShape.n() - blockCoord.n() * tileMNK.n()) : tileMNK.n();
        uint32_t kActual = (blockCoord.k() == (loopsMNK.k() - 1)) ?
            (problemShape.k() - (loopsMNK.k() - 1) * tileMNK.k()) : tileMNK.k();
        return GemmCoord{mActual, nActual, kActual};
    }

};

struct DynamicSingleCoreSplitkGemmIdentityBlockSwizzle: public SingleCoreSplitkGemmIdentityBlockSwizzle<> {
    uint32_t swizzleOffset;
    uint32_t swizzleDirection;

    CATLASS_DEVICE
    DynamicSingleCoreSplitkGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, GemmCoord const &l1TileShape,
        uint32_t swizzleOffset_, uint32_t swizzleDirection_)
        : swizzleOffset(swizzleOffset_), swizzleDirection(swizzleDirection_),
        SingleCoreSplitkGemmIdentityBlockSwizzle<>(problemShape_, l1TileShape) {}

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t loopIdx)
    {
        uint32_t kIdx = (loopIdx / mnLoopRangeLen);
        // mnLoopIdx
        uint32_t innerIdx = mnLoopRangeStart + loopIdx % mnLoopRangeLen;
        if (swizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * loopsMNK.n());
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * loopsMNK.n());
            uint32_t tailOffset = loopsMNK.m() - (tileBlockLoop - 1) * swizzleOffset;
            uint32_t nRow = swizzleOffset;

            if (tileBlockLoop >= 2 && tileBlockIdx >= tileBlockLoop - 2 && tailOffset <= swizzleOffset / 2) {
                tileBlockLoop = (loopsMNK.m() - tailOffset) / swizzleOffset;
                tileBlockIdx = tileBlockLoop - 1;
                nRow = swizzleOffset + tailOffset;
                inTileBlockIdx = innerIdx - swizzleOffset * loopsMNK.n() * tileBlockIdx;
            } else if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - swizzleOffset * tileBlockIdx;
            }

            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMNK.n() - nIdx - 1;
            }

            uint32_t mIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nRow;
            if (nIdx % 2 == 1) {
                mIdx = tileBlockIdx * swizzleOffset + (nRow - inTileBlockIdx % nRow - 1);
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        } else { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * loopsMNK.m());
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * loopsMNK.m());
            uint32_t tailOffset = loopsMNK.n() - (tileBlockLoop - 1) * swizzleOffset;
            uint32_t nCol = swizzleOffset;

            if (tileBlockLoop >= 2 && tileBlockIdx >= tileBlockLoop - 2 && tailOffset < swizzleOffset / 2) {
                tileBlockLoop = (loopsMNK.n() - tailOffset) / swizzleOffset;
                tileBlockIdx = tileBlockLoop - 1;
                nCol = swizzleOffset + tailOffset;
                inTileBlockIdx = innerIdx - swizzleOffset * loopsMNK.m() * tileBlockIdx;
            } else if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - swizzleOffset * tileBlockIdx;
            }

            uint32_t mIdx = inTileBlockIdx / nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }

            uint32_t nIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nCol;
            if (mIdx % 2 == 1) {
                nIdx = tileBlockIdx * swizzleOffset + (nCol - inTileBlockIdx % nCol - 1);
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        }
    }
};

/// Block swizzling function for Gemms
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct GemmIdentityBlockSwizzleL1FullLoad {
    /// Data members

    GemmCoord problemShape;
    MatrixCoord tileMN;
    MatrixCoord loopsMN;

    uint32_t loopsPerCore;
    uint32_t loopsTail;
    uint32_t aicCoreNum;

    /// Methods

    CATLASS_DEVICE
    GemmIdentityBlockSwizzleL1FullLoad() {}

    CATLASS_DEVICE
    GemmIdentityBlockSwizzleL1FullLoad(GemmCoord const &problemShape_, MatrixCoord const &tileMN_)
        : problemShape(problemShape_), tileMN(tileMN_)
    {
        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
        uint32_t loopsTotalNum = GetCoreLoops();
        aicCoreNum = AscendC::GetBlockNum();
        loopsPerCore = loopsTotalNum / aicCoreNum;
        loopsTail = loopsTotalNum % aicCoreNum;
    }

    CATLASS_DEVICE
    GemmIdentityBlockSwizzleL1FullLoad(GemmCoord const &problemShape_, MatrixCoord const &tileMN_,
        MatrixCoord const &loopsMN_)
        : problemShape(problemShape_), tileMN(tileMN_), loopsMN(loopsMN_)
    {
        uint32_t loopsTotalNum = GetCoreLoops();
        aicCoreNum = AscendC::GetBlockNum();
        loopsPerCore = loopsTotalNum / aicCoreNum;
        loopsTail = loopsTotalNum % aicCoreNum;
    }

    CATLASS_DEVICE
    void Update(GemmCoord const &problemShape_, MatrixCoord const &tileMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;
        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);

        uint32_t loopsTotalNum = GetCoreLoops();
        aicCoreNum = AscendC::GetBlockNum();
        loopsPerCore = loopsTotalNum / aicCoreNum;
        loopsTail = loopsTotalNum % aicCoreNum;
    }

    CATLASS_DEVICE
    void Update(GemmCoord const &problemShape_, MatrixCoord const &tileMN_, MatrixCoord const &loopsMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;
        loopsMN = loopsMN_;

        uint32_t loopsTotalNum = GetCoreLoops();
        aicCoreNum = AscendC::GetBlockNum();
        loopsPerCore = loopsTotalNum / aicCoreNum;
        loopsTail = loopsTotalNum % aicCoreNum;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    ////// WARNING: current strategy not support GetBatchIdx()

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        // calculate innerIdx from taskIdx
        uint32_t CoreIdx = taskIdx % aicCoreNum;
        uint32_t innerCoreIdx = taskIdx / aicCoreNum;
        uint32_t innerIdx = CoreIdx * loopsPerCore + innerCoreIdx;
        if (CoreIdx < loopsTail) {
            innerIdx += CoreIdx;
        } else {
            innerIdx += loopsTail;
        }
        // calculate block location in swizzle, using innerIdx
        if constexpr (SwizzleDirection == 0) { // Zn
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
            return GemmCoord{mIdx, nIdx, 0};
        } else if constexpr (SwizzleDirection == 1) { // Nz
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
            return GemmCoord{mIdx, nIdx, 0};
        }
    }

    CATLASS_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord blockCoord)
    {
        uint32_t mActual = (blockCoord.m() == (loopsMN.row() - 1)) ?
            (problemShape.m() - blockCoord.m() * tileMN.row()) : tileMN.row();
        uint32_t nActual = (blockCoord.n() == (loopsMN.column() - 1)) ?
            (problemShape.n() - blockCoord.n() * tileMN.column()) : tileMN.column();
        uint32_t kActual = problemShape.k();
        return GemmCoord{mActual, nActual, kActual};
    }
};

struct DynamicSplitkGemmIdentityBlockSwizzle : public SplitkGemmIdentityBlockSwizzle<> {

    uint32_t swizzleOffset{1};
    uint32_t swizzleDirection{0};

    CATLASS_DEVICE
    DynamicSplitkGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, GemmCoord const &tileMNK_, 
        uint32_t splitkFactor_, uint32_t swizzleOffset_, uint32_t swizzleDirection_) : 
        swizzleOffset(swizzleOffset_), swizzleDirection(swizzleDirection_), 
        SplitkGemmIdentityBlockSwizzle<>(problemShape_, tileMNK_, splitkFactor_) {}

    CATLASS_DEVICE
    DynamicSplitkGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, GemmCoord const &tileMNK_, 
        uint32_t splitkFactor_) : SplitkGemmIdentityBlockSwizzle<>(problemShape_, tileMNK_, splitkFactor_) {}

    CATLASS_DEVICE
    DynamicSplitkGemmIdentityBlockSwizzle() {}


    CATLASS_DEVICE
    void SetSwizzleParams(uint32_t swizzleOffset_, uint32_t swizzleDirection_)
    {
        swizzleOffset = swizzleOffset_;
        swizzleDirection = swizzleDirection_;
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t splitkSliceIdx = GetSplitkSliceIdx(taskIdx);
        uint32_t kIdx = GetKIdxBySplitkSliceIdx(splitkSliceIdx);

        uint32_t innerIdx = taskIdx % (loopsMNK.m() * loopsMNK.n());
        if (swizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * loopsMNK.n());
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * loopsMNK.n());

            uint32_t nRow = swizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - swizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMNK.n() - nIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        } else { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * loopsMNK.m());
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * loopsMNK.m());

            uint32_t nCol = swizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - swizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        }
    }

};

/// Block swizzling function for streamk Gemms
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct StreamkGemmIdentityBlockSwizzle {
    /// Data members

    GemmCoord problemShape;
    GemmCoord tileMNK;
    GemmCoord loopsMNK;
    uint32_t blockDim;
    // Represents the number of ordinary blocks.
    uint32_t normalBlocks;
    // Represents the number of blocks that need to be split according to the Streamk algorithm.
    uint32_t streamkBlocks;
    // If computing streamk blocks, how many kTile should each core process
    uint32_t kTileNumPerCore;
    // Since the number of kTiles may not be evenly divisible by the number of cores,
    // some cores need to compute one additoinal kTile compared to others
    uint32_t kTileRemain;

    struct StreamkBlockDec {
        // If the current block is a ordinary block, set it to false.
        // If it is a streamk block, set it to true.
        bool isStreamkBlock{false};
        // If the current block is a cross block-meaning it consists partly of the tail k portion of the
        // current basic block and partly of the head k portion of the next basic block, set it to true.
        bool isCrossBlock{false};

        GemmCoord blockCoord;
        GemmCoord actualBlockShape;
        GemmCoord streamkBlockCoord;
        GemmCoord streamkActualBlockShape;
    };


    /// Methods

    CATLASS_DEVICE
    StreamkGemmIdentityBlockSwizzle() {}

    CATLASS_DEVICE
    StreamkGemmIdentityBlockSwizzle(
        GemmCoord const &problemShape_, GemmCoord const &tileMNK_, uint32_t blockDim_
    ) : problemShape(problemShape_), tileMNK(tileMNK_), blockDim(blockDim_)
    {
        loopsMNK = CeilDiv(problemShape, tileMNK);
        streamkBlocks = loopsMNK.m() * loopsMNK.n() % blockDim;
        normalBlocks = loopsMNK.m() * loopsMNK.n() - streamkBlocks;
        kTileNumPerCore = streamkBlocks * loopsMNK.k() / blockDim;
        kTileRemain = streamkBlocks * loopsMNK.k() % blockDim;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        uint32_t blocksNum = loopsMNK.m() * loopsMNK.n();
        uint32_t coreLoops = blocksNum / blockDim * blockDim + min(streamkBlocks * loopsMNK.k(), blockDim);
        return coreLoops;
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        if constexpr (SwizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.n());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.n());

            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMNK.n() - nIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, 0};
        } else if constexpr (SwizzleDirection == 1) { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.m());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.m());

            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, 0};
        }
    }

    CATLASS_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord blockCoord)
    {
        uint32_t mActual = 
            (blockCoord.m() == (loopsMNK.m() - 1)) ? (problemShape.m() - blockCoord.m() * tileMNK.m()) : tileMNK.m();
        uint32_t nActual = 
            (blockCoord.n() == (loopsMNK.n() - 1)) ? (problemShape.n() - blockCoord.n() * tileMNK.n()) : tileMNK.n();
        uint32_t kActual = problemShape.k();
        return GemmCoord{mActual, nActual, kActual};
    }

    CATLASS_DEVICE
    uint32_t GetStreamkBlockNum() { return streamkBlocks; }

    CATLASS_DEVICE
    uint32_t GetNormalBlockNum() { return normalBlocks; }

    CATLASS_DEVICE
    void GetStreamkBlockDec(uint32_t loopIdx, StreamkBlockDec &streamkBlockDec)
    {
        if (loopIdx < normalBlocks) {
            streamkBlockDec.isStreamkBlock = false;
            streamkBlockDec.isCrossBlock = false;
            streamkBlockDec.blockCoord = GetBlockCoord(loopIdx);
            streamkBlockDec.actualBlockShape = GetActualBlockShape(streamkBlockDec.blockCoord);
        } else {
            streamkBlockDec.isStreamkBlock = true;
            uint32_t curKTileNum = kTileNumPerCore;
            uint32_t kTileIdx = ((loopIdx - normalBlocks) * kTileNumPerCore + kTileRemain);
            if (loopIdx - normalBlocks < kTileRemain) {
                curKTileNum += 1;
                kTileIdx = ((loopIdx - normalBlocks) * curKTileNum);
            }
            uint32_t streamkBlockIdx = kTileIdx / loopsMNK.k();
            GemmCoord blockCoordTmp = GetBlockCoord(normalBlocks + streamkBlockIdx);
            streamkBlockDec.blockCoord = GemmCoord{blockCoordTmp.m(), blockCoordTmp.n(), kTileIdx % loopsMNK.k()};

            uint32_t nextStreamkBlockIdx = (kTileIdx + curKTileNum) / loopsMNK.k();
            streamkBlockDec.isCrossBlock = false;
            GemmCoord actualBlockShapeTmp = GetActualBlockShape(streamkBlockDec.blockCoord);
            if ((kTileIdx % loopsMNK.k() + curKTileNum) * tileMNK.k() > problemShape.k()) {
                streamkBlockDec.actualBlockShape = GemmCoord{actualBlockShapeTmp.m(), actualBlockShapeTmp.n(),
                    problemShape.k() - (kTileIdx % loopsMNK.k()) * tileMNK.k()};
            } else {
                streamkBlockDec.actualBlockShape = 
                    GemmCoord{actualBlockShapeTmp.m(), actualBlockShapeTmp.n(), curKTileNum * tileMNK.k()};
            }
            if (kTileIdx % loopsMNK.k() + curKTileNum > loopsMNK.k()) {
                streamkBlockDec.isCrossBlock = true;
                streamkBlockDec.streamkBlockCoord = GetBlockCoord(normalBlocks + nextStreamkBlockIdx);
                GemmCoord streamkActualBlockShapeTmp = GetActualBlockShape(streamkBlockDec.streamkBlockCoord);
                streamkBlockDec.streamkActualBlockShape = GemmCoord{streamkActualBlockShapeTmp.m(),
                    streamkActualBlockShapeTmp.n(), ((kTileIdx + curKTileNum) % loopsMNK.k()) * tileMNK.k()};
            }
        }
    }

    CATLASS_DEVICE
    uint32_t GetCoreIdx(uint32_t streamkBlockIdx)
    {
        if (streamkBlockIdx * loopsMNK.k() > kTileRemain * (kTileNumPerCore + 1)) {
            return kTileRemain +
                (streamkBlockIdx * loopsMNK.k() - kTileRemain * (kTileNumPerCore + 1)) / kTileNumPerCore;
        } else {
            return streamkBlockIdx * loopsMNK.k() / (kTileNumPerCore + 1);
        }
    }

    CATLASS_DEVICE
    bool IsCross(uint32_t streamkBlockIdx)
    {
        // If the head part of the current block and the tail part of the previous block are computed by
        // the same core, return true.
        if (streamkBlockIdx * loopsMNK.k() > kTileRemain * (kTileNumPerCore + 1)) {
            if ((streamkBlockIdx * loopsMNK.k() - kTileRemain * (kTileNumPerCore + 1)) % kTileNumPerCore == 0) {
                return false;
            } else {
                return true;
            }
        } else {
            if ((streamkBlockIdx * loopsMNK.k()) % (kTileNumPerCore + 1) == 0) {
                return false;
            } else {
                return true;
            }
        }
    }
};

struct DynamicStreamkGemmIdentityBlockSwizzle : public StreamkGemmIdentityBlockSwizzle<> {

    uint32_t swizzleOffset{1};
    uint32_t swizzleDirection{0};

    CATLASS_DEVICE
    DynamicStreamkGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, GemmCoord const &tileShape_, 
        uint32_t coreNum_, uint32_t swizzleOffset_, uint32_t swizzleDirection_) : 
        swizzleOffset(swizzleOffset_), swizzleDirection(swizzleDirection_), 
        StreamkGemmIdentityBlockSwizzle<>(problemShape_, tileShape_, coreNum_) {}

    CATLASS_DEVICE
    DynamicStreamkGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, GemmCoord const &tileShape_, 
        uint32_t coreNum_) 
        : StreamkGemmIdentityBlockSwizzle<>(problemShape_, tileShape_, coreNum_) {}

    CATLASS_DEVICE
    DynamicStreamkGemmIdentityBlockSwizzle() {}


    CATLASS_DEVICE
    void SetSwizzleParams(uint32_t swizzleOffset_, uint32_t swizzleDirection_)
    {
        swizzleOffset = swizzleOffset_;
        swizzleDirection = swizzleDirection_;
    }

    CATLASS_DEVICE
    void GetStreamkBlockDec(uint32_t loopIdx, StreamkBlockDec &streamkBlockDec)
    {
        if (loopIdx < normalBlocks) {
            streamkBlockDec.isStreamkBlock = false;
            streamkBlockDec.isCrossBlock = false;
            streamkBlockDec.blockCoord = GetBlockCoord(loopIdx);
            streamkBlockDec.actualBlockShape = GetActualBlockShape(streamkBlockDec.blockCoord);
        } else {
            streamkBlockDec.isStreamkBlock = true;
            uint32_t curKTileNum = kTileNumPerCore;
            uint32_t kTileIdx = ((loopIdx - normalBlocks) * kTileNumPerCore + kTileRemain);
            if (loopIdx - normalBlocks < kTileRemain) {
                curKTileNum += 1;
                kTileIdx = ((loopIdx - normalBlocks) * curKTileNum);
            }
            uint32_t streamkBlockIdx = kTileIdx / loopsMNK.k();
            GemmCoord blockCoordTmp = GetBlockCoord(normalBlocks + streamkBlockIdx);
            streamkBlockDec.blockCoord = GemmCoord{blockCoordTmp.m(), blockCoordTmp.n(), kTileIdx % loopsMNK.k()};

            uint32_t nextStreamkBlockIdx = (kTileIdx + curKTileNum) / loopsMNK.k();
            streamkBlockDec.isCrossBlock = false;
            GemmCoord actualBlockShapeTmp = GetActualBlockShape(streamkBlockDec.blockCoord);
            if ((kTileIdx % loopsMNK.k() + curKTileNum) * tileMNK.k() > problemShape.k()) {
                streamkBlockDec.actualBlockShape = GemmCoord{actualBlockShapeTmp.m(), actualBlockShapeTmp.n(),
                    problemShape.k() - (kTileIdx % loopsMNK.k()) * tileMNK.k()};
            } else {
                streamkBlockDec.actualBlockShape = 
                    GemmCoord{actualBlockShapeTmp.m(), actualBlockShapeTmp.n(), curKTileNum * tileMNK.k()};
            }
            if (kTileIdx % loopsMNK.k() + curKTileNum > loopsMNK.k()) {
                streamkBlockDec.isCrossBlock = true;
                streamkBlockDec.streamkBlockCoord = GetBlockCoord(normalBlocks + nextStreamkBlockIdx);
                GemmCoord streamkActualBlockShapeTmp = GetActualBlockShape(streamkBlockDec.streamkBlockCoord);
                streamkBlockDec.streamkActualBlockShape = GemmCoord{streamkActualBlockShapeTmp.m(),
                    streamkActualBlockShapeTmp.n(), ((kTileIdx + curKTileNum) % loopsMNK.k()) * tileMNK.k()};
            }
        }
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        if (swizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * loopsMNK.n());
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * loopsMNK.n());

            uint32_t nRow = swizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - swizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMNK.n() - nIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, 0};
        } else { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * loopsMNK.m());
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * loopsMNK.m());

            uint32_t nCol = swizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - swizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, 0};
        }
    }
};

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_BLOCK_BLOCK_SWIZZLE_HPP