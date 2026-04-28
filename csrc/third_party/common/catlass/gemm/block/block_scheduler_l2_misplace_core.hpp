/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_L2_MISPLACE_CORE_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_L2_MISPLACE_CORE_HPP

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Gemm::Block {

/////////////////////////////////////////////////////////////////////////////////////////////////
constexpr uint32_t L2_TILE_THRESHOLD = 100 * 1024 * 1024;
constexpr uint32_t MIN_SPLIT_TILE_NUM = 4;
constexpr uint32_t MAX_SPLIT_TILE_NUM = 8;

template <bool TransA_ = false, bool TransB_ = false>
struct BlockSchedulerL2MisplaceCore {
    /// Data members
    GemmCoord problemShape;
    MatrixCoord tileMN;
    MatrixCoord loopsMN;

    uint32_t mTileNum_{0};
    uint32_t nTileNum_{0};
    uint32_t kTileNum_{0};
    uint32_t blockNum_{0};

    // l2 spilit attribute
    uint32_t newBlockIdx_{0};
    uint32_t mL2TileNumTmp_{0};
    uint32_t nL2TileNumTmp_{0};
    uint32_t nL2Idx_{0};
    uint32_t mL2Idx_{0};
    uint32_t mL2Num_{0};     // l2 m block num
    uint32_t nL2Num_{0};     // l2 n block num
    uint32_t mL2TileNum_{0}; // a1b1 m tile num of one l2 block
    uint32_t nL2TileNum_{0}; // a1b1 n tile num of one l2 block

    static constexpr bool isTransA = TransA_;
    static constexpr bool isTransB = TransB_;

    CATLASS_DEVICE
    BlockSchedulerL2MisplaceCore() {}

    CATLASS_DEVICE
    BlockSchedulerL2MisplaceCore(GemmCoord const &problemShape_, MatrixCoord const &tileMN_)
        : problemShape(problemShape_), tileMN(tileMN_)
    {
        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
        mTileNum_ = loopsMN.row();
        nTileNum_ = loopsMN.column();
        blockNum_ = AscendC::GetBlockNum();
        InitL2Tile();
    }

    CATLASS_DEVICE
    BlockSchedulerL2MisplaceCore(GemmCoord const &problemShape_, MatrixCoord const &tileMN_,
        MatrixCoord const &loopsMN_)
        : problemShape(problemShape_), tileMN(tileMN_), loopsMN(loopsMN_) {}

    CATLASS_DEVICE
    uint32_t GetTotalSize(uint32_t mL2, uint32_t nL2, uint32_t kL2)
    {
        uint32_t sizeA = mL2 * kL2 * sizeof(half);
        uint32_t sizeB = kL2 * nL2 * sizeof(half);
        uint32_t sizeC = mL2 * nL2 * sizeof(half);
        return sizeA + sizeB + sizeC;
    }

    CATLASS_DEVICE
    bool EnableL2Tile()
    {
        return GetTotalSize(problemShape.m(), problemShape.n(), problemShape.k()) > L2_TILE_THRESHOLD;
    }

    CATLASS_DEVICE
    void InitL2TileTail()
    {
        uint32_t mConflict = INT32_MAX;
        uint32_t nConflict = INT32_MAX;
        uint32_t l1M = tileMN.row();
        uint32_t l1N = tileMN.column();

        bool isInnerBad = false;
        uint32_t maxi = 0;
        uint32_t maxj = 0;
        if (l1N > l1M) {
            isInnerBad = isTransA;
            maxi = (blockNum_ > nTileNum_) ? nTileNum_ : blockNum_;
            maxj = (blockNum_ > mTileNum_) ? mTileNum_ : blockNum_;
        } else {
            isInnerBad = !isTransB;
            maxi = (blockNum_ > mTileNum_) ? mTileNum_ : blockNum_;
            maxj = (blockNum_ > nTileNum_) ? nTileNum_ : blockNum_;
        }
        uint32_t innerMinUseDim = isInnerBad ? MAX_SPLIT_TILE_NUM : MIN_SPLIT_TILE_NUM;

        for (uint32_t i = maxi; i >= MIN_SPLIT_TILE_NUM; i--) { // if l1N greater than l1M, indicates n
            for (uint32_t j = maxj; j >= innerMinUseDim; j--) {
                uint32_t mL2TileNumTmp = (l1N > l1M) ? j : i;
                uint32_t nL2TileNumTmp = (l1N > l1M) ? i : j;
                if (GetTotalSize(mL2TileNumTmp * l1M, nL2TileNumTmp * l1N, problemShape.k()) <= L2_TILE_THRESHOLD) {
                    uint32_t mL2TileNumTailTmp = GetTailNum(mTileNum_, mL2TileNumTmp);
                    uint32_t nL2TileNumTailTmp = GetTailNum(nTileNum_, nL2TileNumTmp);

                    uint32_t mConflictTmp = CeilDiv(blockNum_, mL2TileNumTailTmp);
                    uint32_t nConflictTmp = CeilDiv(blockNum_, nL2TileNumTailTmp);
                    if (mConflict >= mConflictTmp && nConflict >= nConflictTmp) {
                        mConflict = mConflictTmp;
                        nConflict = nConflictTmp;
                        mL2TileNum_ = mL2TileNumTmp;
                        nL2TileNum_ = nL2TileNumTmp;
                    }
                }
            }
        }
        if (mL2TileNum_ == 0 || nL2TileNum_ == 0) {
            mL2TileNum_ = mTileNum_;
            nL2TileNum_ = nTileNum_;
        }
    }

    CATLASS_DEVICE
    void InitL2Tile()
    {
        int64_t l1M = static_cast<int64_t>(tileMN.row());
        int64_t l1N = static_cast<int64_t>(tileMN.column());
        int64_t k = static_cast<int64_t>(problemShape.k());

        if ((mTileNum_ < MIN_SPLIT_TILE_NUM && nTileNum_ < MIN_SPLIT_TILE_NUM) || (!EnableL2Tile())) {
            mL2TileNum_ = mTileNum_;
            nL2TileNum_ = nTileNum_;
            mL2Num_ = 1;
            nL2Num_ = 1;
            return;
        }

        InitL2TileTail();

        mL2Num_ = CeilDiv(mTileNum_, mL2TileNum_);
        nL2Num_ = CeilDiv(nTileNum_, nL2TileNum_);
    }

    CATLASS_DEVICE
    void GetCommonTileIndex(uint32_t tileIdx)
    {
        uint32_t batchTileIdx = tileIdx / (nTileNum_ * mTileNum_);
        if (batchTileIdx != 0) {
            tileIdx = tileIdx - batchTileIdx * nTileNum_ * mTileNum_;
        }
        mL2Idx_ = tileIdx / (mL2TileNum_ * nTileNum_);
        mL2TileNumTmp_ = (mL2Idx_ == mL2Num_ - 1) ? GetTailNum(mTileNum_, mL2TileNum_) : mL2TileNum_;

        nL2Idx_ = (tileIdx % (mL2TileNum_ * nTileNum_)) / (mL2TileNumTmp_ * nL2TileNum_);
        nL2TileNumTmp_ = (nL2Idx_ == nL2Num_ - 1) ? GetTailNum(nTileNum_, nL2TileNum_) : nL2TileNum_;

        uint32_t startIdx = mL2Idx_ * mL2TileNum_ * nTileNum_ + nL2Idx_ * nL2TileNum_ * mL2TileNumTmp_;
        newBlockIdx_ = tileIdx - startIdx;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    CATLASS_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord blockCoord)
    {
        uint32_t mTileIdx = blockCoord.m();
        uint32_t nTileIdx = blockCoord.n();

        // calc tail l1block mnk
        uint32_t tailL1M = (problemShape.m() % tileMN.row() == 0) ? tileMN.row() : problemShape.m() % tileMN.row();
        uint32_t tailL1N = (problemShape.n() % tileMN.column() == 0) ? tileMN.column() : problemShape.n() % tileMN.column();
        uint32_t blockShapeM = IsMTail(mTileIdx, mTileNum_) ? tailL1M : tileMN.row();
        uint32_t blockShapeN = IsNTail(nTileIdx, nTileNum_) ? tailL1N : tileMN.column();

        return {blockShapeM, blockShapeN, problemShape.k()};
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t tileIdx)
    {
        uint32_t batchTileIdx = tileIdx / (nTileNum_ * mTileNum_);
        GetCommonTileIndex(tileIdx);
        uint32_t mTileIdx = newBlockIdx_ % mL2TileNumTmp_;
        mTileIdx = mTileIdx + mL2Idx_ * mL2TileNum_;

        uint32_t nTileIdx = 0;
        if (nL2TileNumTmp_ != 0) {
            uint32_t tmp = newBlockIdx_ / MMLcm(mL2TileNumTmp_, nL2TileNumTmp_);
            nTileIdx = (newBlockIdx_ + tmp) % nL2TileNumTmp_;
        }
        nTileIdx = nTileIdx + nL2Idx_ * nL2TileNum_;

        return {mTileIdx, nTileIdx, 0};
    }

    CATLASS_DEVICE
    uint32_t GetTailNum(uint32_t totalNum, uint32_t normalNum)
    {
        if (normalNum == 0) {
            return 0;
        }
        if (totalNum % normalNum == 0) {
            return normalNum;
        }
        return totalNum % normalNum;
    }

    CATLASS_DEVICE
    uint32_t MMLcm(uint32_t m, uint32_t n)
    {
        if (m == 0 || n == 0) {
            return 0;
        }
        uint64_t total = m * n;
        uint32_t tmp = 0;
        // calc maximum common divisor m
        while (n != 0) {
            tmp = m % n;
            m = n;
            n = tmp;
        }
        return static_cast<uint32_t>(total / m);
    }

    CATLASS_DEVICE bool IsMTail(uint32_t mTileIdx, uint32_t mTileNum)
    {
        return (mTileIdx - (mTileNum - 1)) % mTileNum == 0;
    }

    CATLASS_DEVICE bool IsNTail(uint32_t nTileIdx, uint32_t nTileNum)
    {  
        return nTileIdx == (nTileNum - 1);  
    }
};

}  // namespace Catlass::Gemm::Block

#endif  // CATLASS_GEMM_BLOCK_BLOCK_SCHEDULER_L2_MISPLACE_CORE_HPP