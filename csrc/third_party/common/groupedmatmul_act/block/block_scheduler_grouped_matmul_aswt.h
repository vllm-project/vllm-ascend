/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_scheduler_grouped_matmul_aswt.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_SCHEDULER_GROUPED_MATMUL_ASWT_H
#define MATMUL_BLOCK_BLOCK_SCHEDULER_GROUPED_MATMUL_ASWT_H
#include "./block_scheduler_utils.h"
#include "./block_scheduler_policy.h"
#include "../utils/status_utils.h"

namespace Act {
namespace Gemm {
namespace Block {

#ifndef TILING_TYPE
#if defined(CONST_TILING)
#define TILING_TYPE const int32_t
#else
#define TILING_TYPE __gm__ int32_t
#endif
#endif // TILING_TYPE

template <class ProblemShape_, class L1TileShape_, class L0TileShape_>
class BlockSchedulerGroupedMatmulAswt {
private:
    const int64_t WINDOW_LEN = 4;
    const int64_t EVEN_ROWS = 2;

public:
    int64_t mTileNum_{0};
    int64_t nTileNum_{0};
    int64_t totalTileNum_{0};

    int64_t blockNum_{0};
    int64_t blockIdx_{0};
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t b_{1};
    int32_t baseM_{1};
    int32_t baseN_{1};
    int32_t baseK_{1};
    uint64_t mTailCnt_{1};
    uint64_t nTailCnt_{1};
    uint64_t tailCnt_{1};
    int64_t perCoreBlockNum_{0};
    int64_t mainWindow_{0};
    int64_t tailWindow_{0};
    int64_t mainRow_{0};
    bool tailSplit_{false};

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    static constexpr int64_t l1M = GetIntegralConstant<MNK_M, L1TileShape_>();
    static constexpr int64_t l1N = GetIntegralConstant<MNK_N, L1TileShape_>();
    static constexpr int64_t l1K = GetIntegralConstant<MNK_K, L1TileShape_>();
    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape_>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape_>();
    static constexpr int64_t l0K = GetIntegralConstant<MNK_K, L0TileShape_>();

public:
    __aicore__ inline BlockSchedulerGroupedMatmulAswt(int64_t m, int64_t n, int64_t k, int32_t baseM, int32_t baseN,
                                                      int32_t baseK, int64_t blockIdx, int64_t blockNum,
                                                      uint64_t mTailCnt, uint64_t nTailCnt, bool tailSplit)
        : m_(m), n_(n), k_(k), baseM_(baseM), baseN_(baseN), baseK_(baseK), blockNum_(blockNum), blockIdx_(blockIdx),
          mTailCnt_(mTailCnt), nTailCnt_(nTailCnt), tailSplit_(tailSplit)
    {
        mTileNum_ = Act::Gemm::CeilDiv(m_, baseM_);
        nTileNum_ = Act::Gemm::CeilDiv(n_, baseN_);
        perCoreBlockNum_ = GetPerBlockNum(blockNum_, mTileNum_, nTileNum_, b_);
        totalTileNum_ = mTileNum_ * nTileNum_;
        if ((mTailCnt_ > 1 || nTailCnt_ > 1) && tailSplit) {
            tailCnt_ = mTailCnt_ * nTailCnt_;
            totalTileNum_ += (tailCnt_ - 1) * (totalTileNum_ % blockNum_);
        }
        mainWindow_ = WINDOW_LEN < mTileNum_ ? WINDOW_LEN : mTileNum_;
        mainRow_ = mTileNum_ / mainWindow_ - 1;
        tailWindow_ = mTileNum_ - mainWindow_ * mainRow_;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return totalTileNum_;
    }

    __aicore__ inline BlockShape GetTileIdx(int64_t curBlock, int64_t count)
    {
        uint64_t index = curBlock - count;
        uint64_t mTileIdx = 0;
        uint64_t nTileIdx = 0;
        if (index / blockNum_ == (perCoreBlockNum_ - 1) && tailCnt_ > 1) {
            index = (perCoreBlockNum_ - 1) * blockNum_ + blockIdx_ / tailCnt_;
        }
        uint64_t rowIdx = index / nTileNum_ / mainWindow_;
        if (rowIdx < mainRow_) {
            mTileIdx = rowIdx * mainWindow_ + index % mainWindow_;
            nTileIdx = (index / mainWindow_) % nTileNum_;
        } else {
            rowIdx = mainRow_;
            uint64_t tailIndex = index - mainRow_ * mainWindow_ * nTileNum_;
            mTileIdx = mainRow_ * mainWindow_ + tailIndex % tailWindow_;
            nTileIdx = (tailIndex / tailWindow_) % nTileNum_;
        }
        if (rowIdx % EVEN_ROWS != 0) { // Reverse computation for even-numbered rows
            nTileIdx = nTileNum_ - 1 - nTileIdx;
        }
        return {mTileIdx, nTileIdx, k_, b_};
    }

    __aicore__ inline BlockShape GetBlockShape(int64_t mTileIdx, int64_t nTileIdx, int64_t cureBlock, int64_t c0,
                                               bool weightNzFlag = false)
    {
        int64_t tailL1M = (m_ % baseM_ == 0) ? baseM_ : m_ % baseM_;
        int64_t tailL1N = (n_ % baseN_ == 0) ? baseN_ : n_ % baseN_;
        int64_t blockShapeM = IsMTail(mTileIdx, mTileNum_) ? tailL1M : baseM_;
        int64_t blockShapeN = IsNTail(nTileIdx, nTileNum_) ? tailL1N : baseN_;
        int64_t mSplitAddrOffset = 0;
        int64_t nSplitAddrOffset = 0;
        if (cureBlock / blockNum_ != (perCoreBlockNum_ - 1) || tailCnt_ == 1) {
            return {blockShapeM, blockShapeN, mSplitAddrOffset, nSplitAddrOffset};
        }
        int64_t singleCoreMSplit = Act::Gemm::CeilDiv(blockShapeM, static_cast<int64_t>(mTailCnt_));
        int64_t singleCoreNSplit = Act::Gemm::CeilDiv(blockShapeN, static_cast<int64_t>(nTailCnt_));
        if (weightNzFlag) {
            singleCoreNSplit = Act::Gemm::CeilAlign(singleCoreNSplit, c0);
        }
        mTailCnt_ = Act::Gemm::CeilDiv(blockShapeM, singleCoreMSplit);
        nTailCnt_ = Act::Gemm::CeilDiv(blockShapeN, singleCoreNSplit);
        int64_t mSplitIdx = (blockIdx_ % tailCnt_) % mTailCnt_;
        int64_t nSplitIdx = (blockIdx_ % tailCnt_) / mTailCnt_;
        mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (mSplitAddrOffset >= blockShapeM || nSplitAddrOffset >= blockShapeN) {
            return {0, 0, 0, 0};
        }
        tailL1M = AscendC::Std::min(blockShapeM - mSplitAddrOffset, singleCoreMSplit);
        tailL1N = AscendC::Std::min(blockShapeN - nSplitAddrOffset, singleCoreNSplit);
        return {tailL1M, tailL1N, mSplitAddrOffset, nSplitAddrOffset};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int64_t mTileIdx, int64_t nTileIdx)
    {
        return {mTileIdx * l1M, nTileIdx * l1N, 0, b_};
    }

    static int64_t GetBlockNum(ProblemShape shape)
    {
        return DoGetBlockNum(l1M, l1N, shape);
    }

    __host_aicore__ static size_t GetWorkspaceSize(ProblemShape shape)
    {
        return 0;
    }

    __host_aicore__ static Status CanImplement(ProblemShape shape)
    {
        return Status::success;
    }
};
template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, Act::Gemm::GroupedMatmulAswtScheduler, TransA_,
                              TransB_> {
    using SchedulerOp = BlockSchedulerGroupedMatmulAswt<ProblemShape_, L1TileShape_, L0TileShape_>;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif