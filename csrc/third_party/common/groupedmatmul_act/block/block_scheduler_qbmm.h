/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACT_QBMM_BLOCK_SCHEDULER_H
#define ACT_QBMM_BLOCK_SCHEDULER_H

#include "./block_scheduler_utils.h"
#include "./block_scheduler_policy.h"
#include "../utils/common_utils.h"

/*!
 * \file qbmm_block_scheduler.h
 * \brief
 */

namespace Act {
namespace Gemm {
namespace Block {

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
class BlockSchedulerQuantBatchMatmulV3 {
public:
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t b_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t mCnt_{0};
    int64_t nCnt_{0};
    int64_t singleBatchCnt_{0};
    int64_t totalCnt_{0};
    int64_t mBaseNormCnt_{0};
    int64_t nBaseNormCnt_{0};
    int64_t mBaseTailMain_{0};
    int64_t nBaseTailMain_{0};
    int64_t mBaseTailLast_{0};
    int64_t nBaseTailLast_{0};
    int64_t mCoreNum_{0};
    int64_t mTailCoreNum_{0};
    int64_t blockIdx_{0};
    int64_t blockNum_{0};
    int64_t startBlockIdx_{0};
    int64_t endBlockIdx_{0};
    int64_t roundIdx_{0};
    int64_t round_{0};
    int64_t mTailTile_{0};
    int64_t nTailTile_{0};
    int64_t totalTailTile_{0};
    int64_t mSplitAddrOffset_{0};
    int64_t nSplitAddrOffset_{0};
    int64_t mainRow_{0};
    int64_t usedCoreNum_{0};

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        int64_t usedCoreNum;
        int64_t baseM;
        int64_t baseN;
        int64_t mTailTile;
        int64_t nTailTile;
        int64_t mBaseTailSplitCnt;
        int64_t nBaseTailSplitCnt;
        int64_t mTailMain;
        int64_t nTailMain;
    };

    const int64_t WINDOW_LEN = 4;

public:
    __aicore__ inline BlockSchedulerQuantBatchMatmulV3(const ProblemShape &shape, const Params &params)
    {
        m_ = shape.m;
        n_ = shape.n;
        k_ = shape.k;
        b_ = shape.b;
        baseM_ = static_cast<int64_t>(params.baseM);
        baseN_ = static_cast<int64_t>(params.baseN);
        mCnt_ = Act::Gemm::CeilDiv(m_, baseM_);
        nCnt_ = Act::Gemm::CeilDiv(n_, baseN_);
        singleBatchCnt_ = mCnt_ * nCnt_;
        totalCnt_ = b_ * singleBatchCnt_;
        mBaseNormCnt_ = mCnt_ - params.mBaseTailSplitCnt;
        nBaseNormCnt_ = nCnt_ - params.nBaseTailSplitCnt;
        int64_t mBaseTail = m_ - mBaseNormCnt_ * baseM_;
        int64_t nBaseTail = n_ - nBaseNormCnt_ * baseN_;
        mBaseTailMain_ = params.mBaseTailSplitCnt == 1 ? mBaseTail : params.mTailMain;
        mBaseTailLast_ = mBaseTail - (params.mBaseTailSplitCnt - 1) * mBaseTailMain_;
        nBaseTailMain_ = params.nBaseTailSplitCnt == 1 ? nBaseTail : params.nTailMain;
        nBaseTailLast_ = nBaseTail - (params.nBaseTailSplitCnt - 1) * nBaseTailMain_;
        mCoreNum_ = Act::Gemm::Min(WINDOW_LEN, mCnt_);
        mainRow_ = mCnt_ / mCoreNum_ - 1;
        mTailCoreNum_ = mCnt_ - mCoreNum_ * mainRow_;
        blockIdx_ = AscendC::GetBlockIdx();
        blockNum_ = AscendC::GetBlockNum();
        endBlockIdx_ = singleBatchCnt_ % blockNum_ - 1;
        roundIdx_ = 0;
        round_ = Act::Gemm::CeilDiv(singleBatchCnt_, blockNum_);
        mTailTile_ = static_cast<int64_t>(params.mTailTile);
        nTailTile_ = static_cast<int64_t>(params.nTailTile);
        totalTailTile_ = mTailTile_ * nTailTile_;
        usedCoreNum_ = params.usedCoreNum;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        if (b_ > 1) {
            return totalCnt_;
        } else {
            int64_t finalRoundTileNum = singleBatchCnt_ % blockNum_;
            return singleBatchCnt_ - finalRoundTileNum + finalRoundTileNum * totalTailTile_;
        }
    }

    __aicore__ inline BlockShape GetBlockShape(BlockCoord blockCoord)
    {
        int64_t singleCoreM = baseM_;
        int64_t singleCoreN = baseN_;
        if (Get<MNK_M>(blockCoord) >= mBaseNormCnt_) {
            singleCoreM = Get<MNK_M>(blockCoord) >= mCnt_ - 1 ? mBaseTailLast_ : mBaseTailMain_;
        }
        if (Get<MNK_N>(blockCoord) >= nBaseNormCnt_) {
            singleCoreN = Get<MNK_N>(blockCoord) >= nCnt_ - 1 ? nBaseTailLast_ : nBaseTailMain_;
        }

        if (totalTailTile_ == 1 || b_ > 1) {
            return {singleCoreM, singleCoreN, 0, 0};
        }

        if (roundIdx_ == round_ - 1) {
            int64_t singleCoreMSplit = Act::Gemm::CeilDiv(singleCoreM, mTailTile_);
            int64_t singleCoreNSplit = Act::Gemm::CeilDiv(singleCoreN, nTailTile_);
            int64_t mSplitIdx = (blockIdx_ % totalTailTile_) % mTailTile_;
            int64_t nSplitIdx = (blockIdx_ % totalTailTile_) / mTailTile_;
            mSplitAddrOffset_ = mSplitIdx * singleCoreMSplit;
            nSplitAddrOffset_ = nSplitIdx * singleCoreNSplit;
            if (mSplitAddrOffset_ >= singleCoreM || nSplitAddrOffset_ >= singleCoreN) {
                singleCoreM = 0;
                singleCoreN = 0;
                return {singleCoreM, singleCoreN, mSplitAddrOffset_, nSplitAddrOffset_};
            }
            singleCoreM = Act::Gemm::Min(singleCoreM - mSplitAddrOffset_, singleCoreMSplit);
            singleCoreN = Act::Gemm::Min(singleCoreN - nSplitAddrOffset_, singleCoreNSplit);
            return {singleCoreM, singleCoreN, mSplitAddrOffset_, nSplitAddrOffset_};
        }
    }

    __aicore__ inline BlockShape GetLoadBalanceInfo()
    {
        return {mBaseNormCnt_, mBaseTailMain_, nBaseNormCnt_, nBaseTailMain_};
    }

    __aicore__ inline void ResetAddrOffsets()
    {
        mSplitAddrOffset_ = 0;
        nSplitAddrOffset_ = 0;
    }

    __aicore__ inline void UpdateNextBatchBlockRoundParams()
    {
        startBlockIdx_ = (endBlockIdx_ + 1) % blockNum_;
        endBlockIdx_ = (singleBatchCnt_ + startBlockIdx_ - 1) % blockNum_;

        roundIdx_ = 0;
        round_ = Act::Gemm::CeilDiv(singleBatchCnt_, blockNum_);
        if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        }
    }

    __aicore__ inline bool GetTileIdx(BlockCoord &blockCoord)
    {
        if (roundIdx_ >= round_) {
            return false;
        }

        int64_t newBlockIdx = (roundIdx_ == round_ - 1) ? blockIdx_ / totalTailTile_ : blockIdx_;
        int64_t tileIdx = newBlockIdx + roundIdx_ * usedCoreNum_;

        int64_t batchIdx = tileIdx / singleBatchCnt_;
        Get<MNK_B>(blockCoord) = batchIdx;

        int64_t inBatchIdx = tileIdx % singleBatchCnt_;
        int64_t rowIdx = inBatchIdx / nCnt_ / mCoreNum_;
        if (rowIdx < mainRow_) {
            Get<MNK_M>(blockCoord) = rowIdx * mCoreNum_ + inBatchIdx % mCoreNum_;
            Get<MNK_N>(blockCoord) = (inBatchIdx / mCoreNum_) % nCnt_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIdx = inBatchIdx - mainRow_ * mCoreNum_ * nCnt_;
            Get<MNK_M>(blockCoord) = mainRow_ * mCoreNum_ + tailIdx % mTailCoreNum_;
            Get<MNK_N>(blockCoord) = (tailIdx / mTailCoreNum_) % nCnt_;
        }
        if (rowIdx & 1) {
            Get<MNK_N>(blockCoord) = nCnt_ - 1 - Get<MNK_N>(blockCoord);
        }

        roundIdx_++;
        return true;
    }
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, Act::Gemm::QuantBatchMatmulV3Scheduler,
                              TransA_, TransB_> {
    using SchedulerOp = BlockSchedulerQuantBatchMatmulV3<ProblemShape_, L1TileShape_, L0TileShape_, TransA_, TransB_>;
};
}  // namespace Block
}  // namespace Gemm
}  // namespace Act
#endif