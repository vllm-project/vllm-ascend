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
 * \file block_scheduler_swizzle_in_mn_core.h
 * \brief
 */

#ifndef ACT_INCLUDE_MATMUL_BLOCK_BLOCK_SCHEDULER_SWIZZLE_IN_MN_CORE_H
#define ACT_INCLUDE_MATMUL_BLOCK_BLOCK_SCHEDULER_SWIZZLE_IN_MN_CORE_H

#include <cstdint>

#include "../utils/common_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tuple_utils.h"

/*
iterateOrder = 0
scheduler diagram, c indicate core,b indicate block
| c0b0 | c0b1 | c0b2 | c2b0 | c2b1 |
------------------------------------
| c0b3 | c0b4 | c0b5 | c2b2 | c2b3 |
------------------------------------
| c1b0 | c1b1 | c1b2 | c3b0 | c3b1 |

iterateOrder = 1
| c0b0 | c0b2 | c0b4 | c2b0 | c2b2 |
------------------------------------
| c0b1 | c0b3 | c0b5 | c2b1 | c2b3 |
------------------------------------
| c1b0 | c1b1 | c1b2 | c3b0 | c3b1 |
*/
namespace Act {
namespace Gemm {
namespace Block {

template <class ProblemShape_, class TileShape_, class BlockShape_>
class BlockSchedulerSwizzleInMnCore {
public:
    // Type Aliases
    using ProblemShape = ProblemShape_;
    using TileShape = TileShape_;
    using BlockShape = BlockShape_;

    struct Arguments {};
    struct Params {
        int32_t iterateOrder;
        ProblemShape problemShape;
        TileShape tileShape;
        BlockShape blockShape;
    };

    __aicore__ inline BlockSchedulerSwizzleInMnCore(Params const &params)
    {
        mSize_ = Get<0>(params.problemShape);
        nSize_ = Get<1>(params.problemShape);
        mL1Size_ = Get<0>(params.tileShape);
        nL1Size_ = Get<1>(params.tileShape);
        auto blockDimM = Get<0>(params.blockShape);
        auto blockDimN = Get<1>(params.blockShape);
        blockDim_ = blockDimM * blockDimN;
        uint64_t curBlockIdx;
        if ASCEND_IS_AIV {
            curBlockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        } else {
            curBlockIdx = AscendC::GetBlockIdx();
        }
        auto mDimIdx = curBlockIdx % blockDimM;
        auto nDimIdx = curBlockIdx / blockDimM;
        uint64_t singleM = Act::Gemm::CeilAlign(Act::Gemm::CeilDiv(mSize_, blockDimM), AscendC::BLOCK_CUBE);
        uint64_t singleN = Act::Gemm::CeilAlign(Act::Gemm::CeilDiv(nSize_, blockDimN), AscendC::BLOCK_CUBE);
        auto mSingleCoreSize = mDimIdx != blockDimM - 1 ? singleM : mSize_ - (blockDimM - 1) * singleM;
        auto nSingleCoreSize = nDimIdx != blockDimN - 1 ? singleN : nSize_ - (blockDimN - 1) * singleN;
        mIter_ = Act::Gemm::CeilDiv(mSingleCoreSize, mL1Size_);
        nIter_ = Act::Gemm::CeilDiv(nSingleCoreSize, nL1Size_);
        mBlockOffset_ = mDimIdx * singleM;
        nBlockOffset_ = nDimIdx * singleN;
        coreLoops_ = mIter_ * nIter_ * blockDimM * blockDimN;
        iterateOrder_ = params.iterateOrder;
    }

    __aicore__ inline uint64_t GetCoreLoops() const { return coreLoops_; }

    template <class Coord>
    __aicore__ inline auto GetActualBlockShape(const Coord &blockCoord)
    {
        auto mCoord = Get<0>(blockCoord);
        auto nCoord = Get<1>(blockCoord);
        return AscendC::MakeShape(Min(mSize_ - mCoord, mL1Size_), Min(nSize_ - nCoord, nL1Size_));
    }

    __aicore__ inline auto GetBlockCoord(uint64_t taskIdx)
    {
        auto localTaskIdx = taskIdx / blockDim_;
        uint64_t mTaskIdx;
        uint64_t nTaskIdx;
        if (iterateOrder_ == ORDER_N) {
            mTaskIdx = localTaskIdx % mIter_;
            nTaskIdx = localTaskIdx / mIter_;
        } else {
            mTaskIdx = localTaskIdx / nIter_;
            nTaskIdx = localTaskIdx % nIter_;
        }
        return AscendC::MakeShape(mBlockOffset_ + mTaskIdx * mL1Size_, nBlockOffset_ + nTaskIdx * nL1Size_);
    }

private:
    uint64_t iterateOrder_;
    uint64_t mBlockOffset_;
    uint64_t nBlockOffset_;
    uint64_t mL1Size_;
    uint64_t nL1Size_;
    uint64_t mSize_;
    uint64_t nSize_;
    uint64_t mIter_;
    uint64_t nIter_;
    uint64_t coreLoops_;
    uint64_t blockDim_;
    static constexpr uint64_t ORDER_N = 1;
};
}  // namespace Block
}  // namespace Gemm
}  // namespace Act
#endif