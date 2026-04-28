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
 * \file block_scheduler_utils.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_SCHEDULER_UTILS_H
#define MATMUL_BLOCK_BLOCK_SCHEDULER_UTILS_H
#include "../utils/common_utils.h"
#include "../utils/status_utils.h"
#include "../utils/host_utils.h"

namespace Act {
namespace Gemm {
namespace Block {

constexpr uint64_t WINDOW_LEN = 4UL;

// Base template definition for BlockSchedulerSelector
template <class ProblemShape, class L1TileShape, class L0TileShape, class BlockScheduler = void,
          bool TransA = false, bool TransB = false>
struct BlockSchedulerSelector;

__aicore__ inline int64_t GetPerBlockNum(int64_t coreNum, int64_t mTileNum, int64_t nTileNum, int64_t b = 1)
{
    int64_t perCoreBlockNum = Act::Gemm::CeilDiv(mTileNum * nTileNum * b, coreNum);
    return perCoreBlockNum;
}

__aicore__ inline bool IsMTail(int64_t mTileIdx, int64_t mTileNum)
{
    if ((mTileIdx - (mTileNum - 1)) % mTileNum == 0) {
        return true;
    }
    return false;
}

__aicore__ inline bool IsNTail(int64_t nTileIdx, int64_t nTileNum)
{
    if (nTileIdx == (nTileNum - 1)) {
        return true;
    }
    return false;
}

__aicore__ inline bool IsKTail(int64_t kTileIdx, int64_t kTileNum)
{
    if (kTileIdx == (kTileNum - 1)) {
        return true;
    }
    return false;
}

__aicore__ inline int64_t GetTailNum(int64_t totalNum, int64_t normalNum)
{
    if (normalNum == 0) {
        return 0;
    }
    if (totalNum % normalNum == 0) {
        return normalNum;
    }
    return totalNum % normalNum;
}

__aicore__ inline int64_t MMLcm(int64_t m, int64_t n)
{
    if (m == 0 || n == 0) {
        return 0;
    }
    int64_t total = m * n;
    int64_t tmp = 0;
    // calc maximum common divisor m
    while (n != 0) {
        tmp = m % n;
        m = n;
        n = tmp;
    }
    return total / m;
}

// device -> kernel -> scheduler
static int64_t DoGetBlockNum(int64_t l1M, int64_t l1N, const MatmulShape &shape)
{
    int maxCoreNum = GetCoreNum();
    int64_t mTotalCnt = Act::Gemm::CeilDiv(shape.m, l1M);
    int64_t nTotalCnt = Act::Gemm::CeilDiv(shape.n, l1N);
    int64_t batch = shape.b ? shape.b : 1;
    int64_t blockNum = 0;
    int64_t totalCnt = mTotalCnt * nTotalCnt * batch;
    if (totalCnt < maxCoreNum) {
        blockNum = totalCnt;
    } else {
        int64_t perCoreBlockNum = Act::Gemm::CeilDiv(totalCnt, maxCoreNum);
        blockNum = Act::Gemm::CeilDiv(totalCnt, perCoreBlockNum);
    }
    return blockNum;
}

template <class ProblemShape_>
__host_aicore__ static Status DoCheckArgs(const ProblemShape_ &shape, int64_t l1M, int64_t l1N, int64_t l1K,
                                          int64_t l0M, int64_t l0N, int64_t l0K)
{
    if (l1M > 128 || l0M > 128 || l1N > 256 || l0N > 256) { // 128,256: input limit
        return Status::l1L0ErrorExceedsLimit;
    }

    if (l1M % MATMUL_MNK_ALIGN != 0 || l1N % MATMUL_MNK_ALIGN != 0 || l1K % MATMUL_MNK_ALIGN != 0 ||
        l0M % MATMUL_MNK_ALIGN != 0 || l0N % MATMUL_MNK_ALIGN != 0 || l0K % MATMUL_MNK_ALIGN != 0) {
        return Status::l1L0ErrorNotAlign;
    }

    if (l1M != l0M || l1N != l0N) {
        return Status::l1MnL0MnErrorNotSame;
    }

    if (l1K < l0K) {
        return Status::l1kErrorSmallerL0k;
    }

    if (l1K % l0K != 0) {
        return Status::l1kErrorL0kNotAlign;
    }
    return Status::success;
}

} // namespace Block
} // namespace Gemm
} // namespace Act
#endif