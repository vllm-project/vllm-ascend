/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_mat_mul_v3_iterbatch_basicapi_block_scheduler.h
 * \brief
 */

#ifndef BATCH_MAT_MUL_V3_ITERBATCH_BASICAPI_BLOCK_SCHEDULER_H
#define BATCH_MAT_MUL_V3_ITERBATCH_BASICAPI_BLOCK_SCHEDULER_H

#include "include/matmul/block/block_scheduler_utils.h"
#include "include/matmul/block/block_scheduler_policy.h"
#include "include/utils/status_utils.h"
#include "../../../mat_mul_v3/op_kernel/arch35/mat_mul_tiling_data.h"

namespace Act {
namespace Gemm {
namespace Block {

template <class ProblemShape_, class L1TileShape_, class L0TileShape_>
class Mc2BlockSchedulerIterBatchBuiltIn {
public:
    int64_t m_{0};
    int64_t n_{0};
    int64_t b_{0};
    int64_t k_{0};
    int64_t iterBatchL1_{1};
    int64_t iterBatchL0_{1};
    int64_t baseM_{16};
    int64_t baseN_{16};
    int64_t isHf32_{0};

    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    struct Params {
        const BatchMatMulV3IterBatchBasicTilingData *tilingData;
    };

public:
    __aicore__ inline Mc2BlockSchedulerIterBatchBuiltIn(const ProblemShape &shape, int64_t blockIdx, int64_t blockNum,
                                                        const Params &params)
    {
        m_ = shape.m;
        n_ = shape.n;
        k_ = shape.k;
        b_ = shape.b;
        iterBatchL1_ = params.tilingData->iterBatchL1;
        iterBatchL0_ = params.tilingData->iterBatchL0;
        baseM_ = params.tilingData->baseM;
        baseN_ = params.tilingData->baseN;
        isHf32_ = params.tilingData->isHf32;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return MMV3DivCeil(b_, iterBatchL1_);
    }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetIterBatchTuple()
    {
        return {iterBatchL1_, iterBatchL0_, baseM_, baseN_};
    }

    __aicore__ inline int64_t GetHf32Flag()
    {
        return isHf32_;
    }

    __aicore__ inline int64_t GetBlockNum(ProblemShape shape, int64_t blockNum)
    {
        int64_t tilingBlockNum = 0;
        if (b_ < blockNum) {
            tilingBlockNum = b_;
        } else {
            tilingBlockNum = blockNum;
        }
        return tilingBlockNum;
    }

    __aicore__ inline BlockShape GetBlockShape(int64_t tileIdx)
    {
        return {m_, n_, k_, b_};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int tileIdx)
    {
        return {0, 0, 0, tileIdx * iterBatchL1_};
    }
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, Act::Gemm::BuiltInIterBatchScheduler, TransA_,
                              TransB_> {
    using SchedulerOp = Mc2BlockSchedulerIterBatchBuiltIn<ProblemShape_, L1TileShape_, L0TileShape_>;
};

} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
