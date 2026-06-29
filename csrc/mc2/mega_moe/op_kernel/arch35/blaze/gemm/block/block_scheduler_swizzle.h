/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_scheduler_swizzle.h
 * \brief
 */

#pragma once

#include "tensor_api/tensor.h"
#include "basic_api/kernel_basic_intf.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
class BlockSchedulerSwizzle {
public:
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t>;
    using TileShape = AscendC::Te::Shape<int64_t, int64_t>;

    struct Params {
        TileShape tileShape;
    };

    __aicore__ inline BlockSchedulerSwizzle(const ProblemShape& shape, const Params& params)
        : problemShape_(shape), tileShape_(params.tileShape)
    {
        if constexpr (SwizzleDirection == 0) {
            // m first
            loopFirst_ = AscendC::Std::ceil_division(Get<IDX_M_IDX>(problemShape_), Get<IDX_M_IDX>(tileShape_));
            loopSecond_ = AscendC::Std::ceil_division(Get<IDX_N_IDX>(problemShape_), Get<IDX_N_IDX>(tileShape_));
        } else if constexpr (SwizzleDirection == 1) {
            // n first
            loopSecond_ = AscendC::Std::ceil_division(Get<IDX_M_IDX>(problemShape_), Get<IDX_M_IDX>(tileShape_));
            loopFirst_ = AscendC::Std::ceil_division(Get<IDX_N_IDX>(problemShape_), Get<IDX_N_IDX>(tileShape_));
        }
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return loopFirst_ * loopSecond_;
    }

    __aicore__ inline BlockShape GetBlockShape(const BlockCoord& blockCoord)
    {
        return {
            min(Get<IDX_M_IDX>(tileShape_), Get<IDX_M_IDX>(problemShape_) - Get<IDX_M_IDX>(blockCoord)),
            min(Get<IDX_N_IDX>(tileShape_), Get<IDX_N_IDX>(problemShape_) - Get<IDX_N_IDX>(blockCoord)),
            Get<IDX_K_IDX>(problemShape_)};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int tileIdx)
    {
        int64_t blockSpan = SwizzleOffset * loopSecond_;
        int64_t blockIdx = tileIdx / blockSpan;
        int64_t inBlockIdx = tileIdx % blockSpan;

        int64_t firstValid = Min(loopFirst_ - blockIdx * SwizzleOffset, static_cast<int64_t>(SwizzleOffset));
        // Defensive: firstValid is always >= 1 by construction, this silences static analyzer divide-by-zero warnings.
        if (firstValid == 0) {
            firstValid = 1;
        }
        int64_t firstIdx = blockIdx * SwizzleOffset + inBlockIdx % firstValid;
        int64_t secondIdx = inBlockIdx / firstValid;
        if (blockIdx & 1) {
            secondIdx = loopSecond_ - secondIdx - 1;
        }

        if constexpr (SwizzleDirection == 0) {
            return {firstIdx * Get<IDX_M_IDX>(tileShape_), secondIdx * Get<IDX_N_IDX>(tileShape_), 0};
        } else {
            return {secondIdx * Get<IDX_M_IDX>(tileShape_), firstIdx * Get<IDX_N_IDX>(tileShape_), 0};
        }
    }

private:
    ProblemShape problemShape_;
    TileShape tileShape_;
    int64_t loopFirst_;
    int64_t loopSecond_;
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
