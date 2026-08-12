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
 * \file mat_mul_dasw_block.h
 * \brief
 */
#ifndef MMV3_MATMUL_DASW_BLOCK_H
#define MMV3_MATMUL_DASW_BLOCK_H

#include "mat_mul_asw_block.h"

namespace Mc2MatmulV3Advanced {

using namespace AscendC;
using namespace matmul;

class Mc2MatmulDaswBlock : public Mc2MatmulAswBlock {
public:
    __aicore__ inline Mc2MatmulDaswBlock()
    {
    }
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void *tilingData);
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx, uint64_t newBlockIdx);
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void Mc2MatmulDaswBlock::Init(const void *tilingData)
{
    Mc2MatmulAswBlock::Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
    if (params_.totalCnt % (matmulTilingData_->tCubeTiling.usedCoreNum * 2) == 0) {
        params_.roundToReverse = params_.round;
    } else if (params_.round % 2 == 0) {
        params_.roundToReverse = params_.round - 2; // even round, last two rounds no need to reverse
    } else {
        params_.roundToReverse = params_.round - 1; // odd round, last round no need to reverse
    }
}

__aicore__ inline void Mc2MatmulDaswBlock::UpdateBasicIndex(uint64_t roundIdx, uint64_t newBlockIdx)
{
    uint64_t reversedRoundIdx = roundIdx;
    uint64_t reversedNewBlockIdx = newBlockIdx;
    if ((roundIdx < (params_.roundToReverse >> 1)) &&
        (newBlockIdx >= (matmulTilingData_->tCubeTiling.usedCoreNum >> 1))) {
        reversedRoundIdx = roundIdx + (params_.roundToReverse >> 1);
        reversedNewBlockIdx = newBlockIdx - (matmulTilingData_->tCubeTiling.usedCoreNum >> 1);
    } else if ((roundIdx >= (params_.roundToReverse >> 1)) && (roundIdx < params_.roundToReverse) &&
               (newBlockIdx < (matmulTilingData_->tCubeTiling.usedCoreNum >> 1))) {
        reversedRoundIdx = roundIdx - (params_.roundToReverse >> 1);
        reversedNewBlockIdx = newBlockIdx + (matmulTilingData_->tCubeTiling.usedCoreNum >> 1);
    }
    Mc2MatmulAswBlock::UpdateBasicIndex(reversedRoundIdx, reversedNewBlockIdx);
}

} // namespace Mc2MatmulV3Advanced

#endif // MMV3_MATMUL_DASW_BLOCK_H
