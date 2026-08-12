/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_white_list.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_WHITE_LIST_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_WHITE_LIST_H

#include "common/op_host/math_util.h"
#include "ops_legacy/op_tiling/op_cache_def_tiling.h"

namespace optiling {

class Mc2WhiteListShape {
public:
    bool operator<(const Mc2WhiteListShape &right) const
    {
        return memcmp(this, &right, sizeof(Mc2WhiteListShape)) < 0;
    }

    uint64_t mSize_;
    uint64_t kSize_;
    uint64_t nSize_;
    bool hasBias_;
    bool transA_;
    bool transB_;
    uint64_t aicNum_ : 40;
};

inline void Mc2SetMatmulTilingFromCacheData(WeightQuantBatchMatmulCacheTilingData &cacheTilingData,
                                            AscendC::tiling::TCubeTiling &matmulTiling, uint64_t m, uint64_t n,
                                            int32_t isBias)
{
    matmulTiling.M = m;
    matmulTiling.N = n;
    matmulTiling.Ka = cacheTilingData.ka_;
    matmulTiling.Kb = cacheTilingData.kb_;
    matmulTiling.singleCoreM = ops::CeilDiv(m, static_cast<uint64_t>(cacheTilingData.mDim_));
    matmulTiling.singleCoreN = cacheTilingData.singleCoreN_;
    matmulTiling.singleCoreK = cacheTilingData.singleCoreK_;
    matmulTiling.baseM = cacheTilingData.baseM_;
    matmulTiling.baseN = cacheTilingData.baseN_;
    matmulTiling.baseK = cacheTilingData.baseK_;
    matmulTiling.depthA1 = cacheTilingData.depthA1_;
    matmulTiling.depthB1 = cacheTilingData.depthB1_;
    matmulTiling.stepM = cacheTilingData.stepM_;
    matmulTiling.stepN = cacheTilingData.stepN_;
    matmulTiling.stepKa = cacheTilingData.stepKa_;
    matmulTiling.stepKb = cacheTilingData.stepKb_;
    matmulTiling.isBias = isBias;
    matmulTiling.transLength = cacheTilingData.transLength_;
    matmulTiling.iterateOrder = cacheTilingData.iterateOrder_;
    matmulTiling.shareL1Size = cacheTilingData.shareL1Size_;
    matmulTiling.shareL0CSize = cacheTilingData.shareL0CSize_;
    matmulTiling.dbL0A = cacheTilingData.dbL0A_;
    matmulTiling.dbL0B = cacheTilingData.dbL0B_;
    matmulTiling.dbL0C = cacheTilingData.dbL0C_;
    matmulTiling.usedCoreNum = 1;
    matmulTiling.batchM = 1;
    matmulTiling.batchN = 1;
    matmulTiling.singleBatchM = 1;
    matmulTiling.singleBatchN = 1;
}

} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_WHITE_LIST_H
