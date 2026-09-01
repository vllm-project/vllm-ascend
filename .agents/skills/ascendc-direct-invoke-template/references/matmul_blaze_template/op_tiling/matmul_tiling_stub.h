/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_TILING_STUB_H
#define MATMUL_TILING_STUB_H

#include <algorithm>
#include <cstdint>

#include "matmul_tiling_data.h"

class MatmulTilingStub {
public:
    void GetTilingData(uint64_t m, uint64_t n, uint64_t k,
                       bool transA, bool transB,
                       QuantMatmulTilingData& data)
    {
        (void)transA;
        (void)transB;

        data = {};
        data.m = static_cast<uint32_t>(m);
        data.n = static_cast<uint32_t>(n);
        data.k = static_cast<uint32_t>(k);

        constexpr uint32_t MAX_BASE_M = 128;
        constexpr uint32_t MAX_BASE_N = 128;
        constexpr uint32_t MAX_BASE_K = 128;
        constexpr uint32_t ALIGN_M = 16;
        constexpr uint32_t ALIGN_K = 64;
        constexpr uint32_t MAX_CORES = 48;

        uint32_t baseM = static_cast<uint32_t>(std::min(m, static_cast<uint64_t>(MAX_BASE_M)));
        baseM = ((baseM + ALIGN_M - 1) / ALIGN_M) * ALIGN_M;
        if (baseM == 0) baseM = ALIGN_M;

        uint32_t baseN = static_cast<uint32_t>(std::min(n, static_cast<uint64_t>(MAX_BASE_N)));
        baseN = ((baseN + ALIGN_M - 1) / ALIGN_M) * ALIGN_M;
        if (baseN == 0) baseN = ALIGN_M;

        uint32_t baseK = static_cast<uint32_t>(std::min(k, static_cast<uint64_t>(MAX_BASE_K)));
        baseK = ((baseK + ALIGN_K - 1) / ALIGN_K) * ALIGN_K;
        if (baseK == 0) baseK = ALIGN_K;

        data.baseM = baseM;
        data.baseN = baseN;
        data.baseK = baseK;

        uint32_t mTiles = static_cast<uint32_t>((m + baseM - 1) / baseM);
        uint32_t nTiles = static_cast<uint32_t>((n + baseN - 1) / baseN);

        data.mTailTile = 1;
        data.nTailTile = 1;
        data.mBaseTailSplitCnt = 1;
        data.nBaseTailSplitCnt = 1;
        data.mTailMain = 0;
        data.nTailMain = 0;

        uint32_t stepK = static_cast<uint32_t>((k + baseK - 1) / baseK);
        data.scaleKL1 = stepK * baseK;
        data.stepK = static_cast<uint8_t>(std::min(stepK, 4U));
        data.nBufferNum = 2;
        data.dbL0c = 1;

        uint32_t totalTiles = mTiles * nTiles;
        data.usedCoreNum = std::min(totalTiles, MAX_CORES);
    }
};

#endif
