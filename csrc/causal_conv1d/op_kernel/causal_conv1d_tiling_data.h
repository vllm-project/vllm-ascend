/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_tiling_data.h
 */

#ifndef CAUSAL_CONV1D_TILING_DATA_H_
#define CAUSAL_CONV1D_TILING_DATA_H_

#include <cstdint>

enum FnExecutionPlan : int64_t {
    FN_EXECUTION_PLAN_INVALID = 0,
    FN_EXECUTION_PLAN_CUTBS = 1,
    FN_EXECUTION_PLAN_CUTBSD = 2,
};

inline constexpr int64_t ResolveFnExecutionPlan(int64_t baseDimCnt)
{
    return (baseDimCnt <= 0) ? FN_EXECUTION_PLAN_INVALID
        : (baseDimCnt <= 1) ? FN_EXECUTION_PLAN_CUTBS
        :                     FN_EXECUTION_PLAN_CUTBSD;
}


struct CausalConv1dTilingData {
    int64_t dim;
    int64_t cuSeqlen;
    int64_t seqLen;
    int64_t inputMode;

    int64_t width;

    int64_t stateLen;
    int64_t numCacheLines;
    int64_t batch;
    int64_t activationMode;
    int64_t padSlotId;
    int64_t hasBias;
    int64_t baseDim;
    int64_t baseDimCnt;
    int64_t hasNumAcceptedTokens;
    int64_t hasCacheIndices;
    int64_t hasInitialStateMode;
    int64_t tokenBlockSize;
    int64_t tokenBlockCnt;
    int64_t convStateStride0;
    int64_t convStateStride1;
    int64_t hasExplicitTokenSeqRanges;
    int64_t explicitTokenSeqRangeCount;
    int64_t tokenTileStartSeq[128];
    int64_t tokenTileEndSeq[128];
    int64_t hasInitStateWorkspace;
};
#endif // CAUSAL_CONV1D_TILING_DATA_H_
