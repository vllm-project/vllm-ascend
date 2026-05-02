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

#ifndef CAUSAL_CONV1D_TILING_UTILS_H
#define CAUSAL_CONV1D_TILING_UTILS_H

#include "../tiling_base/tiling_util.h"
#include "../op_kernel/causal_conv1d_tiling_key.h"

namespace optiling::causal_conv1d_host {

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t CONV_STATES_INDEX = 3;
constexpr uint32_t QUERY_START_LOC_INDEX = 4;
constexpr uint32_t CACHE_INDICES_INDEX = 5;
constexpr uint32_t INITIAL_STATE_MODE_INDEX = 6;
constexpr uint32_t NUM_ACCEPTED_TOKENS_INDEX = 7;

constexpr int32_t ATTR_ACTIVATION_MODE_INDEX = 0;
constexpr int32_t ATTR_PAD_SLOT_ID_INDEX = 1;
constexpr int32_t ATTR_RUN_MODE_INDEX = 2;
constexpr int32_t ATTR_CONV_STATE_STRIDE_0 = 3;
constexpr int32_t ATTR_CONV_STATE_STRIDE_1 = 4;
constexpr int64_t ASCENDC_RESERVED_WORKSPACE_SIZE = 16 * 1024 * 1024;

struct CausalConv1dCompileInfo {
    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
};

struct CausalConv1dAttrInfo {
    int64_t activationMode = 0;
    int64_t padSlotId = -1;
    int64_t runMode = 0;
    int64_t convStateStride0 = 0;
    int64_t convStateStride1 = 0;
};

struct DimTileChoice {
    int64_t baseDim = 0;
    int64_t baseDimCnt = 0;
    int64_t gridSize = 0;
};

struct VarlenTokenTileChoice {
    bool enabled = false;
    int64_t tokenBlockSize = 0;
    int64_t tokenBlockCnt = 0;
    int64_t gridSize = 0;
};

enum FnTilingCaseKind : int64_t {
    FN_TILING_CASE_INVALID = 0,
    FN_TILING_CASE_TOKEN_FIRST = 1,
    FN_TILING_CASE_TOKEN_DIM_CO_SPLIT = 2,
};

struct TokenCoreMappingChoice {
    int64_t tokenCoreBudget = 0;
    int64_t tokenBlocksPerCore = 0;
    int64_t tokenCoreTailCnt = 0;
    int64_t blockDim = 0;
};

constexpr int64_t MAX_FN_TOKEN_SEQ_RANGE_COUNT = 128;

struct FnTokenSeqRangePlan {
    bool enabled = false;
    int64_t rangeCount = 0;
    int64_t tokenTileStartSeq[MAX_FN_TOKEN_SEQ_RANGE_COUNT] = {};
    int64_t tokenTileEndSeq[MAX_FN_TOKEN_SEQ_RANGE_COUNT] = {};
};

struct FnHostPlan {
    FnTilingCaseKind caseKind = FN_TILING_CASE_INVALID;
    FnExecutionPlan executionPlan = FN_EXECUTION_PLAN_INVALID;
    DimTileChoice baseDimChoice;
    VarlenTokenTileChoice tokenBlockChoice;
    TokenCoreMappingChoice tokenCoreMapping;
    FnTokenSeqRangePlan tokenSeqRangePlan;
};

constexpr int64_t DIM_ALIGN_BYTES = 32;
constexpr int64_t BF16_FP16_ELEM_BYTES = 2;
constexpr int64_t DIM_ALIGN_ELEMS = DIM_ALIGN_BYTES / BF16_FP16_ELEM_BYTES;
constexpr int64_t MAX_DIM_TILE_SIZE = 4096;
constexpr int64_t FN_UB_RESERVED_BYTES = 512;
constexpr int64_t RING_SLOT_CNT = 5;
constexpr int64_t FN_OUT_SLOT_CNT = 2;
constexpr int64_t FN_CALC_FP32_SLOT_CNT = 8;

inline uint32_t NormalizeFnPlanTilingKey(uint32_t runModeKey, FnExecutionPlan fnExecutionPlan)
{
    if (runModeKey != CAUSAL_CONV1D_TPL_RUN_MODE_FN) {
        return CAUSAL_CONV1D_TPL_FN_PLAN_INVALID;
    }
    switch (fnExecutionPlan) {
        case FN_EXECUTION_PLAN_CUTBS:
            return CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS;
        case FN_EXECUTION_PLAN_CUTBSD:
            return CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD;
        default:
            return CAUSAL_CONV1D_TPL_FN_PLAN_INVALID;
    }
}

inline uint32_t NormalizeWidthTilingKey(uint32_t runModeKey, int32_t width)
{
    if (runModeKey != CAUSAL_CONV1D_TPL_RUN_MODE_FN) {
        return CAUSAL_CONV1D_TPL_WIDTH_RUNTIME;
    }
    switch (width) {
        case 2:
            return CAUSAL_CONV1D_TPL_WIDTH_2;
        case 3:
            return CAUSAL_CONV1D_TPL_WIDTH_3;
        case 4:
            return CAUSAL_CONV1D_TPL_WIDTH_4;
        default:
            return CAUSAL_CONV1D_TPL_WIDTH_RUNTIME;
    }
}

inline int64_t CeilDivInt64(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

inline int64_t AlignDownInt64(int64_t value, int64_t align)
{
    if (align <= 0 || value <= 0) {
        return 0;
    }
    return (value / align) * align;
}

inline int64_t AlignUpInt64(int64_t value, int64_t align)
{
    if (align <= 0 || value <= 0) {
        return 0;
    }
    return CeilDivInt64(value, align) * align;
}

inline const char *GetFnTilingCaseName(FnTilingCaseKind caseKind)
{
    switch (caseKind) {
        case FN_TILING_CASE_TOKEN_FIRST:
            return "token_first";
        case FN_TILING_CASE_TOKEN_DIM_CO_SPLIT:
            return "token_dim_co_split";
        default:
            return "invalid";
    }
}

} // namespace optiling::causal_conv1d_host

#endif // CAUSAL_CONV1D_TILING_UTILS_H
