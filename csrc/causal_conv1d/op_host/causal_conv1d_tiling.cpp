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
 * \file causal_conv1d_tiling.cpp
 */

#include "../tiling_base/tiling_templates_registry.h"
#include "causal_conv1d_tiling_utils.h"
#include "causal_conv1d_tiling_planner.h"
#include "causal_conv1d_tiling_validation.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;
using namespace causal_conv1d_host;

static ge::graphStatus CausalConv1dTilingFunc(gert::TilingContext *context)
{
    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    CausalConv1dTilingData *tiling = context->GetTilingData<CausalConv1dTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(CausalConv1dTilingData), 0, sizeof(CausalConv1dTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    CausalConv1dAttrInfo attrInfo;
    OP_CHECK_IF(GetAttrsInfo(context, attrInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetAttrsInfo error"),
                return ge::GRAPH_FAILED);

    tiling->convStateStride0 = attrInfo.convStateStride0;
    tiling->convStateStride1 = attrInfo.convStateStride1;

    bool hasBias = false;
    OP_CHECK_IF(GetShapeDtypeInfo(context, attrInfo, *tiling, hasBias) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeDtypeInfo error"), return ge::GRAPH_FAILED);

    const int64_t &dim = tiling->dim;
    const int64_t &batch = tiling->batch;
    OP_CHECK_IF(dim <= 0 || batch <= 0, OP_LOGE(context, "dim/batch must be positive"), return ge::GRAPH_FAILED);

    const uint32_t runModeKey = static_cast<uint32_t>(attrInfo.runMode);
    const bool &isFn = (runModeKey == CAUSAL_CONV1D_TPL_RUN_MODE_FN);
    const bool &hasActivation = (attrInfo.activationMode != 0);
    const char *plannerModeTag = "update";
    DimTileChoice baseDimChoice;
    FnExecutionPlan fnExecutionPlan = FN_EXECUTION_PLAN_INVALID;
    FnHostPlan fnHostPlan;
    const int64_t *qslData = nullptr;
    if (isFn && tiling->inputMode == 0) {
        const gert::Tensor *qslTensor = context->GetOptionalInputTensor(QUERY_START_LOC_INDEX);
        qslData = (qslTensor != nullptr) ? qslTensor->GetData<int64_t>() : nullptr;
    }

    if (isFn) {
        fnHostPlan = ChooseFnHostPlan(context, *tiling, ubSize, coreNum);
        plannerModeTag = GetFnTilingCaseName(fnHostPlan.caseKind);
        baseDimChoice = fnHostPlan.baseDimChoice;
        fnExecutionPlan = fnHostPlan.executionPlan;
    } else {
        baseDimChoice = ChooseCanonicalUpdateBaseDimChoice(context, tiling->batch, tiling->dim, coreNum);
    }

    OP_CHECK_IF(baseDimChoice.baseDim <= 0 || baseDimChoice.baseDimCnt <= 0 || baseDimChoice.gridSize <= 0,
                OP_LOGE(context, "invalid dim tile size selection"), return ge::GRAPH_FAILED);

    int64_t effectiveGridSize = baseDimChoice.gridSize;

    if (isFn) {
        OP_CHECK_IF(fnHostPlan.caseKind == FN_TILING_CASE_INVALID || fnExecutionPlan == FN_EXECUTION_PLAN_INVALID ||
                        !fnHostPlan.tokenBlockChoice.enabled || fnHostPlan.tokenBlockChoice.tokenBlockSize <= 0 ||
                        fnHostPlan.tokenBlockChoice.tokenBlockCnt <= 0 || fnHostPlan.tokenBlockChoice.gridSize <= 0 ||
                        fnHostPlan.tokenCoreMapping.tokenCoreBudget <= 0 || fnHostPlan.tokenCoreMapping.blockDim <= 0,
                    OP_LOGE(context, "runMode=0 must resolve a valid unified token tiling plan"),
                    return ge::GRAPH_FAILED);

        tiling->tokenBlockSize = fnHostPlan.tokenBlockChoice.tokenBlockSize;
        tiling->tokenBlockCnt = fnHostPlan.tokenBlockChoice.tokenBlockCnt;
        effectiveGridSize = fnHostPlan.tokenBlockChoice.gridSize;
        if (tiling->inputMode == 0) {
            fnHostPlan.tokenSeqRangePlan =
                BuildFnTokenSeqRangePlan(qslData, tiling->batch, tiling->tokenBlockSize, tiling->tokenBlockCnt);
            if (fnHostPlan.tokenSeqRangePlan.enabled) {
                tiling->hasExplicitTokenSeqRanges = 1;
                tiling->explicitTokenSeqRangeCount = fnHostPlan.tokenSeqRangePlan.rangeCount;
                for (int64_t i = 0; i < fnHostPlan.tokenSeqRangePlan.rangeCount; ++i) {
                    tiling->tokenTileStartSeq[i] = fnHostPlan.tokenSeqRangePlan.tokenTileStartSeq[i];
                    tiling->tokenTileEndSeq[i] = fnHostPlan.tokenSeqRangePlan.tokenTileEndSeq[i];
                }
            } else if (qslData != nullptr && tiling->tokenBlockCnt > MAX_FN_TOKEN_SEQ_RANGE_COUNT) {
                OP_LOGD(context,
                        "FnTokenSeqRanges disabled: tokenBlockCnt[%ld] exceeds fixed tiling capacity[%ld].",
                        tiling->tokenBlockCnt, MAX_FN_TOKEN_SEQ_RANGE_COUNT);
            }
        }
        OP_LOGD(context,
                "FnHostPlan(case=%s): inputMode[%ld], dim[%ld], cuSeqlen[%ld], baseDim[%ld], baseDimCnt[%ld], "
                "tokenCoreBudget[%ld], tokenBlockSize[%ld], tokenBlockCnt[%ld], tokenBlocksPerCore[%ld], "
                "tokenCoreTailCnt[%ld], explicitSeqRanges[%ld], baseGrid[%ld], phase1Grid[%ld], mappedBlockDim[%ld].",
                plannerModeTag, tiling->inputMode, tiling->dim, tiling->cuSeqlen, baseDimChoice.baseDim,
                baseDimChoice.baseDimCnt, fnHostPlan.tokenCoreMapping.tokenCoreBudget,
                fnHostPlan.tokenBlockChoice.tokenBlockSize, fnHostPlan.tokenBlockChoice.tokenBlockCnt,
                fnHostPlan.tokenCoreMapping.tokenBlocksPerCore, fnHostPlan.tokenCoreMapping.tokenCoreTailCnt,
                tiling->hasExplicitTokenSeqRanges,
                baseDimChoice.gridSize, fnHostPlan.tokenBlockChoice.gridSize, fnHostPlan.tokenCoreMapping.blockDim);
    }

    uint32_t blockDim =
        (effectiveGridSize < static_cast<int64_t>(coreNum)) ? static_cast<uint32_t>(effectiveGridSize) : coreNum;
    if (isFn) {
        const int64_t mappedBlockDim = (effectiveGridSize < fnHostPlan.tokenCoreMapping.blockDim) ? effectiveGridSize : fnHostPlan.tokenCoreMapping.blockDim;
        OP_CHECK_IF(mappedBlockDim <= 0, OP_LOGE(context, "invalid mapped blockDim for runMode=0"),
                    return ge::GRAPH_FAILED);
        blockDim = static_cast<uint32_t>(mappedBlockDim);
    }

    OP_LOGD(context,
            "Tiling result: mode[%s], batch[%ld], dim[%ld], baseDim[%ld], baseDimCnt[%ld], gridSize[%ld], "
            "effectiveGrid[%ld], blockDim[%u], coreNum[%u], tokenTiling[%ld,%ld], hasActivation[%d], hasBias[%d], "
            "fnPlan[%ld].",
            plannerModeTag, batch, dim, baseDimChoice.baseDim, baseDimChoice.baseDimCnt, baseDimChoice.gridSize,
            effectiveGridSize, blockDim, coreNum, tiling->tokenBlockSize, tiling->tokenBlockCnt,
            static_cast<int32_t>(hasActivation), static_cast<int32_t>(hasBias), static_cast<int64_t>(fnExecutionPlan));

    context->SetBlockDim(blockDim);
    tiling->baseDim = baseDimChoice.baseDim;
    tiling->baseDimCnt = baseDimChoice.baseDimCnt;
    const uint32_t fnPlanKey = NormalizeFnPlanTilingKey(runModeKey, fnExecutionPlan);
    const uint32_t widthKey = NormalizeWidthTilingKey(runModeKey, static_cast<int32_t>(tiling->width));
    if (isFn && tiling->hasInitialStateMode != 0) {
        constexpr int64_t kDtypeSize = 2;
        constexpr int64_t kSyncBytesPerBlock = 32;
        const int64_t historyCount = (tiling->width - 1 > 0) ? tiling->width - 1 : 0;
        const int64_t syncWorkspaceSize = static_cast<int64_t>(blockDim) * kSyncBytesPerBlock;
        const int64_t snapshotWorkspaceSize = tiling->batch * historyCount * tiling->dim * kDtypeSize;
        const int64_t workspaceSize =
            ASCENDC_RESERVED_WORKSPACE_SIZE + syncWorkspaceSize + snapshotWorkspaceSize;
        OP_CHECK_IF(SetWorkspaceSize(context, static_cast<size_t>(workspaceSize)) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "SetWorkspaceSize error"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "SetScheduleMode(1) error"), return ge::GRAPH_FAILED);
        tiling->hasInitStateWorkspace = 1;
    } else {
        OP_CHECK_IF(SetWorkspaceSize(context, 0) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetWorkspaceSize error"),
                    return ge::GRAPH_FAILED);
        tiling->hasInitStateWorkspace = 0;
    }

    const uint64_t tilingKey = GET_TPL_TILING_KEY(runModeKey, widthKey, fnPlanKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCausalConv1d(gert::TilingParseContext *context)
{
    OP_LOGD(context, "Enter TilingParseForCausalConv1d.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CausalConv1d)
    .Tiling(CausalConv1dTilingFunc)
    .TilingParse<CausalConv1dCompileInfo>(TilingParseForCausalConv1d);

}
