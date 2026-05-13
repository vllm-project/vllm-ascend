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

#ifndef CAUSAL_CONV1D_TILING_PLANNER_H
#define CAUSAL_CONV1D_TILING_PLANNER_H

#include "causal_conv1d_tiling_utils.h"
#include "../op_kernel/causal_conv1d_tiling_data.h"

namespace optiling::causal_conv1d_host {

using namespace Ops::Transformer::OpTiling;

inline DimTileChoice ChooseCanonicalUpdateBaseDimChoice(gert::TilingContext *context, int64_t batch, int64_t dim,
                                                        uint32_t coreNum)
{
    const int64_t candidates[] = {4096, 2048, 1024, 512, 384, 192};

    auto chooseOnce = [&](bool requireExactDiv) -> DimTileChoice {
        DimTileChoice bestOver;
        int64_t bestOverGap = std::numeric_limits<int64_t>::max();
        DimTileChoice bestUnder;

        for (int64_t baseDim : candidates) {
            if (baseDim <= 0) {
                continue;
            }
            if (requireExactDiv && (dim % baseDim != 0)) {
                continue;
            }

            const int64_t baseDimCnt = requireExactDiv ? (dim / baseDim) : CeilDivInt64(dim, baseDim);
            const int64_t gridSize = batch * baseDimCnt;
            if (gridSize <= 0) {
                continue;
            }

            OP_LOGD(context,
                    "DimTile(update) candidate[%s]: baseDim[%ld], baseDimCnt[%ld], gridSize[%ld], coreNum[%u].",
                    requireExactDiv ? "exact" : "tail", baseDim, baseDimCnt, gridSize, coreNum);
            if (gridSize >= static_cast<int64_t>(coreNum)) {
                const int64_t gap = gridSize - static_cast<int64_t>(coreNum);
                if (gap < bestOverGap) {
                    // bestOver = {baseDim, baseDimCnt, gridSize};
                    bestOver.baseDim = baseDim;
                    bestOver.baseDimCnt = baseDimCnt;
                    bestOver.gridSize = gridSize;
                    bestOverGap = gap;
                }
            } else if (gridSize > bestUnder.gridSize ||
                       (gridSize == bestUnder.gridSize && baseDim < bestUnder.baseDim)) {
                // bestUnder = {baseDim, baseDimCnt, gridSize};
                bestUnder.baseDim = baseDim;
                bestUnder.baseDimCnt = baseDimCnt;
                bestUnder.gridSize = gridSize;
            }
        }

        return (bestOver.baseDim != 0) ? bestOver : bestUnder;
    };

    DimTileChoice result = chooseOnce(true);
    if (result.baseDim == 0) {
        result = chooseOnce(false);
    }
    OP_LOGD(context, "DimTile(update) chosen: baseDim[%ld], baseDimCnt[%ld], gridSize[%ld].", result.baseDim,
            result.baseDimCnt, result.gridSize);
    return result;
}

inline int64_t ResolveFnTokenCoreBudget(int64_t baseDimCnt, FnExecutionPlan fnExecutionPlan, uint32_t coreNum)
{
    if (baseDimCnt <= 0 || coreNum == 0 || fnExecutionPlan == FN_EXECUTION_PLAN_INVALID) {
        return 0;
    }

    int64_t tokenCoreBudget = static_cast<int64_t>(coreNum);
    if (fnExecutionPlan == FN_EXECUTION_PLAN_CUTBSD) {
        tokenCoreBudget = std::max<int64_t>(1, tokenCoreBudget / baseDimCnt);
    }
    return tokenCoreBudget;
}

inline VarlenTokenTileChoice ChooseFnTokenBlockChoice(int64_t cuSeqlen, int64_t baseDimCnt,
                                                      FnExecutionPlan fnExecutionPlan, uint32_t coreNum);

inline int64_t ComputeFnUbLimitedBaseDim(uint64_t ubSize)
{
    if (ubSize <= static_cast<uint64_t>(FN_UB_RESERVED_BYTES)) {
        return 0;
    }

    const int64_t bytesPerElem = (RING_SLOT_CNT * BF16_FP16_ELEM_BYTES) + (FN_OUT_SLOT_CNT * BF16_FP16_ELEM_BYTES) +
                                 (FN_CALC_FP32_SLOT_CNT * static_cast<int64_t>(sizeof(float)));
    const int64_t budgetBytes = static_cast<int64_t>(ubSize) - FN_UB_RESERVED_BYTES;
    const int64_t ubLimitedBaseDim = AlignDownInt64(budgetBytes / bytesPerElem, DIM_ALIGN_ELEMS);
    return std::min<int64_t>(MAX_DIM_TILE_SIZE, ubLimitedBaseDim);
}

inline DimTileChoice ChooseFnTokenFirstBaseDimChoice(int64_t dim)
{
    if (dim <= 0 || dim > MAX_DIM_TILE_SIZE) {
        return {};
    }
    DimTileChoice choice;
    choice.baseDim = dim;
    choice.baseDimCnt = 1;
    choice.gridSize = 1;
    return choice;
}

inline DimTileChoice ChooseFnTokenDimCoSplitBaseDimChoice(gert::TilingContext *context, int64_t dim, uint64_t ubSize,
                                                          uint32_t coreNum)
{
    if (dim <= 0) {
        return {};
    }

    const int64_t ubLimitedBaseDim = ComputeFnUbLimitedBaseDim(ubSize);
    if (ubLimitedBaseDim <= 0) {
        OP_LOGD(context, "FnDimCoSplit: UB budget is too small to form a valid baseDim.");
        return {};
    }

    DimTileChoice result;
    result.baseDim = ubLimitedBaseDim;
    result.baseDimCnt = CeilDivInt64(dim, result.baseDim);
    result.gridSize = result.baseDimCnt;

    if (coreNum == 0 || result.baseDimCnt <= 1 || result.baseDimCnt >= static_cast<int64_t>(coreNum) ||
        (coreNum % result.baseDimCnt == 0)) {
        OP_LOGD(context,
                "FnDimCoSplit: dim[%ld], ubLimitedBaseDim[%ld], baseDimCnt[%ld], coreNum[%u], adjusted[%d].", dim,
                result.baseDim, result.baseDimCnt, coreNum, 0);
        return result;
    }

    int64_t adjustedBaseDimCnt = result.baseDimCnt;
    while (adjustedBaseDimCnt < static_cast<int64_t>(coreNum) && (coreNum % adjustedBaseDimCnt != 0)) {
        ++adjustedBaseDimCnt;
    }

    if (adjustedBaseDimCnt >= static_cast<int64_t>(coreNum)) {
        OP_LOGD(context,
                "FnDimCoSplit: keep baseDimCnt[%ld] because no divisible adjustment exists under coreNum[%u].",
                result.baseDimCnt, coreNum);
        return result;
    }

    const int64_t adjustedBaseDim = AlignUpInt64(CeilDivInt64(dim, adjustedBaseDimCnt), DIM_ALIGN_ELEMS);
    if (adjustedBaseDim <= 0 || adjustedBaseDim > ubLimitedBaseDim || adjustedBaseDim > MAX_DIM_TILE_SIZE) {
        OP_LOGD(context,
                "FnDimCoSplit: rejected adjusted baseDim[%ld] with baseDimCnt[%ld], ubLimitedBaseDim[%ld].",
                adjustedBaseDim, adjustedBaseDimCnt, ubLimitedBaseDim);
        return result;
    }

    result.baseDim = adjustedBaseDim;
    result.baseDimCnt = CeilDivInt64(dim, result.baseDim);
    result.gridSize = result.baseDimCnt;
    OP_LOGD(context,
            "FnDimCoSplit: dim[%ld], ubLimitedBaseDim[%ld], adjustedBaseDim[%ld], baseDimCnt[%ld], coreNum[%u].",
            dim, ubLimitedBaseDim, result.baseDim, result.baseDimCnt, coreNum);
    return result;
}

inline TokenCoreMappingChoice BuildFnTokenCoreMappingChoice(int64_t tokenBlockCnt, int64_t baseDimCnt,
                                                            FnExecutionPlan fnExecutionPlan, uint32_t coreNum)
{
    TokenCoreMappingChoice mapping;
    mapping.tokenCoreBudget = ResolveFnTokenCoreBudget(baseDimCnt, fnExecutionPlan, coreNum);
    if (tokenBlockCnt <= 0 || mapping.tokenCoreBudget <= 0 || baseDimCnt <= 0) {
        return mapping;
    }

    mapping.tokenBlocksPerCore = CeilDivInt64(tokenBlockCnt, mapping.tokenCoreBudget);
    mapping.tokenCoreTailCnt =
        tokenBlockCnt - (std::max<int64_t>(0, mapping.tokenBlocksPerCore - 1) * mapping.tokenCoreBudget);
    if (mapping.tokenCoreTailCnt <= 0) {
        mapping.tokenCoreTailCnt = mapping.tokenCoreBudget;
    }
    mapping.blockDim = mapping.tokenCoreBudget * baseDimCnt;
    return mapping;
}

inline FnTokenSeqRangePlan BuildFnTokenSeqRangePlan(const int64_t *qslData, int64_t batch, int64_t tokenBlockSize,
                                                    int64_t tokenBlockCnt)
{
    FnTokenSeqRangePlan plan;
    if (qslData == nullptr || batch <= 0 || tokenBlockSize <= 0 || tokenBlockCnt <= 0 ||
        tokenBlockCnt > MAX_FN_TOKEN_SEQ_RANGE_COUNT) {
        return plan;
    }

    plan.enabled = true;
    plan.rangeCount = tokenBlockCnt;
    int64_t seq = 0;
    for (int64_t tokenTileId = 0; tokenTileId < tokenBlockCnt; ++tokenTileId) {
        const int64_t tokenStart = tokenTileId * tokenBlockSize;
        const int64_t tokenEnd = tokenStart + tokenBlockSize;

        while (seq < batch && qslData[seq + 1] <= tokenStart) {
            ++seq;
        }

        int64_t endSeq = seq;
        while (endSeq < batch && qslData[endSeq] < tokenEnd) {
            ++endSeq;
        }

        plan.tokenTileStartSeq[tokenTileId] = seq;
        plan.tokenTileEndSeq[tokenTileId] = endSeq;
    }
    return plan;
}

inline VarlenTokenTileChoice ChooseUnifiedFnTokenBlockPlan(gert::TilingContext *context,
                                                           const CausalConv1dTilingData &tiling,
                                                           const DimTileChoice &baseDimChoice,
                                                           FnExecutionPlan fnExecutionPlan,
                                                           uint32_t coreNum)
{
    VarlenTokenTileChoice tokenBlockChoice;
    if ((tiling.inputMode != 0 && tiling.inputMode != 1) || tiling.batch <= 0 || tiling.cuSeqlen <= 0 ||
        baseDimChoice.baseDimCnt <= 0 || coreNum == 0 || fnExecutionPlan == FN_EXECUTION_PLAN_INVALID) {
        return tokenBlockChoice;
    }
    if (tiling.hasNumAcceptedTokens != 0) {
        OP_LOGD(context, "Varlen token tiling disabled: speculative decode still uses the existing seq mapping.");
        return tokenBlockChoice;
    }

    tokenBlockChoice = ChooseFnTokenBlockChoice(tiling.cuSeqlen, baseDimChoice.baseDimCnt, fnExecutionPlan, coreNum);

    OP_LOGD(context,
            "FnTokenTile(plan=%ld): cuSeqlen[%ld], baseDimCnt[%ld], tokenBlockSize[%ld], "
            "tokenBlockCnt[%ld], gridSize[%ld].",
            static_cast<int64_t>(fnExecutionPlan), tiling.cuSeqlen, baseDimChoice.baseDimCnt,
            tokenBlockChoice.tokenBlockSize, tokenBlockChoice.tokenBlockCnt, tokenBlockChoice.gridSize);
    return tokenBlockChoice;
}

inline VarlenTokenTileChoice ChooseFnTokenBlockChoice(int64_t cuSeqlen, int64_t baseDimCnt,
                                                      FnExecutionPlan fnExecutionPlan, uint32_t coreNum)
{
    VarlenTokenTileChoice tokenBlockChoice;
    const int64_t tokenCoreBudget = ResolveFnTokenCoreBudget(baseDimCnt, fnExecutionPlan, coreNum);
    if (cuSeqlen <= 0 || tokenCoreBudget <= 0) {
        return tokenBlockChoice;
    }

    tokenBlockChoice.enabled = true;
    const int64_t idealBlockSize = CeilDivInt64(cuSeqlen, tokenCoreBudget);
    tokenBlockChoice.tokenBlockSize = (idealBlockSize > 0) ? idealBlockSize : 1;
    tokenBlockChoice.tokenBlockCnt = CeilDivInt64(cuSeqlen, tokenBlockChoice.tokenBlockSize);
    tokenBlockChoice.gridSize = tokenBlockChoice.tokenBlockCnt * baseDimCnt;
    return tokenBlockChoice;
}

inline FnHostPlan ChooseFnHostPlan(gert::TilingContext *context, const CausalConv1dTilingData &tiling, uint64_t ubSize,
                                   uint32_t coreNum)
{
    FnHostPlan plan;
    if ((tiling.inputMode != 0 && tiling.inputMode != 1) || tiling.batch <= 0 || tiling.cuSeqlen <= 0 ||
        tiling.dim <= 0 || coreNum == 0) {
        return plan;
    }

    if (tiling.dim <= MAX_DIM_TILE_SIZE) {
        plan.caseKind = FN_TILING_CASE_TOKEN_FIRST;
        plan.executionPlan = FN_EXECUTION_PLAN_CUTBS;
        plan.baseDimChoice = ChooseFnTokenFirstBaseDimChoice(tiling.dim);
    } else {
        plan.caseKind = FN_TILING_CASE_TOKEN_DIM_CO_SPLIT;
        plan.executionPlan = FN_EXECUTION_PLAN_CUTBSD;
        plan.baseDimChoice = ChooseFnTokenDimCoSplitBaseDimChoice(context, tiling.dim, ubSize, coreNum);
    }

    if (plan.baseDimChoice.baseDim <= 0 || plan.baseDimChoice.baseDimCnt <= 0) {
        return {};
    }

    plan.baseDimChoice.gridSize = tiling.batch * plan.baseDimChoice.baseDimCnt;
    plan.tokenBlockChoice =
        ChooseUnifiedFnTokenBlockPlan(context, tiling, plan.baseDimChoice, plan.executionPlan, coreNum);
    if (!plan.tokenBlockChoice.enabled || plan.tokenBlockChoice.tokenBlockSize <= 0 ||
        plan.tokenBlockChoice.tokenBlockCnt <= 0 || plan.tokenBlockChoice.gridSize <= 0) {
        return {};
    }

    plan.tokenCoreMapping = BuildFnTokenCoreMappingChoice(plan.tokenBlockChoice.tokenBlockCnt,
                                                          plan.baseDimChoice.baseDimCnt, plan.executionPlan, coreNum);
    if (plan.tokenCoreMapping.tokenCoreBudget <= 0 || plan.tokenCoreMapping.blockDim <= 0) {
        return {};
    }
    if (plan.tokenCoreMapping.blockDim > static_cast<int64_t>(coreNum)) {
        plan.tokenCoreMapping.blockDim = static_cast<int64_t>(coreNum);
    }
    return plan;
}

} // namespace optiling::causal_conv1d_host

#endif // CAUSAL_CONV1D_TILING_PLANNER_H
