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
 * \file scatter_nd_update_struct.h
 * \brief tiling base data
 */

#ifndef SCATTER_ND_UPDATE_STTRUCT_H
#define SCATTER_ND_UPDATE_STTRUCT_H

static constexpr uint16_t MAX_RANK_COUNT = 7;
static constexpr uint16_t MAX_SHAPE_RANK = 8;

class ScatterNdUpdateRegBaseTilingData {
public:
    uint64_t blockNum;
    uint32_t rankSize;
    uint64_t blockTilingSize;
    uint64_t tailBlockTilingSize;
    uint32_t ubTilingSize;
    uint64_t sliceSize;
    uint64_t outPutShape[MAX_SHAPE_RANK];
    uint64_t strideList[MAX_SHAPE_RANK];
    uint64_t outputStorageShapeSize;
    /* for deterministic */
    int64_t varInAxis;
    int64_t varStorageInAxis;
    int64_t indexRankSize;
    int64_t afterAxis;

    int64_t updateLoopSize;
    int64_t updateTailNum;
    int64_t indicesLoopSize;
    int64_t indiceTailNum;

    int64_t usedCoreNumBefore;
    int64_t usedCoreNumAfter;
    int64_t indicesFactor;
    int64_t afterAxisFactor;
    /* split after */
    int64_t eachCoreAfterAxisCount;
    int64_t tailCoreAfterAxisCount;
    int64_t tailUpdateLoopSize;
    int64_t tailUpdateAxisNum;

    int64_t ubQuantaIndxFactor;
    int64_t ubRowFactor;
    int64_t eachCoreIndexCount;
    int64_t tailCoreIndexCount;
    int64_t eachCoreVarCount;
    int64_t tailCoreVarCount;
    int64_t isSplitAfterAxis;
    int64_t isDeterministic;
    int64_t isSimtWithSort;
    int64_t isSimdWithSort;
    int64_t isSimdNonDeterministic;
    int64_t isMask;
    int64_t isSplitOneLine;
    int64_t IsContiguous;

    /* for deterministic */
    int64_t normCoreHandleIdx;
    int64_t tailCoreHandleIdx;
    int32_t calcMaskUsedCoreNum;
    int64_t maskNormBlockLen = 0;
    int64_t maskTailBlockLen = 0;
    int64_t isDeterminSimt = 0;

    int64_t indicesUbFactor = 0;
    int64_t normBlockLoop = 0;
    int64_t tailBlockLoop = 0;
    int64_t normBlockTail = 0;
    int64_t tailBlockTail = 0;
};
#endif
