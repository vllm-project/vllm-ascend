// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
extern "C" {
// A null output selects bounded IDs-only execution; selectedRegionIds is required.
ACLNN_API aclnnStatus aclnnIndexerRegionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensor *weights, const aclTensor *keySummary,
    const aclTensor *actualKvLengths, const aclTensor *blockTable, const aclTensor *cuSeqLens,
    int64_t maxRegionLen, int64_t topK, int64_t regionSize, double headScale,
    int64_t cacheMode, int64_t maskMode, int64_t sortMode,
    const aclTensor *output, const aclTensor *selectedRegionIds,
    uint64_t *workspaceSize, aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnIndexerRegionScore(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
}
