// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
extern "C" {
ACLNN_API aclnnStatus aclnnAttnResFwdPrefillGetWorkspaceSize(
    const aclTensor *prefix, const aclTensor *blocks, const aclTensor *proj,
    const aclTensor *norm, const aclTensor *addend, const aclTensor *outputNorm,
    double eps, int64_t validBlocks, int64_t blockTokenStride, int64_t blockWriteIdx,
    double outputNormEps, bool saveMaterialized, bool mix, bool fuseAdd,
    aclTensor *output, aclTensor *prefixOut, aclTensor *materialized,
    uint64_t *workspaceSize, aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnAttnResFwdPrefill(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
}
