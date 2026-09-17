// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#ifdef __cplusplus
extern "C" {
#endif
ACLNN_API aclnnStatus aclnnAttnResFwdWithAddGetWorkspaceSize(
    const aclTensor *prefix, const aclTensor *blocks, const aclTensor *proj,
    const aclTensor *norm, const aclTensor *addend, double eps,
    aclTensor *output, aclTensor *prefixOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnAttnResFwdWithAdd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
