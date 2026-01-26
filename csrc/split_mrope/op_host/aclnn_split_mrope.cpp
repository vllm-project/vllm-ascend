/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string.h>
#include "graph/types.h"
#include "aclnn_split_mrope.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerSplitMropeGetWorkspaceSize(
    const aclTensor *positions,
    const aclTensor *inQkv,
    const aclTensor *inCosSinCache,
    int64_t numQHeads,
    int64_t numKvHeads,
    int64_t headSize,
    const aclIntArray *mropeSection,
    int64_t isNeoxStyle,
    const aclTensor *outQueryOut,
    const aclTensor *outKeyOut,
    const aclTensor *outValueOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerSplitMrope(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

aclnnStatus aclnnSplitMropeGetWorkspaceSize(
    const aclTensor *positions,
    const aclTensor *inQkv,
    const aclTensor *inCosSinCache,
    int64_t numQHeads,
    int64_t numKvHeads,
    int64_t headSize,
    const aclIntArray *mropeSection,
    int64_t isNeoxStyle,
    const aclTensor *outQueryOut,
    const aclTensor *outKeyOut,
    const aclTensor *outValueOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerSplitMropeGetWorkspaceSize(
        positions,
        inQkv,
        inCosSinCache,
        numQHeads,
        numKvHeads,
        headSize,
        mropeSection,
        isNeoxStyle,
        outQueryOut,
        outKeyOut,
        outValueOut,
        workspaceSize,
        executor
    );
}

aclnnStatus aclnnSplitMrope(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return aclnnInnerSplitMrope(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif