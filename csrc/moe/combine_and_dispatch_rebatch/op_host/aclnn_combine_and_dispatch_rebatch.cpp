// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <string.h>
#include "graph/types.h"
#include "aclnn_combine_and_dispatch_rebatch.h"
#include <cstdio>

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerCombineAndDispatchRebatchGetWorkspaceSize(
    const aclTensor *batchBeforeDispatch,
    const aclTensor *residualsBeforeDispatch,
    const aclTensor *cutIdx,
    const aclTensor *prefixIndex,
    const aclTensor *discardLabel,
    const aclTensor *discardTokens,
    const aclTensor *discardMoeReqIdTensor,
    const aclTensor *reqIdList,
    const aclTensor *discardResiduals,
    const aclTensor *runningTopk,
    const aclTensor *discardTopk,
    const aclTensor *runningTopkWeight,
    const aclTensor *discardTopkWeight,
    const aclTensor *moeCntInput,
    const aclTensor *actualBs,
    const aclTensor *historyBs,
    const aclTensor *runningBs,
    const aclTensor *originalBs,
    int64_t isCombineRebatch,
    int64_t maxBs,
    int64_t k,
    // const aclTensor *rebatchOutput,
    const aclTensor *rebatchResiduals,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerCombineAndDispatchRebatch(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

aclnnStatus aclnnCombineAndDispatchRebatchGetWorkspaceSize(
    const aclTensor *batchBeforeDispatch,
    const aclTensor *residualsBeforeDispatch,
    const aclTensor *cutIdx,
    const aclTensor *prefixIndex,
    const aclTensor *discardLabel,
    const aclTensor *discardTokens,
    const aclTensor *discardMoeReqIdTensor,
    const aclTensor *reqIdList,
    const aclTensor *discardResiduals,
    const aclTensor *runningTopk,
    const aclTensor *discardTopk,
    const aclTensor *runningTopkWeight,
    const aclTensor *discardTopkWeight,
    const aclTensor *moeCntInput,
    const aclTensor *actualBs,
    const aclTensor *historyBs,
    const aclTensor *runningBs,
    const aclTensor *originalBs,
    int64_t isCombineRebatch,
    int64_t maxBs,
    int64_t k,
    // const aclTensor *rebatchOutput,
    const aclTensor *rebatchResiduals,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerCombineAndDispatchRebatchGetWorkspaceSize(
        batchBeforeDispatch,
        residualsBeforeDispatch,
        cutIdx,
        prefixIndex,
        discardLabel,
        discardTokens,
        discardMoeReqIdTensor,
        reqIdList,
        discardResiduals,
        runningTopk,
        discardTopk,
        runningTopkWeight,
        discardTopkWeight,
        moeCntInput,
        actualBs,
        historyBs,
        runningBs,
        originalBs,
        isCombineRebatch,
        maxBs,
        k,
        // rebatchOutput,
        rebatchResiduals,
        workspaceSize,
        executor);
}

aclnnStatus aclnnCombineAndDispatchRebatch(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    return aclnnInnerCombineAndDispatchRebatch(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
