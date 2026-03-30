/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/types.h"
#include "aclnn_lightning_attention_prefill.h"


#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerLightningAttentionPrefillGetWorkspaceSize(
        const aclTensor *query,
        const aclTensor *key,
        const aclTensor *value,
        const aclTensor *slopeRate,
        const aclTensor *kvHistoryOptional,
        int64_t blockSize,
        const aclIntArray *actualSeqLen,
        char *inputLayoutOptional,
        const aclTensor *attentionOut,
        const aclTensor *kvCachesOut,
        uint64_t *workspaceSize,
        aclOpExecutor **executor);

extern aclnnStatus aclnnInnerLightningAttentionPrefill(
        void *workspace,
        uint64_t workspaceSize,
        aclOpExecutor *executor,
        aclrtStream stream);


aclnnStatus aclnnLightningAttentionPrefillGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *slopeRate,
    const aclTensor *kvHistoryOptional,
    int64_t blockSize,
    const aclIntArray *actualSeqLen,
    char *inputLayoutOptional,
    const aclTensor *attentionOut,
    const aclTensor *kvCachesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    aclnnStatus ret = aclnnInnerLightningAttentionPrefillGetWorkspaceSize(
            query, key, value, slopeRate, kvHistoryOptional, blockSize, actualSeqLen,
            inputLayoutOptional, attentionOut, kvCachesOut, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnLightningAttentionPrefill(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerLightningAttentionPrefill(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
