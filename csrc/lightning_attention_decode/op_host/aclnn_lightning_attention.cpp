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
#include "aclnn_lightning_attention_decode.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerLightningAttentionDecodeGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *slopeRate,
    aclTensor *kvCachesRef,
    const aclTensor *slotIds,
    char *inputLayoutOptional,
    const aclTensor *attentionOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerLightningAttentionDecode(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

aclnnStatus aclnnLightningAttentionDecodeGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *slopeRate,
    aclTensor *kvCachesRef,
    const aclTensor *slotIds,
    char *inputLayoutOptional,
    const aclTensor *attentionOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    aclnnStatus ret = aclnnInnerLightningAttentionDecodeGetWorkspaceSize(
        query, key, value, slopeRate, kvCachesRef, slotIds,
        inputLayoutOptional, attentionOut, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnLightningAttentionDecode(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerLightningAttentionDecode(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
