/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_LIGHTNING_ATTENTION_PREFILL_H_
#define ACLNN_LIGHTNING_ATTENTION_PREFILL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnLightningAttentionPrefillGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * value : required
 * slopeRate : required
 * kvHistoryOptional : optional
 * blockSize : required
 * actualSeqLen : required
 * inputLayoutOptional : optional
 * attentionOut : required
 * kvCachesOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
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
    aclOpExecutor **executor);

/* funtion: aclnnLightningAttentionPrefill
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLightningAttentionPrefill(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
