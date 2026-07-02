/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_ACLNN_CAUSAL_CONV1D_H_
#define OP_API_ACLNN_CAUSAL_CONV1D_H_

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnCausalConv1dGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * biasOptional : optional
 * convStates : required
 * queryStartLocOptional : optional
 * cacheIndicesOptional : optional
 * initialStateModeOptional : optional
 * numAcceptedTokensOptional : optional
 * activationMode : optional
 * padSlotId : optional
 * runMode : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCausalConv1dGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *biasOptional,
    const aclTensor *convStates,
    const aclIntArray *queryStartLocOptional,
    const aclIntArray *cacheIndicesOptional,
    const aclIntArray *initialStateModeOptional,
    const aclIntArray *numAcceptedTokensOptional,
    int64_t activationMode,
    int64_t padSlotId,
    int64_t runMode,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnCausalConv1d
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCausalConv1d(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif