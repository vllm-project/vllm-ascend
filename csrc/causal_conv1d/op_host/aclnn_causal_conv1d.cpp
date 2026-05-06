/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_causal_conv1d.h"
#include "aclnn_kernels/common/op_error_check.h"
#include <algorithm>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <climits>

#ifdef __cplusplus
extern "C" {
#endif

constexpr int64_t CONV_STATE_DIM = 3;
constexpr int64_t CONV_STATE_STRIDE_0 = 0;
constexpr int64_t CONV_STATE_STRIDE_1 = 1;
/* function: aclnnInnerCausalConv1dGetWorkspaceSize
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
 * stride0 : required
 * stride1 : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
extern aclnnStatus aclnnInnerCausalConv1dGetWorkspaceSize(
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
    int64_t stride0,
    int64_t stride1,
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
extern aclnnStatus aclnnInnerCausalConv1d(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);


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
    aclOpExecutor **executor)
{

    int64_t* convStateStridesValues = nullptr;
    uint64_t convStateStridesNum = 0;

    auto ret = aclGetViewStrides(convStates, &convStateStridesValues, &convStateStridesNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CHECK_COND(convStateStridesNum == CONV_STATE_DIM, ACLNN_ERR_PARAM_INVALID, "convStateStrideDim must be 3.");

    int64_t stride0 = convStateStridesValues[CONV_STATE_STRIDE_0];
    int64_t stride1 = convStateStridesValues[CONV_STATE_STRIDE_1];

    ret = aclnnInnerCausalConv1dGetWorkspaceSize(x, weight, biasOptional, convStates, queryStartLocOptional,
                                                             cacheIndicesOptional, initialStateModeOptional,
                                                             numAcceptedTokensOptional, activationMode, padSlotId,
                                                             runMode, stride0, stride1,
                                                             out, workspaceSize, executor);
    delete[] convStateStridesValues;
    return ret;
}

aclnnStatus aclnnCausalConv1d(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream) {
    return aclnnInnerCausalConv1d(workspace, workspaceSize, executor, stream);
}


#ifdef __cplusplus
}
#endif