/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_CHUNK_GATED_DELTA_RULE_FWD_H_H
#define OP_API_INC_ACLNN_CHUNK_GATED_DELTA_RULE_FWD_H_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize
 * parameters (order aligned with chunk_gated_delta_rule_fwd_h Python API):
 * k : required
 * w : required
 * u : required
 * gOptional : optional, only non-null aclTensor is supported
 * gkOptional : optional, reserved (must be nullptr)
 * initalStateOptional : optional
 * outputFinalState : required
 * chunkSize : required
 * saveNewValue : reserved (must be true)
 * cuSeqlensOptional : optional
 * chunkIndicesOptional : optional
 * useExp2 : reserved (must be false)
 * transposeStateLayout : reserved (must be false)
 * hOut : required
 * vNewOut : required
 * finalStateOut : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *initalStateOptional,
    bool outputFinalState,
    int64_t chunkSize,
    bool saveNewValue,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool useExp2,
    bool transposeStateLayout,
    const aclTensor *hOut,
    const aclTensor *vNewOut,
    const aclTensor *finalStateOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnChunkGatedDeltaRuleFwdH
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleFwdH(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif
