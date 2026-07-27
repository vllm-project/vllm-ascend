/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_CHUNK_FWD_O_H
#define OP_API_INC_ACLNN_CHUNK_FWD_O_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnChunkFwdOGetWorkspaceSize
 * parameters :
 * q : required
 * k : required
 * v : required
 * h : required
 * g : required
 * cuSeqlensOptional : optional
 * chunkOffsetsOptional : optional
 * scale : required
 * chunkSize : required
 * oOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkFwdOGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *h,
    const aclTensor *g,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkOffsetsOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *oOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnChunkFwdO
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkFwdO(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif
