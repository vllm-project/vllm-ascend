/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#ifndef OP_API_INC_ACLNN_KDA_LAYOUT_SWAP12_H
#define OP_API_INC_ACLNN_KDA_LAYOUT_SWAP12_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnKdaLayoutSwap12GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *dependencyOptional,
    const aclTensor *yOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnKdaLayoutSwap12(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
