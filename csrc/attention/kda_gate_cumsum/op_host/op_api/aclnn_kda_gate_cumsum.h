/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#ifndef OP_API_INC_ACLNN_KDA_GATE_CUMSUM_H
#define OP_API_INC_ACLNN_KDA_GATE_CUMSUM_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnKdaGateCumsumGetWorkspaceSize(
    const aclTensor *g,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    int64_t chunkSize,
    bool useGateInKernel,
    bool safeGate,
    double lowerBound,
    const char *layout,
    const aclTensor *gkOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnKdaGateCumsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
