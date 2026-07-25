/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ACLNN_FUSED_GDN_DECODE_H
#define ACLNN_FUSED_GDN_DECODE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnFusedGdnDecodeGetWorkspaceSize(
    const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
    const aclTensor *aLog, const aclTensor *dtBias, aclTensor *stateRef,
    const aclTensor *ssmStateIndices, float scale, float softplusThreshold,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnFusedGdnDecode(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_FUSED_GDN_DECODE_H
