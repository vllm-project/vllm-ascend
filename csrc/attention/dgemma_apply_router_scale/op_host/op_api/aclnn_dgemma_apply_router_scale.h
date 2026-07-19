/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OP_API_ACLNN_DGEMMA_APPLY_ROUTER_SCALE_H
#define OP_API_ACLNN_DGEMMA_APPLY_ROUTER_SCALE_H
#include "aclnn/aclnn_base.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default"))) aclnnStatus aclnnDgemmaApplyRouterScaleGetWorkspaceSize(
    const aclTensor *weights, const aclTensor *ids, const aclTensor *scale,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnDgemmaApplyRouterScale(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif
