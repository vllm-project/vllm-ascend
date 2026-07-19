/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file aclnn_dgemma_fused_router_front.h */
#ifndef OP_API_ACLNN_DGEMMA_FUSED_ROUTER_FRONT_H
#define OP_API_ACLNN_DGEMMA_FUSED_ROUTER_FRONT_H
#include "aclnn/aclnn_base.h"
#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnDgemmaFusedRouterFrontGetWorkspaceSize(
    const aclTensor *x, const aclTensor *scale, const aclTensor *projWeight,
    const aclTensor *normScratch, const aclTensor *logitsScratch,
    const aclTensor *perExpertScale, const aclTensor *syncScratch,
    float epsilon, int64_t hiddenSize, int64_t numExperts, int64_t topK,
    int64_t syncBase,
    aclTensor *topkWeights, aclTensor *topkIds,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnDgemmaFusedRouterFront(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif // OP_API_ACLNN_DGEMMA_FUSED_ROUTER_FRONT_H
