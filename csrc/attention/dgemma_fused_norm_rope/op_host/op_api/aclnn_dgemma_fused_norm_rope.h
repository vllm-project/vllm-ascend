/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file aclnn_dgemma_fused_norm_rope.h
 *  \brief ACLNN C-API for DgemmaFusedNormRope. */
#ifndef OP_API_ACLNN_DGEMMA_FUSED_NORM_ROPE_H
#define OP_API_ACLNN_DGEMMA_FUSED_NORM_ROPE_H
#include "aclnn/aclnn_base.h"
#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnDgemmaFusedNormRopeGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *qWeight, const aclTensor *kWeight,
    const aclTensor *cos, const aclTensor *sin,
    float epsilon, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    aclTensor *qOut, aclTensor *kOut, aclTensor *vOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnDgemmaFusedNormRope(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif // OP_API_ACLNN_DGEMMA_FUSED_NORM_ROPE_H
