/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file aclnn_dgemma_fused_qkv_proj_norm_rope.h */
#ifndef OP_API_ACLNN_DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_H
#define OP_API_ACLNN_DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_H
#include "aclnn/aclnn_base.h"
#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnDgemmaFusedQkvProjNormRopeGetWorkspaceSize(
    const aclTensor *hidden, const aclTensor *wqkv,
    const aclTensor *qWeight, const aclTensor *kWeight,
    const aclTensor *cos, const aclTensor *sin, aclTensor *qkvScratch,
    float epsilon, int64_t numQHeads, int64_t numKvHeads, int64_t headDim, int64_t hiddenSize,
    int64_t syncBase,
    aclTensor *qkvScratchOut, aclTensor *qOut, aclTensor *kOut, aclTensor *vOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnDgemmaFusedQkvProjNormRope(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif // OP_API_ACLNN_DGEMMA_FUSED_QKV_PROJ_NORM_ROPE_H
