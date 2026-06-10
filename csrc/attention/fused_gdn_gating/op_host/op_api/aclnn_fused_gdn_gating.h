/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file aclnn_fused_gdn_gating.h
 * \brief ACLNN C-API for FusedGdnGating.
 */

#ifndef OP_API_ACLNN_FUSED_GDN_GATING_H
#define OP_API_ACLNN_FUSED_GDN_GATING_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief FusedGdnGating phase-1: compute required workspace size.
 * @param [in]  aLog        : A_log,       [num_heads],           dtype fp32.
 * @param [in]  a           : a,           [batch, num_heads],    dtype bf16/fp16.
 * @param [in]  b           : b,           [batch, num_heads],    dtype bf16/fp16.
 * @param [in]  dtBias      : dt_bias,     [num_heads],           dtype fp32.
 * @param [in]  beta        : softplus beta (default 1.0).
 * @param [out] g           : output gate,   [1, batch, num_heads], dtype fp32.
 * @param [out] betaOutput  : sigmoid(b),    [1, batch, num_heads], same dtype as a/b.
 * @param [out] workspaceSize: required workspace bytes on device.
 * @param [out] executor    : op executor handle.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedGdnGatingGetWorkspaceSize(
    const aclTensor *aLog, const aclTensor *a, const aclTensor *b,
    const aclTensor *dtBias, float beta,
    aclTensor *g, aclTensor *betaOutput,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief FusedGdnGating phase-2: launch the kernel.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedGdnGating(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_FUSED_GDN_GATING_H
