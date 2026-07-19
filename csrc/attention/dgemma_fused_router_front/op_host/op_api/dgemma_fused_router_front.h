/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef PTA_NPU_OP_API_DGEMMA_FUSED_ROUTER_FRONT_H
#define PTA_NPU_OP_API_DGEMMA_FUSED_ROUTER_FRONT_H
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
namespace l0op {
const aclTensor *DgemmaFusedRouterFront(
    const aclTensor *x, const aclTensor *scale, const aclTensor *projWeight,
    const aclTensor *normScratch, const aclTensor *logitsScratch,
    const aclTensor *perExpertScale, const aclTensor *syncScratch,
    float epsilon, int64_t hiddenSize, int64_t numExperts, int64_t topK,
    int64_t syncBase,
    aclTensor **topkIds,
    aclOpExecutor *executor);
} // namespace l0op
#endif
