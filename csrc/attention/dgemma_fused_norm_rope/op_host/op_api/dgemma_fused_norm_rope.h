/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef PTA_NPU_OP_API_DGEMMA_FUSED_NORM_ROPE_H
#define PTA_NPU_OP_API_DGEMMA_FUSED_NORM_ROPE_H
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
namespace l0op {
struct DgemmaFusedNormRopeOutput {
    const aclTensor *q_out;
    const aclTensor *k_out;
    const aclTensor *v_out;
};
DgemmaFusedNormRopeOutput DgemmaFusedNormRope(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *qWeight, const aclTensor *kWeight,
    const aclTensor *cos, const aclTensor *sin,
    float epsilon, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    aclOpExecutor *executor);
} // namespace l0op
#endif
