/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef PTA_NPU_OP_API_DGEMMA_APPLY_ROUTER_SCALE_H
#define PTA_NPU_OP_API_DGEMMA_APPLY_ROUTER_SCALE_H
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
namespace l0op {
const aclTensor *DgemmaApplyRouterScale(
    const aclTensor *weights, const aclTensor *ids, const aclTensor *scale,
    aclOpExecutor *executor);
}
#endif
