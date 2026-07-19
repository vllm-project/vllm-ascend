/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "dgemma_apply_router_scale.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
using namespace op;
namespace l0op {
OP_TYPE_REGISTER(DgemmaApplyRouterScale);

const aclTensor *DgemmaApplyRouterScale(
    const aclTensor *weights, const aclTensor *ids, const aclTensor *scale,
    aclOpExecutor *executor)
{
    L0_DFX(DgemmaApplyRouterScale, weights, ids, scale);
    const Format format = Format::FORMAT_ND;
    auto out = executor->AllocTensor(weights->GetViewShape(), weights->GetDataType(), format);
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out AllocTensor failed."), return nullptr);

    auto ret = INFER_SHAPE(DgemmaApplyRouterScale,
                           OP_INPUT(weights, ids, scale), OP_OUTPUT(out), OP_ATTR());
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "DgemmaApplyRouterScale InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(DgemmaApplyRouterScale,
                                      OP_INPUT(weights, ids, scale), OP_OUTPUT(out), OP_ATTR());
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
        "DgemmaApplyRouterScale ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}
}
