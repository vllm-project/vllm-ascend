/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "kda_layout_swap12.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(KdaLayoutSwap12);

const std::array<const aclTensor *, 1> KdaLayoutSwap12(
    const aclTensor *x,
    const aclTensor *dependency,
    const aclTensor *y,
    aclOpExecutor *executor)
{
    L0_DFX(KdaLayoutSwap12, x, dependency, y);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        KdaLayoutSwap12,
        OP_INPUT(x, dependency),
        OP_OUTPUT(y),
        OP_ATTR());
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE KdaLayoutSwap12 failed.");
        return {nullptr};
    }
    (void)executor;
    return {y};
}

const std::array<const aclTensor *, 1> KdaLayoutSwap12(
    const aclTensor *x,
    const aclTensor *y,
    aclOpExecutor *executor)
{
    return KdaLayoutSwap12(x, nullptr, y, executor);
}
} // namespace l0op
