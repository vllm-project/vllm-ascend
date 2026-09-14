/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file scatter_nd_update_sk.cpp
 * \brief
 */

#include "scatter_nd_update_sk.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(ScatterNdUpdateSk);

// AiCore的执行逻辑 (arch22: 910b/910_93)
inline static const aclTensor* ScatterNdUpdateSkAiCore(const aclTensor* self, const aclTensor* indices,
                                                       const aclTensor* updates, const aclIntArray* strides,
                                                       bool use_locking, aclOpExecutor* executor)
{
    L0_DFX(ScatterNdUpdateSkAiCore, self, indices, updates);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ScatterNdUpdateSk, OP_INPUT(self, indices, updates),
                                                 OP_OUTPUT(self), OP_ATTR(strides, use_locking));
    CHECK_RET(retAicore == ACLNN_SUCCESS, nullptr);
    return self;
}

const aclTensor* ScatterNdUpdateSk(const aclTensor* self, const aclTensor* indices, const aclTensor* updates,
                                   const aclIntArray* strides, bool use_locking, aclOpExecutor* executor)
{
    return ScatterNdUpdateSkAiCore(self, indices, updates, strides, use_locking, executor);
}
} // namespace l0op
