/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file scatter_nd_update_asc.cpp
 * \brief
 */

#include "scatter_nd_update_asc.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(ScatterNdUpdateAsc);

// AiCore的执行逻辑
inline static const aclTensor* ScatterNdUpdateAscAiCore(const aclTensor* self, const aclTensor* indices,
                                                        const aclTensor* updates, const aclIntArray* strides,
                                                        aclOpExecutor* executor)
{
    L0_DFX(ScatterNdUpdateAscAiCore, self, indices, updates);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ScatterNdUpdateAsc, OP_INPUT(self, indices, updates),
                                                 OP_OUTPUT(self), OP_ATTR(strides));
    CHECK_RET(retAicore == ACLNN_SUCCESS, nullptr);
    return self;
}

const aclTensor* ScatterNdUpdateAsc(const aclTensor* self, const aclTensor* indices, const aclTensor* updates,
                                    const aclIntArray* strides, aclOpExecutor* executor)
{
    return ScatterNdUpdateAscAiCore(self, indices, updates, strides, executor);
}
}  // namespace l0op
