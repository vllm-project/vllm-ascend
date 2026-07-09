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
 * \file scatter_nd_update.cpp
 * \brief
 */

#include "scatter_nd_update.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(ScatterNdUpdate);

// AiCore支持的ScatterUpdate类型
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL,  op::DataType::DT_BF16,
    op::DataType::DT_INT64, op::DataType::DT_INT8,    op::DataType::DT_INT32, op::DataType::DT_INT16};

static const std::initializer_list<op::DataType> ASCEND950_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,    op::DataType::DT_FLOAT16, op::DataType::DT_BOOL,        op::DataType::DT_BF16,
    op::DataType::DT_INT64,    op::DataType::DT_INT8,    op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN,
    op::DataType::DT_HIFLOAT8, op::DataType::DT_UINT8};

inline static bool IsAiCoreSupport(const aclTensor* self)
{
    // ScatterNdUpdate只需要判断self
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        return CheckType(self->GetDataType(), ASCEND950_AICORE_DTYPE_SUPPORT_LIST);
    }
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AiCore的执行逻辑
inline static const aclTensor* ScatterNdUpdateAiCore(const aclTensor* self, const aclTensor* indices,
                                                     const aclTensor* updates, bool use_locking,
                                                     aclOpExecutor* executor)
{
    L0_DFX(ScatterNdUpdateAiCore, self, indices, updates, use_locking);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ScatterNdUpdate, OP_INPUT(self, indices, updates), OP_OUTPUT(self),
                                                 OP_ATTR(use_locking));
    CHECK_RET(retAicore == ACLNN_SUCCESS, nullptr);
    return self;
}

// AiCPU的执行逻辑
inline static const aclTensor* ScatterNdUpdateAiCPU(const aclTensor* self, const aclTensor* indices,
                                                    const aclTensor* updates, bool use_locking, aclOpExecutor* executor)
{
    L0_DFX(ScatterNdUpdateAiCPU, self, indices, updates, use_locking);

    static internal::AicpuTaskSpace space("ScatterNdUpdate", ge::DEPEND_IN_SHAPE, true);
    space.SetRef(0);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(ScatterNdUpdate, OP_ATTR_NAMES({"Tindices", "T", "use_locking"}),
                                          OP_INPUT(self, indices, updates), OP_OUTPUT(self),
                                          OP_ATTR(indices->GetDataType(), updates->GetDataType(), use_locking));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return self;
}

const aclTensor* ScatterNdUpdate(const aclTensor* self, const aclTensor* indices, const aclTensor* updates,
                                 bool use_locking, aclOpExecutor* executor)
{
    if (IsAiCoreSupport(self)) {
        return ScatterNdUpdateAiCore(self, indices, updates, use_locking, executor);
    } else {
        return ScatterNdUpdateAiCPU(self, indices, updates, use_locking, executor);
    }
}
} // namespace l0op
