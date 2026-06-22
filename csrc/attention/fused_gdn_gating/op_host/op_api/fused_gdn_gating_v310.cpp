/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_gdn_gating_v310.cpp
 * \brief
 */
#include "fused_gdn_gating_v310.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

// 绑定编译实体到之前 def.cpp 注册的子配置项
OP_TYPE_REGISTER(FusedGdnGatingV310);

static constexpr FusedGdnGatingV310Output kNullOutput{nullptr, nullptr};

FusedGdnGatingV310Output FusedGdnGatingV310(const aclTensor *aLog, const aclTensor *a,
                                            const aclTensor *b, const aclTensor *dtBias,
                                            float beta, float threshold,
                                            aclOpExecutor *executor)
{
    L0_DFX(FusedGdnGatingV310, aLog, a, b, dtBias, beta, threshold);

    Format format = Format::FORMAT_ND;

    // 独立分配 310P 输出临时计算图节点
    auto g = executor->AllocTensor(DataType::DT_FLOAT, format, format);
    OP_CHECK(g != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "g AllocTensor failed."),
             return kNullOutput);

    auto betaOutput = executor->AllocTensor(DataType::DT_FLOAT16, format, format);
    OP_CHECK(betaOutput != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "beta_output AllocTensor failed."),
             return kNullOutput);

    // 执行 Shape 推导 (输入参数顺序与大一统顺序完美镜面对齐)
    auto ret = INFER_SHAPE(FusedGdnGatingV310,
                           OP_INPUT(aLog, a, b, dtBias),
                           OP_OUTPUT(g, betaOutput),
                           OP_ATTR(beta, threshold));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return kNullOutput,
                        "FusedGdnGatingV310 InferShape failed.");

    // 推入硬件 AIV 核心调度链
    ret = ADD_TO_LAUNCHER_LIST_AICORE(FusedGdnGatingV310,
                                      OP_INPUT(aLog, a, b, dtBias),
                                      OP_OUTPUT(g, betaOutput),
                                      OP_ATTR(beta, threshold));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return kNullOutput,
        "FusedGdnGatingV310 ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return FusedGdnGatingV310Output{g, betaOutput};
}

} // namespace l0op
