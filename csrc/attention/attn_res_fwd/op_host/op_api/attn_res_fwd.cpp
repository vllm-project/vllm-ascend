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
 * \file attn_res_fwd.cpp
 * \brief AttnResFwd L0 op_api
 */
#include "attn_res_fwd.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(AttnResFwd);

const std::array<const aclTensor *, 3> AttnResFwd(const aclTensor *prefixSum, const aclTensor *blockResidual,
                                                  const aclTensor *projWeight, const aclTensor *normWeight,
                                                  double normEps, bool needBackward, aclOpExecutor *executor)
{
    L0_DFX(AttnResFwd, prefixSum, blockResidual, projWeight, normWeight, normEps, needBackward);

    DataType outType = prefixSum->GetDataType();
    Format format = Format::FORMAT_ND;
    auto hiddenStates = executor->AllocTensor(outType, format, format);
    auto invRms = executor->AllocTensor(DataType::DT_FLOAT, format, format);
    auto probs = executor->AllocTensor(DataType::DT_FLOAT, format, format);

    auto ret = INFER_SHAPE(AttnResFwd, OP_INPUT(prefixSum, blockResidual, projWeight, normWeight),
                           OP_OUTPUT(hiddenStates, invRms, probs),
                           OP_ATTR(static_cast<float>(normEps), needBackward));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AttnResFwd InferShape failed.");
        return {nullptr, nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(AttnResFwd, OP_INPUT(prefixSum, blockResidual, projWeight, normWeight),
                                      OP_OUTPUT(hiddenStates, invRms, probs),
                                      OP_ATTR(static_cast<float>(normEps), needBackward));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AttnResFwd ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }

    return {hiddenStates, invRms, probs};
}
} // namespace l0op
