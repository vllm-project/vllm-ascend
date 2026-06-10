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
 * \file attention_update_v2.cpp
 * \brief
 */

#include "attention_update_v2.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(AttentionUpdateV2);

const std::tuple<const aclTensor*, const aclTensor*> AttentionUpdateV2(const aclTensor *lse, const aclTensor *go,
                            int64_t updateType, aclOpExecutor *executor) {
    L0_DFX(AttentionUpdateV2, lse, go, updateType);

    // go: [sp, bsh, hd] -> out: [bsh, hd]
    const auto& goShape = go->GetViewShape();
    op::Shape outShape({goShape.GetDim(1), goShape.GetDim(2)});
    auto out = executor->AllocTensor(outShape, go->GetDataType(), op::Format::FORMAT_ND);

    // lse: [sp, bsh] -> lseOut: [bsh]
    const auto& lseShape = lse->GetViewShape();
    op::Shape lseOutShape({lseShape.GetDim(1)});
    auto lseOut = executor->AllocTensor(lseOutShape, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AttentionUpdateV2,
                              OP_INPUT(lse, go),
                              OP_OUTPUT(out, lseOut),
                              OP_ATTR(updateType));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AttentionUpdateV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }

    return std::tuple<aclTensor*, aclTensor*>(out, lseOut);
}
}  // namespace l0op