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
 * \file apply_top_k_top_p_custom.cpp
 * \brief
 */
#include "apply_top_k_top_p_custom.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ApplyTopKTopPCustom);

const aclTensor* ApplyTopKTopPCustom(
  const aclTensor* sortedValue, const aclTensor* sortedIndices, const aclTensor* p, const aclTensor* k,
  aclOpExecutor* executor)
{
    L0_DFX(ApplyTopKTopPCustom, sortedValue, sortedIndices, p, k);
    auto output = executor->AllocTensor(sortedValue->GetViewShape(), sortedValue->GetDataType());
    if (p == nullptr) {
        p = executor->AllocTensor(sortedValue->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (k == nullptr) {
        k = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    ADD_TO_LAUNCHER_LIST_AICORE(ApplyTopKTopPCustom, OP_INPUT(sortedValue, sortedIndices, p, k), OP_OUTPUT(output));

    return output;
}
}  // namespace l0op
