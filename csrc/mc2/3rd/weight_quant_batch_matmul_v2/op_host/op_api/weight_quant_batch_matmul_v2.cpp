/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "weight_quant_batch_matmul_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Mc2WeightQuantBatchMatmulV2);

constexpr int64_t TYPE_FP16 = 1;
constexpr int64_t TYPE_BF16 = 27;

const aclTensor *Mc2WeightQuantBatchMatmulV2(const aclTensor *x, const aclTensor *weight,
                                             const aclTensor *antiquantScale, const aclTensor *antiquantOffsetOptional,
                                             const aclTensor *quantScaleOptional, const aclTensor *quantOffsetOptional,
                                             const aclTensor *biasOptional, bool transposeX, bool transposeWeight,
                                             int antiquantGroupSize, int64_t dtype, int innerPrecise,
                                             aclOpExecutor *executor)
{
    L0_DFX(Mc2WeightQuantBatchMatmulV2, x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional,
           quantOffsetOptional, biasOptional, transposeX, transposeWeight, antiquantGroupSize, dtype, innerPrecise);
    DataType outType = x->GetDataType();
    if (dtype != -1) {
        outType = static_cast<DataType>(dtype);
    }
    auto output = executor->AllocTensor(outType, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret =
        INFER_SHAPE(Mc2WeightQuantBatchMatmulV2,
                    OP_INPUT(x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional,
                             quantOffsetOptional, biasOptional),
                    OP_OUTPUT(output), OP_ATTR(transposeX, transposeWeight, antiquantGroupSize, dtype, innerPrecise));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed.");
        return nullptr;
    }
    ret = ADD_TO_LAUNCHER_LIST_AICORE(Mc2WeightQuantBatchMatmulV2,
                                      OP_INPUT(x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional,
                                               quantOffsetOptional, biasOptional),
                                      OP_OUTPUT(output),
                                      OP_ATTR(transposeX, transposeWeight, antiquantGroupSize, dtype, innerPrecise));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return output;
}
} // namespace l0op
