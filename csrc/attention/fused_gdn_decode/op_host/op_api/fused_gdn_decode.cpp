/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedGdnDecode);

namespace {
void UseViewShapeForNdTensor(aclTensor *tensor)
{
    if (tensor == nullptr) {
        return;
    }
    const auto &viewShape = tensor->GetViewShape();
    tensor->SetStorageShape(viewShape);
    tensor->SetOriginalShape(viewShape);
    tensor->SetStorageFormat(Format::FORMAT_ND);
    tensor->SetOriginalFormat(Format::FORMAT_ND);
    tensor->SetViewFormat(Format::FORMAT_ND);
}
} // namespace

const aclTensor *FusedGdnDecode(const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
                                const aclTensor *aLog, const aclTensor *dtBias, aclTensor *stateRef,
                                const aclTensor *ssmStateIndices, float scale, float softplusThreshold,
                                aclOpExecutor *executor)
{
    L0_DFX(FusedGdnDecode, mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices, scale, softplusThreshold);
    UseViewShapeForNdTensor(stateRef);
    auto out = executor->AllocTensor(mixedQkv->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out AllocTensor failed."), return nullptr);
    auto stateOut = executor->AllocTensor(stateRef->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    OP_CHECK(stateOut != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "stateOut AllocTensor failed."), return nullptr);

    auto ret = INFER_SHAPE(FusedGdnDecode,
                           OP_INPUT(mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices),
                           OP_OUTPUT(out, stateOut),
                           OP_ATTR(scale, softplusThreshold));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "FusedGdnDecode InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(FusedGdnDecode,
                                      OP_INPUT(mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices),
                                      OP_OUTPUT(out, stateRef),
                                      OP_ATTR(scale, softplusThreshold));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "FusedGdnDecode ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}

} // namespace l0op
