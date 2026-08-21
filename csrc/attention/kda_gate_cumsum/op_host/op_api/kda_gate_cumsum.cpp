/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "kda_gate_cumsum.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(KdaGateCumsum);

const std::array<const aclTensor *, 1> KdaGateCumsum(
    const aclTensor *g,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    int64_t chunkSize,
    bool useGateInKernel,
    bool safeGate,
    double lowerBound,
    const char *layout,
    const aclTensor *gkOut,
    aclOpExecutor *executor)
{
    L0_DFX(KdaGateCumsum, g, aLogOptional, dtBiasOptional, cuSeqlensOptional, chunkSize, useGateInKernel,
           safeGate, lowerBound, layout, gkOut);

    const aclTensor *actualCuSeqlens = nullptr;
    if (cuSeqlensOptional != nullptr) {
        actualCuSeqlens = executor->ConvertToTensor(cuSeqlensOptional, DataType::DT_INT64);
        const_cast<aclTensor *>(actualCuSeqlens)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualCuSeqlens)->SetOriginalFormat(Format::FORMAT_ND);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        KdaGateCumsum,
        OP_INPUT(g, aLogOptional, dtBiasOptional, actualCuSeqlens),
        OP_OUTPUT(gkOut),
        OP_ATTR(chunkSize, useGateInKernel, safeGate, static_cast<float>(lowerBound), layout));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE KdaGateCumsum failed.");
        return {nullptr};
    }
    return {gkOut};
}
} // namespace l0op
