/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file recurrent_kda.cpp
 * \brief
 */
#include "recurrent_kda.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(RecurrentKda);

const std::array<const aclTensor *, 3> RecurrentKda(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *gate,
    const aclTensor *beta,
    aclTensor *initialStateRef,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *ssmStateIndicesOptional,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *numAcceptedTokensOptional,
    const char *layout,
    double scale,
    bool outputFinalState,
    bool inplaceFinalState,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool stateVFirst,
    const aclTensor *attnOut,
    const aclTensor *finalState,
    aclOpExecutor *executor)
{
    L0_DFX(RecurrentKda, query, key, value, gate, beta, initialStateRef, cuSeqlensOptional,
           ssmStateIndicesOptional, aLogOptional, dtBiasOptional, numAcceptedTokensOptional, layout, scale,
           outputFinalState, inplaceFinalState, useQkL2normInKernel, useGateInKernel,
           useBetaSigmoidInKernel, allowNegEigval, safeGate, lowerBound, stateVFirst, attnOut,
           initialStateRef, finalState);

    float scaleAttr = static_cast<float>(scale);
    float lowerBoundAttr = static_cast<float>(lowerBound);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        RecurrentKda,
        OP_INPUT(query, key, value, gate, beta, initialStateRef, cuSeqlensOptional, ssmStateIndicesOptional,
                 aLogOptional, dtBiasOptional, numAcceptedTokensOptional),
        OP_OUTPUT(attnOut, initialStateRef, finalState),
        OP_ATTR(layout, scaleAttr, outputFinalState, inplaceFinalState, useQkL2normInKernel,
                useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, safeGate, lowerBoundAttr,
                stateVFirst));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "RecurrentKda ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr};
    }

    return {attnOut, initialStateRef, finalState};
}
} // namespace l0op
