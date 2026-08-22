/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "prompt_flash_attention.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(PromptFlashAttention);

const aclTensor *PromptFlashAttention(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *pseShift,
    const aclTensor *attenMask,
    const aclIntArray *actualSeqLengths,
    const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1,
    const aclTensor *quantScale1,
    const aclTensor *deqScale2,
    const aclTensor *quantScale2,
    const aclTensor *quantOffset2,
    int64_t numHeads,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    const char *inputLayout,
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    int64_t innerPrecise,
    const aclTensor *attentionOut,
    aclOpExecutor *executor) {
    L0_DFX(PromptFlashAttention, query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKv,
           deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
           numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads,
           sparseMode, innerPrecise);

    const aclTensor *actualSeqLengthsTensor = nullptr;
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PromptFlashAttention: executor is nullptr.");
        return nullptr;
    }

    if (actualSeqLengths) {
        actualSeqLengthsTensor = executor->ConvertToTensor(actualSeqLengths, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSeqLengthsTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqLengthsTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqLengthsTensor)->SetOriginalFormat(Format::FORMAT_ND);
    }

    const aclTensor *actualSeqLengthsKvTensor = nullptr;
    if (actualSeqLengthsKv) {
        actualSeqLengthsKvTensor = executor->ConvertToTensor(actualSeqLengthsKv, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSeqLengthsKvTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqLengthsKvTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqLengthsKvTensor)->SetOriginalFormat(Format::FORMAT_ND);
    }

    auto attentionOutOut = executor->AllocTensor(attentionOut->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto ret = INFER_SHAPE(PromptFlashAttention,
        OP_INPUT(query, key, value, pseShift, attenMask, actualSeqLengthsTensor, actualSeqLengthsKvTensor,
                 deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2),
        OP_OUTPUT(attentionOutOut),
        OP_ATTR(numHeads, static_cast<float>(scaleValue), preTokens, nextTokens,
                inputLayout, numKeyValueHeads, sparseMode, innerPrecise));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PromptFlashAttention InferShape failed.");
        return nullptr;
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(PromptFlashAttention,
        OP_INPUT(query, key, value, pseShift, attenMask, actualSeqLengthsTensor, actualSeqLengthsKvTensor,
                 deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2),
        OP_OUTPUT(attentionOutOut),
        OP_ATTR(numHeads, static_cast<float>(scaleValue), preTokens, nextTokens,
                inputLayout, numKeyValueHeads, sparseMode, innerPrecise));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PromptFlashAttention LaunchAicore failed.");
        return nullptr;
    }

    return attentionOutOut;
}
}