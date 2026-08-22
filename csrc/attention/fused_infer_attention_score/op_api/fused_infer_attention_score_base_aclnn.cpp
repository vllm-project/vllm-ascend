/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fused_infer_attention_score.h"

#include "opdev/make_op_executor.h"
#include "opdev/common_types.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedInferAttentionScore);

namespace {
const aclTensor *ConvertIntArrayToTensor(const aclIntArray *array, aclOpExecutor *executor)
{
    if (array == nullptr) {
        return nullptr;
    }

    auto tensor = executor->ConvertToTensor(array, DataType::DT_INT64);
    if (tensor == nullptr) {
        return nullptr;
    }
    const_cast<aclTensor *>(tensor)->SetStorageFormat(Format::FORMAT_ND);
    const_cast<aclTensor *>(tensor)->SetViewFormat(Format::FORMAT_ND);
    const_cast<aclTensor *>(tensor)->SetOriginalFormat(Format::FORMAT_ND);
    return tensor;
}
} // namespace

std::tuple<const aclTensor *, const aclTensor *> FusedInferAttentionScore(
    const aclTensor *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional,
    const aclTensor *deqScale2Optional,
    const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional,
    const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional,
    const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional,
    const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional,
    const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional,
    const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    const aclTensor *keyRopeAntiquantScaleOptional,
    const aclTensor *dequantScaleQueryOptional,
    const aclTensor *learnableSinkOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    int64_t numHeads,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    const char *inputLayout,
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    int64_t innerPrecise,
    int64_t blockSize,
    int64_t antiquantMode,
    bool softmaxLseFlag,
    int64_t keyAntiquantMode,
    int64_t valueAntiquantMode,
    int64_t queryQuantMode,
    int64_t pseType,
    int64_t outDtype,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLse,
    aclOpExecutor *executor)
{
    L0_DFX(FusedInferAttentionScore, query, key, value, pseShiftOptional, attenMaskOptional,
           actualSeqLengthsOptional, actualSeqLengthsKvOptional, deqScale1Optional, quantScale1Optional,
           deqScale2Optional, quantScale2Optional, quantOffset2Optional, antiquantScaleOptional,
           antiquantOffsetOptional, blockTableOptional, queryPaddingSizeOptional, kvPaddingSizeOptional,
           keyAntiquantScaleOptional, keyAntiquantOffsetOptional, valueAntiquantScaleOptional,
           valueAntiquantOffsetOptional, keySharedPrefixOptional, valueSharedPrefixOptional,
           actualSharedPrefixLenOptional, queryRopeOptional, keyRopeOptional, keyRopeAntiquantScaleOptional,
           dequantScaleQueryOptional, learnableSinkOptional, qStartIdxOptional, kvStartIdxOptional,
           numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads, sparseMode, innerPrecise,
           blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, queryQuantMode, pseType,
           outDtype);

    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedInferAttentionScore: executor is nullptr.");
        return {nullptr, nullptr};
    }
    if (attentionOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "FusedInferAttentionScore: attentionOut is nullptr.");
        return {nullptr, nullptr};
    }
    if (softmaxLse == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "FusedInferAttentionScore: softmaxLse is nullptr.");
        return {nullptr, nullptr};
    }

    const aclTensor *actualSeqLengthsTensor = ConvertIntArrayToTensor(actualSeqLengthsOptional, executor);
    const aclTensor *actualSeqLengthsKvTensor = ConvertIntArrayToTensor(actualSeqLengthsKvOptional, executor);
    const aclTensor *actualSharedPrefixLenTensor = ConvertIntArrayToTensor(actualSharedPrefixLenOptional, executor);
    const aclTensor *qStartIdxTensor = ConvertIntArrayToTensor(qStartIdxOptional, executor);
    const aclTensor *kvStartIdxTensor = ConvertIntArrayToTensor(kvStartIdxOptional, executor);
    if ((actualSeqLengthsOptional != nullptr && actualSeqLengthsTensor == nullptr) ||
        (actualSeqLengthsKvOptional != nullptr && actualSeqLengthsKvTensor == nullptr) ||
        (actualSharedPrefixLenOptional != nullptr && actualSharedPrefixLenTensor == nullptr) ||
        (qStartIdxOptional != nullptr && qStartIdxTensor == nullptr) ||
        (kvStartIdxOptional != nullptr && kvStartIdxTensor == nullptr)) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FusedInferAttentionScore: convert IntArray to Tensor failed.");
        return {nullptr, nullptr};
    }

    auto attentionOutOut = executor->AllocTensor(attentionOut->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    if (attentionOutOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FusedInferAttentionScore: alloc attentionOutOut failed.");
        return {nullptr, nullptr};
    }
    auto softmaxLseOut = executor->AllocTensor(softmaxLse->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    if (softmaxLseOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FusedInferAttentionScore: alloc softmaxLseOut failed.");
        return {nullptr, nullptr};
    }
    auto ret = INFER_SHAPE(FusedInferAttentionScore,
        OP_INPUT(query, key, value, pseShiftOptional, attenMaskOptional, actualSeqLengthsTensor,
                 actualSeqLengthsKvTensor, deqScale1Optional, quantScale1Optional, deqScale2Optional,
                 quantScale2Optional, quantOffset2Optional, antiquantScaleOptional, antiquantOffsetOptional,
                 blockTableOptional, queryPaddingSizeOptional, kvPaddingSizeOptional, keyAntiquantScaleOptional,
                 keyAntiquantOffsetOptional, valueAntiquantScaleOptional, valueAntiquantOffsetOptional,
                 keySharedPrefixOptional, valueSharedPrefixOptional, actualSharedPrefixLenTensor,
                 queryRopeOptional, keyRopeOptional, keyRopeAntiquantScaleOptional, dequantScaleQueryOptional,
                 learnableSinkOptional, qStartIdxTensor, kvStartIdxTensor),
        OP_OUTPUT(attentionOutOut, softmaxLseOut),
        OP_ATTR(numHeads, static_cast<float>(scaleValue), preTokens, nextTokens, inputLayout, numKeyValueHeads,
                sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode,
                valueAntiquantMode, queryQuantMode, pseType, outDtype));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedInferAttentionScore InferShape failed.");
        return {nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(FusedInferAttentionScore,
        OP_INPUT(query, key, value, pseShiftOptional, attenMaskOptional, actualSeqLengthsTensor,
                 actualSeqLengthsKvTensor, deqScale1Optional, quantScale1Optional, deqScale2Optional,
                 quantScale2Optional, quantOffset2Optional, antiquantScaleOptional, antiquantOffsetOptional,
                 blockTableOptional, queryPaddingSizeOptional, kvPaddingSizeOptional, keyAntiquantScaleOptional,
                 keyAntiquantOffsetOptional, valueAntiquantScaleOptional, valueAntiquantOffsetOptional,
                 keySharedPrefixOptional, valueSharedPrefixOptional, actualSharedPrefixLenTensor,
                 queryRopeOptional, keyRopeOptional, keyRopeAntiquantScaleOptional, dequantScaleQueryOptional,
                 learnableSinkOptional, qStartIdxTensor, kvStartIdxTensor),
        OP_OUTPUT(attentionOutOut, softmaxLseOut),
        OP_ATTR(numHeads, static_cast<float>(scaleValue), preTokens, nextTokens, inputLayout, numKeyValueHeads,
                sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode,
                valueAntiquantMode, queryQuantMode, pseType, outDtype));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedInferAttentionScore LaunchAicore failed.");
        return {nullptr, nullptr};
    }

    return {attentionOutOut, softmaxLseOut};
}

} // namespace l0op
