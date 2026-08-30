/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_infer_attention_score_v2_sink.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include <acl/acl.h>
using namespace op;

namespace FusedInferAttentionScoreV2Sinkl0op {

OP_TYPE_REGISTER(FusedInferAttentionScoreV2Sink);

const FusedInferAttentionScoreV2SinkOutputs FusedInferAttentionScoreV2Sink(
    const aclTensor *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor *pseShift,
    const aclTensor *attenMask,
    const aclTensor *actualSeqLengths,
    const aclTensor *actualSeqLengthsKv,
    const aclTensor *deqScale1,
    const aclTensor *quantScale1,
    const aclTensor *deqScale2,
    const aclTensor *quantScale2,
    const aclTensor *quantOffset2,
    const aclTensor *antiquantScale,
    const aclTensor *antiquantOffset,
    const aclTensor *blockTable,
    const aclTensor *queryPaddingSize,
    const aclTensor *kvPaddingSize,
    const aclTensor *keyAntiquantScale,
    const aclTensor *keyAntiquantOffset,
    const aclTensor *valueAntiquantScale,
    const aclTensor *valueAntiquantOffset,
    const aclTensor *queryRope,
    const aclTensor *keyRope,
    const aclTensor *keyRopeAntiquantScale,
    const aclTensor *dequantScaleQuery,
    const aclTensor *metaData,
    const aclTensor *learnableSink,
    const aclTensor *keySinkOptional,
    const aclTensor *keyRopeSinkOptional,
    const aclTensor *valueSinkOptional,
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
    int64_t sinkNumber,
    bool batchInvariant,
    bool softmaxMaxSumFlag,
    int64_t outType,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLse,
    const aclTensor *softmaxMax,
    const aclTensor *softmaxSum,
    aclOpExecutor *executor)
{
    L0_DFX(FusedInferAttentionScoreV2Sink, query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKv,
            deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,
            blockTable, queryPaddingSize, kvPaddingSize, keyAntiquantScale, keyAntiquantOffset,
            valueAntiquantScale, valueAntiquantOffset,
            queryRope, keyRope, keyRopeAntiquantScale, dequantScaleQuery, metaData, learnableSink,
            keySinkOptional, keyRopeSinkOptional, valueSinkOptional,
            numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads,
            sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode,
            queryQuantMode, pseType, sinkNumber, batchInvariant, softmaxMaxSumFlag, outType);
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedInferAttentionScoreV2Sink: executor is nullptr.");
        return {nullptr, nullptr, nullptr, nullptr};
    }

    // 分配输出 tensor
    auto attentionOutOut = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxLseOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxMaxOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxSumOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);

    // 形状推导
    auto ret = INFER_SHAPE(FusedInferAttentionScoreV2Sink,
        OP_INPUT(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKv,
                deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
                antiquantScale, antiquantOffset, blockTable, queryPaddingSize, kvPaddingSize,
                keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
                queryRope, keyRope, keyRopeAntiquantScale, dequantScaleQuery, metaData, nullptr,
                keySinkOptional, keyRopeSinkOptional, valueSinkOptional),
        OP_OUTPUT(attentionOutOut, softmaxLseOut, softmaxMaxOut, softmaxSumOut),
        OP_ATTR(numHeads, static_cast<float>(scaleValue), static_cast<int32_t>(preTokens),
                static_cast<int32_t>(nextTokens), inputLayout, numKeyValueHeads,
                static_cast<int32_t>(sparseMode), innerPrecise, blockSize,
                static_cast<int32_t>(antiquantMode), softmaxLseFlag,
                static_cast<int32_t>(keyAntiquantMode), static_cast<int32_t>(valueAntiquantMode),
                static_cast<int32_t>(queryQuantMode), 0, sinkNumber, batchInvariant, softmaxMaxSumFlag, 0));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedInferAttentionScoreV2Sink InferShape failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }

    // 添加到启动列表
    ret = ADD_TO_LAUNCHER_LIST_AICORE(FusedInferAttentionScoreV2Sink,
        OP_INPUT(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKv,
                deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
                antiquantScale, antiquantOffset, blockTable, queryPaddingSize, kvPaddingSize,
                keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
                queryRope, keyRope, keyRopeAntiquantScale, dequantScaleQuery, metaData, nullptr,
                keySinkOptional, keyRopeSinkOptional, valueSinkOptional),
        OP_OUTPUT(attentionOutOut, softmaxLseOut, softmaxMaxOut, softmaxSumOut),
        OP_ATTR(numHeads, static_cast<float>(scaleValue), static_cast<int32_t>(preTokens),
                static_cast<int32_t>(nextTokens), inputLayout, numKeyValueHeads,
                static_cast<int32_t>(sparseMode), innerPrecise, blockSize,
                static_cast<int32_t>(antiquantMode), softmaxLseFlag,
                static_cast<int32_t>(keyAntiquantMode), static_cast<int32_t>(valueAntiquantMode),
                static_cast<int32_t>(queryQuantMode), 0, sinkNumber, batchInvariant, softmaxMaxSumFlag, 0));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedInferAttentionScoreV2Sink LaunchAicore failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return {attentionOutOut, softmaxLseOut, softmaxMaxOut, softmaxSumOut};
}

}