/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback/fallback_comm_2stages.h"
#include "fallback/fallback_2stages.h"
#include "log/log.h"
#include "log/error_code.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;
static const size_t QUERY_INDEX = 0;
static const size_t KEY_INDEX = 1;
static const size_t VALUE_INDEX = 2;
static const size_t PSE_SHIFT_INDEX = 3;
static const size_t ATTEN_MASK_INDEX = 4;
static const size_t ACTUAL_SEQ_LEN_INDEX = 5;
static const size_t DEQUANT_SCALE_1_INDEX = 6;
static const size_t QUANT_SCALE_1_INDEX = 7;
static const size_t DEQUANT_SCALE_2_INDEX = 8;
static const size_t QUANT_SCALE_2_INDEX = 9;
static const size_t QUANT_OFFSET_2_INDEX = 10;
static const size_t ANTIQUANT_SCALE_INDEX = 11;
static const size_t ANTIQUANT_OFFSET_INDEX = 12;
static const size_t BLOCK_TABLE_INDEX = 13;
static const size_t KV_PADDING_SIZE_INDEX = 14;
static const size_t NUM_HEADS_INDEX = 0;
static const size_t SCALE_VALUE_INDEX = 1;
static const size_t LAYOUT_INDEX = 2;
static const size_t KV_HEAD_NUM_INDEX = 3;
static const size_t BLOCK_SIZE_INDEX = 4;
static const size_t INNER_PRECISE_INDEX = 5;

graphStatus IncreHostExecuteFunc(OpExecutePrepareContext *host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr, OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

    auto query = host_api_ctx->GetInputTensor(QUERY_INDEX);
    OP_CHECK_IF(query == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "query is null"), return GRAPH_FAILED);

    auto key = host_api_ctx->GetDynamicInputTensor(KEY_INDEX, 0);
    OP_CHECK_IF(key == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "key is null"), return GRAPH_FAILED);

    auto value = host_api_ctx->GetDynamicInputTensor(VALUE_INDEX, 0);
    OP_CHECK_IF(value == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "value is null"), return GRAPH_FAILED);

    auto output = host_api_ctx->GetOutputTensor(0);
    OP_CHECK_IF(output == nullptr, OP_LOGE(host_api_ctx->GetNodeName(), "output is null"), return GRAPH_FAILED);

    std::vector<const gert::Tensor *> ge_tenserListKey;
    ge_tenserListKey.push_back(key);

    std::vector<const gert::Tensor *> ge_tenserListValue;
    ge_tenserListValue.push_back(value);

    auto pseShiftGe = host_api_ctx->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    auto attenMaskGe = host_api_ctx->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    auto actualSeqLengthsGe = host_api_ctx->GetOptionalInputTensor(ACTUAL_SEQ_LEN_INDEX);
    auto dequantScale1Ge = host_api_ctx->GetOptionalInputTensor(DEQUANT_SCALE_1_INDEX);
    auto quantScale1Ge = host_api_ctx->GetOptionalInputTensor(QUANT_SCALE_1_INDEX);
    auto dequantScale2Ge = host_api_ctx->GetOptionalInputTensor(DEQUANT_SCALE_2_INDEX);
    auto quantScale2Ge = host_api_ctx->GetOptionalInputTensor(QUANT_SCALE_2_INDEX);
    auto quantOffset2Ge = host_api_ctx->GetOptionalInputTensor(QUANT_OFFSET_2_INDEX);
    auto antiquantScaleGe = host_api_ctx->GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    auto antiquantOffsetGe = host_api_ctx->GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    auto blocktableGe = host_api_ctx->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    auto kvPaddingSizeGe = host_api_ctx->GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);

    std::vector<int64_t> actSeqArray;
    if (actualSeqLengthsGe != nullptr) {
        const int64_t *actSeqData = actualSeqLengthsGe->GetData<int64_t>();
        const size_t len = static_cast<size_t>(actualSeqLengthsGe->GetShapeSize());
        for (size_t i = 0; i < len; i++) {
            actSeqArray.push_back(actSeqData[i]);
        }
    }

    auto attrs = host_api_ctx->GetAttrs();
    const uint32_t *num_heads = attrs->GetAttrPointer<uint32_t>(NUM_HEADS_INDEX);
    const float *scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_INDEX);
    const char *layout = attrs->GetAttrPointer<char>(LAYOUT_INDEX);
    const uint32_t *kvHeadNum = attrs->GetAttrPointer<uint32_t>(KV_HEAD_NUM_INDEX);
    const uint32_t *blockSize = attrs->GetAttrPointer<uint32_t>(BLOCK_SIZE_INDEX);
    OP_CHECK_IF(blockSize == nullptr,  
        OP_LOGE(host_api_ctx->GetNodeName(),  
            "In block-wise attention computation, blockSize is a required parameter (controls matrix block dimension), must not be null, but a null was actually passed."), 
        return GRAPH_FAILED);

    const uint32_t *innerPrecise = attrs->GetAttrPointer<uint32_t>(INNER_PRECISE_INDEX);
    OP_CHECK_IF(innerPrecise == nullptr, 
        OP_LOGE(host_api_ctx->GetNodeName(), 
            "In internal operations (e.g., attention matrix multiplication, softmax), innerPrecise is a required parameter (specifies precision), must not be null, but a null was actually passed."), 
        return GRAPH_FAILED);

    OP_CHECK_IF(scaleValue == nullptr,  
        OP_LOGE(host_api_ctx->GetNodeName(),  
            "When computing attention weights, scaleValue is a required parameter (normalizes QK^T to prevent softmax gradient vanishing), must not be null, but a null was actually passed."), 
        return GRAPH_FAILED);

    double dScaleValue = *scaleValue;

    // execute opapi
    auto api_ret =
        EXEC_OPAPI_PREPARE_CMD(aclnnIncreFlashAttentionV4, query, ge_tenserListKey, ge_tenserListValue, pseShiftGe, attenMaskGe,
                       actSeqArray, dequantScale1Ge, quantScale1Ge, dequantScale2Ge, quantScale2Ge, quantOffset2Ge,
                       antiquantScaleGe, antiquantOffsetGe, blocktableGe, kvPaddingSizeGe, *num_heads, dScaleValue,
                       layout, *kvHeadNum, *blockSize, *innerPrecise, output);
    OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE(host_api_ctx->GetNodeName(), "api_ret faild:%u", api_ret), return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(IncreFlashAttention).Op2StageExecuteFuncs(IncreHostExecuteFunc, ExecuteOpLaunch).HostInputs({5});

} // namespace fallback

#ifdef __cplusplus
}
#endif