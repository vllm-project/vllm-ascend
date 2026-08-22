/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fallback/fallback_comm.h"
#include "fallback/fallback.h"
#include "log/log.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

using namespace ge;
using namespace gert;
static const size_t QUERY_INDEX = 0;
static const size_t KEY_INDEX = 1;
static const size_t VALUE_INDEX = 2;
static const int64_t INNER_PRECISE_MIN = 0;
static const int64_t INNER_PRECISE_MAX = 4;

static graphStatus PromptHostExecuteFunc(OpExecuteContext* host_api_ctx)
{
  OP_CHECK_IF(host_api_ctx == nullptr,
    OP_LOGE("aclnnfallback", "host_api_ctx is null"), return GRAPH_FAILED);

  auto query = host_api_ctx->GetInputTensor(QUERY_INDEX);
  OP_CHECK_IF(query == nullptr,
    OP_LOGE("aclnnfallback", "query is null"), return GRAPH_FAILED);

  auto key = host_api_ctx->GetInputTensor(KEY_INDEX);
  OP_CHECK_IF(key == nullptr,
    OP_LOGE("aclnnfallback", "key is null"), return GRAPH_FAILED);

  auto value = host_api_ctx->GetInputTensor(VALUE_INDEX);
  OP_CHECK_IF(value == nullptr,
    OP_LOGE("aclnnfallback", "value is null"), return GRAPH_FAILED);

  auto output = host_api_ctx->GetOutputTensor(0);
  OP_CHECK_IF(output == nullptr,
    OP_LOGE("aclnnfallback", "output is null"), return GRAPH_FAILED);

  auto pseShiftGe = host_api_ctx->GetOptionalInputTensor(3);
  auto attenMaskGe = host_api_ctx->GetOptionalInputTensor(4);
  auto actualSeqLengthsGe = host_api_ctx->GetOptionalInputTensor(5);

  auto actualSeqLengthsGeKv = host_api_ctx->GetOptionalInputTensor(6);
  auto deq_scale1 = host_api_ctx->GetOptionalInputTensor(7);
  auto quant_scale1 = host_api_ctx->GetOptionalInputTensor(8);
  auto deq_scale2 = host_api_ctx->GetOptionalInputTensor(9);
  auto quant_scale2 = host_api_ctx->GetOptionalInputTensor(10);
  auto quant_offset2 = host_api_ctx->GetOptionalInputTensor(11);

  std::vector<int64_t> actSeqArray;
  if (actualSeqLengthsGe != nullptr) {
    const int64_t* actSeqData = actualSeqLengthsGe->GetData<int64_t>();
    OP_CHECK_IF(actSeqData == nullptr,  
      OP_LOGE("aclnnfallback", 
             "When processing attention padding masking, actSeqData is a required parameter (stores valid sequence lengths excluding padding), must not be null, but a null was actually passed."), 
      return GRAPH_FAILED);
    
    const size_t len = static_cast<size_t>(actualSeqLengthsGe->GetShapeSize());
    for (size_t i = 0; i < len; i++) {
      actSeqArray.push_back(actSeqData[i]);
    }
  }

  std::vector<int64_t> actSeqArrayKv;
  if (actualSeqLengthsGeKv != nullptr) {
    const int64_t* actSeqData = actualSeqLengthsGeKv->GetData<int64_t>();
    const size_t len = static_cast<size_t>(actualSeqLengthsGeKv->GetShapeSize());
    for (size_t i = 0; i < len; i++) {
      actSeqArrayKv.push_back(actSeqData[i]);
    }
  }

  auto attrs = host_api_ctx->GetAttrs();
  const uint32_t* getNumHeads = attrs->GetAttrPointer<uint32_t>(0);
  const float* scaleValue = attrs->GetAttrPointer<float>(1);
  const int64_t* getPreTokens = attrs->GetAttrPointer<int64_t>(2);
  const int64_t* getNextTokens = attrs->GetAttrPointer<int64_t>(3);
  const char* layout = attrs->GetAttrPointer<char>(4);
  const uint32_t* getKVHeadNum = attrs->GetAttrPointer<uint32_t>(5);
  const uint32_t* getSparseMode = attrs->GetAttrPointer<uint32_t>(6);
  const uint32_t* getInnerPrecise = attrs->GetAttrPointer<uint32_t>(7);

  int64_t numHeads = *getNumHeads;
  double dScaleValue = *scaleValue;
  int64_t preTokens = *getPreTokens;
  int64_t nextTokens = *getNextTokens;
  int64_t kvHeadNum = *getKVHeadNum;
  int64_t sparseMode = *getSparseMode;
  int64_t innerPrecise = *getInnerPrecise;

  if (innerPrecise < INNER_PRECISE_MIN || innerPrecise > INNER_PRECISE_MAX) {   // innerPrecise=2,3 corresponds to high-performance or high-precision mode with invalid rows.
    // 4 is for approximate computation. only for 310p
    OP_LOGE("aclnnfallback", "invalid innerPrecise(%ld). Only support 0~4 now.", innerPrecise);
    return GRAPH_FAILED;
  }
  OP_LOGD("aclnnFallback",
          "PromptFlashAttentionV3 fallback begin, numHeads = %ld, dScaleValue = %lf",
          numHeads, dScaleValue);
  OP_LOGD("aclnnFallback",
          "preTokens = %ld, nextTokens = %ld, kvHeadNum = %ld, sparseMode = %ld, innerPrecise = %ld",
          preTokens, nextTokens, kvHeadNum, sparseMode, innerPrecise);

  if (sparseMode >= 10 && sparseMode <= 14) {  // 10: min  14: max 
    innerPrecise = 0;
    sparseMode -= 10;  // subtract 10 to modify sparseMode
    OP_LOGD("aclnnFallback",
            "because sparseMode in range [10, 14], after modification, sparseMode = %ld, innerPrecise = %ld.",
            sparseMode, innerPrecise);
  }

  auto api_ret = EXEC_OPAPI_CMD(aclnnPromptFlashAttentionV3, query, key, value, pseShiftGe, attenMaskGe, actSeqArray,
                                actSeqArrayKv, deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2,
                                numHeads, dScaleValue, preTokens, nextTokens, layout, kvHeadNum, sparseMode,
                                innerPrecise, output);

  OP_CHECK_IF(api_ret != GRAPH_SUCCESS,
    OP_LOGE("aclnnfallback", "api_ret faild:%u", api_ret), return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

IMPL_OP(PromptFlashAttention).OpExecuteFunc(PromptHostExecuteFunc).HostInputs({5, 6});
}  // namespace fallback

#ifdef __cplusplus
}
#endif
