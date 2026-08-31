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
#include "aclnn_fused_infer_attention_score_v2_sink_v3.h"

#include <cstring>

#include "aclnn/aclnn_base.h"
#include "aclnn/opdev/op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

#include "graph/types.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

const uint64_t INT4_NUMS_IN_INT32 = 8;

static void TensorPreProcessV3(const aclTensorList *&tensorListKey, const aclTensorList *&tensorListValue)
{
    if (tensorListKey == nullptr) {
        OP_LOGD("TensorListKey is nullptr,TensorPreProcessV3 exit.");
        return;
    }
    if (tensorListValue == nullptr) {
        OP_LOGD("tensorListValue is nullptr,TensorPreProcessV3 exit.");
        return;
    }
    if ((*tensorListKey)[0]->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kv's from OriginalShape is completed.");
        return;
    }
    if ((*tensorListValue)[0]->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kv's from OriginalShape is completed.");
        return;
    }
    auto tempKey = const_cast<aclTensorList *>(tensorListKey);
    for (uint64_t i = 0; i < tempKey->Size(); i++) {
        if ((*tempKey)[i] != nullptr) {
            op::Shape viewShape = (*tempKey)[i]->GetViewShape();
            auto viewShapeDim = viewShape.GetDimNum();
            if (viewShapeDim >= 1) {
                viewShape[viewShapeDim - 1] = viewShape[viewShapeDim - 1] * INT4_NUMS_IN_INT32;
            }
            (*tempKey)[i]->SetViewShape(viewShape);
            (*tempKey)[i]->SetDataType(DataType::DT_INT4);
        }
    }

    auto tempValue = const_cast<aclTensorList *>(tensorListValue);
    for (uint64_t i = 0; i < tempValue->Size(); i++) {
        if ((*tempValue)[i] != nullptr) {
            op::Shape viewShape = (*tempValue)[i]->GetViewShape();
            auto viewShapeDim = viewShape.GetDimNum();
            if (viewShapeDim >= 1) {
                viewShape[viewShapeDim - 1] = viewShape[viewShapeDim - 1] * INT4_NUMS_IN_INT32;
            }
            (*tempValue)[i]->SetViewShape(viewShape);
            (*tempValue)[i]->SetDataType(DataType::DT_INT4);
        }
    }

    OP_LOGD("The conversion of kv from int32 to int4 is completed.");
}

// 创建 shape=[0]、假地址的 ACL_FLOAT 占位 tensor（softmaxLseFlag=false 时为 lse/max/sum 三路占位）
static aclTensor *FusedInferAttentionScoreV2SinkCreateEmptyTensorV3()
{
    std::vector<int64_t> shape = {0};
    int64_t addr = 0xff;
    return aclCreateTensor(shape.data(),
                           shape.size(),
                           aclDataType::ACL_FLOAT,
                           shape.data(),
                           0,
                           ACL_FORMAT_ND,
                           shape.data(),
                           shape.size(),
                           static_cast<void *>(&addr));
}

// V3 三路 softmax 输出占位处理：
//   softmaxLseFlag=false    → lse 建 shape=[0] 假地址 tempTensor 占位；
//   softmaxMaxSumFlag=false → max/sum 建 shape=[0] 假地址 tempTensor 占位；
//   各自独立控制。
// tempLse/Max/Sum 仅在对应 flag=false 时非空，调用方需在末尾 aclDestroyTensor 释放。
static void FusedInferAttentionScoreV2SinkProcessSoftmaxLseV3(bool softmaxLseFlag,
                                                       bool softmaxMaxSumFlag,
                                                       const aclTensor *softmaxLse,
                                                       const aclTensor *softmaxMax,
                                                       const aclTensor *softmaxSum,
                                                       aclTensor *&tempLse,
                                                       aclTensor *&tempMax,
                                                       aclTensor *&tempSum,
                                                       const aclTensor *&placeLse,
                                                       const aclTensor *&placeMax,
                                                       const aclTensor *&placeSum)
{
    if (softmaxLseFlag == false) {
        tempLse = FusedInferAttentionScoreV2SinkCreateEmptyTensorV3();
        placeLse = tempLse;
    } else {
        placeLse = softmaxLse;
    }
    if (softmaxMaxSumFlag == false) {
        tempMax = FusedInferAttentionScoreV2SinkCreateEmptyTensorV3();
        tempSum = FusedInferAttentionScoreV2SinkCreateEmptyTensorV3();
        placeMax = tempMax;
        placeSum = tempSum;
    } else {
        placeMax = softmaxMax;
        placeSum = softmaxSum;
    }
}

// CreateView 保留非连续 tensor 的 stride 信息
static const aclTensor *CalcNoContiguous(const aclTensor *input, aclOpExecutor *executor)
{
    if (input == nullptr) {
        return input;
    }
    aclTensor *newInput = executor->CreateView(input,
                                               input->GetViewShape(),
                                               input->GetStorageShape(),
                                               input->GetViewStrides(),
                                               input->GetViewOffset());
    CHECK_RET(newInput != nullptr, nullptr);
    return newInput;
}

static const aclTensor *ProcessTensorContiguous(const aclTensor *tensor,
                                                aclOpExecutor *executor,
                                                const char *tensorName)
{
    if (tensor == nullptr) {
        return nullptr;
    }
    if (!IsContiguous(tensor)) {
        return CalcNoContiguous(tensor, executor);
    } else {
        tensor = l0op::Contiguous(tensor, executor);
    }
    return tensor;
}

static aclnnStatus ContiguousInput(const aclTensor *&query,
                                   const aclTensor *&attenMaskOptional,
                                   const aclTensor *&blockTableOptional,
                                   const aclTensor *&queryRopeOptional,
                                   const aclTensor *&keySinkOptional,
                                   const aclTensor *&keyRopeSinkOptional,
                                   const aclTensor *&valueSinkOptional,
                                   aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (attenMaskOptional) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (blockTableOptional) {
        blockTableOptional = l0op::Contiguous(blockTableOptional, executor);
        CHECK_RET(blockTableOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (queryRopeOptional) {
        queryRopeOptional = l0op::Contiguous(queryRopeOptional, executor);
        CHECK_RET(queryRopeOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (keySinkOptional) {
        keySinkOptional = l0op::Contiguous(keySinkOptional, executor);
        CHECK_RET(keySinkOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (keyRopeSinkOptional) {
        keyRopeSinkOptional = l0op::Contiguous(keyRopeSinkOptional, executor);
        CHECK_RET(keyRopeSinkOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (valueSinkOptional) {
        valueSinkOptional = l0op::Contiguous(valueSinkOptional, executor);
        CHECK_RET(valueSinkOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static const aclTensorList *ProcessTensorListContiguous(const aclTensorList *tensorList, aclOpExecutor *executor,
                                                        const char *tensorListName)
{
    if (tensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> processedTensorsTmp;
    uint64_t size = tensorList->Size();

    for (uint64_t i = 0; i < size; i++) {
        const aclTensor *tensor = (*tensorList)[i];
        if (tensor == nullptr) {
            processedTensorsTmp.push_back(nullptr);
        }
        const aclTensor *processedTensor = ProcessTensorContiguous(tensor, executor, tensorListName);
        CHECK_RET(processedTensor != nullptr, nullptr);
        processedTensorsTmp.push_back(processedTensor);
    }

    aclTensorList *processedTensors = executor->AllocTensorList(processedTensorsTmp.data(), processedTensorsTmp.size());
    CHECK_RET(processedTensors != nullptr, nullptr);
    return processedTensors;
}

extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void *executor, const size_t irIndex);

aclnnStatus aclnnFusedInferAttentionScoreV2SinkV3GetMaxWorkspaceSize(const aclTensor *query,
                                                                     const aclTensorList *tensorListKey,
                                                                     const aclTensorList *tensorListValue,
                                                                     const aclTensor *pseShiftOptional,
                                                                     const aclTensor *attenMaskOptional,
                                                                     const aclTensor *actualSeqLengthsOptional,
                                                                     const aclTensor *actualSeqLengthsKvOptional,
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
                                                                     const aclTensor *queryRopeOptional,
                                                                     const aclTensor *keyRopeOptional,
                                                                     const aclTensor *keyRopeAntiquantScaleOptional,
                                                                     const aclTensor *dequantScaleQueryOptional,
                                                                     const aclTensor *metaDataOptional,
                                                                     const aclTensor *keySinkOptional,
                                                                     const aclTensor *keyRopeSinkOptional,
                                                                     const aclTensor *valueSinkOptional,
                                                                     int64_t numHeads,
                                                                     double scaleValue,
                                                                     int64_t preTokens,
                                                                     int64_t nextTokens,
                                                                     char *inputLayout,
                                                                     int64_t numKeyValueHeads,
                                                                     int64_t sparseMode,
                                                                     int64_t innerPrecise,
                                                                     int64_t blockSize,
                                                                     int64_t antiquantMode,
                                                                     bool softmaxLseFlag,
                                                                     int64_t keyAntiquantMode,
                                                                     int64_t valueAntiquantMode,
                                                                     int64_t queryQuantMode,
                                                                     int64_t sinkNumber,
                                                                     bool batchInvariant,
                                                                     bool softmaxMaxSumFlag,
                                                                     const aclTensor *attentionOut,
                                                                     const aclTensor *softmaxLse,
                                                                     const aclTensor *softmaxMax,
                                                                     const aclTensor *softmaxSum,
                                                                     uint64_t *workspaceSize,
                                                                     aclOpExecutor **executor)
{
    OP_LOGD("start aclnnFusedInferAttentionScoreV2SinkV3GetMaxWorkspaceSize");
    TensorPreProcessV3(tensorListKey, tensorListValue);

    auto ret = aclnnFusedInferAttentionScoreV2SinkV3GetWorkspaceSize(query,
                                                                     tensorListKey,
                                                                     tensorListValue,
                                                                     pseShiftOptional,
                                                                     attenMaskOptional,
                                                                     actualSeqLengthsOptional,
                                                                     actualSeqLengthsKvOptional,
                                                                     deqScale1Optional,
                                                                     quantScale1Optional,
                                                                     deqScale2Optional,
                                                                     quantScale2Optional,
                                                                     quantOffset2Optional,
                                                                     antiquantScaleOptional,
                                                                     antiquantOffsetOptional,
                                                                     blockTableOptional,
                                                                     queryPaddingSizeOptional,
                                                                     kvPaddingSizeOptional,
                                                                     keyAntiquantScaleOptional,
                                                                     keyAntiquantOffsetOptional,
                                                                     valueAntiquantScaleOptional,
                                                                     valueAntiquantOffsetOptional,
                                                                     queryRopeOptional,
                                                                     keyRopeOptional,
                                                                     keyRopeAntiquantScaleOptional,
                                                                     dequantScaleQueryOptional,
                                                                     metaDataOptional,
                                                                     keySinkOptional,
                                                                     keyRopeSinkOptional,
                                                                     valueSinkOptional,
                                                                     numHeads,
                                                                     scaleValue,
                                                                     preTokens,
                                                                     nextTokens,
                                                                     inputLayout,
                                                                     numKeyValueHeads,
                                                                     sparseMode,
                                                                     innerPrecise,
                                                                     blockSize,
                                                                     antiquantMode,
                                                                     softmaxLseFlag,
                                                                     keyAntiquantMode,
                                                                     valueAntiquantMode,
                                                                     queryQuantMode,
                                                                     sinkNumber,
                                                                     batchInvariant,
                                                                     softmaxMaxSumFlag,
                                                                     attentionOut,
                                                                     softmaxLse,
                                                                     softmaxMax,
                                                                     softmaxSum,
                                                                     workspaceSize,
                                                                     executor);
    return ret;
}

aclnnStatus aclnnFusedInferAttentionScoreV2SinkV3GetWorkspaceSize(const aclTensor *query,
                                                                  const aclTensorList *key,
                                                                  const aclTensorList *value,
                                                                  const aclTensor *pseShiftOptional,
                                                                  const aclTensor *attenMaskOptional,
                                                                  const aclTensor *actualSeqLengthsOptional,
                                                                  const aclTensor *actualSeqLengthsKvOptional,
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
                                                                  const aclTensor *queryRopeOptional,
                                                                  const aclTensor *keyRopeOptional,
                                                                  const aclTensor *keyRopeAntiquantScaleOptional,
                                                                  const aclTensor *dequantScaleQueryOptional,
                                                                  const aclTensor *metaDataOptional,
                                                                  const aclTensor *keySinkOptional,
                                                                  const aclTensor *keyRopeSinkOptional,
                                                                  const aclTensor *valueSinkOptional,
                                                                  int64_t numHeads,
                                                                  double scaleValue,
                                                                  int64_t preTokens,
                                                                  int64_t nextTokens,
                                                                  char *inputLayout,
                                                                  int64_t numKeyValueHeads,
                                                                  int64_t sparseMode,
                                                                  int64_t innerPrecise,
                                                                  int64_t blockSize,
                                                                  int64_t antiquantMode,
                                                                  bool softmaxLseFlag,
                                                                  int64_t keyAntiquantMode,
                                                                  int64_t valueAntiquantMode,
                                                                  int64_t queryQuantMode,
                                                                  int64_t sinkNumber,
                                                                  bool batchInvariant,
                                                                  bool softmaxMaxSumFlag,
                                                                  const aclTensor *attentionOut,
                                                                  const aclTensor *softmaxLse,
                                                                  const aclTensor *softmaxMax,
                                                                  const aclTensor *softmaxSum,
                                                                  uint64_t *workspaceSize,
                                                                  aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFusedInferAttentionScoreV2SinkV3,
            DFX_IN(query,
                   key,
                   value,
                   pseShiftOptional,
                   attenMaskOptional,
                   actualSeqLengthsOptional,
                   actualSeqLengthsKvOptional,
                   deqScale1Optional,
                   quantScale1Optional,
                   deqScale2Optional,
                   quantScale2Optional,
                   quantOffset2Optional,
                   antiquantScaleOptional,
                   antiquantOffsetOptional,
                   blockTableOptional,
                   queryPaddingSizeOptional,
                   kvPaddingSizeOptional,
                   keyAntiquantScaleOptional,
                   keyAntiquantOffsetOptional,
                   valueAntiquantScaleOptional,
                   valueAntiquantOffsetOptional,
                   queryRopeOptional,
                   keyRopeOptional,
                   keyRopeAntiquantScaleOptional,
                   dequantScaleQueryOptional,
                   metaDataOptional,
                   keySinkOptional,
                   keyRopeSinkOptional,
                   valueSinkOptional,
                   numHeads,
                   scaleValue,
                   preTokens,
                   nextTokens,
                   inputLayout,
                   numKeyValueHeads,
                   sparseMode,
                   innerPrecise,
                   blockSize,
                   antiquantMode,
                   softmaxLseFlag,
                   keyAntiquantMode,
                   valueAntiquantMode,
                   queryQuantMode,
                   sinkNumber,
                   batchInvariant,
                   softmaxMaxSumFlag),
            DFX_OUT(attentionOut, softmaxLse, softmaxMax, softmaxSum));
    const aclTensorList *tensorListKey = key;
    const aclTensorList *tensorListValue = value;
    TensorPreProcessV3(tensorListKey, tensorListValue);

    // 创建 executor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (attentionOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    CHECK_RET(ContiguousInput(query,
                              attenMaskOptional,
                              blockTableOptional,
                              queryRopeOptional,
                              keySinkOptional,
                              keyRopeSinkOptional,
                              valueSinkOptional,
                              l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    // 将K\V, k_rope 连续、非连续判断处理
    const aclTensorList *processKeyList = ProcessTensorListContiguous(tensorListKey, l0Executor, "key");
    const aclTensorList *processValueList = ProcessTensorListContiguous(tensorListValue, l0Executor, "value");
    const aclTensor *processKeyRope = ProcessTensorContiguous(keyRopeOptional, l0Executor, "keyRope");
    CHECK_RET(processKeyList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(processValueList != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // V3 softmax 输出占位处理：lse/max/sum 各自独立 flag 控制
    aclTensor *tempLse = nullptr;
    aclTensor *tempMax = nullptr;
    aclTensor *tempSum = nullptr;
    const aclTensor *placeLse = nullptr;
    const aclTensor *placeMax = nullptr;
    const aclTensor *placeSum = nullptr;
    FusedInferAttentionScoreV2SinkProcessSoftmaxLseV3(softmaxLseFlag, softmaxMaxSumFlag, softmaxLse,
                                                      softmaxMax, softmaxSum, tempLse, tempMax, tempSum,
                                                      placeLse, placeMax, placeSum);

    // 调用 L0 接口 - 透传 lse/max/sum 三路占位/真实输出
    auto l0Outputs = FusedInferAttentionScoreV2Sinkl0op::FusedInferAttentionScoreV2Sink(
        query,
        processKeyList,
        processValueList,
        pseShiftOptional,
        attenMaskOptional,
        actualSeqLengthsOptional,
        actualSeqLengthsKvOptional,
        deqScale1Optional,
        quantScale1Optional,
        deqScale2Optional,
        quantScale2Optional,
        quantOffset2Optional,
        antiquantScaleOptional,
        antiquantOffsetOptional,
        blockTableOptional,
        queryPaddingSizeOptional,
        kvPaddingSizeOptional,
        keyAntiquantScaleOptional,
        keyAntiquantOffsetOptional,
        valueAntiquantScaleOptional,
        valueAntiquantOffsetOptional,
        queryRopeOptional,
        processKeyRope,
        keyRopeAntiquantScaleOptional,
        dequantScaleQueryOptional,
        metaDataOptional,
        nullptr,
        keySinkOptional,
        keyRopeSinkOptional,
        valueSinkOptional,
        numHeads,
        scaleValue,
        preTokens,
        nextTokens,
        inputLayout,
        numKeyValueHeads,
        sparseMode,
        innerPrecise,
        blockSize,
        antiquantMode,
        softmaxLseFlag,
        keyAntiquantMode,
        valueAntiquantMode,
        queryQuantMode,
        0,
        sinkNumber,
        batchInvariant,
        softmaxMaxSumFlag,
        0,
        attentionOut,
        placeLse,
        placeMax,
        placeSum,
        l0Executor);
    CHECK_RET(l0Outputs.AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0Outputs.SoftmaxOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 使用 ViewCopy 复制输出（lse/max/sum 三路与 softmaxLse 对称）
    auto viewCopyAttentionOutResult = l0op::ViewCopy(l0Outputs.AttentionOutOut, attentionOut, l0Executor);
    auto viewCopySoftmaxLseOutResult = l0op::ViewCopy(l0Outputs.SoftmaxOutOut, placeLse, l0Executor);
    auto viewCopySoftmaxMaxOutResult = l0op::ViewCopy(l0Outputs.SoftmaxMaxOut, placeMax, l0Executor);
    auto viewCopySoftmaxSumOutResult = l0op::ViewCopy(l0Outputs.SoftmaxSumOut, placeSum, l0Executor);

    CHECK_RET(viewCopyAttentionOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(viewCopySoftmaxLseOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(viewCopySoftmaxMaxOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(viewCopySoftmaxSumOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (softmaxLseFlag == false) {
        aclDestroyTensor(tempLse);
    }
    if (softmaxMaxSumFlag == false) {
        aclDestroyTensor(tempMax);
        aclDestroyTensor(tempSum);
    }

    // 获取 workspace 大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedInferAttentionScoreV2SinkV3(void *workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor *executor,
                                                  const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedInferAttentionScoreV2SinkV3);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
