/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include "graph/types.h"
#include "aclnn_fused_infer_attention_score_v2_sink.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/format_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "fused_infer_attention_score_v2_sink.h"
#include "aclnn/opdev/op_executor.h"

#include "opdev/op_errno.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

const uint64_t INT4_NUMS_IN_INT32 = 8;

static void TensorPreProcess(const aclTensorList *&tensorListKey, const aclTensorList *&tensorListValue)
{
    if (tensorListKey == nullptr) {
        OP_LOGD("TensorListKey is nullptr,TensorPreProcess exit.");
        return;
    }
    if (tensorListValue == nullptr) {
        OP_LOGD("tensorListValue is nullptr,TensorPreProcess exit.");
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

static void PrefixTensorPreProcess(const aclTensor *&tensorKey, const aclTensor *&tensorValue)
{
    if (tensorKey == nullptr) {
        OP_LOGD("TensorListKey is nullptr,TensorPreProcess exit.");
        return;
    }
    if (tensorValue == nullptr) {
        OP_LOGD("tensorListValue is nullptr,TensorPreProcess exit..");
        return;
    }
    if (tensorKey->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kvPrefix's from OriginalShape is completed.");
        return;
    }
    if (tensorValue->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kvPrefix's from OriginalShape is completed.");
        return;
    }
    auto tempKey = const_cast<aclTensor *>(tensorKey);
    op::Shape viewKeyShape = tempKey->GetViewShape();
    auto viewKeyShapeDim = viewKeyShape.GetDimNum();
    viewKeyShape[viewKeyShapeDim - 1] = viewKeyShape[viewKeyShapeDim - 1] * INT4_NUMS_IN_INT32;
    tempKey->SetViewShape(viewKeyShape);
    tempKey->SetDataType(DataType::DT_INT4);

    auto tempValue = const_cast<aclTensor *>(tensorValue);
    op::Shape viewValueShape = tempValue->GetViewShape();
    auto viewValueShapeDim = viewValueShape.GetDimNum();
    viewValueShape[viewValueShapeDim - 1] = viewValueShape[viewValueShapeDim - 1] * INT4_NUMS_IN_INT32;
    tempValue->SetViewShape(viewValueShape);
    tempValue->SetDataType(DataType::DT_INT4);

    OP_LOGD("The conversion of kvPrefix from int32 to int4 is completed.");
}

static aclnnStatus FakeArray(const aclIntArray *inArray, aclTensor *&outTensor)
{
    OP_LOGD("start fake tensor");
    if (inArray != nullptr) {
        OP_LOGD("input array is not nullptr");
        int64_t size = static_cast<int64_t>(inArray->Size());
        std::vector<int64_t> shape = {size};
        outTensor = aclCreateTensor(shape.data(),
                                    shape.size(),
                                    aclDataType::ACL_INT64,
                                    nullptr,
                                    0,
                                    ACL_FORMAT_ND,
                                    shape.data(),
                                    shape.size(),
                                    nullptr);
        if (outTensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc tensor failed");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    OP_LOGD("end fake tensor");
    return ACLNN_SUCCESS;
}

static void FusedInferAttentionScoreV2SinkProcessSoftmaxLse(bool softmaxLseFlag,
                                                     const aclTensor *softmaxLse,
                                                     const aclTensor *tempTensor,
                                                     const aclTensor *&placeHolder)
{
    if (softmaxLseFlag == false) {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        tempTensor = aclCreateTensor(shape.data(),
                                     shape.size(),
                                     aclDataType::ACL_FLOAT,
                                     shape.data(),
                                     0,
                                     ACL_FORMAT_ND,
                                     shape.data(),
                                     shape.size(),
                                     static_cast<void *>(&addr));
        placeHolder = tempTensor;
    } else {
        placeHolder = softmaxLse;
    }
}

// aclnnV1 永不返回真实的 softmaxMax/softmaxSum，故对二者始终创建 shape=[0]、假地址的
// placeholder 占位 tensor，仅用于满足 L0 的 4 出参签名。调用方负责在末尾 aclDestroyTensor 释放。
aclTensor *FusedInferAttentionScoreV2SinkCreateMaxSumPlaceholder()
{
    std::vector<int64_t> shape = {0};
    int64_t addr = 0xff;
    auto *tensor = aclCreateTensor(shape.data(),
                                   shape.size(),
                                   aclDataType::ACL_FLOAT,
                                   shape.data(),
                                   0,
                                   ACL_FORMAT_ND,
                                   shape.data(),
                                   shape.size(),
                                   static_cast<void *>(&addr));
    return tensor;
}

// CreateView 保留非连续 tensor 的 stride 信息
static const aclTensor *CalcNoContiguous(const aclTensor *input, aclOpExecutor *executor)
{
    if (input == nullptr) {
        return input;
    }
    // CreateView 保留非连续 tensor 的 stride 信息
    // 这样底层算子可以正确处理非连续的访存模式
    aclTensor *newInput = executor->CreateView(input,
                                               input->GetViewShape(), // 视图形状
                                               input->GetStorageShape(), // 底层存储形状
                                               input->GetViewStrides(), // 步幅（非连续的关键信息）
                                               input->GetViewOffset() // 偏移量
    );
    CHECK_RET(newInput != nullptr, nullptr);
    return newInput;
}

// 处理非连续 tensor，如果是非连续则使用 CreateView，否则按原样返回
static const aclTensor *ProcessTensorContiguous(
    const aclTensor *tensor,
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

// 处理 TensorList 中所有 tensor 的连续性，返回新的 TensorList
static const aclTensorList *ProcessTensorListContiguous(const aclTensorList *tensorList,
                                                        aclOpExecutor *executor,
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
            // continue;
        }

        // 处理单个 tensor 的连续性
        const aclTensor *processedTensor = ProcessTensorContiguous(tensor, executor, tensorListName);

        CHECK_RET(processedTensor != nullptr, nullptr);
        processedTensorsTmp.push_back(processedTensor);
    }

    // 创建新的 TensorList
    aclTensorList *processedTensors = executor->AllocTensorList(processedTensorsTmp.data(), processedTensorsTmp.size());
    CHECK_RET(processedTensors != nullptr, nullptr);
    return processedTensors;
}

/**
 * @brief for acl graph calculates the max workspace size based on the specific calculation process.
 * declaration here for testcase to use by extern the interface
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedInferAttentionScoreV2SinkGetMaxWorkspaceSize(
    const aclTensor *query,
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
    const aclTensor *attentionOut,
    const aclTensor *softmaxLse,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void *executor, const size_t irIndex);

aclnnStatus aclnnFusedInferAttentionScoreV2SinkGetMaxWorkspaceSize(const aclTensor *query,
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
                                                                   const aclTensor *attentionOut,
                                                                   const aclTensor *softmaxLse,
                                                                   uint64_t *workspaceSize,
                                                                   aclOpExecutor **executor)
{
    OP_LOGD("start aclnnFusedInferAttentionScoreV2SinkGetMaxWorkspaceSize");
    TensorPreProcess(tensorListKey, tensorListValue);

    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;
    FusedInferAttentionScoreV2SinkProcessSoftmaxLse(softmaxLseFlag, softmaxLse, tempTensor, placeHolder);

    auto ret = aclnnFusedInferAttentionScoreV2SinkGetWorkspaceSize(query,
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
                                                                   attentionOut,
                                                                   placeHolder,
                                                                   workspaceSize,
                                                                   executor);
    if (softmaxLseFlag == false) {
        aclDestroyTensor(tempTensor);
    }

    return ret;
}

aclnnStatus aclnnFusedInferAttentionScoreV2SinkGetWorkspaceSize(const aclTensor *query,
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
                                                                const aclTensor *attentionOut,
                                                                const aclTensor *softmaxLse,
                                                                uint64_t *workspaceSize,
                                                                aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFusedInferAttentionScoreV2Sink,
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
                          sinkNumber),
                   DFX_OUT(attentionOut, softmaxLse));
    const aclTensorList *tensorListKey = key;
    const aclTensorList *tensorListValue = value;
    TensorPreProcess(tensorListKey, tensorListValue);

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

    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;
    const aclTensor *learnableSinkOptional = nullptr;
    FusedInferAttentionScoreV2SinkProcessSoftmaxLse(softmaxLseFlag, softmaxLse, tempTensor, placeHolder);

    // aclnnV1 不返回 softmaxMax/softmaxSum，始终创建 shape=[0] placeholder 占位，仅满足 L0 4 出参签名
    auto *softmaxMaxPlaceholder = FusedInferAttentionScoreV2SinkCreateMaxSumPlaceholder();
    auto *softmaxSumPlaceholder = FusedInferAttentionScoreV2SinkCreateMaxSumPlaceholder();

    // 调用 L0 接口 - 使用处理后的 tensor（保留 stride 信息）
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
        false,
        false,
        0,
        attentionOut,
        placeHolder,
        softmaxMaxPlaceholder,
        softmaxSumPlaceholder,
        l0Executor);
    CHECK_RET(l0Outputs.AttentionOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0Outputs.SoftmaxOutOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 使用 ViewCopy 复制输出
    auto viewCopyAttentionOutResult = l0op::ViewCopy(l0Outputs.AttentionOutOut, attentionOut, l0Executor);
    auto viewCopySoftmaxLseOutResult = l0op::ViewCopy(l0Outputs.SoftmaxOutOut, placeHolder, l0Executor);
    // max/sum 占位输出丢弃（v1 不返回），仅 ViewCopy 到 placeholder 保证 launch 图完整
    (void)l0op::ViewCopy(l0Outputs.SoftmaxMaxOut, softmaxMaxPlaceholder, l0Executor);
    (void)l0op::ViewCopy(l0Outputs.SoftmaxSumOut, softmaxSumPlaceholder, l0Executor);

    CHECK_RET(viewCopyAttentionOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(viewCopySoftmaxLseOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (softmaxLseFlag == false) {
        aclDestroyTensor(tempTensor);
    }
    aclDestroyTensor(softmaxMaxPlaceholder);
    aclDestroyTensor(softmaxSumPlaceholder);

    // 获取 workspace 大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedInferAttentionScoreV2Sink(void *workspace,
                                                uint64_t workspaceSize,
                                                aclOpExecutor *executor,
                                                const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedInferAttentionScoreV2Sink);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif