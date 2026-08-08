/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_fused_infer_attention_score_v5.h"

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "fused_infer_attention_score_inner.h"
#include "aclnnInner_fused_infer_attention_score.h" // 该文件为自动生成，在build/autogen/inner路径下
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
/**
 * @brief for acl graph calculates the max workspace size based on the specific calculation process.
 * declaration here for testcase to use by extern the interface
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedInferAttentionScoreV5GetMaxWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional, const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional, const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional, const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, const aclTensor *keyRopeAntiquantScaleOptional, 
    const aclTensor *dequantScaleQueryOptional, const aclTensor *learnableSinkOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, int64_t numHeads, 
    double scaleValue, int64_t preTokens, int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads, 
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag,
    int64_t keyAntiquantMode, int64_t valueAntiquantMode, int64_t queryQuantMode, int64_t pseType,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus CheckTensorContiguous(
    const aclTensorList *key, const aclTensorList *value,
    const aclTensor *keyAntiquantScaleOptional,
    const aclTensor *valueAntiquantScaleOptional,
    const aclTensor *keyRopeOptional);

// 新版本opbase存在TensorV2的新接口，用弱符号判断当前opbase是新版本还是旧版本，旧版本不支持传入非连续tensor
bool NnopbaseSupportTensorV2() __attribute__((weak));

aclnnStatus aclnnFusedInferAttentionScoreV5GetMaxWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
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
    int64_t numHeads, double scaleValue, int64_t preTokens,
    int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize,
    int64_t antiquantMode, bool softmaxLseFlag,
    int64_t keyAntiquantMode, int64_t valueAntiquantMode, int64_t queryQuantMode, int64_t pseType,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("start aclnnFusedInferAttentionScoreV5GetMaxWorkspaceSize");
    const aclTensorList *tensorListKey = key;
    const aclTensorList *tensorListValue = value;
    TensorPreProcess(tensorListKey, tensorListValue);

    const aclTensor *tensorKeySharedPrefixOptional = keySharedPrefixOptional;
    const aclTensor *tensorValueSharedPrefixOptional = valueSharedPrefixOptional;
    PrefixTensorPreProcess(tensorKeySharedPrefixOptional, tensorValueSharedPrefixOptional);

    aclTensor *fakeActualSeqLengthsOptional{nullptr};
    aclTensor *fakeActualSeqLengthsKvOptional{nullptr};
    aclTensor *fakeActualSharedPrefixLenOptional{nullptr};
    aclTensor *fakeQStartIdxOptional{nullptr};
    aclTensor *fakeKVStartIdxOptional{nullptr};

    // nullptr不处理， nullptr是空指针，这样不会影响原来就不传入actual seq length为空的逻辑
    aclnnStatus ret = FakeArray(actualSeqLengthsOptional, fakeActualSeqLengthsOptional);
    CHECK_RET_CODE(ret, "Try alloc fake actualSeqLengthsOptional failed");

    ret = FakeArray(actualSeqLengthsKvOptional, fakeActualSeqLengthsKvOptional);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc fake actualSeqLengthsKvOptional failed");
        aclDestroyTensor(fakeActualSeqLengthsOptional); // 没有返回值无需校验
        return ret;
    }

    ret = FakeArray(actualSharedPrefixLenOptional, fakeActualSharedPrefixLenOptional);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc fake actualSharedPrefixLenOptional failed");
        aclDestroyTensor(fakeActualSeqLengthsOptional); // 没有返回值无需校验
        aclDestroyTensor(fakeActualSeqLengthsKvOptional);
        return ret;
    }

    ret = FakeArray(qStartIdxOptional, fakeQStartIdxOptional);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc fake qStartIdxOptional failed");
        aclDestroyTensor(fakeActualSeqLengthsOptional); // 没有返回值无需校验
        aclDestroyTensor(fakeActualSeqLengthsKvOptional);
        aclDestroyTensor(fakeActualSharedPrefixLenOptional);
        return ret;
    }

    ret = FakeArray(kvStartIdxOptional, fakeKVStartIdxOptional);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc fake kvStartIdxOptional failed");
        aclDestroyTensor(fakeActualSeqLengthsOptional); // 没有返回值无需校验
        aclDestroyTensor(fakeActualSeqLengthsKvOptional);
        aclDestroyTensor(fakeActualSharedPrefixLenOptional);
        aclDestroyTensor(fakeQStartIdxOptional);
        return ret;
    }

    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;
    if (softmaxLseFlag == false) {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        tempTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), static_cast<void*>(&addr));
        placeHolder = tempTensor;
    } else {
        placeHolder = softmaxLse;
    }

    ret = aclnnInnerFusedInferAttentionScoreTensorGetWorkspaceSize(
        query, tensorListKey, tensorListValue, pseShiftOptional, attenMaskOptional, fakeActualSeqLengthsOptional,
        fakeActualSeqLengthsKvOptional, deqScale1Optional, quantScale1Optional, deqScale2Optional, quantScale2Optional,
        quantOffset2Optional, antiquantScaleOptional, antiquantOffsetOptional, blockTableOptional,
        queryPaddingSizeOptional, kvPaddingSizeOptional, keyAntiquantScaleOptional, keyAntiquantOffsetOptional,
        valueAntiquantScaleOptional, valueAntiquantOffsetOptional, tensorKeySharedPrefixOptional,
        tensorValueSharedPrefixOptional, fakeActualSharedPrefixLenOptional, queryRopeOptional,
        keyRopeOptional, keyRopeAntiquantScaleOptional, dequantScaleQueryOptional, learnableSinkOptional, fakeQStartIdxOptional, fakeKVStartIdxOptional, 
        numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads, sparseMode, innerPrecise, 
        blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, queryQuantMode, pseType, 0,
        attentionOut, placeHolder, workspaceSize, executor);
    if (softmaxLseFlag == false) {
        aclDestroyTensor(tempTensor);
    }
    aclDestroyTensor(fakeActualSeqLengthsOptional); // 只会成功，无需校验
    aclDestroyTensor(fakeActualSeqLengthsKvOptional);
    aclDestroyTensor(fakeActualSharedPrefixLenOptional);
    aclDestroyTensor(fakeQStartIdxOptional);
    aclDestroyTensor(fakeKVStartIdxOptional);
    return ret;
}

aclnnStatus CheckTensorContiguous(
    const aclTensorList *key, const aclTensorList *value,
    const aclTensor *keyAntiquantScaleOptional,
    const aclTensor *valueAntiquantScaleOptional,
    const aclTensor *keyRopeOptional)
{
    if ((key != nullptr && !IsContiguous((*key)[0])) || (value != nullptr && !IsContiguous((*value)[0]))) {
        return ACLNN_ERR_INNER_TILING_ERROR;
    }
    if (keyRopeOptional != nullptr && !IsContiguous(keyRopeOptional)) {
        return ACLNN_ERR_INNER_TILING_ERROR;
    }
    if ((keyAntiquantScaleOptional != nullptr && !IsContiguous(keyAntiquantScaleOptional)) ||
        (valueAntiquantScaleOptional != nullptr && !IsContiguous(valueAntiquantScaleOptional))) {
        return ACLNN_ERR_INNER_TILING_ERROR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
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
    int64_t numHeads, double scaleValue, int64_t preTokens,
    int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize,
    int64_t antiquantMode, bool softmaxLseFlag,
    int64_t keyAntiquantMode, int64_t valueAntiquantMode, int64_t queryQuantMode, int64_t pseType,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const aclTensorList *tensorListKey = key;
    const aclTensorList *tensorListValue = value;
    TensorPreProcess(tensorListKey, tensorListValue);

    const aclTensor *tensorKeySharedPrefixOptional = keySharedPrefixOptional;
    const aclTensor *tensorValueSharedPrefixOptional = valueSharedPrefixOptional;
    PrefixTensorPreProcess(tensorKeySharedPrefixOptional, tensorValueSharedPrefixOptional);

    aclnnStatus ret = CheckTensorContiguous(key, value, keyAntiquantScaleOptional, valueAntiquantScaleOptional, keyRopeOptional);
    if (ret != ACLNN_SUCCESS && NnopbaseSupportTensorV2 == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_TILING_ERROR, "When tensor is not contiguous, opbase package version check failed");
        return ret;
    }

    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;
    if (softmaxLseFlag == false) {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        tempTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), static_cast<void*>(&addr));
        placeHolder = tempTensor;
    } else {
        placeHolder = softmaxLse;
    }
    ret = aclnnInnerFusedInferAttentionScoreGetWorkspaceSize(
        query, tensorListKey, tensorListValue, pseShiftOptional, attenMaskOptional, actualSeqLengthsOptional,
        actualSeqLengthsKvOptional, deqScale1Optional, quantScale1Optional, deqScale2Optional, quantScale2Optional,
        quantOffset2Optional, antiquantScaleOptional, antiquantOffsetOptional, blockTableOptional,
        queryPaddingSizeOptional, kvPaddingSizeOptional, keyAntiquantScaleOptional, keyAntiquantOffsetOptional,
        valueAntiquantScaleOptional, valueAntiquantOffsetOptional, tensorKeySharedPrefixOptional,
        tensorValueSharedPrefixOptional, actualSharedPrefixLenOptional, queryRopeOptional,
        keyRopeOptional, keyRopeAntiquantScaleOptional, dequantScaleQueryOptional, learnableSinkOptional, qStartIdxOptional, kvStartIdxOptional,
        numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads, sparseMode, innerPrecise, blockSize, 
        antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, queryQuantMode, pseType, 0, attentionOut, 
        placeHolder, workspaceSize, executor);
    if (softmaxLseFlag == false) {
        aclDestroyTensor(tempTensor);
    }
    return ret;
}

aclnnStatus aclnnFusedInferAttentionScoreV5(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            const aclrtStream stream)
{
    return aclnnInnerFusedInferAttentionScore(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif