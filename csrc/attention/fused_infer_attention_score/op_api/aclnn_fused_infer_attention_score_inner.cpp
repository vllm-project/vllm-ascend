/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_infer_attention_score_inner.h"
#include "fused_infer_attention_score.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/fast_vector.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "acl/acl.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
const uint64_t INT4_NUMS_IN_INT32 = 8;
}

// The weak symbol is available when opbase supports TensorV2 view strides.
bool NnopbaseSupportTensorV2() __attribute__((weak));

void TensorPreProcess(const aclTensorList *&tensorListKey, const aclTensorList *&tensorListValue) {
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


void PrefixTensorPreProcess(const aclTensor *&tensorKey, const aclTensor *&tensorValue) {
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

aclnnStatus FakeArray(const aclIntArray *inArray, aclTensor *&outTensor) {
    OP_LOGD("start fake tensor");
    if (inArray != nullptr) {
        OP_LOGD("input array is not nullptr");
        int64_t size = static_cast<int64_t>(inArray->Size());
        std::vector<int64_t> shape = {size};
        outTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT64, nullptr,
                                    0, ACL_FORMAT_ND, shape.data(), shape.size(), nullptr);
        if (outTensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc tensor failed");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    OP_LOGD("end fake tensor");
    return ACLNN_SUCCESS;
}

void FusedInferAttentionScoreProcessSoftmaxLse(bool softmaxLseFlag, const aclTensor *softmaxLse,
                                               const aclTensor *&tempTensor, const aclTensor *&placeHolder)
{
    if (softmaxLseFlag == false) {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        tempTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), static_cast<void*>(&addr));
        placeHolder = tempTensor;
    } else {
        placeHolder = softmaxLse;
    }
}

#ifdef __cplusplus
}
#endif

namespace {
bool IsCacheScene(const aclTensor *blockTableOptional)
{
    return blockTableOptional != nullptr && blockTableOptional->GetViewShape().GetShapeSize() != 0;
}

bool GetAclTensorViewStrides(const aclTensor *tensor, int64_t *&stridesValue, uint64_t &stridesNum)
{
    stridesValue = nullptr;
    stridesNum = 0;
    auto retView = aclGetViewStrides(tensor, &stridesValue, &stridesNum);
    if (retView != ACL_SUCCESS || stridesValue == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclGetViewStrides failed.");
        delete[] stridesValue;
        stridesValue = nullptr;
        stridesNum = 0;
        return false;
    }
    return true;
}

bool IsFirstAxisOnlyNonContiguous(const aclTensor *tensor, const char *name)
{
    if (tensor == nullptr) {
        return true;
    }
    if (IsContiguous(tensor)) {
        return true;
    }

    auto viewShape = tensor->GetViewShape();
    int64_t dimNum = viewShape.GetDimNum();
    if (dimNum <= 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "%s view shape dim num[%ld] should be greater than 1 in the current cache scene.", name, dimNum);
        return false;
    }

    int64_t *viewStrides = nullptr;
    uint64_t stridesNum = 0;
    if (!GetAclTensorViewStrides(tensor, viewStrides, stridesNum)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Failed to get view strides for %s.", name);
        return false;
    }
    if (stridesNum < static_cast<uint64_t>(dimNum)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "%s view strides num[%lu] is less than view shape dim num[%ld].", name, stridesNum, dimNum);
        delete[] viewStrides;
        return false;
    }

    bool isFirstAxisOnlyNonContiguous = true;
    int64_t expectedStride = 1;
    for (int64_t dim = dimNum - 1; dim >= 1; --dim) {
        if (viewShape.GetDim(dim) != 1 && viewStrides[dim] != expectedStride) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s is non-contiguous at axis[%ld] (0-based): actual stride[%ld], expected stride[%ld], "
                    "shape[%s]. Only axis[0] may be non-contiguous in the current cache scene.",
                    name, dim, viewStrides[dim], expectedStride, op::ToString(viewShape).GetString());
            isFirstAxisOnlyNonContiguous = false;
            break;
        }
        expectedStride *= viewShape.GetDim(dim);
    }

    delete[] viewStrides;
    return isFirstAxisOnlyNonContiguous;
}

aclnnStatus CheckTensorListFirstAxisOnlyNonContiguous(const aclTensorList *tensorList, const char *name)
{
    if (tensorList == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s tensorList is nullptr.", name);
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    for (uint64_t i = 0; i < tensorList->Size(); ++i) {
        auto tensor = (*tensorList)[i];
        std::string tensorName = name;
        if (tensorList->Size() > 1) {
            tensorName += "[" + std::to_string(i) + "]";
        }
        if (!IsFirstAxisOnlyNonContiguous(tensor, tensorName.c_str())) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus MakeTensorListContiguous(const aclTensorList *&tensorList, const char *name, aclOpExecutor *executor)
{
    if (tensorList == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s tensorList is nullptr.", name);
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    std::vector<const aclTensor *> tensors;
    tensors.reserve(tensorList->Size());
    for (uint64_t i = 0; i < tensorList->Size(); ++i) {
        auto tensor = (*tensorList)[i];
        if (tensor == nullptr) {
            tensors.emplace_back(nullptr);
            continue;
        }
        auto contiguousTensor = l0op::Contiguous(tensor, executor);
        if (contiguousTensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try make %s[%lu] contiguous failed.", name, i);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        tensors.emplace_back(contiguousTensor);
    }

    auto contiguousList = executor->AllocTensorList(tensors.data(), tensors.size());
    if (contiguousList == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc contiguous %s tensorList failed.", name);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    tensorList = contiguousList;
    return ACLNN_SUCCESS;
}

void SetTensorFormatToND(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return;
    }
    auto mutableTensor = const_cast<aclTensor *>(tensor);
    mutableTensor->SetStorageFormat(Format::FORMAT_ND);
    mutableTensor->SetViewFormat(Format::FORMAT_ND);
    mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
}

aclnnStatus MakeTensorContiguous(const aclTensor *&tensor, const char *name, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }

    tensor = l0op::Contiguous(tensor, executor);
    if (tensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try make %s contiguous failed.", name);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    SetTensorFormatToND(tensor);
    return ACLNN_SUCCESS;
}

aclnnStatus NormalizeCacheTensorList(const aclTensorList *&tensorList, const char *name, aclOpExecutor *executor)
{
    if (tensorList == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s tensorList is nullptr.", name);
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    std::vector<const aclTensor *> tensors;
    tensors.reserve(tensorList->Size());
    for (uint64_t i = 0; i < tensorList->Size(); ++i) {
        auto tensor = (*tensorList)[i];
        if (tensor == nullptr) {
            tensors.emplace_back(nullptr);
            continue;
        }

        const aclTensor *normalizedTensor = nullptr;
        std::string itemName = std::string(name) + "[" + std::to_string(i) + "]";
        if (IsContiguous(tensor)) {
            normalizedTensor = l0op::Contiguous(tensor, executor);
            if (normalizedTensor == nullptr) {
                OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try normalize contiguous %s failed.", itemName.c_str());
                return ACLNN_ERR_INNER_NULLPTR;
            }
        } else {
            normalizedTensor = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetStorageShape(),
                                                    tensor->GetViewStrides(), tensor->GetViewOffset());
            if (normalizedTensor == nullptr) {
                OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try create view for %s failed.", itemName.c_str());
                return ACLNN_ERR_INNER_NULLPTR;
            }
            const_cast<aclTensor *>(normalizedTensor)->SetStorageShape(tensor->GetViewShape());
            SetTensorFormatToND(normalizedTensor);
        }
        tensors.emplace_back(normalizedTensor);
    }

    auto normalizedList = executor->AllocTensorList(tensors.data(), tensors.size());
    if (normalizedList == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc normalized %s tensorList failed.", name);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    tensorList = normalizedList;
    return ACLNN_SUCCESS;
}

aclnnStatus NormalizeCacheTensor(const aclTensor *&tensor, const char *name, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s tensor is nullptr.", name);
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (IsContiguous(tensor)) {
        tensor = l0op::Contiguous(tensor, executor);
        if (tensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try normalize contiguous %s failed.", name);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    } else {
        tensor = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetStorageShape(),
                                      tensor->GetViewStrides(), tensor->GetViewOffset());
        if (tensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try create view for %s failed.", name);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        const_cast<aclTensor *>(tensor)->SetStorageShape(tensor->GetViewShape());
    }
    SetTensorFormatToND(tensor);
    return ACLNN_SUCCESS;
}

aclnnStatus ProcessKVForL0Input(const aclTensorList *&key, const aclTensorList *&value,
                                const aclTensor *blockTableOptional, aclOpExecutor *executor)
{
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (key == nullptr || value == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "key or value tensorList is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (key->Size() != value->Size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "key tensorList size[%lu] should be equal to value tensorList size[%lu].",
                key->Size(), value->Size());
        return ACLNN_ERR_PARAM_INVALID;
    }

    const bool isCacheScene = IsCacheScene(blockTableOptional);
    const bool supportTensorV2 = NnopbaseSupportTensorV2 != nullptr;
    if (supportTensorV2 && isCacheScene) {
        CHECK_RET(CheckTensorListFirstAxisOnlyNonContiguous(key, "key") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckTensorListFirstAxisOnlyNonContiguous(value, "value") == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(NormalizeCacheTensorList(key, "key", executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(NormalizeCacheTensorList(value, "value", executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    }

    if (!supportTensorV2) {
        OP_LOGW("Current opbase does not support TensorV2, make key and value contiguous.");
    }

    CHECK_RET(MakeTensorListContiguous(key, "key", executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorListContiguous(value, "value", executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus InnerFusedInferAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
    const aclTensor *pseShiftOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *deqScale1Optional, const aclTensor *quantScale1Optional,
    const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional, const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, const aclTensor *keyRopeAntiquantScaleOptional,
    const aclTensor *dequantScaleQueryOptional, const aclTensor *learnableSinkOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    int64_t numHeads, double scaleValue, int64_t preTokens, int64_t nextTokens,
    char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
    int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode,
    int64_t valueAntiquantMode, int64_t queryQuantMode, int64_t pseType, int64_t outDtype,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    OP_CHECK_NULL(query, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(key, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(value, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(attentionOut, return ACLNN_ERR_PARAM_NULLPTR);
    if (workspaceSize == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "workspaceSize is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (softmaxLseFlag) {
        OP_CHECK_NULL(softmaxLse, return ACLNN_ERR_PARAM_NULLPTR);
    }
    if (inputLayout == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "inputLayout is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (attentionOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor *processedQuery = query;
    const aclTensorList *processedKey = key;
    const aclTensorList *processedValue = value;
    const aclTensor *processedPseShift = pseShiftOptional;
    const aclTensor *processedAttenMask = attenMaskOptional;
    const aclTensor *processedBlockTable = blockTableOptional;
    const aclTensor *processedKeySharedPrefix = keySharedPrefixOptional;
    const aclTensor *processedValueSharedPrefix = valueSharedPrefixOptional;
    const aclTensor *processedQueryRope = queryRopeOptional;
    const aclTensor *processedKeyRope = keyRopeOptional;
    const aclTensor *processedDeqScale1 = deqScale1Optional;
    const aclTensor *processedQuantScale1 = quantScale1Optional;
    const aclTensor *processedDeqScale2 = deqScale2Optional;
    const aclTensor *processedQuantScale2 = quantScale2Optional;
    const aclTensor *processedQuantOffset2 = quantOffset2Optional;
    const aclTensor *processedAntiquantScale = antiquantScaleOptional;
    const aclTensor *processedAntiquantOffset = antiquantOffsetOptional;
    const aclTensor *processedQueryPaddingSize = queryPaddingSizeOptional;
    const aclTensor *processedKvPaddingSize = kvPaddingSizeOptional;
    const aclTensor *processedKeyAntiquantScale = keyAntiquantScaleOptional;
    const aclTensor *processedKeyAntiquantOffset = keyAntiquantOffsetOptional;
    const aclTensor *processedValueAntiquantScale = valueAntiquantScaleOptional;
    const aclTensor *processedValueAntiquantOffset = valueAntiquantOffsetOptional;
    const aclTensor *processedKeyRopeAntiquantScale = keyRopeAntiquantScaleOptional;
    const aclTensor *processedDequantScaleQuery = dequantScaleQueryOptional;
    const aclTensor *processedLearnableSink = learnableSinkOptional;

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    CHECK_RET(MakeTensorContiguous(processedQuery, "query", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedPseShift, "pseShift", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedAttenMask, "attenMask", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedBlockTable, "blockTable", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedKeySharedPrefix, "keySharedPrefix", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedValueSharedPrefix, "valueSharedPrefix", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedQueryRope, "queryRope", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    if (NnopbaseSupportTensorV2 != nullptr && IsCacheScene(processedBlockTable) && processedKeyRope != nullptr) {
        CHECK_RET(IsFirstAxisOnlyNonContiguous(processedKeyRope, "keyRope"), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(NormalizeCacheTensor(processedKeyRope, "keyRope", l0Executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
    } else {
        CHECK_RET(MakeTensorContiguous(processedKeyRope, "keyRope", l0Executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
    }
    CHECK_RET(MakeTensorContiguous(processedDeqScale1, "deqScale1", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedQuantScale1, "quantScale1", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedDeqScale2, "deqScale2", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedQuantScale2, "quantScale2", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedQuantOffset2, "quantOffset2", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedAntiquantScale, "antiquantScale", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedAntiquantOffset, "antiquantOffset", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedQueryPaddingSize, "queryPaddingSize", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedKvPaddingSize, "kvPaddingSize", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedKeyAntiquantScale, "keyAntiquantScale", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedKeyAntiquantOffset, "keyAntiquantOffset", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedValueAntiquantScale, "valueAntiquantScale", l0Executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedValueAntiquantOffset, "valueAntiquantOffset", l0Executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedKeyRopeAntiquantScale, "keyRopeAntiquantScale", l0Executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedDequantScaleQuery, "dequantScaleQuery", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeTensorContiguous(processedLearnableSink, "learnableSink", l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    auto kvProcessRet = ProcessKVForL0Input(processedKey, processedValue, processedBlockTable, l0Executor);
    CHECK_RET(kvProcessRet == ACLNN_SUCCESS, kvProcessRet);

    auto l0Out = l0op::FusedInferAttentionScore(
        processedQuery, processedKey, processedValue, processedPseShift, processedAttenMask, actualSeqLengthsOptional,
        actualSeqLengthsKvOptional, processedDeqScale1, processedQuantScale1, processedDeqScale2, processedQuantScale2,
        processedQuantOffset2, processedAntiquantScale, processedAntiquantOffset, processedBlockTable,
        processedQueryPaddingSize, processedKvPaddingSize, processedKeyAntiquantScale, processedKeyAntiquantOffset,
        processedValueAntiquantScale, processedValueAntiquantOffset, processedKeySharedPrefix,
        processedValueSharedPrefix, actualSharedPrefixLenOptional, processedQueryRope, processedKeyRope,
        processedKeyRopeAntiquantScale, processedDequantScaleQuery, processedLearnableSink, qStartIdxOptional,
        kvStartIdxOptional, numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads,
        sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode,
        queryQuantMode, pseType, outDtype, attentionOut, softmaxLse, l0Executor);

    auto l0AttentionOut = std::get<0>(l0Out);
    auto l0SoftmaxLse = std::get<1>(l0Out);
    if (l0AttentionOut == nullptr || l0SoftmaxLse == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto attentionViewCopy = l0op::ViewCopy(l0AttentionOut, attentionOut, l0Executor);
    if (attentionViewCopy == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (softmaxLseFlag) {
        auto softmaxLseViewCopy = l0op::ViewCopy(l0SoftmaxLse, softmaxLse, l0Executor);
        CHECK_RET(softmaxLseViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus InnerFusedInferAttentionScore(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(InnerFusedInferAttentionScore);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
