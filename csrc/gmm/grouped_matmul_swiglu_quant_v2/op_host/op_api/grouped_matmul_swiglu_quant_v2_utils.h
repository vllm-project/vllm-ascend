/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_GROUPED_MATMUL_SWIGLU_QUANT_V2_UTILS_H
#define OP_HOST_OP_API_GROUPED_MATMUL_SWIGLU_QUANT_V2_UTILS_H

#include "grouped_matmul_swiglu_quant_utils.h"
#include "util/math_util.h"

namespace gmmSwigluQuantV2 {

using namespace gmm_dsq;

constexpr int64_t OUTPUT_IDX_0 = 0L;
constexpr int64_t OUTPUT_IDX_1 = 1L;
constexpr size_t MX_SPLIT_K_PER_TOKEN_SCALE_DIM = 3UL;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_THIRD_DIM_INDEX = 3;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2L;
constexpr size_t MX_SPLIT_M_SCALE_DIM = 4UL;
constexpr size_t MX_X_DIM = 2UL;
constexpr size_t MX_X_SCALE_DIM = 3UL;
constexpr size_t MX_WEIGHT_DIM = 3UL;
constexpr size_t MX_WEIGHT_SCALE_DIM = 4UL;
constexpr size_t MX_OUTPUT_DIM = 2UL;
constexpr size_t MX_OUTPUT_SCALE_DIM = 3UL;
constexpr size_t PERTOKEN_X_DIM = 2;
constexpr size_t PERTOKEN_X_SCALE_DIM = 1;
constexpr size_t PERTOKEN_WEIGHT_DIM = 3;
constexpr size_t PERTOKEN_WEIGHT_SCALE_DIM = 2;
constexpr size_t PERTOKEN_OUTPUT_DIM = 2;
constexpr size_t PERTOKEN_OUTPUT_SCALE_DIM = 1;
constexpr int64_t SWIGLU_SPLIT_FACTOR = 2L;
constexpr int64_t SWIGLU_SPLIT_SIZE = 64L;
constexpr int64_t MXFP4_K_CONSTRAINT = 2L;
constexpr int64_t SWIGLU_N_CONSTRAINT = 2L;
constexpr int64_t MXFP4_N_CONSTRAINT = 4L;
constexpr size_t SINGLE_TENSOR_SIZE = 1;
constexpr int64_t MAX_GROUP_LIST_SIZE = 1024L;
constexpr int64_t QUNAT_MODE_MX = 2;
constexpr int64_t QUNAT_MODE_PERTOKEN = 0;

const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2};
const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST_MXFP4 = {DataType::DT_FLOAT4_E2M1};
const std::initializer_list<DataType> XW_DTYPE_SUPPORT_LIST_PERTOKEN = {
    DataType::DT_INT8, DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2, DataType::DT_HIFLOAT8};
const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN,
                                                                   DataType::DT_FLOAT8_E5M2};
const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST_MXFP4 = {DataType::DT_FLOAT4_E2M1};
const std::initializer_list<DataType> WEIGHT_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
const std::initializer_list<DataType> WEIGHT_SCALE_DTYPE_SUPPORT_LIST_PERTOKEN_XINT8 = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT};
const std::initializer_list<DataType> WEIGHT_SCALE_DTYPE_SUPPORT_LIST_PERTOKEN_XFP8HIF8 = {DataType::DT_BF16,
                                                                                           DataType::DT_FLOAT};
const std::initializer_list<DataType> X_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
const std::initializer_list<DataType> X_SCALE_DTYPE_SUPPORT_LIST_PERTOKEN = {DataType::DT_FLOAT};
const std::initializer_list<DataType> GROUP_LIST_DTYPE_SUPPORT_LIST = {DataType::DT_INT64};
const std::initializer_list<DataType> QUANTOUT_DTYPE_SUPPORT_LIST_MXFP4 = {
    DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT4_E2M1};
const std::initializer_list<DataType> QUANTOUT_DTYPE_SUPPORT_LIST_PERTOKEN = {
    DataType::DT_INT8, DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2, DataType::DT_HIFLOAT8};
const std::initializer_list<DataType> QUANTSCALEOUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
const std::initializer_list<DataType> QUANTSCALEOUT_DTYPE_SUPPORT_LIST_PERTOKEN = {DataType::DT_FLOAT};

class GroupedMatmulSwigluQuantBaseHandler : public GroupedMatmulSwigluQuantHandler {
protected:
    bool IsTransposeForMxShape(const aclTensor *tensor) const
    {
        auto shape = tensor->GetViewShape();
        if (shape.GetDimNum() < MX_SPLIT_K_PER_TOKEN_SCALE_DIM) {
            return false;
        }
        int64_t firstLastDim = shape.GetDimNum() - 1;
        int64_t secondLastDim = shape.GetDimNum() - LAST_SECOND_DIM_INDEX;
        int64_t thirdLastDim = shape.GetDimNum() - LAST_THIRD_DIM_INDEX;
        auto strides = tensor->GetViewStrides();
        if (strides[firstLastDim] == 1 && strides[thirdLastDim] == MXFP_MULTI_BASE_SIZE &&
            strides[secondLastDim] == shape.GetDim(thirdLastDim) * MXFP_MULTI_BASE_SIZE) {
            return true;
        }
        return false;
    }

    bool IsTransposeLastTwoDims(const aclTensor *tensor) const
    {
        auto shape = tensor->GetViewShape();
        int64_t dim1 = shape.GetDimNum() - 1;
        int64_t dim2 = shape.GetDimNum() - 2;
        auto strides = tensor->GetViewStrides();
        if (strides[dim2] == 1 && strides[dim1] == shape.GetDim(dim2)) {
            int64_t tmpNxD = shape.GetDim(dim1) * shape.GetDim(dim2);
            for (int64_t batchDim = shape.GetDimNum() - 3; batchDim >= 0; batchDim--) {
                if (strides[batchDim] != tmpNxD) {
                    return false;
                }
                tmpNxD *= shape.GetDim(batchDim);
            }
            return true;
        }
        return false;
    }

    void CreateContiguousTensorListForMXTypeMScale(const aclTensorList *tensorList,
                                                   std::vector<aclTensor *> &newTensorList,
                                                   aclOpExecutor *executor) const
    {
        op::Shape shape;
        for (uint64_t idx = 0; idx < (*tensorList).Size(); idx++) {
            const aclTensor *inputTensor = (*tensorList)[idx];
            op::Shape viewShape = inputTensor->GetViewShape();
            shape.SetScalar();
            if (viewShape.GetDimNum() < MX_SPLIT_M_SCALE_DIM) {
                continue;
            }
            shape.AppendDim(viewShape.GetDim(0));
            shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - LAST_SECOND_DIM_INDEX));
            shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - LAST_THIRD_DIM_INDEX));
            shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - 1));
            aclTensor *tensor =
                executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset()); // use executor to create tensor
            tensor->SetStorageFormat(inputTensor->GetStorageFormat());
            newTensorList.emplace_back(tensor);
        }
    }

    void CreateContiguousTensorList(const aclTensorList *tensorList, std::vector<aclTensor *> &newTensorList,
                                    aclOpExecutor *executor) const
    {
        op::Shape shape;
        for (uint64_t idx = 0; idx < (*tensorList).Size(); idx++) {
            const aclTensor *inputTensor = (*tensorList)[idx];
            op::Shape viewShape = inputTensor->GetViewShape();
            uint32_t viewShapeDimsNum = viewShape.GetDimNum();
            shape.SetScalar();
            // 2: the second last dimension; in for-loops, it indicates dimensions before the second last remain unchanged.
            for (uint32_t i = 0; i < viewShapeDimsNum - 2; ++i) {
                shape.AppendDim(viewShape.GetDim(i));
            }
            // viewShapeDimsNum - 1, the dim value of the last dim. viewShapeDimsNum - 2, the dim value of the second
            // last dim.
            shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 1));
            shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 2)); // 2:the second last dim.
            aclTensor *tensor =
                executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset()); // use executor to create tensor
            tensor->SetStorageFormat(inputTensor->GetStorageFormat());
            newTensorList.emplace_back(tensor);
        }
    }

    static void CheckOptionalTensorListEmpty(const aclTensorList *&tensorList)
    {
        if (tensorList != nullptr) {
            if (tensorList->Size() == 0) {
                tensorList = nullptr;
            } else if ((*tensorList)[0] == nullptr) {
                tensorList = nullptr;
            } else if (tensorList->Size() == 1) {
                op::Shape shape = (*tensorList)[0]->GetViewShape();
                if (shape.GetDimNum() == 1 && shape.GetDim(0) == 0) {
                    tensorList = nullptr;
                }
            }
        }
    }

    bool CheckAttrs()
    {
        CheckOptionalTensorListEmpty(gmmDsqParams_.weightAssistMatrix);
        if (gmmDsqParams_.tuningConfig != nullptr && gmmDsqParams_.tuningConfig->Size() == 0) {
            gmmDsqParams_.tuningConfig = nullptr;
        }
        if (gmmDsqParams_.weightAssistMatrix != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The current version does not support weightAssistMatrix, it should be nullptr.");
            return false;
        }
        if (gmmDsqParams_.bias != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The current version does not support bias, it should be nullptr.");
            return false;
        }
        if (gmmDsqParams_.smoothScale != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The current version does not support smoothScale, it should be nullptr.");
            return false;
        }
        if (gmmDsqParams_.tuningConfig != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The current version does not support tuningConfig, it should be nullptr.");
            return false;
        }
        if ((gmmDsqParams_.dequantMode != QUNAT_MODE_MX && gmmDsqParams_.dequantMode != QUNAT_MODE_PERTOKEN) ||
            (gmmDsqParams_.quantMode != QUNAT_MODE_MX && gmmDsqParams_.quantMode != QUNAT_MODE_PERTOKEN) ||
            (gmmDsqParams_.dequantMode != gmmDsqParams_.quantMode)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Both dequantMode and quantMode must be 0 (pertoken) or 2 (mx), and they must be equal. Actual "
                    "value: dequantMode=%lu, dequantMode=%lu.",
                    gmmDsqParams_.dequantMode, gmmDsqParams_.quantMode);
            return false;
        }
        ge::DataType dequantDtype = static_cast<ge::DataType>(gmmDsqParams_.dequantDtype);
        if (gmmDsqParams_.quantMode == QUNAT_MODE_MX && dequantDtype != ge::DT_FLOAT) {
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "In mx quant mode, dequantDtype should be 0, but actual value is %lu.",
                     gmmDsqParams_.dequantDtype);
             return false;
        }
        if (gmmDsqParams_.quantMode == QUNAT_MODE_PERTOKEN && dequantDtype != ge::DT_FLOAT && dequantDtype != ge::DT_BF16 &&
            dequantDtype != ge::DT_FLOAT16) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "In pertoken quant mode, dequantDtype should be 0, 1, 27, but actual value is %lu.",
                    gmmDsqParams_.dequantDtype);
            return false;
        }
        return true;
    }

    bool CheckMXTranspose()
    {
        // 判断weight和weightScale是否转置，是则对两者进行转置动作
        bool transposeWeightScale = IsTransposeForMxShape((*gmmDsqParams_.weightScale)[0]);
        bool transposeWeight = IsTransposeLastTwoDims((*gmmDsqParams_.weight)[0]);
        bool transposeX = IsTransposeLastTwoDims(gmmDsqParams_.x);
        bool transposeXScale = IsTransposeForMxShape(gmmDsqParams_.xScale);

        if (transposeWeightScale != transposeWeight) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The transposition of weightScale/weight should be equal, but actual transpositions are %s/%s.",
                    transposeWeightScale ? "true" : "false", transposeWeight ? "true" : "false");
            return false;
        }

        if (transposeWeightScale && transposeWeight) {
            gmmDsqParams_.transposeWeight = true;
            auto uniqueExecutor = CREATE_EXECUTOR();
            CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
            aclOpExecutor *executorPtr = uniqueExecutor.get();
            CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
            std::vector<aclTensor *> scaleTensorList;
            std::vector<aclTensor *> weightTensorList;
            CreateContiguousTensorListForMXTypeMScale(gmmDsqParams_.weightScale, scaleTensorList, executorPtr);
            gmmDsqParams_.weightScale = executorPtr->AllocTensorList(scaleTensorList.data(), scaleTensorList.size());
            CreateContiguousTensorList(gmmDsqParams_.weight, weightTensorList, executorPtr);
            gmmDsqParams_.weight = executorPtr->AllocTensorList(weightTensorList.data(), weightTensorList.size());
            uniqueExecutor.ReleaseTo(executor_);
        }

        if ((gmmDsqParams_.x->GetViewShape().GetDim(0) == 1 && gmmDsqParams_.x->GetViewShape().GetDim(1) == 1) ||
            (gmmDsqParams_.xScale->GetViewShape().GetDim(0) == 1 &&
             gmmDsqParams_.xScale->GetViewShape().GetDim(1) == 1)) {
            return true;
        }
        if (transposeX || transposeXScale) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The transposition of x/xScale should be false, but actual transposition are %s/%s.",
                    transposeX ? "true" : "false", transposeXScale ? "true" : "false");
            return false;
        }
        return true;
    }

    bool CheckPertokenTranspose()
    {
        bool transposeWeight = IsTransposeLastTwoDims((*gmmDsqParams_.weight)[0]);
        bool transposeX = IsTransposeLastTwoDims(gmmDsqParams_.x);

        if (transposeWeight) {
            gmmDsqParams_.transposeWeight = true;
            auto uniqueExecutor = CREATE_EXECUTOR();
            CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
            aclOpExecutor *executorPtr = uniqueExecutor.get();
            CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
            std::vector<aclTensor *> weightTensorList;
            CreateContiguousTensorList(gmmDsqParams_.weight, weightTensorList, executorPtr);
            gmmDsqParams_.weight = executorPtr->AllocTensorList(weightTensorList.data(), weightTensorList.size());
            uniqueExecutor.ReleaseTo(executor_);
        }
        if ((gmmDsqParams_.x->GetViewShape().GetDim(0) == 1 && gmmDsqParams_.x->GetViewShape().GetDim(1) == 1) ||
            (gmmDsqParams_.xScale->GetViewShape().GetDim(0) == 1 &&
             gmmDsqParams_.xScale->GetViewShape().GetDim(1) == 1)) {
            return true;
        }
        if (transposeX) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The transposition of x should be false, but actual transposition are %s.",
                    transposeX ? "true" : "false");
            return false;
        }
        return true;
    }

    bool CheckMXShape()
    {
        int64_t m = gmmDsqParams_.x->GetViewShape().GetDim(0); // 从x的第0维获取m
        int64_t k = gmmDsqParams_.x->GetViewShape().GetDim(1); // 从x的第1维获取k
        // 转置情况下从weight的第1维获取n，非转置情况下从weight的第2维获取n
        int64_t n = gmmDsqParams_.transposeWeight ? ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1) :
                                                    ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(2);
        int64_t e = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(0); // 从weight的第0维获取e

        // x的shape期望为[M, K]
        op::Shape xExpectShape = {m, k};
        // xScale的shape期望为[M, CeilDiv(K, 64), 2]
        op::Shape xScaleExpectShape = {m, Ops::Base::CeilDiv(k, SWIGLU_SPLIT_SIZE), SWIGLU_SPLIT_FACTOR};
        // weight的shape期望为[E, K, N]
        op::Shape weightExpectShape = {e, k, n};
        // weightScale的shape期望为[E, CeilDiv(K, 64), N, 2]
        op::Shape weightScaleExpectShape = {e, Ops::Base::CeilDiv(k, SWIGLU_SPLIT_SIZE), n, SWIGLU_SPLIT_FACTOR};
        // weight转置的shape期望为[E, N, K]
        op::Shape weightTransExpectShape = {e, n, k};
        // weightScale转置的shape期望为[E, N, CeilDiv(K, 64), 2]
        op::Shape weightScaleTransExpectShape = {e, n, Ops::Base::CeilDiv(k, SWIGLU_SPLIT_SIZE), SWIGLU_SPLIT_FACTOR};
        int64_t nAfterHalve = static_cast<int64_t>(n / SWIGLU_SPLIT_FACTOR);
        // output的shape期望为[M, N / 2]
        op::Shape outputExpectShape = {m, nAfterHalve};
        // outputScale的shape期望为[M, CeilDiv(N / 2, 64), 2]
        op::Shape outputScaleExpectShape = {m, Ops::Base::CeilDiv(nAfterHalve, SWIGLU_SPLIT_SIZE), SWIGLU_SPLIT_FACTOR};
        const aclTensor *x = gmmDsqParams_.x;
        const aclTensor *xScale = gmmDsqParams_.xScale;
        const aclTensor *output = gmmDsqParams_.output;
        const aclTensor *outputScale = gmmDsqParams_.outputScale;
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(x, xExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(xScale, xScaleExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, outputExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputScale, outputScaleExpectShape, return false);

        const aclTensor *weightScale = (*gmmDsqParams_.weightScale)[0];
        const aclTensor *weight = (*gmmDsqParams_.weight)[0];
        if (gmmDsqParams_.transposeWeight) {
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weightScale, weightScaleTransExpectShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weight, weightTransExpectShape, return false);
        } else {
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weightScale, weightScaleExpectShape, return false);
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weight, weightExpectShape, return false);
        }
        // 进行swiglu操作需满足n为偶数
        if (n % SWIGLU_N_CONSTRAINT != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Swiglu operation requires n to be even , but n actual value is %lu.", n);
            return false;
        }
        // groupList的长度应等于weight的专家数
        int64_t groupListLen = gmmDsqParams_.groupList->GetViewShape().GetDim(0);
        if (groupListLen != e) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Length of 'groupList' should be equal to the number of experts in weight.");
            return false;
        }
        return true;
    }

    bool CheckPertokenShape()
    {
        int64_t m = gmmDsqParams_.x->GetViewShape().GetDim(0); // 从x的第0维获取m
        int64_t k = gmmDsqParams_.x->GetViewShape().GetDim(1); // 从x的第1维获取k
        // 转置情况下从weight的第1维获取n，非转置情况下从weight的第2维获取n
        int64_t n = gmmDsqParams_.transposeWeight ? ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1) :
                                                    ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(2);
        int64_t e = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(0); // 从weight的第0维获取e

        // x的shape期望为[M, K]
        op::Shape xExpectShape = {m, k};
        // xScale的shape期望为[M]
        op::Shape xScaleExpectShape = {m};
        // weight的shape期望为根据转置的情况来具体确认[E, K, N] 或者[E, N, K]
        op::Shape weightExpectShape = gmmDsqParams_.transposeWeight ? op::Shape{e, n, k} : op::Shape{e, k, n};
        // weightScale的shape期望为[E, N]
        op::Shape weightScaleExpectShape = {e, n};
        int64_t nAfterHalve = static_cast<int64_t>(n / SWIGLU_SPLIT_FACTOR);
        // output的shape期望为[M, N / 2]
        op::Shape outputExpectShape = {m, nAfterHalve};
        // outputScale的shape期望为[M]
        op::Shape outputScaleExpectShape = {m};
        const aclTensor *x = gmmDsqParams_.x;
        const aclTensor *xScale = gmmDsqParams_.xScale;
        const aclTensor *weight = (*gmmDsqParams_.weight)[0];
        const aclTensor *weightScale = (*gmmDsqParams_.weightScale)[0];
        const aclTensor *output = gmmDsqParams_.output;
        const aclTensor *outputScale = gmmDsqParams_.outputScale;
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(x, xExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(xScale, xScaleExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weight, weightExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weightScale, weightScaleExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, outputExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputScale, outputScaleExpectShape, return false);
        return true;
    }

    bool CheckFp8DtypeValid(const aclTensor *x, const aclTensor *xScale, const aclTensor *groupList,
                            const aclTensor *output, const aclTensor *outputScale)
    {
        size_t weightLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < weightLength; i++) {
            const aclTensor *weightScale = (*gmmDsqParams_.weightScale)[i];
            const aclTensor *weight = (*gmmDsqParams_.weight)[i];
            OP_CHECK_DTYPE_NOT_SUPPORT(weight, WEIGHT_DTYPE_SUPPORT_LIST, return false);
            OP_CHECK_DTYPE_NOT_SUPPORT(weightScale, WEIGHT_SCALE_DTYPE_SUPPORT_LIST, return false);
        }
        OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(xScale, X_SCALE_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(groupList, GROUP_LIST_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(outputScale, QUANTSCALEOUT_DTYPE_SUPPORT_LIST, return false);
        DataType outputDtype = gmmDsqParams_.output->GetDataType();
        if (outputDtype != DataType::DT_FLOAT8_E4M3FN && outputDtype != DataType::DT_FLOAT8_E5M2) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "When the dtypes of x and weight inputs are DT_FLOAT8_E4M3FN or "
                "DT_FLOAT8_E5M2, the dtypes of output should be DT_FLOAT8_E4M3FN or DT_FLOAT8_E5M2, but actual value "
                "is %s.",
                op::ToString(outputDtype).GetString());
            return false;
        }
        return true;
    }

    bool CheckFp4DtypeValid(const aclTensor *x, const aclTensor *xScale, const aclTensor *groupList,
                            const aclTensor *output, const aclTensor *outputScale)
    {
        size_t weightLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < weightLength; i++) {
            const aclTensor *weightScale = (*gmmDsqParams_.weightScale)[i];
            const aclTensor *weight = (*gmmDsqParams_.weight)[i];
            OP_CHECK_DTYPE_NOT_SUPPORT(weight, WEIGHT_DTYPE_SUPPORT_LIST_MXFP4, return false);
            OP_CHECK_DTYPE_NOT_SUPPORT(weightScale, WEIGHT_SCALE_DTYPE_SUPPORT_LIST, return false);
        }
        OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_SUPPORT_LIST_MXFP4, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(xScale, X_SCALE_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(groupList, GROUP_LIST_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(output, QUANTOUT_DTYPE_SUPPORT_LIST_MXFP4, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(outputScale, QUANTSCALEOUT_DTYPE_SUPPORT_LIST, return false);
        return true;
    }

    bool CheckPertokenDtypeValid(const aclTensor *x, const aclTensor *xScale, const aclTensor *groupList,
                                 const aclTensor *output, const aclTensor *outputScale)
    {
        OP_CHECK_DTYPE_NOT_SUPPORT(x, XW_DTYPE_SUPPORT_LIST_PERTOKEN, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(xScale, X_SCALE_DTYPE_SUPPORT_LIST_PERTOKEN, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(groupList, GROUP_LIST_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(output, QUANTOUT_DTYPE_SUPPORT_LIST_PERTOKEN, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(outputScale, QUANTSCALEOUT_DTYPE_SUPPORT_LIST_PERTOKEN, return false);
        size_t weightLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < weightLength; i++) {
            const aclTensor *weight = (*gmmDsqParams_.weight)[i];
            const aclTensor *weightScale = (*gmmDsqParams_.weightScale)[i];
            OP_CHECK_DTYPE_NOT_SUPPORT(weight, XW_DTYPE_SUPPORT_LIST_PERTOKEN, return false);
            DataType xDtype = gmmDsqParams_.x->GetDataType();
            if (xDtype == DataType::DT_INT8) {
                OP_CHECK_DTYPE_NOT_SUPPORT(weightScale, WEIGHT_SCALE_DTYPE_SUPPORT_LIST_PERTOKEN_XINT8, return false);
            } else {
                OP_CHECK_DTYPE_NOT_SUPPORT(weightScale, WEIGHT_SCALE_DTYPE_SUPPORT_LIST_PERTOKEN_XFP8HIF8,
                                           return false);
            }
        }
        DataType xDtype = gmmDsqParams_.x->GetDataType();
        return IsDtypeCompatiblePertoken(xDtype, ((*gmmDsqParams_.weight)[0])->GetDataType());
    }

    bool IsDtypeCompatiblePertoken(const DataType a, const DataType b) const
    {
        if ((a == DataType::DT_FLOAT8_E4M3FN || a == DataType::DT_FLOAT8_E5M2) &&
            (b == DataType::DT_FLOAT8_E4M3FN || b == DataType::DT_FLOAT8_E5M2)) {
            return true;
        }
        return a == b;
    }

    bool checkMxfp4InputShape()
    {
        int64_t kValue = gmmDsqParams_.x->GetViewShape().GetDim(1);
        // 转置情况下从weight的第1维获取n，非转置情况下从weight的第2维获取n
        int64_t nValue = gmmDsqParams_.transposeWeight ? ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1) :
                                                         ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(2);
        // mxfp4场景不支持k=2
        if (kValue == MXFP4_K_CONSTRAINT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When the dtypes of x and weight inputs are DT_FLOAT4_E2M1, the K value \
should be greater than 2, but actual value is %lu.",
                    kValue);
            return false;
        }

        // 1：检查K是否为偶数
        int64_t kModValue = kValue % MXFP4_K_CONSTRAINT;
        // 2：检查N是否为偶数
        int64_t nModValue = nValue % MXFP4_N_CONSTRAINT;
        if (kModValue != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When the dtypes of x and weight inputs are DT_FLOAT4_E2M1, the K value \
should be even, but actual value is %lu.",
                    kValue);
            return false;
        }

        // mxfp4场景下，当输出类型为fp4时，N需要满足为大于等于4的偶数
        DataType outputDtype = gmmDsqParams_.output->GetDataType();
        if (outputDtype == DataType::DT_FLOAT4_E2M1) {
            if (!(nValue >= MXFP4_N_CONSTRAINT && nModValue == 0)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "When the output dtype is DT_FLOAT4_E2M1, the N value should be even \
and greater or equal to 4, but actual value is %lu.",
                        nValue);
                return false;
            }
        }

        return true;
    }

    bool CheckEmptyTensor() override
    {
        if (gmmDsqParams_.x->GetViewShape().GetDim(1) <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When the M value is not 0, the K value in x should be positive, but actual value is %ld",
                    gmmDsqParams_.x->GetViewShape().GetDim(1));
            return false;
        }
        auto weightKIndex = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDimNum() - LAST_SECOND_DIM_INDEX;
        if (((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(weightKIndex) <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When the N value is not 0, the K value in weight should be positive, but actual value is %ld",
                    ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(weightKIndex));
            return false;
        }
        return true;
    }

    bool CheckInputOutDims() override
    {
        if (!CheckAttrs()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckAttrs failed.");
            return false;
        }

        if (gmmDsqParams_.quantMode == QUNAT_MODE_MX) {
            return CheckInputOutDimsForMX();
        } else if (gmmDsqParams_.quantMode == QUNAT_MODE_PERTOKEN) {
            return CheckInputOutDimsForPertoken();
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Quant mode %d is not supported. Supported modes are 0 (pertoken) and 2 (MX).",
                    gmmDsqParams_.quantMode);
            return false;
        }
        return true;
    }

    bool CheckInputOutDimsForMX()
    {
        auto xDimNumber = gmmDsqParams_.x->GetViewShape().GetDimNum();
        auto xScaleDimNumber = gmmDsqParams_.xScale->GetViewShape().GetDimNum();
        auto outputDimNumber = gmmDsqParams_.output->GetViewShape().GetDimNum();
        auto outputScaleDimNumber = gmmDsqParams_.outputScale->GetViewShape().GetDimNum();
        if (xDimNumber != MX_X_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of x should be equal 2, current dim is %lu.", xDimNumber);
            return false;
        }
        if (xScaleDimNumber != MX_X_SCALE_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of xScale should be equal 3, current dim is %lu.",
                    xScaleDimNumber);
            return false;
        }
        if (gmmDsqParams_.weight->Size() != SINGLE_TENSOR_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of weight should be 1, current size is %lu.",
                    gmmDsqParams_.weight->Size());
            return false;
        }
        if (gmmDsqParams_.weightScale->Size() != SINGLE_TENSOR_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of weightScale should be 1, current size is %lu.",
                    gmmDsqParams_.weightScale->Size());
            return false;
        }
        if (outputDimNumber != MX_OUTPUT_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of output should be equal 2, current dim is %lu.",
                    outputDimNumber);
            return false;
        }
        if (outputScaleDimNumber != MX_OUTPUT_SCALE_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of outputScale should be equal 3, current dim is %lu.",
                    outputScaleDimNumber);
            return false;
        }
        auto weightDimNumber = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDimNum();
        auto weightScaleDimNumber = ((*gmmDsqParams_.weightScale)[0])->GetViewShape().GetDimNum();
        if (weightScaleDimNumber != MX_WEIGHT_SCALE_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of weightScale should be equal 2, current dim is %lu.",
                    weightScaleDimNumber);
            return false;
        }
        if (weightDimNumber != MX_WEIGHT_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of weight should be equal 3, current dim is %lu.",
                    weightDimNumber);
            return false;
        }
        return true;
    }
    bool CheckInputOutDimsForPertoken()
    {
        auto xDimNumber = gmmDsqParams_.x->GetViewShape().GetDimNum();
        auto xScaleDimNumber = gmmDsqParams_.xScale->GetViewShape().GetDimNum();
        auto outputDimNumber = gmmDsqParams_.output->GetViewShape().GetDimNum();
        auto outputScaleDimNumber = gmmDsqParams_.outputScale->GetViewShape().GetDimNum();
        if (xDimNumber != PERTOKEN_X_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of x should be equal 2, current dim is %lu.", xDimNumber);
            return false;
        }
        if (xScaleDimNumber != PERTOKEN_X_SCALE_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of xScale should be equal 1, current dim is %lu.",
                    xScaleDimNumber);
            return false;
        }
        if (outputDimNumber != PERTOKEN_OUTPUT_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of output should be equal 2, current dim is %lu.",
                    outputDimNumber);
            return false;
        }
        if (outputScaleDimNumber != PERTOKEN_OUTPUT_SCALE_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of outputScale should be equal 1, current dim is %lu.",
                    outputScaleDimNumber);
            return false;
        }
        if (gmmDsqParams_.weight->Size() != SINGLE_TENSOR_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of weight should be 1, current size is %lu.",
                    gmmDsqParams_.weight->Size());
            return false;
        }
        if (gmmDsqParams_.weightScale->Size() != SINGLE_TENSOR_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The size of weightScale should be 1, current size is %lu.",
                    gmmDsqParams_.weightScale->Size());
            return false;
        }
        auto weightDimNumber = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDimNum();
        auto weightScaleDimNumber = ((*gmmDsqParams_.weightScale)[0])->GetViewShape().GetDimNum();
        if (weightScaleDimNumber != PERTOKEN_WEIGHT_SCALE_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of weightScale should be equal 2, current dim is %lu.",
                    weightScaleDimNumber);
            return false;
        }
        if (weightDimNumber != PERTOKEN_WEIGHT_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of weight should be equal 3, current dim is %lu.",
                    weightDimNumber);
            return false;
        }
        return true;
    }

    bool CheckInputOutShape() override
    {
        int64_t groupListLen = gmmDsqParams_.groupList->GetViewShape().GetDim(0);
        if (groupListLen > MAX_GROUP_LIST_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The length of groupList should not be greater than 1024, but actual is %ld.", groupListLen);
            return false;
        }
        // 从x的第1维获取k
        int64_t kInX = gmmDsqParams_.x->GetViewShape().GetDim(1);
        // 根据是否转置从weight中读取维度k
        int64_t kInWeight = gmmDsqParams_.transposeWeight ? ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(2) :
                                                    ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1);
        if (kInX != kInWeight) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Expected input tensor x and weight tensor to have consistent k-dimension, but k=%ld in x, while "
                    "k=%ld in weight.",
                    kInX, kInWeight);
            return false;
        }
        if (gmmDsqParams_.quantMode == QUNAT_MODE_MX) {
            return CheckInputOutShapeForMX();
        } else if (gmmDsqParams_.quantMode == QUNAT_MODE_PERTOKEN) {
            return CheckInputOutShapeForPertoken();
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "Quant mode %d is not supported. Supported modes are 0 (pertoken) and 2 (MX).",
                    gmmDsqParams_.quantMode);
            return false;
        }

        return true;
    }

    bool CheckInputOutShapeForMX()
    {
        if (!CheckMXTranspose()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckMXTranspose failed.");
            return false;
        }
        if (!CheckMXShape()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckMXShape failed.");
            return false;
        }
        DataType xDtype = gmmDsqParams_.x->GetDataType();
        DataType weightDtype = ((*gmmDsqParams_.weight)[0])->GetDataType();
        if (xDtype == DataType::DT_FLOAT4_E2M1 && weightDtype == DataType::DT_FLOAT4_E2M1) {
            return checkMxfp4InputShape();
        }
        return true;
    }

    bool CheckInputOutShapeForPertoken()
    {
        if (!CheckPertokenTranspose()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckPertokenTranspose failed.");
            return false;
        }
        if (!CheckPertokenShape()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckPerTokenShape failed.");
            return false;
        }
        return true;
    }

    bool CheckDtypeValid() override
    {
        DataType xDtype = gmmDsqParams_.x->GetDataType();
        DataType weightDtype = ((*gmmDsqParams_.weight)[0])->GetDataType();
        DataType xScaleDtype = gmmDsqParams_.xScale->GetDataType();
        DataType weightScaleDtype = ((*gmmDsqParams_.weightScale)[0])->GetDataType();
        const aclTensor *x = gmmDsqParams_.x;
        const aclTensor *xScale = gmmDsqParams_.xScale;
        const aclTensor *groupList = gmmDsqParams_.groupList;
        const aclTensor *output = gmmDsqParams_.output;
        const aclTensor *outputScale = gmmDsqParams_.outputScale;
        if(std::find(X_DTYPE_SUPPORT_LIST.begin(), X_DTYPE_SUPPORT_LIST.end(), xDtype) == X_DTYPE_SUPPORT_LIST.end() &&
           std::find(X_DTYPE_SUPPORT_LIST_MXFP4.begin(), X_DTYPE_SUPPORT_LIST_MXFP4.end(), xDtype) == X_DTYPE_SUPPORT_LIST_MXFP4.end() &&
           std::find(XW_DTYPE_SUPPORT_LIST_PERTOKEN.begin(), XW_DTYPE_SUPPORT_LIST_PERTOKEN.end(), xDtype) == XW_DTYPE_SUPPORT_LIST_PERTOKEN.end()){
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Quant case with x dtype %s is not supported; supported types are: INT8, FLOAT8_E4M3FN, "
                        "FLOAT8_E5M2, HIFLOAT8, and FLOAT4_E2M1.", op::ToString(xDtype).GetString());
            return false;
        }
        if(std::find(WEIGHT_DTYPE_SUPPORT_LIST.begin(), WEIGHT_DTYPE_SUPPORT_LIST.end(), weightDtype) == WEIGHT_DTYPE_SUPPORT_LIST.end() &&
           std::find(WEIGHT_DTYPE_SUPPORT_LIST_MXFP4.begin(), WEIGHT_DTYPE_SUPPORT_LIST_MXFP4.end(), weightDtype) == WEIGHT_DTYPE_SUPPORT_LIST_MXFP4.end() &&
           std::find(XW_DTYPE_SUPPORT_LIST_PERTOKEN.begin(), XW_DTYPE_SUPPORT_LIST_PERTOKEN.end(), weightDtype) == XW_DTYPE_SUPPORT_LIST_PERTOKEN.end()){
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Quant case with weight dtype %s is not supported; supported types are: INT8, FLOAT8_E4M3FN, "
                        "FLOAT8_E5M2, HIFLOAT8, and FLOAT4_E2M1.", op::ToString(weightDtype).GetString());
            return false;
        }
        if (gmmDsqParams_.quantMode == QUNAT_MODE_MX &&
            (xDtype == DataType::DT_FLOAT8_E4M3FN || xDtype == DataType::DT_FLOAT8_E5M2) &&
            (weightDtype == DataType::DT_FLOAT8_E4M3FN || weightDtype == DataType::DT_FLOAT8_E5M2)) {
            return CheckFp8DtypeValid(x, xScale, groupList, output, outputScale);
        } else if (gmmDsqParams_.quantMode == QUNAT_MODE_MX && xDtype == DataType::DT_FLOAT4_E2M1 &&
                   weightDtype == DataType::DT_FLOAT4_E2M1) {
            return CheckFp4DtypeValid(x, xScale, groupList, output, outputScale);
        } else if (gmmDsqParams_.quantMode == QUNAT_MODE_PERTOKEN &&
                   std::find(XW_DTYPE_SUPPORT_LIST_PERTOKEN.begin(), XW_DTYPE_SUPPORT_LIST_PERTOKEN.end(), xDtype) !=
                       XW_DTYPE_SUPPORT_LIST_PERTOKEN.end() &&
                   std::find(XW_DTYPE_SUPPORT_LIST_PERTOKEN.begin(), XW_DTYPE_SUPPORT_LIST_PERTOKEN.end(),
                             weightDtype) != XW_DTYPE_SUPPORT_LIST_PERTOKEN.end()) {
            return CheckPertokenDtypeValid(x, xScale, groupList, output, outputScale);
        } else {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "In quantization mode %d, the combination of x dtype %s, weight dtype %s is not supported. "
                "Supported combinations are: "
                "Quantmode 0 (pertoken): (x=int8, weight=int8) or (x=float8_e4m3fn/float8_e5m2, "
                "weight=float8_e4m3fn/float8_e5m2) or (x=hifloat8, weight=hifloat8); "
                "Quantmode 2 (mx): (x=float8_e4m3fn/float8_e5m2, weight=float8_e4m3fn/float8_e5m2) or (x=float4_e2m1, weight=float4_e2m1).",
                gmmDsqParams_.quantMode, op::ToString(xDtype).GetString(), op::ToString(weightDtype).GetString());
            return false;
        }
        return true;
    }

    bool CheckFormat() override
    {
        size_t wLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < wLength; i++) {
            const aclTensor *weightScale = (*gmmDsqParams_.weightScale)[i];
            const aclTensor *weight = (*gmmDsqParams_.weight)[i];
            if (op::IsPrivateFormat(weight->GetStorageFormat())) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of weight should be ND, current format is format is %s.",
                        op::ToString(weight->GetStorageFormat()).GetString());
                return false;
            }
            if (op::IsPrivateFormat(weightScale->GetStorageFormat())) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of weightScale should be ND, current format is format is %s.",
                        op::ToString(weightScale->GetStorageFormat()).GetString());
                return false;
            }
        }

        if (op::IsPrivateFormat(gmmDsqParams_.x->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of x should be ND, current format is format is %s.",
                    op::ToString(gmmDsqParams_.x->GetStorageFormat()).GetString());
            return false;
        }
        if (op::IsPrivateFormat(gmmDsqParams_.xScale->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of xScale should be ND, current format is format is %s.",
                    op::ToString(gmmDsqParams_.xScale->GetStorageFormat()).GetString());
            return false;
        }
        if (op::IsPrivateFormat(gmmDsqParams_.groupList->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of groupList should be ND, current format is format is %s.",
                    op::ToString(gmmDsqParams_.groupList->GetStorageFormat()).GetString());
            return false;
        }
        if (op::IsPrivateFormat(gmmDsqParams_.output->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of output should be ND, current format is format is %s.",
                    op::ToString(gmmDsqParams_.output->GetStorageFormat()).GetString());
            return false;
        }
        if (op::IsPrivateFormat(gmmDsqParams_.outputScale->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of outputScale should be ND, current format is format is %s.",
                    op::ToString(gmmDsqParams_.outputScale->GetStorageFormat()).GetString());
            return false;
        }
        return true;
    }
};
} // namespace gmmSwigluQuantV2
#endif