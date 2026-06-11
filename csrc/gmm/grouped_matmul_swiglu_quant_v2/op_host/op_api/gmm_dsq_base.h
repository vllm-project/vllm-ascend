/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_ACLNN_GMM_DSQ_BASE_H
#define OP_HOST_OP_API_ACLNN_GMM_DSQ_BASE_H

#include "grouped_matmul_swiglu_quant_utils.h"

namespace gmm_dsq_base {

using namespace gmm_dsq;

constexpr int64_t SPLIT = 2L;
constexpr int64_t K_LIMIT_A8W8 = 65536L;
constexpr int64_t K_LIMIT_A8W4 = 20000L;
constexpr int64_t N_LIMIT = 10240L;
constexpr int64_t NZ_DIM_4_INT8 = 32L;
constexpr int64_t NZ_DIM_4_INT4 = 64L;
constexpr int64_t NZ_DIM_3 = 16L;
constexpr int64_t OUTPUT_IDX_0 = 0L;
constexpr int64_t OUTPUT_IDX_1 = 1L;
constexpr int64_t DIM_IDX_0 = 0L;
constexpr int64_t DIM_IDX_1 = 1L;
constexpr int64_t DIM_IDX_2 = 2L;
constexpr int64_t DIM_IDX_3 = 4L;
constexpr size_t X_DIM_LIMIT = 2UL;
constexpr size_t MULTI_WEIGHT_NZ_DIM_LIMIT = 4UL;
constexpr size_t MULTI_WEIGHT_ND_DIM_LIMIT = 2UL;
constexpr size_t WEIGHT_SCALE_DIM_LIMIT = 2UL;
constexpr size_t SINGLE_WEIGHT_SCALE_PERGROUP_DIM_LIMIT = 3UL;
constexpr size_t SINGLE_WEIGHT_SCALE_PERCHANNEL_DIM_LIMIT = 2UL;
constexpr size_t MULTI_WEIGHT_SCALE_PERGROUP_DIM_LIMIT = 2UL;
constexpr size_t MULTI_WEIGHT_SCALE_PERCHANNEL_DIM_LIMIT = 1UL;
constexpr size_t TOKEN_SCALE_DIM_LIMIT = 1UL;
constexpr size_t SINGLE_WEIGHT_ASSIST_MATRIX_DIM_LIMIT = 2UL;
constexpr size_t MULTI_WEIGHT_ASSIST_MATRIX_DIM_LIMIT = 1UL;
constexpr size_t GROUP_LIST_DIM_LIMIT = 1UL;
constexpr size_t QUANTOUT_DIM_LIMIT = 2UL;
constexpr size_t QUANTSCALEOUT_DIM_LIMIT = 1UL;
constexpr size_t INT4_PER_INT32 = 8UL;
constexpr size_t NZ_ALIGN_K = 16UL;
constexpr size_t NZ_ALIGN_N = 32UL;
constexpr size_t SMOOTH_SCALE_1D_DIM_LIMIT = 1UL;
constexpr size_t SMOOTH_SCALE_2D_DIM_LIMIT = 2UL;

const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST = {DataType::DT_INT8, DataType::DT_INT4};
const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST = {DataType::DT_INT8, DataType::DT_INT4};
const std::initializer_list<DataType> WEIGHT_SCALE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
const std::initializer_list<DataType> WEIGHT_SCALE_A8W4_DTYPE_SUPPORT_LIST = {DataType::DT_UINT64};
const std::initializer_list<DataType> X_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
const std::initializer_list<DataType> GROUP_LIST_DTYPE_SUPPORT_LIST = {DataType::DT_INT64};
const std::initializer_list<DataType> QUANTOUT_DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
const std::initializer_list<DataType> QUANTSCALEOUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
const std::initializer_list<DataType> WEIGHT_ASSIST_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
const std::initializer_list<DataType> SMOOTH_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};

class GroupedMatmulSwigluQuantBaseHandler : public GroupedMatmulSwigluQuantHandler {
protected:
    bool CheckInputOutDimsA8W8()
    {
        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.x, X_DIM_LIMIT, return false);
        size_t wLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < wLength; i++) {
            const aclTensor* w = (*gmmDsqParams_.weight)[i];
            const aclTensor* wScale = (*gmmDsqParams_.weightScale)[i];
            op::Format wFormat = w->GetViewFormat();
            if (wLength == static_cast<size_t>(1)) { // 单Tensor场景
                if (IsPrivateFormat(wFormat)) {
                    OP_CHECK_WRONG_DIMENSION(w, WEIGHT_NZ_DIM_LIMIT, return false);
                } else {
                    OP_CHECK_WRONG_DIMENSION(w, WEIGHT_ND_DIM_LIMIT, return false);
                }
                OP_CHECK_WRONG_DIMENSION(wScale, WEIGHT_SCALE_DIM_LIMIT, return false);
            } else { // 多Tensor场景
                if (IsPrivateFormat(wFormat)) {
                    OP_CHECK_WRONG_DIMENSION(w, MULTI_WEIGHT_NZ_DIM_LIMIT, return false);
                } else {
                    OP_CHECK_WRONG_DIMENSION(w, MULTI_WEIGHT_ND_DIM_LIMIT, return false);
                }
                OP_CHECK_WRONG_DIMENSION(wScale, MULTI_WEIGHT_SCALE_PERCHANNEL_DIM_LIMIT, return false);
            }
        }

        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.xScale, TOKEN_SCALE_DIM_LIMIT, return false);
        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.groupList, GROUP_LIST_DIM_LIMIT, return false);
        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.output, QUANTOUT_DIM_LIMIT, return false);
        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.outputScale, QUANTSCALEOUT_DIM_LIMIT, return false);
        return true;
    }

    bool CheckInputOutDimsA4W4orA8W4()
    {
        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.x, X_DIM_LIMIT, return false);
        if (gmmDsqParams_.isA4W4 && gmmDsqParams_.weightAssistMatrix != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In the A4W4 scenario, the weightAssistMatrix input must be nullptr.");
            return false;
        } else if (gmmDsqParams_.isA8W4 && gmmDsqParams_.weightAssistMatrix == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In the A8W4 scenario, the weightAssistMatrix input must not be nullptr.");
            return false;
        }
        size_t wLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < wLength; i++) {
            const aclTensor* w = (*gmmDsqParams_.weight)[i];
            const aclTensor* wScale = (*gmmDsqParams_.weightScale)[i];
            op::Format weightViewFormat = w->GetViewFormat();
            bool isSingle = (wLength == 1);
            // 检查权重维度
            OP_CHECK_WRONG_DIMENSION(w,
                (isSingle ?
                    (IsPrivateFormat(weightViewFormat) ? WEIGHT_NZ_DIM_LIMIT : WEIGHT_ND_DIM_LIMIT) :
                    (IsPrivateFormat(weightViewFormat) ? MULTI_WEIGHT_NZ_DIM_LIMIT : MULTI_WEIGHT_ND_DIM_LIMIT)),
                return false);
            // 检查权重Scale维度
            OP_CHECK_WRONG_DIMENSION(wScale,
                (isSingle ?
                    (gmmDsqParams_.dequantMode == 0 ? SINGLE_WEIGHT_SCALE_PERCHANNEL_DIM_LIMIT : SINGLE_WEIGHT_SCALE_PERGROUP_DIM_LIMIT) :
                    (gmmDsqParams_.dequantMode == 0 ? MULTI_WEIGHT_SCALE_PERCHANNEL_DIM_LIMIT : MULTI_WEIGHT_SCALE_PERGROUP_DIM_LIMIT)),
                return false);
            // 检查辅助矩阵（A8W4模式）
            if (gmmDsqParams_.isA8W4) {
                const aclTensor* weightAssistMatrix = (*gmmDsqParams_.weightAssistMatrix)[i];
                OP_CHECK_WRONG_DIMENSION(weightAssistMatrix,
                    (isSingle ? SINGLE_WEIGHT_ASSIST_MATRIX_DIM_LIMIT : MULTI_WEIGHT_ASSIST_MATRIX_DIM_LIMIT),
                    return false);
            }
        }
        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.xScale, TOKEN_SCALE_DIM_LIMIT, return false);
        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.groupList, GROUP_LIST_DIM_LIMIT, return false);
        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.output, QUANTOUT_DIM_LIMIT, return false);
        OP_CHECK_WRONG_DIMENSION(gmmDsqParams_.outputScale, QUANTSCALEOUT_DIM_LIMIT, return false);
        return true;
    }

    bool CheckSingleTensorListTypeA8W8(int64_t e, int64_t k, int64_t n)
    {
        // weight的NDshape期望为[E, K, N]
        op::Shape weightNDExpectShape1 = {e, k, n};
        // 单tesnsor weight的NZshape期望为[E, N // 32, K // 16, 16, 32]
        op::Shape weightNZExpectShape1 = {e, static_cast<int64_t>(n / NZ_DIM_4_INT8), static_cast<int64_t>(k / NZ_DIM_3),
                                        NZ_DIM_3, NZ_DIM_4_INT8};
        // weight的NDshape期望为[K, N]
        op::Shape weightNDExpectShape2 = {k, n};
        // weight的NZshape期望为[N // 32, K // 16, 16, 32]
        op::Shape weightNZExpectShape2 = {static_cast<int64_t>(n / NZ_DIM_4_INT8), static_cast<int64_t>(k / NZ_DIM_3),
                                        NZ_DIM_3, NZ_DIM_4_INT8};

        // weightScale的shape期望为[E, N]
        op::Shape weightScaleExpectShape1 = {e, n};
        op::Shape weightScaleExpectShape2 = {n};

        const aclTensor* w = (*gmmDsqParams_.weight)[0];
        const aclTensor* wScale = (*gmmDsqParams_.weightScale)[0];

        op::Format wFormat = w->GetViewFormat();
        op::Format storageFormat = w->GetStorageFormat();
        if (IsPrivateFormat(wFormat)) {
            if (!(w->GetViewShape() == weightNZExpectShape1 || w->GetViewShape() == weightNZExpectShape2)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected tensor for weight to have same size as %s or %s, but got %s.",
                        op::ToString(weightNZExpectShape1).GetString(),
                        op::ToString(weightNZExpectShape2).GetString(),
                        op::ToString(w->GetViewShape()).GetString());
                return false;
            }
        } else {
            if (!(w->GetViewShape() == weightNDExpectShape1 || w->GetViewShape() == weightNDExpectShape2)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected tensor for weight to have same size as %s or %s, but got %s.",
                        op::ToString(weightNDExpectShape1).GetString(),
                        op::ToString(weightNDExpectShape2).GetString(),
                        op::ToString(w->GetViewShape()).GetString());
                return false;
            }

            if (IsPrivateFormat(storageFormat) && (k % NZ_ALIGN_K != 0 || n % NZ_ALIGN_N != 0)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In W8a8 Nz mode, k should align to 16, n align to 32");
                return false;
            }
        }

        if (!(wScale->GetViewShape() == weightScaleExpectShape1 || wScale->GetViewShape() == weightScaleExpectShape2)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected tensor for weight_scale to have same size as %s or %s, but got %s.",
                    op::ToString(weightScaleExpectShape1).GetString(),
                    op::ToString(weightScaleExpectShape2).GetString(),
                    op::ToString(wScale->GetViewShape()).GetString());
            return false;
        }
        return true;
    }

    bool CheckMultiTensorTypeA8W8(int64_t k, int64_t n)
    {
        // weight的NDshape期望为[K, N]
        op::Shape weightNDExpectShape = {k, n};
        // weight的NZshape期望为[N // 32, K // 16, 16, 32]
        op::Shape weightNZExpectShape = {static_cast<int64_t>(n / NZ_DIM_4_INT8), static_cast<int64_t>(k / NZ_DIM_3),
                                        NZ_DIM_3, NZ_DIM_4_INT8};

        // weightScale的shape期望为[N]
        op::Shape weightScaleExpectShape = {n};
        size_t wLength = gmmDsqParams_.weight->Size();

        for (size_t i = 0; i < wLength; i++) {
            const aclTensor* w = (*gmmDsqParams_.weight)[0];
            const aclTensor* wScale = (*gmmDsqParams_.weightScale)[0];
            op::Format wFormat = w->GetViewFormat();
            op::Format storageFormat = w->GetStorageFormat();
            if (IsPrivateFormat(wFormat)) {
                OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(w, weightNZExpectShape, return false);
            } else {
                OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(w, weightNDExpectShape, return false);
                if (IsPrivateFormat(storageFormat) && (k % NZ_ALIGN_K != 0 || n % NZ_ALIGN_N != 0)) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In W8a8 Nz mode, k should align to 16, n align to 32");
                    return false;
                }
            }
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(wScale, weightScaleExpectShape, return false);
        }

        return true;
    }

    bool CheckTensorListShapeA8W8(int64_t e,int64_t k, int64_t n)
    {
        size_t wLength = gmmDsqParams_.weight->Size();
        if (wLength == static_cast<size_t>(1)) {
            return CheckSingleTensorListTypeA8W8(e, k, n);
        }

        return CheckMultiTensorTypeA8W8(k, n);
    }

    bool CheckInputOutShapeA8W8()
    {
        int64_t m = gmmDsqParams_.x->GetViewShape().GetDim(0);
        int64_t k = gmmDsqParams_.x->GetViewShape().GetDim(1);
        auto n_index = ((*gmmDsqParams_.weightScale)[0])->GetViewShape().GetDimNum() - 1;
        int64_t n = ((*gmmDsqParams_.weightScale)[0])->GetViewShape().GetDim(n_index);
        size_t wLength = gmmDsqParams_.weight->Size();
        int64_t e = wLength;
        if (wLength == static_cast<size_t>(1)) {
            e = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(0);
        }
        if (n % SPLIT != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s, N is %ld , not an even number.", interfaceName_.c_str(), n);
            return false;
        }
        int64_t nAfterHalve = static_cast<int64_t>(n / SPLIT);
        // x的shape期望为[M, K]
        op::Shape xExpectShape = {m, k};
        // xScale的shape期望为[E, N]
        op::Shape xScaleExpectShape = {m};
        // output的shape期望为[M, N / 2]
        op::Shape outputExpectShape = {m, nAfterHalve};
        // outputScale的shape期望为[M]
        op::Shape outputScaleExpectShape = {m};

        auto ret = CheckTensorListShapeA8W8(e, k, n);
        if (!ret) {
            return false;
        }

        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.x, xExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.xScale, xScaleExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.output, outputExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.outputScale, outputScaleExpectShape, return false);
        // groupList的长度应小于等于weight的专家数
        int64_t groupListLen = gmmDsqParams_.groupList->GetViewShape().GetDim(0);
        if (groupListLen > e) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s A8W8, Length of 'groupList' out of range (expected to be in range of [1, "
                    "%ld], but got %ld)", interfaceName_.c_str(),
                    e, groupListLen);
            return false;
        }
        if (n > N_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s A8W8: The current version does not support the scenario that "
                    "N(%ld) is greater than %ld.", interfaceName_.c_str(),
                    n, N_LIMIT);
            return false;
        }
        if (k >= K_LIMIT_A8W8) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s A8W8, The current version does not support the scenario."
                    "The tail axis dimension of input0(x) is %ld, which need lower than %ld.",
                    interfaceName_.c_str(), k, K_LIMIT_A8W8);
            return false;
        }
        if (gmmDsqParams_.smoothScale != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s, smoothScale must be nullptr in A8W8 scenario.", interfaceName_.c_str());
            return false;
        }
        return true;
    }

    bool CheckSingleTensorListTypeA8W4orA4W4(int64_t e, int64_t k, int64_t n)
    {
        // weight的NDshape期望为[E, K, N]
        op::Shape weightNDExpectShape = {e, k, n};
        // 单tesnsor weight的NZshape期望为[E, N // 64, K // 16, 16, 64]
        op::Shape weightNZExpectShape = {e, static_cast<int64_t>(n / NZ_DIM_4_INT4), static_cast<int64_t>(k / NZ_DIM_3),
                                        NZ_DIM_3, NZ_DIM_4_INT4};
        // 单tensor NZ转置
        op::Shape weightNZTransposeExpectShape1 = {e, static_cast<int64_t>(k / NZ_DIM_4_INT4), static_cast<int64_t>(n / NZ_DIM_3),
                                        NZ_DIM_4_INT4, NZ_DIM_3};
        op::Shape weightNZTransposeExpectShape2 = {e, static_cast<int64_t>(k / NZ_DIM_4_INT4), static_cast<int64_t>(n / NZ_DIM_3),
                                        NZ_DIM_3, NZ_DIM_4_INT4};

        // 辅助矩阵的shape期望为[E, N]
        op::Shape weightAssistMatrixExpectShape = {e, n};

        const aclTensor* w = (*gmmDsqParams_.weight)[0];
        const aclTensor* weightAssistMatrix = nullptr;
        if (gmmDsqParams_.weightAssistMatrix != nullptr && (*gmmDsqParams_.weightAssistMatrix)[0] != nullptr) {
            weightAssistMatrix = (*gmmDsqParams_.weightAssistMatrix)[0];
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weightAssistMatrix, weightAssistMatrixExpectShape, return false);
        }
        op::Format weightViewFormat = w->GetViewFormat();
        if (IsPrivateFormat(weightViewFormat)) {
            if (!(w->GetViewShape() == weightNZExpectShape || w->GetViewShape() == weightNZTransposeExpectShape1 || w->GetViewShape() == weightNZTransposeExpectShape2)) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected tensor for weight to have same size as %s %s or %s, but got %s.",
                        op::ToString(weightNZExpectShape).GetString(),
                        op::ToString(weightNZTransposeExpectShape1).GetString(),
                        op::ToString(weightNZTransposeExpectShape2).GetString(),
                        op::ToString(w->GetViewShape()).GetString());
                return false;
            }
        } else {
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(w, weightNDExpectShape, return false);
        }
        return true;
    }

    bool CheckMultiTensorTypeA8W4orA4W4(int64_t k, int64_t n)
    {
        // weight的NDshape期望为[K, N]
        op::Shape weightNDExpectShape = {k, n};
        // weight的NZshape期望为[N // 64, K // 16, 16, 64]
        op::Shape weightNZExpectShape = {static_cast<int64_t>(n / NZ_DIM_4_INT4), static_cast<int64_t>(k / NZ_DIM_3),
                                        NZ_DIM_3, NZ_DIM_4_INT4};

        op::Shape weightAssistMatrixExpectShape = {n};
        size_t wLength = gmmDsqParams_.weight->Size();

        for (size_t i = 0; i < wLength; i++) {
            const aclTensor* w = (*gmmDsqParams_.weight)[i];
            const aclTensor* weightAssistMatrix = nullptr;
            if (gmmDsqParams_.weightAssistMatrix != nullptr && (*gmmDsqParams_.weightAssistMatrix)[i] != nullptr) {
                weightAssistMatrix = (*gmmDsqParams_.weightAssistMatrix)[i];
                OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weightAssistMatrix, weightAssistMatrixExpectShape, return false);
            }
            op::Format weightViewFormat = w->GetViewFormat();
            if (IsPrivateFormat(weightViewFormat)) {
                OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(w, weightNZExpectShape, return false);
            } else {
                OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(w, weightNDExpectShape, return false);
            }
        }

        return true;
    }

    bool CheckSmoothScaleA4W4(int64_t e, int64_t nAfterHalve)
    {
        if (gmmDsqParams_.smoothScale == nullptr) {
            return true;
        }
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.smoothScale, SMOOTH_SCALE_DTYPE_SUPPORT_LIST, return false);
        size_t dimNum = gmmDsqParams_.smoothScale->GetViewShape().GetDimNum();
        if (dimNum == SMOOTH_SCALE_1D_DIM_LIMIT) {
            op::Shape expectShape = {e};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.smoothScale, expectShape, return false);
        } else if (dimNum == SMOOTH_SCALE_2D_DIM_LIMIT) {
            op::Shape expectShape = {e, nAfterHalve};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.smoothScale, expectShape, return false);
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s, smoothScale dimNum should be 1 or 2 in A4W4 scenario, but got %lu.",
                    interfaceName_.c_str(), dimNum);
            return false;
        }
        return true;
    }

    bool CheckTensorListShapeA8W4orA4W4(int64_t e, int64_t k, int64_t n)
    {
        size_t wLength = gmmDsqParams_.weight->Size();
        if (wLength == static_cast<size_t>(1)) {
            return CheckSingleTensorListTypeA8W4orA4W4(e, k, n);
        }

        return CheckMultiTensorTypeA8W4orA4W4(k, n);
    }

    bool CheckInputOutShapeA8W4orA4W4()
    {
        int64_t m = gmmDsqParams_.x->GetViewShape().GetDim(0);
        int64_t k = gmmDsqParams_.x->GetViewShape().GetDim(1);
        int64_t e = 1;
        int64_t n = 1;
        int64_t KGroupCount = 1; // K轴的组数，perchannel场景相当于pergroup场景中的组数为1
        int64_t KGroupSize = k;  // K轴每组的元素个数
        op::Shape weightScaleExpectShape;
        size_t wLength = gmmDsqParams_.weight->Size();
        if (gmmDsqParams_.dequantMode == 0 && wLength == static_cast<size_t>(1)) {
            e = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(0);
            // weightScale入参在perchannel单tensor场景期望shape [E, N]
            n = ((*gmmDsqParams_.weightScale)[0])->GetViewShape().GetDim(DIM_IDX_1);
            weightScaleExpectShape = {e, n}; // 单
        } else if (gmmDsqParams_.dequantMode == 0 && wLength != static_cast<size_t>(1)) {
            e = wLength;
            // weightScale入参在perchannel多tensor场景期望shape [N]
            n = ((*gmmDsqParams_.weightScale)[0])->GetViewShape().GetDim(DIM_IDX_0);
            weightScaleExpectShape = {n}; // 多
        } else if (gmmDsqParams_.dequantMode == 1 && wLength == static_cast<size_t>(1)) {
            e = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(0);
            // weightScale入参在pergroup单tensor场景期望shape [E, KGroupCount, N]
            n = ((*gmmDsqParams_.weightScale)[0])->GetViewShape().GetDim(DIM_IDX_2);
            KGroupCount = ((*gmmDsqParams_.weightScale)[0])->GetViewShape().GetDim(DIM_IDX_1);
            KGroupSize = KGroupCount > 0 ? k / KGroupCount : k;
            weightScaleExpectShape = {e, KGroupCount, n}; // 单
        } else if (gmmDsqParams_.dequantMode == 1 && wLength != static_cast<size_t>(1)) {
            e = wLength;
            // weightScale入参在pergroup多tensor场景期望shape [KGroupCount, N]
            n = ((*gmmDsqParams_.weightScale)[0])->GetViewShape().GetDim(DIM_IDX_1);
            KGroupCount = ((*gmmDsqParams_.weightScale)[0])->GetViewShape().GetDim(DIM_IDX_0);
            KGroupSize = KGroupCount > 0 ? k / KGroupCount : k;
            weightScaleExpectShape = {KGroupCount, n}; // 多
        }
        if (KGroupCount == 0 || k % KGroupCount != 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "%s, "
                "The number of groups along the k-axis is %ld, and the length of the k-axis is %ld, which is illegal. "
                "The number of groups must be greater than 0, and k-axis length %% number of groups == 0 must be true.",
                interfaceName_.c_str(), KGroupCount, k);
            return false;
        }
        if (n % SPLIT != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s, N is %ld , not an even number.", interfaceName_.c_str(), n);
            return false;
        }
        int64_t nAfterHalve = static_cast<int64_t>(n / SPLIT);
        // x的shape期望为[M, K]
        op::Shape xExpectShape = {m, k};
        // xScale的shape期望为[E, N]
        op::Shape xScaleExpectShape = {m};
        // output的shape期望为[M, N / 2]
        op::Shape outputExpectShape = {m, nAfterHalve};
        // outputScale的shape期望为[M]
        op::Shape outputScaleExpectShape = {m};
        auto ret = CheckTensorListShapeA8W4orA4W4(e, k, n);
        if (!ret) {
            return false;
        }

        for (size_t i = 0; i < wLength; i++) {
            const aclTensor* wScale = (*gmmDsqParams_.weightScale)[i];
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(wScale, weightScaleExpectShape, return false);
        }
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.x, xExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.xScale, xScaleExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.output, outputExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.outputScale, outputScaleExpectShape, return false);
        // groupList的长度应小于等于weight的专家数
        int64_t groupListLen = gmmDsqParams_.groupList->GetViewShape().GetDim(0);
        if (groupListLen > e) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s A8W4 or A4W4, Length of 'groupList' out of range (expected to be in range of [1, "
                    "%ld], but got %ld)", interfaceName_.c_str(),
                    e, groupListLen);
            return false;
        }
        if (n > N_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s A8W4 or A4W4: The current version does not support the scenario that "
                    "N(%ld) is greater than %ld.", interfaceName_.c_str(),
                    n, N_LIMIT);
            return false;
        }
        if (k >= K_LIMIT_A8W4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s A8W4 or A4W4, The current version does not support the scenario."
                    "The tail axis dimension of input0(x) is %ld, which need lower than %ld.",
                    interfaceName_.c_str(), k, K_LIMIT_A8W4);
            return false;
        }
        if (gmmDsqParams_.isA4W4) {
            if (!CheckSmoothScaleA4W4(e, nAfterHalve)) {
                return false;
            }
        } else if (gmmDsqParams_.isA8W4 && gmmDsqParams_.smoothScale != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s, smoothScale must be nullptr in A8W4 scenario.", interfaceName_.c_str());
            return false;
        }
        (void)KGroupSize;
        return true;
    }

    bool IsTransposeLastTwoDims(const aclTensor *tensor)
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

    void UnpackInt32ToInt4(const aclTensor *&tensorS32, const std::string &tensorType)
    {
        OP_LOGD("Unpack %s from int32 to int4 start.", tensorType.c_str());
        auto tensorS4 = const_cast<aclTensor *>(tensorS32);
        op::Shape tensorShape = tensorS4->GetViewShape();
        auto viewShapeDim = tensorShape.GetDimNum();
        op::Strides newStride = tensorS4->GetViewStrides();
        bool transposeTensor = false;
        auto changeDimIdx = viewShapeDim - 1;
        // 轴大于等于2才判断是否转置
        if (viewShapeDim >= DIM_IDX_2 && IsTransposeLastTwoDims(tensorS4)) {
            transposeTensor = true;
            changeDimIdx = viewShapeDim - DIM_IDX_2;
        }
        tensorShape[changeDimIdx] = tensorShape.GetDim(changeDimIdx) * INT4_PER_INT32;
        bool isNz = tensorS4->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ;
        tensorS4->SetViewShape(tensorShape);
        tensorS4->SetDataType(DataType::DT_INT4);
        if (isNz){
            OP_LOGD("Reset %s storageShape because tensor is NZ format.", tensorType.c_str());
            auto storageShape = tensorS4->GetStorageShape();
            auto storageShapeDim = storageShape.GetDimNum();
            storageShape[storageShapeDim - 1] *= INT4_PER_INT32;
            tensorS4->SetStorageShape(storageShape);
        }
        if (transposeTensor) {
            OP_LOGD("Reset %s stride because tensor is transposed.", tensorType.c_str());
            auto strideSize = newStride.size();
            // 转置场景，B32承载B4时Strides缩小了8倍，需要调整回来
            newStride[strideSize - 1] *= INT4_PER_INT32;
            for(int64_t batchDim = strideSize - 3; batchDim >= 0; batchDim--) {
                newStride[batchDim] *= INT4_PER_INT32;
            }
            tensorS4->SetViewStrides(newStride);
        }
        OP_LOGD("Unpack %s from int32 to int4 finished.", tensorType.c_str());
    }

    bool CheckInputOutDims() override
    {
        if (gmmDsqParams_.x->GetDataType() == DataType::DT_INT8
                && ((*gmmDsqParams_.weight)[0])->GetDataType() == DataType::DT_INT8) {
            return CheckInputOutDimsA8W8();
        }
        // A8W4或者A4W4场景 INT32为兼容torch_npu考虑，实际计算时，1个INT32数据会被视为8个INT4数据
        if (gmmDsqParams_.isA8W4 || gmmDsqParams_.isA4W4) {
            bool transposeWeight = IsTransposeLastTwoDims((*gmmDsqParams_.weight)[0]);
            gmmDsqParams_.transposeWeight = transposeWeight;
            // 将INT32视为8个Int4数据，调整viewShape和dtype便于后续统一校验
            if (gmmDsqParams_.x->GetDataType() == DataType::DT_INT32) {
                UnpackInt32ToInt4(gmmDsqParams_.x, "x");
            }
            if (((*gmmDsqParams_.weight)[0])->GetDataType() == DataType::DT_INT32) {
                size_t wLength = gmmDsqParams_.weight->Size();
                for (size_t i = 0; i < wLength; i++) {
                    const aclTensor *w = (*gmmDsqParams_.weight)[i];
                    UnpackInt32ToInt4(w, "weight");
                }
            }

            if (transposeWeight == true){
                const aclTensor* w = (*gmmDsqParams_.weight)[0];
                bool isNZ = w->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ;
                if (!isNZ) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                            "In weight Transpose scenario.weight Format expect is FRACTAL_NZ when weight is transposed, but got [%s].",
                            op::ToString(w->GetStorageFormat()).GetString());
                    return false;
                }
                if (!gmmDsqParams_.isA4W4) {
                    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                            "In weight Transpose scenario, only A4W4 is supported.");
                    return false;
                }
            }
            if (((*gmmDsqParams_.weightScale)[0])->GetDataType() == DataType::DT_INT64) {
                size_t weightScaleLength = gmmDsqParams_.weightScale->Size();
                for (size_t i = 0; i < weightScaleLength; i++) {
                    auto weightScale_fix = const_cast<aclTensor *>((*gmmDsqParams_.weightScale)[i]);
                    weightScale_fix->SetDataType(DataType::DT_UINT64);
                }
            }
            return CheckInputOutDimsA4W4orA8W4();
        }
        return false;
    }

    bool CheckInputOutShape() override
    {
        if (gmmDsqParams_.x->GetDataType() == DataType::DT_INT8
                && ((*gmmDsqParams_.weight)[0])->GetDataType() == DataType::DT_INT8) {
            return CheckInputOutShapeA8W8();
        }
        // A8W4场景或A4W4场景
        if (gmmDsqParams_.isA8W4 || gmmDsqParams_.isA4W4) {
            return CheckInputOutShapeA8W4orA4W4();
        }
        return false;
    }

    bool CheckDtypeValid() override
    {
        size_t wLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < wLength; i++) {
            const aclTensor* wScale = (*gmmDsqParams_.weightScale)[i];
            const aclTensor* w = (*gmmDsqParams_.weight)[i];

            OP_CHECK_DTYPE_NOT_SUPPORT(w, WEIGHT_DTYPE_SUPPORT_LIST, return false);

            if (w->GetDataType() == DataType::DT_INT4) {
                OP_CHECK_DTYPE_NOT_SUPPORT(wScale, WEIGHT_SCALE_A8W4_DTYPE_SUPPORT_LIST, return false);
                if (gmmDsqParams_.weightAssistMatrix != nullptr && (*gmmDsqParams_.weightAssistMatrix)[i] != nullptr) {
                    const aclTensor* weightAssistMatrix = (*gmmDsqParams_.weightAssistMatrix)[i];
                    OP_CHECK_DTYPE_NOT_SUPPORT(weightAssistMatrix, WEIGHT_ASSIST_DTYPE_SUPPORT_LIST, return false);
                }
            } else if (w->GetDataType() == DataType::DT_INT8) {
                OP_CHECK_DTYPE_NOT_SUPPORT(wScale, WEIGHT_SCALE_DTYPE_SUPPORT_LIST, return false);
            }
        }
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.x, X_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.xScale, X_SCALE_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.groupList, GROUP_LIST_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.output, QUANTOUT_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(gmmDsqParams_.outputScale, QUANTSCALEOUT_DTYPE_SUPPORT_LIST, return false);

        return true;
    }

    bool CheckFormat() override
    {
        const aclTensor* w = (*gmmDsqParams_.weight)[0];
        bool isNZ = w->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ;
        if ((gmmDsqParams_.x->GetDataType() == DataType::DT_INT8 && w->GetDataType() == DataType::DT_INT8) && !isNZ) {
            // fp16 in fp32 out that is split k template, not precision-advanced now
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s, The current version does not support the scenario."
                    "weight Format expect is FRACTAL_NZ, but got [%s].", interfaceName_.c_str(),
                    op::ToString(w->GetStorageFormat()).GetString());
            return false;
        }
        if (IsPrivateFormat(gmmDsqParams_.x->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s, The current version does not support the scenario."
                    "x Format Not support Private Format.", interfaceName_.c_str());
            return false;
        }
        if (IsPrivateFormat(gmmDsqParams_.output->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "%s, The current version does not support the scenario."
                    "output Format Not support Private Format.", interfaceName_.c_str());
            return false;
        }
        return true;
    }
};
}
#endif