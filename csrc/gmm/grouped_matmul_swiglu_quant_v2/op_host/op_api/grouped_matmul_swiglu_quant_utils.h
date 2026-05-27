/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_GROUPED_MATMUL_SWIGLU_QUANT_UTILS_H
#define OP_HOST_OP_API_GROUPED_MATMUL_SWIGLU_QUANT_UTILS_H

#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "grouped_matmul_swiglu_quant_v2.h"

namespace gmm_dsq {
using namespace op;
constexpr int64_t OUTPUT_IDX_0 = 0L;
constexpr int64_t OUTPUT_IDX_1 = 1L;
constexpr size_t WEIGHT_NZ_DIM_LIMIT = 5UL;
constexpr size_t WEIGHT_ND_DIM_LIMIT = 3UL;

struct GroupedMatmulSwigluQuantParamsBase {
    const aclTensor *x = nullptr;
    const aclTensorList *weight = nullptr;
    const aclTensorList *weightScale = nullptr;
    const aclTensorList *weightAssistMatrix = nullptr;
    const aclTensor *xScale = nullptr;
    const aclTensor *bias = nullptr;
    const aclTensor *smoothScale = nullptr;
    const aclTensor *groupList = nullptr;
    const aclTensor *output = nullptr;
    const aclTensor *outputScale = nullptr;
    const aclIntArray *tuningConfig = nullptr;
    int64_t dequantMode = 0;
    int64_t dequantDtype = 0;
    int64_t quantMode = 0;
    int64_t quantDtype = 0;
    int64_t groupListType = 0;
    bool transposeWeight = false;
    double swigluLimit=0;
    bool isA8W4 = false;
    bool isA4W4 = false;
};

class GroupedMatmulSwigluQuantParamsBuilder {
public:
    static GroupedMatmulSwigluQuantParamsBuilder Create(const aclTensor *x, const aclTensorList *weight,
        const aclTensorList *weightScale, const aclTensor *output, const aclTensor *outputScale)
    {
        GroupedMatmulSwigluQuantParamsBuilder b;
        b.p_.x = x;
        b.p_.weight = weight;
        b.p_.weightScale = weightScale;
        b.p_.output = output;
        b.p_.outputScale = outputScale;
        return b;
    }

    GroupedMatmulSwigluQuantParamsBuilder &SetWeightAssistMatrix(const aclTensorList *weightAssistMatrix)
    {
        p_.weightAssistMatrix = weightAssistMatrix;
        return *this;
    }

    GroupedMatmulSwigluQuantParamsBuilder &SetXScale(const aclTensor *xScale)
    {
        p_.xScale = xScale;
        return *this;
    }

    GroupedMatmulSwigluQuantParamsBuilder &SetSmoothScale(const aclTensor *smoothScale)
    {
        p_.smoothScale = smoothScale;
        return *this;
    }

    GroupedMatmulSwigluQuantParamsBuilder &SetBias(const aclTensor *bias)
    {
        p_.bias = bias;
        return *this;
    }

    GroupedMatmulSwigluQuantParamsBuilder &SetGroupList(const aclTensor *groupList)
    {
        p_.groupList = groupList;
        return *this;
    }

    GroupedMatmulSwigluQuantParamsBuilder &SetGroupListType(const int64_t groupListType)
    {
        p_.groupListType = groupListType;
        return *this;
    }

    GroupedMatmulSwigluQuantParamsBuilder &SetTuningConfig(const aclIntArray *tuningConfig)
    {
        p_.tuningConfig = tuningConfig;
        return *this;
    }

    GroupedMatmulSwigluQuantParamsBuilder &SetDequantAttr(int64_t dequantMode, int64_t dequantDtype)
    {
        p_.dequantMode = dequantMode;
        p_.dequantDtype = dequantDtype;
        return *this;
    }

    GroupedMatmulSwigluQuantParamsBuilder &SetQuantAttr(int64_t quantMode, int64_t quantDtype)
    {
        p_.quantMode = quantMode;
        p_.quantDtype = quantDtype;
        return *this;
    }

    GroupedMatmulSwigluQuantParamsBuilder &SetTransposeAttr(bool transposeWeight)
    {
        p_.transposeWeight = transposeWeight;
        return *this;
    }
    GroupedMatmulSwigluQuantParamsBuilder &SetLimitAttr(double swigluLimit)
    {
        p_.swigluLimit = swigluLimit;
        return *this;
    }
    GroupedMatmulSwigluQuantParamsBuilder &SetScenario()
    {
        p_.isA8W4 = ((this->p_.x->GetDataType() == DataType::DT_INT8 &&
                    ((*this->p_.weight)[0])->GetDataType() == DataType::DT_INT4) ||
                    (this->p_.x->GetDataType() == DataType::DT_INT8 &&
                    ((*this->p_.weight)[0])->GetDataType() == DataType::DT_INT32));
        p_.isA4W4 = ((this->p_.x->GetDataType() == DataType::DT_INT4 &&
                    ((*this->p_.weight)[0])->GetDataType() == DataType::DT_INT4) ||
                    (this->p_.x->GetDataType() == DataType::DT_INT4 &&
                    ((*this->p_.weight)[0])->GetDataType() == DataType::DT_INT32) ||
                    (this->p_.x->GetDataType() == DataType::DT_INT32 &&
                    ((*this->p_.weight)[0])->GetDataType() == DataType::DT_INT4) ||
                    (this->p_.x->GetDataType() == DataType::DT_INT32 &&
                    ((*this->p_.weight)[0])->GetDataType() == DataType::DT_INT32));
        return *this;
    }

    GroupedMatmulSwigluQuantParamsBase Build() const
    {
        return p_;
    }

private:
    GroupedMatmulSwigluQuantParamsBase p_;
};

class GroupedMatmulSwigluQuantHandler {
public:
    virtual ~GroupedMatmulSwigluQuantHandler() = default;

protected:
    bool CheckTensorListNull(const aclTensorList *&tensors) const
    {
        OP_CHECK_NULL(tensors, return false);
        if (tensors->Size() == 0) {
            return true;
        } else if ((tensors->Size() == 1) && ((*tensors)[0] == nullptr)) {
            return true;
        }

        return false;
    }

    virtual bool CheckNotNull(void)
    {
        OP_CHECK_NULL(gmmDsqParams_.x, return false);
        OP_CHECK_NULL(gmmDsqParams_.weight, return false);
        OP_CHECK_NULL(gmmDsqParams_.weightScale, return false);
        OP_CHECK_NULL(gmmDsqParams_.xScale, return false);
        OP_CHECK_NULL(gmmDsqParams_.groupList, return false);
        OP_CHECK_NULL(gmmDsqParams_.output, return false);
        OP_CHECK_NULL(gmmDsqParams_.outputScale, return false);

        auto ret = CheckTensorListNull(gmmDsqParams_.weight);
        if (ret) {
            return false;
        }

        ret = CheckTensorListNull(gmmDsqParams_.weightScale);
        if (ret) {
            return false;
        }

        if (!gmmDsqParams_.weight || !gmmDsqParams_.weightScale) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
            "The weight or weightScale is nullptr.");
            return false;
        }
        return true;
    }

    virtual bool CheckEmptyTensor(void)
    {
        if ((*gmmDsqParams_.weight)[0]->IsEmpty() || (*gmmDsqParams_.weightScale)[0]->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The weight or weightScale is an empty container.");
            return false;
        }
        return true;
    }

    virtual bool CheckInputOutDims() = 0;
    virtual bool CheckInputOutShape() = 0;
    virtual bool CheckDtypeValid() = 0;
    virtual bool CheckFormat() = 0;

    virtual aclnnStatus CheckParams()
    {
        // 1. 检查参数是否为空指针、空tensor
        CHECK_RET(CheckNotNull(), ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(CheckEmptyTensor(), ACLNN_ERR_PARAM_INVALID);

        // 2. 校验输入、输出参数维度
        CHECK_RET(CheckInputOutDims(), ACLNN_ERR_PARAM_INVALID);

        // 3. 校验输入、输出shape参数
        CHECK_RET(CheckInputOutShape(), ACLNN_ERR_PARAM_INVALID);

        // 4. 检查输入的数据类型是否在支持的数据类型范围之内
        CHECK_RET(CheckDtypeValid(), ACLNN_ERR_PARAM_INVALID);

        // 5. 检查数据形状是否支持
        CHECK_RET(CheckFormat(), ACLNN_ERR_PARAM_INVALID);

        return ACLNN_SUCCESS;
    }

    void CheckOptionalTensorListEmpty(const aclTensorList *&tensorList) const
    {
        if (tensorList == nullptr) {
            return;
        }

        if (tensorList->Size() == 0) {
            tensorList = nullptr;
        } else if (tensorList->Size() == 1) {
            op::Shape shape = (*tensorList)[0]->GetViewShape();
            if (shape.GetDimNum() == 1 && shape.GetDim(0) == 0) {
                tensorList = nullptr;
            }
        }
    }

    void CreateEmptyTensor(const aclDataType dataType, const aclTensorList *&tensorList,
            aclTensorList *&emptyTensorList) const
    {
        if (tensorList != nullptr) {
            return;
        }

        FVector<aclTensor*> emptyTensors;
        aclTensor *emptyTensor = l0Executor_->AllocTensor({0}, static_cast<op::DataType>(dataType));
        emptyTensors.emplace_back(emptyTensor);
        emptyTensorList = l0Executor_->AllocTensorList(emptyTensors.data(), emptyTensors.size());
        tensorList = emptyTensorList;
    }

    aclnnStatus DataContiguous(const aclTensorList *&tensors) const
    {
        std::vector<const aclTensor *> tensorsVec;
        const aclTensor *contiguousTensor = nullptr;
        for (size_t i = 0; i < tensors->Size(); ++i) {
            const aclTensor *tensor = (*tensors)[i];
            contiguousTensor = l0op::Contiguous(tensor, l0Executor_);
            CHECK_RET(contiguousTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
            tensorsVec.push_back(contiguousTensor);
        }
        tensors = l0Executor_->AllocTensorList(tensorsVec.data(), tensorsVec.size());
        return ACLNN_SUCCESS;
    }

    aclnnStatus DataContiguousWeight(const aclTensorList *&tensors) const
    {
        std::vector<const aclTensor *> tensorsVec;
        const aclTensor *contiguousTensor = nullptr;
        for (size_t i = 0; i < tensors->Size(); ++i) {
            const aclTensor *tensor = (*tensors)[i];
            if (!IsPrivateFormat(tensor->GetStorageFormat())) {
                contiguousTensor = l0op::Contiguous(tensor, l0Executor_);
                CHECK_RET(contiguousTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
                tensorsVec.push_back(contiguousTensor);
            } else {
                tensorsVec.push_back(tensor);
            }
        }
        tensors = l0Executor_->AllocTensorList(tensorsVec.data(), tensorsVec.size());
        return ACLNN_SUCCESS;
    }

    virtual aclnnStatus CovertDataContiguous()
    {
        aclTensorList *emptyWeightAssistMatrixList = nullptr;
        CreateEmptyTensor(aclDataType::ACL_FLOAT, gmmDsqParams_.weightAssistMatrix,
            emptyWeightAssistMatrixList);

        CHECK_COND(DataContiguousWeight(gmmDsqParams_.weight) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR,
                    "Contiguous weight failed.");
        CHECK_COND(DataContiguous(gmmDsqParams_.weightScale) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR,
                    "Contiguous weightScale failed.");
        if (gmmDsqParams_.weightAssistMatrix != nullptr && gmmDsqParams_.weightAssistMatrix->Size() != 0) {
            CHECK_COND(DataContiguous(gmmDsqParams_.weightAssistMatrix) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR,
                    "Contiguous weightAssistMatrix failed.");
        }

        gmmDsqParams_.x = l0op::Contiguous(gmmDsqParams_.x, l0Executor_);
        CHECK_COND(gmmDsqParams_.x != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous groupList failed.");
        gmmDsqParams_.xScale = l0op::Contiguous(gmmDsqParams_.xScale, l0Executor_);
        CHECK_COND(gmmDsqParams_.xScale != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous xScale failed.");
        gmmDsqParams_.groupList = l0op::Contiguous(gmmDsqParams_.groupList, l0Executor_);
        CHECK_COND(gmmDsqParams_.groupList != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous groupList failed.");

        return ACLNN_SUCCESS;
    }

public:
    void Initialize(const char *interfaceName, GroupedMatmulSwigluQuantParamsBase &params, uint64_t *workspaceSize, aclOpExecutor **executor)
    {
        interfaceName_ = interfaceName;
        gmmDsqParams_ = params;
        workspaceSize_ = workspaceSize;
        executor_ = executor;
    }

    aclnnStatus Process()
    {
        // 固定写法，创建OpExecutor
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        l0Executor_ = uniqueExecutor.get();

        auto ret = CheckParams();
        CHECK_RET(ret == ACLNN_SUCCESS, ret);

        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            auto x1MDim = gmmDsqParams_.x->GetViewShape().GetDim(0);
            auto x2NIndex = (*gmmDsqParams_.weight)[0]->GetViewShape().GetDimNum() - 1;
            auto x2NDim = (*gmmDsqParams_.weight)[0]->GetViewShape().GetDim(x2NIndex);
            if (x1MDim == 0 || x2NDim == 0) {
                *workspaceSize_ = 0ULL;
                uniqueExecutor.ReleaseTo(executor_);
                return ACLNN_SUCCESS;
            }
        }
        for (size_t i = 0; i < gmmDsqParams_.weight->Size(); i++) {
            auto *w = (*gmmDsqParams_.weight)[i];
            if (IsPrivateFormat(w->GetStorageFormat())) {
                w->SetOriginalShape(w->GetViewShape());
            }
        }
        // 空Tensor场景
        if (gmmDsqParams_.output->IsEmpty() || gmmDsqParams_.groupList->IsEmpty() || gmmDsqParams_.outputScale->IsEmpty()) {
            *workspaceSize_ = 0ULL;
            uniqueExecutor.ReleaseTo(executor_);
            return ACLNN_SUCCESS;
        }

        ret = CovertDataContiguous();
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        auto ret0 = l0op::GroupedMatmulSwigluQuantV2(gmmDsqParams_.x, gmmDsqParams_.weight, gmmDsqParams_.weightScale,
                                                gmmDsqParams_.xScale, gmmDsqParams_.weightAssistMatrix,
                                                gmmDsqParams_.bias,
                                                gmmDsqParams_.smoothScale, gmmDsqParams_.groupList,
                                                gmmDsqParams_.dequantMode, gmmDsqParams_.dequantDtype,
                                                gmmDsqParams_.quantMode, gmmDsqParams_.quantDtype,
                                                gmmDsqParams_.transposeWeight, gmmDsqParams_.groupListType,
                                                gmmDsqParams_.tuningConfig,gmmDsqParams_.swigluLimit, uniqueExecutor.get());
        CHECK_RET(ret0 != std::tuple(nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);

        auto out0 = std::get<OUTPUT_IDX_0>(ret0);
        auto ret1 = l0op::ViewCopy(out0, gmmDsqParams_.output, uniqueExecutor.get());
        CHECK_RET(ret1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto out1 = std::get<OUTPUT_IDX_1>(ret0);
        auto ret2 = l0op::ViewCopy(out1, gmmDsqParams_.outputScale, uniqueExecutor.get());
        CHECK_RET(ret2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *workspaceSize_ = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor_);
        return ACLNN_SUCCESS;
    }

protected:
    string interfaceName_;
    GroupedMatmulSwigluQuantParamsBase gmmDsqParams_;
    uint64_t *workspaceSize_;
    aclOpExecutor **executor_;
    aclOpExecutor *l0Executor_;
};

} // namespace gmm_dsq
#endif