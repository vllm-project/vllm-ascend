/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <new>
#include <memory>
#include <unordered_map>
#include "gmm_dsq_base.h"
#include "grouped_matmul_swiglu_quant_v2_utils.h"
#include "grouped_matmul_swiglu_quant_v2.h"
#include "aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"
#include "aclnn_grouped_matmul_swiglu_quant_v2.h"

using namespace op;
using namespace gmm_dsq;
using namespace gmm_dsq_base;

class GmmDsqHandlerFactory {
private:
    std::unordered_map<NpuArch, std::unique_ptr<GroupedMatmulSwigluQuantHandler>> handlers_;

public:
    void registerHandler(NpuArch npuArch, std::unique_ptr<GroupedMatmulSwigluQuantHandler> handler)
    {
        handlers_[npuArch] = std::move(handler);
    }

    GroupedMatmulSwigluQuantHandler *getHandler(NpuArch npuArch)
    {
        auto it = handlers_.find(npuArch);
        return it != handlers_.end() ? it->second.get() : nullptr;
    }
};

static aclnnStatus aclnnGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(const char* interfaceName,
    GroupedMatmulSwigluQuantParamsBase &params, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    GmmDsqHandlerFactory factory;
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    factory.registerHandler(NpuArch::DAV_2201,
        std::make_unique<gmm_dsq_base::GroupedMatmulSwigluQuantBaseHandler>());
    factory.registerHandler(NpuArch::DAV_3510,
        std::make_unique<gmmSwigluQuantV2::GroupedMatmulSwigluQuantBaseHandler>());

    if (auto *handler = factory.getHandler(npuArch)) {
        handler->Initialize(interfaceName, params, workspaceSize, executor);
        return handler->Process();
    } else {
         OP_LOGE(ACLNN_ERR_PARAM_INVALID, "interfaceName failed: the soc version is not support");
    }

    return ACLNN_ERR_PARAM_INVALID;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(const aclTensor *x,
        const aclTensorList *weight, const aclTensorList *weightScale,
        const aclTensorList *weightAssistMatrix, const aclTensor *bias,
        const aclTensor *xScale, const aclTensor *smoothScale,
        const aclTensor *groupList, int64_t dequantMode,
        int64_t dequantDtype, int64_t quantMode,
        int64_t groupListType, const aclIntArray *tuningConfigOptional,  double swigluLimit,
        aclTensor *output, aclTensor *outputScale,
        uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnGroupedMatmulSwigluQuantV2,
                   DFX_IN(x, weight, weightScale, xScale, groupList),
                   DFX_OUT(output, outputScale));
    CHECK_COND((output != nullptr), ACLNN_ERR_PARAM_INVALID,
               "Expected a proper Tensor but got null for argument output.");

    GroupedMatmulSwigluQuantParamsBase params =
        GroupedMatmulSwigluQuantParamsBuilder::Create(x, weight, weightScale, output, outputScale)
        .SetXScale(xScale).SetSmoothScale(smoothScale)
        .SetGroupList(groupList).SetGroupListType(groupListType)
        .SetWeightAssistMatrix(weightAssistMatrix)
        .SetDequantAttr(dequantMode, dequantDtype)
        .SetQuantAttr(quantMode, static_cast<int64_t> (output->GetDataType()))
        .SetTransposeAttr(false).SetBias(bias)
        .SetLimitAttr(swigluLimit)
        .SetScenario()
        .SetTuningConfig(tuningConfigOptional).Build();
    // 调用公共接口
    return aclnnGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(__FUNCTION__, params, workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(const aclTensor *x,
        const aclTensorList *weight, const aclTensorList *weightScale,
        const aclTensorList *weightAssistMatrix, const aclTensor *bias,
        const aclTensor *xScale, const aclTensor *smoothScale,
        const aclTensor *groupList, int64_t dequantMode,
        int64_t dequantDtype, int64_t quantMode,
        int64_t groupListType,  const aclIntArray *tuningConfigOptional, double swigluLimit,
        aclTensor *output, aclTensor *outputScale,
        uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnGroupedMatmulSwigluQuantWeightNzV2,
                   DFX_IN(x, weight, weightScale, xScale, groupList),
                   DFX_OUT(output, outputScale));
    // weight在该场景下强制绑定StorageFormat 和 ViewFormat 为NZ
    CHECK_RET(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    size_t wLength = weight->Size();
    if (wLength == 1) {
        // 单Tensor场景
        auto w = (*weight)[0];
        auto storgeShape = w->GetStorageShape();
        auto viewShape = w->GetViewShape();
        aclTensor *weightNZ = const_cast<aclTensor *>(w);
        auto storageShape = w->GetStorageShape();
        auto groupListViewShape = groupList->GetViewShape();
        auto expertNum = groupListViewShape[0];
        auto weightScale0 = (*weightScale)[0];
        auto weightScaleStorageShape = weightScale0->GetViewShape();
        auto n = weightScaleStorageShape[1];
        auto xViewShape = x->GetViewShape();
        auto k = xViewShape[1];
        storageShape = {expertNum, n / 64, k / 16, 16, 8};
        w->SetStorageShape(storageShape);
        CHECK_COND((storgeShape.GetDimNum() == WEIGHT_NZ_DIM_LIMIT), ACLNN_ERR_PARAM_INVALID,
                   "aclnnGroupedMatmulSwigluQuantWeightNzV2, The dimnum of storageShape for second input (weight)"
                 "must be 5. \n But StorageShape got %s , and dimNum is %lu.",
                   op::ToString(storgeShape).GetString(), storgeShape.GetDimNum());
        // weight的StorageFormat无条件视为NZ
        weightNZ->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
        if (viewShape.GetDimNum() == WEIGHT_NZ_DIM_LIMIT) {
            // 若weight的viewShape为5维则视为NZ
            weightNZ->SetViewFormat(op::Format::FORMAT_FRACTAL_NZ);
        } else if (viewShape.GetDimNum() == WEIGHT_ND_DIM_LIMIT) {
            // 若weight的viewShape为3维则视为ND
            weightNZ->SetViewFormat(op::Format::FORMAT_ND);
        }
    } else {
        // 多Tensor场景
        for (size_t i = 0; i < wLength; i++) {
            auto w = (*weight)[i];
            auto storgeShape = w->GetStorageShape();
            auto viewShape = w->GetViewShape();
            aclTensor *weightNZ = const_cast<aclTensor *>(w);
            auto storageShape = w->GetStorageShape();
            auto groupListViewShape = groupList->GetViewShape();
            auto weightScale0 = (*weightScale)[i];
            auto weightScaleStorageShape = weightScale0->GetViewShape();
            auto n = weightScaleStorageShape[0];
            auto xViewShape = x->GetViewShape();
            auto k = xViewShape[1];
            storageShape = {n / 64, k / 16, 16, 8};
            w->SetStorageShape(storageShape);
            CHECK_COND((storgeShape.GetDimNum() == MULTI_WEIGHT_NZ_DIM_LIMIT), ACLNN_ERR_PARAM_INVALID,
                       "aclnnGroupedMatmulSwigluQuantWeightNzV2, The dimnum of storageShape for second input (weight)"
                     "must be 4. \n But StorageShape got %s , and dimNum is %lu.",
                       op::ToString(storgeShape).GetString(), storgeShape.GetDimNum());
            // weight的StorageFormat无条件视为NZ
            weightNZ->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
            if (viewShape.GetDimNum() == MULTI_WEIGHT_NZ_DIM_LIMIT) {
                // 若weight的viewShape为4维则视为NZ
                weightNZ->SetViewFormat(op::Format::FORMAT_FRACTAL_NZ);
            } else if (viewShape.GetDimNum() == MULTI_WEIGHT_ND_DIM_LIMIT) {
                // 若weight的viewShape为2维则视为ND
                weightNZ->SetViewFormat(op::Format::FORMAT_ND);
            }
        }
    }
    GroupedMatmulSwigluQuantParamsBase params =
        GroupedMatmulSwigluQuantParamsBuilder::Create(x, weight, weightScale, output, outputScale)
        .SetXScale(xScale).SetSmoothScale(smoothScale)
        .SetGroupList(groupList).SetGroupListType(groupListType)
        .SetWeightAssistMatrix(weightAssistMatrix)
        .SetDequantAttr(dequantMode, dequantDtype)
        .SetLimitAttr(swigluLimit)
        .SetScenario()
        .SetTuningConfig(tuningConfigOptional).Build();
    // 调用公共接口
    return aclnnGroupedMatmulSwigluQuantGetWorkspaceSizeCommon(__FUNCTION__, params, workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmulSwigluQuantV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulSwigluQuantV2);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in GroupedMatmulSwigluQuantV2 launch aicore");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulSwigluQuantWeightNzV2);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in GroupedMatmulSwigluQuantWeightNzV2 launch aicore");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif