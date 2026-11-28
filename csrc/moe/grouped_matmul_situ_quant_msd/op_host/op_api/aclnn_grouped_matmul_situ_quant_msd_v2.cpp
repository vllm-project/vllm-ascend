/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_grouped_matmul_situ_quant_msd_v2.cpp
 * \brief GroupedMatmulSituQuantMsd 的 A3 int32 打包入口。
 *
 * 只补一件事：torch 侧 4bit 权重按 int32 承载（1 个 int32 装 8 个 int4），而 op 原型
 * 按 INT4 声明 weight、tiling 也按逻辑 INT4 形状取 N，因此这里把描述符改写成语义上
 * 的 INT4 张量（view 末维 x8、dtype=INT4；FRACTAL_NZ 排布同时改写 storage 末维），
 * weightScale 的 int64 同理改写为 uint64。改写完成后直接转给 op 原型自动生成的
 * aclnnGroupedMatmulSituQuantMsd 入口。
 *
 * 该改写手法与仓内其它 A3 int4 分组矩阵算子的 op_api 一致
 * （见 csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/gmm_dsq_base.h）。
 */
#include "aclnn_grouped_matmul_situ_quant_msd_v2.h"
#include "grouped_matmul_situ_quant_packed_tensor.h"

#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

// op 原型自动生成的同名入口：本文件只做描述符改写，其余全部转交给它。
// 注意 beta / linearBeta 在生成原型里是 double（Float 属性统一按 double 导出），此处声明不得改成 float，
// 否则 extern "C" 下 ABI 不一致，float 的位模式会被当成 double 解读（4.0f 读出来约 2e-314，打印即 0）。
extern "C" aclnnStatus aclnnGroupedMatmulSituQuantMsdGetWorkspaceSize(
    const aclTensor *x, const aclTensor *xScale, const aclTensorList *weight, const aclTensorList *weightScale,
    const aclTensor *groupList, const aclTensorList *weightAssistMatrix, const aclTensor *bias,
    const aclTensor *smoothScale, int64_t groupListType, double beta, double linearBeta, int64_t dequantMode,
    int64_t dequantDtype, int64_t quantMode, int64_t quantDtype, bool transposeWeight,
    const aclIntArray *tuningConfigOptional, bool activateLeft, aclTensor *output, aclTensor *outputScale,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnGroupedMatmulSituQuantMsd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                   aclrtStream stream);

namespace {

void UnpackWeightList(const aclTensorList *weight)
{
    if (weight == nullptr) {
        return;
    }
    for (size_t i = 0; i < weight->Size(); ++i) {
        situ_quant::UnpackInt32ToInt4((*weight)[i]);
    }
}

void CastScaleListToUint64(const aclTensorList *weightScale)
{
    if (weightScale == nullptr) {
        return;
    }
    for (size_t i = 0; i < weightScale->Size(); ++i) {
        situ_quant::CastInt64ScaleToUint64((*weightScale)[i]);
    }
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGroupedMatmulSituQuantMsdV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *xScale, const aclTensorList *weight, const aclTensorList *weightScale,
    const aclTensor *groupList, const aclTensorList *weightAssistMatrix, const aclTensor *bias,
    const aclTensor *smoothScale, int64_t groupListType, double beta, double linearBeta, int64_t dequantMode,
    int64_t dequantDtype, int64_t quantMode, int64_t quantDtype, bool transposeWeight,
    const aclIntArray *tuningConfigOptional, bool activateLeft, aclTensor *output, aclTensor *outputScale,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "workspaceSize or executor is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    // 形状 / 维数仍交给后续入口按 INT4 语义校验，这里只做 4bit 打包形态的归一化。
    UnpackWeightList(weight);
    CastScaleListToUint64(weightScale);

    return aclnnGroupedMatmulSituQuantMsdGetWorkspaceSize(
        x, xScale, weight, weightScale, groupList, weightAssistMatrix, bias, smoothScale, groupListType, beta,
        linearBeta, dequantMode, dequantDtype, quantMode, quantDtype, transposeWeight, tuningConfigOptional,
        activateLeft, output, outputScale, workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmulSituQuantMsdV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    return aclnnGroupedMatmulSituQuantMsd(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
