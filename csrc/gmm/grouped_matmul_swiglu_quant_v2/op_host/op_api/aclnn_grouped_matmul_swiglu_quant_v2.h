/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_QUANT_V2_H
#define OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SWIGLU_QUANT_V2_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmulSwigluQuantV2 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: 表示公式中的x，数据类型支持INT8、FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8数据类型，数据格式支持ND。
 * @param [in] weight:
 * 表示公式中的weight，数据类型支持INT4、FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2、INT8、HIFLOAT8数据类型，数据格式支持ND。
 * @param [in] weightScale:
 * 表示量化参数，数据类型支持UINT64、FLOAT32、FLOAT8_E8M0、BF16、FLOAT16数据类型，数据格式支持ND。
 * @param [in] weightAssistMatrix:
 * 表示weight辅助矩阵，数据类型支持FLOAT32数据类型。
 * @param [in] bias:
 * 表示偏移，数据类型支持FLOAT32数据类型，数据格式支持ND。
 * @param [in] xScale:
 * 表示perToken量化参数，数据类型支持FLOAT8_E8M0、FLOAT32数据类型，数据格式支持ND。
 * @param [in] smoothScale:
 * 左矩阵的的量化因子，数据类型支持FLOAT32数据类型，数据格式支持ND。
 * @param [in] groupList: 必选参数，表示每个分组参与计算的Token个数，数据类型支持INT64。
 * @param [in] dequantMode: 表示反量化计算类型，用于确定激活矩阵与权重矩阵的反量化方式。
 * @param [in] dequantDtype: 表示中间GroupedMatmul的结果数据类型。
 * @param [in] quantMode: 表示量化计算类型，用于确定swiglu结果的量化模式。
 * @param [in] groupListType: 表示指定分组的解释方式，用于确定groupList的语义。
 * @param [in] tuningConfig: 用于算子预估m/e的大小，走不同的算子模板，以适配不不同场景性能要求。
 * @param [in] swigluLimit: clamp。
 * @param [out] quantOutput: 表示公式中的out，数据类型支持INT8、FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8数据类型，数据格式支持ND。
 * @param [out] quantScaleOutput: 表示公式中的outQuantScale，数据类型支持FLOAT32、FLOAT8_E8M0数据类型。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
aclnnStatus aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(const aclTensor *x,
        const aclTensorList *weight, const aclTensorList *weightScale,
        const aclTensorList *weightAssistMatrix, const aclTensor *bias,
        const aclTensor *xScale, const aclTensor *smoothScale,
        const aclTensor *groupList,  int64_t dequantMode,
        int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
        const aclIntArray *tuningConfigOptional, double swigluLimit,
        aclTensor *output, aclTensor *outputScale,
        uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnGroupedMatmulSwigluQuantV2的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulSwigluQuantV2(void *workspace,
                                                                                   uint64_t workspaceSize,
                                                                                   aclOpExecutor *executor,
                                                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif