/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_H
#define OP_API_INC_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmulSwigluQuantWeightNZTensorList的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] x: 表示公式中的x，数据类型支持INT8数据类型，数据格式支持ND。
 * @param [in] weight:
 * 表示公式中的weight，数据类型支持INT8数据类型，数据格式支持NZ。
 * @param [in] weightScale: 表示量化参数，数据类型支持FLOAT16、BFLOAT16、FLOAT32数据类型，数据格式支持ND，支持的最大长度为128个。
 * 表示per Channel参数，数据类型支持FLOAT16，BFLOAT16数据类型，数据格式支持ND。
 * @param [in] xScale:
 * 表示per Token量化参数，数据类型支持FLOAT32数据类型，数据格式支持ND。
 * @param [in] groupList: 必选参数，代表输入和输出分组轴上的索引情况，数据类型支持INT64。
 * @param [out] quantOutput: 表示公式中的out，数据类型支持INT8数据类型，数据格式支持ND。
 * @param [out] quantScaleOutput: 表示公式中的outQuantScale，数据类型支持Float32数据类型。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZTensorListGetWorkspaceSize(
    const aclTensor *x, const aclTensorList *weight, const aclTensor *bias, const aclTensor *offset,
    const aclTensorList *weightScale, const aclTensor *xScale, const aclTensor *groupList,  
    aclTensor *output, aclTensor *outputScale, aclTensor *outputOffset, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnGroupedMatmulSwigluQuantWeightNZTensorList的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulSwigluQuantWeightNZTensorListGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZTensorList(void* workspace,
    uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif