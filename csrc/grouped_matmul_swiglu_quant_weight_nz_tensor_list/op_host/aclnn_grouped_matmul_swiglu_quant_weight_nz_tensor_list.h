/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_H
#define OP_API_INC_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnGroupedMatmulSwigluQuantWeightNzTensorList, which calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] x: Represents x in the formula. The data type supports INT8, and the data format supports ND.
 * @param [in] weight:
 * Represents weight in the formula. The data type supports INT8, and the data format supports NZ.
 * @param [in] weightScale: Represents quantization parameters. The data type supports FLOAT16, BFLOAT16, and FLOAT32. The data format supports ND, with a maximum length of 128.
 * Represents per Channel parameters. The data type supports FLOAT16 and BFLOAT16. The data format supports ND.
 * @param [in] xScale:
 * Represents per Token quantization parameters. The data type supports FLOAT32, and the data format supports ND.
 * @param [in] groupList: Required parameter, representing the index situation on the input and output grouping axes. The data type supports INT64.
 * @param [out] quantOutput: Represents out in the formula. The data type supports INT8, and the data format supports ND.
 * @param [out] quantScaleOutput: Represents outQuantScale in the formula. The data type supports Float32.
 * @param [out] workspaceSize: Returns the workspace size that users need to apply for on the npu device side.
 * @param [out] executor: Returns the op executor, containing the operator calculation process.
 * @return aclnnStatus: Returns the status code.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzTensorListGetWorkspaceSize(
    const aclTensor *x, const aclTensorList *weight, const aclTensor *bias, const aclTensor *offset,
    const aclTensorList *weightScale, const aclTensor *xScale, const aclTensor *groupList,  
    aclTensor *output, aclTensor *outputScale, aclTensor *outputOffset, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnGroupedMatmulSwigluQuantWeightNzTensorList, used to execute calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the npu device side.
 * @param [in] workspaceSize: The workspace size applied for on the npu device side, obtained from the first interface aclnnGroupedMatmulSwigluQuantWeightNzTensorListGetWorkspaceSize.
 * @param [in] stream: acl stream.
 * @param [in] executor: op executor, containing the operator calculation process.
 * @return aclnnStatus: Returns the status code.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzTensorList(void* workspace,
    uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
