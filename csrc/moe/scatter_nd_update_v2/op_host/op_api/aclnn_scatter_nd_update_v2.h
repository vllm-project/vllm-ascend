/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_SCATTER_ND_UPDATE_V2_H_
#define OP_API_INC_SCATTER_ND_UPDATE_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnScatterNdUpdateV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnnop_ops_infer
 * @domain aclnnop_ops_train
 * 算子功能： 将tensor updates中的值按指定的索引indices逐个更新tensor var中的值。
 * @param [in] varRef: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, BOOL
 * INT64，BFLOAT16，支持非连续的Tensor，数据格式支持ND。
 * @param [in] indices: npu device侧的aclTensor，数据类型支持INT32, INT64类型。支持非连续的Tensor，数据格式支持ND。
 * @param [in] updates: npu device侧的aclTensor，数据类型支持FLOAT16, FLOAT32, BOOL
 * INT64，BFLOAT16，支持非连续的Tensor，数据格式支持ND,
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnScatterNdUpdateV2GetWorkspaceSize(aclTensor* varRef, const aclTensor* indices,
                                                           const aclTensor* updates, const aclIntArray* strides, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor);

/**
 * @brief: aclnnScatterNdUpdateV2的第二段接口，用于执行计算
 * @domain aclnnop_ops_infer
 * @domain aclnnop_ops_train
 * 算子功能： 将tensor updates中的值按指定的索引indices逐个更新tensor var中的值。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnScatterNdUpdateV2GetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnScatterNdUpdateV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SCATTER_ND_UPDATE_V2_H_