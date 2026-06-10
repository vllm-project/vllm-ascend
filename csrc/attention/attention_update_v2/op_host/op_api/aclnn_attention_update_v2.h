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
 * \file aclnn_attention_update_v2.h
 * \brief
 */

#ifndef OP_API_INC_ATTENTION_UPDATE_V2_H_
#define OP_API_INC_ATTENTION_UPDATE_V2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAttentionUpdateV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * @param [in] lse: npu
 * device侧的aclTensorList, 数据类型支持FLOAT32,数据格式支持ND。
 * @param [in] localOut: npu
 * device侧的aclTensorList, 数据类型支持FLOAT32，FLOAT16，BFLOAT16,数据格式支持ND。
 * @param [in] updateType：表示AttentionUpdateV2类型。
 * int64_t, 控制lseOut是否输出，支持0,1，分别表示不输出lseOut，输出lseOut。
 * @param [out] out: npu
 * device侧的aclTensor, 数据类型支持FLOAT32，FLOAT16，BFLOAT16，数据格式支持ND。
 * @param [out] lseOut: npu
 * device侧的aclTensor, 作为lse_m可选输出，数据类型支持FLOAT32。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小.
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAttentionUpdateV2GetWorkspaceSize(
    const aclTensor* lse, const aclTensor* localOut, int64_t updateType,
    aclTensor* out, aclTensor* lseOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/* @brief aclnnAttentionUpdateV2的第二段接口，用于执行计算.
 * @param [in] workspace: 在npu device侧申请的workspace内存起址
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnBitwiseNotGetWorkspaceSize获取.
 * @param [in] executor: op执行器，包含了算子计算流程.
 * @param [in] stream: acl stream流.
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAttentionUpdateV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif