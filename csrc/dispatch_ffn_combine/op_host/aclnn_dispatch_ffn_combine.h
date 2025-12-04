/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_DISPATCH_FFN_COMBINE_
#define OP_API_INC_DISPATCH_FFN_COMBINE_

#include <string>

#include "aclnn/aclnn_base.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现分布式MoE从InitRouting到Unpermute全部算子的融合
 * @brief aclnnDispatchFFNCombine的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] a: matmul左矩阵，数据类型支持：float16, bf16。
 * @param [in] b: matmul右矩阵，数据类型支持：float16, bf16。
 * @param [in] bias: 偏置，数据类型支持：float16, bf16。
 * @param [in] group: 标识通信域名称的字符串。
 * @param [in] worldsize: 通信域size，支持2/4/8卡。
 * @param [in] epRankId: ep本卡Id。取值范围[0, worldSize)，各卡的rankId不能重复
 * @param [out] c: 计算+通信的结果，数据类型：同输入。
 * @param [out] workspaceSize: 返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDispatchFFNCombineGetWorkspaceSize(const aclTensor* x, const aclTensor* weight1, const aclTensor* weight2,
                                                                                        const aclTensor* expertId, const aclTensor* scale1, const aclTensor* scale2,
                                                                                        const aclTensor* probs,
                                                                                        const char* group, int64_t maxOutputSize,
                                                                                        const aclTensor* out,
                                                                                        uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnDispatchGmmCombine的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnDispatchFFNCombineGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDispatchFFNCombine(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GMM_ALLTOALLV_