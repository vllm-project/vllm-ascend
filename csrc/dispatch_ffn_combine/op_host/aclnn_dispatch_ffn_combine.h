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
 * Operator function: fuse all distributed MoE ops from InitRouting through Unpermute.
 * @brief First-stage interface of aclnnDispatchFFNCombine that calculates workspace size based on the specific compute flow.
 * @domain aclnn_ops_infer
 * @param [in] x: The input tensor.
 * @param [in] weight1: The first weight tensor.
 * @param [in] weight2: The second weight tensor.
 * @param [in] expertId: The expert ID tensor.
 * @param [in] scale1: The first scale tensor.
 * @param [in] scale2: The second scale tensor.
 * @param [in] probs: The probabilities tensor.
 * @param [in] group: string identifying the communication domain name.
 * @param [in] maxOutputSize: The maximum output size.
 * @param [out] out: result of computation + communication; same dtype as input.
 * @param [out] workspaceSize: workspace size to allocate on the NPU device side.
 * @param [out] executor: op executor containing the operator compute flow.
 * @return aclnnStatus: status code.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDispatchFFNCombineGetWorkspaceSize(const aclTensor* x, const aclTensorList* weight1, const aclTensorList* weight2,
                                                                                        const aclTensor* expertId, const aclTensorList* scale1, const aclTensorList* scale2,
                                                                                        const aclTensor* probs,
                                                                                        const char* group, int64_t maxOutputSize,
                                                                                        const aclTensor* out,
                                                                                        uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief Second-stage interface of aclnnDispatchFFNCombine to execute computation.
 * @param [in] workspace: workspace memory address allocated on the NPU device side.
 * @param [in] workspace_size: workspace size allocated on the NPU device side, obtained from aclnnDispatchFFNCombineGetWorkspaceSize.
 * @param [in] executor: op executor containing the operator compute flow.
 * @param [in] stream: acl stream.
 * @return aclnnStatus: status code.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDispatchFFNCombine(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_DISPATCH_FFN_COMBINE_
