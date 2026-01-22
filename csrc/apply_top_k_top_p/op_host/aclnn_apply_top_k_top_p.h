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
 * \file aclnn_apply_top_k_top_p.h
 * \brief Custom implementation of ApplyTopKTopP operator to avoid name conflict with CANN built-in operator.
 */
#ifndef OP_API_INC_APPLY_TOP_K_TOP_P_CUSTOM_H_
#define OP_API_INC_APPLY_TOP_K_TOP_P_CUSTOM_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnApplyTopKTopPCustom first phase interface, calculates workspace size based on the computation flow.
 * @domain aclnn_ops_infer
 * @param [in] logits: aclTensor on NPU device, supports FLOAT, FLOAT16, BFLOAT16, supports non-contiguous Tensor, ND format.
 * @param [in] p: aclTensor on NPU device, supports FLOAT, FLOAT16, BFLOAT16, supports non-contiguous Tensor, ND format.
 * @param [in] k: aclTensor on NPU device, supports INT32, supports non-contiguous Tensor, ND format.
 * @param [in] out: aclTensor on NPU device, supports FLOAT, FLOAT16, BFLOAT16, supports non-contiguous Tensor, ND format.
 * @param [out] workspaceSize: returns the workspace size needed on NPU device.
 * @param [out] executor: returns the op executor containing the computation flow.
 * @return aclnnStatus: returns status code.
 */
aclnnStatus aclnnApplyTopKTopPCustomGetWorkspaceSize(const aclTensor* logits, const aclTensor* p,
                                                         const aclTensor* k, aclTensor* out, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor);

/**
 * @brief aclnnApplyTopKTopPCustom second phase interface, executes the computation.
 * @param [in] workspace: workspace memory address on NPU device.
 * @param [in] workspaceSize: workspace size on NPU device, obtained from aclnnApplyTopKTopPCustomGetWorkspaceSize.
 * @param [in] stream: acl stream.
 * @param [in] executor: op executor containing the computation flow.
 * @return aclnnStatus: returns status code.
 */
aclnnStatus aclnnApplyTopKTopPCustom(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_APPLY_TOP_K_TOP_P_CUSTOM_H_
