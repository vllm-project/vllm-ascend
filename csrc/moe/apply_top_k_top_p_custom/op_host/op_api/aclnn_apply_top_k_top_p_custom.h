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
 * \file aclnn_apply_top_k_top_p_custom.h
 * \brief
 */
#ifndef OP_API_INC_APPLY_TOP_K_TOP_P_CUSTOM_H_
#define OP_API_INC_APPLY_TOP_K_TOP_P_CUSTOM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Phase 1 of aclnnApplyTopKTopPCustom. Calculates workspace size.
 * @domain aclnn_ops_infer
 * @param [in] logits: npu
 * device aclTensor. Supports FLOAT, FLOAT16, BFLOAT16, non-contiguous tensor and ND format.
 * @param [in] p: NPU device aclTensor. Supports FLOAT, FLOAT16, BFLOAT16, non-contiguous tensor and ND format.
 * @param [in] k: NPU device aclTensor. Supports INT32, non-contiguous tensor and ND format.
 * @param [in] out: NPU device aclTensor. Supports FLOAT, FLOAT16, BFLOAT16, non-contiguous tensor and ND format.
 * @param [out] workspaceSize: Workspace size required on NPU device.
 * @param [out] executor: Op executor that contains the computation flow.
 * @return aclnnStatus: Status code.
 */
ACLNN_API aclnnStatus aclnnApplyTopKTopPCustomGetWorkspaceSize(const aclTensor* logits, const aclTensor* p,
                                                         const aclTensor* k, aclTensor* out, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor);

/**
 * @brief Phase 2 of aclnnApplyTopKTopPCustom. Executes the computation.
 * @param [in] workspace: Workspace memory allocated on NPU device.
 * @param [in] workspaceSize: Workspace size returned by aclnnApplyTopKTopPCustomGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: Op executor that contains the computation flow.
 * @return aclnnStatus: Status code.
 */
ACLNN_API aclnnStatus aclnnApplyTopKTopPCustom(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_APPLY_TOP_K_TOP_P_CUSTOM_H_
