/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_FREQUENCY_REGULATOR_
#define OP_API_INC_FREQUENCY_REGULATOR_

#include <string>

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief First-stage interface of aclnnFrequencyRegulator that calculates 
 *        workspace size based on the specific compute flow.
 * @domain aclnn_ops_infer
 * @param [in] freq: The target frequency to be set (uint32_t).
 * @param [in] out: The output tensor.
 * @param [out] workspaceSize: workspace size to allocate on the NPU device side.
 * @param [out] executor: op executor containing the operator compute flow.
 * @return aclnnStatus: status code.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFrequencyRegulatorGetWorkspaceSize(
    uint32_t freq,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Second-stage interface of aclnnFrequencyRegulator to execute computation.
 * @param [in] workspace: workspace memory address allocated on the NPU device side.
 * @param [in] workspaceSize: workspace size allocated on the NPU device side, obtained from aclnnFrequencyRegulatorGetWorkspaceSize.
 * @param [in] executor: op executor containing the operator compute flow.
 * @param [in] stream: acl stream.
 * @return aclnnStatus: status code.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFrequencyRegulator(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_FREQUENCY_REGULATOR_