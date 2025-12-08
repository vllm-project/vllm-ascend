/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef OP_API_INC_MOE_INIT_ROUTING_V3_H_
#define OP_API_INC_MOE_INIT_ROUTING_V3_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMoeInitRoutingV3的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnMoeInitRoutingV3GetWorkspaceSize(const aclTensor *x, 
                                                            const aclTensor *expertIdx,
                                                            const aclTensor *scaleOptional,
                                                            const aclTensor *offsetOptional, 
                                                            int64_t activeNum, 
                                                            int64_t expertCapacity, 
                                                            int64_t expertNum, 
                                                            int64_t dropPadMode, 
                                                            int64_t expertTokensNumType, 
                                                            bool expertTokensNumFlag, 
                                                            int64_t quantMode, 
                                                            const aclIntArray *activeExpertRangeOptional, 
                                                            int64_t rowIdxType, 
                                                            const aclTensor *expandedXOut, 
                                                            const aclTensor *expandedRowIdxOut, 
                                                            const aclTensor *expertTokensCountOrCumsumOut, 
                                                            const aclTensor *expandedScaleOut, 
                                                            uint64_t *workspaceSize, 
                                                            aclOpExecutor **executor);
                                                            
/* @brief aclnnMoeInitRoutingV3的第二段接口，用于执行计算。 */
ACLNN_API aclnnStatus aclnnMoeInitRoutingV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif