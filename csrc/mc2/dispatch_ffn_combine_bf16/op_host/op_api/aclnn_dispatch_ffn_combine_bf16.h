/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_DISPATCH_FFN_COMBINE_BF16_
#define OP_API_INC_DISPATCH_FFN_COMBINE_BF16_

#include <string>

#include "aclnn/aclnn_base.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnDispatchFFNCombineBF16GetWorkspaceSize(const aclTensor* x, const aclTensorList* weight1, const aclTensorList* weight2,
                                                                                        const aclTensor* expertId, const aclTensorList* scale1, const aclTensorList* scale2,
                                                                                        const aclTensor* probs,
                                                                                        const char* group, int64_t maxOutputSize,
                                                                                        const aclTensor* out, const aclTensor* expertTokenNums,
                                                                                        uint64_t* workspaceSize, aclOpExecutor** executor);


__attribute__((visibility("default"))) aclnnStatus aclnnDispatchFFNCombineBF16(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_DISPATCH_FFN_COMBINE_BF16_