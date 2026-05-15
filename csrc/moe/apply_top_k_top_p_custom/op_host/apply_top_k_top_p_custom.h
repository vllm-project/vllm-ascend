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
 * \file apply_top_k_top_p_custom.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_OP_APPLY_TOP_K_TOP_P_CUSTOM_OP_H_
#define OP_API_INC_LEVEL0_OP_APPLY_TOP_K_TOP_P_CUSTOM_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* ApplyTopKTopPCustom(const aclTensor* sortedValue, const aclTensor* sortedIndices,
                                         const aclTensor* p, const aclTensor* k, aclOpExecutor* executor);
}
#endif // OP_API_INC_LEVEL0_OP_APPLY_TOP_K_TOP_P_CUSTOM_OP_H_