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
 * \file sort.h
 * \brief
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_SORT_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_SORT_OP_H_

#include "opdev/op_executor.h"
#include "opdev/fast_vector.h"

namespace l0op {
const std::tuple<aclTensor*, aclTensor*> Sort(const aclTensor* self, int64_t dim, bool descending, bool stable,
                                              op::DataType indicesType, aclOpExecutor* executor);
}

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_SORT_OP_H_
