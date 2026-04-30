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
* \file padv3.h
* \brief
*/

#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_REDUCE_SUM_OP_H_
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_REDUCE_SUM_OP_H_

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const aclTensor *ReduceSumOp(const aclTensor *x, const aclIntArray *axes,
                             bool keep_dims, aclOpExecutor *executor);
}

#endif // OP_API_OP_API_COMMON_INC_LEVEL0_OP_REDUCE_SUM_OP_H_