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
 * \file scatter_nd_update_v2.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_OP_SCATTER_ND_UPDATE_V2_OP_H_
#define OP_API_INC_LEVEL0_OP_SCATTER_ND_UPDATE_V2_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* ScatterNdUpdateV2(const aclTensor* self, const aclTensor* indices, const aclTensor* updates,
                                 const aclIntArray* strides, bool use_locking, aclOpExecutor* executor);
}  // namespace l0op

#endif  // OP_API_INC_LEVEL0_OP_SCATTER_ND_UPDATE_V2_OP_H_
