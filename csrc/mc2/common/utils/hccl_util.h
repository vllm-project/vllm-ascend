/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_HCCL_UTIL_H_
#define OP_API_INC_HCCL_UTIL_H_

#include "hccl/hccl_types.h"
#include "op_mc2_def.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace op {
#define OP_API_CHECK(cond, exec_expr)                                                                                  \
    do {                                                                                                               \
        if (cond) {                                                                                                    \
            exec_expr;                                                                                                 \
        }                                                                                                              \
    } while (0)
} // namespace op

#ifdef __cplusplus
}
#endif
#endif // OP_API_INC_HCCL_UTIL_H_
