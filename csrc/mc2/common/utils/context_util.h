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
 * \file context_util.h
 * \brief
 */

#ifndef CANN_OPS_BUILT_IN_CONTEXT_UTIL_H_
#define CANN_OPS_BUILT_IN_CONTEXT_UTIL_H_

#include "runtime/infer_shape_context.h"
#include "runtime/tiling_context.h"
#include "mc2_log.h"

namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                                      \
    if ((ptr) == nullptr) {                                                                                            \
        const char *name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();                   \
        OP_LOGE_WITHOUT_REPORT(name, "%s is nullptr!", #ptr);                                                          \
        REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                                          \
        return ge::GRAPH_FAILED;                                                                                       \
    }

#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                                             \
    if ((ptr) == nullptr) {                                                                                            \
        const char *name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();                   \
        OP_LOGE_WITHOUT_REPORT(name, "%s is nullptr!", #ptr);                                                          \
        REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                                          \
        return ret;                                                                                                    \
    }
} // namespace ops
#endif // CANN_OPS_BUILT_IN_CONTEXT_UTIL_H_
