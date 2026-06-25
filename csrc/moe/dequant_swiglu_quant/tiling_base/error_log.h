/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "log/log.h"

#ifndef OP_LOGE_FOR_INVALID_DTYPE
#define OP_LOGE_FOR_INVALID_DTYPE(opname, param, actual, expected) \
    OP_LOGE(opname, "Invalid dtype for %s, actual: %s, expected: %s", param, actual, expected)
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opname, param, actual, reason) \
    OP_LOGE(opname, "Invalid dtype for %s, actual: %s, reason: %s", param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPE
#define OP_LOGE_FOR_INVALID_SHAPE(opname, param, actual, expected) \
    OP_LOGE(opname, "Invalid shape for %s, actual: %s, expected: %s", param, actual, expected)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opname, param, actual, reason) \
    OP_LOGE(opname, "Invalid shape for %s, actual: %s, reason: %s", param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opname, param, actual, reason) \
    OP_LOGE(opname, "Invalid shapes for %s, actual: %s, reason: %s", param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM
#define OP_LOGE_FOR_INVALID_SHAPEDIM(opname, param, actual, expected) \
    OP_LOGE(opname, "Invalid shape dim for %s, actual: %s, expected: %s", param, actual, expected)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opname, param, actual, reason) \
    OP_LOGE(opname, "Invalid shape dims for %s, actual: %s, reason: %s", param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPESIZE
#define OP_LOGE_FOR_INVALID_SHAPESIZE(opname, param, actual, expected) \
    OP_LOGE(opname, "Invalid shape size for %s, actual: %s, expected: %s", param, actual, expected)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opname, param, actual, reason) \
    OP_LOGE(opname, "Invalid shape size for %s, actual: %s, reason: %s", param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opname, param, actual, reason) \
    OP_LOGE(opname, "Invalid value for %s, actual: %s, reason: %s", param, actual, reason)
#endif
