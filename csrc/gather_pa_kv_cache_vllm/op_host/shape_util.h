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
 * \file shape_util.h
 * \brief
 */

#ifndef OP_COMMON_OP_HOST_UTIL_SHAPE_UTIL_H
#define OP_COMMON_OP_HOST_UTIL_SHAPE_UTIL_H

#include "exe_graph/runtime/shape.h"
#include "opbase_export.h"

namespace Ops {
namespace Base {
template <typename T1, typename T2>
bool CheckAxisBounds(const T1 dimNum, const T2 axis)
{
    if (dimNum == 0) {
        return axis == 0;
    }
    const int64_t minimumNum = static_cast<int64_t>(dimNum) * (-1);
    const int64_t maximumNum = static_cast<int64_t>(dimNum) - 1;

    return static_cast<int64_t>(axis) >= minimumNum && static_cast<int64_t>(axis) <= maximumNum;
}

OPBASE_API void SetUnknownRank(gert::Shape &shape);

OPBASE_API bool IsUnknownRank(const gert::Shape &shape);

OPBASE_API void SetUnknownShape(int64_t rank, gert::Shape &shape);

OPBASE_API bool IsUnknownShape(const gert::Shape &shape);
}  // namespace Base
}  // namespace Ops
#endif  // OP_COMMON_OP_HOST_UTIL_SHAPE_UTIL_H