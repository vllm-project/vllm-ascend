/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdarg>
#include <cstddef>
#include <string>
#include "securec.h"

namespace Ops {
namespace NN {
constexpr std::size_t FORMAT_STRING_BUFFER_SIZE = 1024U;

inline std::string FormatString(const char* format, ...)
{
    char buffer[FORMAT_STRING_BUFFER_SIZE] = {};
    va_list args;
    va_start(args, format);
    auto ret = vsnprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1U, format, args);
    va_end(args);
    if (ret < 0) {
        return std::string(format);
    }
    return std::string(buffer);
}

inline const char* BoolToString(bool value) { return value ? "true" : "false"; }

inline std::string StripEnclosingSquareBrackets(const char* value)
{
    if (value == nullptr) {
        return {};
    }
    std::string str(value);
    if (str.size() >= 2U && str.front() == '[' && str.back() == ']') {
        return str.substr(1U, str.size() - 2U);
    }
    return str;
}
} // namespace NN
} // namespace Ops
