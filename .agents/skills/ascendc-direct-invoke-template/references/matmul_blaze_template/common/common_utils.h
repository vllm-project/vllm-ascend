/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)
#define CHECK_COND(cond, msg)                                                                                  \
    do {                                                                                                       \
        if (!(cond)) {                                                                                         \
            throw std::runtime_error(                                                                          \
                std::string("Error: ") + msg + "\nFile: " + __FILE__ + "\nLine: " + std::to_string(__LINE__)); \
        }                                                                                                      \
    } while (0)

template <typename T>
inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b + static_cast<T>(a % b != 0);
}

template <typename T>
inline T Align(T a, T b)
{
    return CeilDiv(a, b) * b;
}

inline uint64_t ParsePositiveUint64(const char* arg, const char* name)
{
    std::string value(arg);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument(std::string("ERROR: ") + name + " must be a positive integer");
    }

    try {
        uint64_t parsed = std::stoull(value);
        if (parsed == 0UL) {
            throw std::invalid_argument(std::string("ERROR: ") + name + " must be greater than 0");
        }
        return parsed;
    } catch (const std::out_of_range&) {
        throw std::invalid_argument(std::string("ERROR: ") + name + " is out of range for uint64_t");
    }
}

inline void CheckUint32Shape(uint64_t value, const char* name)
{
    constexpr uint64_t uint32Max = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    if (value > uint32Max) {
        throw std::invalid_argument(std::string("ERROR: ") + name + " must not exceed UINT32_MAX");
    }
}

inline void PrintUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " m k n transA transB" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "  transA: transdata of matrix A" << std::endl;
    std::cerr << "  transB: transdata of matrix B" << std::endl;
    std::cerr << "Example: " << programName << " 16 128 16384 false true" << std::endl;
}

#endif
