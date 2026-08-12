/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file common_utils.h
 * \brief Host-side argument parsing, shape helpers, and error utilities for matmul examples.
 */

#pragma once
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace mm {
// Quantized MX element layouts for non-type template parameters.
enum class DataType {
    DT_FLOAT4_E2M1,
    DT_FLOAT8_E4M3FN,
};
} // namespace mm

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

template <typename T>
inline T FloorAlign(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b * b;
}

enum class DataType {
    FP4,
    FP8
};

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
    // QuantMatmulTilingData serializes public shape fields as uint32_t.
    constexpr uint64_t uint32Max = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    if (value > uint32Max) {
        throw std::invalid_argument(std::string("ERROR: ") + name + " must not exceed UINT32_MAX");
    }
}

inline void PrintUsage(const std::string& programName)
{
    if (programName == "./matmul_a16w16_swat") {
        std::cerr << "Usage: " << programName << " m k n transA transB" << std::endl;
    } else {
        std::cerr << "Usage: " << programName << " m k n" << std::endl;
    }
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    if (programName == "./matmul_a16w16_swat") {
        std::cerr << "  transA: transdata of matrix A" << std::endl;
        std::cerr << "  transB: transdata of matrix B" << std::endl;
    }
    std::cerr << "Example: " << programName << " 100 50 200" << std::endl;
    if (programName == "./matmul_a16w16_swat") {
        std::cerr << "Example: " << programName << " 100 50 200 false true"<< std::endl;
    }
}

inline void ParseArguments(int argc, char* argv[], uint64_t& m, uint64_t& k, uint64_t& n)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        PrintUsage(argv[0]);
        std::exit(1);
    }
    if (argc != 4) {
        throw std::invalid_argument("ERROR: Invalid number of arguments, expected exactly 3 arguments: m k n");
    }
    m = ParsePositiveUint64(argv[1], "m");
    k = ParsePositiveUint64(argv[2], "k");
    n = ParsePositiveUint64(argv[3], "n");
    CheckUint32Shape(m, "m");
    CheckUint32Shape(k, "k");
    CheckUint32Shape(n, "n");
}

inline void ParseArguments(int argc, char* argv[], uint64_t& m, uint64_t& k, uint64_t& n, bool& transA, bool& transB)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        PrintUsage(argv[0]);
        std::exit(1);
    }
    if (argc != 4 && argc != 6) {
        throw std::invalid_argument(
            "ERROR: Invalid number of arguments, expected least 4 arguments: m k n, or 6 arguments: m k n transA "
            "transB");
    }
    m = ParsePositiveUint64(argv[1], "m");
    k = ParsePositiveUint64(argv[2], "k");
    n = ParsePositiveUint64(argv[3], "n");
    if (argc == 6) {
        transA = (std::string(argv[4]) == "true");
        transB = (std::string(argv[5]) == "true");
    } else {
        transA = false;
        transB = true;
    }
    CheckUint32Shape(m, "m");
    CheckUint32Shape(k, "k");
    CheckUint32Shape(n, "n");
}

template <mm::DataType dataType, typename T>
constexpr T GetShapeWithDataType(T size)
{
    if constexpr (dataType == mm::DataType::DT_FLOAT4_E2M1) {
        return size << 1;
    } else {
        return size;
    }
}

template <mm::DataType dataType, typename T>
constexpr T GetSizeWithDataType(T shape)
{
    if constexpr (dataType == mm::DataType::DT_FLOAT4_E2M1) {
        return (shape + 1) >> 1;
    } else {
        return shape;
    }
}

