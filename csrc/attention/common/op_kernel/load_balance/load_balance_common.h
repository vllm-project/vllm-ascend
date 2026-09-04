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
 * \file load_balance_common.h
 * \brief
 */

#ifndef LOAD_BALANCE_COMMON_H
#define LOAD_BALANCE_COMMON_H

#include <cstdint>
#include <string>
#include <array>
#include <functional>
#include <unordered_map>

namespace load_balance {

/******************************** CLASS DEF  ********************************/

template<class T>
using Range = std::pair<T, T>;

template<class T, size_t ROW, size_t COL>
using Table = std::array<std::array<T, COL>, ROW>;

using CostFunc = std::function<int64_t(uint32_t, uint32_t)>;       // The cost function of the algo

enum class SocVersion : uint32_t {
    BUTT
};

struct DeviceInfo {
    uint32_t aicCoreMaxNum { 0U };      // At most amount of aic core would be turned on, this is not final used number
    uint32_t aicCoreMinNum { 0U };      // At least amount of aic core would be turned on, this is not final used number
    uint32_t aivCoreMaxNum { 0U };      // At most amount of aiv core would be turned on, this is not final used number
    uint32_t aivCoreMinNum { 0U };      // At least amount of aiv core would be turned on, this is not final used number
    SocVersion version { SocVersion::BUTT };
};

struct GeneralBalanceParam {
    uint32_t mBaseSize { 1U };              // At least one
    uint32_t s2BaseSize { 1U };             // At least one
    int64_t faToleranceRatio { 2U };        // Larger the value, the smaller the tolerance is
    bool fdOn { true };                     // Turn on to activate FD
    int64_t fdTolerance { 0U };             // if fd - full_block_cost * fdTolerance <= fd, then choose no fd,
                                            // full_block_cost is costFunc(mBaseSize, s2BaseSize)
    int64_t fdLeastBlock { 0U };            // if noFd.maxCost <= fdLeastBlock * full_block_cost, then choose no fd
    CostFunc costFunc { nullptr };          // Customize cost func. Set nullptr to use default cost func
};

enum class SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    BUTT,
};

enum class Layout : uint8_t {
    BSND = 0,
    BNSD,
    BSH,
    NBSD,
    TND,
    NTD,
    PA_NZ,
    PA_BBND,
    PA_BNBD,
    BUTT
};

enum class DataType : uint8_t {
    FP32 = 0,
    FP16,
    INT4,
    INT8,
    INT32,
    BUTT
};

/******************************** UTIL FUNC ********************************/
template<class T>
inline T CeilDiv(T a, T b)
{
    static_assert(std::is_integral_v<T>, "must be integer type");
    if (b == 0) {
        return 0;
    }
    return a / b + (a % b != 0 ? 1 : 0);        // avoid overflow
}

template<class T>
inline T FloorDiv(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return a / b;
}

template<class T>
inline T SafeFloorDiv(T a, T b, T val)
{
    static_assert(std::is_integral_v<T>, "must be integer type");
    if (b == 0) {
        return val;
    }
    return a / b;
}

template<typename T>
inline T AddOne(T val)
{
    static_assert(std::is_integral_v<T>, "must be integer type");
    return val + 1;
}

template<typename T>
inline T MinusOne(T val)
{
    static_assert(std::is_integral_v<T>, "must be integer type");
    return val - 1;
}

template<typename T>
inline T ToOpenInterval(T val)
{
    return AddOne(val);
}

template<typename T>
inline T ToClosedInterval(T val)
{
    return MinusOne(val);
}

template<typename T>
inline T IndexToNum(T val)
{
    return AddOne(val);
}

template<typename T>
inline T NumToIndex(T val)
{
    return MinusOne(val);
}

template<typename T>
T Clip(T value, T minValue, T maxValue)
{
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

template<typename T>
inline bool IsWithinTolerance(T limit, T tolerance, T value)
{
    return limit + tolerance >= value;
}

static inline Layout ConvertToLayout(const std::string &layoutStr)
{
    static std::unordered_map<std::string, Layout> layoutTable {
        { "BSND",  Layout::BSND },
        { "BNSD",  Layout::BNSD },
        { "BSH",   Layout::BSH },
        { "NBSD",  Layout::NBSD },
        { "TND",   Layout::TND },
        { "NTD",   Layout::NTD },
        { "PA_NZ", Layout::PA_NZ }
    };
    if (layoutTable.find(layoutStr) != layoutTable.end()) {
        return layoutTable[layoutStr];
    }
    return Layout::BUTT;
}

static inline uint32_t GetDataTypeByteSize(DataType type)
{
    switch (type) {
        case (DataType::FP32):
        case (DataType::INT32):
            return 4U;
        case (DataType::FP16):
            return 2U;
        case (DataType::INT8):
            return 1U;
        default:
            return 2U;
    }
}

}
#endif