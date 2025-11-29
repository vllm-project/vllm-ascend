/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tiling_type.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace optiling {

enum class AxisEnum {
    B = 0,
    N2 = 1,
    G = 2,
    S1 = 3,
    S2 = 4,
    D = 5,
    NONE = 9,
};

enum class DtypeEnum {
    FLOAT16 = 0,
    FLOAT32 = 1,
    BFLOAT16 = 2,
    FLOAT16_PRECISION = 3,
};

enum class PerformanceOrientedEnum {
    BIG_BUFFER = 1,
    BIG_DOUBLE_BUFFER = 2,
};

enum class MatmulConfig {
    NULL_CONFIG = 0,
    NORMAL_CONFIG = 1,
    MDL_CONFIG = 2
};

enum class PseConfig {
    NO_PSE = 0,
    EXIST_PSE = 1
};

enum class AttenMaskConfig {
    NO_ATTEN_MASK = 0,
    EXIST_ATTEN_MASK = 1
};

enum class DropOutConfig {
    NO_DROP_OUT = 0,
    EXIST_DROP_OUT = 1
};

enum class CubeFormatEnum {
    ND = 0,
    NZ = 1
};
enum class LayoutEnum {
    BSND = 0,
    SBND = 1,
    BNSD = 2,
    TND = 3
};

enum class CubeInputSourceEnum {
    GM = 0,
    L1 = 1
};

enum class OptionEnum {
    DISABLE = 0,
    ENABLE = 1
};

enum class SparseEnum {
    ALL = 0,
    NONE = 1,
    ANY = 2,
    CAUSAL = 3,
    BAND = 4,
    PREFIX = 5,
    BAND_COMPRESS = 6,
    RIGHT_DOWN_CAUSAL = 7,
    RIGHT_DOWN_CAUSAL_BAND = 8,
    BAND_LEFT_UP_CAUSAL = 9
};

constexpr uint64_t RecursiveSum()
{
    return 0;
}

template <typename T, typename... Args> constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + 10 * RecursiveSum(templateIds...);
}

// TilingKey 的生成规则：
// FlashAttentionScore/FlashAttentionScoreGrad 十进制位组装tiling key，包含以下关键参数，从低位到高位依次是：Ub0, Ub1,
// Block, DataType, Format, Sparse, 特化模板 Ub0、Ub1:
//     表示Ub核内切分的轴，使用枚举AxisEnum表示，因为我们允许最多切分两根轴，所以存在UB0和UB1，如果没有UB核内切分，
//     那么填AXIS_NONE。UB0和UB1各占一个十进制位;
//     Block: 表示UB用来分核的轴，使用枚举AxisEnum表示，占一个十进制位;
//     DataType: 表示当前tiling key支持的输入输出的数据类型，使用枚举SupportedDtype来表示，占一个十进制位
//     Format: 表示当前tiling key支持的Format, 使用枚举InputLayout表示，占一个十进制位
//     Sparse: 表示当前tiling key是否支持Sparse，使用枚举SparseCapability表示，占一个十进制位
//     其余特化场景，定义自己的位域和值
// usage: get tilingKey from inputed types
//     uint64_t tilingKey = GET_FLASHATTENTION_TILINGKEY(AxisEnum::AXIS_S1, AxisEnum::AXIS_S2, AxisEnum::AXIS_N2,
//                                     SupportedDtype::FLOAT32, InputLayout::BSH, SparseCapability::SUPPORT_ALL)

constexpr uint64_t TILINGKEYOFFSET = uint64_t(10000000000000000000UL); // 10^19
template <typename... Args> constexpr uint64_t GET_TILINGKEY(Args... templateIds)
{
    return TILINGKEYOFFSET + RecursiveSum(templateIds...);
}

// usage: get tilingKey from inputed types
//     uint64_t tilingKey = TILINGKEY(S2, S1, N2, FLOAT32, BSND, ALL)

#define TILINGKEY(ub2, ub1, block, dtype, layout, sparse)                                                              \
    (GET_TILINGKEY(AxisEnum::ub2, AxisEnum::ub1, AxisEnum::block, DtypeEnum::dtype, LayoutEnum::layout,                \
                   SparseEnum::sparse))

} // namespace optiling
