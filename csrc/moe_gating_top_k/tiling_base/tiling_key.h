/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tiling_key.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace Ops {
namespace Transformer {
namespace OpTiling {
constexpr uint64_t RecursiveSum()
{
    return 0;
}

constexpr uint64_t kBase = 10; // 10进制进位基数
template <typename T, typename... Args> constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + kBase * RecursiveSum(templateIds...);
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

} // namespace Optiling
} // namespace Transformer
} // namespace Ops
