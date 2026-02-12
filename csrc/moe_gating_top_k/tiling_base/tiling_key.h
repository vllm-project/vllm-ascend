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

constexpr uint64_t kBase = 10; // Base-10 carry base
template <typename T, typename... Args> constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + kBase * RecursiveSum(templateIds...);
}

// TilingKey generation rules:
// FlashAttentionScore/FlashAttentionScoreGrad assembles tiling key using decimal digits, containing the following key parameters from low to high: Ub0, Ub1,
// Block, DataType, Format, Sparse. Specialized template Ub0, Ub1:
//     Represents the axis for UB intra-core splitting, using AxisEnum. Since we allow at most two axes to be split, UB0 and UB1 exist. If there is no UB intra-core splitting,
//     fill with AXIS_NONE. UB0 and UB1 each occupy one decimal digit;
//     Block: Represents the axis used by UB for multi-core splitting, using AxisEnum, occupies one decimal digit;
//     DataType: Represents the input/output data types supported by the current tiling key, using SupportedDtype enum, occupies one decimal digit
//     Format: Represents the Format supported by the current tiling key, using InputLayout enum, occupies one decimal digit
//     Sparse: Represents whether the current tiling key supports Sparse, using SparseCapability enum, occupies one decimal digit
//     For other specialized scenarios, define your own bit fields and values
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
