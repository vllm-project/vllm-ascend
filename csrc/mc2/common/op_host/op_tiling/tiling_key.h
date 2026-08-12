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
 * \file tiling_key.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace Ops {
namespace Transformer {
namespace Optiling {
constexpr uint64_t RecursiveSum()
{
    return 0;
}
constexpr uint64_t BASE_MULTIPLIER_SCALE = 10;

template <typename T, typename... Args>
constexpr uint64_t RecursiveSum(T templatedId, Args... templatedIds)
{
    return static_cast<uint64_t>(templatedId) + BASE_MULTIPLIER_SCALE * RecursiveSum(templatedIds...);
}

constexpr uint64_t TILINGKEYOFFSET = uint64_t(10000000000000000000UL);
template <typename... Args>
constexpr uint64_t GET_TILINGKEY(Args... templatedIds)
{
    return TILINGKEYOFFSET + RecursiveSum(templatedIds...);
}

#ifndef TILINGKEY
#define TILINGKEY(ub2, ub1, block, dtype, layout, sparse)                                                              \
    (GER_TILINGKEY(AxisEnum::ub2, AxisEnum::ub1, AxisEnum::block, DtypeEnum::dtype, LayoutEnum::layout,                \
                   SparseEnum::sparse))
#endif
} // namespace Optiling
} // namespace Transformer
} // namespace Ops
