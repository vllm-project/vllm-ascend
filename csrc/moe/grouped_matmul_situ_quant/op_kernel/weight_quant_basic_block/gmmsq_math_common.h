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
 * \file gmmsq_math_common.h
 * \brief 内核自带的 Ops::Base 数学小工具，替代 third_party/opbase 的 op_kernel/math_util.h。
 *
 * 本算子只用 `Ops::Base::CeilDiv` / `CeilAlign` 两个模板。原先 include
 * "op_kernel/math_util.h"（来自 third_party/opbase 的 include 路径），下游移植树
 * （如 vllm-ascend 把本目录整体落到 csrc/moe/grouped_matmul_situ_quant/op_kernel/
 * vendor/gmsq2/）没有该路径，会编译失败。这里按 opbase 原实现逐字内联一份
 * （ceildiv/ceilalign 的中间类型同样提升到 uint32/uint64，避免窄类型溢出），
 * 使本目录对第三方 include 路径零依赖——只需本目录文件 + CANN 公共头即可编译。
 */
#ifndef GMMSQ_MATH_COMMON_H
#define GMMSQ_MATH_COMMON_H

#include <cstdint>
#include <type_traits>

namespace Ops {
namespace Base {
template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    using type = typename std::conditional<sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t,
                                           uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b);
    return static_cast<T>(res);
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    using type = typename std::conditional<sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t,
                                           uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b) * static_cast<type>(b);
    return static_cast<T>(res);
}
} // namespace Base
} // namespace Ops

#endif // GMMSQ_MATH_COMMON_H
