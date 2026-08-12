/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fragment_tensor_api.h
 * \brief FragmentTensor API 总入口
 */

#pragma once

#include "fragment_tensor.h"

namespace Apace {
namespace Basic {

/*!
 * \brief 创建FragmentTensor
 */
template<uint32_t Dims, uint32_t MaxFragments = MAX_FRAGMENT_COUNT,
    typename LayoutFactory = void, typename ElementType = uint8_t>
__aicore__ inline auto MakeFragmentTensor(
    const FragmentParam<Dims>& fragParam,
    GM_ADDR* addrList) {
    return FragmentTensor<Dims, MaxFragments, LayoutFactory, ElementType>(fragParam, addrList);
}

} // namespace Basic
} // namespace Apace
