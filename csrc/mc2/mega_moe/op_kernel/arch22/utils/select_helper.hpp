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
 * \file select_helper.hpp
 * \brief
 */

#ifndef SELECT_HELPER_HPP
#define SELECT_HELPER_HPP

#include "../template_linear_algebra_v2/layout/layout.hpp"
using namespace AscendC;
using namespace Catlass;

template <typename Layout, typename ElementType, typename = void>
struct LayoutBInitializer {
    CATLASS_DEVICE
    static Layout create(uint32_t k, uint32_t n)
    {
        return Layout{k, n};
    }
};

template <typename Layout, typename ElementType>
struct LayoutBInitializer<Layout, ElementType, std::enable_if_t<std::is_same_v<Layout, layout::zN>>> {
    CATLASS_DEVICE
    static Layout create(uint32_t k, uint32_t n)
    {
        return Layout::template MakeLayout<ElementType>(k, n);
    }
};
#endif // SELECT_HELPER_HPP