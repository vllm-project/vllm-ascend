/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_NUMERIC_MATH_HPP
#define TLA_NUMERIC_MATH_HPP

#include "tla/type_traits.hpp"

namespace tla {

//
// Common Operations
//

template <class T, class U,
          __TLA_REQUIRES(std::is_arithmetic<T>::value &&
                         std::is_arithmetic<U>::value)>
CATLASS_HOST_DEVICE constexpr
auto
max(T const& t, U const& u) {
    return t < u ? u : t;
}

template <class T, class U,
          __TLA_REQUIRES(std::is_arithmetic<T>::value &&
                         std::is_arithmetic<U>::value)>
CATLASS_HOST_DEVICE constexpr
auto
min(T const& t, U const& u) {
    return t < u ? t : u;
}

} // namespace tla

#endif // TLA_NUMERIC_MATH_HPP