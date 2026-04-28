/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_DISPATCH_POLICY_HPP
#define CATLASS_CONV_DISPATCH_POLICY_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"

namespace Catlass::Conv {

// Block Mmad Policies

template <bool ASYNC_ = false>
struct ConvAtlasA2Base {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t ASYNC = ASYNC_;
};

using ConvAtlasA2 = ConvAtlasA2Base<false>;
using ConvAtlasA2Async = ConvAtlasA2Base<true>;

template <uint32_t L1A_STAGES_, uint32_t L1B_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
    uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_>
struct ConvAtlasA2Pingpong : public ConvAtlasA2 {
    static constexpr uint32_t L1A_STAGES = L1A_STAGES_;
    static constexpr uint32_t L1B_STAGES = L1B_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

}  // namespace Catlass::Conv

#endif  // CATLASS_CONV_DISPATCH_POLICY_HPP
