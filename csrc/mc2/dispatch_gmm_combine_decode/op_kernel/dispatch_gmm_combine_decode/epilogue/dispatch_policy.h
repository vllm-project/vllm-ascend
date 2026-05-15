/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "catlass/epilogue/dispatch_policy.hpp"

namespace Catlass::Epilogue {

template <uint32_t UB_STAGES_, uint32_t EXEC_FLAG_>
struct EpilogueAtlasA2PerTokenDequantSwiglu {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr uint32_t EXEC_FLAG = EXEC_FLAG_;
};

template <uint32_t UB_STAGES_, uint32_t EXEC_FLAG_>
struct EpilogueAtlasA2PerTokenDequantCombine {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr uint32_t EXEC_FLAG = EXEC_FLAG_;
};

}  // namespace Catlass::Epilogue
