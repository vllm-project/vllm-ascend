/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_DISPATCH_POLICY_HPP
#define EPILOGUE_DISPATCH_POLICY_HPP

#include "../../attn_infra/base_defs.hpp"
#include "../../attn_infra/arch/arch.hpp"

namespace NpuArch::Epilogue
{

enum class LseMode {NONE = 0, OUT_ONLY = 1};
enum class SinkMode {DISABLE = 0, ENABLE = 1};

// For AtlasA2, FA Infer online Softmax
template <LseMode LSE_MODE_, SinkMode SINK_MODE_, typename SM_DTYPE_>
struct EpilogueAtlasA2OnlineSoftmax {
    using ArchTag = Arch::AtlasA2;
    using IntermPrec = SM_DTYPE_;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
    static constexpr SinkMode SINK_MODE = SINK_MODE_;
};

// For AtlasA2, FA Infer RescaleO
template <LseMode LSE_MODE_, typename SM_DTYPE_>
struct EpilogueAtlasA2RescaleO {
    using ArchTag = Arch::AtlasA2;
    using IntermPrec = SM_DTYPE_;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
};

// For AtlasA2, FA Infer Deal kv-len=0
template <LseMode LSE_MODE_>
struct EpilogueAtlasA2InitOutWhenZero {
    using ArchTag = Arch::AtlasA2;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
};

}  // namespace NpuArch::Epilogue

#endif  // EPILOGUE_DISPATCH_POLICY_HPP