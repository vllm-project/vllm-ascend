/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATTN_INFRA_EPILOGUE_COMMON_HPP_
#define ATTN_INFRA_EPILOGUE_COMMON_HPP_

#include "common.hpp"
#include "arch.hpp"
#include "layout.hpp"
#include "gemm_common.hpp"

namespace NpuArch::Epilogue 
{

enum class LseMode {NONE = 0, OUT_ONLY = 1};
enum class MaskMode {
    NO_MASK = 0,
    MASK_CAUSAL = 1,
    MASK_SPEC = 2,
    MASK_SWA = 4
};
// For AtlasA2, FA Infer online Softmax
template <LseMode LSE_MODE_, MaskMode MASK_MODE_, typename SM_DTYPE_>
struct EpilogueAtlasA2OnlineSoftmax {
    using ArchTag = Arch::AtlasA2;
    using IntermPrec = SM_DTYPE_;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
    static constexpr MaskMode MASK_MODE = MASK_MODE_;
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

namespace NpuArch::Epilogue::Block {

// Primary template: specializations live in epilogue_online_softmax.hpp /
// epilogue_rescale_o.hpp / epilogue_init_outputs.hpp.
template <
    class DispatchPolicy,
    class... Args
>
class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

}  // namespace NpuArch::Epilogue::Block
#endif // ATTN_INFRA_EPILOGUE_COMMON_HPP_
