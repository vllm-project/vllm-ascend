/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_DISPATCH_POLICY_HPP
#define CATLASS_EPILOGUE_DISPATCH_POLICY_HPP

#include "catlass/arch/arch.hpp"

namespace Catlass::Epilogue {

// For AtlasA2, an element wise epilogue of the form D = C + X, where X is an additional source
struct EpilogueAtlasA2ElemWiseOneSource {
    using ArchTag = Arch::AtlasA2;
    // Number of operands. Including C, X, and D 3 operands
    static constexpr uint32_t OPERANDS_NUM = 3;
};

// For AtlasA2, FA Softmax
struct EpilogueAtlasA2FASoftmax {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, FA RescaleO
struct EpilogueAtlasA2FARescaleO {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, MLA Softmax
struct EpilogueAtlasA2MLASoftmax {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, MLA RescaleO
struct EpilogueAtlasA2MLARescaleO {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, MLA FD RescaleO
template <uint32_t COMPUTE_ELE_NUM_>
struct EpilogueAtlasA2MLAFDRescaleO {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t KV_SPLIT_MAX = 64;
    static constexpr uint32_t HEADS_PROCESS_MAX = 16;
    static constexpr uint32_t COMPUTE_ELE_NUM = COMPUTE_ELE_NUM_;
};

// For AtlasA2, MLA TP1 Softmax
struct EpilogueAtlasA2MLATP1Softmax {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, MLA TP1 RescaleO
struct EpilogueAtlasA2MLATP1RescaleO {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, per token dequant
template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequant {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};
////////////////////////////
/// new add
// For AtlasA2, GEMM
struct EpilogueAtlasA2Gemm {
    using ArchTag = Arch::AtlasA2;
};

// For AtlasA2, GEMV
struct EpilogueAtlasA2Gemv {
    using ArchTag = Arch::AtlasA2;
};
///////////////////////////
}  // namespace Catlass::Epilogue

#endif  // CATLASS_EPILOGUE_DISPATCH_POLICY_HPP
