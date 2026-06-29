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
 * \file dispatch_policy_custom.hpp
 * \brief
 */

#ifndef DISPATH_POLICY_CUSTOM_W4A8_HPP
#define DISPATH_POLICY_CUSTOM_W4A8_HPP

namespace Catlass::Gemm {

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2PreloadFixpipeQuant : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2PreloadAsyncFixpipe
    : public MmadAtlasA2PreloadAsync<PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_,
                                     ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_> {};

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2W4A4MatmulPerChannelDequant
    : public MmadAtlasA2PreloadAsyncWithCallback<PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_,
                                                 ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_> {};

// ------------------------------------------------------------------
// Unified dispatch-policy & BlockMmad selector (used by the unified
// MegaMoe kernel).
//
// Goal: keep a SINGLE name for the dispatch policy and BlockMmad in the
// caller; switch the underlying primitive purely from the weight element
// type (`BType_`). The selection matches what the three legacy
// implementations historically used:
//   - BType_ == int4b_t  -> W4A4 path  -> MmadAtlasA2W4A4MatmulPerChannelDequant
//                                           + BlockMmad with QuantTileCopy
//                                             (ScaleGranularity::PER_CHANNEL)
//   - other (int8 / bf16 / fp16) -> MmadAtlasA2PreloadAsyncFixpipe
//                                           + BlockMmad with default tile copy
//
// The caller no longer carries a `std::conditional_t<isW4A8, …, …>` in its
// own scope: it just spells the element type and gets back the right
// BlockMmad shape.
// ------------------------------------------------------------------
template <typename BType_, uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadDispatchPolicyFor {
    // For W4A4 we need the WithCallback variant so the per-channel scale
    // can be applied during fixpipe.
    using type =
        std::conditional_t<std::is_same_v<BType_, AscendC::int4b_t>,
                           MmadAtlasA2W4A4MatmulPerChannelDequant<PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_,
                                                                  L0C_STAGES_, ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>,
                           MmadAtlasA2PreloadAsyncFixpipe<PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_,
                                                          L0C_STAGES_, ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>>;
};

template <typename BType_, uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
using MmadDispatchPolicyFor_t =
    typename MmadDispatchPolicyFor<BType_, PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_,
                                   ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>::type;
} // namespace Catlass::Gemm

namespace Catlass::Epilogue {

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2UnQuant {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequantQuant {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequantSwigluQuant {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequantV2 {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequantInt8 {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequantBF16 {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequantSwigluQuantInt8 {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

// A2-specific BF16 SwiGLU (non-quantized). Uses a distinct tag so the
// A2 epilogue specialization does not conflict with the generic
// EpilogueAtlasA2PerTokenDequantSwigluQuant specialization.
template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequantSwigluQuantBF16 {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequantV2Int8 {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2PerTokenDequantV2BF16 {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA2W4A8PostPerTokenDequantV2 {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

} // namespace Catlass::Epilogue
#endif // DISPATH_POLICY_CUSTOM_HPP