/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_TILE_COPY_L0C_TO_DST_HPP
#define GEMM_TILE_COPY_L0C_TO_DST_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/arch.hpp"
#include "../../../attn_infra/gemm/gemm_type.hpp"
#include "../../../tla/tensor.hpp"

namespace NpuArch::Gemm::Tile {

enum class ScaleGranularity {
    UNDEFINED = -1,
    NO_QUANT = 0,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP
};

template <
    class ArchTag,
    class ElementSrc,
    class ElementDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT
>
struct CopyL0CToDstQuantMode {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

// CopyL0CToGm fp32 to fp32
template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    float, float,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

#if (__CCE_AICORE__ == 310)
template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    float, float,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::QF322F32_PRE;
};
#endif

// CopyL0CToGm cast fp32 to fp16
template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    float, half,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    float, half,
    ScaleGranularity::PER_TENSOR
> {
    // static constexpr auto VALUE = QuantMode_t::F322F16;
    static constexpr auto VALUE = QuantMode_t::QF322F16_PRE;
};

// CopyL0CToGm cast fp32 to bf16
template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    float, bfloat16_t,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::F322BF16;
};

template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    float, bfloat16_t,
    ScaleGranularity::PER_TENSOR
> {
    // static constexpr auto VALUE = QuantMode_t::F322BF16;
    static constexpr auto VALUE = QuantMode_t::QF322BF16_PRE;
};

// CopyL0CToGm cast float to uint8/int8
template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    float, uint8_t,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::QF322B8_PRE;
};

template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    float, uint8_t,
    ScaleGranularity::PER_CHANNEL
> {
    static constexpr auto VALUE = QuantMode_t::VQF322B8_PRE;
};

template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    float, int8_t,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::QF322B8_PRE;
};

template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    float, int8_t,
    ScaleGranularity::PER_CHANNEL
> {
    static constexpr auto VALUE = QuantMode_t::VQF322B8_PRE;
};

// CopyL0CToGm output int32
template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    int32_t, int32_t,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::NoQuant;
};

// CopyL0CToGm cast int32_t to fp16
template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    int32_t, half,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::DEQF16;
};

template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    int32_t, half,
    ScaleGranularity::PER_CHANNEL
> {
    static constexpr auto VALUE = QuantMode_t::VDEQF16;
};

// CopyL0CToGm cast int32 to uint8/int8
template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    int32_t, uint8_t,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::REQ8;
};

template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    int32_t, uint8_t,
    ScaleGranularity::PER_CHANNEL
> {
    static constexpr auto VALUE = QuantMode_t::VREQ8;
};

template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    int32_t, int8_t,
    ScaleGranularity::PER_TENSOR
> {
    static constexpr auto VALUE = QuantMode_t::REQ8;
};

template <class ArchTag>
struct CopyL0CToDstQuantMode<
    ArchTag,
    int32_t, int8_t,
    ScaleGranularity::PER_CHANNEL
> {
    static constexpr auto VALUE = QuantMode_t::VREQ8;
};

template <
    class ArchTag,
    class ElementAccumulator,
    class GmType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
>
struct CopyL0CToGm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

///////////////////////////////////////////CopyL0CToGmTla/////////////////////////////////////////////////
// L0C copy mode
struct CopyToGM {};
struct CopyToL1 {};

template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false,
    class Enable = void
>
struct CopyL0CToGmTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum class CopyL0CToUBMode {
    NO_SPLIT = 0,
    SPLIT_M,
    SPLIT_N,
    RESERVED
};

template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    CopyL0CToUBMode CopyMode = CopyL0CToUBMode::NO_SPLIT,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false,
    class Enable = void
>
struct CopyL0CToUBTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to ub, can not find the specialization.");
};

}  // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_COPY_L0C_TO_DST_HPP