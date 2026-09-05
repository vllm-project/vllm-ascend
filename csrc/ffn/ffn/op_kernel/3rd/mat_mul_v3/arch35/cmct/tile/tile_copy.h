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
 * \file tile_copy.h
 * \brief
 */
#pragma once

#include "copy_out/copy_out_split_m_with_params.h"

#include "tile_copy_policy.h"

namespace Cmct {
namespace Gemm {
namespace Tile {
/**
 * @struct TileCopy
 * @brief This struct template provides different copy operations based on the architecture and dispatch policy
 * @param [in] ArchTag: architecture tag
 * @param [in] DispatchPolicy: dispatch policy
 *
 */
template <class ArchTag, class DispatchPolicy>
struct TileCopy {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy>, "TileCopy is not implemented for this DispatchPolicy");
};

/**
 * @struct TileCopy
 * @brief Specialization of TileCopy for CopyWithParams dispatch policy
 * @param [in] ArchTag: architecture tag
 * @param [in] CopyWithParams: copy policy with params
 */
template <class ArchTag>
struct TileCopy<ArchTag, CopyWithParams> {
    using CopyPolicy = CopyWithParams;

    template <class InputType, const auto& COPY_CFG>
    using CopyGmToA1 = Copy<ArchTag, CopyPolicy, void, void, InputType, void, COPY_CFG>;

    template <class InputType, const auto& COPY_CFG>
    using CopyGmToB1 = Copy<ArchTag, CopyPolicy, void, void, InputType, void, COPY_CFG>;

    template <class InputType, class OutputType, typename T = void>
    using CopyCo1ToOut = Copy<ArchTag, CopyPolicy, void, OutputType, InputType>;
};

/**
 * @struct TileCopy
 * @brief Specialization of TileCopy for CopyWithLayout dispatch policy
 * @param [in] ArchTag: architecture tag
 * @param [in] CopyWithLayout: copy policy with layouts
 */
template <class ArchTag>
struct TileCopy<ArchTag, CopyWithLayout> {
    using CopyPolicy = CopyWithLayout;

    template <class InputType, class DstTrait, class SrcTrait>
    using CopyGmToA1 = Copy<ArchTag, CopyPolicy, InputType, DstTrait, SrcTrait>;

    template <class InputType, class DstTrait, class SrcTrait>
    using CopyGmToB1 = Copy<ArchTag, CopyPolicy, InputType, DstTrait, SrcTrait>;

    template <class AType, class DstTrait, class SrcTrait>
    using CopyA1ToA2 = Copy<ArchTag, CopyPolicy, AType, DstTrait, SrcTrait>;

    template <class BType, class DstTrait, class SrcTrait>
    using CopyB1ToB2 = Copy<ArchTag, CopyPolicy, BType, DstTrait, SrcTrait>;

    template <class OutputType, class DstTrait, class SrcTrait>
    using CopyCo1ToOut = Copy<ArchTag, CopyPolicy, OutputType, DstTrait, SrcTrait>;
};

/**
 * @struct TileCopy
 * @brief Specialization of TileCopy for CopyEnUnitFlagWithLayout dispatch policy
 * @param [in] ArchTag: architecture tag
 * @param [in] CopyEnUnitFlagWithLayout: copy enUnitFlag policy with layouts
 */
template <>
struct TileCopy<Arch::DAV2201, CopyEnUnitFlagWithLayout> {
    using ArchTag = Arch::DAV2201;
    using CopyPolicy = CopyWithLayout;

    template <class InputType, class DstTrait, class SrcTrait>
    using CopyGmToA1 = Copy<ArchTag, CopyPolicy, InputType, DstTrait, SrcTrait>;

    template <class InputType, class DstTrait, class SrcTrait>
    using CopyGmToB1 = Copy<ArchTag, CopyPolicy, InputType, DstTrait, SrcTrait>;

    template <class OutputType, class DstTrait, class SrcTrait>
    using CopyCo1ToOut = Copy<ArchTag, CopyEnUnitFlagWithLayout, OutputType, DstTrait, SrcTrait>;

    template <class AType, class DstTrait, class SrcTrait>
    using CopyA1ToA2 = Copy<ArchTag, CopyPolicy, AType, DstTrait, SrcTrait>;

    template <class BType, class DstTrait, class SrcTrait>
    using CopyB1ToB2 = Copy<ArchTag, CopyPolicy, BType, DstTrait, SrcTrait>;
};

/**
 * @struct TileCopy
 * @brief Specialization of TileCopy for CopySparseWithLayout dispatch policy
 * @param [in] ArchTag: architecture tag
 * @param [in] CopySparseWithLayout: copy sparse policy with layouts
 */
template <>
struct TileCopy<Arch::DAV2201, CopySparseWithLayout> {
    using ArchTag = Arch::DAV2201;
    using CopyPolicy = CopyWithLayout;

    template <class InputType, class DstTrait, class SrcTrait>
    using CopyGmToA1 = Copy<ArchTag, CopyPolicy, InputType, DstTrait, SrcTrait>;

    template <class InputType, class DstTrait, class SrcTrait>
    using CopyGmToB1 = Copy<ArchTag, CopyPolicy, InputType, DstTrait, SrcTrait>;

    template <class OutputType, class DstTrait, class SrcTrait>
    using CopyCo1ToOut = Copy<ArchTag, CopyPolicy, OutputType, DstTrait, SrcTrait>;

    template <class AType, class DstTrait, class SrcTrait>
    using CopyA1ToA2 = Copy<ArchTag, CopyPolicy, AType, DstTrait, SrcTrait>;

    template <class BType, class DstTrait, class SrcTrait>
    using CopyB1ToB2 = Copy<ArchTag, CopySparseWithLayout, BType, DstTrait, SrcTrait>;
};

/**
 * @struct TileCopy
 * @brief Specialization of TileCopy for CopyInAndCopyOutSplitMWithParams dispatch policy
 * @param [in] ArchTag: architecture tag
 * @param [in] CopyInAndCopyOutSplitMWithParams: copy in and copy out split m policy with params
 */
template <>
struct TileCopy<Arch::DAV3510, CopyInAndCopyOutSplitMWithParams> {
    using ArchTag = Arch::DAV3510;
    using CopyPolicy = CopyWithParams;

    template <class InputType, const auto& COPY_CFG>
    using CopyGmToA1 = Copy<ArchTag, CopyPolicy, void, void, InputType, void, COPY_CFG>;

    template <class InputType, const auto& COPY_CFG>
    using CopyGmToB1 = Copy<ArchTag, CopyPolicy, void, void, InputType, void, COPY_CFG>;

    template <class InputType, class OutputType, typename T = void>
    using CopyCo1ToOut = Copy<ArchTag, CopyOutSplitMWithParams, void, OutputType, InputType>;
};

/**
 * @struct TileCopy
 * @brief Specialization of TileCopy for CopyInAndCopyOutSplitMWithLayout dispatch policy
 * @param [in] ArchTag: architecture tag
 * @param [in] CopyInAndCopyOutSplitMWithLayout: copy in and copy out split m policy with layout
 */
template <>
struct TileCopy<Arch::DAV3510, CopyInAndCopyOutSplitMWithLayout> {
    using ArchTag = Arch::DAV3510;
    using CopyPolicy = CopyWithLayout;

    template <class InputType, class DstTrait, class SrcTrait>
    using CopyGmToA1 = Copy<ArchTag, CopyPolicy, InputType, DstTrait, SrcTrait>;

    template <class InputType, class DstTrait, class SrcTrait>
    using CopyGmToB1 = Copy<ArchTag, CopyPolicy, InputType, DstTrait, SrcTrait>;

    template <class OutputType, class DstTrait, class SrcTrait>
    using CopyCo1ToOut = Copy<ArchTag, CopyOutSplitMWithLayout, OutputType, DstTrait, SrcTrait>;

    template <class AType, class DstTrait, class SrcTrait>
    using CopyA1ToA2 = Copy<ArchTag, CopyPolicy, AType, DstTrait, SrcTrait>;

    template <class BType, class DstTrait, class SrcTrait>
    using CopyB1ToB2 = Copy<ArchTag, CopyPolicy, BType, DstTrait, SrcTrait>;
};

/**
 * @struct TileCopy
 * @brief Specialization of TileCopy for CopyOutSplitMWithParams dispatch policy
 * @param [in] ArchTag: architecture tag
 * @param [in] CopyOutSplitMWithParams: copy out split m  policy with params
 */
template <>
struct TileCopy<Arch::DAV3510, CopyOutSplitMWithParams> {
    using ArchTag = Arch::DAV3510;

    template <class InputType, class OutputType, typename T = void>
    using CopyCo1ToOut = Copy<ArchTag, CopyOutSplitMWithParams, void, OutputType, InputType>;
};

/**
 * @struct TileCopy
 * @brief Specialization of TileCopy for CopyOutSplitNWithParams dispatch policy
 * @param [in] ArchTag: architecture tag
 * @param [in] CopyOutSplitNWithParams: copy out split n policy with params
 */
template <>
struct TileCopy<Arch::DAV3510, CopyOutSplitNWithParams> {
    using ArchTag = Arch::DAV3510;

    template <class InputType, class OutputType, typename T = void>
    using CopyCo1ToOut = Copy<ArchTag, CopyOutSplitNWithParams, void, OutputType, InputType>;
};

/**
 * @struct TileCopy
 * @brief Specialization of TileCopy for CopyNoGmIn dispatch policy
 * @param [in] ArchTag: architecture tag
 * @param [in] CopyNoGmIn: copy policy without Gm input
 */
template <>
struct TileCopy<Arch::DAV3510, CopyNoGmIn> {
    using ArchTag = Arch::DAV3510;
    using CopyPolicy = CopyWithParams;

    template <class InputType, class OutputType, typename T = void>
    using CopyCo1ToOut = Copy<ArchTag, CopyPolicy, void, OutputType, InputType>;
};

/**
 * @struct TileCopy
 * @brief Specialization of TileCopy for CopyBasedBaseK dispatch policy
 * @param [in] ArchTag: architecture tag
 * @param [in] CopyBasedBaseK: copy based basek policy
 */
template <>
struct TileCopy<Arch::DAV3510, CopyBasedBaseK> {
    using CopyPolicy = CopyBasedBaseK;
    using ArchTag = Arch::DAV3510;

    template <class AType, class DstTrait, class SrcTrait>
    using CopyA1ToA2 = Copy<ArchTag, CopyPolicy, AType, DstTrait, SrcTrait>;

    template <class BType, class DstTrait, class SrcTrait>
    using CopyB1ToB2 = Copy<ArchTag, CopyPolicy, BType, DstTrait, SrcTrait>;

    template <class InputType, class OutputType, typename T = void>
    using CopyCo1ToOut = Copy<ArchTag, CopyBasedBaseK, void, OutputType, InputType>;
};

namespace Detail {
constexpr int32_t DUMMY_CFG = 0;

template <class TileCopy, typename Enable = void>
struct HasCopyGmToA1WithParams : public AscendC::Std::false_type {};

template <class TileCopy>
struct HasCopyGmToA1WithParams<TileCopy, std::void_t<typename TileCopy::template CopyGmToA1<void, DUMMY_CFG>>>
    : public AscendC::Std::true_type {};

template <class TileCopy, typename Enable = void>
struct HasCopyGmToB1WithParams : public AscendC::Std::false_type {};

template <class TileCopy>
struct HasCopyGmToB1WithParams<TileCopy, std::void_t<typename TileCopy::template CopyGmToB1<void, DUMMY_CFG>>>
    : public AscendC::Std::true_type {};

template <class TileCopy, typename Enable = void>
struct HasCopyCo1ToOutWithParams : public AscendC::Std::false_type {};

template <class TileCopy>
struct HasCopyCo1ToOutWithParams<TileCopy, std::void_t<typename TileCopy::template CopyCo1ToOut<void, void, void>>>
    : public AscendC::Std::true_type {};
} // namespace Detail

template <class TileCopy>
struct HasCopyGmToA1 : public AscendC::Std::bool_constant<Detail::HasCopyGmToA1WithParams<TileCopy>::value> {};

template <class TileCopy>
inline constexpr bool HasCopyGmToA1V = HasCopyGmToA1<TileCopy>::value;

template <class TileCopy>
struct HasCopyGmToB1 : public AscendC::Std::bool_constant<Detail::HasCopyGmToB1WithParams<TileCopy>::value> {};

template <class TileCopy>
inline constexpr bool HasCopyGmToB1V = HasCopyGmToB1<TileCopy>::value;

template <class TileCopy>
struct HasCopyCo1ToOut : public AscendC::Std::bool_constant<Detail::HasCopyCo1ToOutWithParams<TileCopy>::value> {};

template <class TileCopy>
inline constexpr bool HasCopyCo1ToOutV = HasCopyCo1ToOut<TileCopy>::value;
} // namespace Tile
} // namespace Gemm
} // namespace Cmct
