/**
 * Copyright (c) 2025 Tianjin University, Ltd.
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
#ifndef MATMUL_TILE_TILE_COPY_H
#define MATMUL_TILE_TILE_COPY_H

#include "copy_in/copy_in_with_params.h"
#include "copy_in/copy_in_with_layout.h"

#include "copy_out/copy_out_with_params_v220.h"
#include "copy_out/copy_out_to_gm_with_params_v310.h"
#include "copy_out/copy_out_to_ub_with_params_v310.h"
#include "copy_out/copy_out_split_m_with_params.h"
#include "copy_out/copy_out_split_n_with_params.h"
#include "copy_out/copy_out_with_layout.h"
#include "copy_out/copy_out_enunitflag_with_layout.h"

#include "load_data/copy_a1_to_a2_with_layout.h"
#include "load_data/copy_b1_to_b2_sparse_with_layout.h"
#include "load_data/copy_b1_to_b2_with_layout.h"
#include "tile_copy_policy.h"

namespace Act {
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
struct TileCopy<Arch::Ascend910B, CopyEnUnitFlagWithLayout> {
    using ArchTag = Arch::Ascend910B;
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
struct TileCopy<Arch::Ascend910B, CopySparseWithLayout> {
    using ArchTag = Arch::Ascend910B;
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
struct TileCopy<Arch::Ascend910_95, CopyInAndCopyOutSplitMWithParams> {
    using ArchTag = Arch::Ascend910_95;
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
 * @brief Specialization of TileCopy for CopyOutSplitMWithParams dispatch policy
 * @param [in] ArchTag: architecture tag
 * @param [in] CopyOutSplitMWithParams: copy out split m  policy with params
 */
template <>
struct TileCopy<Arch::Ascend910_95, CopyOutSplitMWithParams> {
    using ArchTag = Arch::Ascend910_95;

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
struct TileCopy<Arch::Ascend910_95, CopyOutSplitNWithParams> {
    using ArchTag = Arch::Ascend910_95;

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
struct TileCopy<Arch::Ascend910_95, CopyNoGmIn> {
    using ArchTag = Arch::Ascend910_95;
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
struct TileCopy<Arch::Ascend910_95, CopyBasedBaseK> {
    using CopyPolicy = CopyBasedBaseK;
    using ArchTag = Arch::Ascend910_95;

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
} // namespace Act
#endif
