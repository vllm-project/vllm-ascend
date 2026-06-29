/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_TILE_COPY_HPP
#define CATLASS_GEMM_TILE_TILE_COPY_HPP

#include <type_traits>
#include "../../../template_linear_algebra_v2/catlass.hpp"
#include "../../../template_linear_algebra_v2/gemm/tile/copy_gm_to_l1.hpp"
#include "../../../template_linear_algebra_v2/gemm/tile/copy_l1_to_fp.hpp"
#include "../../../template_linear_algebra_v2/gemm/tile/copy_l0c_to_gm.hpp"
#include "../../../template_linear_algebra_v2/gemm/tile/copy_l1_to_l0a.hpp"
#include "../../../template_linear_algebra_v2/gemm/tile/copy_l1_to_l0b.hpp"
#include "../../../template_linear_algebra_v2/gemm/tile/copy_l1_to_bt.hpp"
#include "../../../template_linear_algebra_v2/gemm/tile/copy_gm_to_ub.hpp"
#include "../../../template_linear_algebra_v2/gemm/tile/copy_ub_to_gm.hpp"
#include "../../../template_linear_algebra_v2/gemm/tile/cast_int4_to_int8.hpp"
#include "../../../template_linear_algebra_v2/gemm/tile/cast_int8_to_fp16.hpp"
#include "../../../template_linear_algebra_v2/gemm/tile/cast_fp8_to_fp16.hpp"
#include "../../../template_linear_algebra_v2/gemm/helper.hpp"


namespace Catlass::Gemm::Tile {

template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void
>
struct TileCopy {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;
    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<
        ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<
        ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
    using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyGmToL1<ArchTag,
            typename BiasTypeSelector::GMBiasType,
            typename BiasTypeSelector::L1BiasType>>;
    using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyL1ToBT<ArchTag,
            typename BiasTypeSelector::L1BiasType,
            typename BiasTypeSelector::L0BiasType>>;
};

template <
    class ArchTag,
    class AType,
    class BType,
    class CType,
    class PrologueA_,
    class PrologueB_,
    class BiasType = void
>
struct TileCopyWithPrologueDeqPerTensor {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;

    using PrologueA = PrologueA_;
    using PrologueB = PrologueB_;

    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<
        ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<
        ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<
        ArchTag, ElementAccumulator, CType, Tile::ScaleGranularity::PER_TENSOR>;
    using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyGmToL1<ArchTag,
            typename BiasTypeSelector::GMBiasType,
            typename BiasTypeSelector::L1BiasType>>;
    using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyL1ToBT<ArchTag,
            typename BiasTypeSelector::L1BiasType,
            typename BiasTypeSelector::L0BiasType>>;
};

template <
    class ArchTag,
    class AType,
    class BType,
    class CType,
    class PrologueA_,
    class PrologueB_,
    class BiasType = void
>
struct TileCopyWithPrologue {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;

    using PrologueA = PrologueA_;
    using PrologueB = PrologueB_;

    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<
        ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<
        ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
    using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyGmToL1<ArchTag,
            typename BiasTypeSelector::GMBiasType,
            typename BiasTypeSelector::L1BiasType>>;
    using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyL1ToBT<ArchTag,
            typename BiasTypeSelector::L1BiasType,
            typename BiasTypeSelector::L0BiasType>>;
};

///////////////////////////////////
/// new add
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmTpe type for Bias operand
    class BiasType = void
>
struct TileCopyGemm {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    // change structual
    using L1AType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L1AType;
    using L1BType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L1BType;
    using L0AType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L0AType;
    using L0BType = typename Gemm::helper::L1AndL0TypeSelectorGemm<AType, BType>::L0BType;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType, L1BType>;
    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, L1AType, L0AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, L1BType, L0BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
};
//////////////////////////////
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType
>
struct ConvTileCopy {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;
    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<
        ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<
        ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
    using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyGmToL1<ArchTag,
            typename BiasTypeSelector::GMBiasType,
            typename BiasTypeSelector::L1BiasType>>;
    using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyL1ToBT<ArchTag,
            typename BiasTypeSelector::L1BiasType,
            typename BiasTypeSelector::L0BiasType>>;
};

// fixpipe开启relu开关
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void
>
struct ReluTileCopy : public TileCopy<ArchTag, AType, BType, CType, BiasType> {
    // 重写 CopyL0CToGm
    using ElementAccumulator = typename TileCopy<ArchTag, AType, BType, CType, BiasType>::ElementAccumulator;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType,
        Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT, true>;
};

// fixpipe开启随路量化
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void,
    /// GemmType type for Bias operand
    ScaleGranularity SCALE_GRANU = ScaleGranularity::PER_TENSOR
>
struct QuantTileCopy : public TileCopy<ArchTag, AType, BType, CType, BiasType> {
    // 重写 CopyL0CToGm
    using ElementAccumulator = typename TileCopy<ArchTag, AType, BType, CType, BiasType>::ElementAccumulator;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType, SCALE_GRANU, false>;

    using CopyGmToL1Scale = Gemm::Tile::CopyGmToL1<ArchTag,
        Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::GM>,
        Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::A1>
    >;

    using CopyL1ToFP = Gemm::Tile::CopyL1ToFP<ArchTag,
        Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::A1>,
        Gemm::GemmType<uint64_t, layout::VectorLayout, AscendC::TPosition::C2PIPE2GM>
    >;
};

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_TILE_COPY_HPP
