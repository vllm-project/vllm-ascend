/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_TILE_COPY_HPP
#define CATLASS_GEMM_TILE_TILE_COPY_HPP

#include <type_traits>
#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "tla/tensor.hpp"
namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    class Enable = void
>
struct TileCopyTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopyTla, can not find the specialization.");
};

// Extended template for TileCopyTla that supports manually specifying LayoutTagSrc and LayoutTagDst.
// Users can specialize the copy class by LayoutTagSrc and LayoutTagDst.
template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    class LayoutTagSrc,
    class LayoutTagDst
>
struct TileCopyTlaExt {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopyTlaExt, can not find the specialization.");
};

} // namespace Catlass::Gemm::Tile

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
#include "catlass/gemm/tile/copy_l1_to_bt.hpp"
#include "catlass/gemm/tile/copy_gm_to_ub.hpp"
#include "catlass/gemm/tile/copy_ub_to_gm.hpp"
#include "catlass/gemm/helper.hpp"


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
    /// Tag indicating architecture
    class ArchTag,
    class TensorA,
    class LayoutTagA,
    class TensorB,
    class LayoutTagB,
    class TensorC,
    class LayoutTagC,
    class TensorBias = void,
    class LayoutTagBias = void
>
struct PackedTileCopyTla {
    using ElementA = typename TensorA::Element;
    using ElementB = typename TensorB::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using LayoutL1A = detail::TagToLayout_t<ElementA,
        typename helper::L1ATypeSelector<Gemm::GemmType<ElementA, LayoutTagA>>::L1AType::Layout>;
    using LayoutL1B = detail::TagToLayout_t<ElementB,
        typename helper::L1BTypeSelector<Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout>;
    using LayoutL0A = detail::TagToLayout_t<ElementA, layout::zZ>;
    using LayoutL0B = detail::TagToLayout_t<ElementB, layout::nZ>;
    using LayoutL0C = typename detail::LayoutL0C;

    using TensorL1A = tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, AscendC::TPosition::A1>;
    using TensorL1B = tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, AscendC::TPosition::A1>;
    using TensorL0A = tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, AscendC::TPosition::A2>;
    using TensorL0B = tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, AscendC::TPosition::B2>;
    using TensorL0C = tla::Tensor<AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, AscendC::TPosition::CO1>;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutTagB>;

    using CopyGmToL1A = Gemm::Tile::TileCopyTla<ArchTag, TensorA, TensorL1A>;
    using CopyGmToL1B = Gemm::Tile::TileCopyTla<ArchTag, TensorB, TensorL1B>;
    using CopyL1ToL0A = Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC>;
};

template <
    /// Tag indicating architecture
    class ArchTag,
    class TensorA,
    class LayoutTagA,
    class TensorB,
    class LayoutTagB,
    class TensorC,
    class LayoutTagC,
    class TensorBias = void,
    class LayoutTagBias = void,
    bool IS_PADDING_A = false,
    bool IS_PADDING_B = false
>
struct PaddingPackedTileCopyTla {
    static_assert(std::is_same_v<LayoutTagA, layout::RowMajor> || std::is_same_v<LayoutTagA, layout::ColumnMajor>,
        "Unsupported layout, only can be RowMajor and ColumnMajor");
    static_assert(std::is_same_v<LayoutTagB, layout::RowMajor> || std::is_same_v<LayoutTagB, layout::ColumnMajor>,
        "Unsupported layout, only can be RowMajor and ColumnMajor");
    using ElementA = typename TensorA::Element;
    using ElementB = typename TensorB::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using LayoutTagL1A = typename helper::L1ATypeSelector<Gemm::GemmType<ElementA, LayoutTagA>>::L1AType::Layout;
    using LayoutTagL1B = typename helper::L1BTypeSelector<Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
    using LayoutL1A = detail::TagToLayout_t<ElementA, LayoutTagL1A>;
    using LayoutL1B = detail::TagToLayout_t<ElementB, LayoutTagL1B>;
    using LayoutL0A = detail::TagToLayout_t<ElementA, layout::zZ>;
    using LayoutL0B = detail::TagToLayout_t<ElementB, layout::nZ>;
    using LayoutL0C = typename detail::LayoutL0C;

    using TensorL1A = tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, AscendC::TPosition::A1>;
    using TensorL1B = tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, AscendC::TPosition::A1>;
    using TensorL0A = tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, AscendC::TPosition::A2>;
    using TensorL0B = tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, AscendC::TPosition::B2>;
    using TensorL0C = tla::Tensor<AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, AscendC::TPosition::CO1>;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutTagB>;

    using LayoutPaddingTagA = std::conditional_t<std::is_same_v<LayoutTagA, layout::RowMajor>,
        layout::PaddingRowMajor, layout::PaddingColumnMajor>;
    using LayoutPaddingTagB = std::conditional_t<std::is_same_v<LayoutTagB, layout::RowMajor>,
        layout::PaddingRowMajor, layout::PaddingColumnMajor>;

    using CopyGmToL1A = std::conditional_t<
        IS_PADDING_A,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorA, TensorL1A, LayoutPaddingTagA, LayoutTagL1A>,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorA, TensorL1A, LayoutTagA, LayoutTagL1A>
    >;
    using CopyGmToL1B = std::conditional_t<
        IS_PADDING_B,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorB, TensorL1B, LayoutPaddingTagB, LayoutTagL1B>,
        Gemm::Tile::TileCopyTlaExt<ArchTag, TensorB, TensorL1B, LayoutTagB, LayoutTagL1B>
    >;

    using CopyL1ToL0A = Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC>;
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
} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_TILE_COPY_HPP
