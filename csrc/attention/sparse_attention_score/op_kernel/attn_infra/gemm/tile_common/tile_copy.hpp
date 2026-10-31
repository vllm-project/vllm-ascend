/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TILE_COPY_HPP
#define TILE_COPY_HPP

#include <type_traits>
#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/detail/tag_to_layout.hpp"
#include "../../../tla/tensor.hpp"
#if (__CCE_AICORE__ == 310)
#include "../../../attn_infra/gemm/tile_common/copy_gm_to_l1_a5.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l0c_to_ub_a5.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l1_to_l0a_a5.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l1_to_l0b_a5.hpp"
#endif
#if (__CCE_AICORE__ == 220)
#include "../../../attn_infra/gemm/tile_common/copy_gm_to_l1_a2.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l0c_to_gm_a2.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l1_to_l0a_a2.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l1_to_l0b_a2.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l1_to_bt.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_gm_to_ub.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_ub_to_gm.hpp"
#endif
#include "../../../attn_infra/gemm/helper.hpp"


namespace NpuArch::Gemm::Tile {

#if (__CCE_AICORE__ == 220)
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
};
#endif

template <
    /// Tag indicating architecture
    class ArchTag,
    class ElementA_,
    class LayoutTagA_,
    class ElementB_,
    class LayoutTagB_,
    class ElementC_,
    class LayoutTagC_,
    class ElementBias = void,
    bool ReluEnable_ = false,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    class L0CCopyMode = CopyToGM
>
struct PackedTileCopyTla {
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using LayoutTagA = LayoutTagA_;
    using LayoutTagB = LayoutTagB_;
    using LayoutTagC = LayoutTagC_;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    static constexpr bool ReluEnable = ReluEnable_;

    static constexpr bool HAS_BIAS = !std::is_void_v<ElementBias>;

    using LayoutTagL1A = typename helper::L1ATypeSelector<Gemm::GemmType<ElementA, LayoutTagA>>::L1AType::Layout;
    using LayoutTagL1B = typename helper::L1BTypeSelector<Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
    using LayoutTagL0A = typename helper::L0ALayoutSelector<ArchTag>::Layout;
    using LayoutTagL0B = layout::nZ;

    using LayoutA = detail::TagToLayout_t<ElementA, LayoutTagA>;
    using LayoutB = detail::TagToLayout_t<ElementB, LayoutTagB>;
    using LayoutC = detail::TagToLayout_t<ElementC_, LayoutTagC>;

    using LayoutL1A = detail::TagToLayout_t<ElementA, LayoutTagL1A>;
    using LayoutL1B = detail::TagToLayout_t<ElementB, LayoutTagL1B>;
    using LayoutL0A = detail::TagToLayout_t<ElementA, LayoutTagL0A>;
    using LayoutL0B = detail::TagToLayout_t<ElementB, LayoutTagL0B>;
    using LayoutL0C = typename detail::LayoutL0C;

    using TensorL1AVectorLayout =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0>, AscendC::TPosition::A1>;
    using TensorL1ALayout =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    
    using TensorL1A = std::conditional_t<tla::detail::isVector<LayoutTagA>::value, TensorL1AVectorLayout, TensorL1ALayout>;
    using TensorL1B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL0A =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A2>;
    using TensorL0B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::B2>;
    using TensorL0C = tla::Tensor<AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, tla::Coord<tla::_0, tla::_0>,
        AscendC::TPosition::CO1>;
    using TensorL1Bias = std::conditional_t<
        HAS_BIAS,
        tla::Tensor<AscendC::LocalTensor<ElementBias>, detail::TagToLayout_t<ElementBias, layout::VectorLayout>,
                    tla::Coord<tla::_0>, AscendC::TPosition::A1>,
        EmptyClass>;
    using TensorL0Bias = tla::Tensor<
        AscendC::LocalTensor<ElementAccumulator>,
        detail::TagToLayout_t<ElementAccumulator, layout::VectorLayout>,
        tla::Coord<tla::_0>,
        AscendC::TPosition::C2>;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutTagB>;

    template <class TensorA>
    using CopyGmToL1A = Gemm::Tile::TileCopyTla<ArchTag, TensorA, TensorL1A>;

    template <class TensorB>
    using CopyGmToL1B = Gemm::Tile::TileCopyTla<ArchTag, TensorB, TensorL1B>;

    template <class TensorBias>
    using CopyGmToL1Bias = std::conditional_t<
        HAS_BIAS,
        Gemm::Tile::TileCopyTla<ArchTag, TensorBias, TensorL1Bias>,
        EmptyClass>;

    using CopyL1ToL0A = Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
    using CopyL1ToBT = std::conditional_t<
        HAS_BIAS,
        Gemm::Tile::TileCopyTla<ArchTag, TensorL1Bias, TensorL0Bias>,
        EmptyClass>;

    template <class TensorC>
    using CopyL0CToDst = Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>;
};

#if (__CCE_AICORE__ == 310)
template <
    /// Tag indicating architecture
    class ArchTag,
    class ElementA_,
    class LayoutTagA,
    class ElementB_,
    class LayoutTagB,
    class ElementC_,
    class LayoutTagC,
    class ElementBias = void,
    CopyL0CToUBMode CopyMode_ = CopyL0CToUBMode::NO_SPLIT,
    bool ReluEnable = false,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT
>
struct PackedTileCopyTlaToUB : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA, ElementB_, LayoutTagB,
                                                        ElementC_, LayoutTagC, ElementBias> {
    static constexpr CopyL0CToUBMode CopyMode = CopyMode_;
    // 重写 CopyL0CToDst
    using TensorL0C = typename PackedTileCopyTla<
        ArchTag,
        ElementA_,
        LayoutTagA,
        ElementB_,
        LayoutTagB,
        ElementC_,
        LayoutTagC,
        ElementBias>::TensorL0C;

    template <class TensorC>
    using CopyL0CToDst =
        Gemm::Tile::CopyL0CToUBTla<ArchTag, TensorL0C, TensorC, CopyMode, DEQUANT_GRANULARITY, ReluEnable>;
};
#endif

//////////////////////////////
} // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_TILE_COPY_HPP