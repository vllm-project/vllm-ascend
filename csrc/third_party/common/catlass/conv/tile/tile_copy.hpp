/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_TILE_TILE_COPY_HPP
#define CATLASS_CONV_TILE_TILE_COPY_HPP

#include <type_traits>
#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"
#include "catlass/conv/tile/copy_gm_to_l1.hpp"
#include "catlass/conv/tile/copy_l0c_to_gm.hpp"
#include "catlass/conv/tile/copy_l1_to_l0a.hpp"
#include "catlass/conv/tile/copy_l1_to_l0b.hpp"
#include "catlass/gemm/helper.hpp"

namespace Catlass::Conv::Tile {

template <
    /// Tag indicating architecture
    class ArchTag,
    /// ConvType for Fmap operand
    class FmapType,
    /// ConvType type for Filter operand
    class FilterType,
    /// ConvType type for Output operand
    class OutputType,
    /// ConvType type for Bias operand
    class BiasType = void
>
struct TileCopy {
    using ElementFmap = typename FmapType::Element;
    using ElementFilter = typename FilterType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementFmap, ElementFilter>::ElementAccumulator;

    using CopyGmToL1A = Conv::Tile::CopyGmToL1<ArchTag, FmapType>;
    using CopyGmToL1B = Conv::Tile::CopyGmToL1<ArchTag, FilterType>;
    using CopyL1ToL0A = Conv::Tile::CopyL1ToL0A<
        ArchTag, typename Gemm::helper::L1ATypeSelector<FmapType>::L1AType>;
    using CopyL1ToL0B = Conv::Tile::CopyL1ToL0B<
        ArchTag, typename Gemm::helper::L1BTypeSelector<FilterType>::L1BType>;
    using CopyL0CToGm = Conv::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, OutputType>;
};

} // namespace Catlass::Conv::Tile

#endif // CATLASS_CONV_TILE_TILE_COPY_HPP
