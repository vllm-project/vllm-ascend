/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_BLOCK_BLOCK_CONV_HPP
#define CATLASS_CONV_BLOCK_BLOCK_CONV_HPP

#include "catlass/catlass.hpp"
#include "catlass/conv/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace Catlass::Conv::Block {

template <
    class DispatchPolicy,
    class CoreTileShape,
    class FmapL1TileShape,
    class FilterL1TileShape,
    class L0TileShape,
    class FmapType,
    class FilterType,
    class OutType,
    class BiasType,
    class TileCopy = Gemm::Tile::ConvTileCopy<typename DispatchPolicy::ArchTag, FmapType, FilterType, OutType, BiasType>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, FmapType, FilterType, BiasType>
>
struct BlockConv {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockConv is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class FmapL1TileShape,
    class FilterL1TileShape,
    class L0TileShape,
    class FmapType,
    class FilterType,
    class OutputType,
    class BiasType = void,
    class TileCopy = Conv::Tile::TileCopy<typename DispatchPolicy::ArchTag, FmapType, FilterType, OutputType, BiasType>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, FmapType, FilterType, BiasType>
>
struct BlockConv2d {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockConv2d is not implemented for this DispatchPolicy");
};

} // namespace Catlass::Conv::Block

#include "catlass/conv/block/block_conv3d_pingpong_bias.hpp"
#include "catlass/conv/block/block_conv2d_pingpong.hpp"

#endif // CATLASS_CONV_BLOCK_BLOCK_CONV_HPP