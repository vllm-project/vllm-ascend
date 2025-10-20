/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_TILE_TILE_COPY_HPP
#define CATLASS_GEMV_TILE_TILE_COPY_HPP

#include "catlass/catlass.hpp"
#include "catlass/detail/tag_to_layout.hpp"

#include "catlass/gemv/tile/vec_copy_gm_to_ub.hpp"
#include "catlass/gemv/tile/vec_copy_ub_to_gm.hpp"
#include "catlass/gemv/tile/matrix_copy_gm_to_ub.hpp"

#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"

#include "catlass/gemm/helper.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace Catlass::Gemv::Tile {

template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyGemvAiv {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aiv
    using VecCopyGmToUb = Gemv::Tile::VecCopyGmToUB<ArchTag, XType>;
    static constexpr bool is_atoadd = Gemv::helper::AtomicAddSelector<AType>::value;
    using VecCopyUbToGm = Gemv::Tile::VecCopyUBToGm<ArchTag, YType,is_atoadd>;
    using MatrixCopyGmToUb = Gemv::Tile::MatrixCopyGmToUB<ArchTag, AType>;
};


template <
    /// Tag indicating architecture
    class ArchTag,
    /// MatmulType for A matrix operand
    class AType,
    /// MatmulType type for X vector operand
    class XType,
    /// MatmulType type for Y vector operand
    class YType,
    /// MatmulTpe type for Bias operand
    class BiasType = void
>
struct TileCopyGemvAic {
    using ElementA = typename AType::Element;
    using ElementX = typename XType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    // the function of aic
    using L1XType = typename Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1AType;
    using L1AType = typename Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L1BType;
    using L0AType = typename Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0AType;
    using L0BType = typename Gemv::helper::L1AndL0TypeSelectorGemv<XType, AType>::L0BType;


    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, XType, L1XType>;   
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, AType, L1AType>;   

    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, L1XType, L0AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, L1AType, L0BType>; 
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, YType>;

};

} // namespace Catlass::Gemv::Tile

#endif // CATLASS_GEMV_TILE_TILE_COPY_HPP
