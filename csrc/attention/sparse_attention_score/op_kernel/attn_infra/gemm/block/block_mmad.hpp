/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_MMAD_HPP
#define BLOCK_MMAD_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/tile_mmad.hpp"

namespace NpuArch::Gemm::Block {

#if (__CCE_AICORE__ == 220)
template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};
#endif

#if (__CCE_AICORE__ == 310)
template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class ElementA,
    class ElementB,
    class ElementC,
    class ElementBias = void,
    class TileCopy = Gemm::Tile::PackedTileCopyTla<typename DispatchPolicy::ArchTag, ElementA, layout::RowMajor,
        ElementB, layout::RowMajor, ElementC, layout::RowMajor, ElementBias>,
    class TileMmad =
        Gemm::Tile::TileMmadTla<typename DispatchPolicy::ArchTag, ElementA, typename TileCopy::LayoutTagL1A>
>
struct BlockMmadTla {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadTla is not implemented for this DispatchPolicy");
};
#endif

} // namespace NpuArch::Gemm::Block
#if (__CCE_AICORE__ == 220)
#include "../../../attn_infra/gemm/block/block_mmad_qk.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_pv.hpp"
#endif
#if (__CCE_AICORE__ == 310)
#include "../../../attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_qk_arch35_ABf16_C_to_UB.hpp"
#endif
#endif // GEMM_BLOCK_BLOCK_MMAD_HPP