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
 * \file block_quant_matmul_builder.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_QUANT_MATMUL_BUILDER_H
#define MATMUL_BLOCK_BLOCK_QUANT_MATMUL_BUILDER_H

#include "kernel_operator.h"

#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tuple_utils.h"

#include "../tile/tile_copy.h"
#include "./block_mmad.h"
#include "./block_quant_mmad_multi_block.h"
#include "../policy/dispatch_policy.h"

using namespace AscendC;

namespace Act {
namespace Gemm {
namespace Block {
template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class L1TileShape_,
          class L0TileShape_, class BlockScheduler_, class BlockMatmulPolicy_ = QuantMatmulMultiBlock<>,
          typename Enable_ = void>
class BlockQuantMatmulBuilder {
    static_assert(AscendC::Std::always_false_v<BlockMatmulPolicy_>,
                  "BlockQuantMatmulBuilder is not implemented for this BlockMatmulPolicy");
};

template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class L1TileShape_,
          class L0TileShape_, class BlockScheduler_, class BlockMatmulPolicy_>
class BlockQuantMatmulBuilder<AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, L1TileShape_, L0TileShape_,
                              BlockScheduler_, BlockMatmulPolicy_,
                              AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<QuantMatmulMultiBlock<>, BlockMatmulPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using BlockMatmulPolicy = BlockMatmulPolicy_;

    // transA and transB are deduced from LayoutA and LayoutB
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    static constexpr CubeFormat formatA = TagToFormat<LayoutA>::format;
    static constexpr CubeFormat formatB = TagToFormat<LayoutB>::format;
    static constexpr CubeFormat formatC = TagToFormat<LayoutC>::format;

    using AMatmulType =
        matmul::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, formatA, AType, transA>;
    using BMatmulType =
        matmul::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, formatB, BType, transB>;
    using CMatmulType = MatmulType<AscendC::TPosition::GM, formatC, CType>;
    using BiasMatmulType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>; // null, placeholder

    using BlockMmadOp =
        Block::BlockMmad<BlockMatmulPolicy, L1TileShape, L0TileShape, AMatmulType, BMatmulType, CMatmulType,
                         BiasMatmulType, Tile::TileCopy<Arch::Ascend910_95, Tile::CopyWithParams>>;

    static constexpr int64_t l1M = GetIntegralConstant<0, L1TileShape>();
    static constexpr int64_t l1N = GetIntegralConstant<1, L1TileShape>();
    static constexpr int64_t l1K = GetIntegralConstant<2, L1TileShape>();

    static constexpr int64_t l0M = GetIntegralConstant<0, L0TileShape>();
    static constexpr int64_t l0N = GetIntegralConstant<1, L0TileShape>();
    static constexpr int64_t l0K = GetIntegralConstant<2, L0TileShape>();

    // host side kernel arguments
    struct Arguments {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR groupListGmAddr{nullptr};
    };

    // params
    using Params = Arguments;

    __aicore__ inline BlockQuantMatmulBuilder() {}

    __aicore__ inline ~BlockQuantMatmulBuilder() {}

    __host_aicore__ static size_t GetWorkspaceSize()
    {
        return 0;
    }

    __host_aicore__ static Status CanImplement(Arguments const& args)
    {
        if (l0M * l0K * sizeof(AType) * DOUBLE_BUFFER_COUNT > L0A_SIZE ||
            l0N * l0K * sizeof(BType) * DOUBLE_BUFFER_COUNT > L0B_SIZE || l0M * l0N * sizeof(CType) > L0C_SIZE ||
            (l1M * l1K * sizeof(AType) + l1K * l1N * sizeof(BType)) * DOUBLE_BUFFER_COUNT > L1_SIZE) {
            return Status::tileShapeErrorExceedsLimit;
        }
        return Status::success;
    }

    __host_aicore__ static Params InitParams(Arguments args)
    {
        Params params = {args.aGmAddr, args.bGmAddr, args.x1ScaleGmAddr, args.x2ScaleGmAddr, args.cGmAddr};
        return params;
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif