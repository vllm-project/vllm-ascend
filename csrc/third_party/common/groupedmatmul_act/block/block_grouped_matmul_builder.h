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
 * \file block_grouped_matmul_builder.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_GROUPED_MATMUL_BUILDER_H
#define MATMUL_BLOCK_BLOCK_GROUPED_MATMUL_BUILDER_H

#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "block_mmad_multi_block_bias.h"

#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tuple_utils.h"

#include "../block/block_mmad.h"
#include "../policy/dispatch_policy.h"

namespace Act {
namespace Gemm {
namespace Block {
template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_,
          class LayoutBias_, class L1TileShape_, class L0TileShape_, class BlockScheduler_,
          class BlockMatmulPolicy_ = MatmulMultiBlockBias<>, typename Enable_ = void>
class BlockGroupedMatmulBuilder {
    static_assert(AscendC::Std::always_false_v<BlockMatmulPolicy_>,
                  "BlockGroupedMatmulBuilder is not implemented for this BlockMatmulPolicy");
};

template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_,
          class LayoutBias_, class L1TileShape_, class L0TileShape_, class BlockScheduler_, class BlockMatmulPolicy_>
class BlockGroupedMatmulBuilder<AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_,
                                L1TileShape_, L0TileShape_, BlockScheduler_, BlockMatmulPolicy_,
                                AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<MatmulMultiBlockWithLayout<>,
                                                          BlockMatmulPolicy_> ||
                                                          AscendC::Std::is_base_of_v<MatmulMultiBlockBias<>,
                                                          BlockMatmulPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using BlockMatmulPolicy = BlockMatmulPolicy_;

    // transA and transB are deduced from LayoutA and LayoutB
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    static constexpr CubeFormat formatA = TagToFormat<LayoutA>::format;
    static constexpr CubeFormat formatB = TagToFormat<LayoutB>::format;
    static constexpr CubeFormat formatC = TagToFormat<LayoutC>::format;
    static constexpr CubeFormat formatBias = TagToFormat<LayoutBias>::format;

    using AMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, formatA, AType, transA>;
    using BMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, formatB, BType, transB>;
    using CMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, formatC, CType>;
    using BiasMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, formatBias, BiasType>;

    using BlockMmadOp =
        Block::BlockMmad<BlockMatmulPolicy, L1TileShape, L0TileShape, AMatmulType, BMatmulType, CMatmulType,
                         BiasMatmulType, Tile::TileCopy<Arch::Ascend910_95, Tile::CopyWithParams>>;

    static constexpr int64_t l1M = GetIntegralConstant<MNK_M, L1TileShape>();
    static constexpr int64_t l1N = GetIntegralConstant<MNK_N, L1TileShape>();
    static constexpr int64_t l1K = GetIntegralConstant<MNK_K, L1TileShape>();

    static constexpr int64_t l0M = GetIntegralConstant<MNK_M, L0TileShape>();
    static constexpr int64_t l0N = GetIntegralConstant<MNK_N, L0TileShape>();
    static constexpr int64_t l0K = GetIntegralConstant<MNK_K, L0TileShape>();

    // host side kernel arguments
    struct Arguments {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR groupListGmAddr{nullptr};
    };

    // params
    using Params = Arguments;

    __aicore__ inline BlockGroupedMatmulBuilder() {}

    __aicore__ inline ~BlockGroupedMatmulBuilder() {}

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
        Params params = {args.aGmAddr, args.bGmAddr, args.cGmAddr, args.biasGmAddr};
        return params;
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif