/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad_builder.h
 * \brief
 */

#pragma once

#include <type_traits>

#define ASCENDC_CUBE_ONLY
#include "../utils/host_utils.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tuple_utils.h"

#include "block_mmad.h"
#include "../policy/dispatch_policy.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_,
          class LayoutBias_, class L1TileShape_, class L0TileShape_, class BlockScheduler_,
          class BlockMatmulPolicy_ = MatmulMultiBlockBias<>, class TileCopyParam_ = void, typename Enable_ = void>
class BlockMmadBuilder {
    static_assert(AscendC::Std::always_false_v<BlockMatmulPolicy_>,
                  "BlockMmadBuilder is not implemented for this BlockMatmulPolicy");
};

template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_,
          class LayoutBias_, class L1TileShape_, class L0TileShape_, class BlockScheduler_, class BlockMatmulPolicy_,
          class TileCopyParam_>
class BlockMmadBuilder<
    AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_, L1TileShape_, L0TileShape_,
    BlockScheduler_, BlockMatmulPolicy_, TileCopyParam_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithLayout<>, BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulNaivePipelineWithLayout<>, BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithOutQue<>, BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE>,
                                   BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, B_FULL_LOAD_MODE>,
                                   BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_ADD>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE, OP_TYPE_ADD>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, B_FULL_LOAD_MODE, OP_TYPE_ADD>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_MUL>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE, OP_TYPE_MUL>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, B_FULL_LOAD_MODE, OP_TYPE_MUL>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_RELU>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE, OP_TYPE_RELU>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, B_FULL_LOAD_MODE, OP_TYPE_RELU>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_QUANT>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE, OP_TYPE_QUANT>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, B_FULL_LOAD_MODE, OP_TYPE_QUANT>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_RELU_QUANT>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE, OP_TYPE_RELU_QUANT>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, B_FULL_LOAD_MODE, OP_TYPE_RELU_QUANT>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_GELU_ERF>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_GELU_TANH>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_SILU>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, NONE_FULL_LOAD_MODE, OP_TYPE_SWIGLU_SINGLE>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulIterBatch<MatMulL0C2Out::ON_THE_FLY, AscendC::Shape<_0, _0, _0, _0>, OP_TYPE_EMPTY>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulIterBatch<MatMulL0C2Out::ON_THE_FLY, AscendC::Shape<_0, _0, _0, _0>, OP_TYPE_RELU>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulIterBatch<MatMulL0C2Out::ON_THE_FLY, AscendC::Shape<_0, _0, _0, _0>, OP_TYPE_QUANT>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulIterBatch<MatMulL0C2Out::ON_THE_FLY, AscendC::Shape<_0, _0, _0, _0>, OP_TYPE_RELU_QUANT>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMergeBatch<>, BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMergeBatch<AscendC::Shape<_0, _0, _0, _0>, OP_TYPE_RELU>,
                                   BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMergeBatch<AscendC::Shape<_0, _0, _0, _0>, OP_TYPE_ADD>, BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMergeBatch<AscendC::Shape<_0, _0, _0, _0>, OP_TYPE_MUL>, BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulIterBatch<MatMulL0C2Out::ND_FIXPIPE_1_2, AscendC::Shape<_0, _0, _0, _0>, OP_TYPE_EMPTY>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulIterBatch<MatMulL0C2Out::ND_FIXPIPE_1_2, AscendC::Shape<_0, _0, _0, _0>, OP_TYPE_ADD>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulIterBatch<MatMulL0C2Out::ND_FIXPIPE_1_2, AscendC::Shape<_0, _0, _0, _0>, OP_TYPE_MUL>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<
            MatmulIterBatch<MatMulL0C2Out::ND_FIXPIPE_1_2, AscendC::Shape<_0, _0, _0, _0>, OP_TYPE_RELU>,
            BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlock<>, BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulFlatQuant<>, BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulRotateQuant<>, BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY>, BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2>, BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_RELU>,
                                   BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_RELU>,
                                   BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_ADD>,
                                   BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_ADD>,
                                   BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_MUL>,
                                   BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_MUL>,
                                   BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<BatchMatmulToMul<>, BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulToMul<>, BlockMatmulPolicy_> ||
        AscendC::Std::is_base_of_v<MatmulToVector<>, BlockMatmulPolicy_>>> {
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
    using BlockMatmulPolicy = BlockMatmulPolicy_;
    using TileCopyParam = TileCopyParam_;
    // transA and transB are deduced from LayoutA and LayoutB
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    static constexpr CubeFormat formatA = TagToFormat<LayoutA>::format;
    static constexpr CubeFormat formatB = TagToFormat<LayoutB>::format;
    static constexpr CubeFormat formatC = TagToFormat<LayoutC>::format;
    static constexpr CubeFormat formatBias = TagToFormat<LayoutBias_>::format;

    using AMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, formatA, AType, transA>;
    using BMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, formatB, BType, transB>;
    using CMatmulType = AscendC::MatmulType<AscendC::TPosition::VECIN, formatC, CType>;
    using BiasMatmulType = AscendC::MatmulType<AscendC::TPosition::GM, formatBias, BiasType>;

    // AType -> AMatmulType
    using BlockMmadOp = Block::BlockMmad<BlockMatmulPolicy, L1TileShape, L0TileShape, AMatmulType, BMatmulType,
                                         CMatmulType, BiasMatmulType, TileCopyParam>;

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
        GM_ADDR workspaceGmAddr{nullptr};
        GM_ADDR quantScaleGmAddr{nullptr};
    };

    // params
    using Params = Arguments;

    __aicore__ inline BlockMmadBuilder() {}

    __aicore__ inline ~BlockMmadBuilder() {}

    __host_aicore__ static size_t GetWorkspaceSize() { return 0; }

    __host_aicore__ static Status CanImplement(Arguments const& args)
    {
        if (AscendC::Std::is_same_v<bfloat16_t, AType> && args.biasGmAddr != nullptr) {
            return Status::bf16BiasErrorInvalidDataType;
        }

        if (l0M * l0K * sizeof(AType) * DOUBLE_BUFFER_COUNT > L0A_SIZE ||
            l0N * l0K * sizeof(BType) * DOUBLE_BUFFER_COUNT > L0B_SIZE || l0M * l0N * sizeof(CType) > L0C_SIZE ||
            (l1M * l1K * sizeof(AType) + l1K * l1N * sizeof(BType)) * DOUBLE_BUFFER_COUNT > L1_SIZE) {
            return Status::tileShapeErrorExceedsLimit;
        }
        return Status::success;
    }

    __host_aicore__ static Params InitParams(const Arguments& args) { return args; }
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
