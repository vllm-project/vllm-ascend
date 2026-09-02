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
 * \file kernel_matmul_without_que.h
 * \brief
 */

#pragma once

#define ASCENDC_CUBE_ONLY
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/status_utils.h"
#include "../block/block_mmad_builder.h"
#include "../block/block_mmad_pingpong_without_que.h"
#include "../epilogue/block_epilogue_empty.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_policy.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {

template <class ProblemShape, class BlockMmadBuilder, class BlockEpilogue, class BlockScheduler, typename Enable = void>
class KernelMatmulWithoutQue;

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelMatmulWithoutQue<
    ProblemShape_, BlockMmadBuilder_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<BlockEpilogue_, Block::BlockEpilogueEmpty>>> {
public:
    __aicore__ inline KernelMatmulWithoutQue() {}
    __aicore__ inline ~KernelMatmulWithoutQue() {}

    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    static constexpr bool transA = BlockMmadBuilder::transA;
    static constexpr bool transB = BlockMmadBuilder::transB;
    // schedulerOp
    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<
        ProblemShape, typename BlockMmadBuilder::L1TileShape, typename BlockMmadBuilder::L0TileShape, BlockScheduler,
        transA, transB>::SchedulerOp;
    // mmadOp
    using BlockMmadOp = typename BlockMmadBuilder::BlockMmadOp;
    using BlockMmadArguments = typename BlockMmadBuilder::Arguments;
    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;
    using BlockMmadParams = typename BlockMmadBuilder::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    // come from cann
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;
    using AType = typename BlockMmadBuilder::AType;
    using BType = typename BlockMmadBuilder::BType;
    using CType = typename BlockMmadBuilder::CType;
    using BiasType = typename BlockMmadBuilder::BiasType;
    using LayoutB = typename BlockMmadBuilder::LayoutB;
    using TupleL1L0Shape = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;

    // ND layout
    using NDLayout = AscendC::Layout<AscendC::Shape<int64_t, int64_t>, AscendC::Stride<int64_t, int64_t>>;

    static constexpr CubeFormat formatB = TagToFormat<LayoutB>::format;
    static constexpr bool isNdFormat = (formatB == CubeFormat::ND);
    static constexpr bool isFp32 = (std::is_same_v<BType, float>);

    // no need to have tensortrait
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    // shape
    TupleShape problemShape_{};
    bool isBias_ = false;

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmadArgs;
        BlockEpilogueArguments epilogueArgs;
        Arguments() = default;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        Params() = default;
    };

    __aicore__ inline static TupleShape ToShapeTuple(ProblemShape const& shape)
    {
        return {shape.m, shape.n, shape.k, shape.b};
    }

    __aicore__ inline void Init(Params const& params)
    {
        problemShape_ = ToShapeTuple(params.problemShape);
        BlockMmadParams blockMmadParams_ = params.mmadParams;
        int64_t m = Get<MNK_M>(problemShape_);
        int64_t n = Get<MNK_N>(problemShape_);
        int64_t k = Get<MNK_K>(problemShape_);
        // Init GlobalTensor
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(blockMmadParams_.aGmAddr));
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(blockMmadParams_.bGmAddr));
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(blockMmadParams_.cGmAddr));
        if (blockMmadParams_.biasGmAddr != nullptr) {
            isBias_ = true;
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(blockMmadParams_.biasGmAddr));
        }
    }

    __aicore__ inline void UnsetHf32(bool isHf32)
    {
        if (isHf32) {
            AscendC::SetHF32Mode(0);
            AscendC::SetHF32TransMode(0);
        }
    }

    __host_aicore__ static Status CheckShape(ProblemShape const& shape)
    {
        int64_t m = shape.m;
        int64_t n = shape.n;
        int64_t k = shape.k;
        int64_t b = shape.b;
        if (b > INT32_MAX) {
            return Status::batchErrorExcceedsLimit;
        }
        // Check m, n, k overlimit data type
        if (m > INT32_MAX || n > INT32_MAX || k > INT32_MAX) {
            return Status::mnkErrorExceedsLimit;
        }
        // Check matrix size exceeds limit
        if (!transA && k > MATRIX_INNER_DIM_LIMIT_SIZE) { // mk matrix k limit
            return Status::mkErrorMatrixExceedsLimit;
        }

        if (transA && m > MATRIX_INNER_DIM_LIMIT_SIZE) { // km matrix m limit
            return Status::kmErrorMatrixExceedsLimit;
        }
        if (!transB && n > MATRIX_INNER_DIM_LIMIT_SIZE) { // kn matrix n limit
            return Status::knErrorMatrixExceedsLimit;
        }

        if (transB && k > MATRIX_INNER_DIM_LIMIT_SIZE) { // nk matrix k limit
            return Status::nkErrorMatrixExceedsLimit;
        }
        return Status::success;
    }

    __host_aicore__ static Status CanImplement(Arguments const& args)
    {
        // Check shape in kernel
        CHECK_AND_RETURN(CheckShape(args.problemShape));
        // Check mmad args
        CHECK_AND_RETURN(BlockMmadBuilder::CanImplement(args.mmadArgs));

        return Status::success;
    }

    __host_aicore__ static size_t GetWorkspaceSize(ProblemShape shape, int64_t blockNum)
    {
        size_t workSpaceSize = 0;
        // Calculate extra workspace size for mmad
        workSpaceSize += BlockMmadBuilder::GetWorkspaceSize();

        return workSpaceSize;
    }

    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR workspace)
    {
        BlockMmadParams mmadParams = BlockMmadBuilder::InitParams(args.mmadArgs);
        // mmad params with epiligue takes workspaceGm as output
        Params params = {args.problemShape, mmadParams, {}, {}};
        return params;
    }

    __aicore__ inline void operator()(Params const& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        // Instantiate mmadOp
        BlockMmadOp blockMmadOp;
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        // Init
        Init(params);

        BlockSchedulerOp bs(params.problemShape, curBlockIdx, blockNum, params.schParams, isFp32, isNdFormat);
        if (bs.GetBL2CacheDisable()) {
            bGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if (bs.GetAL2CacheDisable()) {
            aGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        int64_t tileNum = bs.GetTileNum();
        TupleShape tileL1 = bs.GetTileL1Shape();
        TupleShape tileL0 = bs.GetTileL0Shape();
        int64_t realBlockNum = bs.GetBlockNum(params.problemShape, blockNum);
        if (curBlockIdx >= realBlockNum) {
            return;
        }

        // come from the block_mmad_pingpong_without_que.h
#if __NPU_ARCH__ != 5102
        bool isHf32 = bs.GetHf32Flag();
        if (isHf32) {
            AscendC::SetHF32Mode(1);
            AscendC::SetHF32TransMode(1);
        }
#endif
        SetMMLayoutTransform(true);
        blockMmadOp.template Init<BlockScheduler::FULL_LOAD_MODE>(
            problemShape_, tileL1, tileL0, isBias_, bs.GetL1BuferNum_(), bs.GetL0cDB(), bs.GetNonContinuousParams(),
            bs.isSplitSingleK_, static_cast<uint8_t>(bs.GetShiftValue()));
        if constexpr (BlockMmadOp::DispatchPolicy::enableQuant) {
            blockMmadOp.CacheQuantScalar(LoadQuantScalarFromGm(params.mmadParams.quantScaleGmAddr));
        }
        // Process tiles in ping-pong mode
        if constexpr (BlockScheduler::FULL_LOAD_MODE == B_FULL_LOAD_MODE) {
            blockMmadOp.template CopyInB1<BlockMmadBuilder::formatB>(bGlobal_, Get<MNK_N>(problemShape_),
                                                                     Get<MNK_K>(problemShape_));
            blockMmadOp.CopyInC1(biasGlobal_, Get<MNK_N>(problemShape_));
        } else if constexpr (BlockScheduler::FULL_LOAD_MODE == A_FULL_LOAD_MODE) {
            blockMmadOp.CopyInA1(aGlobal_, Get<MNK_M>(problemShape_), Get<MNK_K>(problemShape_));
        }
        uint64_t curML1 = Get<MNK_M>(tileL1);
        uint64_t curNL1 = Get<MNK_N>(tileL1);
        int64_t n = Get<MNK_N>(problemShape_);
        for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
            // mIter
            for (uint64_t mOffset = 0; mOffset < curML1; mOffset += Get<0>(tileL0)) {
                // nIter
                for (uint64_t nOffset = 0; nOffset < curNL1; nOffset += Get<1>(tileL0)) {
                    // kIter
                    for (uint64_t kOffset = 0; kOffset < bs.k_; kOffset += bs.blkK_) {
                        TupleL1L0Shape blockShape = bs.template GetBlockShape<BlockMmadBuilder::formatB, transB, BType>(
                            tileIdx, mOffset, nOffset, kOffset);
                        auto blockCoord = bs.GetBlockCoord(tileIdx);
                        if constexpr (BlockMmadBuilder::formatB == CubeFormat::NZ) {
                            blockCoord = bs.GetSingleBlockCoord(tileIdx);
                        }
                        if (bs.isSplitSingleK_) {
                            blockCoord = bs.GetSplitKBlockCoord(tileIdx);
                        }
                        // cal offset between blocks
                        auto blockOffset = GetOffsetWithoutLayout<BlockCoord, TupleShape, BlockMmadBuilder::formatB,
                                                                  BType>(
                            blockCoord, problemShape_, transA, transB, isBias_, bs.GetNonContinuousParams(), blockShape,
                            tileL1, bs.GetSplitOffset(), bs.GetTailParams(), bs.isSplitSingleK_);
                        if (Get<0>(blockShape) <= 0 || Get<1>(blockShape) <= 0) {
#if __NPU_ARCH__ != 5102
                            UnsetHf32(isHf32);
#endif
                            return;
                        }
                        int64_t offsetA = Get<0>(blockOffset);
                        int64_t offsetB = Get<1>(blockOffset);
                        int64_t offsetC = Get<2>(blockOffset);
                        int64_t offsetBias = Get<3>(blockOffset);
                        offsetC += mOffset * n + nOffset;
                        blockMmadOp.template operator()<AscendC::GlobalTensor<CType>, BlockMmadBuilder::formatB>(
                            cGlobal_[offsetC], aGlobal_[offsetA], bGlobal_[offsetB], biasGlobal_[offsetBias],
                            blockShape, mOffset, nOffset, tileIdx == curBlockIdx, false, bs.blkK_,
                            bs.splitSingleKIdx_ == 0, bs.splitSingleKIdx_ == (bs.splitSingleKRound_ - 1));
                    }
                }
            }
        }

        SetMMLayoutTransform(false);
#if __NPU_ARCH__ != 5102
        UnsetHf32(isHf32);
#endif
    }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct
