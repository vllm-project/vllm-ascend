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
 * \file kernel_matmul_streamk.h
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
#include "../block/block_mmad_streamk.h"
#include "../block/block_mmad_builder.h"
#include "../epilogue/block_epilogue_streamk.h"
#include "../epilogue/block_epilogue_streamk_fusion.h"
#include "../epilogue/block_epilogue_streamk_gelu.h"
#include "../epilogue/block_epilogue_elementwise.h"
#include "../epilogue/block_epilogue_empty.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_policy.h"

namespace Cmct {
namespace Gemm {
namespace Kernel {
// specialization of streamk basicapi kernel
template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_,
          uint64_t FUSED_OP_TYPE_ = 0, class OutType_ = void, class FusedEpilogue_ = Block::BlockEpilogueEmpty,
          typename Enable_ = void>
class KernelMatmulStreamK {
    static_assert(AscendC::Std::always_false_v<BlockEpilogue_>,
                  "KernelStreamk is not implemented for this BlockEpilogue");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_,
          uint64_t FUSED_OP_TYPE_, class OutType_, class FusedEpilogue_>
class KernelMatmulStreamK<
    ProblemShape_, BlockMmadBuilder_, BlockEpilogue_, BlockScheduler_, FUSED_OP_TYPE_, OutType_, FusedEpilogue_,
    AscendC::Std::enable_if_t<
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamK<float, float, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamK<float, float, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamK<float, bfloat16_t, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKGelu<float, bfloat16_t, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_, Block::BlockEpilogueStreamK<float, bfloat16_t,
                                                        MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamK<float, half, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamK<float, half, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_, Block::BlockEpilogueStreamK<
                                float, float, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_RELU>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamK<float, float,
                                        MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_RELU>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamK<float, bfloat16_t,
                                        MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_RELU>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamK<float, bfloat16_t,
                                        MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_RELU>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_, Block::BlockEpilogueStreamK<
                                float, half, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_RELU>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamK<float, half,
                                        MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_RELU>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_, Block::BlockEpilogueStreamKFusion<
                                float, float, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_ADD>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, float, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_ADD>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<float, bfloat16_t,
                                              MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_ADD>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, bfloat16_t, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_ADD>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_, Block::BlockEpilogueStreamKFusion<
                                float, half, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_ADD>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, half, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_ADD>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_, Block::BlockEpilogueStreamKFusion<
                                float, float, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_MUL>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, float, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_MUL>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<float, bfloat16_t,
                                              MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_MUL>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, bfloat16_t, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_MUL>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_, Block::BlockEpilogueStreamKFusion<
                                float, half, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_MUL>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, half, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_MUL>>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, float, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_ADD>, 1>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, float, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_ADD>, 1>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, bfloat16_t, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_ADD>, 1>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, bfloat16_t, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_ADD>, 1>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_, Block::BlockEpilogueStreamKFusion<
                                float, half, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_ADD>, 1>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, half, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_ADD>, 1>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, float, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_MUL>, 1>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, float, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_MUL>, 1>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, bfloat16_t, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_MUL>, 1>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, bfloat16_t, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_MUL>, 1>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_, Block::BlockEpilogueStreamKFusion<
                                float, half, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY, OP_TYPE_MUL>, 1>> ||
        AscendC::Std::is_base_of_v<
            BlockEpilogue_,
            Block::BlockEpilogueStreamKFusion<
                float, half, MatmulMultiBlockWithStreamK<MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_MUL>, 1>>>> {
public:
    __aicore__ inline KernelMatmulStreamK() {}
    __aicore__ inline ~KernelMatmulStreamK() {}

    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;
    using FusedEpilogue = FusedEpilogue_;
    using OutType = OutType_;
    static constexpr uint64_t FUSED_OP_TYPE = FUSED_OP_TYPE_;

    static constexpr bool transA = BlockMmadBuilder::transA;
    static constexpr bool transB = BlockMmadBuilder::transB;
    // schedulerOp
    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<
        ProblemShape, typename BlockMmadBuilder::L1TileShape, typename BlockMmadBuilder::L0TileShape, BlockScheduler,
        transA, transB>::SchedulerOp;
    // mmadOp
    using BlockMmadOp = typename BlockMmadBuilder::BlockMmadOp;
    using BlockMmadArguments = typename BlockMmadBuilder::Arguments;
    using BlockMmadParams = typename BlockMmadBuilder::Params;
    using BlockSchedulerParams = typename BlockSchedulerOp::Params;
    using AType = typename BlockMmadBuilder::AType;
    using BType = typename BlockMmadBuilder::BType;
    using CType = typename BlockMmadBuilder::CType;
    using BiasType = typename BlockMmadBuilder::BiasType;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;

    // ND layout
    using NDLayout = AscendC::Layout<AscendC::Shape<int64_t, int64_t>, AscendC::Stride<int64_t, int64_t>>;

    static constexpr bool isNdFormat = BlockMmadBuilder::formatB == CubeFormat::ND;
    static constexpr bool isFp32 = std::is_same_v<CType, float>;

    // no need to have tensortrait
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    AscendC::GlobalTensor<float> workspaceGlobal_;

    // basic args
    int64_t m_ = 0;
    int64_t n_ = 0;
    int64_t k_ = 0;
    int64_t usedCoreNum_ = 0;

    // shape
    TupleShape problemShape_{};
    bool isBias_ = false;

    constexpr static uint16_t NUM_TWO = 2;
    constexpr static uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    constexpr static uint16_t AIV_SYNC_AIC_FLAG = 6;
    constexpr static uint16_t AIC_SYNC_AIV_FLAG = 8;
    constexpr static uint16_t FLAG_ID_MAX = 16;
    constexpr static uint16_t BLOCK_BASE_M = 256;
    constexpr static uint16_t BLOCK_BASE_N = 256;

    struct EpilogueArgs {
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        GM_ADDR x3GmAddr{nullptr};
        bool x3BatchBroadcast{false};
        uint64_t x3M{0};
    };

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmadArgs;
        EpilogueArgs epilogueArgs;
        Arguments() = default;
    };

    using BlockEpilogueParams = typename BlockEpilogue::Params;

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        EpilogueArgs epilogueArgs;
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
        m_ = Get<MNK_M>(problemShape_);
        n_ = Get<MNK_N>(problemShape_);
        k_ = Get<MNK_K>(problemShape_);
        usedCoreNum_ = params.schParams.tilingData->usedCoreNum;
        // Init GlobalTensor
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(blockMmadParams_.aGmAddr));
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(blockMmadParams_.bGmAddr));
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(blockMmadParams_.cGmAddr));
        workspaceGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(blockMmadParams_.workspaceGmAddr));
        if (blockMmadParams_.biasGmAddr != nullptr) {
            isBias_ = true;
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(blockMmadParams_.biasGmAddr));
        }
    }

    __aicore__ inline void run(Params const& params)
    {
        // Init
        Init(params);
        BlockSchedulerOp bs(params.problemShape, params.schParams, isFp32, isNdFormat);
        TupleShape tileL1 = bs.GetTileL1Shape();
        int64_t mL1 = Get<MNK_M>(tileL1);
        int64_t nL1 = Get<MNK_N>(tileL1);
        int64_t kL1 = Get<MNK_K>(tileL1);
        int64_t mTileNum = Get<MNK_M>(bs.GetMNKTileNum());
        int64_t nTileNum = Get<MNK_N>(bs.GetMNKTileNum());
        int64_t skKTileNum = Get<MNK_K>(bs.GetMNKTileNum()); // it only used in sk
        int64_t batchNum = Get<MNK_B>(bs.GetMNKTileNum());
        int64_t tileNum = bs.GetTotalTileNum();

        if ASCEND_IS_AIC {
            // Instantiate mmadOp
            BlockMmadOp blockMmadOp;
            int64_t curBlockIdx = AscendC::GetBlockIdx();

            TupleShape tileL0 = bs.GetTileL0Shape();
            int64_t isHf32 = bs.GetHf32Flag();

            if (curBlockIdx >= bs.GetBlockNum(usedCoreNum_)) {
                CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
                CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
                return;
            }
            if (isHf32) {
                AscendC::SetHF32Mode(1);
                AscendC::SetHF32TransMode(1);
            }
            SetMMLayoutTransform(true); // copy out with nfirst, try to make cube and fixp pairing.
            blockMmadOp.Init(problemShape_, tileL1, tileL0, isBias_, bs.isSplitSingleK_);
            int64_t tailSKTotalTileNum = static_cast<int64_t>(((mTileNum * nTileNum * batchNum) % usedCoreNum_) *
                                                              skKTileNum);
            for (int64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += usedCoreNum_) {
                int64_t tmpTileIdx = tileIdx;
                if (!bs.CheckIsSkScene(0)) { // SK Preload in DP+SK
                    if (tileIdx % usedCoreNum_ < tailSKTotalTileNum &&
                        (CeilDiv(tileIdx + 1, usedCoreNum_) == (CeilDiv(tileNum, usedCoreNum_) - 1))) {
                        tmpTileIdx = tileIdx + usedCoreNum_;
                    } else if (tileIdx % usedCoreNum_ < tailSKTotalTileNum &&
                               (CeilDiv(tileIdx + 1, usedCoreNum_) == CeilDiv(tileNum, usedCoreNum_))) {
                        tmpTileIdx = tileIdx - usedCoreNum_;
                    }
                }
                auto singleCoreShape = bs.GetSingleCoreShape(tmpTileIdx);
                auto singleCoreCoord = bs.GetSingleCoreCoord(tmpTileIdx);
                for (int64_t kOffset = 0; kOffset < bs.realSingleCoreK_; kOffset += bs.blkK_) {
                    bs.UpdateSplitKIdx(kOffset);
                    auto blockOffset = GetOffsetStreamK<BlockCoord, TupleShape, BlockMmadBuilder::formatB, BType>(
                        singleCoreCoord, problemShape_, tileL1, bs.GetCurKSingleCore(tmpTileIdx), transA, transB,
                        isBias_, bs.splitSingleK_, bs.splitSingleKIdx_);
                    int64_t offsetA = Get<MNK_M>(blockOffset);
                    int64_t offsetB = Get<MNK_N>(blockOffset);
                    int64_t offsetC = Get<2>(blockOffset);
                    int64_t offsetBias = Get<3>(blockOffset);
                    int64_t offsetWorkspace = (((tmpTileIdx % usedCoreNum_) / skKTileNum) * skKTileNum +
                                               Get<MNK_K>(singleCoreCoord)) *
                                              BLOCK_BASE_M * BLOCK_BASE_N;
                    blockMmadOp.template operator()<BlockMmadBuilder::formatB>(
                        cGlobal_[offsetC], aGlobal_[offsetA], bGlobal_[offsetB], biasGlobal_[offsetBias],
                        workspaceGlobal_[offsetWorkspace], singleCoreShape, Get<MNK_K>(singleCoreCoord),
                        bs.CheckIsSkScene(tmpTileIdx), bs.blkK_, bs.splitSingleKIdx_ == 0,
                        bs.splitSingleKIdx_ == (bs.splitSingleKRound_ - 1));
                }
                if (tmpTileIdx + usedCoreNum_ >= tileNum) {
                    CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
                    CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
                }
            }
            SetMMLayoutTransform(false);
            if (isHf32) {
                AscendC::SetHF32Mode(0);
                AscendC::SetHF32TransMode(0);
            }
        }

        if ASCEND_IS_AIV {
            uint64_t lastLoopTotalCnt = (mTileNum * nTileNum * batchNum % usedCoreNum_) * skKTileNum;
            uint64_t curBlockIdxInAiv = AscendC::GetBlockIdx();
            if (curBlockIdxInAiv >= lastLoopTotalCnt * AscendC::GetTaskRation()) {
                CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
                SyncAll();
                return;
            }

            CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(AIC_SYNC_AIV_FLAG);
            SyncAll();
            BlockEpilogue epilogueOp;
            BlockEpilogueParams skEpilogueParams{};
            skEpilogueParams.cGmAddr = params.epilogueArgs.cGmAddr;
            skEpilogueParams.workspaceGmAddr = params.epilogueArgs.workspaceGmAddr;
            skEpilogueParams.x3GmAddr = params.epilogueArgs.x3GmAddr;
            if constexpr (FUSED_OP_TYPE == OP_TYPE_ADD || FUSED_OP_TYPE == OP_TYPE_MUL) {
                skEpilogueParams.x3BatchBroadcast = params.epilogueArgs.x3BatchBroadcast;
                skEpilogueParams.x3M = params.epilogueArgs.x3M;
            }
            epilogueOp.Init(skEpilogueParams, problemShape_, tileL1, bs.GetMNKTileNum(), usedCoreNum_,
                            bs.CheckIsSkScene(0));
            epilogueOp();
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

    __host_aicore__ static Status CheckArgs(Arguments const& args)
    {
        // Check shape in kernel
        CHECK_AND_RETURN(CheckShape(args.problemShape));
        // Check mmad args
        CHECK_AND_RETURN(BlockMmadBuilder::CheckArgs(args.mmadArgs));

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
        Params params = {args.problemShape, mmadParams, {}};
        return params;
    }

    __aicore__ inline void operator()(Params const& params) { run(params); }
};

} // namespace Kernel
} // namespace Gemm
} // namespace Cmct
