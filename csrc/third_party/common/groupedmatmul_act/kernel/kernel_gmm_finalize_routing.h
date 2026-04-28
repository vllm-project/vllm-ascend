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
 * \file kernel_gmm_finalize_routing.h
 * \brief
 */

#ifndef KERNEL_GMM_FINALIZE_ROUTING_H
#define KERNEL_GMM_FINALIZE_ROUTING_H
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/status_utils.h"

#include "./semaphore.h"
#include "../block/block_mx_mm_aic_to_aiv_builder.h"
#include "../epilogue/block_epilogue_finalize_routing.h"
#include "../prologue/block_prologue_finalize_routing.h"

#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_gmm_aswt_with_tail_split.h"

namespace Act {
namespace Gemm {
namespace Kernel {

namespace {
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_X1SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_X2SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_BIAS_OFFSET = 4UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_LOGIT_OFFSET = 6UL;
constexpr uint64_t IDX_M_TILEIDX = 0UL;
constexpr uint64_t IDX_N_TILEIDX = 1UL;
constexpr uint64_t IDX_M_TAIL_SPLIT_TILEIDX = 2UL;
constexpr uint64_t IDX_N_TAIL_SPLIT_TILEIDX = 3UL;
constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t FLAG_ID_MAX = 16;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
} // namespace

using namespace AscendC;
template <class ProblemShape_, class BlockMmadBuilder_, class BlockPrologue_, class BlockEpilogue_, class BlockScheduler_,
          typename Enable_ = void>
class KernelGmmFinalizeRouting {
    static_assert(AscendC::Std::always_false_v<BlockScheduler_>,
                  "KernelGmmFinalizeRouting is not implemented for this scheduler");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockPrologue_, class BlockEpilogue_, class BlockScheduler_>
class KernelGmmFinalizeRouting<
    ProblemShape_, BlockMmadBuilder_, BlockPrologue_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<BlockScheduler_, GroupedMatmulAswtWithTailSplitScheduler>>> {
public:
    __aicore__ inline KernelGmmFinalizeRouting() {}

    __aicore__ inline ~KernelGmmFinalizeRouting() {}

    using BlockPrologue = BlockPrologue_;
    using BlockEpilogue = BlockEpilogue_;
    using BlockMmadBuilder = BlockMmadBuilder_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    static constexpr bool transA = BlockMmadBuilder::transA;
    static constexpr bool transB = BlockMmadBuilder::transB;
    static constexpr int64_t l1M = BlockMmadBuilder::l1M;
    static constexpr int64_t l1N = BlockMmadBuilder::l1N;
    static constexpr int64_t l1K = BlockMmadBuilder::l1K;
    // schedulerOp
    using BlockSchedulerOp =
        typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmadBuilder::L1TileShape,
                                               typename BlockMmadBuilder::L0TileShape, BlockScheduler, transA,
                                               transB>::SchedulerOp;
    // mmadOp
    using BlockMmadOp = typename BlockMmadBuilder::BlockMmadOp;
    using BlockMmadArguments = typename BlockMmadBuilder::Arguments;
    using BlockPrologueArguments = typename BlockPrologue::Arguments;
    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;
    using BlockMmadParams = typename BlockMmadBuilder::Params;
    using BlockPrologueParams = typename BlockPrologue::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using AType = typename BlockMmadBuilder::AType;
    using BType = typename BlockMmadBuilder::BType;
    using CType = typename BlockMmadBuilder::CType;
    using BiasType = typename BlockMmadBuilder::BiasType;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // a, b, x1scale, x2scale, bias, y, logit
    using BaseOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // a, b, x1scale, x2scale, bias, y
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // coordinate
    using CoordClass =
        Coordinate<transA, transB, BlockMmadBuilder::formatA, BlockMmadBuilder::formatB, BlockMmadBuilder::formatC>;

    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> x1ScaleGlobal_;
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> x2ScaleGlobal_;
    AscendC::GlobalTensor<bfloat16_t> biasGlobal_;
    AscendC::GlobalTensor<int64_t> groupListGm_;
    // shape
    TupleShape problemShape_{};
    BaseOffset baseOffset_{0, 0, 0, 0, 0, 0, 0};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};
    uint64_t preOffset_ = 0;
    BlockMmadOp mmadOp_;

    BlockPrologue prologueOp_;
    BlockEpilogue epilogueOp_;
    AscendC::LocalTensor<CType> l0cOutUb_;
    bool isVecSetSyncCom_ = false;

    struct GMMTiling {
        uint32_t groupNum;
        uint8_t groupListType;
        int32_t baseM;
        int32_t baseN;
        int32_t baseK;
        uint8_t hasBias;
        const TCubeTiling* __restrict matmulTiling;
        __aicore__ GMMTiling() {}
        __aicore__ GMMTiling(uint32_t groupNum_, uint8_t groupListType_, int32_t baseM_, int32_t baseN_,
                             int32_t baseK_, uint8_t hasBias_) :
            groupNum(groupNum_), groupListType(groupListType_), baseM(baseM_), baseN(baseN_), baseK(baseK_), hasBias(hasBias_)
        {}
    };

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmadArgs;
        BlockPrologueArguments prologueArgs;
        BlockEpilogueArguments epilogueArgs;
        GMMTiling gmmArgs;
        Arguments() = default;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockPrologueParams prologueParams;
        BlockEpilogueParams epilogueParams;
        GMMTiling gmmParams;
        Params() = default;
    };

    __aicore__ inline void NotifyCube()
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG);
    }
    __aicore__ inline void WaitForVector()
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG);
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + FLAG_ID_MAX);
    }
    __aicore__ inline void NotifyVector()
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG);
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
    }
    __aicore__ inline void WaitForCube()
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG);
    }

    __aicore__ inline void End()
    {
        if ASCEND_IS_AIC {
            if (isVecSetSyncCom_) {
                WaitForVector();
            }
        }
    }

    __aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, uint8_t groupListType)
    {
        int32_t splitValue = 0;
        if (groupListType == 0) {
            int32_t offset = static_cast<int32_t>(groupListGm_.GetValue(groupIdx));
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else {
            splitValue = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
        }
        return splitValue;
    }

    __aicore__ inline void UpdateGlobalBuffer(const Params& params)
    {
        if ASCEND_IS_AIC {
            aGlobal_.SetGlobalBuffer((__gm__ AType*)params.mmadParams.aGmAddr + Get<IDX_A_OFFSET>(baseOffset_));
            bGlobal_.SetGlobalBuffer((__gm__ BType*)params.mmadParams.bGmAddr + Get<IDX_B_OFFSET>(baseOffset_));
            x1ScaleGlobal_.SetGlobalBuffer((__gm__ AscendC::fp8_e8m0_t*)params.mmadParams.x1ScaleGmAddr + Get<IDX_X1SCALE_OFFSET>(baseOffset_));
            x2ScaleGlobal_.SetGlobalBuffer((__gm__ AscendC::fp8_e8m0_t*)params.mmadParams.x2ScaleGmAddr + Get<IDX_X2SCALE_OFFSET>(baseOffset_));
            if (params.gmmParams.hasBias == 1)
            {
                biasGlobal_.SetGlobalBuffer((__gm__ bfloat16_t*)params.mmadParams.biasGmAddr + Get<IDX_BIAS_OFFSET>(baseOffset_));
            }
        }
        if ASCEND_IS_AIV {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                0L, 0L, 0L, 0L, Get<IDX_LOGIT_OFFSET>(baseOffset_), Get<IDX_LOGIT_OFFSET>(baseOffset_)};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    }

    __aicore__ inline void UpdateOffset(uint32_t groupIdx)
    {
        // baseOffset is 0 when groupIdx = 0
        if (groupIdx == 0) {
            return;
        }
        uint64_t m = Get<MNK_M>(problemShape_);
        uint64_t n = Get<MNK_N>(problemShape_);
        uint64_t k = Get<MNK_K>(problemShape_);
        if (AscendC::IsSameTypeV<AType, fp4x2_e2m1_t> || AscendC::IsSameTypeV<AType, fp4x2_e1m2_t>) {
            Get<IDX_A_OFFSET>(baseOffset_) += (m * k) >> 1;
            Get<IDX_B_OFFSET>(baseOffset_) += (n * k) >> 1;
        } else {
            Get<IDX_A_OFFSET>(baseOffset_) += m * k;
            Get<IDX_B_OFFSET>(baseOffset_) += n * k;
        }
        Get<IDX_BIAS_OFFSET>(baseOffset_) += n;
        auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        // scaleAAxisBaseOffset (m, ceil(k,64), 2)
        Get<IDX_X1SCALE_OFFSET>(baseOffset_) += m * scaleK;
        // scaleBAxisBaseOffset (g, n, ceil(k,64), 2) or (g, ceil(k,64), n, 2)
        Get<IDX_X2SCALE_OFFSET>(baseOffset_) += n * scaleK;
        Get<IDX_C_OFFSET>(baseOffset_) += m * n;
        Get<IDX_LOGIT_OFFSET>(baseOffset_) += m;
    }

    __aicore__ inline bool UpdateGroupParams(const Params& params, uint32_t groupIdx)
    {
        UpdateOffset(groupIdx);
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, params.gmmParams.groupListType);
        Get<MNK_M>(problemShape_) = splitValue;
        // split_m, when m=0, skip
        if (Get<MNK_M>(problemShape_) == 0) {
            return false;
        }
        return true;
    }

    __aicore__ inline void InitParamsAndTensor(const Params& params)
    {
        Get<MNK_N>(problemShape_) = params.gmmParams.matmulTiling->N;
        Get<MNK_K>(problemShape_) = params.gmmParams.matmulTiling->Ka;
        groupListGm_.SetGlobalBuffer((__gm__ int64_t*)params.mmadParams.groupListGmAddr);
    }

    __aicore__ inline void ProcessSingleGroup(const Params& params, BlockSchedulerOp& bs, uint32_t groupIdx)
    {
        int64_t m = Get<MNK_M>(problemShape_);
        int64_t n = Get<MNK_N>(problemShape_);
        int64_t k = Get<MNK_K>(problemShape_);
        TupleShape resProblemShape {Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_),
                                    Get<MNK_K>(problemShape_), 0};
        bs.UpdateNextProblem(resProblemShape);
        epilogueOp_.UpdateNextProblem(resProblemShape);
        UpdateGlobalBuffer(params);
        CoordClass coord(m, n, k, params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        BlockCoord tileIdx;
        while (bs.GetTileIdx(tileIdx)) {
            BlockShape singleShape = bs.GetBlockShape(tileIdx);
            blockOffset_ = coord.template GetQuantOffset<GroupedMatmul::QuantMode::MX_PERGROUP_MODE>(
                Get<IDX_M_TILEIDX>(tileIdx), Get<IDX_N_TILEIDX>(tileIdx), Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
                Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape));
            if ASCEND_IS_AIC {
                if (isVecSetSyncCom_)
                {
                    WaitForVector();
                }
                AscendC::Std::tuple<int32_t, int32_t, int32_t> mmSingleShape{Get<MNK_M>(singleShape),
                                                                             Get<MNK_N>(singleShape), k};
                if (params.gmmParams.hasBias == 1)
                {
                    mmadOp_(aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)], bGlobal_[Get<IDX_B_OFFSET>(blockOffset_)],
                            x1ScaleGlobal_[Get<IDX_X1SCALE_OFFSET>(blockOffset_)], x2ScaleGlobal_[Get<IDX_X2SCALE_OFFSET>(blockOffset_)],
                            biasGlobal_[Get<IDX_BIAS_OFFSET>(blockOffset_)], l0cOutUb_, mmSingleShape, transA, transB);
                } else {
                    mmadOp_(aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)], bGlobal_[Get<IDX_B_OFFSET>(blockOffset_)],
                            x1ScaleGlobal_[Get<IDX_X1SCALE_OFFSET>(blockOffset_)], x2ScaleGlobal_[Get<IDX_X2SCALE_OFFSET>(blockOffset_)],
                            l0cOutUb_, mmSingleShape, transA, transB);
                }
                NotifyVector();
            }
            isVecSetSyncCom_ = true;
            if ASCEND_IS_AIV {
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<MNK_M>(singleShape),
                                                                                      Get<MNK_N>(singleShape), 0, 0};
                int64_t y = Get<IDX_C_OFFSET>(blockOffset_);
                int64_t mOffset = y / n;
                int64_t nOffset = y - mOffset * n;
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                    nOffset, 0, 0, 0, mOffset, mOffset};
                WaitForCube();
                epilogueOp_(epilogueShape, epilogueOffset);
                NotifyCube();
            }
        }
    }

    __host_aicore__ static Status CheckShape(const ProblemShape& shape)
    {
        int64_t m = shape.m;
        int64_t n = shape.n;
        int64_t k = shape.k;
        // Check m, n, k overlimit data type
        if (m > INT32_MAX || n > INT32_MAX || k > INT32_MAX) {
            return Status::mnkErrorExceedsLimit;
        }
        return Status::success;
    }

    __host_aicore__ static Status CanImplement(const Arguments& args)
    {
        // Check shape in kernel
        CHECK_AND_RETURN(CheckShape(args.problemShape));
        // Check mmad args
        CHECK_AND_RETURN(BlockMmadBuilder::CanImplement(args.mmadArgs));
        // Check args for block scheduler
        CHECK_AND_RETURN(BlockSchedulerOp::CanImplement(args.problemShape));
        return Status::success;
    }

    __host_aicore__ static size_t GetWorkspaceSize(ProblemShape shape, int64_t blockNum)
    {
        return 0;
    }

    __host_aicore__ static Params InitParams(const Arguments& args, GM_ADDR workspace)
    {
        BlockMmadParams mmadParams = BlockMmadBuilder::InitParams(args.mmadArgs);
        // mmad params with epiligue takes workspaceGm as output
        if constexpr (!AscendC::Std::is_same_v<BlockEpilogue, Block::BlockEpilogueEmpty>) {
            mmadParams.cGmAddr = workspace;
        }
        // epilogue params takes workspaceGm as input
        BlockPrologueParams prologueParams = BlockPrologue::InitParams(args.prologueArgs, workspace);
        // epilogue params takes workspaceGm as input
        BlockEpilogueParams epilogueParams = BlockEpilogue::InitParams(args.epilogueArgs, workspace);
        Params params = {args.problemShape, mmadParams, prologueParams, epilogueParams, args.gmmArgs};
        return params;
    }

    __host_aicore__ static int64_t GetBlockNum(ProblemShape shape)
    {
        return BlockSchedulerOp::GetBlockNum(shape);
    }

    __aicore__ inline void operator()(const Params& params)
    {
        if ASCEND_IS_AIV
        {
            prologueOp_.Init(params.prologueParams);
            prologueOp_();
        }
        if ASCEND_IS_AIC 
        {
            mmadOp_.Init(const_cast<TCubeTiling* __restrict>(params.gmmParams.matmulTiling), GetTPipePtr());
            l0cOutUb_ = epilogueOp_.GetL0c2UbTensor();
        }
        InitParamsAndTensor(params);
        BlockSchedulerOp bs(params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        SyncAll();
        if ASCEND_IS_AIV
        {
            AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(AIV_SYNC_AIC_FLAG);
        }
        if ASCEND_IS_AIC
        {
            WaitForVector();
        }
        if ASCEND_IS_AIV
        {
            epilogueOp_.Init(params.epilogueParams);
        }
        uint32_t groupNum = params.gmmParams.groupNum;
        for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
            if (!UpdateGroupParams(params, groupIdx)) {
                continue;
            }
            ProcessSingleGroup(params, bs, groupIdx);
        }
        End();
    }
};
} // namespace Kernel
} // namespace Gemm
} // namespace Act
#endif