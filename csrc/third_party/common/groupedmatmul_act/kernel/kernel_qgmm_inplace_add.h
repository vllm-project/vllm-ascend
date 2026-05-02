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
 * \file kernel_qgmm_inplace_add.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_QGMM_INPLACE_ADD_H
#define MATMUL_KERNEL_KERNEL_QGMM_INPLACE_ADD_H
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

#include "../utils/common_utils.h"
#include "../utils/grouped_matmul_constant.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/status_utils.h"

#include "./semaphore.h"
#include "../block/block_quant_matmul_builder.h"
#include "../epilogue/block_epilogue_empty.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_gmm_aswt_with_tail_split.h"

namespace Act {
namespace Gemm {
namespace Kernel {

namespace {
constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_X1SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_X2SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_BIAS_OFFSET = 4UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_M_TILEIDX = 0UL;
constexpr uint64_t IDX_N_TILEIDX = 1UL;
constexpr uint64_t IDX_M_TAIL_SPLIT_TILEIDX = 2UL;
constexpr uint64_t IDX_N_TAIL_SPLIT_TILEIDX = 3UL;
constexpr int64_t MATRIX_INNER_DIM_LIMIT_SIZE_V35 = 2097151L; // 21bits
} // namespace

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_,
          typename Enable_ = void>
class KernelQGmmInplaceAdd {
    static_assert(AscendC::Std::always_false_v<BlockScheduler_>,
                  "KernelQGmmInplaceAdd is not implemented for this scheduler");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelQGmmInplaceAdd<
    ProblemShape_, BlockMmadBuilder_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<BlockScheduler_, GroupedMatmulAswtWithTailSplitScheduler>>> {
public:
    __aicore__ inline KernelQGmmInplaceAdd() {}

    __aicore__ inline ~KernelQGmmInplaceAdd() {}

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
    using BlockEpilogueArguments = typename BlockEpilogue::Arguments;
    using BlockMmadParams = typename BlockMmadBuilder::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using AType = typename BlockMmadBuilder::AType;
    using BType = typename BlockMmadBuilder::BType;
    using CType = typename BlockMmadBuilder::CType;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // order by {x1, x2, scale1, scale2, bias, y}
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // coordinate
    using CoordClass =
        Coordinate<transA, transB, BlockMmadBuilder::formatA, BlockMmadBuilder::formatB, BlockMmadBuilder::formatC>;

    // attribute
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> x1ScaleGlobal_;
    AscendC::GlobalTensor<AscendC::fp8_e8m0_t> x2ScaleGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<int64_t> groupListGm_;
    // shape
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};
    uint64_t preOffset_ = 0;

    struct GMMTiling {
        uint32_t groupNum;
        uint8_t groupListType;
        int32_t baseM;
        int32_t baseN;
        int32_t baseK;
        const TCubeTiling* __restrict matmulTiling;
        __aicore__ GMMTiling() {}
        __aicore__ GMMTiling(uint32_t groupNum_, uint8_t groupListType_, int32_t baseM_, int32_t baseN_,
                             int32_t baseK_) :
            groupNum(groupNum_), groupListType(groupListType_), baseM(baseM_), baseN(baseN_), baseK(baseK_)
        {}
    };

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmadArgs;
        BlockEpilogueArguments epilogueArgs;
        GMMTiling gmmArgs;
        Arguments() = default;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        GMMTiling gmmParams;
        Params() = default;
    };

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
        aGlobal_.SetGlobalBuffer((__gm__ AType*)params.mmadParams.aGmAddr + Get<IDX_A_OFFSET>(baseOffset_));
        bGlobal_.SetGlobalBuffer((__gm__ BType*)params.mmadParams.bGmAddr + Get<IDX_B_OFFSET>(baseOffset_));
        x1ScaleGlobal_.SetGlobalBuffer((__gm__ AscendC::fp8_e8m0_t*)params.mmadParams.x1ScaleGmAddr +
                                       Get<IDX_X1SCALE_OFFSET>(baseOffset_));
        x2ScaleGlobal_.SetGlobalBuffer((__gm__ AscendC::fp8_e8m0_t*)params.mmadParams.x2ScaleGmAddr +
                                       Get<IDX_X2SCALE_OFFSET>(baseOffset_));
        cGlobal_.SetGlobalBuffer((__gm__ CType*)params.mmadParams.cGmAddr + Get<IDX_C_OFFSET>(baseOffset_));
    }

    __aicore__ inline void UpdateOffset(uint32_t groupIdx)
    {
        // baseOffset is 0 when groupIdx = 0
        if (groupIdx == 0) {
            return;
        }
        uint64_t m = Get<M_VALUE>(problemShape_);
        uint64_t n = Get<N_VALUE>(problemShape_);
        uint64_t k = Get<K_VALUE>(problemShape_);
        // aBaseOffset += m * k
        Get<IDX_A_OFFSET>(baseOffset_) = Get<IDX_A_OFFSET>(baseOffset_) + m * k;
        // bBaseOffset += n * k
        Get<IDX_B_OFFSET>(baseOffset_) = Get<IDX_B_OFFSET>(baseOffset_) + n * k;
        uint64_t scaleK = (Get<IDX_B_OFFSET>(baseOffset_) / n / MXFP_DIVISOR_SIZE + groupIdx) * MXFP_MULTI_BASE_SIZE;
        // scaleAAxisBaseOffset = (k / 64 + g) * m * 2
        Get<IDX_X1SCALE_OFFSET>(baseOffset_) = scaleK * m;
        // scaleBAxisBaseOffset = (k / 64 + g) * n * 2
        Get<IDX_X2SCALE_OFFSET>(baseOffset_) = scaleK * n;
        // yBaseOffset += m * n
        Get<IDX_C_OFFSET>(baseOffset_) = Get<IDX_C_OFFSET>(baseOffset_) + m * n;
    }

    __aicore__ inline bool UpdateGroupParams(const Params& params, uint32_t groupIdx)
    {
        UpdateOffset(groupIdx);
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, params.gmmParams.groupListType);
        Get<K_VALUE>(problemShape_) = splitValue;
        // split_k, when k=0，0 + y = y, skip
        if (Get<K_VALUE>(problemShape_) == 0) {
            return false;
        }
        UpdateGlobalBuffer(params);
        return true;
    }

    __aicore__ inline void InitParamsAndTensor(const Params& params)
    {
        Get<M_VALUE>(problemShape_) = params.gmmParams.matmulTiling->M;
        Get<N_VALUE>(problemShape_) = params.gmmParams.matmulTiling->N;
        cGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.mmadParams.groupListGmAddr));
    }

    __aicore__ inline void ProcessSingleGroup(const Params& params, BlockSchedulerOp& bs, BlockMmadOp& blockMmadOp)
    {
        int64_t m = Get<M_VALUE>(problemShape_);
        int64_t n = Get<N_VALUE>(problemShape_);
        int64_t k = Get<K_VALUE>(problemShape_);
        CoordClass coord(m, n, k, params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        BlockCoord tileIdx;
        while (bs.GetTileIdx(tileIdx)) {
            BlockShape singleShape = bs.GetBlockShape(tileIdx);
            blockOffset_ = coord.template GetQuantOffset<GroupedMatmul::QuantMode::MX_PERGROUP_MODE>(
                Get<IDX_M_TILEIDX>(tileIdx), Get<IDX_N_TILEIDX>(tileIdx), Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
                Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape));
            AscendC::Std::tuple<int32_t, int32_t, int32_t> mmSingleShape{Get<M_VALUE>(singleShape),
                                                                         Get<N_VALUE>(singleShape), k};
            blockMmadOp(aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)], bGlobal_[Get<IDX_B_OFFSET>(blockOffset_)],
                        x1ScaleGlobal_[Get<IDX_X1SCALE_OFFSET>(blockOffset_)],
                        x2ScaleGlobal_[Get<IDX_X2SCALE_OFFSET>(blockOffset_)],
                        cGlobal_[Get<IDX_C_OFFSET>(blockOffset_)], mmSingleShape, transA, transB);
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
        // Check matrix size exceeds limit
        if (!transA && k > MATRIX_INNER_DIM_LIMIT_SIZE_V35) { // mk matrix k limit
            return Status::mkErrorMatrixExceedsLimit;
        }

        if (transA && m > MATRIX_INNER_DIM_LIMIT_SIZE_V35) { // km matrix m limit
            return Status::kmErrorMatrixExceedsLimit;
        }
        if (!transB && n > MATRIX_INNER_DIM_LIMIT_SIZE_V35) { // kn matrix n limit
            return Status::knErrorMatrixExceedsLimit;
        }

        if (transB && k > MATRIX_INNER_DIM_LIMIT_SIZE_V35) { // nk matrix k limit
            return Status::nkErrorMatrixExceedsLimit;
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
        // Check args for block epilogue
        CHECK_AND_RETURN(BlockEpilogue::CanImplement(args.epilogueArgs));
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
        BlockEpilogueParams epilogueParams = BlockEpilogue::InitParams(args.epilogueArgs, workspace);
        Params params = {args.problemShape, mmadParams, epilogueParams, args.gmmArgs};
        return params;
    }

    __host_aicore__ static int64_t GetBlockNum(ProblemShape shape)
    {
        return BlockSchedulerOp::GetBlockNum(shape);
    }

    __aicore__ inline void operator()(const Params& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        // Get blockIdx
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        AscendC::SetAtomicAdd<float>(); // GMM res + yIn
        // Instantiate mmadOp
        BlockMmadOp blockMmadOp;
        blockMmadOp.Init(const_cast<TCubeTiling* __restrict>(params.gmmParams.matmulTiling), GetTPipePtr());
        InitParamsAndTensor(params);
        uint32_t groupNum = params.gmmParams.groupNum;
        BlockSchedulerOp bs(params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
            if (!UpdateGroupParams(params, groupIdx)) {
                continue;
            }
            bs.UpdateNextProblem(problemShape_);
            ProcessSingleGroup(params, bs, blockMmadOp);
        }
        AscendC::SetAtomicNone();
    }
};
} // namespace Kernel
} // namespace Gemm
} // namespace Act
#endif // AOT_KERNEL_ASWT_GROUPED_MATMUL_H