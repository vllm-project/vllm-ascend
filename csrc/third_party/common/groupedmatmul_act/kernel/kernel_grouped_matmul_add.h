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
 * \file kernel_grouped_matmul add.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_GROUPED_MATMUL_ADD_H
#define MATMUL_KERNEL_KERNEL_GROUPED_MATMUL_ADD_H
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"

#include "../utils/common_utils.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/tensor_utils.h"
#include "../utils/status_utils.h"

#include "../block/block_grouped_matmul_builder.h"
#include "../epilogue/block_epilogue_empty.h"
#include "../block/block_scheduler_utils.h"
#include "../block/block_scheduler_grouped_matmul_aswt.h"

namespace Act {
namespace Gemm {
namespace Kernel {

constexpr uint64_t BLOCK_BYTE_SIZE = 32UL;
constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t NUM_ZERO = 0UL;
constexpr uint64_t NUM_ONE = 1UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t NUM_THREE = 3UL;

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_,
          typename Enable_ = void>
class KernelGroupedMatmulAdd {
    static_assert(AscendC::Std::always_false_v<BlockScheduler_>,
                  "KernelGroupedMatmulAdd is not implemented for this scheduler");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelGroupedMatmulAdd<
    ProblemShape_, BlockMmadBuilder_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<BlockScheduler_, GroupedMatmulAswtScheduler>>> {
public:
    __aicore__ inline KernelGroupedMatmulAdd() {}
    __aicore__ inline ~KernelGroupedMatmulAdd() {}

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
    using BiasType = typename BlockMmadBuilder::BiasType;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t>;
    // coordinate
    using CoordClass =
        Coordinate<transA, transB, BlockMmadBuilder::formatA, BlockMmadBuilder::formatB, BlockMmadBuilder::formatC>;

    // attribute
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<int64_t> groupListGm_;
    // mmad
    BlockMmadParams blockMmadParams_{};
    // shape
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0};
    int64_t preOffset_{0};

    struct GMMAddTiling {
        uint32_t groupNum;
        uint32_t groupListType;
        uint64_t mTailCnt;
        uint64_t nTailCnt;
        const TCubeTiling* __restrict matmulTiling;
        __aicore__ GMMAddTiling() {}
        __aicore__ GMMAddTiling(uint32_t groupNum_, uint32_t groupListType_, uint64_t mTailCnt_ = 1,
                                uint64_t nTailCnt_ = 1) :
            groupNum(groupNum_),
            groupListType(groupListType_), mTailCnt(mTailCnt_), nTailCnt(nTailCnt_)
        {}
    };

    struct Arguments {
        ProblemShape problemShape;
        BlockMmadArguments mmadArgs;
        BlockEpilogueArguments epilogueArgs;
        GMMAddTiling gmmArgs;
        Arguments() = default;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        GMMAddTiling gmmParams;
        Params() = default;
    };

    __aicore__ inline int64_t GetSplitValueFromGroupList(uint64_t groupIdx, uint64_t groupListType)
    {
        int64_t splitValue = 0;
        if (groupListType == 0) {
            int64_t offset = static_cast<int64_t>(groupListGm_.GetValue(groupIdx));
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else {
            splitValue = static_cast<int64_t>(groupListGm_.GetValue(groupIdx));
        }
        return splitValue;
    }

    __aicore__ inline void InitGlobalBuffer(const Params& params, uint64_t groupIdx)
    {
        aGlobal_.SetGlobalBuffer((__gm__ AType*)params.mmadParams.aGmAddr + Get<NUM_ZERO>(baseOffset_));
        bGlobal_.SetGlobalBuffer((__gm__ BType*)params.mmadParams.bGmAddr + Get<NUM_ONE>(baseOffset_));
        cGlobal_.SetGlobalBuffer((__gm__ CType*)params.mmadParams.cGmAddr + Get<NUM_TWO>(baseOffset_));
    }

    __aicore__ inline void UpdateOffset(uint64_t groupIdx)
    {
        if (groupIdx == 0) {
            return;
        }
        int64_t m = Get<M_VALUE>(problemShape_);
        int64_t n = Get<N_VALUE>(problemShape_);
        int64_t k = Get<K_VALUE>(problemShape_);
        // xBaseOffset += m * k
        Get<NUM_ZERO>(baseOffset_) = Get<NUM_ZERO>(baseOffset_) + m * k;
        // wBaseOffset += n * k
        Get<NUM_ONE>(baseOffset_) = Get<NUM_ONE>(baseOffset_) + n * k;
        // yBaseOffset += m * n
        Get<NUM_TWO>(baseOffset_) = Get<NUM_TWO>(baseOffset_) + m * n;
    }

    __aicore__ inline bool Init(const Params& params, uint64_t groupIdx)
    {
        UpdateOffset(groupIdx);
        int64_t splitValue = GetSplitValueFromGroupList(groupIdx, params.gmmParams.groupListType);
        Get<K_VALUE>(problemShape_) = splitValue;
        // Group by k-axis: skip
        if (splitValue <= 0) {
            return false;
        }
        InitGlobalBuffer(params, groupIdx);
        return true;
    }

    __aicore__ inline void InitParamsAndGm(const Params& params)
    {
        if (params.mmadParams.groupListGmAddr != nullptr) {
            groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(params.mmadParams.groupListGmAddr));
        }
        Get<M_VALUE>(problemShape_) = params.gmmParams.matmulTiling->M;
        Get<N_VALUE>(problemShape_) = params.gmmParams.matmulTiling->N;
        cGlobal_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }

    __host_aicore__ static Status CheckShape(const ProblemShape& shape)
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
        size_t workSpaceSize = 0;
        // Calculate extra workspace size for mmad
        workSpaceSize += BlockMmadBuilder::GetWorkspaceSize();
        // Calculate extra workspace size for epilogue
        workSpaceSize += BlockEpilogue::GetWorkspaceSize(blockNum, l1M, l1N);
        // Calculate extra workspace size for block scheduler
        workSpaceSize += BlockSchedulerOp::GetWorkspaceSize(shape);
        return workSpaceSize;
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
        // Instantiate mmadOp
        BlockMmadOp blockMmadOp;
        // Get blockIdx
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        if (curBlockIdx >= blockNum || blockNum == 0) {
            return;
        }
        InitParamsAndGm(params);
        int32_t baseM = params.gmmParams.matmulTiling->baseM;
        int32_t baseN = params.gmmParams.matmulTiling->baseN;
        int32_t baseK = params.gmmParams.matmulTiling->baseK;
        blockMmadOp.Init(const_cast<TCubeTiling* __restrict>(params.gmmParams.matmulTiling), GetTPipePtr());
        for (uint64_t groupIdx = 0, count = 0; groupIdx < params.gmmParams.groupNum; groupIdx++) {
            if (!Init(params, groupIdx)) {
                continue;
            }
            int64_t m = Get<M_VALUE>(problemShape_);
            int64_t n = Get<N_VALUE>(problemShape_);
            int64_t k = Get<K_VALUE>(problemShape_);
            CoordClass coord(m, n, k, baseM, baseN, baseK);
            BlockSchedulerOp bs(m, n, k, baseM, baseN, baseK, curBlockIdx, blockNum, params.gmmParams.mTailCnt,
                                params.gmmParams.nTailCnt, true);
            blockMmadOp.SetOrgShape(m, n, k);
            uint64_t curCount = count + bs.GetTileNum();
            uint64_t curBlock = curBlockIdx >= count ? curBlockIdx : curBlockIdx + blockNum;
            for (; curBlock < curCount; curBlock += blockNum) {
                BlockShape tileIdx = bs.GetTileIdx(curBlock, count);
                BlockShape singleShape = bs.GetBlockShape(Get<NUM_ZERO>(tileIdx), Get<NUM_ONE>(tileIdx), curBlock,
                                                          BLOCK_BYTE_SIZE / sizeof(BType));

                int64_t aOffset = coord.GetAOffset(Get<NUM_ZERO>(tileIdx), 0, 0, Get<NUM_TWO>(singleShape));
                int64_t bOffset = coord.GetBOffset(Get<NUM_ONE>(tileIdx), 0, 0, BLOCK_BYTE_SIZE / sizeof(BType),
                                                   Get<NUM_THREE>(singleShape));
                int64_t cOffset = coord.GetCOffset(Get<NUM_ZERO>(tileIdx), Get<NUM_ONE>(tileIdx), 0,
                                                   Get<NUM_TWO>(singleShape), Get<NUM_THREE>(singleShape));
                if (Get<NUM_ZERO>(singleShape) <= 0 || Get<NUM_ONE>(singleShape) <= 0) {
                    continue;
                }
                blockMmadOp.SetSingleShape(Get<NUM_ZERO>(singleShape), Get<NUM_ONE>(singleShape), k);
                blockMmadOp.SetTensorA(aGlobal_[aOffset], transA);
                blockMmadOp.SetTensorB(bGlobal_[bOffset], transB);
                blockMmadOp.IterateAll(cGlobal_[cOffset], 1);
            }
            count = curCount % blockNum;
        }
    }
};
} // namespace Kernel
} // namespace Gemm
} // namespace Act
#endif