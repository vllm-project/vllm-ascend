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
 * \file kernel_grouped_matmul.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_GROUPED_MATMUL_H
#define MATMUL_KERNEL_KERNEL_GROUPED_MATMUL_H
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

constexpr uint64_t SPLIT_M = 0UL;
constexpr uint64_t SPLIT_K = 2UL;
constexpr uint64_t MKN_LIST_LEN = 128UL;
constexpr uint64_t BLOCK_BYTE_SIZE = 32UL;
constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t NUM_ZERO = 0UL;
constexpr uint64_t NUM_ONE = 1UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t NUM_THREE = 3UL;
constexpr uint64_t NUM_FOUR = 4UL;

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_,
          typename Enable_ = void>
class KernelGroupedMatmul {
    static_assert(AscendC::Std::always_false_v<BlockScheduler_>,
                  "KernelGroupedMatmul is not implemented for this scheduler");
};

template <class ProblemShape_, class BlockMmadBuilder_, class BlockEpilogue_, class BlockScheduler_>
class KernelGroupedMatmul<
    ProblemShape_, BlockMmadBuilder_, BlockEpilogue_, BlockScheduler_,
    AscendC::Std::enable_if_t<AscendC::Std::is_same_v<BlockScheduler_, GroupedMatmulAswtScheduler>>> {
public:
    __aicore__ inline KernelGroupedMatmul()
    {
    }
    __aicore__ inline ~KernelGroupedMatmul()
    {
    }

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
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t>;
    // coordinate
    using CoordClass =
        Coordinate<transA, transB, BlockMmadBuilder::formatA, BlockMmadBuilder::formatB, BlockMmadBuilder::formatC>;

    // attribute
    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<BiasType> biasGlobal_;
    AscendC::GlobalTensor<CType> cGlobal_;
    AscendC::GlobalTensor<int64_t> groupListGm_;
    // mmad
    BlockMmadParams blockMmadParams_{};
    // shape
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0};
    int64_t preOffset_{0};
    TILING_TYPE *mListGm_;
    TILING_TYPE *kListGm_;
    TILING_TYPE *nListGm_;
    bool weightNzFlag_{false};
    bool tailSplit_{true};
    int64_t blockNum_{0};

    struct GMMTiling {
        uint32_t groupNum;
        int32_t groupType;
        uint32_t groupListType;
        int32_t baseM;
        int32_t baseN;
        int32_t baseK;
        uint64_t singleX;
        uint64_t singleWeight;
        uint64_t singleY;
        uint64_t singleTensor;
        uint32_t hasBias;
        uint64_t mTailCnt;
        uint64_t nTailCnt;
        const TCubeTiling *__restrict matmulTiling;
        TILING_TYPE *gmmArrayAddrIn;
        __aicore__ GMMTiling()
        {
        }
        __aicore__ GMMTiling(uint32_t groupNum_, int32_t groupType_, uint32_t groupListType_, int32_t baseM_,
                             int32_t baseN_, int32_t baseK_, uint64_t singleX_, uint64_t singleWeight_,
                             uint64_t singleY_, uint32_t hasBias_ = 0, uint64_t mTailCnt_ = 1, uint64_t nTailCnt_ = 1)
            : groupNum(groupNum_), groupType(groupType_), groupListType(groupListType_), baseM(baseM_), baseN(baseN_),
              baseK(baseK_), singleX(singleX_), singleWeight(singleWeight_), singleY(singleY_), hasBias(hasBias_),
              mTailCnt(mTailCnt_), nTailCnt(nTailCnt_)
        {
            singleTensor = singleX == 1 && singleWeight == 1 && singleY == 1;
        }
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

    __aicore__ inline int64_t GetSplitValueFromGroupList(uint64_t groupIdx, uint64_t groupListType, int32_t groupType)
    {
        int64_t splitValue = 0;
        if (groupType != -1) {
            if (groupListType == 0) {
                int64_t offset = static_cast<int64_t>(groupListGm_.GetValue(groupIdx));
                splitValue = offset - preOffset_;
                preOffset_ = offset;
            } else {
                splitValue = static_cast<int64_t>(groupListGm_.GetValue(groupIdx));
            }
        }
        return splitValue;
    }

    template <typename T>
    __aicore__ inline __gm__ T *GetTensorAddr(uint64_t groupIdx, GM_ADDR tensorPtr)
    {
        AscendC::ListTensorDesc listTensorDesc(reinterpret_cast<__gm__ void *>(tensorPtr));
        return listTensorDesc.GetDataPtr<T>(groupIdx);
    }

    __aicore__ inline void SetMKN(Params const &params, const int32_t splitValue, const uint32_t groupIdx)
    {
        uint32_t valueIdx = params.gmmParams.singleTensor ? 0 : groupIdx;
        if (params.gmmParams.groupType == SPLIT_M) {
            Get<M_VALUE>(problemShape_) = splitValue;
            Get<N_VALUE>(problemShape_) = nListGm_[valueIdx];
            Get<K_VALUE>(problemShape_) = kListGm_[valueIdx];
            return;
        }

        if (params.gmmParams.groupType == SPLIT_K) {
            Get<M_VALUE>(problemShape_) = mListGm_[valueIdx];
            Get<N_VALUE>(problemShape_) = nListGm_[valueIdx];
            Get<K_VALUE>(problemShape_) = splitValue;
            return;
        }

        Get<M_VALUE>(problemShape_) = mListGm_[groupIdx];
        Get<N_VALUE>(problemShape_) = nListGm_[groupIdx];
        Get<K_VALUE>(problemShape_) = kListGm_[groupIdx];
        return;
    }

    __aicore__ inline void InitGlobalBuffer(Params const &params, uint64_t groupIdx)
    {
        if (params.gmmParams.singleX == 0) {
            aGlobal_.SetGlobalBuffer(GetTensorAddr<AType>(groupIdx, params.mmadParams.aGmAddr));
        } else {
            aGlobal_.SetGlobalBuffer(GetTensorAddr<AType>(0, params.mmadParams.aGmAddr) + Get<NUM_ZERO>(baseOffset_));
        }

        if (params.gmmParams.singleWeight == 0) {
            bGlobal_.SetGlobalBuffer(GetTensorAddr<BType>(groupIdx, params.mmadParams.bGmAddr));
            if (params.gmmParams.hasBias != 0) {
                biasGlobal_.SetGlobalBuffer(GetTensorAddr<BiasType>(groupIdx, params.mmadParams.biasGmAddr));
            }
        } else {
            bGlobal_.SetGlobalBuffer(GetTensorAddr<BType>(0, params.mmadParams.bGmAddr) + Get<NUM_ONE>(baseOffset_));
            if (params.gmmParams.hasBias != 0) {
                biasGlobal_.SetGlobalBuffer(GetTensorAddr<BiasType>(0, params.mmadParams.biasGmAddr) +
                                            Get<NUM_THREE>(baseOffset_));
            }
        }

        if (params.gmmParams.singleY == 0) {
            cGlobal_.SetGlobalBuffer(GetTensorAddr<CType>(groupIdx, params.mmadParams.cGmAddr));
        } else {
            cGlobal_.SetGlobalBuffer(GetTensorAddr<CType>(0, params.mmadParams.cGmAddr) + Get<NUM_FOUR>(baseOffset_));
        }
    }

    __aicore__ inline void UpdateOffset()
    {
        int64_t m = Get<M_VALUE>(problemShape_);
        int64_t n = Get<N_VALUE>(problemShape_);
        int64_t k = Get<K_VALUE>(problemShape_);
        // xBaseOffset += m * k
        Get<NUM_ZERO>(baseOffset_) = Get<NUM_ZERO>(baseOffset_) + m * k;
        if (weightNzFlag_) {
            int64_t c0 = BLOCK_BYTE_SIZE / sizeof(BType);
            if (transB) {
                // wBaseOffset += Align(n, 16) * Align(k, 16)
                Get<NUM_ONE>(baseOffset_) = Get<NUM_ONE>(baseOffset_) + CeilAlign(n, OUTER_SIZE) * CeilAlign(k, c0);
            } else {
                // wBaseOffset += Align(n, 16) * Align(k, 16)
                Get<NUM_ONE>(baseOffset_) = Get<NUM_ONE>(baseOffset_) + CeilAlign(n, c0) * CeilAlign(k, OUTER_SIZE);
            }
        } else {
            // wBaseOffset += n * k
            Get<NUM_ONE>(baseOffset_) = Get<NUM_ONE>(baseOffset_) + n * k;
        }
        // mAxisBaseOffset += m
        Get<NUM_TWO>(baseOffset_) = Get<NUM_TWO>(baseOffset_) + m;
        // nAxisBaseOffset += n
        Get<NUM_THREE>(baseOffset_) = Get<NUM_THREE>(baseOffset_) + n;
        // yBaseOffset += m * n
        Get<NUM_FOUR>(baseOffset_) = Get<NUM_FOUR>(baseOffset_) + m * n;
    }

    __aicore__ inline bool Init(Params const &params, uint64_t groupIdx)
    {
        weightNzFlag_ = BlockMmadBuilder::formatB == CubeFormat::NZ;
        UpdateOffset();
        int64_t splitValue =
            GetSplitValueFromGroupList(groupIdx, params.gmmParams.groupListType, params.gmmParams.groupType);
        SetMKN(params, splitValue, groupIdx);
        // when group num equal 1 and split m less than shape m , revert aswt tail split
        if (params.gmmParams.groupType == SPLIT_M && params.gmmParams.groupNum == 1) {
            tailSplit_ = Get<M_VALUE>(problemShape_) == params.gmmParams.matmulTiling->M;
        }
        // Group along m-axis: skip current group and continue when m=0c
        if (params.gmmParams.groupType == SPLIT_M && Get<M_VALUE>(problemShape_) <= 0) {
            return false;
        }
        // Group by k-axis: executed by AIV when k=0
        if (params.gmmParams.groupType == SPLIT_K && Get<K_VALUE>(problemShape_) <= 0) {
            return false;
        }
        InitGlobalBuffer(params, groupIdx);
        return true;
    }

    __aicore__ inline void setMKNGm(Params const &params)
    {
        mListGm_ = params.gmmParams.gmmArrayAddrIn;
        kListGm_ = params.gmmParams.gmmArrayAddrIn + MKN_LIST_LEN;
        nListGm_ = params.gmmParams.gmmArrayAddrIn + MKN_LIST_LEN * NUM_TWO;
    }

    __host_aicore__ static Status CheckShape(ProblemShape const &shape)
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

    __host_aicore__ static Status CanImplement(Arguments const &args)
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

    __host_aicore__ static Params InitParams(Arguments const &args, GM_ADDR workspace)
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

    __aicore__ inline uint64_t IterateOneGroup(Params const &params, BlockMmadOp &blockMmadOp, int64_t curBlockIdx,
                                               uint64_t count)
    {
        int64_t m = Get<M_VALUE>(problemShape_);
        int64_t n = Get<N_VALUE>(problemShape_);
        int64_t k = Get<K_VALUE>(problemShape_);
        CoordClass coord(m, n, k, params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
        BlockSchedulerOp bs(m, n, k, params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK,
                            curBlockIdx, blockNum_, params.gmmParams.mTailCnt, params.gmmParams.nTailCnt, tailSplit_);
        blockMmadOp.SetOrgShape(m, n, k);
        uint64_t curCount = count + bs.GetTileNum();
        uint64_t curBlock = curBlockIdx >= count ? curBlockIdx : curBlockIdx + blockNum_;
        for (; curBlock < curCount; curBlock += blockNum_) {
            BlockShape tileIdx = bs.GetTileIdx(curBlock, count);
            BlockShape singleShape = bs.GetBlockShape(Get<NUM_ZERO>(tileIdx), Get<NUM_ONE>(tileIdx), curBlock,
                                                      BLOCK_BYTE_SIZE / sizeof(BType), weightNzFlag_);
            if (Get<NUM_ZERO>(singleShape) <= 0 || Get<NUM_ONE>(singleShape) <= 0) {
                continue;
            }
            int64_t aOffset = coord.GetAOffset(Get<NUM_ZERO>(tileIdx), 0, 0, Get<NUM_TWO>(singleShape));
            int64_t bOffset = coord.GetBOffset(Get<NUM_ONE>(tileIdx), 0, 0, BLOCK_BYTE_SIZE / sizeof(BType),
                                               Get<NUM_THREE>(singleShape));
            int64_t cOffset = coord.GetCOffset(Get<NUM_ZERO>(tileIdx), Get<NUM_ONE>(tileIdx), 0,
                                               Get<NUM_TWO>(singleShape), Get<NUM_THREE>(singleShape));
            blockMmadOp.SetSingleShape(Get<NUM_ZERO>(singleShape), Get<NUM_ONE>(singleShape), k);
            blockMmadOp.SetTensorA(aGlobal_[aOffset], transA);
            blockMmadOp.SetTensorB(bGlobal_[bOffset], transB);
            if (params.gmmParams.hasBias != 0) {
                int64_t biasOffset = coord.GetBiasOffset(Get<NUM_ONE>(tileIdx), Get<NUM_THREE>(singleShape));
                blockMmadOp.SetBias(biasGlobal_[biasOffset]);
            }
            blockMmadOp.IterateAll(cGlobal_[cOffset]);
        }
        return curCount % blockNum_;
    }

    __aicore__ inline void operator()(Params const &params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        // Instantiate mmadOp
        BlockMmadOp blockMmadOp;
        // Get blockIdx
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        blockNum_ = AscendC::GetBlockNum();
        if (curBlockIdx >= blockNum_ || blockNum_ == 0) {
            return;
        }
        if (params.mmadParams.groupListGmAddr != nullptr) {
            groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(params.mmadParams.groupListGmAddr));
        }
        setMKNGm(params);
        blockMmadOp.Init(const_cast<TCubeTiling *__restrict>(params.gmmParams.matmulTiling), GetTPipePtr());
        for (uint64_t groupIdx = 0, count = 0; groupIdx < params.gmmParams.groupNum; groupIdx++) {
            if (!Init(params, groupIdx)) {
                continue;
            }
            count = IterateOneGroup(params, blockMmadOp, curBlockIdx, count);
        }
    }
};
} // namespace Kernel
} // namespace Gemm
} // namespace Act
#endif