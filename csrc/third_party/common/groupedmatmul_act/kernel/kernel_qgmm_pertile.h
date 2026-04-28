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
 * \file kernel_qgmm_pertile.h
 * \brief
 */

#ifndef MATMUL_KERNEL_KERNEL_QGMM_PERTILE_H
#define MATMUL_KERNEL_KERNEL_QGMM_PERTILE_H
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "../utils/common_utils.h"
#include "../utils/fill_utils.h"
#include "../utils/grouped_matmul_constant.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../utils/coord_utils.h"
#include "../utils/tensor_utils.h"

namespace Act {
namespace Gemm {
namespace Kernel {
#define QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS                                                                           \
    template <class ProblemShape, class BlockMmad, class BlockEpilogue, class BlockScheduler>
#define QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler

using namespace Act::Gemm::GroupedMatmul;

namespace {
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
} // namespace

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
class QuantMmGroupedPerTile {
public:
    __aicore__ inline QuantMmGroupedPerTile() {}
    __aicore__ inline ~QuantMmGroupedPerTile() {}

    static constexpr bool transA = BlockMmad::transA;
    static constexpr bool transB = BlockMmad::transB;

    // schedulerOp
    using BlockSchedulerOp = typename Block::BlockSchedulerSelector<ProblemShape, typename BlockMmad::L1TileShape,
                                                                    typename BlockMmad::L0TileShape, BlockScheduler,
                                                                    transA, transB>::SchedulerOp;

    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using YType = typename BlockEpilogue::YType;
    using LayoutB = typename BlockMmad::LayoutB;

    static constexpr CubeFormat FormatB = TagToFormat<LayoutB>::format;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    // x1,x2,x1Scale,x2Scale,bias,y
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using CoordClass = Coordinate<transA, transB, CubeFormat::ND, FormatB, CubeFormat::ND>;

    struct GMMTiling {
        int32_t m;
        int32_t n;
        int32_t k;
        int32_t baseM;
        int32_t baseN;
        int32_t baseK;
        int32_t stepM;
        int32_t stepN;
        int32_t stepKa;
        int32_t stepKb;
        uint32_t groupNum;
        int8_t groupType;
        uint8_t groupListType;
        __aicore__ GMMTiling() {}
        __aicore__ GMMTiling(int32_t m_, int32_t n_, int32_t k_, int32_t baseM_, int32_t baseN_, int32_t baseK_,
                             int32_t stepM_, int32_t stepN_, int32_t stepKa_, int32_t stepKb_, uint32_t groupNum_,
                             int8_t groupType_, uint8_t groupListType_) :
            m(m_), n(n_), k(k_), baseM(baseM_), baseN(baseN_), baseK(baseK_), stepM(stepM_), stepN(stepN_),
            stepKa(stepKa_), stepKb(stepKb_), groupNum(groupNum_), groupType(groupType_), groupListType(groupListType_)
        {}
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        GMMTiling gmmParams;
        Params() = default;
    };

public:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Run(const Params& params);
    __aicore__ inline void operator()(const Params& params)
    {
        Run(params);
    }

private:
    __aicore__ inline void SetMNK(uint32_t groupIdx);
    __aicore__ inline void ProcessSingleGroup(const Params& params, BlockSchedulerOp& bs, uint32_t groupIdx);
    __aicore__ inline void UpdateOffset(uint32_t groupIdx);
    __aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx);
    __aicore__ inline void UpdateMMGlobalAddr();
    __aicore__ inline void Iterate(int64_t singleCoreM, int64_t singleCoreN);
    __aicore__ inline void End();

private:
    BlockMmad mmadOp_;
    BlockEpilogue epilogueOp_;
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0};
    BlockOffset blockOffset_{0, 0, 0, 0, 0, 0};

    AscendC::GlobalTensor<AType> aGlobal_;
    AscendC::GlobalTensor<BType> bGlobal_;
    AscendC::GlobalTensor<int64_t> groupListGlobal_;
    AscendC::LocalTensor<CType> mmResPing_;
    AscendC::LocalTensor<CType> mmResPong_;
    AscendC::LocalTensor<YType> initLocal_;

    GM_ADDR groupListPtr_;
    GM_ADDR xTensorPtr_;
    GM_ADDR wTensorPtr_;
    GM_ADDR yTensorPtr_;

    uint32_t blockIdx_;
    int32_t preOffset_ = 0;
    uint32_t groupNum_;
    int8_t groupType_;
    uint8_t groupListType_;
};

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::Run(const Params& params)
{
    Init(params);
    bool isKZeroInit = false;
    BlockSchedulerOp bs(params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
    for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
        UpdateOffset(groupIdx);
        // Update input parameters M, N, K within the group
        SetMNK(groupIdx);
        if (Get<MNK_M>(problemShape_) <= 0 || Get<MNK_N>(problemShape_) <= 0) {
            continue;
        }
        if (Get<MNK_K>(problemShape_) <= 0) {
            // With K-axis grouping: output (m,n) required. int8 inputs disable K-axis grouping.
            // No bias (all zeros) when K-axis grouped.
            if ASCEND_IS_AIV {
                if (groupType_ == GMM_SPLIT_K) {
                    AscendC::GlobalTensor<YType> yInitGlobal;
                    yInitGlobal.SetGlobalBuffer(GetTensorAddr<YType>(0, yTensorPtr_) + Get<IDX_C_OFFSET>(baseOffset_));
                    InitOutputWithZero(yInitGlobal, initLocal_,
                                       static_cast<uint64_t>(Get<MNK_M>(problemShape_)) * Get<MNK_N>(problemShape_),
                                       AscendC::GetBlockNum(), isKZeroInit);
                }
            }
            continue;
        }
        if ASCEND_IS_AIC {
            mmadOp_.UpdateParamsForNextProblem(problemShape_);
        }
        if ASCEND_IS_AIV {
            epilogueOp_.UpdateParamsForNextProblem(problemShape_);
        }

        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> bsProblemShape{
            Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_), 0L};
        bs.UpdateNextProblem(bsProblemShape);

        UpdateMMGlobalAddr();
        ProcessSingleGroup(params, bs, groupIdx);
    }
    End();
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::Init(const Params& params)
{
    xTensorPtr_ = params.mmadParams.aGmAddr;
    wTensorPtr_ = params.mmadParams.bGmAddr;
    groupListPtr_ = params.mmadParams.groupListGmAddr;
    yTensorPtr_ = params.mmadParams.cGmAddr;

    groupNum_ = params.gmmParams.groupNum;
    groupType_ = params.gmmParams.groupType;
    groupListType_ = params.gmmParams.groupListType;

    blockIdx_ = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        blockIdx_ = blockIdx_ / AscendC::GetTaskRation();
    }

    if (groupListPtr_ != nullptr) {
        groupListGlobal_.SetGlobalBuffer((__gm__ int64_t*)groupListPtr_);
    }
    TupleShape l0Shape{static_cast<int64_t>(params.gmmParams.baseM), static_cast<int64_t>(params.gmmParams.baseN),
                       static_cast<int64_t>(params.gmmParams.baseK)};
    BlockShape tileL12L0{static_cast<int64_t>(params.gmmParams.stepM), static_cast<int64_t>(params.gmmParams.stepN),
                         static_cast<int64_t>(params.gmmParams.stepKa), static_cast<int64_t>(params.gmmParams.stepKb)};
    auto mmResPing_ = epilogueOp_.GetL0c2UbPingTensor();
    auto mmResPong_ = epilogueOp_.GetL0c2UbPongTensor();
    mmadOp_.Init(l0Shape, tileL12L0, &mmResPing_, &mmResPong_);
    epilogueOp_.Init(&params.epilogueParams);

    Get<MNK_M>(problemShape_) = params.gmmParams.m;
    Get<MNK_N>(problemShape_) = params.gmmParams.n;
    Get<MNK_K>(problemShape_) = params.gmmParams.k;
    if ASCEND_IS_AIV {
        // k = 0, init out
        if (AscendC::GetSubBlockIdx() == 0 && groupType_ == GMM_SPLIT_K) {
            uint32_t initSize = AscendC::MAX_REPEAT_TIMES * AscendC::ONE_BLK_SIZE;
            initLocal_ = AscendC::LocalTensor<YType>(AscendC::TPosition::VECCALC,
                                                     AscendC::GetUBSizeInBytes() - initSize, initSize / sizeof(YType));
        }
    }
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::UpdateOffset(uint32_t groupIdx)
{
    // baseOffset is 0 when groupIdx = 0
    if (groupIdx == 0) {
        return;
    }
    int64_t m = Get<MNK_M>(problemShape_);
    int64_t n = Get<MNK_N>(problemShape_);
    int64_t k = Get<MNK_K>(problemShape_);
    // aBaseOffset += m * k
    Get<IDX_A_OFFSET>(baseOffset_) += m * k;
    // bBaseOffset += n * k
    Get<IDX_B_OFFSET>(baseOffset_) += n * k;
    // G-B
    if constexpr (transA) { // split k, x1Scale:(k/gs+g, m) x2Scale:(k/gs+g, ceil(n/gs))
        int64_t scaleK = (Get<IDX_B_OFFSET>(baseOffset_) / n / PER_BLOCK_SIZE + groupIdx);
        Get<IDX_X1SCALE_OFFSET>(baseOffset_) = m * scaleK;
        Get<IDX_X2SCALE_OFFSET>(baseOffset_) = CeilDiv(n, PER_BLOCK_SIZE) * scaleK;
    } else { // split m, x1Scale:(m, ceil(k/gs)) x2Scale:(g, ceil(n/gs), ceil(k/gs)) or (g, ceil(k/gs), ceil(n/gs))
        int64_t scaleK = CeilDiv(k, PER_BLOCK_SIZE);
        Get<IDX_X1SCALE_OFFSET>(baseOffset_) += m * scaleK;
        Get<IDX_X2SCALE_OFFSET>(baseOffset_) += CeilDiv(n, PER_BLOCK_SIZE) * scaleK;
    }
    // yBaseOffset += m * n
    Get<IDX_C_OFFSET>(baseOffset_) += m * n;
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void
QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::ProcessSingleGroup(const Params& params,
                                                                              BlockSchedulerOp& bs, uint32_t groupIdx)
{
    CoordClass coord(Get<MNK_M>(problemShape_), Get<MNK_N>(problemShape_), Get<MNK_K>(problemShape_),
                     params.gmmParams.baseM, params.gmmParams.baseN, params.gmmParams.baseK);
    BlockCoord tileIdx;
    while (bs.GetTileIdx(tileIdx)) {
        BlockShape singleShape = bs.GetBlockShape(tileIdx);
        if (Get<MNK_M>(singleShape) <= 0 || Get<MNK_N>(singleShape) <= 0) {
            return;
        }
        blockOffset_ = coord.template GetQuantOffset<QuantMode::PERGROUP_MODE>(
            Get<IDX_M_TILEIDX>(tileIdx), Get<IDX_N_TILEIDX>(tileIdx), Get<IDX_M_TAIL_SPLIT_TILEIDX>(singleShape),
            Get<IDX_N_TAIL_SPLIT_TILEIDX>(singleShape));
        Iterate(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape));
    }
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::Iterate(int64_t singleCoreM,
                                                                                          int64_t singleCoreN)
{
    AscendC::Std::tuple<int64_t, int64_t, int64_t> blockShape{singleCoreM, singleCoreN,
                                                              static_cast<int64_t>(Get<MNK_K>(problemShape_))};
    if ASCEND_IS_AIC {
        mmadOp_(blockShape, aGlobal_[Get<IDX_A_OFFSET>(blockOffset_)], bGlobal_[Get<IDX_B_OFFSET>(blockOffset_)]);
    }
    if ASCEND_IS_AIV {
        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> blockCoord{
            static_cast<int64_t>(Get<IDX_C_OFFSET>(blockOffset_)),
            static_cast<int64_t>(Get<IDX_X2SCALE_OFFSET>(blockOffset_)),
            static_cast<int64_t>(Get<IDX_X1SCALE_OFFSET>(blockOffset_)), 0L};
        epilogueOp_(blockShape, blockCoord);
    }
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::SetMNK(uint32_t groupIdx)
{
    int32_t splitValue = GetSplitValueFromGroupList(groupIdx);
    if (groupType_ == GMM_SPLIT_M) {
        Get<MNK_M>(problemShape_) = splitValue;
    } else {
        Get<MNK_K>(problemShape_) = splitValue;
    }
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::UpdateMMGlobalAddr()
{
    if ASCEND_IS_AIC {
        // single MM
        aGlobal_.SetGlobalBuffer(GetTensorAddr<AType>(0, xTensorPtr_) + Get<IDX_A_OFFSET>(baseOffset_));
        bGlobal_.SetGlobalBuffer(GetTensorAddr<BType>(0, wTensorPtr_) + Get<IDX_B_OFFSET>(baseOffset_));
    }
    if ASCEND_IS_AIV {
        AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> baseOffset{
            static_cast<int64_t>(Get<IDX_C_OFFSET>(baseOffset_)),
            static_cast<int64_t>(Get<IDX_X2SCALE_OFFSET>(baseOffset_)),
            static_cast<int64_t>(Get<IDX_X1SCALE_OFFSET>(baseOffset_)), 0L};
        epilogueOp_.UpdateGlobalAddr(baseOffset);
    }
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::End()
{
    if ASCEND_IS_AIC {
        mmadOp_.End();
    }
}

QGMM_PERTILE_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline int32_t
QuantMmGroupedPerTile<QGMM_PERTILE_KERNEL_FUN_TEM_PARAMS>::GetSplitValueFromGroupList(uint32_t groupIdx)
{
    int32_t splitValue = 0;
    if (likely(groupType_ != -1)) { // -1: no  need to split
        if (groupListType_ == 0) {
            int32_t offset = static_cast<int32_t>(groupListGlobal_.GetValue(groupIdx));
            splitValue = offset - preOffset_;
            preOffset_ = offset;
        } else {
            splitValue = static_cast<int32_t>(groupListGlobal_.GetValue(groupIdx));
        }
    }
    return splitValue;
}

} // namespace Kernel
} // namespace Gemm
} // namespace Act

#endif
