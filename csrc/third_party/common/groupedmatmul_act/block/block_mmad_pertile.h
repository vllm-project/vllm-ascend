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
 * \file block_mmad_pertile.h
 * \brief
 */
#ifndef MATMUL_BLOCK_BLOCK_MMAD_PERTILE_H
#define MATMUL_BLOCK_BLOCK_MMAD_PERTILE_H

#include "block_scheduler_utils.h"
#include "lib/matmul/matmul.h"
#include "lib/matmul/tiling.h"

#include "../utils/grouped_matmul_constant.h"
#include "../utils/layout_utils.h"
#include "../utils/tuple_utils.h"
#include "../policy/dispatch_policy.h"
#include "../tile/tile_copy.h"
#include "block_mmad_pertile_param.h"

namespace Act {
namespace Gemm {
namespace Block {

struct PerBlockMmParam {
    bool fixpipeSplitN = false;
    uint64_t fixpipeN;
    uint64_t fixpipeM;
    uint64_t srcStride;
    uint64_t ndNum;
};

#define QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS                                                                             \
    template <class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_,                \
              class BiasType_, class LayoutBias_, class L1TileShape_, class L0TileShape_, class TileCopyParam_>

#define QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS                                                                              \
    GMMPerTile<>, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_, L1TileShape_,          \
        L0TileShape_, TileCopyParam_

using namespace Act::Gemm::GroupedMatmul;

template <class BlockMatmulPolicy_, class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_,
          class LayoutC_, class BiasType_, class LayoutBias_, class L1TileShape_, class L0TileShape_,
          class TileCopyParam_ = void>
class BlockMmadGmm {
    static_assert(AscendC::Std::always_false_v<BlockMatmulPolicy_>, "Should not be here!");
};

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
class BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS> {
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
    using TileCopyParam = TileCopyParam_;

    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;

    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;              // m,n,k
    using TupleTileShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>; // m,n,ka,kb
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

public:
    __aicore__ inline BlockMmadGmm();
    __aicore__ inline void Init(const TupleShape& l0Shape, const TupleTileShape& tileL12L0,
                                AscendC::LocalTensor<CType>* ping, AscendC::LocalTensor<CType>* pong);
    __aicore__ inline void operator()(const TupleShape& actualSingleShape, const AscendC::GlobalTensor<AType>& aGlobal,
                                      const AscendC::GlobalTensor<BType>& bGlobal);
    __aicore__ inline void UpdateParamsForNextProblem(const TupleShape& problemShape);
    __aicore__ inline void End();
    __aicore__ inline ~BlockMmadGmm();

private:
    __aicore__ inline void AicBaseMadProcess(AscendC::LocalTensor<AType>& aL1, AscendC::LocalTensor<BType>& bL1,
                                             uint64_t kInner, uint64_t kAL1Offset, bool isTailAL1, uint64_t kBL1Offset,
                                             bool isTailBL1);
    __aicore__ inline void CopyInA1Nd2Nz(const AscendC::GlobalTensor<AType>& aGlobal, uint64_t kOffset, bool isTailAL1);
    __aicore__ inline void CopyInB1Nd2Nz(const AscendC::GlobalTensor<BType>& bGlobal, uint64_t kOffset, bool isTailBL1);
    __aicore__ inline void CopyInA2(AscendC::LocalTensor<AType>& aL1, uint64_t mAL1Offset, uint64_t kAL1Offset,
                                    uint64_t kOffset, bool isTailAL1);
    __aicore__ inline void CopyInB2(AscendC::LocalTensor<BType>& bL1, uint64_t nBL1Offset, uint64_t kBL1Offset,
                                    uint64_t kOffset, bool isTailBL1);
    __aicore__ inline void MmadBase(uint64_t kOffset);

    __aicore__ inline void UpdatePerBlockMmParam();

    __aicore__ inline void WaitForVector(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreWaitFlag<GMM_AIC_SYNC_AIV_MODE, PIPE_FIX>(GMM_AIC_SYNC_AIV_FLAG + crossPingPongID);
        AscendC::CrossCoreWaitFlag<GMM_AIC_SYNC_AIV_MODE, PIPE_FIX>(GMM_AIC_SYNC_AIV_FLAG + crossPingPongID +
                                                                    GMM_FLAG_ID_MAX);
    }
    __aicore__ inline void NotifyVector(uint16_t crossPingPongID)
    {
        AscendC::CrossCoreSetFlag<GMM_AIC_SYNC_AIV_MODE, PIPE_FIX>(GMM_AIV_SYNC_AIC_FLAG + crossPingPongID);
        AscendC::CrossCoreSetFlag<GMM_AIC_SYNC_AIV_MODE, PIPE_FIX>(GMM_AIV_SYNC_AIC_FLAG + crossPingPongID +
                                                                   GMM_FLAG_ID_MAX);
    }

public:
    AscendC::LocalTensor<CType>* mmResPing_;
    AscendC::LocalTensor<CType>* mmResPong_;

    AscendC::LocalTensor<AType> aL1Ping_;
    AscendC::LocalTensor<AType> aL1Pong_;
    AscendC::LocalTensor<BType> bL1Ping_;
    AscendC::LocalTensor<BType> bL1Pong_;
    AscendC::LocalTensor<AType> aL0Ping_;
    AscendC::LocalTensor<AType> aL0Pong_;
    AscendC::LocalTensor<BType> bL0Ping_;
    AscendC::LocalTensor<BType> bL0Pong_;
    AscendC::LocalTensor<CType> cL0Ping_;
    AscendC::LocalTensor<CType> cL0Pong_;

private:
    TupleShape problemShape_;
    TupleShape actualSingleShape_;
    PerBlockMmParam mmParams_;
    MatMulCommonParam<transA, transB> matmulParam_;
    uint64_t baseCount_ = 0;
    uint64_t maxStepK_ = 0;
    uint64_t minStepK_ = 0;
    uint32_t baseM_;
    uint32_t baseN_;
    uint32_t baseK_;
    uint32_t stepM_;
    uint32_t stepN_;
    uint32_t stepKa_;
    uint32_t stepKb_;
    uint16_t crossPingPongID_ = 0;
    int32_t aL1PingPongID_ = 0;
    int32_t bL1PingPongID_= 0;
    int32_t l0PingPongID_ = 0;
    bool needAicWait_ = false;
    bool orderAL1BL1_ = false;
};

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::BlockMmadGmm()
{
    if ASCEND_IS_AIC {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0 + GMM_BUFFER_NUM);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1 + GMM_BUFFER_NUM);
        // 框架会使用AscendC::HardEvent::M_MTE1的0、1、2 eventID,后续都将使用GMM_CUBE_SYNC_MTE1_FLAG来避免eventID冲突
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0 + GMM_CUBE_SYNC_MTE1_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1 + GMM_CUBE_SYNC_MTE1_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
    }
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::Init(const TupleShape& l0Shape,
                                                                             const TupleTileShape& tileL12L0,
                                                                             AscendC::LocalTensor<CType>* ping,
                                                                             AscendC::LocalTensor<CType>* pong)
{
    if ASCEND_IS_AIC {
        baseM_ = Get<MNK_M>(l0Shape);
        baseN_ = Get<MNK_N>(l0Shape);
        baseK_ = Get<MNK_K>(l0Shape);
        stepM_ = Get<MNK_M>(tileL12L0);
        stepN_ = Get<MNK_N>(tileL12L0);
        stepKa_ = Get<2>(tileL12L0);  // 2: idx of stepKa in tileshape
        stepKb_ = Get<3>(tileL12L0);  // 3: idx of stepKb in tileshape
        orderAL1BL1_ = stepKa_ >= stepKb_;
        maxStepK_ = (orderAL1BL1_ ? stepKa_ : stepKb_) * baseK_;
        minStepK_ = (orderAL1BL1_ ? stepKb_ : stepKa_) * baseK_;
        matmulParam_.Init(l0Shape, tileL12L0);
        mmResPing_ = ping;
        mmResPong_ = pong;

        aL1Ping_ = AscendC::LocalTensor<AType>(AscendC::TPosition::A1, 0, baseM_ * baseK_ * stepKa_);
        aL1Pong_ = AscendC::LocalTensor<AType>(AscendC::TPosition::A1, baseM_ * baseK_ * stepKa_ * sizeof(AType),
                                               baseM_ * baseK_ * stepKa_);
        bL1Ping_ = AscendC::LocalTensor<BType>(AscendC::TPosition::B1,
                                               baseM_ * baseK_ * stepKa_ * sizeof(AType) * GMM_BUFFER_NUM,
                                               baseN_ * baseK_ * stepKb_);
        bL1Pong_ = AscendC::LocalTensor<BType>(
            AscendC::TPosition::B1,
            baseM_ * baseK_ * stepKa_ * sizeof(AType) * GMM_BUFFER_NUM + baseN_ * baseK_ * stepKb_ * sizeof(BType),
            baseN_ * baseK_ * stepKb_);
        aL0Ping_ = AscendC::LocalTensor<AType>(AscendC::TPosition::A2, 0, baseM_ * baseK_);
        aL0Pong_ =
            AscendC::LocalTensor<AType>(AscendC::TPosition::A2, baseM_ * baseK_ * sizeof(AType), baseM_ * baseK_);
        bL0Ping_ = AscendC::LocalTensor<BType>(AscendC::TPosition::B2, 0, baseN_ * baseK_);
        bL0Pong_ =
            AscendC::LocalTensor<BType>(AscendC::TPosition::B2, baseN_ * baseK_ * sizeof(BType), baseN_ * baseK_);
        cL0Ping_ = AscendC::LocalTensor<CType>(AscendC::TPosition::CO1, 0, baseM_ * baseN_);
        cL0Pong_ =
            AscendC::LocalTensor<CType>(AscendC::TPosition::CO1, baseM_ * baseN_ * sizeof(CType), baseM_ * baseN_);
    }
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::UpdateParamsForNextProblem(const TupleShape& problemShape)
{
    problemShape_ = problemShape;
    matmulParam_.UpdateForNextGroup(problemShape_);
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::UpdatePerBlockMmParam()
{
    mmParams_.fixpipeSplitN = Get<MNK_N>(actualSingleShape_) > PER_BLOCK_SIZE || Get<MNK_M>(actualSingleShape_) == 1;

    if constexpr (transA) {
        mmParams_.srcStride = Align(Get<MNK_M>(actualSingleShape_), static_cast<uint64_t>(AscendC::ONE_BLK_SIZE));
    } else {
        mmParams_.srcStride = Align(Get<MNK_M>(actualSingleShape_), static_cast<uint64_t>(AscendC::BLOCK_CUBE));
    }
    mmParams_.fixpipeM = mmParams_.fixpipeSplitN ?
                             Get<MNK_M>(actualSingleShape_) :
                             Align(Get<MNK_M>(actualSingleShape_), static_cast<uint64_t>(GetAicAivTaskRation()));
    if (mmParams_.fixpipeSplitN) {
        mmParams_.ndNum = Get<MNK_N>(actualSingleShape_) > UB_TWO_BANK_ELEMS_B32 ? 2 : 1; // 2: 2 ND
        int64_t alignedNBase =
            Get<MNK_N>(actualSingleShape_) > PER_BLOCK_SIZE ? PER_BLOCK_SIZE : AscendC::ONE_BLK_SIZE * mmParams_.ndNum;
        mmParams_.fixpipeN = Align(Get<MNK_N>(actualSingleShape_), static_cast<uint64_t>(alignedNBase));
    } else {
        mmParams_.ndNum = Get<MNK_N>(actualSingleShape_) > UB_SUB_BANK_ELEMS_B32 ? 2 : 1;
        mmParams_.fixpipeN =
            Align(Get<MNK_N>(actualSingleShape_), static_cast<uint64_t>(AscendC::BLOCK_CUBE) * mmParams_.ndNum);
    }
    mmParams_.fixpipeN /= mmParams_.ndNum;
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::operator()(const TupleShape& actualSingleShape,
                                                            const AscendC::GlobalTensor<AType>& aGlobal,
                                                            const AscendC::GlobalTensor<BType>& bGlobal)
{
    actualSingleShape_ = actualSingleShape;
    matmulParam_.UpdateNextBlockParams(actualSingleShape_);
    UpdatePerBlockMmParam();
    bool isTailAL1 = false;
    bool isTailBL1 = false;
    if (orderAL1BL1_) {
        for (uint64_t kOuter = 0; kOuter < Get<MNK_K>(problemShape_); kOuter += maxStepK_) {
            isTailAL1 = (kOuter + maxStepK_) >= Get<MNK_K>(problemShape_);
            CopyInA1Nd2Nz(aGlobal, kOuter, isTailAL1);
            for (uint64_t kInner = kOuter;
                 kInner < AscendC::Std::min(kOuter + maxStepK_, static_cast<uint64_t>(Get<MNK_K>(problemShape_)));
                 kInner += minStepK_) {
                isTailBL1 = (kInner + minStepK_) >= Get<MNK_K>(problemShape_);
                CopyInB1Nd2Nz(bGlobal, kInner, isTailBL1);
                uint64_t kAL1Offset = kInner - kOuter;
                AicBaseMadProcess(aL1PingPongID_ == 0 ? aL1Ping_ : aL1Pong_, bL1PingPongID_ == 0 ? bL1Ping_ : bL1Pong_,
                                  kInner, kAL1Offset, isTailAL1, 0UL, isTailBL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bL1PingPongID_ + GMM_BUFFER_NUM);
                bL1PingPongID_ = bL1PingPongID_ ^ 1;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1PingPongID_);
            aL1PingPongID_ = aL1PingPongID_ ^ 1;
        }
    } else {
        for (uint64_t kOuter = 0; kOuter < Get<MNK_K>(problemShape_); kOuter += maxStepK_) {
            isTailBL1 = (kOuter + maxStepK_) >= Get<MNK_K>(problemShape_);
            CopyInB1Nd2Nz(bGlobal, kOuter, isTailBL1);
            for (uint64_t kInner = kOuter;
                 kInner < AscendC::Std::min(kOuter + maxStepK_, static_cast<uint64_t>(Get<MNK_K>(problemShape_)));
                 kInner += minStepK_) {
                isTailAL1 = (kInner + minStepK_) >= Get<MNK_K>(problemShape_);
                CopyInA1Nd2Nz(aGlobal, kInner, isTailAL1);
                uint64_t kBL1Offset = kInner - kOuter;
                AicBaseMadProcess(aL1PingPongID_ == 0 ? aL1Ping_ : aL1Pong_, bL1PingPongID_ == 0 ? bL1Ping_ : bL1Pong_,
                                  kInner, 0UL, isTailAL1, kBL1Offset, isTailBL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1PingPongID_);
                aL1PingPongID_ = aL1PingPongID_ ^ 1;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bL1PingPongID_ + GMM_BUFFER_NUM);
            bL1PingPongID_ = bL1PingPongID_ ^ 1;
        }
    }
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::AicBaseMadProcess(
    AscendC::LocalTensor<AType>& aL1, AscendC::LocalTensor<BType>& bL1, uint64_t kInner, uint64_t kAL1Offset,
    bool isTailAL1, uint64_t kBL1Offset, bool isTailBL1)
{
    for (uint64_t kb = kInner;
         kb < AscendC::Std::min(kInner + minStepK_, static_cast<uint64_t>(Get<MNK_K>(problemShape_))); kb += baseK_) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPongID_ + GMM_CUBE_SYNC_MTE1_FLAG);
        CopyInA2(aL1, 0, kAL1Offset, kb, isTailAL1);
        CopyInB2(bL1, 0, kBL1Offset, kb, isTailBL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPongID_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPongID_);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0PingPongID_);
        MmadBase(kb);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPongID_ + GMM_CUBE_SYNC_MTE1_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0PingPongID_);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0PingPongID_);
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams(
            mmParams_.fixpipeN, mmParams_.fixpipeM, mmParams_.srcStride, UB_TWO_BANK_ELEMS_B32); // dstStride is 128
        fixpipeParams.dualDstCtl = mmParams_.fixpipeSplitN ? 2 : 1; // 2 means splitting N with ratio 1:2
        // When nz2nd loop in copyout, srcndstride is unit of c0Size, dstndstride is unit of one element.
        fixpipeParams.params.ndNum = mmParams_.ndNum;
        fixpipeParams.params.srcNdStride = mmParams_.srcStride * (mmParams_.fixpipeN / AscendC::BLOCK_CUBE);
        fixpipeParams.params.dstNdStride = UB_TWO_BANK_ELEMS_B32 * PER_BLOCK_SIZE;
        if (needAicWait_) {
            WaitForVector(crossPingPongID_);
        }
        AscendC::Fixpipe<CType, CType, AscendC::Impl::CFG_ROW_MAJOR_UB>(
            crossPingPongID_ == 0 ? *mmResPing_ : *mmResPong_, l0PingPongID_ == 0 ? cL0Ping_ : cL0Pong_, fixpipeParams);
        NotifyVector(crossPingPongID_);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0PingPongID_);
        needAicWait_ = needAicWait_ || crossPingPongID_ == 1;
        crossPingPongID_ = (crossPingPongID_ + 1) & 1;
        l0PingPongID_ = l0PingPongID_ ^ 1;
        kAL1Offset = kAL1Offset + baseK_;
        kBL1Offset = kBL1Offset + baseK_;
        baseCount_++;
    }
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::CopyInA2(AscendC::LocalTensor<AType>& aL1,
                                                                                 uint64_t mAL1Offset,
                                                                                 uint64_t kAL1Offset, uint64_t kOffset,
                                                                                 bool isTailAL1)
{
    uint64_t offsetAL1 = matmulParam_.CalcAL1Offset(mAL1Offset, kAL1Offset, isTailAL1);
    AscendC::LoadData2DParamsV2 loadData2dParams;
    matmulParam_.LoadData2dParamsA(loadData2dParams, kOffset, isTailAL1);
    AscendC::LoadData(l0PingPongID_ == 0 ? aL0Ping_ : aL0Pong_, aL1[offsetAL1], loadData2dParams);
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::CopyInB2(AscendC::LocalTensor<BType>& bL1,
                                                                                 uint64_t nBL1Offset,
                                                                                 uint64_t kBL1Offset, uint64_t kOffset,
                                                                                 bool isTailBL1)
{
    uint64_t offsetBL1 = matmulParam_.CalcBL1Offset(nBL1Offset, kBL1Offset, isTailBL1);
    AscendC::LoadData2DParamsV2 loadData2dParams;
    matmulParam_.LoadData2dParamsB(loadData2dParams, kOffset, isTailBL1);
    AscendC::LoadData(l0PingPongID_ == 0 ? bL0Ping_ : bL0Pong_, bL1[offsetBL1], loadData2dParams);
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::MmadBase(uint64_t kOffset)
{
    uint32_t mmadK = AscendC::Std::min(static_cast<uint64_t>(baseK_), Get<MNK_K>(problemShape_) - kOffset);
    AscendC::MmadParams mmadParams;
    if constexpr (transA) {
        mmadParams.m = Align(Get<MNK_M>(actualSingleShape_), static_cast<uint64_t>(GMM_DATA_BLOCK));
    } else {
        mmadParams.m = Get<MNK_M>(actualSingleShape_);
    }
    if constexpr (transB) {
        mmadParams.n = Get<MNK_N>(actualSingleShape_);
    } else {
        mmadParams.n = Align(Get<MNK_N>(actualSingleShape_), static_cast<uint64_t>(GMM_DATA_BLOCK));
    }
    mmadParams.k = mmadK;
    mmadParams.disableGemv = true;
    AscendC::Mmad(l0PingPongID_ == 0 ? cL0Ping_ : cL0Pong_, l0PingPongID_ == 0 ? aL0Ping_ : aL0Pong_,
                  l0PingPongID_ == 0 ? bL0Ping_ : bL0Pong_, mmadParams);
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::CopyInA1Nd2Nz(const AscendC::GlobalTensor<AType>& aGlobal,
                                                               uint64_t kOffset, bool isTailAL1)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1PingPongID_);   
    uint64_t offset = matmulParam_.CalcAGMOffsetInnerLoop(0, kOffset);
    AscendC::Nd2NzParams nd2nzParam;
    matmulParam_.CalNd2NzParamA(nd2nzParam, isTailAL1);
    AscendC::DataCopy(aL1PingPongID_ == 0 ? aL1Ping_ : aL1Pong_, aGlobal[offset], nd2nzParam);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aL1PingPongID_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aL1PingPongID_);
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::CopyInB1Nd2Nz(const AscendC::GlobalTensor<BType>& bGlobal,
                                                               uint64_t kOffset, bool isTailBL1)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(bL1PingPongID_ + GMM_BUFFER_NUM);      
    uint64_t offset = matmulParam_.CalcBGMOffsetInnerLoop(0, kOffset);
    AscendC::Nd2NzParams nd2nzParam;
    matmulParam_.CalNd2NzParamB(nd2nzParam, isTailBL1);
    AscendC::DataCopy(bL1PingPongID_ == 0 ? bL1Ping_ : bL1Pong_, bGlobal[offset], nd2nzParam);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bL1PingPongID_ + GMM_BUFFER_NUM);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bL1PingPongID_ + GMM_BUFFER_NUM);
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::End()
{
    if ASCEND_IS_AIC {
        if (baseCount_ > 1) {
            WaitForVector(0); // ping
            WaitForVector(1); // pong
        } else if (baseCount_ == 1) {
            WaitForVector(0);
        }
    }
}

QGMM_BLOCK_MMAD_CLASS_LOCAL_PARAMS
__aicore__ inline BlockMmadGmm<QGMM_BLOCK_MMAD_FUNC_LOCAL_PARAMS>::~BlockMmadGmm()
{
    if ASCEND_IS_AIC {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0 + GMM_BUFFER_NUM);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1 + GMM_BUFFER_NUM);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0 + GMM_CUBE_SYNC_MTE1_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1 + GMM_CUBE_SYNC_MTE1_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    }
}
}  // namespace Block
} // namespace Gemm
} // namespace Act
#endif
