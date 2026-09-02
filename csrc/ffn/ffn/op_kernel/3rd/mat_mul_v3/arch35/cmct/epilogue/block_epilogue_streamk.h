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
 * \file block_epilogue_streamk.h
 * \brief
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "fusion/default_fusion_op.h"
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/status_utils.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <class WorkspaceType_, class OutType_, class DispatchPolicy_>
class BlockEpilogueStreamK {
public:
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;

    struct Arguments {
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        GM_ADDR x3GmAddr{nullptr};
    };

    using Params = Arguments;

    __aicore__ inline BlockEpilogueStreamK() {}
    __aicore__ inline ~BlockEpilogueStreamK() {}

    using WorkspaceType = WorkspaceType_;
    using OutType = OutType_;
    using DispatchPolicy = DispatchPolicy_;

    AscendC::GlobalTensor<OutType> cGlobal_;
    AscendC::GlobalTensor<WorkspaceType> workspaceGlobal_;

    // basic args
    uint64_t m_ = 0;
    uint64_t n_ = 0;
    uint64_t mL1_ = 0;
    uint64_t nL1_ = 0;
    uint64_t mCnt_ = 0;
    uint64_t nCnt_ = 0;
    uint64_t kCnt_ = 0;
    uint64_t bCnt_ = 0;
    uint64_t round_ = 1;
    uint64_t usedCoreNum_ = 0;
    uint64_t aivMte2Num_ = 0;

    constexpr static uint16_t ZERO_FLAG = 0;

    struct AivParams {
        uint64_t indexParams = 0;
        uint64_t mCntIndex = 0;
        uint64_t nCntIndex = 0;
        uint64_t kCntIndex = 0;
        uint64_t bCntIndex = 0;
        uint64_t curML1InAiv = 0;
        uint64_t curNL1InAiv = 0;
        uint64_t curAlignedNInAiv = 0;
    };
    AivParams aivParams_;

    uint64_t mBurstBase_ = 0;

    struct CopyGm2UbParams {
        uint64_t offsetWorkspaceGM = 0;
        uint64_t kCnt = 0;
        uint64_t mBurstOri = 0;
        uint64_t mBurst = 0;
        uint64_t burstLen = 0;
        uint64_t srcGap = 0;
    };
    CopyGm2UbParams copyGm2UbParams_;

    struct CopyUb2GmParams {
        uint64_t offsetCGm = 0;
        uint64_t mLength = 0;
        uint64_t burstLen = 0;
        uint64_t dstGap = 0;
        uint64_t srcGap = 0;
    };
    CopyUb2GmParams copyUb2GmParams_;

    __aicore__ inline void Init(Params const& params, BlockShape blockShapeInAiv, BlockShape tileL1ShapeInAiv,
                                BlockCoord coordInAiv, uint64_t usedCoreNum, bool checkIsSkScene)
    {
        m_ = Get<MNK_M>(blockShapeInAiv);
        n_ = Get<MNK_N>(blockShapeInAiv);
        mL1_ = Get<MNK_M>(tileL1ShapeInAiv);
        nL1_ = Get<MNK_N>(tileL1ShapeInAiv);
        mCnt_ = Get<MNK_M>(coordInAiv);
        nCnt_ = Get<MNK_N>(coordInAiv);
        kCnt_ = Get<MNK_K>(coordInAiv);
        bCnt_ = Get<MNK_B>(coordInAiv);
        usedCoreNum_ = usedCoreNum;
        // Decrease tile size of per vector core to prevent data race of cube and vector
        aivMte2Num_ = checkIsSkScene ? AscendC::GetTaskRation() : AscendC::BLOCK_CUBE;
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ OutType*>(params.cGmAddr));
        workspaceGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ WorkspaceType*>(params.workspaceGmAddr));
        ICachePreLoad(NUM_TWO);
        // Ensure cube to pair with vector, add sync flag in dp+sk scene
        if (!checkIsSkScene) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
        }
    }

    __aicore__ inline void Run()
    {
        UpdateAivBasicIndex();
        UpdateAivBasicBlock();
        for (uint64_t index = 0; index < aivMte2Num_; ++index) {
            UpdateAivParams(index);
            LocalTensor<float> ubAddTensor{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
            DataCopyExtParams dataCopyExtParams{static_cast<uint16_t>(copyGm2UbParams_.kCnt),
                                                static_cast<uint32_t>(copyGm2UbParams_.burstLen * sizeof(float)),
                                                static_cast<uint32_t>(copyGm2UbParams_.srcGap * sizeof(float)), 0, 0};
            if (copyGm2UbParams_.mBurst == 0) {
                return;
            }
            DataCopyPad<float>(ubAddTensor, workspaceGlobal_[copyGm2UbParams_.offsetWorkspaceGM], dataCopyExtParams,
                               {false, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);

            for (uint64_t i = 1; i < copyGm2UbParams_.kCnt; ++i) {
                Add(ubAddTensor, ubAddTensor, ubAddTensor[i * copyGm2UbParams_.burstLen], copyGm2UbParams_.burstLen);
            }

            DataCopyExtParams ub2gmExtParams{
                static_cast<uint16_t>(copyUb2GmParams_.mLength),
                static_cast<uint32_t>(copyUb2GmParams_.burstLen * sizeof(OutType)),
                static_cast<uint32_t>(copyUb2GmParams_.srcGap * sizeof(OutType) / UB2GM_SRCGAP_UNIT),
                static_cast<uint32_t>(copyUb2GmParams_.dstGap * sizeof(OutType)), 0};

            if constexpr (DispatchPolicy::enableRelu &&
                          (sizeof(OutType) != sizeof(half) || AscendC::IsSameType<OutType, bfloat16_t>::value)) {
                AscendC::Relu(ubAddTensor, ubAddTensor, copyGm2UbParams_.burstLen);
            }
            if constexpr (sizeof(OutType) == sizeof(half)) {
                LocalTensor<OutType> ubCastDst{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
                Cast(ubCastDst, ubAddTensor, RoundMode::CAST_RINT, copyGm2UbParams_.burstLen);
                if constexpr (DispatchPolicy::enableRelu && !AscendC::IsSameType<OutType, bfloat16_t>::value) {
                    // Relu not support bfloat16_t
                    AscendC::Relu(ubCastDst, ubCastDst, copyGm2UbParams_.burstLen);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
                DataCopyPad<OutType, (DispatchPolicy::fixpOpti_ == MatMulL0C2Out::ND_FIXPIPE_1_2) ?
                                         PaddingMode::Normal :
                                         PaddingMode::Compact>(cGlobal_[copyUb2GmParams_.offsetCGm], ubCastDst,
                                                               ub2gmExtParams);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ZERO_FLAG);
                DataCopyPad<OutType, (DispatchPolicy::fixpOpti_ == MatMulL0C2Out::ND_FIXPIPE_1_2) ?
                                         PaddingMode::Normal :
                                         PaddingMode::Compact>(cGlobal_[copyUb2GmParams_.offsetCGm], ubAddTensor,
                                                               ub2gmExtParams);
            }
        }
    }

    __aicore__ inline void UpdateAivBasicIndex()
    {
        uint64_t newBlockIdx = AscendC::GetBlockIdx() / (AscendC::GetTaskRation() * kCnt_);
        // kCntIndex为m方向的切分
        aivParams_.kCntIndex = AscendC::GetBlockIdx() % (AscendC::GetTaskRation() * kCnt_);
        aivParams_.bCntIndex = newBlockIdx / (mCnt_ * nCnt_);
        aivParams_.indexParams = newBlockIdx;
        uint64_t mnIndex = (aivParams_.indexParams + (bCnt_ * mCnt_ * nCnt_ - bCnt_ * mCnt_ * nCnt_ % usedCoreNum_)) %
                           (mCnt_ * nCnt_);
        uint64_t mainWindow = AscendC::Std::min(MAIN_WINDOW, mCnt_);
        uint64_t mainRow = mCnt_ / mainWindow - 1UL;
        uint64_t tailWindow = mCnt_ - mainRow * mainWindow;
        uint64_t rowIdx = mnIndex / nCnt_ / mainWindow;
        if (rowIdx < mainRow) {
            aivParams_.mCntIndex = rowIdx * mainWindow + mnIndex % mainWindow;
            aivParams_.nCntIndex = (mnIndex / mainWindow) % nCnt_;
        } else {
            rowIdx = mainRow;
            uint64_t tailIndex = mnIndex - mainRow * mainWindow * nCnt_;
            aivParams_.mCntIndex = mainRow * mainWindow + tailIndex % tailWindow;
            aivParams_.nCntIndex = (tailIndex / tailWindow) % nCnt_;
        }
        // mod 2 means even row, need reverse scan
        if (rowIdx % NUM_TWO != 0UL) {
            aivParams_.nCntIndex = nCnt_ - 1UL - aivParams_.nCntIndex;
        }
    }

    __aicore__ inline void UpdateAivBasicBlock()
    {
        if (round_ < NUM_TWO) {
            aivParams_.curML1InAiv = aivParams_.mCntIndex != (mCnt_ - 1) ? mL1_ : (m_ - (mCnt_ - 1) * mL1_);
            aivParams_.curNL1InAiv = aivParams_.nCntIndex != (nCnt_ - 1) ? nL1_ : (n_ - (nCnt_ - 1) * nL1_);
            if constexpr (DispatchPolicy::fixpOpti_ == MatMulL0C2Out::ND_FIXPIPE_1_2) {
                aivParams_.curAlignedNInAiv = CeilAlign(aivParams_.curNL1InAiv, AscendC::ONE_BLK_SIZE);
            } else {
                aivParams_.curAlignedNInAiv = aivParams_.curNL1InAiv;
            }
        }
    }

    __aicore__ inline void UpdateAivParams(uint64_t index)
    {
        mBurstBase_ = CeilAlign(CeilDiv(aivParams_.curML1InAiv, kCnt_ * AscendC::GetTaskRation()),
                                CeilDiv(UB2GM_SRCGAP_UNIT, aivParams_.curAlignedNInAiv));
        uint64_t mBurstCnt = CeilDiv(aivParams_.curML1InAiv, mBurstBase_);
        uint64_t mBurstTail = aivParams_.curML1InAiv - (mBurstCnt - 1) * mBurstBase_;
        if (aivParams_.kCntIndex >= mBurstCnt) {
            copyGm2UbParams_.mBurstOri = 0;
        } else {
            copyGm2UbParams_.mBurstOri = (aivParams_.kCntIndex == mBurstCnt - 1) ? mBurstTail : mBurstBase_;
        }

        copyGm2UbParams_.kCnt = kCnt_;
        copyGm2UbParams_.mBurst = CeilDiv(copyGm2UbParams_.mBurstOri, aivMte2Num_);
        // Calculate init address of workspace for moving into UB.
        copyGm2UbParams_.offsetWorkspaceGM = aivParams_.indexParams * kCnt_ * BLOCK_BASE_M * BLOCK_BASE_N +
                                             (aivParams_.kCntIndex * mBurstBase_ + copyGm2UbParams_.mBurst * index) *
                                                 aivParams_.curAlignedNInAiv;
        // Calculate init address of GM for moving out to GM.
        copyUb2GmParams_.offsetCGm = aivParams_.nCntIndex * nL1_ + aivParams_.mCntIndex * mL1_ * n_ +
                                     (aivParams_.kCntIndex * mBurstBase_ + copyGm2UbParams_.mBurst * index) * n_ +
                                     aivParams_.bCntIndex * m_ * n_;
        uint64_t singleCnt = 1;
        if (index == singleCnt - 1) {
            copyGm2UbParams_.mBurst = copyGm2UbParams_.mBurstOri - (singleCnt - 1) * copyGm2UbParams_.mBurst;
        } else if (index >= singleCnt) {
            copyGm2UbParams_.mBurst = 0;
        }
        // datasize for moving in ub, align to 32B
        copyGm2UbParams_.burstLen = CeilAlign(copyGm2UbParams_.mBurst * aivParams_.curAlignedNInAiv, BLOCK_SIZE);
        // gap of src between cur burst and next burst
        copyGm2UbParams_.srcGap = BLOCK_BASE_M * BLOCK_BASE_N - copyGm2UbParams_.burstLen;

        // args for ub2gm
        copyUb2GmParams_.mLength = copyGm2UbParams_.mBurst;
        copyUb2GmParams_.burstLen = aivParams_.curNL1InAiv;
        copyUb2GmParams_.dstGap = n_ - aivParams_.curNL1InAiv;
        copyUb2GmParams_.srcGap = aivParams_.curAlignedNInAiv - aivParams_.curNL1InAiv;
    }

    __aicore__ inline void operator()() { Run(); }

private:
    constexpr static uint64_t BLOCK_BASE_M = 256UL;
    constexpr static uint64_t BLOCK_BASE_N = 256UL;
    constexpr static uint64_t NUM_TWO = 2UL;
    constexpr static uint64_t MAIN_WINDOW = 4UL;
    constexpr static uint64_t UB2GM_SRCGAP_UNIT = 32UL;
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
