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
 * \file block_epilogue_streamk_fusion.h
 * \brief BlockEpilogue for StreamK path with FusionOp (Add/Mul).
 *        After accumulating K partial sums from workspace in UB, this epilogue
 *        reads x3 from GM, does Add/Mul in UB, then writes the fused result to GM.
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "fusion/default_fusion_op.h"
#include "../utils/common_utils.h"
#include "../utils/status_utils.h"
#include "../utils/device_utils.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <class WorkspaceType_, class OutType_, class DispatchPolicy_, int8_t INNER_PRECISE = 0>
class BlockEpilogueStreamKFusion {
public:
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;

    struct Arguments {
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        GM_ADDR x3GmAddr{nullptr};
        bool x3BatchBroadcast{false};
        uint64_t x3M{0};
    };

    using Params = Arguments;

    __aicore__ inline BlockEpilogueStreamKFusion() {}
    __aicore__ inline ~BlockEpilogueStreamKFusion() {}

    using WorkspaceType = WorkspaceType_;
    using OutType = OutType_;
    using DispatchPolicy = DispatchPolicy_;
    constexpr static MatMulL0C2Out L0C2OutModel = DispatchPolicy_::fixpOpti_;

    AscendC::GlobalTensor<OutType> cGlobal_;
    AscendC::GlobalTensor<WorkspaceType> workspaceGlobal_;
    AscendC::GlobalTensor<OutType> x3Global_;

    // basic args
    uint64_t m_ = 0;
    uint64_t n_ = 0;
    uint64_t mL1_ = 0;
    uint64_t nL1_ = 0;
    uint64_t bCnt_ = 0;
    uint64_t mCnt_ = 0;
    uint64_t nCnt_ = 0;
    uint64_t kCnt_ = 0;
    uint64_t round_ = 1;
    uint64_t aivMte2Num_ = 0;
    uint64_t usedCoreNum_ = 0;
    bool x3BatchBroadcast_ = false;
    uint64_t x3M_ = 0;

    constexpr static uint16_t FLAG_0 = 0;

    struct AivParams {
        uint64_t indexParams = 0;
        uint64_t bCntIndex = 0;
        uint64_t mCntIndex = 0;
        uint64_t nCntIndex = 0;
        uint64_t kCntIndex = 0;
        uint64_t curML1InAiv = 0;
        uint64_t curNL1InAiv = 0;
        uint64_t curAlignedNInAiv = 0;
    };
    AivParams aivParams_;

    uint64_t mBurstBase_ = 0;

    struct CopyGm2UbParams {
        uint64_t kCnt = 0;
        uint64_t offsetWorkspaceGM = 0;
        uint64_t mBurstOri = 0;
        uint64_t mBurst = 0;
    };
    CopyGm2UbParams copyGm2UbParams_;

    struct CopyUb2GmParams {
        uint64_t mLength = 0;
        uint64_t offsetCGm = 0;
    };
    CopyUb2GmParams copyUb2GmParams_;

    __aicore__ inline void Init(Params const& params, BlockShape blockShapeInAiv, BlockShape tileL1ShapeInAiv,
                                BlockCoord coordInAiv, uint64_t usedCoreNum, bool checkIsSkScene)
    {
        m_ = Get<MNK_M>(blockShapeInAiv);
        n_ = Get<MNK_N>(blockShapeInAiv);
        mL1_ = Get<MNK_M>(tileL1ShapeInAiv);
        nL1_ = Get<MNK_N>(tileL1ShapeInAiv);
        bCnt_ = Get<MNK_B>(coordInAiv);
        mCnt_ = Get<MNK_M>(coordInAiv);
        nCnt_ = Get<MNK_N>(coordInAiv);
        kCnt_ = Get<MNK_K>(coordInAiv);
        usedCoreNum_ = usedCoreNum;
        aivMte2Num_ = checkIsSkScene ? AscendC::GetTaskRation() : AscendC::BLOCK_CUBE;
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ OutType*>(params.cGmAddr));
        workspaceGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ WorkspaceType*>(params.workspaceGmAddr));
        x3Global_.SetGlobalBuffer(reinterpret_cast<__gm__ OutType*>(params.x3GmAddr));
        x3BatchBroadcast_ = params.x3BatchBroadcast;
        x3M_ = params.x3M;
        ICachePreLoad(NUM_TWO);
        if (!checkIsSkScene) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(FLAG_0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(FLAG_0);
        }
    }

    __aicore__ inline void Run()
    {
        UpdateAivBasicIndex();
        UpdateAivBasicBlock();
        for (uint64_t index = 0; index < aivMte2Num_; index++) {
            UpdateAivParams(index);
            if (copyGm2UbParams_.mBurst == 0) {
                return;
            }
            RunFusion();
        }
    }

    __aicore__ inline void RunFusion()
    {
        LocalTensor<float> ubAddTensor{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
        uint64_t paddedN = CeilAlign(aivParams_.curNL1InAiv, BLOCK_SIZE);
        uint64_t tilePaddedSize = copyGm2UbParams_.mBurst * paddedN;

        uint32_t wsRowBytes = static_cast<uint32_t>(aivParams_.curNL1InAiv * sizeof(float));
        uint32_t wsSrcGap = static_cast<uint32_t>((aivParams_.curAlignedNInAiv - aivParams_.curNL1InAiv) *
                                                  sizeof(float));
        uint32_t wsDstGap = static_cast<uint32_t>(
            (paddedN * sizeof(float) - CeilAlign(aivParams_.curNL1InAiv * sizeof(float), DATABLOCK_SIZE)) /
            DATABLOCK_SIZE);
        uint64_t wsBaseGM = copyGm2UbParams_.offsetWorkspaceGM;
        for (uint64_t k = 0; k < copyGm2UbParams_.kCnt; ++k) {
            uint64_t gmOff = wsBaseGM + k * BLOCK_BASE_M * BLOCK_BASE_N;
            DataCopyExtParams wsParams{static_cast<uint16_t>(copyGm2UbParams_.mBurst), wsRowBytes, wsSrcGap, wsDstGap,
                                       0};
            DataCopyPad<float>(ubAddTensor[k * tilePaddedSize], workspaceGlobal_[gmOff], wsParams, {false, 0, 0, 0});
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(FLAG_0);

        for (uint64_t i = 1; i < copyGm2UbParams_.kCnt; ++i) {
            Add(ubAddTensor, ubAddTensor, ubAddTensor[i * tilePaddedSize], tilePaddedSize);
        }

        if constexpr (sizeof(OutType) == sizeof(float)) {
            RunFusionFloat(ubAddTensor, paddedN, tilePaddedSize);
        } else if constexpr (INNER_PRECISE == 1) {
            RunFusionNormalPrecision(ubAddTensor, paddedN, tilePaddedSize);
        } else {
            RunFusionHighPrecision(ubAddTensor, paddedN, tilePaddedSize);
        }
    }

    __aicore__ inline void RunFusionHighPrecision(LocalTensor<float>& ubAddTensor, uint64_t paddedN,
                                                  uint64_t tilePaddedSize)
    {
        uint64_t castBufOffset = copyGm2UbParams_.kCnt * tilePaddedSize;
        AscendC::LocalTensor<OutType> ubX3Half =
            ubAddTensor[castBufOffset].template ReinterpretCast<OutType>();
        LoadX3Padded(ubX3Half, paddedN);

        AscendC::LocalTensor<float> ubX3Float = ubAddTensor[tilePaddedSize];
        AscendC::PipeBarrier<PIPE_V>();
        Cast(ubX3Float, ubX3Half, AscendC::RoundMode::CAST_NONE, tilePaddedSize);
        AscendC::PipeBarrier<PIPE_V>();

        if constexpr (DispatchPolicy::enableAdd) {
            AscendC::Add(ubAddTensor, ubAddTensor, ubX3Float, tilePaddedSize);
        } else if constexpr (DispatchPolicy::enableMul) {
            AscendC::Mul(ubAddTensor, ubAddTensor, ubX3Float, tilePaddedSize);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::LocalTensor<OutType> ubResult =
            ubAddTensor[tilePaddedSize].template ReinterpretCast<OutType>();
        Cast(ubResult, ubAddTensor, AscendC::RoundMode::CAST_RINT, tilePaddedSize);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FLAG_0);
        WriteResultToGm(ubResult, paddedN);
    }

    __aicore__ inline void RunFusionNormalPrecision(LocalTensor<float>& ubAddTensor, uint64_t paddedN,
                                                    uint64_t tilePaddedSize)
    {
        uint64_t castBufOffset = copyGm2UbParams_.kCnt * tilePaddedSize;
        AscendC::LocalTensor<OutType> ubX3Half =
            ubAddTensor[castBufOffset].template ReinterpretCast<OutType>();
        LoadX3Padded(ubX3Half, paddedN);

        AscendC::LocalTensor<OutType> ubResultHalf =
            ubAddTensor[castBufOffset + tilePaddedSize].template ReinterpretCast<OutType>();
        AscendC::PipeBarrier<PIPE_V>();
        Cast(ubResultHalf, ubAddTensor, AscendC::RoundMode::CAST_RINT, tilePaddedSize);
        AscendC::PipeBarrier<PIPE_V>();

        if constexpr (DispatchPolicy::enableAdd) {
            AscendC::Add(ubResultHalf, ubResultHalf, ubX3Half, tilePaddedSize);
        } else if constexpr (DispatchPolicy::enableMul) {
            AscendC::Mul(ubResultHalf, ubResultHalf, ubX3Half, tilePaddedSize);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FLAG_0);
        WriteResultToGm(ubResultHalf, paddedN);
    }

    __aicore__ inline void RunFusionFloat(LocalTensor<float>& ubAddTensor, uint64_t paddedN, uint64_t tilePaddedSize)
    {
        uint64_t castBufOffset = copyGm2UbParams_.kCnt * tilePaddedSize;
        AscendC::LocalTensor<OutType> ubX3 = ubAddTensor[castBufOffset].template ReinterpretCast<OutType>();
        LoadX3Padded(ubX3, paddedN);
        if constexpr (DispatchPolicy::enableAdd) {
            AscendC::Add(ubAddTensor, ubAddTensor, ubX3, tilePaddedSize);
        } else if constexpr (DispatchPolicy::enableMul) {
            AscendC::Mul(ubAddTensor, ubAddTensor, ubX3, tilePaddedSize);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FLAG_0);
        WriteResultToGm(ubAddTensor, paddedN);
    }

    __aicore__ inline void LoadX3Padded(AscendC::LocalTensor<OutType>& ubX3, uint64_t paddedN)
    {
        uint32_t x3RowBytes = static_cast<uint32_t>(aivParams_.curNL1InAiv * sizeof(OutType));
        uint32_t x3SrcGap = static_cast<uint32_t>((n_ - aivParams_.curNL1InAiv) * sizeof(OutType));
        uint32_t x3DstGap = static_cast<uint32_t>(
            (paddedN * sizeof(OutType) - CeilAlign(aivParams_.curNL1InAiv * sizeof(OutType),
            DATABLOCK_SIZE)) / DATABLOCK_SIZE);
        DataCopyExtParams x3Params{static_cast<uint16_t>(copyGm2UbParams_.mBurst),
            x3RowBytes, x3SrcGap, x3DstGap, 0};
        uint64_t x3Offset = copyUb2GmParams_.offsetCGm;
        if (x3BatchBroadcast_ && x3M_ > 0) {
            x3Offset = x3Offset % (x3M_ * n_);
        }
        AscendC::DataCopyPad(ubX3, x3Global_[x3Offset],
                             x3Params, {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(FLAG_0);
    }

    template <typename SrcTensor, typename X3Tensor>
    __aicore__ inline void FuseFlat(SrcTensor& mmTensor, X3Tensor& ubX3, uint64_t elemCount)
    {
        if constexpr (DispatchPolicy::enableAdd) {
            AscendC::Add(mmTensor, mmTensor, ubX3, elemCount);
        } else if constexpr (DispatchPolicy::enableMul) {
            AscendC::Mul(mmTensor, mmTensor, ubX3, elemCount);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(FLAG_0);
    }

    template <typename SrcTensor>
    __aicore__ inline void WriteResultToGm(SrcTensor& srcTensor, uint64_t paddedN)
    {
        uint32_t outRowBytes = static_cast<uint32_t>(aivParams_.curNL1InAiv * sizeof(OutType));
        uint32_t outSrcGap = static_cast<uint32_t>(
            (paddedN * sizeof(OutType) - CeilAlign(aivParams_.curNL1InAiv * sizeof(OutType), DATABLOCK_SIZE)) /
            DATABLOCK_SIZE);
        uint32_t outDstGap = static_cast<uint32_t>((n_ - aivParams_.curNL1InAiv) * sizeof(OutType));
        DataCopyExtParams outParams{static_cast<uint16_t>(copyUb2GmParams_.mLength), outRowBytes, outSrcGap, outDstGap,
                                    0};
        DataCopyPad<OutType>(cGlobal_[copyUb2GmParams_.offsetCGm], srcTensor, outParams);
    }

    __aicore__ inline void UpdateAivBasicBlock()
    {
        if (round_ < NUM_TWO) {
            aivParams_.curNL1InAiv = aivParams_.nCntIndex != (nCnt_ - 1) ? nL1_ : (n_ - (nCnt_ - 1) * nL1_);
            aivParams_.curML1InAiv = aivParams_.mCntIndex != (mCnt_ - 1) ? mL1_ : (m_ - (mCnt_ - 1) * mL1_);

            if constexpr (L0C2OutModel == MatMulL0C2Out::ND_FIXPIPE_1_2) {
                aivParams_.curAlignedNInAiv = CeilAlign(aivParams_.curNL1InAiv, AscendC::ONE_BLK_SIZE);
            } else {
                aivParams_.curAlignedNInAiv = aivParams_.curNL1InAiv;
            }
        }
    }

    __aicore__ inline void UpdateAivBasicIndex()
    {
        uint64_t newBlockIdx = AscendC::GetBlockIdx() / (AscendC::GetTaskRation() * kCnt_);
        aivParams_.kCntIndex = AscendC::GetBlockIdx() % (AscendC::GetTaskRation() * kCnt_);
        aivParams_.indexParams = newBlockIdx;
        aivParams_.bCntIndex = newBlockIdx / (mCnt_ * nCnt_);
        uint64_t mnIndex = (aivParams_.indexParams + (bCnt_ * mCnt_ * nCnt_ - bCnt_ * mCnt_ * nCnt_ % usedCoreNum_)) %
                           (mCnt_ * nCnt_);
        uint64_t mainWindow = AscendC::Std::min(MAIN_WINDOW, mCnt_);
        uint64_t mainRow = mainWindow == 0 ? 0 : mCnt_ / mainWindow - 1UL;
        uint64_t rowIdx = mainWindow == 0 ? 0 : mnIndex / nCnt_ / mainWindow;
        uint64_t tailWindow = mCnt_ - mainRow * mainWindow;
        if (rowIdx < mainRow) {
            aivParams_.nCntIndex = mainWindow == 0 ? 0 : (mnIndex / mainWindow) % nCnt_;
            aivParams_.mCntIndex = mainWindow == 0 ? 0 : rowIdx * mainWindow + mnIndex % mainWindow;
        } else {
            rowIdx = mainRow;
            uint64_t tailIndex = mnIndex - mainRow * mainWindow * nCnt_;
            aivParams_.nCntIndex = (tailIndex / tailWindow) % nCnt_;
            aivParams_.mCntIndex = mainRow * mainWindow + tailIndex % tailWindow;
        }
        if (rowIdx % NUM_TWO != 0UL) {
            aivParams_.nCntIndex = nCnt_ - aivParams_.nCntIndex - 1UL;
        }
    }

    __aicore__ inline void UpdateAivParams(uint64_t index)
    {
        mBurstBase_ = CeilAlign(CeilDiv(aivParams_.curML1InAiv, kCnt_ * AscendC::GetTaskRation()),
                                CeilDiv(UB2GM_SRCGAP_UNIT, aivParams_.curAlignedNInAiv));
        uint64_t mBurstCnt = CeilDiv(aivParams_.curML1InAiv, mBurstBase_);
        uint64_t mBurstTail = aivParams_.curML1InAiv - (mBurstCnt - 1) * mBurstBase_;

        if (aivParams_.kCntIndex < mBurstCnt) {
            copyGm2UbParams_.mBurstOri = (aivParams_.kCntIndex == mBurstCnt - 1) ? mBurstTail : mBurstBase_;
        } else {
            copyGm2UbParams_.mBurstOri = 0;
        }

        copyGm2UbParams_.kCnt = kCnt_;
        copyGm2UbParams_.mBurst = CeilDiv(copyGm2UbParams_.mBurstOri, aivMte2Num_);
        // moving out to GM.
        copyUb2GmParams_.offsetCGm = aivParams_.nCntIndex * nL1_ + aivParams_.mCntIndex * mL1_ * n_ +
                                     (aivParams_.kCntIndex * mBurstBase_ + copyGm2UbParams_.mBurst * index) * n_ +
                                     aivParams_.bCntIndex * m_ * n_;
        // moving into UB.
        copyGm2UbParams_.offsetWorkspaceGM = aivParams_.indexParams * kCnt_ * BLOCK_BASE_M * BLOCK_BASE_N +
                                             (aivParams_.kCntIndex * mBurstBase_ + copyGm2UbParams_.mBurst * index) *
                                                 aivParams_.curAlignedNInAiv;
        uint64_t singleCnt = 1;
        if (index >= singleCnt) {
            copyGm2UbParams_.mBurst = 0;
        } else if (index == singleCnt - 1) {
            copyGm2UbParams_.mBurst = copyGm2UbParams_.mBurstOri - (singleCnt - 1) * copyGm2UbParams_.mBurst;
        }
        copyUb2GmParams_.mLength = copyGm2UbParams_.mBurst;
    }

    __aicore__ inline void operator()() { Run(); }

private:
    constexpr static uint64_t NUM_TWO = 2UL;
    constexpr static uint64_t BLOCK_BASE_N = 256UL;
    constexpr static uint64_t BLOCK_BASE_M = 256UL;
    constexpr static uint64_t MAIN_WINDOW = 4UL;
    constexpr static uint64_t UB2GM_SRCGAP_UNIT = 32UL;
    constexpr static uint64_t DATABLOCK_SIZE = 32UL;
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
