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
 * \file hc_pre_m_k_split_core.h
 * \brief
 */

#ifndef HC_PRE_M_K_SPLIT_CORE_ARACH35_H
#define HC_PRE_M_K_SPLIT_CORE_ARACH35_H

#include "kernel_operator.h"
#include "hc_pre_base_arch35.h"
#include "hc_pre_cube_compute_arch35.h"

namespace HcPreNs {
using namespace AscendC;

template <typename T>
class HcPreMKSplitCorePart1 {
public:
    __aicore__ inline HcPreMKSplitCorePart1()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR hcFn, GM_ADDR workspace, const HcPreTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        xGm.SetGlobalBuffer((__gm__ T*)x);
        hcFnGm.SetGlobalBuffer((__gm__ float*)hcFn);
        mmGm.SetGlobalBuffer((__gm__ float*)workspace);
        rmsGm.SetGlobalBuffer((__gm__ float*)workspace + tilingData->kBlockFactor * tilingData->bs * tilingData->hcMix);

        TBuf<TPosition::A1> l1Buffer;
        pipe->InitBuffer(l1Buffer, L1_ALLOC_SIZE);
        xL1_ = l1Buffer.Get<float>();
        wL1_ = l1Buffer.Get<float>()[L1_BUF_NUM * L1_BUF_OFFSET];

        // InQue
        pipe->InitBuffer(xQue, 2, tilingData->mUbSize * RoundUp<T>(tilingData->kUbSize) * sizeof(T));

        // OutQue
        pipe->InitBuffer(rmsQue, 2, RoundUp<float>(tilingData->mUbSize) * sizeof(float));

        // Calc Buf
        pipe->InitBuffer(castBuf, tilingData->mUbSize * (RoundUp<float>(tilingData->kUbSize) * sizeof(float) + BLOCK_SIZE));
        pipe->InitBuffer(nd2NzBuf, CeilAlign(tilingData->mUbSize, C0_SIZE) * RoundUp<float>(tilingData->kUbSize) * sizeof(float) * DOUBLE_BUFFER);

        if ASCEND_IS_AIC {
            mmService_.Init();
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
        }
        xCastLocal = castBuf.Get<float>();
        xNd2NzLocal = nd2NzBuf.Get<float>();
    }

    __aicore__ inline void Process()
    {
        int64_t curBlockIdx = GetBlockIdx();
        int64_t totalBlockNum = GetBlockNum();

        uint64_t mBlkDimIdx = curBlockIdx / tilingData->cubeBlockDimK;
        uint64_t kBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimK;

        // todo 移到tiling计算
        uint64_t mCnt = CeilDiv(tilingData->bs, tilingData->mL1Size);
        uint64_t singleCoreMaxRound = CeilDiv(mCnt, tilingData->cubeBlockDimM);
        uint64_t mainCoreCount = mCnt % tilingData->cubeBlockDimM;
        uint64_t singleCoreRound = (mainCoreCount == 0 || curBlockIdx < mainCoreCount) ? singleCoreMaxRound : singleCoreMaxRound - 1;
        uint64_t mGmOffset = 0;
        uint64_t nd2NzBufSize = CeilAlign(tilingData->mUbSize, C0_SIZE) * RoundUp<float>(tilingData->kUbSize);
        if ASCEND_IS_AIC {
            mGmOffset = (curBlockIdx / tilingData->cubeBlockDimK) * singleCoreMaxRound * tilingData->mL1Size;
        } else {
            mGmOffset = ((curBlockIdx / 2) / tilingData->cubeBlockDimK) * singleCoreMaxRound * tilingData->mL1Size;
        }
        int64_t xGmBaseOffset = 0;
        int64_t rmsGmBaseOffset = 0;
        if ASCEND_IS_AIV {
            int64_t aivCurBlockIdx = GetBlockIdx();
            xGmBaseOffset = ((aivCurBlockIdx / 2) / tilingData->cubeBlockDimK) * singleCoreMaxRound * tilingData->mL1Size * tilingData->hcMult * tilingData->d +
                ((aivCurBlockIdx / 2) % tilingData->cubeBlockDimK) * tilingData->multCoreSplitKSize;
            rmsGmBaseOffset = ((aivCurBlockIdx / 2) / tilingData->cubeBlockDimK) * singleCoreMaxRound * tilingData->mL1Size;
        }

        // todo 移到tiling计算
        int64_t xSplitOffset = 0;
        int64_t rmsSplitOffset = 0;
        // m轴切分 按照0 0 1 1..分核
        int64_t bufferIdx = 0;
        int64_t curAicBlockIdx = 0;
        if ASCEND_IS_AIC {
            curAicBlockIdx = curBlockIdx;
        } else {
            curAicBlockIdx = curBlockIdx / 2;
        }

        if (curAicBlockIdx < tilingData->cubeBlockDimK * tilingData->cubeBlockDimM) {
            if ASCEND_IS_AIV {
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(1));
            }
            for (uint64_t roundIdx = 0; roundIdx < singleCoreRound; mGmOffset += tilingData->mL1Size, ++roundIdx)
            {
                uint64_t mL1RealSize = AscendC::Std::min(tilingData->bs - mGmOffset, (uint64_t)tilingData->mL1Size);
                uint64_t kGmStartOffset = 0;
                uint64_t kGmEndOffset = AscendC::Std::min(tilingData->multCoreSplitKSize,
                    tilingData->k -  (curAicBlockIdx % tilingData->cubeBlockDimK) * tilingData->multCoreSplitKSize);
                if ASCEND_IS_AIV {
                    if (GetBlockIdx() % 2 != 0) {
                        xSplitOffset = (mL1RealSize / 2) * tilingData->hcMult * tilingData->d;
                        rmsSplitOffset = mL1RealSize / 2;
                    }
                    rmsNormLocal = rmsQue.template AllocTensor<float>();
                }

                int64_t curRowFactor = 0;
                for (int64_t kGmOffset = kGmStartOffset; kGmOffset < kGmEndOffset; kGmOffset += tilingData->kL1Size) {
                    uint64_t kL1RealSize = AscendC::Std::min(kGmEndOffset - kGmOffset, (uint64_t)tilingData->kL1Size);
                    if ASCEND_IS_AIC {
                        bool isFirstKL1 = kGmOffset == kGmStartOffset;
                        bool isLastKL1 = (kGmOffset + tilingData->kL1Size) >= kGmEndOffset;
                        mmService_.CopyInB1Nd2Nz(tilingData->hcMult * tilingData->d, kL1RealSize,
                                                tilingData->hcMix, hcFnGm[kGmOffset + kBlkDimIdx * tilingData->multCoreSplitKSize],
                                                wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
                        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
                        uint64_t mL1AlignSize = Align(mL1RealSize, AscendC::BLOCK_CUBE);
                        uint64_t nL1AlignSize = Align((uint64_t)tilingData->hcMix, AscendC::BLOCK_CUBE);

                        mmService_.Process(tilingData->bs, tilingData->hcMix, mL1RealSize, (256 / AscendC::Std::max(mL1AlignSize, nL1AlignSize)) * 32,
                                        isFirstKL1, isLastKL1, xL1_[aL1BufferID_ * L1_BUF_OFFSET], wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                        if (isLastKL1) {
                            mmService_.CopyOut(mmGm[mBlkDimIdx * tilingData->mL1Size * singleCoreMaxRound * tilingData->hcMix + kBlkDimIdx * tilingData->bs * tilingData->hcMix + roundIdx * tilingData->mL1Size]);
                        }
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG); // 写出ub搬出，cv流水同步比较复杂，暂不讨论
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                        int64_t rowFactor = mL1RealSize / 2;
                        int64_t tailRowFactor = mL1RealSize - rowFactor;
                        curRowFactor = rowFactor;
                        int64_t mL1SizeAlign = CeilAlign(mL1RealSize, AscendC::BLOCK_CUBE);
                        if (curBlockIdx % 2 == 1) {
                            curRowFactor = tailRowFactor;
                        }
                        uint64_t cvLoopKSize = CeilDiv(kL1RealSize, tilingData->kUbSize);
                        uint64_t kReminderSize = kL1RealSize - (cvLoopKSize - 1) * tilingData->kUbSize;
                        float coeff = 1 / static_cast<float>(tilingData->hcMult * tilingData->d);
                        for (int64_t cvLoopIdx = 0; cvLoopIdx < cvLoopKSize; cvLoopIdx++) {
                            uint64_t kRealSize = (cvLoopIdx == cvLoopKSize - 1) ? kReminderSize : tilingData->kUbSize;
                            xLocal = xQue.template AllocTensor<T>();
                            CopyIn(xGm[xGmBaseOffset + xSplitOffset + roundIdx * tilingData->mL1Size * tilingData->hcMult * tilingData->d + kGmOffset + cvLoopIdx * tilingData->kUbSize],
                                xLocal, curRowFactor, kRealSize, tilingData->hcMult * tilingData->d - kRealSize);
                            xQue.template EnQue(xLocal);
                            xLocal = xQue.template DeQue<T>();
                            if (kGmOffset == kGmStartOffset && cvLoopIdx == 0) {
                                VFProcessCastAndInvRmsPart1<T, false>(rmsNormLocal, xCastLocal, xLocal, coeff, curRowFactor, kRealSize);
                            } else {
                                VFProcessCastAndInvRmsPart1<T, true>(rmsNormLocal, xCastLocal, xLocal, coeff, curRowFactor, kRealSize);
                            }
                            xQue.template FreeTensor(xLocal);

                            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                            VFTransND2NZ(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)], xCastLocal, curRowFactor, kRealSize);
                            SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));
                            WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));

                            if (curBlockIdx % 2 == 0) {
                                DataCopyParams dataCopyXParams;
                                dataCopyXParams.blockCount = CeilDiv(kRealSize, C0_SIZE);
                                dataCopyXParams.blockLen = curRowFactor * C0_SIZE * sizeof(float) / BLOCK_SIZE;
                                dataCopyXParams.srcStride = CeilAlign(curRowFactor, C0_SIZE) - curRowFactor;
                                dataCopyXParams.dstStride = CeilAlign(mL1RealSize, 16) - curRowFactor;
                                CopyToL1(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)], xL1_[(aL1BufferID_ * L1_BUF_OFFSET) + cvLoopIdx * tilingData->kUbSize * mL1SizeAlign], dataCopyXParams);
                            } else {
                                DataCopyParams dataCopyXParams;
                                dataCopyXParams.blockCount = CeilDiv(kRealSize, C0_SIZE);
                                dataCopyXParams.blockLen = curRowFactor * C0_SIZE * sizeof(float) / BLOCK_SIZE;
                                dataCopyXParams.srcStride = CeilAlign(curRowFactor, C0_SIZE) -  curRowFactor;
                                dataCopyXParams.dstStride = CeilAlign(mL1RealSize, 16) - curRowFactor;
                                CopyToL1(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)], xL1_[(aL1BufferID_ * L1_BUF_OFFSET) + rowFactor * (BLOCK_SIZE / sizeof(float)) + cvLoopIdx * tilingData->kUbSize * mL1SizeAlign], dataCopyXParams);
                            }
                            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                            bufferIdx++;
                        }
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
                    }
                    aL1BufferID_ ^= 1;
                }
                if ASCEND_IS_AIV {
                    int64_t kBaseOffset = (GetBlockIdx() / 2) % tilingData->kBlockFactor * tilingData->bs;
                    rmsQue.template EnQue(rmsNormLocal);
                    rmsNormLocal = rmsQue.template DeQue<float>();
                    CopyOut(rmsNormLocal, rmsGm[kBaseOffset + rmsGmBaseOffset + rmsSplitOffset + roundIdx * tilingData->mL1Size], 1, curRowFactor);
                    rmsQue.template FreeTensor(rmsNormLocal);
                }
            }
            if ASCEND_IS_AIV {
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(1));
            }
        }
        SyncAll<false>();
    }

private:
    TPipe* pipe;
    const HcPreTilingData* tilingData;
    // (M, K) * (N, K)

    GlobalTensor<T> xGm;
    GlobalTensor<float> hcFnGm;
    GlobalTensor<float> mmGm;
    GlobalTensor<float> rmsGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> rmsQue;
    TBuf<QuePosition::VECCALC> castBuf;
    TBuf<QuePosition::VECCALC> nd2NzBuf;

    LocalTensor<T> xLocal;
    LocalTensor<float> mmXLocal;
    LocalTensor<float> rmsNormLocal;
    LocalTensor<float> xCastLocal;
    LocalTensor<float> xNd2NzLocal;

    HcPreCubeCompute mmService_;
    LocalTensor<float> xL1_;
    LocalTensor<float> wL1_;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
    static constexpr uint64_t SYNC_AIV_AIC_PRE_POST_FLAG = 10;
    static constexpr uint64_t SYNC_AIC_AIV_PRE_POST_FLAG = 11;
    static constexpr uint64_t FLAG_ID_MAX = 16;
    uint64_t cvLoopIdx_ = 0;
    uint8_t aL1BufferID_ = 0;
};


template <typename T>
class HcPreMKSplitCorePart2 {
public:
    __aicore__ inline HcPreMKSplitCorePart2()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR y, GM_ADDR post,
        GM_ADDR combFrag, GM_ADDR workspace, const HcPreTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T*)x);
        hcScaleGm.SetGlobalBuffer((__gm__ float*)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float*)hcBase);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        postGm.SetGlobalBuffer((__gm__ float*)post);
        combFragGm.SetGlobalBuffer((__gm__ float*)combFrag);
        mmGm.SetGlobalBuffer((__gm__ float*)workspace);
        rmsGm.SetGlobalBuffer((__gm__ float*)workspace + tilingData->kBlockFactor * tilingData->bs * tilingData->hcMix);


        // InQue
        pipe->InitBuffer(
            xQue, 2, tilingData->stage2RowFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        int64_t rmsAndmmQueSize = tilingData->kBlockFactor * RoundUp<float>(tilingData->stage2RowFactor) * sizeof(float) +
                                  tilingData->kBlockFactor * tilingData->stage2RowFactor * RoundUp<float>(tilingData->hcMix) * sizeof(float);
        pipe->InitBuffer(rmsAndmmQue, 2, rmsAndmmQueSize);

        // OutQue
        pipe->InitBuffer(
            yQue, 2, tilingData->stage2RowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        pipe->InitBuffer(postQue, 2, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(combFragQue, DOUBLE_BUFFER,
            tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMult * sizeof(float));

        // TBuf
        pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(mixesBuf, tilingData->stage2RowFactor * RoundUp<float>(tilingData->hcMix) * sizeof(float));

        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        hcBase2Local = hcBaseBuf2.Get<float>();
        mixesLocal = mixesBuf.Get<float>();
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
            int64_t curBlockIdx = GetBlockIdx();
            int64_t stage2UsedCoreNum = tilingData->secondUsedCoreNum;
            if (curBlockIdx >= stage2UsedCoreNum) {
                return;
            }
            int64_t rowOuterLoop =
                (curBlockIdx == stage2UsedCoreNum - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
            int64_t tailRowFactor = (curBlockIdx == stage2UsedCoreNum - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                        tilingData->tailRowFactorOfFormerBlock;

            CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult * 2], hcBase2Local, 1, tilingData->hcMult * tilingData->hcMult);
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
            int64_t mmGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMix;
            int64_t rmsGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock;
            int64_t xGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult * tilingData->d;
            int64_t mmLocalSize = stage1UsedCoreNum * tilingData->stage2RowFactor * RoundUp<float>(tilingData->hcMix);
            for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
                int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->stage2RowFactor;
                rmsAndmmLocal = rmsAndmmQue.AllocTensor<float>();
                CopyInWithLoopMode(
                    mmGm[mmGmBaseOffset + rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMix], rmsAndmmLocal, tilingData->kBlockFactor, curRowFactor, tilingData->hcMix, tilingData->bs * tilingData->hcMix);
                CopyIn(
                    rmsGm[rmsGmBaseOffset + rowOuterIdx * tilingData->stage2RowFactor],
                    rmsAndmmLocal[mmLocalSize], tilingData->kBlockFactor, curRowFactor, tilingData->bs - curRowFactor);

                rmsAndmmQue.EnQue(rmsAndmmLocal);
                rmsAndmmLocal = rmsAndmmQue.DeQue<float>();

                VFProcessInvRmsPart3WithGroupReduce(mixesLocal, rmsAndmmLocal, rmsAndmmLocal[mmLocalSize], tilingData->normEps, tilingData->kBlockFactor, curRowFactor, tilingData->hcMix);

                VFProcessPre(
                    mixesLocal, mixesLocal, hcBase0Local, hcScaleGm.GetValue(0), tilingData->hcEps,
                    curRowFactor, tilingData->hcMult, tilingData->hcMix);
                for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                    int64_t curDFactor =
                        (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                    xLocal = xQue.template AllocTensor<T>();
                    CopyIn(
                        xGm[xGmBaseOffset + rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMult * tilingData->d +
                            dLoopIdx * tilingData->dFactor],
                        xLocal, tilingData->stage2RowFactor * tilingData->hcMult, curDFactor, tilingData->d - curDFactor);
                    xQue.template EnQue(xLocal);
                    xLocal = xQue.template DeQue<T>();

                    yLocal = yQue.template AllocTensor<T>();
                    VFProcessY(yLocal, mixesLocal, xLocal, curRowFactor, tilingData->hcMult, curDFactor, tilingData->hcMix);
                    xQue.template FreeTensor(xLocal);
                    yQue.template EnQue(yLocal);
                    yLocal = yQue.template DeQue<T>();
                    CopyOut(yLocal, yGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->d + rowOuterIdx * tilingData->stage2RowFactor * tilingData->d + dLoopIdx * tilingData->dFactor], curRowFactor, curDFactor, tilingData->d - curDFactor);
                    yQue.template FreeTensor(yLocal);
                }

                // post
                postLocal = postQue.AllocTensor<float>();
                VFProcessPost(
                    postLocal, mixesLocal[tilingData->hcMult], hcBase1Local,
                    hcScaleGm.GetValue(1), tilingData->hcEps, curRowFactor, tilingData->hcMult, tilingData->hcMix);

                postQue.EnQue(postLocal);
                postLocal = postQue.DeQue<float>();
                CopyOut(postLocal, postGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult + rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMult], curRowFactor, tilingData->hcMult);
                postQue.FreeTensor(postLocal);

                // combFrag
                combFragLocal = combFragQue.AllocTensor<float>();
                VFProcessCombFragPacked(
                    combFragLocal, mixesLocal[tilingData->hcMult * 2], hcBase2Local, hcScaleGm.GetValue(2), tilingData->hcEps,
                    tilingData->iterTimes - 1, curRowFactor, tilingData->hcMult, tilingData->hcMix);
                rmsAndmmQue.FreeTensor(rmsAndmmLocal);

                combFragQue.EnQue(combFragLocal);
                combFragLocal = combFragQue.DeQue<float>();
                int64_t combLen = tilingData->hcMult * tilingData->hcMult;
                int64_t combOutOffset = (curBlockIdx * tilingData->rowOfFormerBlock +
                                         rowOuterIdx * tilingData->stage2RowFactor) * combLen;
                CopyOut(combFragLocal, combFragGm[combOutOffset], curRowFactor, combLen);
                combFragQue.FreeTensor(combFragLocal);
            }
        }
    }

private:
    TPipe* pipe;
    const HcPreTilingData* tilingData;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;

    GlobalTensor<float> mmGm;
    GlobalTensor<float> rmsGm;

    TQue<QuePosition::VECIN, 1> rmsAndmmQue;
    TQue<QuePosition::VECIN, 1> xQue;

    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECOUT, 1> combFragQue;

    TBuf<QuePosition::VECCALC> mixesBuf;
    TBuf<QuePosition::VECCALC> hcBaseBuf0;
    TBuf<QuePosition::VECCALC> hcBaseBuf1;
    TBuf<QuePosition::VECCALC> hcBaseBuf2;

    LocalTensor<float> mixesLocal;
    LocalTensor<float> rmsAndmmLocal;
    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> postLocal;
    LocalTensor<float> combFragLocal;
    LocalTensor<float> hcBase0Local;
    LocalTensor<float> hcBase1Local;
    LocalTensor<float> hcBase2Local;
};
} // namespace HCPreSinkhorn

#endif