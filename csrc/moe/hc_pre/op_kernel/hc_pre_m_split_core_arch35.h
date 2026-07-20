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

#ifndef HC_PRE_M_SPLIT_CORE_H
#define HC_PRE_M_SPLIT_CORE_H

#include "kernel_operator.h"
#include "hc_pre_base_arch35.h"
#include "hc_pre_cube_compute_arch35.h"

namespace HcPreNs {
using namespace AscendC;

template <typename T>
class HcPreMSplitCorePart1 {
public:
    __aicore__ inline HcPreMSplitCorePart1()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR hcFn, GM_ADDR hcScale, GM_ADDR hcBase,
        GM_ADDR y, GM_ADDR post, GM_ADDR combFrag, const HcPreTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        xGm.SetGlobalBuffer((__gm__ T*)x);
        hcFnGm.SetGlobalBuffer((__gm__ float*)hcFn);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        hcScaleGm.SetGlobalBuffer((__gm__ float*)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float*)hcBase);
        postGm.SetGlobalBuffer((__gm__ float*)post);
        combFragGm.SetGlobalBuffer((__gm__ float*)combFrag);

        TBuf<TPosition::A1> l1Buffer;
        pipe->InitBuffer(l1Buffer, L1_ALLOC_SIZE);
        xL1_ = l1Buffer.Get<float>();
        wL1_ = l1Buffer.Get<float>()[L1_BUF_NUM * L1_BUF_OFFSET];

        pipe->InitBufPool(tbufPool0, tilingData->bufferPool0Size);
        tbufPool0.InitBuffer(mmXBuf, CeilDiv(tilingData->mL1Size, 2) * RoundUp<float>(tilingData->hcMix) * sizeof(float));
        mmXLocal = mmXBuf.Get<float>();

        if ASCEND_IS_AIC {
            mmService_.Init();
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
        } else {
            tbufPool0.InitBuffer(rmsNormBuf, RoundUp<float>(CeilDiv(tilingData->mL1Size, 2)) * sizeof(float));
            tbufPool0.InitBufPool(tbufPool1, tilingData->bufferPool1Size);

            tbufPool0.InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
            tbufPool0.InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
            tbufPool0.InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));

            hcBase0Local = hcBaseBuf0.Get<float>();
            hcBase1Local = hcBaseBuf1.Get<float>();
            hcBase2Local = hcBaseBuf2.Get<float>();
        }
    }

    __aicore__ inline void Process()
    {
        int64_t curBlockIdx = GetBlockIdx();
        int64_t logicalBlockIdx = curBlockIdx;
        if ASCEND_IS_AIV {
            logicalBlockIdx = curBlockIdx / 2;
        }
        if (logicalBlockIdx >= tilingData->cubeBlockDimM) {
            return;
        }

        if ASCEND_IS_AIV {
          CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
          CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
          CopyIn(hcBaseGm[tilingData->hcMult * 2], hcBase2Local, 1, tilingData->hcMult * tilingData->hcMult);
          event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
          SetFlag<HardEvent::MTE2_V>(eventId);
          WaitFlag<HardEvent::MTE2_V>(eventId);
        }

        int64_t totalBlockNum = GetBlockNum();

        uint64_t mBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimM;
        uint64_t kBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimK;

        // todo 移到tiling计算
        uint64_t mCnt = CeilDiv(tilingData->bs, tilingData->mL1Size);
        uint64_t singleCoreMaxRound = CeilDiv(mCnt, tilingData->cubeBlockDimM);
        uint64_t mainCoreCount = mCnt % tilingData->cubeBlockDimM;
        uint64_t singleCoreRound = (mainCoreCount == 0 || logicalBlockIdx < mainCoreCount) ? singleCoreMaxRound : singleCoreMaxRound - 1;
        uint64_t mGmOffset = 0;
        if ASCEND_IS_AIC {
            if (mainCoreCount == 0 || curBlockIdx <= mainCoreCount) {
                mGmOffset = curBlockIdx * singleCoreMaxRound * tilingData->mL1Size;
            } else {
                mGmOffset = (mainCoreCount * singleCoreMaxRound + (curBlockIdx - mainCoreCount) * (singleCoreMaxRound - 1)) * tilingData->mL1Size;
            }
        } else {
            if (mainCoreCount == 0 || (curBlockIdx / 2) <= mainCoreCount) {
                mGmOffset = curBlockIdx / 2 * singleCoreMaxRound * tilingData->mL1Size;
            } else {
                mGmOffset = (mainCoreCount * singleCoreMaxRound + (curBlockIdx / 2 - mainCoreCount) * (singleCoreMaxRound - 1)) * tilingData->mL1Size;
            }
        }
        int64_t xGmBaseOffset = 0;
        int64_t yGmBaseOffset = 0;
        int64_t postGmBaseOffset = 0;
        int64_t combFragGmBaseOffset = 0;
        if ASCEND_IS_AIV {
            xGmBaseOffset = mGmOffset * tilingData->hcMult * tilingData->d;
            yGmBaseOffset = mGmOffset * tilingData->d;
            postGmBaseOffset = mGmOffset * tilingData->hcMult;
            combFragGmBaseOffset = mGmOffset * tilingData->hcMult * tilingData->hcMult;
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
        }

        uint64_t cvLoopKSize = tilingData->kL1Size / tilingData->kUbSize;
        int64_t xSplitOffset = 0;
        int64_t ySplitOffset = 0;
        int64_t postSplitOffset = 0;
        int64_t combFragSplitOffset = 0;
        int64_t xOutSplitOffset = 0;
        // m轴切分 按照0 0 1 1..分核
        for (uint64_t roundIdx = 0; roundIdx < singleCoreRound; mGmOffset += tilingData->mL1Size, ++roundIdx)
        {
            uint64_t mL1RealSize = AscendC::Std::min(tilingData->bs - mGmOffset, (uint64_t)tilingData->mL1Size);
            uint64_t kGmStartOffset = 0;
            uint64_t kGmEndOffset = tilingData->multCoreSplitKSize;
            uint64_t nd2NzBufSize = CeilAlign(tilingData->mUbSize, C0_SIZE) * RoundUp<float>(tilingData->kUbSize);
            if ASCEND_IS_AIV {
                tbufPool1.Reset();
                tbufPool1.InitBuffer(xQue, 2, tilingData->mUbSize * RoundUp<T>(tilingData->kUbSize) * sizeof(T));
                tbufPool1.InitBuffer(castBuf, tilingData->mUbSize * (RoundUp<float>(tilingData->kUbSize) * sizeof(float) + BLOCK_SIZE));
                tbufPool1.InitBuffer(nd2NzBuf, nd2NzBufSize * sizeof(float) * DOUBLE_BUFFER);

                xCastLocal = castBuf.Get<float>();
                xNd2NzLocal = nd2NzBuf.Get<float>();
                rmsNormLocal = rmsNormBuf.Get<float>();
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
                if (GetBlockIdx() % 2 != 0) {
                    xSplitOffset = CeilDiv(mL1RealSize, 2) * tilingData->hcMult * tilingData->d;
                    xOutSplitOffset = CeilDiv(mL1RealSize, 2) * tilingData->hcMult * tilingData->d;
                    ySplitOffset = CeilDiv(mL1RealSize, 2) * tilingData->d;
                    postSplitOffset = CeilDiv(mL1RealSize, 2) * tilingData->hcMult;
                    combFragSplitOffset = CeilDiv(mL1RealSize, 2) * tilingData->hcMult * tilingData->hcMult;
                }
            }
            // k轴切分（kCoreDim=1）
            int64_t bufferIdx = 0;
            if ASCEND_IS_AIV {
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(1));
            }
            for (int64_t kGmOffset = kGmStartOffset; kGmOffset < kGmEndOffset; kGmOffset += tilingData->kL1Size) {
                if ASCEND_IS_AIC {
                    bool isFirstKL1 = kGmOffset == kGmStartOffset;
                    bool isLastKL1 = (kGmOffset + tilingData->kL1Size) >= kGmEndOffset;
                    uint64_t kL1RealSize = AscendC::Std::min(kGmEndOffset - kGmOffset, (uint64_t)tilingData->kL1Size);
                    mmService_.CopyInB1Nd2Nz(tilingData->multCoreSplitKSize, kL1RealSize,
                                             tilingData->hcMix, hcFnGm[kGmOffset],
                                             wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
                    uint64_t mL1AlignSize = Align(mL1RealSize, AscendC::BLOCK_CUBE);
                    uint64_t nL1AlignSize = Align((uint64_t)tilingData->hcMix, AscendC::BLOCK_CUBE);
                    mmService_.Process(tilingData->bs, tilingData->hcMix, mL1RealSize, (256 / AscendC::Std::max(mL1AlignSize, nL1AlignSize)) * 32,
                                       isFirstKL1, isLastKL1, xL1_[aL1BufferID_ * L1_BUF_OFFSET], wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                    if (isLastKL1) {
                        mmService_.CopyOut(mmXLocal);
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(SYNC_AIC_AIV_PRE_POST_FLAG);
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(SYNC_AIC_AIV_PRE_POST_FLAG + FLAG_ID_MAX);
                    }
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG); // 写出ub搬出，cv流水同步比较复杂，暂不讨论
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
                } else {
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                    // Even cores take the first half with CeilDiv, and odd cores take the second half.
                    // This must match the Sinkhorn/output row split; otherwise odd mL1RealSize shifts the
                    // second half by one row.
                    int64_t rowFactor = CeilDiv(mL1RealSize, 2);
                    int64_t tailRowFactor = mL1RealSize - rowFactor;
                    int64_t curRowFactor = rowFactor;
                    int64_t mL1SizeAlign = CeilAlign(mL1RealSize, AscendC::BLOCK_CUBE);
                    if (curBlockIdx % 2 == 1) {
                        curRowFactor = tailRowFactor;
                    }
                    float coeff = 1 / static_cast<float>(tilingData->hcMult * tilingData->d);
                    for (int64_t cvLoopIdx = 0; cvLoopIdx < cvLoopKSize; cvLoopIdx++) {
                        uint64_t kRealSize = kGmOffset + tilingData->kUbSize >= kGmEndOffset ? kGmEndOffset - kGmOffset : tilingData->kUbSize;

                        xLocal = xQue.template AllocTensor<T>();
                        CopyIn(xGm[xGmBaseOffset + xSplitOffset + roundIdx * tilingData->mL1Size * tilingData->hcMult * tilingData->d + kGmOffset + cvLoopIdx * tilingData->kUbSize],
                               xLocal, curRowFactor, tilingData->kUbSize, tilingData->hcMult * tilingData->d - tilingData->kUbSize);
                        xQue.template EnQue(xLocal);
                        xLocal = xQue.template DeQue<T>();
                        if (kGmOffset == kGmStartOffset && cvLoopIdx == 0) {
                            VFProcessCastAndInvRmsPart1<T, false>(rmsNormLocal, xCastLocal, xLocal, coeff, curRowFactor, tilingData->kUbSize);
                        } else {
                            VFProcessCastAndInvRmsPart1<T, true>(rmsNormLocal, xCastLocal, xLocal, coeff, curRowFactor, tilingData->kUbSize);
                        }
                        xQue.template FreeTensor(xLocal);

                        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                        VFTransND2NZ(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)], xCastLocal, curRowFactor, tilingData->kUbSize);
                        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));
                        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));

                        if (curBlockIdx % 2 == 0) {
                            DataCopyParams dataCopyXParams;
                            dataCopyXParams.blockCount = CeilDiv(tilingData->kUbSize, C0_SIZE);
                            dataCopyXParams.blockLen = curRowFactor * C0_SIZE * sizeof(float) / BLOCK_SIZE;
                            dataCopyXParams.srcStride = CeilAlign(curRowFactor, C0_SIZE) - curRowFactor;
                            dataCopyXParams.dstStride = CeilAlign(mL1RealSize, 16) - curRowFactor;
                            CopyToL1(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)], xL1_[(aL1BufferID_ * L1_BUF_OFFSET) + cvLoopIdx * tilingData->kUbSize * mL1SizeAlign], dataCopyXParams);
                        } else {
                            DataCopyParams dataCopyXParams;
                            dataCopyXParams.blockCount = CeilDiv(tilingData->kUbSize, C0_SIZE);
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
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(1));
                CrossCoreWaitFlag<SYNC_MODE4, PIPE_V>(SYNC_AIC_AIV_PRE_POST_FLAG);
                // mm计算结果存入mmXLocal，mmXLocal每轮循环需要累加;
                tbufPool1.Reset();
                tbufPool1.InitBuffer(xQue, 2, tilingData->rowInnerFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor) * sizeof(T));
                tbufPool1.InitBuffer(
                    yQue, 2, tilingData->rowInnerFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
                tbufPool1.InitBuffer(postQue, 2, tilingData->rowInnerFactor * tilingData->hcMultAlign * sizeof(float));
                tbufPool1.InitBuffer(combFragQue, DOUBLE_BUFFER,
                    tilingData->rowInnerFactor * tilingData->hcMult * tilingData->hcMult * sizeof(float));

                // TBuf
                tbufPool1.InitBuffer(mixesBuf, tilingData->rowInnerFactor * RoundUp<float>(tilingData->hcMix) * sizeof(float));

                mixesLocal = mixesBuf.Get<float>();

                SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);

                // m内层循环
                int64_t currentRow = mL1RealSize / 2;
                if (mL1RealSize % 2 == 1 && curBlockIdx % 2 == 0) {
                    // m不整除时偶数核多处理一行
                    currentRow += 1;
                }
                for (int64_t innerRowIdx = 0; innerRowIdx < currentRow; innerRowIdx += tilingData->rowInnerFactor) {
                    int64_t currentInnerRowFactor = innerRowIdx + tilingData->rowInnerFactor >= currentRow ? currentRow - innerRowIdx :
                                                    tilingData->rowInnerFactor;
                    VFProcessInvRmsPart3(mixesLocal, mmXLocal[innerRowIdx * tilingData->hcMix], rmsNormLocal[innerRowIdx],
                                         tilingData->normEps, currentInnerRowFactor, tilingData->hcMix);

                    VFProcessPre(
                        mixesLocal, mixesLocal, hcBase0Local, hcScaleGm.GetValue(0), tilingData->hcEps,
                        currentInnerRowFactor, tilingData->hcMult, tilingData->hcMix);
                    for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++)
                    {
                        int64_t curDFactor =
                            (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                        xLocal = xQue.template AllocTensor<T>();
                        CopyIn(
                            xGm[xGmBaseOffset + xOutSplitOffset + roundIdx * tilingData->mL1Size * tilingData->hcMult * tilingData->d +
                                innerRowIdx * tilingData->hcMult * tilingData->d + dLoopIdx * tilingData->dFactor],
                            xLocal, currentInnerRowFactor * tilingData->hcMult, curDFactor, tilingData->d - curDFactor);
                        xQue.template EnQue(xLocal);
                        xLocal = xQue.template DeQue<T>();

                        yLocal = yQue.template AllocTensor<T>();
                        VFProcessY(yLocal, mixesLocal, xLocal, currentInnerRowFactor, tilingData->hcMult, curDFactor, tilingData->hcMix);
                        xQue.template FreeTensor(xLocal);
                        yQue.template EnQue(yLocal);
                        yLocal = yQue.template DeQue<T>();
                        CopyOut(yLocal, yGm[yGmBaseOffset + ySplitOffset + roundIdx * tilingData->mL1Size * tilingData->d + innerRowIdx * tilingData->d + dLoopIdx * tilingData->dFactor],
                                currentInnerRowFactor, curDFactor, tilingData->d - curDFactor);
                        yQue.template FreeTensor(yLocal);
                    }

                    // post
                    postLocal = postQue.AllocTensor<float>();
                    VFProcessPost(
                        postLocal, mixesLocal[tilingData->hcMult], hcBase1Local,
                        hcScaleGm.GetValue(1), tilingData->hcEps, currentInnerRowFactor, tilingData->hcMult, tilingData->hcMix);

                    postQue.EnQue(postLocal);
                    postLocal = postQue.DeQue<float>();
                    CopyOut(postLocal, postGm[postGmBaseOffset + postSplitOffset + roundIdx * tilingData->mL1Size * tilingData->hcMult + innerRowIdx * tilingData->hcMult], currentInnerRowFactor, tilingData->hcMult);
                    postQue.FreeTensor(postLocal);

                    // combFrag
                    combFragLocal = combFragQue.AllocTensor<float>();
                    VFProcessCombFragPacked(
                        combFragLocal, mixesLocal[tilingData->hcMult * 2], hcBase2Local, hcScaleGm.GetValue(2), tilingData->hcEps,
                        tilingData->iterTimes - 1, currentInnerRowFactor, tilingData->hcMult, tilingData->hcMix);

                    combFragQue.EnQue(combFragLocal);
                    combFragLocal = combFragQue.DeQue<float>();
                    CopyOut(combFragLocal, combFragGm[combFragGmBaseOffset + combFragSplitOffset + roundIdx * tilingData->mL1Size * tilingData->hcMult * tilingData->hcMult + innerRowIdx * tilingData->hcMult * tilingData->hcMult],
                            currentInnerRowFactor, tilingData->hcMult * tilingData->hcMult);
                    combFragQue.FreeTensor(combFragLocal);
                }
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            }
        }
        if ASCEND_IS_AIV {
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
        } else {
            mmService_.End();
        }
    }

private:
    TPipe *pipe;
    const HcPreTilingData *tilingData;
    // (M, K) * (N, K)

    GlobalTensor<T> xGm;
    GlobalTensor<float> hcFnGm;
    GlobalTensor<float> workspaceGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> invRmsGm;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECOUT, 1> combFragQue;

    TBuf<QuePosition::VECCALC> castBuf;
    TBuf<QuePosition::VECCALC> nd2NzBuf;

    TQue<QuePosition::VECIN, 1> squareSumQue;

    TBuf<QuePosition::VECCALC> hcBaseBuf0;
    TBuf<QuePosition::VECCALC> hcBaseBuf1;
    TBuf<QuePosition::VECCALC> hcBaseBuf2;

    TBuf<QuePosition::VECCALC> rowBrcbBuf0;
    TBuf<QuePosition::VECCALC> hcBrcbBuf1;
    TBuf<QuePosition::VECCALC> reduceBuf;

    TBuf<QuePosition::VECCALC> rsqrtBuf;
    TBuf<QuePosition::VECCALC> squareReduceBuf;
    TBuf<QuePosition::VECCALC> mixes01ReduceBuf;

    TBuf<QuePosition::VECCALC> xCastBuf;
    TBuf<QuePosition::VECCALC> yCastBuf;

    TBuf<QuePosition::VECCALC> mixesBuf;
    TBuf<QuePosition::VECCALC> rmsNormBuf;
    TBuf<QuePosition::VECCALC> mmXBuf;

    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> mmXLocal;
    LocalTensor<float> rmsNormLocal;
    LocalTensor<float> xCastLocal;
    LocalTensor<float> xNd2NzLocal;

    LocalTensor<float> mixesLocal;
    LocalTensor<float> rmsAndmmLocal;
    LocalTensor<float> postLocal;
    LocalTensor<float> combFragLocal;
    LocalTensor<float> hcBase0Local;
    LocalTensor<float> hcBase1Local;
    LocalTensor<float> hcBase2Local;

    HcPreCubeCompute mmService_;
    LocalTensor<float> xL1_;
    LocalTensor<float> wL1_;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
    static constexpr uint64_t SYNC_AIC_AIV_PRE_POST_FLAG = 10;
    static constexpr uint64_t FLAG_ID_MAX = 16;
    uint64_t cvLoopIdx_ = 0;
    uint8_t aL1BufferID_{0};

    TBufPool<QuePosition::VECCALC, 12> tbufPool0;
    TBufPool<QuePosition::VECCALC, 12> tbufPool1;
};

} // namespace HCPreSinkhorn

#endif
