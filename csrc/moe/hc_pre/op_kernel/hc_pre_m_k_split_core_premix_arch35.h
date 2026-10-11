/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HC_PRE_M_K_SPLIT_CORE_PREMIX_ARCH35_H
#define HC_PRE_M_K_SPLIT_CORE_PREMIX_ARCH35_H

#include "hc_pre_base_arch35.h"
#include "hc_pre_cube_compute_arch35.h"
#include "kernel_operator.h"

namespace HcPreNs {
using namespace AscendC;

// Premix MK tiling contract:
//   multCoreSplitKSize is the aggregate hcMult*d capacity of one K core and is
//   divisible by hcMult. Each core owns dPerCore=multCoreSplitKSize/hcMult
//   contiguous d columns and visits all hc segments for those columns.
//   cubeBlockDimK must equal CeilDiv(d, dPerCore). kUbSize is C0 aligned,
//   kL1Size contains at most two kUbSize subwindows, and mUbSize tiles each
//   AIV-owned half of mL1Size.
template <typename T> class HcPreMKSplitCorePremixPart1 {
public:
    __aicore__ inline HcPreMKSplitCorePremixPart1() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcFn, GM_ADDR preMix, GM_ADDR y, GM_ADDR workspace,
                                const HcPreTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        xGm.SetGlobalBuffer((__gm__ T *)x);
        hcFnGm.SetGlobalBuffer((__gm__ float *)hcFn);
        preMixGm.SetGlobalBuffer((__gm__ float *)preMix);
        yGm.SetGlobalBuffer((__gm__ T *)y);
        mmGm.SetGlobalBuffer((__gm__ float *)workspace);
        rmsGm.SetGlobalBuffer((__gm__ float *)workspace +
                              tilingData->kBlockFactor * tilingData->bs * tilingData->hcMix);

        TBuf<TPosition::A1> l1Buffer;
        pipe->InitBuffer(l1Buffer, L1_ALLOC_SIZE);
        xL1_ = l1Buffer.Get<float>();
        wL1_ = l1Buffer.Get<float>()[L1_BUF_NUM * L1_BUF_OFFSET];

        pipe->InitBuffer(xQue, DOUBLE_BUFFER, tilingData->mUbSize * RoundUp<T>(tilingData->kUbSize) * sizeof(T));
        uint64_t maxMRows = CeilDiv(tilingData->mL1Size, 2);
        uint64_t dPerCore = tilingData->multCoreSplitKSize / tilingData->hcMult;
        uint64_t maxGroupK =
            AscendC::Std::min((uint64_t)tilingData->kL1Size, AscendC::Std::min(dPerCore, (uint64_t)tilingData->d));
        pipe->InitBuffer(preMixQue, 1, maxMRows * RoundUp<float>(tilingData->hcMix) * sizeof(float));
        pipe->InitBuffer(rmsQue, DOUBLE_BUFFER, RoundUp<float>(maxMRows) * sizeof(float));
        pipe->InitBuffer(yQue, DOUBLE_BUFFER, tilingData->mUbSize * RoundUp<T>(tilingData->kUbSize) * sizeof(T));
        pipe->InitBuffer(castBuf,
                         tilingData->mUbSize * (RoundUp<float>(tilingData->kUbSize) * sizeof(float) + BLOCK_SIZE));
        pipe->InitBuffer(nd2NzBuf, CeilAlign(tilingData->mUbSize, C0_SIZE) * RoundUp<float>(tilingData->kUbSize) *
                                       sizeof(float) * DOUBLE_BUFFER);
        pipe->InitBuffer(yAccBuf, maxMRows * RoundUp<float>(tilingData->kUbSize) * sizeof(float) *
                                      CeilDiv(maxGroupK, (uint64_t)tilingData->kUbSize));

        xCastLocal = castBuf.Get<float>();
        xNd2NzLocal = nd2NzBuf.Get<float>();
        yAccLocal = yAccBuf.Get<float>();
        if ASCEND_IS_AIC {
            mmService_.Init();
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
        }
    }

    __aicore__ inline void Process()
    {
        int64_t curBlockIdx = GetBlockIdx();
        int64_t logicalBlockIdx = curBlockIdx;
        if ASCEND_IS_AIV {
            logicalBlockIdx = curBlockIdx / 2;
        }
        int64_t activeBlockNum = tilingData->cubeBlockDimM * tilingData->cubeBlockDimK;
        if (logicalBlockIdx >= activeBlockNum) {
            FinishStage1();
            SyncAll<false>();
            return;
        }

        uint64_t mBlockIdx = logicalBlockIdx / tilingData->cubeBlockDimK;
        uint64_t kBlockIdx = logicalBlockIdx % tilingData->cubeBlockDimK;
        uint64_t mCnt = CeilDiv(tilingData->bs, tilingData->mL1Size);
        uint64_t maxRound = CeilDiv(mCnt, tilingData->cubeBlockDimM);
        uint64_t longCoreCount = mCnt % tilingData->cubeBlockDimM;
        uint64_t coreRound = (longCoreCount == 0 || mBlockIdx < longCoreCount) ? maxRound : maxRound - 1;
        uint64_t mGmOffset;
        if (longCoreCount == 0 || mBlockIdx < longCoreCount) {
            mGmOffset = mBlockIdx * maxRound * tilingData->mL1Size;
        } else {
            mGmOffset = (longCoreCount * maxRound + (mBlockIdx - longCoreCount) * (maxRound - 1)) * tilingData->mL1Size;
        }

        uint64_t dPerCore = tilingData->multCoreSplitKSize / tilingData->hcMult;
        uint64_t dCoreStart = kBlockIdx * dPerCore;
        uint64_t dCoreSize = AscendC::Std::min(dPerCore, (uint64_t)tilingData->d - dCoreStart);
        uint64_t dGroupCount = CeilDiv(dCoreSize, tilingData->kL1Size);
        uint64_t nd2NzBufSize = CeilAlign(tilingData->mUbSize, C0_SIZE) * RoundUp<float>(tilingData->kUbSize);
        uint64_t yAccSliceSize = CeilDiv(tilingData->mL1Size, 2) * RoundUp<float>(tilingData->kUbSize);
        int64_t bufferIdx = 0;
        if ASCEND_IS_AIV {
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(1));
        }

        for (uint64_t roundIdx = 0; roundIdx < coreRound; ++roundIdx, mGmOffset += tilingData->mL1Size) {
            uint64_t mReal = AscendC::Std::min((uint64_t)tilingData->bs - mGmOffset, (uint64_t)tilingData->mL1Size);
            uint64_t firstHalfRows = CeilDiv(mReal, 2);
            uint64_t localRowOffset = 0;
            uint64_t localRows = firstHalfRows;
            if ASCEND_IS_AIV {
                if (curBlockIdx & 1) {
                    localRowOffset = firstHalfRows;
                    localRows = mReal - firstHalfRows;
                }
            }
            if ASCEND_IS_AIV {
                rmsNormLocal = rmsQue.AllocTensor<float>();
                preMixLocal = preMixQue.AllocTensor<float>();
                CopyInWithUbStride(preMixGm[(mGmOffset + localRowOffset) * tilingData->hcMult], preMixLocal, localRows,
                                   tilingData->hcMult, 0, UbRowGapBlocks(tilingData->hcMult, tilingData->hcMix));
                preMixQue.EnQue(preMixLocal);
                preMixLocal = preMixQue.DeQue<float>();
            }

            uint64_t mAlign = Align(mReal, AscendC::BLOCK_CUBE);
            for (uint64_t groupIdx = 0; groupIdx < dGroupCount; ++groupIdx) {
                uint64_t dGroupStart = dCoreStart + groupIdx * tilingData->kL1Size;
                uint64_t currentGroupK =
                    AscendC::Std::min((uint64_t)tilingData->kL1Size, dCoreStart + dCoreSize - dGroupStart);
                uint64_t subCount = CeilDiv(currentGroupK, tilingData->kUbSize);
                for (uint64_t hcIdx = 0; hcIdx < (uint64_t)tilingData->hcMult; ++hcIdx) {
                    uint64_t sourceGroupKOffset = hcIdx * tilingData->d + dGroupStart;
                    uint64_t groupHcIdx = groupIdx * tilingData->hcMult + hcIdx;
                    bool isFirstK = groupHcIdx == 0;
                    bool isLastK = groupHcIdx + 1 == dGroupCount * tilingData->hcMult;

                    if ASCEND_IS_AIC {
                        mmService_.CopyInB1Nd2Nz(tilingData->hcMult * tilingData->d, currentGroupK, tilingData->hcMix,
                                                 hcFnGm[sourceGroupKOffset],
                                                 wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
                        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
                        uint64_t nAlign = Align((uint64_t)tilingData->hcMix, AscendC::BLOCK_CUBE);
                        mmService_.Process(tilingData->bs, tilingData->hcMix, mReal,
                                           (256 / AscendC::Std::max(mAlign, nAlign)) * 32, isFirstK, isLastK,
                                           xL1_[aL1BufferID_ * L1_BUF_OFFSET],
                                           wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                        if (isLastK) {
                            mmService_.CopyOut(
                                mmGm[kBlockIdx * tilingData->bs * tilingData->hcMix + mGmOffset * tilingData->hcMix]);
                        }
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
                    } else {
                        CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                        for (uint64_t mInner = 0; mInner < localRows; mInner += tilingData->mUbSize) {
                            uint64_t currentM = AscendC::Std::min((uint64_t)tilingData->mUbSize, localRows - mInner);
                            LocalTensor<float> rmsChunk = rmsNormLocal[mInner];
                            LocalTensor<float> preMixChunk = preMixLocal[mInner * RoundUp<float>(tilingData->hcMix)];
                            for (uint64_t subIdx = 0; subIdx < subCount; ++subIdx) {
                                uint64_t dOffset = dGroupStart + subIdx * tilingData->kUbSize;
                                uint64_t currentK = AscendC::Std::min((uint64_t)tilingData->kUbSize,
                                                                      dGroupStart + currentGroupK - dOffset);
                                uint64_t sourceKOffset = hcIdx * tilingData->d + dOffset;
                                LocalTensor<float> yAccSlice =
                                    yAccLocal[subIdx * yAccSliceSize + mInner * RoundUp<float>(tilingData->kUbSize)];
                                xLocal = xQue.AllocTensor<T>();
                                CopyIn(xGm[(mGmOffset + localRowOffset + mInner) * tilingData->hcMult * tilingData->d +
                                           sourceKOffset],
                                       xLocal, currentM, currentK, tilingData->hcMult * tilingData->d - currentK);
                                xQue.EnQue(xLocal);
                                xLocal = xQue.DeQue<T>();
                                float coeff = 1.0f / static_cast<float>(tilingData->hcMult * tilingData->d);
                                bool isFirstSub = groupIdx == 0 && hcIdx == 0 && subIdx == 0;
                                if (isFirstSub) {
                                    VFProcessCastInvRmsAndPremixYPart1<T, false, false>(
                                        rmsChunk, yAccSlice, xCastLocal, preMixChunk, xLocal, coeff, hcIdx,
                                        RoundUp<float>(tilingData->hcMix), RoundUp<float>(tilingData->kUbSize),
                                        currentM, currentK);
                                } else if (hcIdx == 0) {
                                    VFProcessCastInvRmsAndPremixYPart1<T, true, false>(
                                        rmsChunk, yAccSlice, xCastLocal, preMixChunk, xLocal, coeff, hcIdx,
                                        RoundUp<float>(tilingData->hcMix), RoundUp<float>(tilingData->kUbSize),
                                        currentM, currentK);
                                } else {
                                    VFProcessCastInvRmsAndPremixYPart1<T, true, true>(
                                        rmsChunk, yAccSlice, xCastLocal, preMixChunk, xLocal, coeff, hcIdx,
                                        RoundUp<float>(tilingData->hcMix), RoundUp<float>(tilingData->kUbSize),
                                        currentM, currentK);
                                }
                                xQue.FreeTensor(xLocal);

                                WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                                VFTransND2NZ(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)], xCastLocal, currentM,
                                             currentK);
                                SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));
                                WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));
                                DataCopyParams copyParams;
                                copyParams.blockCount = CeilDiv(currentK, C0_SIZE);
                                copyParams.blockLen = currentM * C0_SIZE * sizeof(float) / BLOCK_SIZE;
                                copyParams.srcStride = CeilAlign(currentM, C0_SIZE) - currentM;
                                copyParams.dstStride = mAlign - currentM;
                                uint64_t l1RowOffset = (localRowOffset + mInner) * (BLOCK_SIZE / sizeof(float));
                                uint64_t l1SubOffset = subIdx * tilingData->kUbSize * mAlign;
                                CopyToL1(xNd2NzLocal[nd2NzBufSize * (bufferIdx & 1)],
                                         xL1_[aL1BufferID_ * L1_BUF_OFFSET + l1SubOffset + l1RowOffset], copyParams);
                                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                                ++bufferIdx;

                                if (hcIdx + 1 == (uint64_t)tilingData->hcMult) {
                                    yLocal = yQue.AllocTensor<T>();
                                    VFProcessPremixYCast<T>(yLocal, yAccSlice, RoundUp<float>(tilingData->kUbSize),
                                                            currentM, currentK);
                                    yQue.EnQue(yLocal);
                                    yLocal = yQue.DeQue<T>();
                                    CopyOut(yLocal,
                                            yGm[(mGmOffset + localRowOffset + mInner) * tilingData->d + dOffset],
                                            currentM, currentK, tilingData->d - currentK);
                                    yQue.FreeTensor(yLocal);
                                }
                            }
                        }
                        CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
                    }
                    aL1BufferID_ ^= 1;
                }
            }

            if ASCEND_IS_AIV {
                rmsQue.EnQue(rmsNormLocal);
                rmsNormLocal = rmsQue.DeQue<float>();
                CopyOut(rmsNormLocal, rmsGm[kBlockIdx * tilingData->bs + mGmOffset + localRowOffset], 1, localRows);
                rmsQue.FreeTensor(rmsNormLocal);
                preMixQue.FreeTensor(preMixLocal);
            }
        }

        if ASCEND_IS_AIV {
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(1));
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(2));
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(2));
        }
        FinishStage1();
        SyncAll<false>();
    }

private:
    __aicore__ inline void FinishStage1()
    {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
            CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        } else {
            mmService_.End();
        }
    }

    TPipe *pipe;
    const HcPreTilingData *tilingData;
    GlobalTensor<T> xGm;
    GlobalTensor<float> hcFnGm;
    GlobalTensor<float> preMixGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> mmGm;
    GlobalTensor<float> rmsGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECIN, 1> preMixQue;
    TQue<QuePosition::VECOUT, 1> rmsQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TBuf<QuePosition::VECCALC> castBuf;
    TBuf<QuePosition::VECCALC> nd2NzBuf;
    TBuf<QuePosition::VECCALC> yAccBuf;

    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> preMixLocal;
    LocalTensor<float> rmsNormLocal;
    LocalTensor<float> xCastLocal;
    LocalTensor<float> xNd2NzLocal;
    LocalTensor<float> yAccLocal;

    HcPreCubeCompute mmService_;
    LocalTensor<float> xL1_;
    LocalTensor<float> wL1_;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
    static constexpr uint64_t FLAG_ID_MAX = 16;
    uint8_t aL1BufferID_ = 0;
};

template <typename T> class HcPreMKSplitCorePremixPart2 {
public:
    __aicore__ inline HcPreMKSplitCorePremixPart2() {}

    // x, preMix and y stay in the signature so the new dispatch can mirror the
    // legacy call. y has already been produced by Part1.
    __aicore__ inline void Init(GM_ADDR, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR, GM_ADDR, GM_ADDR post,
                                GM_ADDR combFrag, GM_ADDR pre, GM_ADDR workspace, const HcPreTilingData *tilingDataPtr,
                                TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        hcScaleGm.SetGlobalBuffer((__gm__ float *)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float *)hcBase);
        postGm.SetGlobalBuffer((__gm__ float *)post);
        combFragGm.SetGlobalBuffer((__gm__ float *)combFrag);
        hasPreOut_ = pre != nullptr;
        if (hasPreOut_) {
            preGm.SetGlobalBuffer((__gm__ float *)pre);
        }
        ubRowGapBlocks_ = UbRowGapBlocks(tilingData->hcMult, tilingData->hcMix);
        mmGm.SetGlobalBuffer((__gm__ float *)workspace);
        rmsGm.SetGlobalBuffer((__gm__ float *)workspace +
                              tilingData->kBlockFactor * tilingData->bs * tilingData->hcMix);

        int64_t rmsAndMmSize =
            tilingData->kBlockFactor * RoundUp<float>(tilingData->stage2RowFactor) * sizeof(float) +
            tilingData->kBlockFactor * tilingData->stage2RowFactor * RoundUp<float>(tilingData->hcMix) * sizeof(float);
        pipe->InitBuffer(rmsAndmmQue, DOUBLE_BUFFER, rmsAndMmSize);
        if (hasPreOut_) {
            pipe->InitBuffer(preQue, DOUBLE_BUFFER,
                             tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        }
        pipe->InitBuffer(postQue, DOUBLE_BUFFER, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
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
            int64_t curBlockIdx = GetBlockIdx();
            if (curBlockIdx >= tilingData->secondUsedCoreNum) {
                return;
            }
            CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult * 2], hcBase2Local, tilingData->hcMult, tilingData->hcMult);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

            int64_t outerLoop = (curBlockIdx == tilingData->secondUsedCoreNum - 1) ? tilingData->rowLoopOfTailBlock
                                                                                   : tilingData->rowLoopOfFormerBlock;
            int64_t tailRows = (curBlockIdx == tilingData->secondUsedCoreNum - 1)
                                   ? tilingData->tailRowFactorOfTailBlock
                                   : tilingData->tailRowFactorOfFormerBlock;
            int64_t rowBase = curBlockIdx * tilingData->rowOfFormerBlock;
            int64_t mmLocalSize =
                tilingData->kBlockFactor * tilingData->stage2RowFactor * RoundUp<float>(tilingData->hcMix);
            for (int64_t outerIdx = 0; outerIdx < outerLoop; ++outerIdx) {
                int64_t rows = outerIdx == outerLoop - 1 ? tailRows : tilingData->stage2RowFactor;
                rmsAndmmLocal = rmsAndmmQue.AllocTensor<float>();
                CopyInWithLoopMode(mmGm[(rowBase + outerIdx * tilingData->stage2RowFactor) * tilingData->hcMix],
                                   rmsAndmmLocal, tilingData->kBlockFactor, rows, tilingData->hcMix,
                                   tilingData->bs * tilingData->hcMix);
                CopyIn(rmsGm[rowBase + outerIdx * tilingData->stage2RowFactor], rmsAndmmLocal[mmLocalSize],
                       tilingData->kBlockFactor, rows, tilingData->bs - rows);
                rmsAndmmQue.EnQue(rmsAndmmLocal);
                rmsAndmmLocal = rmsAndmmQue.DeQue<float>();
                VFProcessInvRmsPart3WithGroupReduce(mixesLocal, rmsAndmmLocal, rmsAndmmLocal[mmLocalSize],
                                                    tilingData->normEps, tilingData->kBlockFactor, rows,
                                                    tilingData->hcMix);

                int64_t outputRow = rowBase + outerIdx * tilingData->stage2RowFactor;
                if (hasPreOut_) {
                    VFProcessPre(mixesLocal, mixesLocal, hcBase0Local, hcScaleGm.GetValue(0), tilingData->hcEps, rows,
                                 tilingData->hcMult, tilingData->hcMix);
                    preLocal = preQue.AllocTensor<float>();
                    CopyOut(mixesLocal, preLocal, rows, tilingData->hcMult, 0, ubRowGapBlocks_);
                    preQue.EnQue(preLocal);
                    preLocal = preQue.DeQue<float>();
                    CopyOut(preLocal, preGm[outputRow * tilingData->hcMult], rows, tilingData->hcMult);
                    preQue.FreeTensor(preLocal);
                }

                postLocal = postQue.AllocTensor<float>();
                VFProcessPost(postLocal, mixesLocal[tilingData->hcMult], hcBase1Local, hcScaleGm.GetValue(1),
                              tilingData->hcEps, rows, tilingData->hcMult, tilingData->hcMix);
                postQue.EnQue(postLocal);
                postLocal = postQue.DeQue<float>();
                CopyOut(postLocal, postGm[outputRow * tilingData->hcMult], rows, tilingData->hcMult);
                postQue.FreeTensor(postLocal);

                VFProcessCombFragPremixElementMajor(mixesLocal, hcBase2Local, hcScaleGm.GetValue(2), tilingData->hcEps,
                                                    tilingData->iterTimes - 1, rows, tilingData->hcMix);
                rmsAndmmQue.FreeTensor(rmsAndmmLocal);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                CopyOut(mixesLocal, combFragGm[outputRow * tilingData->hcMult * tilingData->hcMult], 1,
                        rows * tilingData->hcMult * tilingData->hcMult);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        }
    }

private:
    TPipe *pipe;
    const HcPreTilingData *tilingData;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;
    GlobalTensor<float> preGm;
    GlobalTensor<float> mmGm;
    GlobalTensor<float> rmsGm;

    TQue<QuePosition::VECIN, 1> rmsAndmmQue;
    TQue<QuePosition::VECOUT, 1> preQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TBuf<QuePosition::VECCALC> mixesBuf;
    TBuf<QuePosition::VECCALC> hcBaseBuf0;
    TBuf<QuePosition::VECCALC> hcBaseBuf1;
    TBuf<QuePosition::VECCALC> hcBaseBuf2;

    LocalTensor<float> mixesLocal;
    LocalTensor<float> rmsAndmmLocal;
    LocalTensor<float> preLocal;
    LocalTensor<float> postLocal;
    LocalTensor<float> hcBase0Local;
    LocalTensor<float> hcBase1Local;
    LocalTensor<float> hcBase2Local;
    bool hasPreOut_ = false;
    uint32_t ubRowGapBlocks_ = 0;
};

} // namespace HcPreNs

#endif
