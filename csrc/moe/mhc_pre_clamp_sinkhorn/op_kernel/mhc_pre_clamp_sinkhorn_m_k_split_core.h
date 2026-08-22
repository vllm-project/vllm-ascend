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
 * \file mhc_pre_clamp_sinkhorn_m_split_core.h
 * \brief
 */

#ifndef MHC_PRE_SINKHORN_M_K_SPLIT_A3_CORE_H
#define MHC_PRE_SINKHORN_M_K_SPLIT_A3_CORE_H

#include "kernel_operator.h"
#include "mhc_pre_clamp_sinkhorn_base.h"
#include "mhc_pre_clamp_sinkhorn_cube_compute.h"

namespace MhcPreClampSinkhorn {
using namespace AscendC;
template <typename T>
class MhcPreClampSinkhornMembaseKSplitCorePart1 {
public:
    __aicore__ inline MhcPreClampSinkhornMembaseKSplitCorePart1()
    {
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcFn, GM_ADDR workspace,
                                const MhcPreClampSinkhornTilingData *tilingDataPtr, TPipe *pipePtr, int64_t bsOffset, int64_t curBs, bool isTailBsLoop = false)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        isTailBsLoop_ = isTailBsLoop;
        
        int64_t curML1Size = isTailBsLoop ? tilingData->tailBsML1Size : tilingData->mL1Size;
        int64_t curCubeBlockDimM = isTailBsLoop ? tilingData->tailBsCubeBlockDimM : tilingData->cubeBlockDimM;
        
        xGm.SetGlobalBuffer((__gm__ T *)x + (bsOffset * tilingData->hcMult *tilingData->d));
        hcFnGm.SetGlobalBuffer((__gm__ float *)hcFn);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace);

        uint64_t curVectorBlockIdx = GetBlockIdx();
        uint64_t curCubeBlockIdx = curVectorBlockIdx;
        if ASCEND_IS_AIV {
            curCubeBlockIdx = curCubeBlockIdx / CV_RATIO;
        }
        xCastFp32BufSize_ = curML1Size * CeilAlign(tilingData->cvLoopKSize, MM_CACHE_LINE_BYTES / sizeof(float)) * sizeof(float);
        int64_t SingleCubeCoreXCastSize = xCastFp32BufSize_ * DOUBLE_BUFFER / sizeof(float);
        xCastFp32WsGm.SetGlobalBuffer((__gm__ float *)workspace + curCubeBlockIdx * SingleCubeCoreXCastSize);
        uint64_t wsOffset = tilingData->cubeCoreNum * SingleCubeCoreXCastSize;
        mmOutFp32WsGm.SetGlobalBuffer((__gm__ float *)workspace + wsOffset);
        mmOuterInnerSize_ = CeilAlign(N_SIZE, MM_CACHE_LINE_BYTES / sizeof(float));
        mmOutFp32BufSize_ = curBs * mmOuterInnerSize_;
        wsOffset += tilingData->cubeBlockDimK / curCubeBlockDimM * mmOutFp32BufSize_;
        squareSumFp32WsGm.SetGlobalBuffer((__gm__ float *)workspace + wsOffset);
        squareSumFp32BufSize_ = CeilAlign(curBs, BLOCK_CUBE) * BLOCK_CUBE;

        if ASCEND_IS_AIC {
            cubeCompute_.Init(xCastFp32WsGm, hcFnGm, pipePtr);
            return;
        }
        int64_t xQueNum = tilingData->stage1MFactor * RoundUp<T>(tilingData->cvLoopKSize);
        pipe->InitBuffer(xQue, NUM_TWO, xQueNum * sizeof(T));
        pipe->InitBuffer(mmInQue, NUM_TWO, xQueNum * sizeof(float));
    }

    __aicore__ inline void Process(int64_t bsOffset, int64_t curBs, bool isTailBsLoop = false)
    {
        isTailBsLoop_ = isTailBsLoop;
        int64_t curML1Size = isTailBsLoop ? tilingData->tailBsML1Size : tilingData->mL1Size;
        int64_t curKL1Size = isTailBsLoop ? tilingData->tailBsKL1Size : tilingData->kL1Size;
        int64_t curCubeBlockDimM = isTailBsLoop ? tilingData->tailBsCubeBlockDimM : tilingData->cubeBlockDimM;
        
        if ASCEND_IS_AIC{
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
        }

        uint64_t curBlockIdx = GetBlockIdx();
        uint64_t curVectorBlockIdx = curBlockIdx;
        if ASCEND_IS_AIV {
            curBlockIdx = curBlockIdx / NUM_TWO;
        }
        uint64_t mBlkDimIdx = curBlockIdx / tilingData->cubeBlockDimK;
        uint64_t kBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimK;
        uint64_t kGmStartOffset = tilingData->multCoreSplitKSize * kBlkDimIdx;
        uint64_t kGmEndOffset = kGmStartOffset + tilingData->multCoreSplitKSize;
        if (kGmEndOffset > tilingData->k) {
            kGmEndOffset = tilingData->k;
        }
        uint64_t curCvLoopKSize = tilingData->cvLoopKSize;
        for (uint64_t mGmOffset = mBlkDimIdx * curML1Size; mGmOffset < curBs;
                mGmOffset += curCubeBlockDimM * curML1Size) {
            uint64_t realMSize = mGmOffset + curML1Size > curBs ? curBs - mGmOffset : curML1Size;
            for (uint64_t kGmBaseOffset = kGmStartOffset; kGmBaseOffset < kGmEndOffset; kGmBaseOffset += tilingData->cvLoopKSize) {
                uint64_t realKGmSize = kGmBaseOffset + tilingData->cvLoopKSize > kGmEndOffset ? kGmEndOffset - kGmBaseOffset : tilingData->cvLoopKSize;
                if ASCEND_IS_AIC{
                    MmParams mmParams;
                    mmParams.curML1 = realMSize;
                    mmParams.curKL1 = curKL1Size;
                    mmParams.curNL1 = N_SIZE;
                    mmParams.singleCoreK = realKGmSize;
                    mmParams.xWsKSize = tilingData->cvLoopKSize;
                    mmParams.nOutSize = mmOuterInnerSize_;
                    mmParams.kGmBaseOffset = kGmBaseOffset;
                    mmParams.nGmSize = N_SIZE;
                    mmParams.kGmSize = tilingData->k;
                    mmParams.isLastK = kGmBaseOffset + tilingData->cvLoopKSize > kGmEndOffset;
                    mmParams.isFirstK = kGmBaseOffset == kGmStartOffset;

                    cubeCompute_.WaitBL1Mte1ToMte2Flag();
                    cubeCompute_.CopyInB1(mGmOffset, kGmBaseOffset, realKGmSize, mmParams);

                    CrossCoreWaitFlag(SYNC_AIV_TO_AIC_FLAG);
                    cubeCompute_.ComputeDecode(xCastFp32WsGm[cvLoopIdx_ % DOUBLE_BUFFER * xCastFp32BufSize_],
                        squareSumFp32WsGm[kBlkDimIdx * squareSumFp32BufSize_ + mGmOffset * BLOCK_CUBE],
                        mmOutFp32WsGm[kBlkDimIdx * mmOutFp32BufSize_ + mGmOffset * mmOuterInnerSize_],
                        mmParams);
                    cubeCompute_.SetBL1Mte1ToMte2Flag();
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
                } else {
                    CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
                    int64_t mVectorOffset = mGmOffset;
                    int64_t mVectorLength = (realMSize + 1) / NUM_TWO;
                    if ((curVectorBlockIdx % NUM_TWO) == 1) {
                        mVectorOffset = mGmOffset + (realMSize + 1) / NUM_TWO;
                        mVectorLength = realMSize - mVectorLength;
                    }
                    int64_t curUbLoops = (mVectorLength + tilingData->stage1MFactor - 1) / tilingData->stage1MFactor;
                    int64_t curUbMfactorTail = mVectorLength - ((curUbLoops - 1) * tilingData->stage1MFactor);
                    for (int64_t i = 0; i < curUbLoops; ++i) {
                        int64_t curUbMFactor = (i != (curUbLoops - 1)) ? tilingData->stage1MFactor : curUbMfactorTail;
                        xLocal = xQue.template AllocTensor<T>();
                        int64_t curGlobalxOffset = (mVectorOffset + i * tilingData->stage1MFactor) * tilingData->k + kGmBaseOffset;
                        CopyIn(xGm[curGlobalxOffset], xLocal, curUbMFactor, realKGmSize, tilingData->k - realKGmSize);
                        xQue.template EnQue(xLocal);
                        xLocal = xQue.template DeQue<T>();
                        xCastLocal = mmInQue.AllocTensor<float>();
                        CastTwoDim(xCastLocal, xLocal, curUbMFactor, realKGmSize);
                        xQue.template FreeTensor(xLocal);
                        mmInQue.template EnQue(xCastLocal);
                        xCastLocal = mmInQue.template DeQue<float>();
                        int64_t cutMmInOffset = cvLoopIdx_ % DOUBLE_BUFFER * xCastFp32BufSize_ + (i * tilingData->stage1MFactor + mVectorOffset) * tilingData->cvLoopKSize;
                        CopyOut(xCastLocal, xCastFp32WsGm[cutMmInOffset], curUbMFactor, realKGmSize, tilingData->cvLoopKSize - realKGmSize);
                        mmInQue.FreeTensor(xCastLocal);
                    }
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_TO_AIC_FLAG);
                }
                cvLoopIdx_++;
            }
        }
        if ASCEND_IS_AIC {
            cubeCompute_.End();
        } else {
            CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
            CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
        }
        SyncAll<false>(); // cv全部同步
    }

private:
    TPipe *pipe;
    const MhcPreClampSinkhornTilingData *tilingData;
    GlobalTensor<float> workspaceGm;
    GlobalTensor<float> xCastFp32WsGm;
    GlobalTensor<float> mmOutFp32WsGm;
    GlobalTensor<float> squareSumFp32WsGm;
    GlobalTensor<float> hcFnGm;
    GlobalTensor<T> xGm;

    TQue<QuePosition::VECIN, 1> xQue;

    TQue<QuePosition::VECOUT, 1> mmInQue;

    LocalTensor<T> xLocal;
    LocalTensor<float> xCastLocal;
    LocalTensor<float> yCastLocal;
    LocalTensor<float> mmOutLocal;
    
    HcCubeComputeSplitK<false> cubeCompute_;
    static constexpr uint64_t SYNC_AIV_TO_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_TO_AIV_FLAG = 9;
    static constexpr uint64_t SYNC_MODE2 = NUM_TWO;
    
    uint64_t cvLoopIdx_ = 0;
    uint64_t xCastFp32BufSize_;
    uint64_t mmOuterInnerSize_;
    uint64_t mmOutFp32BufSize_;
    uint64_t squareSumFp32BufSize_;
    bool isTailBsLoop_ = false;
};

template <typename T>
class MhcPreClampSinkhornMembaseKSplitCorePart2 {
public:
    __aicore__ inline MhcPreClampSinkhornMembaseKSplitCorePart2()
    {
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR y,
                                GM_ADDR post, GM_ADDR combFrag,
                                GM_ADDR hPre, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
                                GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR hResLogits,
                                GM_ADDR workspace,
                                const MhcPreClampSinkhornTilingData *tilingDataPtr, TPipe *pipePtr, int64_t bsOffset, bool isTailBsLoop = false)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        needGrad_ = tilingData->needGrad;
        isTailBsLoop_ = isTailBsLoop;

        xGm.SetGlobalBuffer((__gm__ T *)x + (bsOffset * tilingData->hcMult *tilingData->d));
        hcScaleGm.SetGlobalBuffer((__gm__ float *)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float *)hcBase);
        yGm.SetGlobalBuffer((__gm__ T *)y + (bsOffset * tilingData->d));
        postGm.SetGlobalBuffer((__gm__ float *)post + (bsOffset * tilingData->hcMult));
        combFragGm.SetGlobalBuffer((__gm__ float *)combFrag + (bsOffset * tilingData->hcMult * tilingData->hcMult));
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace);

        if (needGrad_) {
            hPreGm.SetGlobalBuffer((__gm__ float *)hPre + (bsOffset * tilingData->hcMult));
            hcBeforeNormGm.SetGlobalBuffer((__gm__ float *)hcBeforeNorm + (bsOffset * tilingData->hcMult *(tilingData->hcMult + 2)));
            invRmsGm.SetGlobalBuffer((__gm__ float *)invRms + bsOffset);
            sumOutGm.SetGlobalBuffer((__gm__ float *)sumOut);
            normOutGm.SetGlobalBuffer((__gm__ float *)normOut);
            reduceOutGm.SetGlobalBuffer((__gm__ float *)hResLogits);
        }

        // InQue
        int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
        int64_t mixesQue01Size = stage1UsedCoreNum * tilingData->stage2RowFactor * tilingData->hcMultAlign * NUM_TWO * sizeof(float);
        pipe->InitBuffer(mixesQue01, NUM_TWO, mixesQue01Size);
        pipe->InitBuffer(mixesQue2, NUM_TWO,
                         stage1UsedCoreNum * tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(squareSumQue, NUM_TWO, stage1UsedCoreNum * tilingData->stage2RowFactor * SQUARE_SUM_SIZE * sizeof(float));

        int64_t xQueNum2 = tilingData->stage2RowFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor);
        pipe->InitBuffer(xQue, NUM_TWO, xQueNum2 * sizeof(T));

        pipe->InitBuffer(squareSumQue, NUM_TWO, stage1UsedCoreNum * tilingData->stage2RowFactor * SQUARE_SUM_SIZE * sizeof(float));

        // OutQue
        pipe->InitBuffer(yQue, NUM_TWO, tilingData->stage2RowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        pipe->InitBuffer(postQue, NUM_TWO, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(combFragQue, NUM_TWO,
                         tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));

        // TBuf
        pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(rowBrcbBuf0, RoundUp<float>(tilingData->stage2RowFactor) * BLOCK_SIZE);
        pipe->InitBuffer(hcBrcbBuf1, RoundUp<float>(tilingData->stage2RowFactor * tilingData->hcMultAlign * NUM_TWO) * BLOCK_SIZE);
        pipe->InitBuffer(reduceBuf, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(mxies01ReduceBuf, tilingData->stage2RowFactor * tilingData->hcMultAlign * NUM_TWO * sizeof(float));
        pipe->InitBuffer(mxies02ReduceBuf, tilingData->stage2RowFactor * tilingData->hcMultAlign * tilingData->hcMult * sizeof(float));
        pipe->InitBuffer(squareReduceBuf, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(xCastBuf, xQueNum2 * sizeof(float));
        pipe->InitBuffer(yCastBuf, tilingData->stage2RowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(float));
        pipe->InitBuffer(rsqrtBuf, RoundUp<float>(tilingData->stage2RowFactor) * sizeof(float));
        // Tiling UB need 512B
        pipe->InitBuffer(maskPatternBuf, RoundUp<uint32_t>(NUM_SIXTEEN * NUM_EIGHT) * sizeof(uint32_t));
        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        hcBase2Local = hcBaseBuf2.Get<float>();
        rowBrcbLocal0 = rowBrcbBuf0.Get<float>();
        hcBrcbLocal1 = hcBrcbBuf1.Get<float>();
        reduceLocal = reduceBuf.Get<float>();
        squareReduceLocal = squareReduceBuf.Get<float>();
        mxies01ReduceLocal = mxies01ReduceBuf.Get<float>();
        mxies02ReduceLocal = mxies02ReduceBuf.Get<float>();
        xCastLocal = xCastBuf.Get<float>();
        yCastLocal = yCastBuf.Get<float>();
        rsqrtLocal = rsqrtBuf.Get<float>();
        maskPatternLocal = maskPatternBuf.Get<uint32_t>();
        SetGatherMaskPattern(maskPatternLocal);
    }

    __aicore__ inline void Process(int64_t bsOffset, int64_t curBs, bool isTailBsLoop = false)
    {
        isTailBsLoop_ = isTailBsLoop;
        
        int64_t curRowOfFormerBlock = isTailBsLoop ? tilingData->tailBsRowOfFormerBlock : tilingData->rowOfFormerBlock;
        int64_t curRowLoopOfFormerBlock = isTailBsLoop ? tilingData->tailBsRowLoopOfFormerBlock : tilingData->rowLoopOfFormerBlock;
        int64_t curRowLoopOfTailBlock = isTailBsLoop ? tilingData->tailBsRowLoopOfTailBlock : tilingData->rowLoopOfTailBlock;
        int64_t curSecondUsedCoreNum = isTailBsLoop ? tilingData->tailBsUsedCoreNum : tilingData->secondUsedCoreNum;
        int64_t curStage2RowFactor = isTailBsLoop ? tilingData->tailBsRowFactor : tilingData->stage2RowFactor;
        int64_t curTailRowFactorOfFormerBlock = isTailBsLoop ? tilingData->tailBsTailRowFactorOfFormerBlock : tilingData->tailRowFactorOfFormerBlock;
        int64_t curTailRowFactorOfTailBlock = isTailBsLoop ? tilingData->tailBsTailRowFactorOfTailBlock : tilingData->tailRowFactorOfTailBlock;
        int64_t curML1Size = isTailBsLoop ? tilingData->tailBsML1Size : tilingData->mL1Size;
        
        if ASCEND_IS_AIV {
            int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
            int64_t stage2BlockIdx = GetBlockIdx();
            int64_t stage2UsedCoreNum = curSecondUsedCoreNum;
            if (stage2BlockIdx >= stage2UsedCoreNum) {
                return;
            }
            int64_t mmLastAxisSize = CeilAlign(tilingData->hcMix, MM_CACHE_LINE_BYTES / sizeof(float));
            int64_t xCastFp32BufSize = curML1Size * CeilAlign(tilingData->cvLoopKSize, MM_CACHE_LINE_BYTES / sizeof(float)) * sizeof(float);
            int64_t workspaceSize1 = (tilingData->cubeCoreNum * DOUBLE_BUFFER * xCastFp32BufSize) / sizeof(float);
            int64_t workspaceSize2 = CeilAlign(stage1UsedCoreNum * curBs * mmLastAxisSize * sizeof(float), WORKSPACE_ALIGN_SIZE) / sizeof(float);
            CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult * NUM_TWO], hcBase2Local, tilingData->hcMult, tilingData->hcMult);
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);

            int64_t rowOuterLoop =
                (stage2BlockIdx == stage2UsedCoreNum - 1) ? curRowLoopOfTailBlock : curRowLoopOfFormerBlock;
            int64_t tailRowFactor = (stage2BlockIdx == stage2UsedCoreNum - 1) ? curTailRowFactorOfTailBlock :
                                                                        curTailRowFactorOfFormerBlock;
            int64_t xGmBlockBaseOffsetPart2 = stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMult * tilingData->d;

            for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
                int64_t xGmBsBaseOffsetPart2 = rowOuterIdx * curStage2RowFactor * tilingData->hcMult * tilingData->d;
                int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : curStage2RowFactor;
                squareSumOutLocal = squareSumQue.AllocTensor<float>();
                
                CopyIn(workspaceGm[workspaceSize1 + workspaceSize2 + stage2BlockIdx * curRowOfFormerBlock * SQUARE_SUM_SIZE + rowOuterIdx * curStage2RowFactor * SQUARE_SUM_SIZE],
                        squareSumOutLocal, stage1UsedCoreNum, curRowFactor * SQUARE_SUM_SIZE, CeilAlign(curBs, SQUARE_SUM_SIZE) * SQUARE_SUM_SIZE - curRowFactor * SQUARE_SUM_SIZE);
                squareSumQue.EnQue(squareSumOutLocal);
                squareSumOutLocal = squareSumQue.DeQue<float>();
                ReduceSumARAPerf(squareReduceLocal, squareSumOutLocal, 1, stage1UsedCoreNum, curRowFactor * SQUARE_SUM_SIZE);
                int64_t curBsIdxForAll = stage2BlockIdx * curRowOfFormerBlock + rowOuterIdx * curStage2RowFactor;
                GatherMaskByDiagonal(rsqrtLocal, squareReduceLocal, maskPatternLocal[(curBsIdxForAll % SQUARE_SUM_SIZE) * 8], curRowFactor);
                float coeff = 1.0f / static_cast<float>(tilingData->k);
                Muls(rsqrtLocal, rsqrtLocal, coeff, curRowFactor);
                PipeBarrier<PIPE_V>();
                Adds(rsqrtLocal, rsqrtLocal, tilingData->normEps, curRowFactor);
                PipeBarrier<PIPE_V>();
                Sqrt(rsqrtLocal, rsqrtLocal, curRowFactor);
                Duplicate(rowBrcbLocal0, static_cast<float>(1.0f), curRowFactor);
                PipeBarrier<PIPE_V>();
                Div(rsqrtLocal, rowBrcbLocal0, rsqrtLocal, curRowFactor);

                if (needGrad_) {
                    VToMTE3Sync();
                    CopyOut(rsqrtLocal, invRmsGm[stage2BlockIdx * curRowOfFormerBlock + rowOuterIdx * curStage2RowFactor], 1,
                            curRowFactor);
                }

                mxies01Local = mixesQue01.AllocTensor<float>();
                uint64_t mixBaseOffset = workspaceSize1 + stage2BlockIdx * curRowOfFormerBlock * CeilAlign(tilingData->hcMix, WORKSPACE_ALIGN_SIZE / sizeof(float)) +
                                         rowOuterIdx * curStage2RowFactor * CeilAlign(tilingData->hcMix, WORKSPACE_ALIGN_SIZE / sizeof(float));
                CopyInWithOuterFor(workspaceGm[mixBaseOffset], mxies01Local, stage1UsedCoreNum, curRowFactor, tilingData->hcMult,
                                curBs, CeilAlign(tilingData->hcMix, WORKSPACE_ALIGN_SIZE / sizeof(float)));
                CopyInWithOuterFor(workspaceGm[mixBaseOffset + tilingData->hcMult],
                                mxies01Local[stage1UsedCoreNum * curRowFactor * tilingData->hcMultAlign],
                                stage1UsedCoreNum, curRowFactor, tilingData->hcMult, curBs,
                                CeilAlign(tilingData->hcMix, WORKSPACE_ALIGN_SIZE / sizeof(float)));

                // wk:[2, kcorenum, bs, n^2 +2n]
                // mx0[2,kcorenum, curRowfator, n]
                mixesQue01.EnQue(mxies01Local);
                mxies01Local = mixesQue01.DeQue<float>();
                ReduceSumARAPerf(mxies01ReduceLocal, mxies01Local, NUM_TWO, stage1UsedCoreNum, curRowFactor * tilingData->hcMultAlign);

                if (needGrad_) {
                    int64_t hcBeforeNormBaseOffset = stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMix +
                                                     rowOuterIdx * curStage2RowFactor * tilingData->hcMix;
                    // mxies01ReduceLocal 布局: [mix0所有行, mix1所有行]
                    //   mix0部分: [row0: hcMultAlign] [row1: hcMultAlign] ...
                    //   mix1部分: [row0: hcMultAlign] [row1: hcMultAlign] ...
                    // hcBeforeNormGm 每行布局: [mix0: hcMult] [mix1: hcMult] [mix2: hcMult*hcMult]
                    VToMTE3Sync();
                    CopyOut(mxies01ReduceLocal, hcBeforeNormGm[hcBeforeNormBaseOffset], curRowFactor,
                            tilingData->hcMult, tilingData->hcMix - tilingData->hcMult);

                    // 搬运 mix1: 从 mxies01ReduceLocal 的后半部分开始
                    CopyOut(mxies01ReduceLocal[curRowFactor * tilingData->hcMultAlign],
                            hcBeforeNormGm[hcBeforeNormBaseOffset + tilingData->hcMult], curRowFactor,
                            tilingData->hcMult, tilingData->hcMix - tilingData->hcMult);
                    MTE3ToVSync();
                }

                ProcessPre(mxies01ReduceLocal, mxies01ReduceLocal, hcBase0Local, rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1,
                            hcScaleGm.GetValue(0), tilingData->hcEps, curRowFactor, tilingData->hcMult);

                if (needGrad_) {
                    int64_t hPreBaseOffset = stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMult +
                                             rowOuterIdx * curStage2RowFactor * tilingData->hcMult;
                    VToMTE3Sync();
                    CopyOut(mxies01ReduceLocal, hPreGm[hPreBaseOffset], curRowFactor, tilingData->hcMult);
                    MTE3ToVSync();
                }
                for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                    int64_t curDFactor =
                        (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                    xLocal = xQue.template AllocTensor<T>();
                    CopyIn(xGm[xGmBlockBaseOffsetPart2 + xGmBsBaseOffsetPart2 +
                                dLoopIdx * tilingData->dFactor],
                            xLocal, curStage2RowFactor * tilingData->hcMult, curDFactor, tilingData->d - curDFactor);
                    xQue.template EnQue(xLocal);
                    xLocal = xQue.template DeQue<T>();
                    yLocal = yQue.template AllocTensor<T>();

                    ProcessY(yLocal, xLocal, mxies01ReduceLocal, hcBrcbLocal1, xCastLocal, yCastLocal, curRowFactor,
                                tilingData->hcMult, curDFactor);
                    xQue.template FreeTensor(xLocal);
                    yQue.template EnQue(yLocal);
                    yLocal = yQue.template DeQue<T>();
                    CopyOut(yLocal,
                            yGm[stage2BlockIdx * curRowOfFormerBlock * tilingData->d +
                                rowOuterIdx * curStage2RowFactor * tilingData->d + dLoopIdx * tilingData->dFactor],
                            curRowFactor, curDFactor, tilingData->d - curDFactor);
                    yQue.template FreeTensor(yLocal);
                }
                // post
                postLocal = postQue.AllocTensor<float>();
                ProcessPost(postLocal, mxies01ReduceLocal[curRowFactor * tilingData->hcMultAlign], hcBase1Local,
                            rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1, hcScaleGm.GetValue(1), curRowFactor,
                            tilingData->hcMult);
                mixesQue01.template FreeTensor(mxies01Local);
                postQue.EnQue(postLocal);
                postLocal = postQue.DeQue<float>();
                CopyOut(postLocal,
                        postGm[stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMult +
                                rowOuterIdx * curStage2RowFactor * tilingData->hcMult],
                        curRowFactor, tilingData->hcMult);
                postQue.FreeTensor(postLocal);

                // combFrag
                mixes2Local = mixesQue2.AllocTensor<float>();
                for (int64_t i = 0; i < stage1UsedCoreNum; ++i) {
                    for (int64_t j = 0; j < curRowFactor; ++j) {
                        CopyIn(workspaceGm[workspaceSize1 + i * curBs * mmLastAxisSize + j * mmLastAxisSize + stage2BlockIdx * curRowOfFormerBlock * mmLastAxisSize + rowOuterIdx * curStage2RowFactor * mmLastAxisSize + tilingData->hcMult * NUM_TWO],
                                        mixes2Local[(i * curRowFactor + j) * tilingData->hcMult * tilingData->hcMultAlign], tilingData->hcMult, tilingData->hcMult);
                    }
                }
                mixesQue2.EnQue(mixes2Local);
                mixes2Local = mixesQue2.DeQue<float>();
                ReduceSumARAPerf(mxies02ReduceLocal, mixes2Local, 1, stage1UsedCoreNum, curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);

                if (needGrad_) {
                    int64_t hcBeforeNormMix2Offset = stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMix +
                                                     rowOuterIdx * curStage2RowFactor * tilingData->hcMix +
                                                     tilingData->hcMult * NUM_TWO;
                    VToMTE3Sync();
                    for (int64_t i = 0 ; i< curRowFactor; i++) {
                        CopyOut(mxies02ReduceLocal[i * tilingData->hcMult * tilingData->hcMultAlign], hcBeforeNormGm[hcBeforeNormMix2Offset + tilingData->hcMix * i],
                                tilingData->hcMult, tilingData->hcMult);
                    }
                    MTE3ToVSync();
                }

                combFragLocal = combFragQue.AllocTensor<float>();

                MulABLastDimBrcInline<float, false>(mxies02ReduceLocal, mxies02ReduceLocal, rsqrtLocal, rowBrcbLocal0, curRowFactor,
                                                    tilingData->hcMult * tilingData->hcMultAlign);
                Muls(mxies02ReduceLocal, mxies02ReduceLocal, hcScaleGm.GetValue(NUM_TWO),
                        curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);
                PipeBarrier<PIPE_V>();
                AddBAFirstDimBrcInline<float>(mxies02ReduceLocal, mxies02ReduceLocal, hcBase2Local, curRowFactor,
                                                tilingData->hcMult * tilingData->hcMultAlign);
                SoftmaxFP32Perf(mxies02ReduceLocal, mxies02ReduceLocal, reduceLocal, hcBrcbLocal1, curRowFactor * tilingData->hcMult,
                                tilingData->hcMult, tilingData->hcEps);
                
                if (needGrad_) {
                    VToMTE3Sync();
                    int64_t normOutBaseOffset = (bsOffset +
                                                stage2BlockIdx * curRowOfFormerBlock +
                                                rowOuterIdx * curStage2RowFactor) * tilingData->hcMult * tilingData->hcMult;
                    CopyOut(mxies02ReduceLocal, normOutGm[normOutBaseOffset], curRowFactor * tilingData->hcMult,
                            tilingData->hcMult);
                    MTE3ToVSync();
                }

                if (needGrad_) {
                    VToMTE3Sync();
                    int64_t sumOutBaseOffset = (bsOffset +
                                               stage2BlockIdx * curRowOfFormerBlock +
                                               rowOuterIdx * curStage2RowFactor) * tilingData->hcMult;
                    CopyOut(reduceLocal, sumOutGm[sumOutBaseOffset], curRowFactor, tilingData->hcMult);
                    MTE3ToVSync();
                }

                ReduceSumARAPerf(reduceLocal, mxies02ReduceLocal, curRowFactor, tilingData->hcMult, tilingData->hcMult);
                if (needGrad_) {
                    VToMTE3Sync();
                    int64_t sumOutBaseOffset = (tilingData->bs +
                                               bsOffset +
                                               stage2BlockIdx * curRowOfFormerBlock +
                                               rowOuterIdx * curStage2RowFactor) * tilingData->hcMult;
                    CopyOut(reduceLocal, sumOutGm[sumOutBaseOffset], curRowFactor, tilingData->hcMult);
                    MTE3ToVSync();
                }
                Adds(reduceLocal, reduceLocal, tilingData->hcEps, curRowFactor * tilingData->hcMult);
                PipeBarrier<PIPE_V>();
                DivABABrcInline(combFragLocal, mxies02ReduceLocal, reduceLocal, curRowFactor, tilingData->hcMult,
                                tilingData->hcMult);

                if (needGrad_) {
                    VToMTE3Sync();
                    int64_t normOutBaseOffset = (tilingData->bs +
                                                bsOffset +
                                                stage2BlockIdx * curRowOfFormerBlock +
                                                rowOuterIdx * curStage2RowFactor) * tilingData->hcMult * tilingData->hcMult;
                    CopyOut(combFragLocal, normOutGm[normOutBaseOffset], curRowFactor * tilingData->hcMult,
                            tilingData->hcMult);
                    MTE3ToVSync();
                }

                for (int64_t iter = 1; iter < tilingData->iterTimes; iter++) {
                    LastDimReduceSumPerf(reduceLocal, combFragLocal, curRowFactor * tilingData->hcMult, tilingData->hcMult);
                    Adds(reduceLocal, reduceLocal, tilingData->hcEps, curRowFactor * tilingData->hcMult);

                    if (needGrad_) {
                        VToMTE3Sync();
                        int64_t sumOutBaseOffset = ((iter * 2) * tilingData->bs +
                                                   bsOffset +
                                                   stage2BlockIdx * curRowOfFormerBlock +
                                                   rowOuterIdx * curStage2RowFactor) * tilingData->hcMult;
                        CopyOut(reduceLocal, sumOutGm[sumOutBaseOffset], curRowFactor, tilingData->hcMult);
                        MTE3ToVSync();
                    }

                    PipeBarrier<PIPE_V>();
                    DivABLastDimBrcInline<float, true>(combFragLocal, combFragLocal, reduceLocal, hcBrcbLocal1,
                                                        curRowFactor * tilingData->hcMult, tilingData->hcMult);

                    if (needGrad_) {
                        VToMTE3Sync();
                        int64_t normOutBaseOffset = ((iter * 2) * tilingData->bs +
                                                    bsOffset +
                                                    stage2BlockIdx * curRowOfFormerBlock +
                                                    rowOuterIdx * curStage2RowFactor) * tilingData->hcMult * tilingData->hcMult;
                        CopyOut(combFragLocal, normOutGm[normOutBaseOffset], curRowFactor * tilingData->hcMult,
                                tilingData->hcMult);
                        MTE3ToVSync();
                    }

                    ReduceSumARAPerf(reduceLocal, combFragLocal, curRowFactor, tilingData->hcMult, tilingData->hcMult);
                    Adds(reduceLocal, reduceLocal, tilingData->hcEps, curRowFactor * tilingData->hcMult);

                    if (needGrad_) {
                        int64_t sumOutBaseOffset = ((iter * 2 + 1) * tilingData->bs +
                                                   bsOffset +
                                                   stage2BlockIdx * curRowOfFormerBlock +
                                                   rowOuterIdx * curStage2RowFactor) * tilingData->hcMult;
                        VToMTE3Sync();
                        CopyOut(reduceLocal, sumOutGm[sumOutBaseOffset], curRowFactor, tilingData->hcMult);
                        MTE3ToVSync();
                    }

                    PipeBarrier<PIPE_V>();
                    DivABABrcInline(combFragLocal, combFragLocal, reduceLocal, curRowFactor, tilingData->hcMult,
                                    tilingData->hcMult);

                    if (needGrad_) {
                        int64_t normOutBaseOffset = ((iter * 2 + 1) * tilingData->bs +
                                                    bsOffset +
                                                    stage2BlockIdx * curRowOfFormerBlock +
                                                    rowOuterIdx * curStage2RowFactor) * tilingData->hcMult * tilingData->hcMult;
                        VToMTE3Sync();
                        CopyOut(combFragLocal, normOutGm[normOutBaseOffset], curRowFactor * tilingData->hcMult,
                                tilingData->hcMult);
                        MTE3ToVSync();
                    }
                }
                mixesQue2.FreeTensor(mixes2Local);
                squareSumQue.template FreeTensor(squareSumOutLocal);

                combFragQue.EnQue(combFragLocal);
                combFragLocal = combFragQue.DeQue<float>();
                CopyOut(combFragLocal,
                        combFragGm[stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMult * tilingData->hcMult +
                                    rowOuterIdx * curStage2RowFactor * tilingData->hcMult * tilingData->hcMult],
                        curRowFactor * tilingData->hcMult, tilingData->hcMult);
                combFragQue.FreeTensor(combFragLocal);
            }
        }
    }

private:
    TPipe *pipe;
    const MhcPreClampSinkhornTilingData *tilingData;
    GlobalTensor<float> mixesGm;
    GlobalTensor<float> rsqrtGm;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<float> workspaceGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;

    GlobalTensor<float> hPreGm;
    GlobalTensor<float> hcBeforeNormGm;
    GlobalTensor<float> invRmsGm;
    GlobalTensor<float> sumOutGm;
    GlobalTensor<float> normOutGm;
    GlobalTensor<float> reduceOutGm;

    bool needGrad_ = false;
    bool isTailBsLoop_ = false;

    TQue<QuePosition::VECIN, 1> mixesQue01;
    TQue<QuePosition::VECIN, 1> mixesQue2;
    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECOUT, 1> combFragQue;

    TQue<QuePosition::VECIN, 1> squareSumQue;

    TBuf<QuePosition::VECCALC> hcBaseBuf0;
    TBuf<QuePosition::VECCALC> hcBaseBuf1;
    TBuf<QuePosition::VECCALC> hcBaseBuf2;

    TBuf<QuePosition::VECCALC> rowBrcbBuf0;
    TBuf<QuePosition::VECCALC> hcBrcbBuf1;
    TBuf<QuePosition::VECCALC> reduceBuf;

    TBuf<QuePosition::VECCALC> rsqrtBuf;
    TBuf<QuePosition::VECCALC> squareReduceBuf;
    TBuf<QuePosition::VECCALC> mxies01ReduceBuf;
    TBuf<QuePosition::VECCALC> mxies02ReduceBuf;

    TBuf<QuePosition::VECCALC> xCastBuf;
    TBuf<QuePosition::VECCALC> yCastBuf;
    TBuf<QuePosition::VECCALC> maskPatternBuf;

    LocalTensor<float> mxies01Local;
    LocalTensor<float> mixes2Local;
    LocalTensor<float> rsqrtLocal;
    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> postLocal;
    LocalTensor<float> combFragLocal;
    LocalTensor<float> hcBase0Local;
    LocalTensor<float> hcBase1Local;
    LocalTensor<float> hcBase2Local;
    LocalTensor<float> rowBrcbLocal0;
    LocalTensor<float> hcBrcbLocal1;
    LocalTensor<float> reduceLocal;
    LocalTensor<float> squareReduceLocal;
    LocalTensor<float> mxies01ReduceLocal;
    LocalTensor<float> mxies02ReduceLocal;
    LocalTensor<float> xCastLocal;
    LocalTensor<float> yCastLocal;
    LocalTensor<float> squareSumOutLocal;
    LocalTensor<uint32_t> maskPatternLocal;
};

} // namespace MhcPreClampSinkhorn

#endif