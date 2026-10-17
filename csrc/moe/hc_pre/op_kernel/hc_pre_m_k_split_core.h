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
 * \file hc_pre_m_split_core.h
 * \brief
 */

#ifndef HC_PRE_M_K_SPLIT_A3_CORE_H
#define HC_PRE_M_K_SPLIT_A3_CORE_H

#include "kernel_operator.h"
#include "hc_pre_base.h"
#include "hc_pre_cube_compute.h"

namespace HcPre {
using namespace AscendC;
template <typename T>
class HcPreMembaseKSplitCorePart1 {
public:
    __aicore__ inline HcPreMembaseKSplitCorePart1()
    {
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcFn, GM_ADDR workspace,
                                const HcPreTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T *)x);
        hcFnGm.SetGlobalBuffer((__gm__ float *)hcFn);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace);

        uint64_t curVectorBlockIdx = GetBlockIdx();
        uint64_t curCubeBlockIdx = curVectorBlockIdx;
        if ASCEND_IS_AIV {
            curCubeBlockIdx = curCubeBlockIdx / CV_RATIO;
        }

        xCastFp32BufSize_ = tilingData->mL1Size *
            CeilAlign(tilingData->cvLoopKSize, MM_CACHE_LINE_BYTES / sizeof(float));
        int64_t SingleCubeCoreXCastSize = xCastFp32BufSize_ * DOUBLE_BUFFER;
        xCastFp32WsGm.SetGlobalBuffer((__gm__ float *)workspace + curCubeBlockIdx * SingleCubeCoreXCastSize);

        uint64_t wsOffset = tilingData->cubeCoreNum * SingleCubeCoreXCastSize;
        mmOutFp32WsGm.SetGlobalBuffer((__gm__ float *)workspace + wsOffset);
        mmOuterInnerSize_ = CeilAlign(N_SIZE, MM_CACHE_LINE_BYTES / sizeof(float));
        mmOutFp32BufSize_ = tilingData->bs * mmOuterInnerSize_;
        wsOffset += CeilAlign(tilingData->cubeBlockDimK * mmOutFp32BufSize_ * sizeof(float),
            static_cast<uint64_t>(WORKSPACE_ALIGN_SIZE)) / sizeof(float);
        squareSumFp32WsGm.SetGlobalBuffer((__gm__ float *)workspace + wsOffset);
        squareSumFp32BufSize_ = CeilAlign(tilingData->bs, BLOCK_CUBE) * BLOCK_CUBE;

        if ASCEND_IS_AIC {
            cubeCompute_.Init(xCastFp32WsGm, hcFnGm, pipePtr);
            return;
        }
        // InQue
        int64_t xQueNum = tilingData->stage1MFactor * RoundUp<T>(tilingData->cvLoopKSize);
        pipe->InitBuffer(xQue, NUM_TWO, xQueNum * sizeof(T));

        // OutQue
        pipe->InitBuffer(mmInQue, NUM_TWO, xQueNum * sizeof(float));
    }

    // 计算 (m,k) 网格分块索引与 GM 范围; curVectorBlockIdx 为 AIV 原始 blockIdx(奇偶分半用)
    __aicore__ inline void GetMkBlockInfo(uint64_t &mGmBaseOffset, uint64_t &mGmEndOffset,
        uint64_t &kGmStartOffset, uint64_t &kGmEndOffset, uint64_t &curVectorBlockIdx)
    {
        uint64_t curBlockIdx = GetBlockIdx();
        curVectorBlockIdx = curBlockIdx;
        if ASCEND_IS_AIV {
            curBlockIdx = curBlockIdx / NUM_TWO;
        }
        uint64_t mBlkDimIdx = curBlockIdx / tilingData->cubeBlockDimK;
        uint64_t kBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimK;
        kBlkDimIdx_ = kBlkDimIdx;
        kGmStartOffset = tilingData->multCoreSplitKSize * kBlkDimIdx;
        kGmEndOffset = kGmStartOffset + tilingData->multCoreSplitKSize;
        if (kGmEndOffset > static_cast<uint64_t>(tilingData->k)) {
            kGmEndOffset = static_cast<uint64_t>(tilingData->k);
        }
        mGmBaseOffset = mBlkDimIdx * tilingData->multCoreSplitMSize;
        mGmEndOffset = mGmBaseOffset + tilingData->multCoreSplitMSize;
        if (mGmEndOffset > static_cast<uint64_t>(tilingData->bs)) {
            mGmEndOffset = static_cast<uint64_t>(tilingData->bs);
        }
    }

    // 单个 m-chunk 的完整 k 循环(原 Process 的 m 循环体). b1/a1 预取状态每 chunk 组重置,
    // 跨 m-chunk 调用间不残留预取依赖. cvLoopIdx_ 跨调用连续, 维持 ws1 双 buffer 槽位交替.
    __aicore__ inline void ProcessMChunk(uint64_t mGmOffset, uint64_t mGmEndOffset,
        uint64_t kGmStartOffset, uint64_t kGmEndOffset, uint64_t curVectorBlockIdx)
    {
        uint64_t realMSize = mGmOffset + static_cast<uint64_t>(tilingData->mL1Size) > mGmEndOffset ?
            mGmEndOffset - mGmOffset : static_cast<uint64_t>(tilingData->mL1Size);
        bool b1Prefetched = false;
        bool a1Prefetched = false;
        for (uint64_t kGmBaseOffset = kGmStartOffset; kGmBaseOffset < kGmEndOffset;
            kGmBaseOffset += tilingData->cvLoopKSize) {
            uint64_t realKGmSize = kGmBaseOffset + tilingData->cvLoopKSize > kGmEndOffset ?
                kGmEndOffset - kGmBaseOffset : tilingData->cvLoopKSize;
            if ASCEND_IS_AIC{
                MmParams mmParams;
                mmParams.curML1 = realMSize;
                mmParams.curKL1 = tilingData->kL1Size;
                mmParams.curNL1 = N_SIZE;
                mmParams.singleCoreK = realKGmSize;
                mmParams.xWsKSize = tilingData->cvLoopKSize;
                mmParams.nOutSize = mmOuterInnerSize_;
                mmParams.kGmBaseOffset = kGmBaseOffset;
                mmParams.nGmSize = N_SIZE;
                mmParams.kGmSize = tilingData->k;
                mmParams.isLastK = kGmBaseOffset + tilingData->cvLoopKSize >= kGmEndOffset;
                mmParams.isFirstK = kGmBaseOffset == kGmStartOffset;

                if (!b1Prefetched) {
                    cubeCompute_.WaitBL1Mte1ToMte2Flag();
                    cubeCompute_.CopyInB1(mGmOffset, kGmBaseOffset, realKGmSize, mmParams);
                }

                uint64_t nextB1Offset = kGmBaseOffset + tilingData->cvLoopKSize;
                bool preloadNextB1 = nextB1Offset < kGmEndOffset;
                uint64_t nextB1Size = 0;
                if (preloadNextB1) {
                    uint64_t nextB1Remain = kGmEndOffset - nextB1Offset;
                    nextB1Size = nextB1Remain < static_cast<uint64_t>(tilingData->cvLoopKSize) ?
                        nextB1Remain : static_cast<uint64_t>(tilingData->cvLoopKSize);
                }

                if (!a1Prefetched) {
                    CrossCoreWaitFlag(SYNC_AIV_TO_AIC_FLAG);
                }
                cubeCompute_.ComputeDecode(
                    xCastFp32WsGm[cvLoopIdx_ % DOUBLE_BUFFER * xCastFp32BufSize_],
                    xCastFp32WsGm[(cvLoopIdx_ + 1) % DOUBLE_BUFFER * xCastFp32BufSize_],
                    squareSumFp32WsGm[kBlkDimIdx_ * squareSumFp32BufSize_ + mGmOffset * BLOCK_CUBE],
                    mmOutFp32WsGm[kBlkDimIdx_ * mmOutFp32BufSize_ + mGmOffset * mmOuterInnerSize_],
                    mmParams, a1Prefetched, preloadNextB1, nextB1Offset, nextB1Size);
                cubeCompute_.SetBL1Mte1ToMte2Flag();
                b1Prefetched = preloadNextB1;
                a1Prefetched = preloadNextB1;
                CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
            } else {
                CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
                // vec.compute();
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
                    int64_t curGlobalxOffset = (mVectorOffset + i * tilingData->stage1MFactor) *
                    tilingData->k + kGmBaseOffset;
                    CopyIn(xGm[curGlobalxOffset], xLocal, curUbMFactor, realKGmSize, tilingData->k - realKGmSize);
                    xQue.template EnQue(xLocal);
                    xLocal = xQue.template DeQue<T>();
                    xCastLocal = mmInQue.AllocTensor<float>();
                    CastTwoDim(xCastLocal, xLocal, curUbMFactor, realKGmSize);
                    xQue.template FreeTensor(xLocal);
                    mmInQue.template EnQue(xCastLocal);
                    xCastLocal = mmInQue.template DeQue<float>();
                    int64_t cutMmInOffset = cvLoopIdx_ % DOUBLE_BUFFER * xCastFp32BufSize_ +
                    (i * tilingData->stage1MFactor + mVectorOffset - mGmOffset) *
                    tilingData->cvLoopKSize;
                    CopyOut(xCastLocal, xCastFp32WsGm[cutMmInOffset], curUbMFactor, realKGmSize,
                    tilingData->cvLoopKSize - realKGmSize);
                    mmInQue.FreeTensor(xCastLocal);
                }
                CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_TO_AIC_FLAG);
            }
            cvLoopIdx_++;
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC{
            // 初始设置
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
        }

        uint64_t mGmBaseOffset = 0;
        uint64_t mGmEndOffset = 0;
        uint64_t kGmStartOffset = 0;
        uint64_t kGmEndOffset = 0;
        uint64_t curVectorBlockIdx = 0;
        GetMkBlockInfo(mGmBaseOffset, mGmEndOffset, kGmStartOffset, kGmEndOffset, curVectorBlockIdx);
        for (uint64_t mGmOffset = mGmBaseOffset; mGmOffset < mGmEndOffset;
                mGmOffset += tilingData->mL1Size) {
            ProcessMChunk(mGmOffset, mGmEndOffset, kGmStartOffset, kGmEndOffset, curVectorBlockIdx);
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
    const HcPreTilingData *tilingData;
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

    HcCubeCompute<false> cubeCompute_;
    static constexpr uint64_t SYNC_AIV_TO_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_TO_AIV_FLAG = 9;
    static constexpr uint64_t SYNC_MODE2 = NUM_TWO;

    uint64_t cvLoopIdx_ = 0;
    uint64_t kBlkDimIdx_ = 0;
    uint64_t xCastFp32BufSize_;
    uint64_t mmOuterInnerSize_;
    uint64_t mmOutFp32BufSize_;
    uint64_t squareSumFp32BufSize_;
};

template <typename T, bool useOriginal>
class HcPreMembaseKSplitCorePart2 {
public:
    __aicore__ inline HcPreMembaseKSplitCorePart2()
    {
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR x, GM_ADDR hcScale,
        GM_ADDR hcBase, GM_ADDR y, GM_ADDR post, GM_ADDR combFrag,
        GM_ADDR preMix, GM_ADDR pre, GM_ADDR workspace)
    {
        xGm.SetGlobalBuffer((__gm__ T *)x);
        hcScaleGm.SetGlobalBuffer((__gm__ float *)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float *)hcBase);
        yGm.SetGlobalBuffer((__gm__ T *)y);
        postGm.SetGlobalBuffer((__gm__ float *)post);
        combFragGm.SetGlobalBuffer((__gm__ float *)combFrag);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace);
    }

    __aicore__ inline void InitCommonBuffers(int64_t stage1UsedCoreNum, int64_t rowFactor)
    {
        pipe->InitBuffer(squareSumQue, NUM_TWO,
            stage1UsedCoreNum * rowFactor * SQUARE_SUM_SIZE * sizeof(float));
        pipe->InitBuffer(squareReduceBuf, rowFactor * SQUARE_SUM_SIZE * sizeof(float));
        pipe->InitBuffer(rsqrtBuf, RoundUp<float>(rowFactor) * sizeof(float));
        pipe->InitBuffer(rowBrcbBuf0, RoundUp<float>(rowFactor) * BLOCK_SIZE);
        pipe->InitBuffer(maskPatternBuf,
            RoundUp<uint32_t>((rowFactor + MASK_PATTERN_DIM_SIZE - 1) * MASK_PATTERN_REPEAT_SIZE) * sizeof(uint32_t));

        squareReduceLocal = squareReduceBuf.Get<float>();
        rsqrtLocal = rsqrtBuf.Get<float>();
        rowBrcbLocal0 = rowBrcbBuf0.Get<float>();
        maskPatternLocal = maskPatternBuf.Get<uint32_t>();
        SetGatherMaskPattern(maskPatternLocal, static_cast<uint16_t>(rowFactor + MASK_PATTERN_DIM_SIZE - 1));
    }

    __aicore__ inline void InitPrePostBuffers(int64_t stage1UsedCoreNum)
    {
        int64_t rowFactor = tilingData->prePostRowFactor;
        // Pre/post and y have different working-set shapes. Keep the y
        // buffers at the stage2 row factor (normally one row for full-D
        // tiles), while batching the row-only pre/post path separately.
        int64_t yRowFactor = tilingData->stage2RowFactor;
        int64_t xQueSize = yRowFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor);
        InitCommonBuffers(stage1UsedCoreNum, rowFactor);
        pipe->InitBuffer(mixesQue01, NUM_TWO,
            stage1UsedCoreNum * rowFactor * tilingData->hcMultAlign * NUM_TWO * sizeof(float));
        pipe->InitBuffer(xQue, NUM_TWO, xQueSize * sizeof(T));
        pipe->InitBuffer(yQue, NUM_TWO,
            yRowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        // ProcessPost 按 prePostRowFactor 批量处理行，postQue 需容纳整个批次。
        pipe->InitBuffer(postQue, NUM_TWO,
            tilingData->prePostRowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBrcbBuf1,
            RoundUp<float>(rowFactor * tilingData->hcMultAlign) * BLOCK_SIZE);
        pipe->InitBuffer(mixes01ReduceBuf,
            rowFactor * tilingData->hcMultAlign * NUM_TWO * sizeof(float));
        pipe->InitBuffer(xCastBuf, xQueSize * sizeof(float));
        pipe->InitBuffer(yCastBuf,
            yRowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(float));

        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        hcBrcbLocal1 = hcBrcbBuf1.Get<float>();
        mixes01ReduceLocal = mixes01ReduceBuf.Get<float>();
        xCastLocal = xCastBuf.Get<float>();
        yCastLocal = yCastBuf.Get<float>();
    }

    __aicore__ inline void InitSinkhornBuffers(int64_t stage1UsedCoreNum)
    {
        int64_t rowFactor = tilingData->sinkhornRowFactor;
        int64_t rowFactorAlign = RoundUp<float>(rowFactor);
        InitCommonBuffers(stage1UsedCoreNum, rowFactor);
        pipe->InitBuffer(mixesQue2, DOUBLE_BUFFER,
            stage1UsedCoreNum * rowFactor * COMPACT_MATRIX_SIZE * sizeof(float));
        pipe->InitBuffer(combFragQue, DOUBLE_BUFFER,
            rowFactor * COMPACT_MATRIX_SIZE * sizeof(float));
        pipe->InitBuffer(hcBaseBuf2, COMPACT_MATRIX_SIZE * sizeof(float));
        pipe->InitBuffer(hcBrcbBuf1, COMPACT_MATRIX_SIZE * BLOCK_SIZE);
        pipe->InitBuffer(reduceBuf,
            COMPACT_HC_MULT * rowFactorAlign * sizeof(float));
        pipe->InitBuffer(mixes02ReduceBuf,
            COMPACT_MATRIX_SIZE * rowFactorAlign * sizeof(float));

        hcBase2Local = hcBaseBuf2.Get<float>();
        hcBrcbLocal1 = hcBrcbBuf1.Get<float>();
        reduceLocal = reduceBuf.Get<float>();
        mixes02ReduceLocal = mixes02ReduceBuf.Get<float>();
    }

    __aicore__ inline void InitOriginalBuffers(int64_t stage1UsedCoreNum)
    {
        int64_t rowFactor = tilingData->stage2RowFactor;
        int64_t xQueSize = rowFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor);
        InitCommonBuffers(stage1UsedCoreNum, rowFactor);

        pipe->InitBuffer(mixesQue01, NUM_TWO,
            stage1UsedCoreNum * rowFactor * tilingData->hcMultAlign * NUM_TWO * sizeof(float));
        pipe->InitBuffer(mixesQue2, NUM_TWO,
            stage1UsedCoreNum * rowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(xQue, NUM_TWO, xQueSize * sizeof(T));
        pipe->InitBuffer(yQue, NUM_TWO,
            rowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        pipe->InitBuffer(postQue, NUM_TWO,
            rowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(combFragQue, NUM_TWO,
            rowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));

        pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf2,
            tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBrcbBuf1,
            RoundUp<float>(rowFactor * tilingData->hcMultAlign) * BLOCK_SIZE);
        pipe->InitBuffer(reduceBuf,
            rowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(mixes01ReduceBuf,
            rowFactor * tilingData->hcMultAlign * NUM_TWO * sizeof(float));
        pipe->InitBuffer(mixes02ReduceBuf,
            rowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(xCastBuf, xQueSize * sizeof(float));
        pipe->InitBuffer(yCastBuf,
            rowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(float));

        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        hcBase2Local = hcBaseBuf2.Get<float>();
        hcBrcbLocal1 = hcBrcbBuf1.Get<float>();
        reduceLocal = reduceBuf.Get<float>();
        mixes01ReduceLocal = mixes01ReduceBuf.Get<float>();
        mixes02ReduceLocal = mixes02ReduceBuf.Get<float>();
        xCastLocal = xCastBuf.Get<float>();
        yCastLocal = yCastBuf.Get<float>();
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcScale, GM_ADDR hcBase,
        GM_ADDR y, GM_ADDR post, GM_ADDR combFrag, GM_ADDR preMix, GM_ADDR pre,
        GM_ADDR workspace, const HcPreTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        InitGlobalBuffers(x, hcScale, hcBase, y, post, combFrag, preMix, pre, workspace);
        if ASCEND_IS_AIC {
            return;
        }
        int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
        if constexpr (useOriginal) {
            InitOriginalBuffers(stage1UsedCoreNum);
        } else {
            if (GetBlockIdx() < tilingData->prePostCoreNum) {
                InitPrePostBuffers(stage1UsedCoreNum);
            } else {
                InitSinkhornBuffers(stage1UsedCoreNum);
            }
        }
    }

    __aicore__ inline void CalcRsqrt(int64_t stage1UsedCoreNum, int64_t globalRowOffset,
                                    int64_t curRowFactor, int64_t workspaceSize1, int64_t workspaceSize2)
    {
        squareSumOutLocal = squareSumQue.AllocTensor<float>();
        CopyIn(workspaceGm[workspaceSize1 + workspaceSize2 + globalRowOffset * SQUARE_SUM_SIZE],
            squareSumOutLocal, stage1UsedCoreNum, curRowFactor * SQUARE_SUM_SIZE,
            CeilAlign(tilingData->bs, SQUARE_SUM_SIZE) * SQUARE_SUM_SIZE - curRowFactor * SQUARE_SUM_SIZE);
        squareSumQue.EnQue(squareSumOutLocal);
        squareSumOutLocal = squareSumQue.DeQue<float>();
        ReduceSumARAPerf(squareReduceLocal, squareSumOutLocal, 1, stage1UsedCoreNum,
            curRowFactor * SQUARE_SUM_SIZE);
        GatherMaskByDiagonal(rsqrtLocal, squareReduceLocal, maskPatternLocal, curRowFactor,
            globalRowOffset % SQUARE_SUM_SIZE);
        float coeff = 1.0f / static_cast<float>(tilingData->k);
        Muls(rsqrtLocal, rsqrtLocal, coeff, curRowFactor);
        PipeBarrier<PIPE_V>();
        Adds(rsqrtLocal, rsqrtLocal, tilingData->normEps, curRowFactor);
        PipeBarrier<PIPE_V>();
        Sqrt(rsqrtLocal, rsqrtLocal, curRowFactor);
        Duplicate(rowBrcbLocal0, static_cast<float>(1.0f), curRowFactor);
        PipeBarrier<PIPE_V>();
        Div(rsqrtLocal, rowBrcbLocal0, rsqrtLocal, curRowFactor);
    }

    __aicore__ inline void CopyInPartialCombCompact(int64_t stage1UsedCoreNum, int64_t workspaceSize1,
                                                    int64_t mmLastAxisSize, int64_t globalRowOffset,
                                                    int64_t curRowFactor)
    {
        int64_t combOffset = tilingData->hcMult * NUM_TWO;
        for (int64_t kBlockIdx = 0; kBlockIdx < stage1UsedCoreNum; kBlockIdx++) {
            int64_t gmOffset = workspaceSize1 + kBlockIdx * tilingData->bs * mmLastAxisSize +
                globalRowOffset * mmLastAxisSize + combOffset;
            int64_t ubOffset = kBlockIdx * curRowFactor * COMPACT_MATRIX_SIZE;
            CopyIn(workspaceGm[gmOffset], mixes2Local[ubOffset], curRowFactor,
                COMPACT_MATRIX_SIZE, mmLastAxisSize - COMPACT_MATRIX_SIZE);
        }
    }

    __aicore__ inline void GatherCompactToElementMajor(const LocalTensor<float> &output,
                                                        const LocalTensor<float> &input,
                                                        int64_t curRowFactor)
    {
        int64_t rowFactorAlign = RoundUp<float>(curRowFactor);
        Duplicate(output, 0.0f, COMPACT_MATRIX_SIZE * rowFactorAlign);
        PipeBarrier<PIPE_V>();
        for (int64_t matrixElement = 0; matrixElement < COMPACT_MATRIX_SIZE; matrixElement++) {
            uint64_t rsvdCnt = 0;
            GatherMask(output[matrixElement * rowFactorAlign], input,
                maskPatternLocal[matrixElement * MASK_PATTERN_REPEAT_SIZE], true,
                COMPACT_MATRIX_SIZE, {1, static_cast<uint16_t>(curRowFactor), 2, 0}, rsvdCnt);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GatherElementMajorToCompact(const LocalTensor<float> &output,
                                                        const LocalTensor<float> &input,
                                                        int64_t curRowFactor)
    {
        int64_t rowFactorAlign = RoundUp<float>(curRowFactor);
        int64_t rowStride = rowFactorAlign / (BLOCK_SIZE / sizeof(float));
        for (int64_t rowIdx = 0; rowIdx < curRowFactor; rowIdx++) {
            int64_t rowGroupOffset = rowIdx / MASK_PATTERN_DIM_SIZE * MASK_PATTERN_DIM_SIZE;
            int64_t rowInGroup = rowIdx % MASK_PATTERN_DIM_SIZE;
            int64_t rowsInGroup = curRowFactor - rowGroupOffset;
            if (rowsInGroup > MASK_PATTERN_DIM_SIZE) {
                rowsInGroup = MASK_PATTERN_DIM_SIZE;
            }
            uint64_t rsvdCnt = 0;
            // GatherMask patterns repeat every 16 elements. For row factors above 16,
            // advance the source to the current group instead of aliasing row 16 to row 0.
            GatherMask(output[rowIdx * COMPACT_MATRIX_SIZE], input[rowGroupOffset],
                maskPatternLocal[rowInGroup * MASK_PATTERN_REPEAT_SIZE], true,
                rowsInGroup, {1, COMPACT_MATRIX_SIZE, static_cast<uint16_t>(rowStride), 0}, rsvdCnt);
        }
        PipeBarrier<PIPE_V>();
    }

    template <bool reduceMax>
    __aicore__ inline void ReduceCompactRows(const LocalTensor<float> &output,
                                              const LocalTensor<float> &input,
                                              int64_t curRowFactor)
    {
        int64_t rowFactorAlign = RoundUp<float>(curRowFactor);
        int64_t rowStride = rowFactorAlign / (BLOCK_SIZE / sizeof(float));
        BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = rowStride;
        params.src0RepStride = COMPACT_HC_MULT * rowStride;
        params.src1RepStride = COMPACT_HC_MULT * rowStride;
        if constexpr (reduceMax) {
            Max(output, input, input[rowFactorAlign], curRowFactor, COMPACT_HC_MULT, params);
            PipeBarrier<PIPE_V>();
            params.src0RepStride = rowStride;
            Max(output, output, input[NUM_TWO * rowFactorAlign], curRowFactor, COMPACT_HC_MULT, params);
            PipeBarrier<PIPE_V>();
            Max(output, output, input[3 * rowFactorAlign], curRowFactor, COMPACT_HC_MULT, params);
        } else {
            Add(output, input, input[rowFactorAlign], curRowFactor, COMPACT_HC_MULT, params);
            PipeBarrier<PIPE_V>();
            params.src0RepStride = rowStride;
            Add(output, output, input[NUM_TWO * rowFactorAlign], curRowFactor, COMPACT_HC_MULT, params);
            PipeBarrier<PIPE_V>();
            Add(output, output, input[3 * rowFactorAlign], curRowFactor, COMPACT_HC_MULT, params);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceCompactCols(const LocalTensor<float> &output,
                                              const LocalTensor<float> &input,
                                              int64_t curRowFactor)
    {
        int64_t rowFactorAlign = RoundUp<float>(curRowFactor);
        int64_t rowStride = rowFactorAlign / (BLOCK_SIZE / sizeof(float));
        BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = rowStride;
        params.src0RepStride = rowStride;
        params.src1RepStride = rowStride;
        Add(output, input, input[COMPACT_HC_MULT * rowFactorAlign],
            curRowFactor, COMPACT_HC_MULT, params);
        PipeBarrier<PIPE_V>();
        Add(output, output, input[NUM_TWO * COMPACT_HC_MULT * rowFactorAlign],
            curRowFactor, COMPACT_HC_MULT, params);
        PipeBarrier<PIPE_V>();
        Add(output, output, input[3 * COMPACT_HC_MULT * rowFactorAlign],
            curRowFactor, COMPACT_HC_MULT, params);
        PipeBarrier<PIPE_V>();
    }

    template <bool divide>
    __aicore__ inline void ApplyCompactRowValues(const LocalTensor<float> &output,
                                                  const LocalTensor<float> &input,
                                                  const LocalTensor<float> &rowValues,
                                                  int64_t curRowFactor)
    {
        int64_t rowFactorAlign = RoundUp<float>(curRowFactor);
        int64_t rowStride = rowFactorAlign / (BLOCK_SIZE / sizeof(float));
        BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = COMPACT_HC_MULT * rowStride;
        params.src0RepStride = COMPACT_HC_MULT * rowStride;
        params.src1RepStride = rowStride;
        for (int64_t colIdx = 0; colIdx < COMPACT_HC_MULT; colIdx++) {
            if constexpr (divide) {
                Div(output[colIdx * rowFactorAlign], input[colIdx * rowFactorAlign], rowValues,
                    curRowFactor, COMPACT_HC_MULT, params);
            } else {
                Sub(output[colIdx * rowFactorAlign], input[colIdx * rowFactorAlign], rowValues,
                    curRowFactor, COMPACT_HC_MULT, params);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplyCompactColValues(const LocalTensor<float> &output,
                                                  const LocalTensor<float> &input,
                                                  const LocalTensor<float> &colValues,
                                                  int64_t curRowFactor)
    {
        int64_t rowFactorAlign = RoundUp<float>(curRowFactor);
        int64_t rowStride = rowFactorAlign / (BLOCK_SIZE / sizeof(float));
        BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = rowStride;
        params.src0RepStride = rowStride;
        params.src1RepStride = rowStride;
        for (int64_t matrixRow = 0; matrixRow < COMPACT_HC_MULT; matrixRow++) {
            Div(output[matrixRow * COMPACT_HC_MULT * rowFactorAlign],
                input[matrixRow * COMPACT_HC_MULT * rowFactorAlign], colValues,
                curRowFactor, COMPACT_HC_MULT, params);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void NormalizeCompactRows(const LocalTensor<float> &data,
                                                 int64_t curRowFactor)
    {
        int64_t rowFactorAlign = RoundUp<float>(curRowFactor);
        ReduceCompactRows<false>(reduceLocal, data, curRowFactor);
        Adds(reduceLocal, reduceLocal, tilingData->hcEps,
            COMPACT_HC_MULT * rowFactorAlign);
        PipeBarrier<PIPE_V>();
        ApplyCompactRowValues<true>(data, data, reduceLocal, curRowFactor);
    }

    __aicore__ inline void NormalizeCompactCols(const LocalTensor<float> &data,
                                                 int64_t curRowFactor)
    {
        int64_t rowFactorAlign = RoundUp<float>(curRowFactor);
        ReduceCompactCols(reduceLocal, data, curRowFactor);
        Adds(reduceLocal, reduceLocal, tilingData->hcEps,
            COMPACT_HC_MULT * rowFactorAlign);
        PipeBarrier<PIPE_V>();
        ApplyCompactColValues(data, data, reduceLocal, curRowFactor);
    }

    __aicore__ inline void SoftmaxCompactRows(const LocalTensor<float> &data,
                                               int64_t curRowFactor)
    {
        int64_t rowFactorAlign = RoundUp<float>(curRowFactor);
        ReduceCompactRows<true>(reduceLocal, data, curRowFactor);
        ApplyCompactRowValues<false>(data, data, reduceLocal, curRowFactor);
        Exp(data, data, COMPACT_MATRIX_SIZE * rowFactorAlign);
        PipeBarrier<PIPE_V>();
        ReduceCompactRows<false>(reduceLocal, data, curRowFactor);
        ApplyCompactRowValues<true>(data, data, reduceLocal, curRowFactor);
        Adds(data, data, tilingData->hcEps, COMPACT_MATRIX_SIZE * rowFactorAlign);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessCompactSinkhorn(const LocalTensor<float> &data,
                                                   int64_t curRowFactor, float hcScale)
    {
        int64_t rowFactorAlign = RoundUp<float>(curRowFactor);
        int64_t rowStride = rowFactorAlign / (BLOCK_SIZE / sizeof(float));
        Brcb(hcBrcbLocal1, hcBase2Local, NUM_TWO,
            {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();

        BinaryRepeatParams mulParams;
        mulParams.dstBlkStride = 1;
        mulParams.src0BlkStride = 1;
        mulParams.src1BlkStride = 1;
        mulParams.dstRepStride = rowStride;
        mulParams.src0RepStride = rowStride;
        mulParams.src1RepStride = 0;
        Mul(data, data, rsqrtLocal, curRowFactor, COMPACT_MATRIX_SIZE, mulParams);
        PipeBarrier<PIPE_V>();
        Muls(data, data, hcScale, COMPACT_MATRIX_SIZE * rowFactorAlign);
        PipeBarrier<PIPE_V>();

        BinaryRepeatParams biasParams = mulParams;
        biasParams.src1BlkStride = 0;
        biasParams.src1RepStride = 1;
        Add(data, data, hcBrcbLocal1, curRowFactor, COMPACT_MATRIX_SIZE, biasParams);
        PipeBarrier<PIPE_V>();

        SoftmaxCompactRows(data, curRowFactor);
        NormalizeCompactCols(data, curRowFactor);
        for (int64_t iter = 0; iter < tilingData->iterTimes - 1; iter++) {
            NormalizeCompactRows(data, curRowFactor);
            NormalizeCompactCols(data, curRowFactor);
        }
    }

    __aicore__ inline void PrePostLoadBase(float &hcScale0, float &hcScale1)
    {
        CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
        CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        hcScale0 = hcScaleGm.GetValue(0);
        hcScale1 = hcScaleGm.GetValue(1);
    }

    // 连续 actualRow 行的一批 pre/post 处理(原 ProcessPrePost 行循环体主体)
    __aicore__ inline void ProcessPrePostBatch(int64_t stage1UsedCoreNum, int64_t workspaceSize1,
        int64_t workspaceSize2, int64_t mmLastAxisSize, int64_t globalRowOffset,
        int64_t curRowFactor, float hcScale0, float hcScale1)
    {
        CalcRsqrt(stage1UsedCoreNum, globalRowOffset, curRowFactor, workspaceSize1, workspaceSize2);

        mixes01Local = mixesQue01.AllocTensor<float>();

        uint64_t mixBaseOffset = workspaceSize1 + globalRowOffset * mmLastAxisSize;
        CopyInWithOuterFor(workspaceGm[mixBaseOffset], mixes01Local, stage1UsedCoreNum,
            curRowFactor, tilingData->hcMult, tilingData->bs,
            mmLastAxisSize);
        CopyInWithOuterFor(workspaceGm[mixBaseOffset + tilingData->hcMult],
            mixes01Local[stage1UsedCoreNum * curRowFactor *
            tilingData->hcMultAlign], stage1UsedCoreNum, curRowFactor, tilingData->hcMult,
            tilingData->bs, mmLastAxisSize);

        mixesQue01.EnQue(mixes01Local);
        mixes01Local = mixesQue01.DeQue<float>();
        ReduceSumARAPerf(mixes01ReduceLocal, mixes01Local, NUM_TWO, stage1UsedCoreNum,
            curRowFactor * tilingData->hcMultAlign);
        ProcessPre(mixes01ReduceLocal, mixes01ReduceLocal, hcBase0Local, rsqrtLocal,
            rowBrcbLocal0, hcBrcbLocal1, hcScale0, tilingData->hcEps,
            curRowFactor, tilingData->hcMult);
        mixesQue01.template FreeTensor(mixes01Local);
        squareSumQue.template FreeTensor(squareSumOutLocal);

        // Keep the D tile resident for each row. This lets the row-only
        // pre/post transforms run in a larger batch without inflating the
        // x/y UB buffers that are sized for one full-D row.
        for (int64_t rowIdx = 0; rowIdx < curRowFactor; rowIdx++) {
            int64_t rowOffset = globalRowOffset + rowIdx;
            const LocalTensor<float> preRow =
                mixes01ReduceLocal[rowIdx * tilingData->hcMultAlign];
            for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                int64_t curDFactor = dLoopIdx == tilingData->dLoop - 1 ?
                    tilingData->tailDFactor : tilingData->dFactor;
                xLocal = xQue.template AllocTensor<T>();
                CopyIn(xGm[rowOffset * tilingData->hcMult * tilingData->d +
                    dLoopIdx * tilingData->dFactor], xLocal,
                    tilingData->hcMult, curDFactor, tilingData->d - curDFactor);
                xQue.template EnQue(xLocal);
                xLocal = xQue.template DeQue<T>();
                yLocal = yQue.template AllocTensor<T>();
                ProcessY(yLocal, xLocal, preRow, hcBrcbLocal1, xCastLocal, yCastLocal,
                    1, tilingData->hcMult, curDFactor);
                xQue.template FreeTensor(xLocal);
                yQue.template EnQue(yLocal);
                yLocal = yQue.template DeQue<T>();
                CopyOut(yLocal, yGm[rowOffset * tilingData->d + dLoopIdx * tilingData->dFactor],
                    1, curDFactor, tilingData->d - curDFactor);
                yQue.template FreeTensor(yLocal);
            }
        }

        // 在行循环外批量处理 post，减少 sigmoid/mul 链的逐行同步开销。
        // rowBrcbLocal0 已按行广播；rsqrtLocal 在行循环中保持不变。
        postLocal = postQue.AllocTensor<float>();
        ProcessPost(postLocal, mixes01ReduceLocal[curRowFactor * tilingData->hcMultAlign],
            hcBase1Local, rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1, hcScale1,
            curRowFactor, tilingData->hcMult);
        postQue.EnQue(postLocal);
        postLocal = postQue.template DeQue<float>();
        // post的UB布局行距hcMultAlign(列4 pad到8), GM布局行距hcMult, 非整
        // block倍数关系, 保留逐行CopyOut.
        for (int64_t rowIdx = 0; rowIdx < curRowFactor; rowIdx++) {
            int64_t rowOffset = globalRowOffset + rowIdx;
            CopyOut(postLocal[rowIdx * tilingData->hcMultAlign],
                postGm[rowOffset * tilingData->hcMult], 1, tilingData->hcMult);
        }
        postQue.FreeTensor(postLocal);
    }

    __aicore__ inline void ProcessPrePost(int64_t stage1UsedCoreNum, int64_t workspaceSize1,
                                         int64_t workspaceSize2, int64_t mmLastAxisSize)
    {
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx >= tilingData->secondUsedCoreNum) {
            return;
        }
        float hcScale0 = 0.0f;
        float hcScale1 = 0.0f;
        PrePostLoadBase(hcScale0, hcScale1);

        int64_t rowOuterLoop = blockIdx == tilingData->secondUsedCoreNum - 1 ?
            tilingData->prePostRowLoopOfTailBlock : tilingData->prePostRowLoopOfFormerBlock;
        int64_t tailRowFactor = blockIdx == tilingData->secondUsedCoreNum - 1 ?
            tilingData->prePostTailRowFactorOfTailBlock : tilingData->prePostTailRowFactorOfFormerBlock;

        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = rowOuterIdx == rowOuterLoop - 1 ?
                tailRowFactor : tilingData->prePostRowFactor;
            int64_t globalRowOffset = blockIdx * tilingData->prePostRowOfFormerBlock +
                rowOuterIdx * tilingData->prePostRowFactor;
            ProcessPrePostBatch(stage1UsedCoreNum, workspaceSize1, workspaceSize2,
                mmLastAxisSize, globalRowOffset, curRowFactor, hcScale0, hcScale1);
        }
    }

    // 连续 actualRow 行的一批 sinkhorn 处理(原 ProcessSinkhorn 行循环体原样抽出)
    __aicore__ inline void ProcessSinkhornBatch(int64_t stage1UsedCoreNum, int64_t workspaceSize1,
        int64_t workspaceSize2, int64_t mmLastAxisSize, int64_t globalRowOffset,
        int64_t curRowFactor, float hcScale2)
    {
        CalcRsqrt(stage1UsedCoreNum, globalRowOffset, curRowFactor, workspaceSize1, workspaceSize2);

        mixes2Local = mixesQue2.AllocTensor<float>();
        CopyInPartialCombCompact(stage1UsedCoreNum, workspaceSize1, mmLastAxisSize,
            globalRowOffset, curRowFactor);
        mixesQue2.EnQue(mixes2Local);
        mixes2Local = mixesQue2.DeQue<float>();
        combFragLocal = combFragQue.AllocTensor<float>();

        ReduceSumARAPerf(combFragLocal, mixes2Local, 1, stage1UsedCoreNum,
            curRowFactor * COMPACT_MATRIX_SIZE);
        GatherCompactToElementMajor(mixes02ReduceLocal, combFragLocal, curRowFactor);
        ProcessCompactSinkhorn(mixes02ReduceLocal, curRowFactor, hcScale2);
        GatherElementMajorToCompact(combFragLocal, mixes02ReduceLocal, curRowFactor);
        mixesQue2.FreeTensor(mixes2Local);
        squareSumQue.template FreeTensor(squareSumOutLocal);

        combFragQue.EnQue(combFragLocal);
        combFragLocal = combFragQue.DeQue<float>();
        CopyOut(combFragLocal,
            combFragGm[globalRowOffset * COMPACT_MATRIX_SIZE],
            1, curRowFactor * COMPACT_MATRIX_SIZE);
        combFragQue.FreeTensor(combFragLocal);
    }

    __aicore__ inline void ProcessSinkhorn(int64_t stage1UsedCoreNum, int64_t workspaceSize1,
                                          int64_t workspaceSize2, int64_t mmLastAxisSize)
    {
        int64_t blockIdx = GetBlockIdx() - tilingData->prePostCoreNum;
        if (blockIdx < 0 || blockIdx >= tilingData->sinkhornUsedCoreNum) {
            return;
        }
        CopyIn(hcBaseGm[COMPACT_HC_MULT * NUM_TWO], hcBase2Local,
            1, COMPACT_MATRIX_SIZE);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        float hcScale2 = hcScaleGm.GetValue(NUM_TWO);

        int64_t rowOuterLoop = blockIdx == tilingData->sinkhornUsedCoreNum - 1 ?
            tilingData->sinkhornRowLoopOfTailBlock : tilingData->sinkhornRowLoopOfFormerBlock;
        int64_t tailRowFactor = blockIdx == tilingData->sinkhornUsedCoreNum - 1 ?
            tilingData->sinkhornTailRowFactorOfTailBlock : tilingData->sinkhornTailRowFactorOfFormerBlock;

        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = rowOuterIdx == rowOuterLoop - 1 ?
                tailRowFactor : tilingData->sinkhornRowFactor;
            int64_t globalRowOffset = blockIdx * tilingData->sinkhornRowOfFormerBlock +
                rowOuterIdx * tilingData->sinkhornRowFactor;
            ProcessSinkhornBatch(stage1UsedCoreNum, workspaceSize1, workspaceSize2,
                mmLastAxisSize, globalRowOffset, curRowFactor, hcScale2);
        }
    }

    __aicore__ inline void ProcessOriginal()
    {
        if ASCEND_IS_AIC {
            return;
        }
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx >= tilingData->secondUsedCoreNum) {
            return;
        }

        int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
        int64_t mmLastAxisSize = CeilAlign(tilingData->hcMix, MM_CACHE_LINE_BYTES / sizeof(float));
        int64_t xCastFp32BufSize = tilingData->mL1Size *
            CeilAlign(tilingData->cvLoopKSize, MM_CACHE_LINE_BYTES / sizeof(float));
        int64_t workspaceSize1 = tilingData->cubeCoreNum * DOUBLE_BUFFER * xCastFp32BufSize;
        int64_t workspaceSize2 = CeilAlign(stage1UsedCoreNum * tilingData->bs *
            mmLastAxisSize * sizeof(float), WORKSPACE_ALIGN_SIZE) / sizeof(float);

        CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
        CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
        CopyIn(hcBaseGm[tilingData->hcMult * NUM_TWO], hcBase2Local,
            tilingData->hcMult, tilingData->hcMult);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        float hcScale0 = hcScaleGm.GetValue(0);
        float hcScale1 = hcScaleGm.GetValue(1);
        float hcScale2 = hcScaleGm.GetValue(NUM_TWO);

        int64_t rowOuterLoop = blockIdx == tilingData->secondUsedCoreNum - 1 ?
            tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        int64_t tailRowFactor = blockIdx == tilingData->secondUsedCoreNum - 1 ?
            tilingData->tailRowFactorOfTailBlock : tilingData->tailRowFactorOfFormerBlock;

        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = rowOuterIdx == rowOuterLoop - 1 ?
                tailRowFactor : tilingData->stage2RowFactor;
            int64_t globalRowOffset = blockIdx * tilingData->rowOfFormerBlock +
                rowOuterIdx * tilingData->stage2RowFactor;
            CalcRsqrt(stage1UsedCoreNum, globalRowOffset, curRowFactor,
                workspaceSize1, workspaceSize2);

            mixes01Local = mixesQue01.AllocTensor<float>();
            uint64_t mixBaseOffset = workspaceSize1 + globalRowOffset * mmLastAxisSize;
            CopyInWithOuterFor(workspaceGm[mixBaseOffset], mixes01Local,
                stage1UsedCoreNum, curRowFactor, tilingData->hcMult,
                tilingData->bs, mmLastAxisSize);
            CopyInWithOuterFor(workspaceGm[mixBaseOffset + tilingData->hcMult],
                mixes01Local[stage1UsedCoreNum * tilingData->stage2RowFactor * tilingData->hcMultAlign],
                stage1UsedCoreNum, curRowFactor, tilingData->hcMult,
                tilingData->bs, mmLastAxisSize);
            mixesQue01.EnQue(mixes01Local);
            mixes01Local = mixesQue01.DeQue<float>();
            ReduceSumARAPerf(mixes01ReduceLocal, mixes01Local, NUM_TWO,
                stage1UsedCoreNum, curRowFactor * tilingData->hcMultAlign);

            ProcessPre(mixes01ReduceLocal, mixes01ReduceLocal, hcBase0Local,
                rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1, hcScale0,
                tilingData->hcEps, curRowFactor, tilingData->hcMult);
            for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                int64_t curDFactor = dLoopIdx == tilingData->dLoop - 1 ?
                    tilingData->tailDFactor : tilingData->dFactor;
                xLocal = xQue.template AllocTensor<T>();
                CopyIn(xGm[globalRowOffset * tilingData->hcMult * tilingData->d +
                    dLoopIdx * tilingData->dFactor], xLocal,
                    curRowFactor * tilingData->hcMult, curDFactor,
                    tilingData->d - curDFactor);
                xQue.template EnQue(xLocal);
                xLocal = xQue.template DeQue<T>();
                yLocal = yQue.template AllocTensor<T>();
                ProcessY(yLocal, xLocal, mixes01ReduceLocal, hcBrcbLocal1,
                    xCastLocal, yCastLocal, curRowFactor, tilingData->hcMult, curDFactor);
                xQue.template FreeTensor(xLocal);
                yQue.template EnQue(yLocal);
                yLocal = yQue.template DeQue<T>();
                CopyOut(yLocal,
                    yGm[globalRowOffset * tilingData->d + dLoopIdx * tilingData->dFactor],
                    curRowFactor, curDFactor, tilingData->d - curDFactor);
                yQue.template FreeTensor(yLocal);
            }

            postLocal = postQue.AllocTensor<float>();
            ProcessPost(postLocal,
                mixes01ReduceLocal[curRowFactor * tilingData->hcMultAlign],
                hcBase1Local, rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1,
                hcScale1, curRowFactor, tilingData->hcMult);
            mixesQue01.template FreeTensor(mixes01Local);
            postQue.EnQue(postLocal);
            postLocal = postQue.DeQue<float>();
            CopyOut(postLocal, postGm[globalRowOffset * tilingData->hcMult],
                curRowFactor, tilingData->hcMult);
            postQue.FreeTensor(postLocal);

            mixes2Local = mixesQue2.AllocTensor<float>();
            for (int64_t kBlockIdx = 0; kBlockIdx < stage1UsedCoreNum; kBlockIdx++) {
                for (int64_t rowIdx = 0; rowIdx < curRowFactor; rowIdx++) {
                    int64_t gmOffset = workspaceSize1 +
                        kBlockIdx * tilingData->bs * mmLastAxisSize +
                        (globalRowOffset + rowIdx) * mmLastAxisSize + tilingData->hcMult * NUM_TWO;
                    int64_t ubOffset = (kBlockIdx * curRowFactor + rowIdx) *
                        tilingData->hcMult * tilingData->hcMultAlign;
                    CopyIn(workspaceGm[gmOffset], mixes2Local[ubOffset],
                        tilingData->hcMult, tilingData->hcMult);
                }
            }
            mixesQue2.EnQue(mixes2Local);
            mixes2Local = mixesQue2.DeQue<float>();
            ReduceSumARAPerf(mixes02ReduceLocal, mixes2Local, 1,
                stage1UsedCoreNum,
                curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);
            combFragLocal = combFragQue.AllocTensor<float>();

            MulABLastDimBrcInline<float, false>(mixes02ReduceLocal,
                mixes02ReduceLocal, rsqrtLocal, rowBrcbLocal0, curRowFactor,
                tilingData->hcMult * tilingData->hcMultAlign);
            Muls(mixes02ReduceLocal, mixes02ReduceLocal, hcScale2,
                curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);
            PipeBarrier<PIPE_V>();
            AddBAFirstDimBrcInline<float>(mixes02ReduceLocal, mixes02ReduceLocal,
                hcBase2Local, curRowFactor,
                tilingData->hcMult * tilingData->hcMultAlign);
            SoftmaxFP32Perf(mixes02ReduceLocal, mixes02ReduceLocal, reduceLocal,
                hcBrcbLocal1, curRowFactor * tilingData->hcMult,
                tilingData->hcMult, tilingData->hcEps);
            ReduceSumARAPerf(reduceLocal, mixes02ReduceLocal, curRowFactor,
                tilingData->hcMult, tilingData->hcMult);
            Adds(reduceLocal, reduceLocal, tilingData->hcEps,
                curRowFactor * tilingData->hcMult);
            PipeBarrier<PIPE_V>();
            DivABABrcInline(combFragLocal, mixes02ReduceLocal, reduceLocal,
                curRowFactor, tilingData->hcMult, tilingData->hcMult);
            for (int64_t iter = 0; iter < tilingData->iterTimes - 1; iter++) {
                LastDimReduceSumPerf(reduceLocal, combFragLocal,
                    curRowFactor * tilingData->hcMult, tilingData->hcMult);
                Adds(reduceLocal, reduceLocal, tilingData->hcEps,
                    curRowFactor * tilingData->hcMult);
                PipeBarrier<PIPE_V>();
                DivABLastDimBrcInline<float, true>(combFragLocal, combFragLocal,
                    reduceLocal, hcBrcbLocal1, curRowFactor * tilingData->hcMult,
                    tilingData->hcMult);
                ReduceSumARAPerf(reduceLocal, combFragLocal, curRowFactor,
                    tilingData->hcMult, tilingData->hcMult);
                Adds(reduceLocal, reduceLocal, tilingData->hcEps,
                    curRowFactor * tilingData->hcMult);
                PipeBarrier<PIPE_V>();
                DivABABrcInline(combFragLocal, combFragLocal, reduceLocal,
                    curRowFactor, tilingData->hcMult, tilingData->hcMult);
            }
            mixesQue2.FreeTensor(mixes2Local);
            squareSumQue.template FreeTensor(squareSumOutLocal);

            combFragQue.EnQue(combFragLocal);
            combFragLocal = combFragQue.DeQue<float>();
            CopyOut(combFragLocal,
                combFragGm[globalRowOffset * tilingData->hcMult * tilingData->hcMult],
                curRowFactor * tilingData->hcMult, tilingData->hcMult);
            combFragQue.FreeTensor(combFragLocal);
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (useOriginal) {
            ProcessOriginal();
            return;
        }
        if ASCEND_IS_AIC {
            return;
        }
        int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
        int64_t mmLastAxisSize = CeilAlign(tilingData->hcMix, MM_CACHE_LINE_BYTES / sizeof(float));
        int64_t xCastFp32BufSize = tilingData->mL1Size *
            CeilAlign(tilingData->cvLoopKSize, MM_CACHE_LINE_BYTES / sizeof(float));
        int64_t workspaceSize1 = tilingData->cubeCoreNum * DOUBLE_BUFFER * xCastFp32BufSize;
        int64_t workspaceSize2 = CeilAlign(stage1UsedCoreNum * tilingData->bs *
            mmLastAxisSize * sizeof(float), WORKSPACE_ALIGN_SIZE) / sizeof(float);
        if (GetBlockIdx() < tilingData->prePostCoreNum) {
            ProcessPrePost(stage1UsedCoreNum, workspaceSize1, workspaceSize2, mmLastAxisSize);
        } else {
            ProcessSinkhorn(stage1UsedCoreNum, workspaceSize1, workspaceSize2, mmLastAxisSize);
        }
    }

private:
    static constexpr int64_t COMPACT_HC_MULT = 4;
    static constexpr int64_t COMPACT_MATRIX_SIZE = COMPACT_HC_MULT * COMPACT_HC_MULT;

    TPipe *pipe;
    const HcPreTilingData *tilingData;
    GlobalTensor<float> mixesGm;
    GlobalTensor<float> rsqrtGm;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<float> workspaceGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;

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
    TBuf<QuePosition::VECCALC> mixes01ReduceBuf;
    TBuf<QuePosition::VECCALC> mixes02ReduceBuf;

    TBuf<QuePosition::VECCALC> xCastBuf;
    TBuf<QuePosition::VECCALC> yCastBuf;
    TBuf<QuePosition::VECCALC> maskPatternBuf;

    LocalTensor<float> mixes01Local;
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
    LocalTensor<float> mixes01ReduceLocal;
    LocalTensor<float> mixes02ReduceLocal;
    LocalTensor<float> xCastLocal;
    LocalTensor<float> yCastLocal;
    LocalTensor<float> squareSumOutLocal;
    LocalTensor<uint32_t> maskPatternLocal;
};

// preMix 场景专用的 Part2 实现，对应 tilingKey=2。
template <typename T>
class HcPreMembaseKSplitCorePremixPart2 {
public:
    __aicore__ inline HcPreMembaseKSplitCorePremixPart2()
    {
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR x, GM_ADDR hcScale,
        GM_ADDR hcBase, GM_ADDR y, GM_ADDR post, GM_ADDR combFrag,
        GM_ADDR preMix, GM_ADDR pre, GM_ADDR workspace)
    {
        xGm.SetGlobalBuffer((__gm__ T *)x);
        hcScaleGm.SetGlobalBuffer((__gm__ float *)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float *)hcBase);
        yGm.SetGlobalBuffer((__gm__ T *)y);
        postGm.SetGlobalBuffer((__gm__ float *)post);
        combFragGm.SetGlobalBuffer((__gm__ float *)combFrag);
        preMixGm.SetGlobalBuffer((__gm__ float *)preMix);
        preGm.SetGlobalBuffer((__gm__ float *)pre);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace);
    }

    __aicore__ inline void InitQueBuffers(int64_t stage1UsedCoreNum,
        int64_t xQueNum2)
    {
        int64_t mixesQue01Size = stage1UsedCoreNum * tilingData->stage2RowFactor *
        tilingData->hcMultAlign * NUM_TWO * sizeof(float);
        pipe->InitBuffer(mixesQue01, NUM_TWO, mixesQue01Size);
        pipe->InitBuffer(mixesQue2, NUM_TWO,
            stage1UsedCoreNum * tilingData->stage2RowFactor *
            tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(squareSumQue, NUM_TWO, stage1UsedCoreNum *
        tilingData->stage2RowFactor * SQUARE_SUM_SIZE * sizeof(float));
        pipe->InitBuffer(xQue, NUM_TWO, xQueNum2 * sizeof(T));
        pipe->InitBuffer(yQue, NUM_TWO,
            tilingData->stage2RowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        pipe->InitBuffer(postQue, NUM_TWO,
            tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(combFragQue, NUM_TWO,
            tilingData->stage2RowFactor * tilingData->hcMult *
            tilingData->hcMultAlign * sizeof(float));
    }

    __aicore__ inline void InitTBufBuffers(int64_t xQueNum2)
    {
        pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf2,
            tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(rowBrcbBuf0,
            RoundUp<float>(tilingData->stage2RowFactor) * BLOCK_SIZE);
        pipe->InitBuffer(hcBrcbBuf1,
            RoundUp<float>(tilingData->stage2RowFactor *
            tilingData->hcMultAlign) * BLOCK_SIZE);
        pipe->InitBuffer(reduceBuf,
            tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(mixes01ReduceBuf, tilingData->stage2RowFactor *
            tilingData->hcMultAlign * NUM_TWO * sizeof(float));
        pipe->InitBuffer(mixes02ReduceBuf, tilingData->stage2RowFactor *
            tilingData->hcMultAlign * tilingData->hcMult * sizeof(float));
        pipe->InitBuffer(squareReduceBuf,
            tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(xCastBuf, xQueNum2 * sizeof(float));
        pipe->InitBuffer(yCastBuf, tilingData->stage2RowFactor *
            RoundUp<T>(tilingData->dFactor) * sizeof(float));
        pipe->InitBuffer(rsqrtBuf,
            RoundUp<float>(tilingData->stage2RowFactor) * sizeof(float));
        pipe->InitBuffer(maskPatternBuf,
            RoundUp<uint32_t>(MASK_PATTERN_DIM_SIZE * MASK_PATTERN_REPEAT_SIZE) *
            sizeof(uint32_t));
        pipe->InitBuffer(preMixBuf,
            tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
    }

    __aicore__ inline void GetLocalTensors()
    {
        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        hcBase2Local = hcBaseBuf2.Get<float>();
        rowBrcbLocal0 = rowBrcbBuf0.Get<float>();
        hcBrcbLocal1 = hcBrcbBuf1.Get<float>();
        reduceLocal = reduceBuf.Get<float>();
        squareReduceLocal = squareReduceBuf.Get<float>();
        mixes01ReduceLocal = mixes01ReduceBuf.Get<float>();
        mixes02ReduceLocal = mixes02ReduceBuf.Get<float>();
        xCastLocal = xCastBuf.Get<float>();
        yCastLocal = yCastBuf.Get<float>();
        rsqrtLocal = rsqrtBuf.Get<float>();
        maskPatternLocal = maskPatternBuf.Get<uint32_t>();
        preMixLocal = preMixBuf.Get<float>();
        SetGatherMaskPattern(maskPatternLocal, static_cast<uint16_t>(MASK_PATTERN_DIM_SIZE));
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcScale, GM_ADDR hcBase,
        GM_ADDR y, GM_ADDR post, GM_ADDR combFrag, GM_ADDR preMix, GM_ADDR pre,
        GM_ADDR workspace, const HcPreTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        InitGlobalBuffers(x, hcScale, hcBase, y, post, combFrag, preMix, pre, workspace);
        int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
        int64_t xQueNum2 = tilingData->stage2RowFactor * tilingData->hcMult *
        RoundUp<T>(tilingData->dFactor);
        InitQueBuffers(stage1UsedCoreNum, xQueNum2);
        InitTBufBuffers(xQueNum2);
        GetLocalTensors();
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
            int64_t stage2BlockIdx = GetBlockIdx();
            int64_t stage2UsedCoreNum = tilingData->secondUsedCoreNum;
            if (stage2BlockIdx >= stage2UsedCoreNum) {
                return;
            }
            int64_t mmLastAxisSize = CeilAlign(tilingData->hcMix, MM_CACHE_LINE_BYTES / sizeof(float));
            int64_t xCastFp32BufSize = tilingData->mL1Size *
            CeilAlign(tilingData->cvLoopKSize, MM_CACHE_LINE_BYTES / sizeof(float));
            int64_t workspaceSize1 = tilingData->cubeCoreNum * DOUBLE_BUFFER * xCastFp32BufSize;
            int64_t workspaceSize2 = CeilAlign(stage1UsedCoreNum * tilingData->bs *
            mmLastAxisSize * sizeof(float), WORKSPACE_ALIGN_SIZE) / sizeof(float);
            CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult * NUM_TWO], hcBase2Local, tilingData->hcMult, tilingData->hcMult);
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);

            int64_t rowOuterLoop =
                (stage2BlockIdx == stage2UsedCoreNum - 1) ?
                tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
            int64_t tailRowFactor = (stage2BlockIdx == stage2UsedCoreNum - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                        tilingData->tailRowFactorOfFormerBlock;
            int64_t xGmBlockBaseOffsetPart2 = stage2BlockIdx *
            tilingData->rowOfFormerBlock * tilingData->hcMult * tilingData->d;

            for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
                int64_t xGmBsBaseOffsetPart2 = rowOuterIdx * tilingData->stage2RowFactor *
                tilingData->hcMult * tilingData->d;
                int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->stage2RowFactor;
                squareSumOutLocal = squareSumQue.AllocTensor<float>();
                CopyIn(workspaceGm[workspaceSize1 + workspaceSize2 +
                       stage2BlockIdx * tilingData->rowOfFormerBlock * SQUARE_SUM_SIZE +
                       rowOuterIdx * tilingData->stage2RowFactor * SQUARE_SUM_SIZE],
                       squareSumOutLocal, stage1UsedCoreNum, curRowFactor * SQUARE_SUM_SIZE,
                       CeilAlign(tilingData->bs, SQUARE_SUM_SIZE) * SQUARE_SUM_SIZE -
                       curRowFactor * SQUARE_SUM_SIZE);
                squareSumQue.EnQue(squareSumOutLocal);
                squareSumOutLocal = squareSumQue.DeQue<float>();
                ReduceSumARAPerf(squareReduceLocal, squareSumOutLocal, 1, stage1UsedCoreNum,
                curRowFactor * SQUARE_SUM_SIZE);
                int64_t curBsIdxForAll = (stage2BlockIdx * tilingData->rowLoopOfFormerBlock +
                rowOuterIdx) * tilingData->stage2RowFactor;
                GatherMaskByDiagonal(rsqrtLocal, squareReduceLocal, maskPatternLocal,
                curRowFactor, static_cast<uint16_t>(curBsIdxForAll % SQUARE_SUM_SIZE));
                float coeff = 1.0f / static_cast<float>(tilingData->k);
                Muls(rsqrtLocal, rsqrtLocal, coeff, curRowFactor);
                PipeBarrier<PIPE_V>();
                Adds(rsqrtLocal, rsqrtLocal, tilingData->normEps, curRowFactor);
                PipeBarrier<PIPE_V>();
                Sqrt(rsqrtLocal, rsqrtLocal, curRowFactor);
                Duplicate(rowBrcbLocal0, static_cast<float>(1.0f), curRowFactor);
                PipeBarrier<PIPE_V>();
                Div(rsqrtLocal, rowBrcbLocal0, rsqrtLocal, curRowFactor);

                mixes01Local = mixesQue01.AllocTensor<float>();

                uint64_t mixBaseOffset = workspaceSize1 +
                stage2BlockIdx * tilingData->rowOfFormerBlock *
                CeilAlign(tilingData->hcMix, WORKSPACE_ALIGN_SIZE / sizeof(float)) +
                rowOuterIdx * tilingData->stage2RowFactor *
                CeilAlign(tilingData->hcMix, WORKSPACE_ALIGN_SIZE / sizeof(float));
                CopyInWithOuterFor(workspaceGm[mixBaseOffset], mixes01Local, stage1UsedCoreNum,
                curRowFactor, tilingData->hcMult, tilingData->bs,
                CeilAlign(tilingData->hcMix, WORKSPACE_ALIGN_SIZE / sizeof(float)));
                CopyInWithOuterFor(workspaceGm[mixBaseOffset + tilingData->hcMult],
                mixes01Local[stage1UsedCoreNum * tilingData->stage2RowFactor *
                tilingData->hcMultAlign], stage1UsedCoreNum, curRowFactor, tilingData->hcMult,
                tilingData->bs,
                CeilAlign(tilingData->hcMix, WORKSPACE_ALIGN_SIZE / sizeof(float)));

                mixesQue01.EnQue(mixes01Local);
                mixes01Local = mixesQue01.DeQue<float>();
                ReduceSumARAPerf(mixes01ReduceLocal, mixes01Local, NUM_TWO, stage1UsedCoreNum,
                curRowFactor * tilingData->hcMultAlign);
                ProcessPre(mixes01ReduceLocal, mixes01ReduceLocal, hcBase0Local, rsqrtLocal,
                rowBrcbLocal0, hcBrcbLocal1, hcScaleGm.GetValue(0), tilingData->hcEps,
                curRowFactor, tilingData->hcMult);
                if (tilingData->hasPreOut != 0) {
                    event_t eventIdPreOut = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(eventIdPreOut);
                    WaitFlag<HardEvent::V_MTE3>(eventIdPreOut);
                    CopyOut(mixes01ReduceLocal,
                        preGm[stage2BlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult +
                        rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMult],
                        1, curRowFactor * tilingData->hcMult);
                }
                if (tilingData->hasPreMix != 0) {
                    CopyIn(preMixGm[stage2BlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult +
                        rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMult],
                        preMixLocal, 1, curRowFactor * tilingData->hcMult);
                    event_t eventIdPreMix = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    SetFlag<HardEvent::MTE2_V>(eventIdPreMix);
                    WaitFlag<HardEvent::MTE2_V>(eventIdPreMix);
                }
                for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                    int64_t curDFactor =
                        (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                    xLocal = xQue.template AllocTensor<T>();
                    CopyIn(xGm[xGmBlockBaseOffsetPart2 + xGmBsBaseOffsetPart2 +
                    dLoopIdx * tilingData->dFactor], xLocal,
                    tilingData->stage2RowFactor * tilingData->hcMult, curDFactor,
                    tilingData->d - curDFactor);
                    xQue.template EnQue(xLocal);
                    xLocal = xQue.template DeQue<T>();
                    yLocal = yQue.template AllocTensor<T>();
                    ProcessY(yLocal, xLocal, tilingData->hasPreMix != 0 ? preMixLocal : mixes01ReduceLocal,
                                hcBrcbLocal1, xCastLocal, yCastLocal, curRowFactor,
                                tilingData->hcMult, curDFactor);
                    xQue.template FreeTensor(xLocal);
                    yQue.template EnQue(yLocal);
                    yLocal = yQue.template DeQue<T>();
                    CopyOut(yLocal,
                        yGm[stage2BlockIdx * tilingData->rowOfFormerBlock * tilingData->d +
                        rowOuterIdx * tilingData->stage2RowFactor * tilingData->d +
                        dLoopIdx * tilingData->dFactor],
                        curRowFactor, curDFactor, tilingData->d - curDFactor);
                    yQue.template FreeTensor(yLocal);
                }
                // post
                postLocal = postQue.AllocTensor<float>();
                ProcessPost(postLocal,
                mixes01ReduceLocal[tilingData->stage2RowFactor * tilingData->hcMultAlign],
                hcBase1Local, rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1, hcScaleGm.GetValue(1),
                curRowFactor, tilingData->hcMult);
                mixesQue01.template FreeTensor(mixes01Local);
                postQue.EnQue(postLocal);
                postLocal = postQue.DeQue<float>();
                CopyOut(postLocal,
                        postGm[stage2BlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult +
                        rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMult],
                        curRowFactor, tilingData->hcMult);
                postQue.FreeTensor(postLocal);

                // combFrag
                mixes2Local = mixesQue2.AllocTensor<float>();
                for (int64_t i = 0; i < stage1UsedCoreNum; ++i) {
                    for (int64_t j = 0; j < curRowFactor; ++j) {
                        CopyIn(workspaceGm[workspaceSize1 + i * tilingData->bs * mmLastAxisSize +
                        j * mmLastAxisSize +
                        stage2BlockIdx * tilingData->rowOfFormerBlock * mmLastAxisSize +
                        rowOuterIdx * tilingData->stage2RowFactor * mmLastAxisSize +
                        tilingData->hcMult * NUM_TWO],
                        mixes2Local[(i * curRowFactor + j) * tilingData->hcMult *
                        tilingData->hcMultAlign], tilingData->hcMult, tilingData->hcMult);
                    }
                }
                mixesQue2.EnQue(mixes2Local);
                mixes2Local = mixesQue2.DeQue<float>();
                ReduceSumARAPerf(mixes02ReduceLocal, mixes2Local, 1, stage1UsedCoreNum,
                curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);
                combFragLocal = combFragQue.AllocTensor<float>();

                MulABLastDimBrcInline<float, false>(mixes02ReduceLocal, mixes02ReduceLocal,
                rsqrtLocal, rowBrcbLocal0, curRowFactor,
                tilingData->hcMult * tilingData->hcMultAlign);
                Muls(mixes02ReduceLocal, mixes02ReduceLocal, hcScaleGm.GetValue(NUM_TWO),
                        curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);
                PipeBarrier<PIPE_V>();
                AddBAFirstDimBrcInline<float>(mixes02ReduceLocal, mixes02ReduceLocal, hcBase2Local, curRowFactor,
                                                tilingData->hcMult * tilingData->hcMultAlign);
                SoftmaxFP32Perf(mixes02ReduceLocal, mixes02ReduceLocal, reduceLocal, hcBrcbLocal1,
                curRowFactor * tilingData->hcMult, tilingData->hcMult, tilingData->hcEps);
                ReduceSumARAPerf(reduceLocal, mixes02ReduceLocal, curRowFactor, tilingData->hcMult, tilingData->hcMult);
                Adds(reduceLocal, reduceLocal, tilingData->hcEps, curRowFactor * tilingData->hcMult);
                PipeBarrier<PIPE_V>();
                DivABABrcInline(combFragLocal, mixes02ReduceLocal, reduceLocal, curRowFactor, tilingData->hcMult,
                                tilingData->hcMult);
                for (int64_t iter = 0; iter < tilingData->iterTimes - 1; iter++) {
                    LastDimReduceSumPerf(reduceLocal, combFragLocal,
                    curRowFactor * tilingData->hcMult, tilingData->hcMult);
                    Adds(reduceLocal, reduceLocal, tilingData->hcEps,
                    curRowFactor * tilingData->hcMult);
                    PipeBarrier<PIPE_V>();
                    DivABLastDimBrcInline<float, true>(combFragLocal, combFragLocal,
                    reduceLocal, hcBrcbLocal1, curRowFactor * tilingData->hcMult,
                    tilingData->hcMult);
                    ReduceSumARAPerf(reduceLocal, combFragLocal, curRowFactor,
                    tilingData->hcMult, tilingData->hcMult);
                    Adds(reduceLocal, reduceLocal, tilingData->hcEps,
                    curRowFactor * tilingData->hcMult);
                    PipeBarrier<PIPE_V>();
                    DivABABrcInline(combFragLocal, combFragLocal, reduceLocal,
                    curRowFactor, tilingData->hcMult, tilingData->hcMult);
                }
                mixesQue2.FreeTensor(mixes2Local);
                squareSumQue.template FreeTensor(squareSumOutLocal);

                combFragQue.EnQue(combFragLocal);
                combFragLocal = combFragQue.DeQue<float>();
                CopyOut(combFragLocal,
                        combFragGm[stage2BlockIdx * tilingData->rowOfFormerBlock *
                        tilingData->hcMult * tilingData->hcMult +
                        rowOuterIdx * tilingData->stage2RowFactor * tilingData->hcMult *
                        tilingData->hcMult], curRowFactor * tilingData->hcMult,
                        tilingData->hcMult);
                combFragQue.FreeTensor(combFragLocal);
            }
        }
    }

private:
    TPipe *pipe;
    const HcPreTilingData *tilingData;
    GlobalTensor<float> mixesGm;
    GlobalTensor<float> rsqrtGm;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<float> workspaceGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;
    GlobalTensor<float> preMixGm;
    GlobalTensor<float> preGm;

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
    TBuf<QuePosition::VECCALC> mixes01ReduceBuf;
    TBuf<QuePosition::VECCALC> mixes02ReduceBuf;

    TBuf<QuePosition::VECCALC> xCastBuf;
    TBuf<QuePosition::VECCALC> yCastBuf;
    TBuf<QuePosition::VECCALC> maskPatternBuf;
    TBuf<QuePosition::VECCALC> preMixBuf;

    LocalTensor<float> mixes01Local;
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
    LocalTensor<float> mixes01ReduceLocal;
    LocalTensor<float> mixes02ReduceLocal;
    LocalTensor<float> xCastLocal;
    LocalTensor<float> yCastLocal;
    LocalTensor<float> squareSumOutLocal;
    LocalTensor<uint32_t> maskPatternLocal;
    LocalTensor<float> preMixLocal;
};

} // namespace HcPreSinkhorn

#endif
