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

#ifndef MHC_PRE_SINKHORN_M_SPLIT_A3_CORE_H
#define MHC_PRE_SINKHORN_M_SPLIT_A3_CORE_H

#include "kernel_operator.h"
#include "mhc_pre_clamp_sinkhorn_base.h"
#include "mhc_pre_clamp_sinkhorn_cube_compute.h"

namespace MhcPreClampSinkhorn {
using namespace AscendC;
template <typename T>
class MhcPreClampSinkhornStage1 {
public:
    __aicore__ inline MhcPreClampSinkhornStage1() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR invRms, GM_ADDR hcBeforeNorm,
                                GM_ADDR workspace, const MhcPreClampSinkhornTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        needGrad_ = tilingData->needGrad;

        // 获取当前核心 ID
        uint64_t blockIdx = GetBlockIdx();
        uint64_t cubeCoreId = blockIdx;
        uint64_t vecCoreId = blockIdx;

        // 计算 workspace 偏移
        int64_t ncSize = tilingData->hcMult * tilingData->d;

        // 计算 Cube Core ID（Vector Core ID / 2）
        if ASCEND_IS_AIV {
            cubeCoreId = blockIdx / CV_RATIO;
            int64_t maxBsPerLoop = 32;
            int64_t xQueSize = maxBsPerLoop * tilingData->stage1NcFactor;
            pipe->InitBuffer(xQue, NUM_TWO, xQueSize * sizeof(T));
            pipe->InitBuffer(xCastQue, NUM_TWO, xQueSize * sizeof(float));
            pipe->InitBuffer(tmpQue, maxBsPerLoop * tilingData->stage1NcFactor * sizeof(float));
            pipe->InitBuffer(sumQue, maxBsPerLoop * sizeof(float));
            LocalTensor<float> sumLocal = sumQue.Get<float>();
            pipe->InitBuffer(invRmsQue, NUM_TWO, maxBsPerLoop * sizeof(float));
        }

        // 设置 GM 地址
        xGm.SetGlobalBuffer((__gm__ T *)x);
        phiGm.SetGlobalBuffer((__gm__ float *)phi);
        xCastWsGm.SetGlobalBuffer((__gm__ float *)workspace);

        // 设置额外的 workspace 地址（needGrad=false 时使用）
        if (needGrad_) {
            invRmsGm.SetGlobalBuffer((__gm__ float *)invRms);
            hcBeforeNormGm.SetGlobalBuffer((__gm__ float *)hcBeforeNorm);
        } else {
            int64_t xCastWsTotalSize = tilingData->stage1XCastWsSize / sizeof(float);
            invRmsWsGm.SetGlobalBuffer((__gm__ float *)workspace + xCastWsTotalSize);
            int64_t invRmsWsTotalSize = tilingData->bs * sizeof(float) / sizeof(float);
            hcBeforeNormWsGm.SetGlobalBuffer((__gm__ float *)workspace + xCastWsTotalSize + invRmsWsTotalSize);
        }

        // 初始化 Cube Compute
        if ASCEND_IS_AIC {
            // 计算当前 Cube Core 的输出偏移
            int64_t mmOutSize = tilingData->hcMult * tilingData->hcMult + 2 * tilingData->hcMult;
            // 根据 needGrad 决定输出地址
            if (needGrad_) {
                cubeCompute_.Init(xCastWsGm, phiGm, hcBeforeNormGm,
                                 tilingData->stage1BsFactor, tilingData->hcMult, tilingData->d, tilingData->stage1VecCoreNum);
            } else {
                cubeCompute_.Init(xCastWsGm, phiGm, hcBeforeNormWsGm,
                                 tilingData->stage1BsFactor, tilingData->hcMult, tilingData->d, tilingData->stage1VecCoreNum);
            }
            return;
        }
    }

    __aicore__ inline void Process()
    {
        uint64_t blockIdx = GetBlockIdx();
        uint64_t vecCoreId = blockIdx;
        uint64_t cubeCoreId = blockIdx;

        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
        }

        // 循环处理 BS（每轮最多 32 个）
        // ASCEND_IS_AIC 直接根据总BS数，计算当前处理的BS数
        int32_t totalTasksAligned_ = AlignUp(tilingData->bs, tilingData->stage1VecCoreNum * tilingData->stage1BsFactor);

        // Cube Core 流程
        if ASCEND_IS_AIC {
            for (int32_t taskOffset = blockIdx * 2 * tilingData->stage1BsFactor; taskOffset < totalTasksAligned_;
                 taskOffset += tilingData->stage1CubeCoreNum * 2 * tilingData->stage1BsFactor) {
                int32_t tileTaskCount = min(static_cast<int32_t>(2 * tilingData->stage1BsFactor),
                                            static_cast<int32_t>(tilingData->bs - taskOffset));
                // 等待 Vector Core 完成 Cast
                CrossCoreWaitFlag(SYNC_AIV_TO_AIC_FLAG);
                // 执行矩阵乘
                if (tileTaskCount > 0) {
                    cubeCompute_.ProcessMatmulXPhi(taskOffset, tileTaskCount);
                }
                // 通知 Vector Core 完成
                CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
            }
        }

        if ASCEND_IS_AIV {
            int64_t ncSize = tilingData->hcMult * tilingData->d;
            for (int32_t taskOffset = blockIdx * tilingData->stage1BsFactor; taskOffset < totalTasksAligned_;
                 taskOffset += tilingData->stage1VecCoreNum * tilingData->stage1BsFactor) {
                CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
                int32_t curBs = min(static_cast<int32_t>(tilingData->stage1BsFactor),
                                            static_cast<int32_t>(tilingData->bs - taskOffset));
                if (curBs > 0) {
                    int64_t curBsOffset = taskOffset;
                    // 初始化 sum 为 0
                    LocalTensor<float> invRmsLocal = invRmsQue.AllocTensor<float>();
                    LocalTensor<float> squareLocal = tmpQue.Get<float>();
                    LocalTensor<float> sumLocal = sumQue.Get<float>();
                    Duplicate(sumLocal, 0.0f, curBs);

                    // 分段 ReduceSum
                    for (int64_t ncIdx = 0; ncIdx < tilingData->stage1NcLoop; ncIdx++) {
                        int64_t curNcSize = (ncIdx == tilingData->stage1NcLoop - 1)
                                            ? tilingData->stage1TailNcFactor
                                            : tilingData->stage1NcFactor;
                        int64_t ncOffset = ncIdx * tilingData->stage1NcFactor;
                        int64_t curNcSizeAlign = AlignUp(static_cast<uint64_t>(curNcSize), ELEMENTS_SIZE_PER_BLOCK);

                        // 1. CopyIn X (BF16)
                        LocalTensor<T> xLocal = xQue.AllocTensor<T>();
                        CopyIn(xGm[curBsOffset * ncSize + ncOffset], xLocal, curBs, curNcSize, ncSize - curNcSize);
                        xQue.EnQue(xLocal);

                        // 2. Cast X (BF16) → X_cast (FP32)
                        xLocal = xQue.DeQue<T>();
                        LocalTensor<float> xCastLocal = xCastQue.AllocTensor<float>();
                        Cast(xCastLocal, xLocal, RoundMode::CAST_NONE, curBs * curNcSizeAlign);
                        xQue.FreeTensor(xLocal);
                        xCastQue.EnQue(xCastLocal);

                        // 3. CopyOut X_cast to workspace
                        xCastLocal = xCastQue.DeQue<float>();
                        int64_t xCastoffset = blockIdx * tilingData->stage1BsFactor * ncSize + ping4vec * tilingData->stage1VecCoreNum * tilingData->stage1BsFactor *  ncSize;
                        CopyOut(xCastLocal, xCastWsGm[xCastoffset + ncOffset], curBs, curNcSize, ncSize - curNcSize);
                        xCastQue.FreeTensor(xCastLocal);

                        // 4. 计算 inv_rms
                        // 4.1 Square
                        PipeBarrier<PIPE_V>();
                        Mul(squareLocal, xCastLocal, xCastLocal, curBs * curNcSizeAlign);
                        PipeBarrier<PIPE_V>();

                        // 对当前段进行 ReduceSum
                        uint32_t srcShape[2] = {static_cast<uint32_t>(curBs), static_cast<uint32_t>(curNcSize)};
                        AscendC::ReduceSum<float, Pattern::Reduce::AR, true>(squareLocal, squareLocal, srcShape, true);
                        PipeBarrier<PIPE_V>();

                        // 累加到总和
                        Add(sumLocal, sumLocal, squareLocal, curBs);
                        PipeBarrier<PIPE_V>();
                    }
                    // 4.3 Compute inv_rms
                    // inv_rms = 1 / sqrt(sum / (n*c) + eps)
                    float invNc = 1.0f / static_cast<float>(ncSize);
                    Muls(invRmsLocal, sumLocal, invNc, curBs);
                    PipeBarrier<PIPE_V>();
                    Adds(invRmsLocal, invRmsLocal, tilingData->normEps, curBs);
                    PipeBarrier<PIPE_V>();
                    Sqrt(invRmsLocal, invRmsLocal, curBs);
                    PipeBarrier<PIPE_V>();
                    Duplicate(sumLocal, 1.0f, curBs);
                    PipeBarrier<PIPE_V>();
                    Div(invRmsLocal, sumLocal, invRmsLocal, curBs);
                    invRmsQue.EnQue(invRmsLocal);

                    // 5. CopyOut inv_rms to output
                    invRmsLocal = invRmsQue.DeQue<float>();
                    if (needGrad_) {
                        CopyOut(invRmsLocal, invRmsGm[curBsOffset], 1, curBs);
                    } else {
                        CopyOut(invRmsLocal, invRmsWsGm[curBsOffset], 1, curBs);
                    }
                    invRmsQue.FreeTensor(invRmsLocal);
                    ping4vec = 1 - ping4vec;
                }
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_TO_AIC_FLAG);
            }
            CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
            CrossCoreWaitFlag(SYNC_AIC_TO_AIV_FLAG);
        }
    }

public:
    TPipe *pipe;
    const MhcPreClampSinkhornTilingData *tilingData;

    // GM 地址
    GlobalTensor<T> xGm;                    // 输入 X (BF16)
    GlobalTensor<float> phiGm;              // phi 矩阵 (FP32)
    GlobalTensor<float> invRmsGm;           // inv_rms 输出 (FP32)
    GlobalTensor<float> hcBeforeNormGm;     // hcBeforeNorm 输出 (FP32)
    GlobalTensor<float> xCastWsGm;          // X_cast workspace
    GlobalTensor<float> invRmsWsGm;         // invRms workspace (needGrad=false)
    GlobalTensor<float> hcBeforeNormWsGm;   // hcBeforeNorm workspace (needGrad=false)

    // Queue
    TQue<QuePosition::VECIN, NUM_TWO> xQue;         // X 输入队列
    TQue<QuePosition::VECOUT, NUM_TWO> xCastQue;    // X_cast 输出队列
    TBuf<QuePosition::VECCALC> tmpQue;
    TBuf<QuePosition::VECCALC> sumAddQue;      // sum 队列
    TBuf<QuePosition::VECCALC> sumQue;         // sum 队列
    TQue<QuePosition::VECOUT, NUM_TWO> invRmsQue;   // inv_rms 队列

    // Cube Compute
    HcCubeCompute<float> cubeCompute_;

    // 标志
    bool needGrad_ = true;
    uint64_t ping4vec = 1;

    // 同步标志
    static constexpr uint64_t SYNC_AIV_TO_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_TO_AIV_FLAG = 9;
    static constexpr uint64_t SYNC_MODE2 = NUM_TWO;
    static constexpr uint64_t CV_RATIO = 2;
};

template <typename T>
class MhcPreClampSinkhornStage2 {
public:
    __aicore__ inline MhcPreClampSinkhornStage2()
    {
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR y,
                                GM_ADDR post, GM_ADDR combFrag,
                                GM_ADDR hPre, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
                                GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR hResLogits,
                                GM_ADDR workspace,
                                const MhcPreClampSinkhornTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        needGrad_ = tilingData->needGrad;

        xGm.SetGlobalBuffer((__gm__ T *)x);
        hcScaleGm.SetGlobalBuffer((__gm__ float *)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float *)hcBase);
        yGm.SetGlobalBuffer((__gm__ T *)y);
        postGm.SetGlobalBuffer((__gm__ float *)post);
        combFragGm.SetGlobalBuffer((__gm__ float *)combFrag);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace);

        if (needGrad_) {
            hPreGm.SetGlobalBuffer((__gm__ float *)hPre);
            hcBeforeNormGm.SetGlobalBuffer((__gm__ float *)hcBeforeNorm);
            invRmsGm.SetGlobalBuffer((__gm__ float *)invRms);
            sumOutGm.SetGlobalBuffer((__gm__ float *)sumOut);
            normOutGm.SetGlobalBuffer((__gm__ float *)normOut);
            hResLogitsGm.SetGlobalBuffer((__gm__ float *)hResLogits);
        } else {
            int64_t xCastWsTotalSize = tilingData->stage1XCastWsSize / sizeof(float);
            invRmsGm.SetGlobalBuffer((__gm__ float *)workspace + xCastWsTotalSize);
            int64_t invRmsWsTotalSize = tilingData->bs * sizeof(float) / sizeof(float);
            hcBeforeNormGm.SetGlobalBuffer((__gm__ float *)workspace + xCastWsTotalSize + invRmsWsTotalSize);
        }
        // InQue
        int64_t mixesQue01Size =  tilingData->stage2RowFactor * tilingData->hcMultAlign * NUM_TWO * sizeof(float);
        pipe->InitBuffer(mixesQue01, NUM_TWO, mixesQue01Size);
        pipe->InitBuffer(mixesQue2, NUM_TWO, tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(squareSumQue, NUM_TWO, tilingData->stage2RowFactor * SQUARE_SUM_SIZE * sizeof(float));
        int64_t xQueNum2 = tilingData->stage2RowFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor);
        pipe->InitBuffer(xQue, NUM_TWO, xQueNum2 * sizeof(T));
        // OutQue
        pipe->InitBuffer(yQue, NUM_TWO, tilingData->stage2RowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        pipe->InitBuffer(postQue, NUM_TWO, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(combFragQue, NUM_TWO,
                         tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(reduceQue, NUM_TWO, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        // TBuf
        pipe->InitBuffer(combFragBuf,
                         tilingData->stage2RowFactor * tilingData->hcMult * tilingData->hcMultAlign * NUM_TWO * sizeof(float));
        pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(rowBrcbBuf0, RoundUp<float>(tilingData->stage2RowFactor) * BLOCK_SIZE);
        pipe->InitBuffer(hcBrcbBuf1, RoundUp<float>(tilingData->stage2RowFactor * tilingData->hcMultAlign * NUM_TWO) * BLOCK_SIZE);
        pipe->InitBuffer(reduceBuf, tilingData->stage2RowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(mxies01ReduceBuf, tilingData->stage2RowFactor * tilingData->hcMultAlign * NUM_TWO * sizeof(float));
        pipe->InitBuffer(mxies02ReduceBuf, tilingData->stage2RowFactor * tilingData->hcMultAlign * tilingData->hcMult * sizeof(float));
        pipe->InitBuffer(xCastBuf, xQueNum2 * sizeof(float));
        pipe->InitBuffer(yCastBuf, tilingData->stage2RowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(float));

        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        hcBase2Local = hcBaseBuf2.Get<float>();
        rowBrcbLocal0 = rowBrcbBuf0.Get<float>();
        hcBrcbLocal1 = hcBrcbBuf1.Get<float>();
        reduceLocal = reduceBuf.Get<float>();
        mxies01ReduceLocal = mxies01ReduceBuf.Get<float>();
        mxies02ReduceLocal = mxies02ReduceBuf.Get<float>();
        xCastLocal = xCastBuf.Get<float>();
        yCastLocal = yCastBuf.Get<float>();
        combFragBufLocal = combFragBuf.Get<float>();
    }

    __aicore__ inline void Process(bool isTailBsLoop = false)
    {
        SyncAll();
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
            int64_t workspaceSize2 = CeilAlign(stage1UsedCoreNum * tilingData->bs * mmLastAxisSize * sizeof(float), WORKSPACE_ALIGN_SIZE) / sizeof(float);
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
            uint64_t mixBaseOffset = 0;
            for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
                int64_t xGmBsBaseOffsetPart2 = rowOuterIdx * curStage2RowFactor * tilingData->hcMult * tilingData->d;
                int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : curStage2RowFactor;

                squareSumOutLocal = squareSumQue.AllocTensor<float>();
                Duplicate(rowBrcbLocal0, static_cast<float>(1.0f), curRowFactor);

                CopyIn(invRmsGm[stage2BlockIdx * curRowOfFormerBlock + rowOuterIdx * curStage2RowFactor],
                        squareSumOutLocal, 1, curRowFactor);

                squareSumQue.EnQue(squareSumOutLocal);
                squareSumOutLocal = squareSumQue.DeQue<float>();
                // 搬运矩阵乘的前两段结果--> 内存格式的变更： 连续地址改为 bs *n ; bs *n ,n为4 非对齐，则需要pad出对齐的8
                mxies01Local = mixesQue01.AllocTensor<float>();

                mixBaseOffset = stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMix +
                                            rowOuterIdx * curStage2RowFactor * tilingData->hcMix;
                CopyInWithOuterFor(hcBeforeNormGm[mixBaseOffset], mxies01Local, 1, curRowFactor, tilingData->hcMult,
                                tilingData->bs, tilingData->hcMix);
                CopyInWithOuterFor(hcBeforeNormGm[mixBaseOffset + tilingData->hcMult],
                                mxies01Local[curRowFactor * tilingData->hcMultAlign],
                                1, curRowFactor, tilingData->hcMult, tilingData->bs,
                                tilingData->hcMix);
                // // wk:[2, kcorenum, bs, n^2 +2n]
                // // mx0[2,kcorenum, curRowfator, n]
                mixesQue01.EnQue(mxies01Local);
                mxies01Local = mixesQue01.DeQue<float>();

                ProcessPre(mxies01ReduceLocal, mxies01Local, hcBase0Local, squareSumOutLocal, rowBrcbLocal0, hcBrcbLocal1,
                            hcScaleGm.GetValue(0), tilingData->hcEps, curRowFactor, tilingData->hcMult);

                if (needGrad_) {
                    int64_t hPreBaseOffset = stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMult +
                                             rowOuterIdx * curStage2RowFactor * tilingData->hcMult;
                    VToMTE3Sync();
                    CopyOut(mxies01ReduceLocal, hPreGm[hPreBaseOffset], curRowFactor, tilingData->hcMult);
                    MTE3ToVSync();
                }
                // --- pre --
                for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                    int64_t curDFactor =
                        (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                    xLocal = xQue.template AllocTensor<T>();
                    CopyIn(xGm[xGmBlockBaseOffsetPart2 + xGmBsBaseOffsetPart2 +
                                dLoopIdx * tilingData->dFactor],
                            xLocal, curRowFactor * tilingData->hcMult, curDFactor, tilingData->d - curDFactor);
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
                ProcessPost(postLocal, mxies01Local[curRowFactor * tilingData->hcMultAlign], hcBase1Local,
                            squareSumOutLocal, rowBrcbLocal0, hcBrcbLocal1, hcScaleGm.GetValue(1), curRowFactor,
                            tilingData->hcMult);
                mixesQue01.FreeTensor(mxies01Local); // 这里对应的申请在上面
                postQue.EnQue(postLocal);
                postLocal = postQue.DeQue<float>();

                CopyOut(postLocal,
                        postGm[stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMult +
                                rowOuterIdx * curStage2RowFactor * tilingData->hcMult],
                        curRowFactor, tilingData->hcMult);
                postQue.FreeTensor(postLocal);

                // combFrag
                mixes2Local = mixesQue2.AllocTensor<float>();
                mixBaseOffset = stage2BlockIdx * curRowOfFormerBlock * tilingData->hcMix +
                                rowOuterIdx * curStage2RowFactor * tilingData->hcMix;
                for (int64_t j = 0; j < curRowFactor; ++j) {
                    CopyIn(hcBeforeNormGm[mixBaseOffset + j * tilingData->hcMix + tilingData->hcMult * NUM_TWO],
                           mixes2Local[j * tilingData->hcMult * tilingData->hcMultAlign], tilingData->hcMult,
                           tilingData->hcMult);
                }
                mixesQue2.EnQue(mixes2Local);
                mixes2Local = mixesQue2.DeQue<float>();

                combFragLocal = combFragQue.AllocTensor<float>();
                MulABLastDimBrcInline<float, false>(mxies02ReduceLocal, mixes2Local, rsqrtLocal, rowBrcbLocal0, curRowFactor,
                                                    tilingData->hcMult * tilingData->hcMultAlign);
                mixesQue2.FreeTensor(mixes2Local);
                Muls(mxies02ReduceLocal, mxies02ReduceLocal, hcScaleGm.GetValue(NUM_TWO),
                        curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);
                PipeBarrier<PIPE_V>();
                AddBAFirstDimBrcInline<float>(mxies02ReduceLocal, mxies02ReduceLocal, hcBase2Local, curRowFactor,
                                                tilingData->hcMult * tilingData->hcMultAlign);
                if (needGrad_) {
                    int64_t hResLogitsBaseOffset = (stage2BlockIdx * curRowOfFormerBlock +
                                                   rowOuterIdx * curStage2RowFactor) * tilingData->hcMult * tilingData->hcMult;
                    VToMTE3Sync();
                    CopyOut(mxies02ReduceLocal, hResLogitsGm[hResLogitsBaseOffset], curRowFactor * tilingData->hcMult, tilingData->hcMult);
                    MTE3ToVSync();
                }

                //* CLAMP Start
                if (tilingData->clamp_min != 0.0f || tilingData->clamp_max != 0.0f) {
                    // AscendC::printf("[clamped]\n");
                    // AscendC::printf("[clamp debug] hcMult=%ld hcMultAlign=%ld curRowFactor=%ld\n", tilingData->hcMult, tilingData->hcMultAlign, curRowFactor);
                    // float min_val = mxies02ReduceLocal.GetValue(0);
                    // float max_val = mxies02ReduceLocal.GetValue(0);
                    // int64_t clamp_elem_cnt = tilingData->hcMult * tilingData->hcMultAlign;
                    // for (int64_t idx = 1; idx < clamp_elem_cnt; ++idx) {
                    //     float val = mxies02ReduceLocal.GetValue(idx);
                    //     if (val < min_val) min_val = val;
                    //     if (val > max_val) max_val = val;
                    // }
                    // AscendC::printf("[clamp debug] min=%f, max=%f\n", min_val, max_val);

                    // AscendC::DumpTensor(mxies02ReduceLocal, 9, tilingData->hcMult * tilingData->hcMultAlign);
                    PipeBarrier<PIPE_V>();
                    Maxs(mxies02ReduceLocal, mxies02ReduceLocal, tilingData->clamp_min,
                        curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);
                    // AscendC::DumpTensor(mxies02ReduceLocal, 10, tilingData->hcMult * tilingData->hcMultAlign);

                    PipeBarrier<PIPE_V>();
                    Mins(mxies02ReduceLocal, mxies02ReduceLocal, tilingData->clamp_max,
                        curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);
                    PipeBarrier<PIPE_V>();
                    // AscendC::DumpTensor(mxies02ReduceLocal, 11, tilingData->hcMult * tilingData->hcMultAlign);
                }
                //* CLAMP End

                SoftmaxFP32Perf(mxies02ReduceLocal, mxies02ReduceLocal, reduceLocal, hcBrcbLocal1, curRowFactor * tilingData->hcMult,
                                tilingData->hcMult, tilingData->hcEps);

                if (needGrad_) {
                    int64_t normOutBaseOffset = (stage2BlockIdx * curRowOfFormerBlock +
                                                rowOuterIdx * curStage2RowFactor) * tilingData->hcMult * tilingData->hcMult;
                    VToMTE3Sync();
                    CopyOut(mxies02ReduceLocal, normOutGm[normOutBaseOffset], curRowFactor * tilingData->hcMult, tilingData->hcMult);
                    MTE3ToVSync();
                }

                if (needGrad_) {
                    VToMTE3Sync();
                    int64_t sumOutBaseOffset = (stage2BlockIdx * curRowOfFormerBlock +
                                               rowOuterIdx * curStage2RowFactor) * tilingData->hcMult;
                    CopyOut(reduceLocal, sumOutGm[sumOutBaseOffset], curRowFactor, tilingData->hcMult);
                    MTE3ToVSync();
                }

                ReduceSumARAPerf(reduceLocal, mxies02ReduceLocal, curRowFactor, tilingData->hcMult, tilingData->hcMult);
                if (needGrad_) {
                    VToMTE3Sync();
                    int64_t sumOutBaseOffset = (tilingData->bs +
                                               stage2BlockIdx * curRowOfFormerBlock +
                                               rowOuterIdx * curStage2RowFactor) * tilingData->hcMult;
                    CopyOut(reduceLocal, sumOutGm[sumOutBaseOffset], curRowFactor, tilingData->hcMult);
                    MTE3ToVSync();
                }
                Adds(reduceLocal, reduceLocal, tilingData->hcEps, curRowFactor * tilingData->hcMult);
                PipeBarrier<PIPE_V>();
                DivABABrcInline(combFragLocal, mxies02ReduceLocal, reduceLocal, curRowFactor, tilingData->hcMult,
                                tilingData->hcMult);

                // 循环内norm输出通过拼接,减少搬运
                for (int64_t iter = 1; iter < tilingData->iterTimes; iter++) {
                    auto reduceLocal2 = reduceQue.AllocTensor<float>();
                    LastDimReduceSumPerf(reduceLocal2, combFragLocal, curRowFactor * tilingData->hcMult, tilingData->hcMult);
                    Adds(reduceLocal2, reduceLocal2, tilingData->hcEps, curRowFactor * tilingData->hcMult);
                    PipeBarrier<PIPE_V>();
                    DivABLastDimBrcInline<float, true>(combFragLocal, combFragLocal, reduceLocal2, hcBrcbLocal1,
                                                        curRowFactor * tilingData->hcMult, tilingData->hcMult);
                    if (needGrad_) {
                        reduceQue.EnQue(reduceLocal2);
                        reduceLocal2 = reduceQue.DeQue<float>();
                        int64_t sumOutBaseOffset = ((iter * 2) * tilingData->bs +
                                                   stage2BlockIdx * curRowOfFormerBlock +
                                                   rowOuterIdx * curStage2RowFactor) * tilingData->hcMult;
                        CopyOut(reduceLocal2, sumOutGm[sumOutBaseOffset], curRowFactor, tilingData->hcMult);
                    }
                    reduceQue.template FreeTensor(reduceLocal2);

                    if (needGrad_) {
                        int64_t tileRepeatTimes_ = CeilDiv(tilingData->hcMult * curStage2RowFactor * 2 * tilingData->hcMult, 64);
                        auto yLocal = yQue.template AllocTensor<float>();
                        Copy(yLocal, combFragLocal, curStage2RowFactor * tilingData->hcMult * tilingData->hcMultAlign, 1, {1, 1, 0, 0});
                        PipeBarrier<PIPE_V>();
                        Cast(yCastLocal.template ReinterpretCast<int64_t>(),
                            yLocal.template ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,
                            curStage2RowFactor * tilingData->hcMult * 8);
                        PipeBarrier<PIPE_V>();

                        Copy(yLocal, yCastLocal, 64, tileRepeatTimes_, {1, 2, 8, 16});
                        PipeBarrier<PIPE_V>();

                        Cast(yLocal.template ReinterpretCast<int32_t>(),
                            yLocal.template ReinterpretCast<int64_t>(), RoundMode::CAST_NONE,
                            curStage2RowFactor * tilingData->hcMult * tilingData->hcMult);
                        PipeBarrier<PIPE_V>();
                        yQue.template EnQue(yLocal);
                        yLocal = yQue.template DeQue<float>();
                        int64_t normOutBaseOffset = ((iter * 2) * tilingData->bs +
                                                    stage2BlockIdx * curRowOfFormerBlock +
                                                    rowOuterIdx * curStage2RowFactor) * tilingData->hcMult * tilingData->hcMult;
                        CopyOut(yLocal, normOutGm[normOutBaseOffset], 1, curRowFactor * tilingData->hcMult *tilingData->hcMult);
                        yQue.template FreeTensor(yLocal);
                    }
                    auto reduceLocal3 = reduceQue.AllocTensor<float>();
                    ReduceSumARAPerf(reduceLocal3, combFragLocal, curRowFactor, tilingData->hcMult, tilingData->hcMult);
                    Adds(reduceLocal3, reduceLocal3, tilingData->hcEps, curRowFactor * tilingData->hcMult);
                    PipeBarrier<PIPE_V>();

                    if (needGrad_) {
                        reduceQue.EnQue(reduceLocal3);
                        reduceLocal3 = reduceQue.DeQue<float>();
                        int64_t sumOutBaseOffset = ((iter * 2 + 1) * tilingData->bs +
                                                   stage2BlockIdx * curRowOfFormerBlock +
                                                   rowOuterIdx * curStage2RowFactor) * tilingData->hcMult;
                        CopyOut(reduceLocal3, sumOutGm[sumOutBaseOffset], curRowFactor, tilingData->hcMult);
                    }
                    reduceQue.template FreeTensor(reduceLocal3);
                    DivABABrcInline(combFragLocal, combFragLocal, reduceLocal3, curRowFactor, tilingData->hcMult,
                                    tilingData->hcMult);
                }

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
    GlobalTensor<float> hResLogitsGm;

    bool needGrad_ = false;
    bool isTailBsLoop_ = false;

    TQue<QuePosition::VECIN, 1> mixesQue01;
    TQue<QuePosition::VECIN, 1> mixesQue2;
    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECOUT, 1> combFragQue;

    TQue<QuePosition::VECOUT, 1> reduceQue;
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
    TBuf<QuePosition::VECCALC> combFragBuf;

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

    LocalTensor<float> reduceLocal2;
    LocalTensor<float> combFragBufLocal;
};

} // namespace MhcPreClampSinkhorn

#endif