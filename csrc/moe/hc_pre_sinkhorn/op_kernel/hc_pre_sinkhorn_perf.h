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
 * \file hc_pre_sinkhorn_perf.h
 * \brief
 */

#ifndef HC_PRE_SINKHORN_PERF_H
#define HC_PRE_SINKHORN_PERF_H

#include "kernel_operator.h"
#include "hc_pre_sinkhorn_base.h"

namespace HcPreSinkhorn {
using namespace AscendC;
template <typename T>
class HcPreSinkhornPerf {
public:
    __aicore__ inline HcPreSinkhornPerf()
    {
    }

    __aicore__ inline void Init(GM_ADDR mixes, GM_ADDR rsqrt, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR x, GM_ADDR y,
                                GM_ADDR post, GM_ADDR combFrag, GM_ADDR workspace,
                                const HcPreSinkhornTilingData *tilingDataPtr, TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        mixesGm.SetGlobalBuffer((__gm__ float *)mixes);
        rsqrtGm.SetGlobalBuffer((__gm__ float *)rsqrt);
        hcScaleGm.SetGlobalBuffer((__gm__ float *)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float *)hcBase);
        xGm.SetGlobalBuffer((__gm__ T *)x);
        yGm.SetGlobalBuffer((__gm__ T *)y);
        postGm.SetGlobalBuffer((__gm__ float *)post);
        combFragGm.SetGlobalBuffer((__gm__ float *)combFrag);

        // InQue
        int64_t mixesQue01Size = tilingData->rowFactor * tilingData->hcMultAlign * 2 * sizeof(float);
        pipe->InitBuffer(mixesQue01, 2, mixesQue01Size);
        pipe->InitBuffer(mixesQue2, 2,
                         tilingData->rowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(rsqrtQue, 2, RoundUp<float>(tilingData->rowFactor) * sizeof(float));
        pipe->InitBuffer(xQue, 2,
                         tilingData->rowFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor) * sizeof(T));

        // OutQue
        pipe->InitBuffer(yQue, 2,
                         tilingData->rowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        pipe->InitBuffer(postQue, 2, tilingData->rowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(combFragQue, 2,
                         tilingData->rowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));

        // TBuf
        pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(rowBrcbBuf0, RoundUp<float>(tilingData->rowFactor) * BLOCK_SIZE);
        pipe->InitBuffer(hcBrcbBuf1, RoundUp<float>(tilingData->rowFactor * tilingData->hcMultAlign) * BLOCK_SIZE);
        pipe->InitBuffer(reduceBuf, tilingData->rowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(xCastBuf, tilingData->rowFactor * tilingData->hcMult * RoundUp<float>(tilingData->dFactor) *
                                       sizeof(float));
        pipe->InitBuffer(yCastBuf, tilingData->rowFactor * RoundUp<float>(tilingData->dFactor) *
                                       sizeof(float));

        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        hcBase2Local = hcBaseBuf2.Get<float>();
        rowBrcbLocal0 = rowBrcbBuf0.Get<float>();
        hcBrcbLocal1 = hcBrcbBuf1.Get<float>();
        reduceLocal = reduceBuf.Get<float>();
        xCastLocal = xCastBuf.Get<float>();
        yCastLocal = yCastBuf.Get<float>();
    }

    __aicore__ inline void Process()
    {
        int64_t curBlockIdx = GetBlockIdx();
        int64_t totalBlockNum = GetBlockNum();

        int64_t rowOuterLoop =
            (curBlockIdx == totalBlockNum - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        int64_t tailRowFactor = (curBlockIdx == totalBlockNum - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                     tilingData->tailRowFactorOfFormerBlock;

        CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
        CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
        CopyIn(hcBaseGm[tilingData->hcMult * 2], hcBase2Local, tilingData->hcMult, tilingData->hcMult);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        int64_t mixGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMix;
        int64_t xGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult * tilingData->d;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            mixes01Local = mixesQue01.AllocTensor<float>();
            CopyIn(mixesGm[mixGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->hcMix], mixes01Local,
                   curRowFactor, tilingData->hcMult, tilingData->hcMix - tilingData->hcMult);
            CopyIn(
                mixesGm[mixGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->hcMix + tilingData->hcMult],
                mixes01Local[tilingData->rowFactor * tilingData->hcMultAlign], curRowFactor, tilingData->hcMult,
                tilingData->hcMix - tilingData->hcMult);
            mixesQue01.EnQue(mixes01Local);
            rsqrtLocal = rsqrtQue.AllocTensor<float>();
            CopyIn(rsqrtGm[curBlockIdx * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor],
                   rsqrtLocal, 1, curRowFactor);
            rsqrtQue.EnQue(rsqrtLocal);

            mixes01Local = mixesQue01.DeQue<float>();
            rsqrtLocal = rsqrtQue.DeQue<float>();
            ProcessPre(mixes01Local, mixes01Local, hcBase0Local, rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1,
                       hcScaleGm.GetValue(0), tilingData->eps, curRowFactor, tilingData->hcMult);
            for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                int64_t curDFactor =
                    (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                xLocal = xQue.template AllocTensor<T>();
                CopyIn(xGm[xGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->hcMult * tilingData->d +
                           dLoopIdx * tilingData->dFactor],
                       xLocal, tilingData->rowFactor * tilingData->hcMult, curDFactor, tilingData->d - curDFactor);
                xQue.template EnQue(xLocal);
                xLocal = xQue.template DeQue<T>();
                yLocal = yQue.template AllocTensor<T>();
                ProcessY(yLocal, xLocal, mixes01Local, hcBrcbLocal1, xCastLocal, yCastLocal, curRowFactor,
                         tilingData->hcMult, curDFactor);
                xQue.template FreeTensor(xLocal);
                yQue.template EnQue(yLocal);
                yLocal = yQue.template DeQue<T>();
                CopyOut(yLocal,
                        yGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->d +
                            rowOuterIdx * tilingData->rowFactor * tilingData->d + dLoopIdx * tilingData->dFactor],
                        curRowFactor, curDFactor, tilingData->d - curDFactor);
                yQue.template FreeTensor(yLocal);
            }
            // post
            postLocal = postQue.AllocTensor<float>();
            ProcessPost(postLocal, mixes01Local[tilingData->rowFactor * tilingData->hcMultAlign], hcBase1Local,
                        rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1, hcScaleGm.GetValue(1), curRowFactor,
                        tilingData->hcMult);
            mixesQue01.template FreeTensor(mixes01Local);
            postQue.EnQue(postLocal);
            postLocal = postQue.DeQue<float>();
            CopyOut(postLocal,
                    postGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult +
                           rowOuterIdx * tilingData->rowFactor * tilingData->hcMult],
                    curRowFactor, tilingData->hcMult);
            postQue.FreeTensor(postLocal);

            // combFrag
            mixes2Local = mixesQue2.AllocTensor<float>();
            CopyInWithOuterFor(mixesGm[mixGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->hcMix +
                                       tilingData->hcMult * 2],
                               mixes2Local, curRowFactor, tilingData->hcMult, tilingData->hcMult, tilingData->hcMix);
            mixesQue2.EnQue(mixes2Local);
            mixes2Local = mixesQue2.DeQue<float>();

            combFragLocal = combFragQue.AllocTensor<float>();

            MulABLastDimBrcInline<float, false>(mixes2Local, mixes2Local, rsqrtLocal, rowBrcbLocal0, curRowFactor,
                                                tilingData->hcMult * tilingData->hcMultAlign);
            Muls(mixes2Local, mixes2Local, hcScaleGm.GetValue(2),
                 curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);
            PipeBarrier<PIPE_V>();
            AddBAFirstDimBrcInline<float>(mixes2Local, mixes2Local, hcBase2Local, curRowFactor,
                                          tilingData->hcMult * tilingData->hcMultAlign);
            SoftmaxFP32Perf(mixes2Local, mixes2Local, reduceLocal, hcBrcbLocal1, curRowFactor * tilingData->hcMult,
                            tilingData->hcMult, tilingData->eps);
            ReduceSumARAPerf(reduceLocal, mixes2Local, curRowFactor, tilingData->hcMult, tilingData->hcMult);
            Adds(reduceLocal, reduceLocal, tilingData->eps, curRowFactor * tilingData->hcMult);
            PipeBarrier<PIPE_V>();
            DivABABrcInline(combFragLocal, mixes2Local, reduceLocal, curRowFactor, tilingData->hcMult,
                            tilingData->hcMult);
            for (int64_t iter = 0; iter < tilingData->iterTimes - 1; iter++) {
                LastDimReduceSumPerf(reduceLocal, combFragLocal, curRowFactor * tilingData->hcMult, tilingData->hcMult);
                Adds(reduceLocal, reduceLocal, tilingData->eps, curRowFactor * tilingData->hcMult);
                PipeBarrier<PIPE_V>();
                DivABLastDimBrcInline<float, true>(combFragLocal, combFragLocal, reduceLocal, hcBrcbLocal1,
                                                   curRowFactor * tilingData->hcMult, tilingData->hcMult);
                ReduceSumARAPerf(reduceLocal, combFragLocal, curRowFactor, tilingData->hcMult, tilingData->hcMult);
                Adds(reduceLocal, reduceLocal, tilingData->eps, curRowFactor * tilingData->hcMult);
                PipeBarrier<PIPE_V>();
                DivABABrcInline(combFragLocal, combFragLocal, reduceLocal, curRowFactor, tilingData->hcMult,
                                tilingData->hcMult);
            }
            mixesQue2.FreeTensor(mixes2Local);
            rsqrtQue.FreeTensor(rsqrtLocal);

            combFragQue.EnQue(combFragLocal);
            combFragLocal = combFragQue.DeQue<float>();
            CopyOut(combFragLocal,
                    combFragGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult * tilingData->hcMult +
                               rowOuterIdx * tilingData->rowFactor * tilingData->hcMult * tilingData->hcMult],
                    curRowFactor * tilingData->hcMult, tilingData->hcMult);
            combFragQue.FreeTensor(combFragLocal);
        }
    }

private:
    TPipe *pipe;
    const HcPreSinkhornTilingData *tilingData;
    GlobalTensor<float> mixesGm;
    GlobalTensor<float> rsqrtGm;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;

    TQue<QuePosition::VECIN, 1> mixesQue01;
    TQue<QuePosition::VECIN, 1> mixesQue2;
    TQue<QuePosition::VECIN, 1> rsqrtQue;
    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECOUT, 1> combFragQue;

    TBuf<QuePosition::VECCALC> hcBaseBuf0;
    TBuf<QuePosition::VECCALC> hcBaseBuf1;
    TBuf<QuePosition::VECCALC> hcBaseBuf2;

    TBuf<QuePosition::VECCALC> rowBrcbBuf0;
    TBuf<QuePosition::VECCALC> hcBrcbBuf1;
    TBuf<QuePosition::VECCALC> reduceBuf;

    TBuf<QuePosition::VECCALC> xCastBuf;
    TBuf<QuePosition::VECCALC> yCastBuf;

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
    LocalTensor<float> xCastLocal;
    LocalTensor<float> yCastLocal;
};

} // namespace HcPreSinkhorn

#endif