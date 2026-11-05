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
 * \file swiglu_fp8_quant_per_token.h
 * \brief
 */

#ifndef SWIGLU_FP8_QUANT_PER_TOKEN_H
#define SWIGLU_FP8_QUANT_PER_TOKEN_H

#include "kernel_operator.h"
#include "swiglu_group_quant_base.h"

namespace SwigluGroupQuant {
using namespace AscendC;
template <typename T0, typename T1, typename T2, bool outputOrigin>
class SwigluFp8QuantPerToken {
public:
    __aicore__ inline SwigluFp8QuantPerToken()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR topkWeight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale, GM_ADDR yOrigin, GM_ADDR workspace, const SwigluGroupQuantTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T0*)x);
        yGm.SetGlobalBuffer((__gm__ T1*)y);
        scaleGm.SetGlobalBuffer((__gm__ T2*)scale);

        pipe->InitBufPool(tBufPool, tilingData->ubSize);
        if (groupIndex != nullptr) {
            hasGroupIndex_ = true;
            groupIndexGm.SetGlobalBuffer((__gm__ int64_t*)groupIndex);
            if (tilingData->groupListType == 0) {
                tBufPool.InitBuffer(groupIndexQue, 2, RoundUp<int64_t>(tilingData->gFactor) * sizeof(int64_t));
                tBufPool.InitBuffer(groupIndexSumBuf, BLOCK_SIZE);
                groupSumLocal = groupIndexSumBuf.Get<int64_t>();
                for (int64_t idx = 0; idx < tilingData->gLoop; idx++) {
                    int64_t curGFactor = (idx == tilingData->gLoop - 1) ? tilingData->tailGFactor : tilingData->gFactor;
                    groupIndexLocal = groupIndexQue.template AllocTensor<int64_t>();
                    CopyIn(groupIndexGm[idx * tilingData->gFactor], groupIndexLocal, 1, curGFactor);
                    groupIndexQue.template EnQue(groupIndexLocal);
                    groupIndexLocal = groupIndexQue.template DeQue<int64_t>();
                    if (idx == 0) {
                        VFProcessGroupIndex<int64_t, false>(groupSumLocal, groupIndexLocal, curGFactor);
                    } else {
                        VFProcessGroupIndex<int64_t, true>(groupSumLocal, groupIndexLocal, curGFactor);
                    }
                    groupIndexQue.template FreeTensor(groupIndexLocal);
                }
                event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventId);
                WaitFlag<HardEvent::V_S>(eventId);
                int64_t realBs = groupSumLocal.GetValue(0) > tilingData->bs ? tilingData->bs : groupSumLocal.GetValue(0);

                rowOfFormerBlock = CeilDiv(realBs, static_cast<int64_t>(tilingData->coreNum));
                usedCoreNums = CeilDiv(realBs, rowOfFormerBlock) < tilingData->coreNum ? CeilDiv(realBs, rowOfFormerBlock) : tilingData->coreNum;
                rowOfTailBlock = realBs - (usedCoreNums - 1) * rowOfFormerBlock;

                rowLoopOfFormerBlock = CeilDiv(rowOfFormerBlock, tilingData->rowFactor);
                rowLoopOfTailBlock = CeilDiv(rowOfTailBlock, tilingData->rowFactor);
                tailRowFactorOfFormerBlock = rowOfFormerBlock % tilingData->rowFactor == 0 ? tilingData->rowFactor : rowOfFormerBlock % tilingData->rowFactor;
                tailRowFactorOfTailBlock = rowOfTailBlock % tilingData->rowFactor == 0 ? tilingData->rowFactor : rowOfTailBlock % tilingData->rowFactor;
                tBufPool.Reset();
            }
        } else {
            rowOfFormerBlock = tilingData->rowOfFormerBlock;
            rowOfTailBlock = tilingData->rowOfTailBlock;
            rowLoopOfFormerBlock = tilingData->rowLoopOfFormerBlock;
            rowLoopOfTailBlock = tilingData->rowLoopOfTailBlock;
            tailRowFactorOfFormerBlock = tilingData->tailRowFactorOfFormerBlock;
            tailRowFactorOfTailBlock = tilingData->tailRowFactorOfTailBlock;
            usedCoreNums = GetBlockNum();
        }

        if (topkWeight != nullptr) {
            hasTopkWeight_ = true;
            topkWeightGm.SetGlobalBuffer((__gm__ float*)topkWeight);
            tBufPool.InitBuffer(topkWeightQue, 2, RoundUp<float>(tilingData->rowFactor) * sizeof(float));
        }
        if constexpr (outputOrigin) {
            yOriginGm.SetGlobalBuffer((__gm__ T0*)yOrigin);
            tBufPool.InitBuffer(yOriginQue, 2, tilingData->rowFactor * RoundUp<T0>(tilingData->dFactor) * sizeof(T0));
        }

        tBufPool.InitBuffer(x0Que, 2, tilingData->rowFactor * RoundUp<T0>(tilingData->dFactor) * sizeof(T0));
        tBufPool.InitBuffer(x1Que, 2, tilingData->rowFactor * RoundUp<T0>(tilingData->dFactor) * sizeof(T0));
        tBufPool.InitBuffer(
            yQue, 2, tilingData->rowFactor * RoundUp<T1>(tilingData->dFactor) * sizeof(T1));
        // scale 在ub内连续写，拷出时采用Compact模式进行搬出
        int64_t scaleColNum = CeilDiv(tilingData->dFactor, PER_BLOCK_FP16);
        tBufPool.InitBuffer(scaleQue, 2, RoundUp<T2>(tilingData->rowFactor * scaleColNum) * sizeof(T2));
        hasClampValue_ = (tilingData->hasClampValue == 1);
        hasRoundScale_ = (tilingData->roundScale == 1);
        clampValue_ = tilingData->clampValue;
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= usedCoreNums) {
            return;
        }
        int64_t curBlockIdx = GetBlockIdx();
        int64_t rowOuterLoop =
            (curBlockIdx == usedCoreNums - 1) ? rowLoopOfTailBlock : rowLoopOfFormerBlock;
        int64_t tailRowFactor = (curBlockIdx == usedCoreNums - 1) ? tailRowFactorOfTailBlock :
                                                                     tailRowFactorOfFormerBlock;
        int64_t x0GmBaseOffset = curBlockIdx * rowOfFormerBlock * tilingData->d;
        int64_t x1GmBaseOffset = x0GmBaseOffset + tilingData->splitD;
        int64_t yGmBaseOffset = curBlockIdx * rowOfFormerBlock * tilingData->splitD;
        int64_t scaleGmBaseOffset = curBlockIdx * rowOfFormerBlock * tilingData->scaleCol;
        int64_t topkWeightGmBaseOffset = curBlockIdx * rowOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            // copy in topkWeight
            if (hasTopkWeight_) {
                topkWeightLocal = topkWeightQue.template AllocTensor<float>();
                CopyIn(topkWeightGm[topkWeightGmBaseOffset + rowOuterIdx * tilingData->rowFactor], topkWeightLocal, 1, curRowFactor);
                topkWeightQue.template EnQue(topkWeightLocal);
                topkWeightLocal = topkWeightQue.template DeQue<float>();
            }

            for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                int64_t curDFactor =
                    (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                int64_t scaleDFactor = CeilDiv(curDFactor, PER_BLOCK_FP16);
                int64_t xBaseOffset = rowOuterIdx * tilingData->rowFactor * tilingData->d + dLoopIdx * tilingData->dFactor;
                x0Local = x0Que.template AllocTensor<T0>();
                CopyIn(
                    xGm[x0GmBaseOffset + xBaseOffset],
                    x0Local, curRowFactor, curDFactor, tilingData->d - curDFactor);
                x0Que.template EnQue(x0Local);
                x0Local = x0Que.template DeQue<T0>();

                x1Local = x1Que.template AllocTensor<T0>();
                CopyIn(
                    xGm[x1GmBaseOffset + xBaseOffset],
                    x1Local, curRowFactor, curDFactor, tilingData->d - curDFactor);
                x1Que.template EnQue(x1Local);
                x1Local = x1Que.template DeQue<T0>();

                if constexpr (outputOrigin) {
                    yOriginLocal = yOriginQue.template AllocTensor<T0>();
                }

                yLocal = yQue.template AllocTensor<T1>();
                scaleLocal = scaleQue.template AllocTensor<T2>();

                int32_t maskBit = (hasRoundScale_ << 2) | (hasClampValue_ << 1) | hasTopkWeight_;

                if constexpr (outputOrigin) {
                    Fp8QuantPerTokenDispatcherYOrigin<T1, T0, T2>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local,
                                                      topkWeightLocal, clampValue_, curRowFactor, curDFactor, maskBit);
                } else {
                    Fp8QuantPerTokenDispatcher<T1, T0, T2>(yLocal, yOriginLocal, scaleLocal, x0Local, x1Local,
                                                      topkWeightLocal, clampValue_, curRowFactor, curDFactor, maskBit);
                }


                x0Que.template FreeTensor(x0Local);
                x1Que.template FreeTensor(x1Local);

                yQue.template EnQue(yLocal);
                yLocal = yQue.template DeQue<T1>();
                CopyOut(yLocal, yGm[yGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->splitD + dLoopIdx * tilingData->dFactor],
                    curRowFactor, curDFactor, tilingData->splitD - curDFactor);
                yQue.template FreeTensor(yLocal);

                scaleQue.template EnQue(scaleLocal);
                scaleLocal = scaleQue.template DeQue<T2>();
                CopyOut<T2, AscendC::PaddingMode::Compact>(scaleLocal,
                    scaleGm[scaleGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->scaleCol + dLoopIdx * CeilDiv(tilingData->dFactor, PER_BLOCK_FP16)],
                    curRowFactor, scaleDFactor, tilingData->scaleCol - scaleDFactor);
                scaleQue.template FreeTensor(scaleLocal);

                // copy yOrigin to gm
                if constexpr (outputOrigin) {
                    int64_t yOriginGmBaseOffset = curBlockIdx * rowOfFormerBlock * tilingData->splitD;
                    yOriginQue.template EnQue(yOriginLocal);
                    yOriginLocal = yOriginQue.template DeQue<T0>();
                    CopyOut(yOriginLocal, yOriginGm[yOriginGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->splitD + dLoopIdx * tilingData->dFactor],
                        curRowFactor, curDFactor, tilingData->splitD - curDFactor);
                    yOriginQue.template FreeTensor(yOriginLocal);
                }
            }
            if (hasTopkWeight_) {
                topkWeightQue.template FreeTensor(topkWeightLocal);
            }
        }
    }

private:
    TPipe* pipe;
    const SwigluGroupQuantTilingData* tilingData;
    GlobalTensor<T0> xGm;
    GlobalTensor<T1> yGm;
    GlobalTensor<T2> scaleGm;
    GlobalTensor<float> topkWeightGm;
    GlobalTensor<T0> yOriginGm;
    GlobalTensor<int64_t> groupIndexGm;

    TQue<QuePosition::VECIN, 1> x0Que;
    TQue<QuePosition::VECIN, 1> x1Que;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> scaleQue;
    TQue<QuePosition::VECIN, 1> topkWeightQue;
    TQue<QuePosition::VECOUT, 1> yOriginQue;

    TQue<QuePosition::VECIN, 1> groupIndexQue;
    TBuf<QuePosition::VECCALC> groupIndexSumBuf;
    TBufPool<QuePosition::VECCALC, 12> tBufPool;

    LocalTensor<T0> x0Local;
    LocalTensor<T0> x1Local;
    LocalTensor<T1> yLocal;
    LocalTensor<T2> scaleLocal;
    LocalTensor<float> topkWeightLocal;
    LocalTensor<T0> yOriginLocal;

    LocalTensor<int64_t> groupIndexLocal;
    LocalTensor<int64_t> groupSumLocal;

    float clampValue_ = 448.0f;
    bool hasTopkWeight_ = false;
    bool hasRoundScale_ = false;
    bool hasClampValue_ = false;

    bool hasGroupIndex_ = false;
    int64_t tailRowFactorOfTailBlock = 0;
    int64_t tailRowFactorOfFormerBlock = 0;
    int64_t rowLoopOfTailBlock = 0;
    int64_t rowLoopOfFormerBlock = 0;
    int64_t usedCoreNums = 0;
    int64_t rowOfFormerBlock = 0;
    int64_t rowOfTailBlock = 0;
};

} // namespace SwigluGroupQuant

#endif