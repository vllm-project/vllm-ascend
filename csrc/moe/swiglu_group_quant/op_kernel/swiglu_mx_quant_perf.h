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
 * \file swiglu_mx_quant_perf.h
 * \brief
 */

#ifndef SWIGLU_MX_QUANT_PERF_H
#define SWIGLU_MX_QUANT_PERF_H

#include "kernel_operator.h"
#include "swiglu_group_quant_base.h"

namespace SwigluGroupQuant {
using namespace AscendC;
template <typename T0, typename T1, typename T2>
class SwigluMxQuantPerf {
public:
    __aicore__ inline SwigluMxQuantPerf()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR topkWeight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace, const SwigluGroupQuantTilingData* tilingDataPtr, TPipe* pipePtr)
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

        tBufPool.InitBuffer(xQue, 2, tilingData->rowFactor * RoundUp<T0>(tilingData->dFactor) * sizeof(T0) * 2);
        tBufPool.InitBuffer(
            yQue, 2, tilingData->rowFactor * RoundUp<T1>(tilingData->dFactor) * sizeof(T1));
        scaleColNum = CeilDiv(tilingData->dFactor, PER_MX_FP16);
        tBufPool.InitBuffer(scaleQue, 2, tilingData->rowFactor * RoundUp<T2>(scaleColNum) * sizeof(T2));

        tBufPool.InitBuffer(swigluBuf, tilingData->rowFactor * RoundUp<T0>(tilingData->dFactor) * sizeof(T0));
        tBufPool.InitBuffer(maxExpBuf, tilingData->rowFactor * RoundUp<uint16_t>(scaleColNum) * sizeof(uint16_t));
        tBufPool.InitBuffer(invScaleBuf, tilingData->rowFactor * RoundUp<uint16_t>(scaleColNum) * sizeof(uint16_t));

        swigluLocal = swigluBuf.Get<T0>();
        maxExpLocal = maxExpBuf.Get<uint16_t>();
        invScaleLocal = invScaleBuf.Get<uint16_t>();

        if constexpr (IsSameType<T1, fp8_e4m3fn_t>::value) {
            lowerBoundOfB16MaxExp = LOWER_BOUND_OF_MAX_EXP_FOR_E4M3;
        } else {
            lowerBoundOfB16MaxExp = LOWER_BOUND_OF_MAX_EXP_FOR_E5M2;
        }

        scaleCols = CeilDiv(tilingData->scaleCol, 2) * 2;
        perLoopScaleCols = CeilDiv(tilingData->dFactor, 32);
        lastLoopValidScaleCols = tilingData->scaleCol - (tilingData->dLoop - 1) * perLoopScaleCols;
        lastLoopScaleCols = scaleCols - (tilingData->dLoop - 1) * perLoopScaleCols;
        hasClampValue_ = (tilingData->hasClampValue == 1);
        clampValue_ = tilingData->clampValue;
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= usedCoreNums) {
            return;
        }
        SetMaxValue();
        int64_t curBlockIdx = GetBlockIdx();
        int64_t rowOuterLoop =
            (curBlockIdx == usedCoreNums - 1) ? rowLoopOfTailBlock :rowLoopOfFormerBlock;
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
                int64_t scaleDFactor = CeilDiv(curDFactor, PER_MX_FP16);
                int64_t xBaseOffset = rowOuterIdx * tilingData->rowFactor * tilingData->d + dLoopIdx * tilingData->dFactor;
                xLocal = xQue.template AllocTensor<T0>();
                CopyIn(
                    xGm[x0GmBaseOffset + xBaseOffset],
                    xLocal, curRowFactor, curDFactor, tilingData->d - curDFactor);

                CopyIn(
                    xGm[x1GmBaseOffset + xBaseOffset],
                    xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)], curRowFactor, curDFactor, tilingData->d - curDFactor);
                xQue.template EnQue(xLocal);
                xLocal = xQue.template DeQue<T0>();
                if (hasTopkWeight_) {
                    if (hasClampValue_) {
                        VFProcessSwigluGroupQuant<T0, true, true>(swigluLocal, xLocal, xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)], topkWeightLocal, curRowFactor, curDFactor, clampValue_);
                    } else {
                        VFProcessSwigluGroupQuant<T0, true, false>(swigluLocal, xLocal, xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)], topkWeightLocal, curRowFactor, curDFactor, clampValue_);
                    }
                } else {
                    if (hasClampValue_) {
                        VFProcessSwigluGroupQuant<T0, false, true>(swigluLocal, xLocal, xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)], topkWeightLocal, curRowFactor, curDFactor, clampValue_);
                    } else {
                        VFProcessSwigluGroupQuant<T0, false, false>(swigluLocal, xLocal, xLocal[curRowFactor * RoundUp<T0>(tilingData->dFactor)], topkWeightLocal, curRowFactor, curDFactor, clampValue_);
                    }
                }
                uint32_t loopScaleCols = (dLoopIdx == tilingData->dLoop - 1) ? lastLoopScaleCols : perLoopScaleCols;
                uint32_t loopValidScaleCols = (dLoopIdx == tilingData->dLoop - 1) ? lastLoopValidScaleCols : perLoopScaleCols;
                // 开始进行MxFp8量化
                VFComputeMaxExp(maxExpLocal, swigluLocal, curRowFactor, curDFactor);
                scaleLocal = scaleQue.template AllocTensor<T2>();
                VFComputeScale(scaleLocal.template ReinterpretCast<uint16_t>(), invScaleLocal, maxExpLocal, curRowFactor, loopScaleCols, loopValidScaleCols,  lowerBoundOfB16MaxExp);

                scaleQue.template EnQue(scaleLocal);
                scaleLocal = scaleQue.template DeQue<T2>();
                CopyOut<T2>(scaleLocal, scaleGm[scaleGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->scaleCol + dLoopIdx * CeilDiv(tilingData->dFactor, 32)],
                curRowFactor, scaleDFactor, tilingData->scaleCol - scaleDFactor);
                scaleQue.template FreeTensor(scaleLocal);

                yLocal = yQue.template AllocTensor<T1>();
                VFComputeData(yLocal, swigluLocal, invScaleLocal, curRowFactor, curDFactor);
                xQue.template FreeTensor(xLocal);
                yQue.template EnQue(yLocal);

                yLocal = yQue.template DeQue<T1>();
                CopyOut(yLocal, yGm[yGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->splitD + dLoopIdx * tilingData->dFactor], curRowFactor, curDFactor, tilingData->splitD - curDFactor);
                yQue.template FreeTensor(yLocal);
            }
            if (hasTopkWeight_) {
                topkWeightQue.template FreeTensor(topkWeightLocal);
            }
        }
    }

    __aicore__ inline void SetMaxValue() {
        if constexpr (IsSameType<T1, fp8_e5m2_t>::value) {
            maxValue = static_cast<float>(1.0) / FP8_E5M2_MAX_VALUE;
        } else if constexpr (IsSameType<T1, fp8_e4m3fn_t>::value) {
            maxValue = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
        }
    }

private:
    TPipe* pipe;
    const SwigluGroupQuantTilingData* tilingData;
    GlobalTensor<T0> xGm;
    GlobalTensor<int64_t> groupIndexGm;
    GlobalTensor<T1> yGm;
    GlobalTensor<T2> scaleGm;
    GlobalTensor<float> topkWeightGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> scaleQue;

    TBuf<QuePosition::VECCALC> swigluBuf;
    TBuf<QuePosition::VECCALC> maxExpBuf;
    TBuf<QuePosition::VECCALC> invScaleBuf;
    TQue<QuePosition::VECIN, 1> groupIndexQue;
    TBuf<QuePosition::VECCALC> groupIndexSumBuf;
    TQue<QuePosition::VECIN, 1> topkWeightQue;
    TBufPool<QuePosition::VECCALC, 12> tBufPool;

    LocalTensor<T0> xLocal;
    LocalTensor<T1> yLocal;
    LocalTensor<T2> scaleLocal;
    LocalTensor<T0> swigluLocal;
    LocalTensor<uint16_t> maxExpLocal;
    LocalTensor<uint16_t> invScaleLocal;
    LocalTensor<int64_t> groupIndexLocal;
    LocalTensor<int64_t> groupSumLocal;
    LocalTensor<float> topkWeightLocal;

    float maxValue = 0.0f;
    int64_t scaleColNum = 0;
    uint16_t lowerBoundOfB16MaxExp = 0;
    uint32_t perLoopScaleCols;
    uint32_t lastLoopValidScaleCols;
    uint32_t lastLoopScaleCols;
    uint32_t scaleCols;
    float clampValue_ = 448.0f;
    bool hasClampValue_ = false;
    bool hasGroupIndex_ = false;
    bool hasTopkWeight_ = false;

    int64_t tailRowFactorOfTailBlock = 0;
    int64_t tailRowFactorOfFormerBlock = 0;
    int64_t rowLoopOfTailBlock = 0;
    int64_t rowLoopOfFormerBlock = 0;
    int64_t usedCoreNums = 0;
    int64_t rowOfFormerBlock = 0;
    int64_t rowOfTailBlock = 0;
};

} // namespace HCPre

#endif