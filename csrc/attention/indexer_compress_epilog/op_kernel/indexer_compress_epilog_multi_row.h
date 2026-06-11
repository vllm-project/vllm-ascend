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
 * \file indexer_compress_epilog_multi_row.h
 * \brief
 */

#ifndef INDEXER_COMPRESS_EPILOG_MULTI_ROW_H
#define INDEXER_COMPRESS_EPILOG_MULTI_ROW_H

#include "kernel_operator.h"
#include "indexer_compress_epilog_base.h"

namespace IndexerCompressEpilog {
using namespace AscendC;
template <typename T0, typename T1, typename T2>
class IndexerCompressEpilogMultiRow {
public:
    __aicore__ inline IndexerCompressEpilogMultiRow()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR slotMapping, GM_ADDR indexerCompressCache, GM_ADDR indexerCompressCacheScale,
        GM_ADDR workspace, const IndexerCompressEpilogTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T0*)x);
        slotMappingGm.SetGlobalBuffer((__gm__ int32_t*)slotMapping);
        indexerCompressCacheGm.SetGlobalBuffer((__gm__ T1*)indexerCompressCache);
        indexerCompressCacheScaleGm.SetGlobalBuffer((__gm__ float*)indexerCompressCacheScale);

        pipe->InitBuffer(xQue, 2, tilingData->rowFactor * RoundUp<T0>(tilingData->d) * sizeof(T0));
        pipe->InitBuffer(indexerCompressCacheQue, 2, tilingData->rowFactor * RoundUp<T1>(tilingData->d) * sizeof(T1));
        pipe->InitBuffer(
            indexerCompressCacheScaleQue, 2,
            tilingData->rowFactor * RoundUp<float>(tilingData->scaleCol) * sizeof(float));
        pipe->InitBuffer(indexBuf, RoundUp<int32_t>(tilingData->rowFactor) * sizeof(int32_t));
        indexLocal = indexBuf.Get<int32_t>();
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    }

    __aicore__ inline void Process()
    {
        SetMaxValue();
        int64_t curBlockIdx = GetBlockIdx();
        int64_t rowOuterLoop =
            (curBlockIdx == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        int64_t tailRowFactor = (curBlockIdx == GetBlockNum() - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                     tilingData->tailRowFactorOfFormerBlock;
        int64_t xGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->d;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            xLocal = xQue.template AllocTensor<T0>();
            validIdx = 0;
            for (int64_t rowInnerIdx = 0; rowInnerIdx < curRowFactor; rowInnerIdx++) {
                int64_t curSlotIdx = curBlockIdx * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor + rowInnerIdx;
                int64_t slot = slotMappingGm.GetValue(curSlotIdx);
                if (slot == -1) {
                    continue;
                }
                CopyIn(
                    xGm[xGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->d +
                        rowInnerIdx * tilingData->d],
                    xLocal[validIdx * RoundUp<T0>(tilingData->d)], 1, tilingData->d);
                indexLocal.SetValue(validIdx, slot);
                validIdx++;

                event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventId);
                WaitFlag<HardEvent::S_MTE3>(eventId);
            }
            xQue.template EnQue(xLocal);
            xLocal = xQue.template DeQue<T0>();

            indexerCompressCacheLocal = indexerCompressCacheQue.template AllocTensor<T1>();
            indexerCompressCacheScaleLocal = indexerCompressCacheScaleQue.AllocTensor<float>();
            VFProcessDynamicBlockQuant(
                indexerCompressCacheLocal, indexerCompressCacheScaleLocal, xLocal, maxValue, validIdx, tilingData->d);
            xQue.template FreeTensor(xLocal);
            indexerCompressCacheQue.template EnQue(indexerCompressCacheLocal);
            indexerCompressCacheScaleQue.template EnQue(indexerCompressCacheScaleLocal);

            indexerCompressCacheLocal = indexerCompressCacheQue.template DeQue<T1>();
            indexerCompressCacheScaleLocal = indexerCompressCacheScaleQue.template DeQue<float>();

            for (int64_t curValidIdx = 0; curValidIdx < validIdx; curValidIdx++) {
                int64_t curSlotIdx = indexLocal.GetValue(curValidIdx);
                CopyOut(
                    indexerCompressCacheLocal[curValidIdx * RoundUp<T1>(tilingData->d)],
                    indexerCompressCacheGm[curSlotIdx * tilingData->d], 1, tilingData->d);
                CopyOut(
                    indexerCompressCacheScaleLocal[curValidIdx * RoundUp<float>(tilingData->scaleCol)],
                    indexerCompressCacheScaleGm[curSlotIdx * CeilDiv(tilingData->d, PER_BLOCK_FP16)], 1,
                    tilingData->scaleCol);
            }
            indexerCompressCacheQue.template FreeTensor(indexerCompressCacheLocal);
            indexerCompressCacheScaleQue.template FreeTensor(indexerCompressCacheScaleLocal);
        }
    }

    __aicore__ inline void SetMaxValue()
    {
        if constexpr (IsSameType<T1, fp8_e5m2_t>::value) {
            maxValue = static_cast<float>(1.0) / FP8_E5M2_MAX_VALUE;
            fp8Max = FP8_E5M2_MAX_VALUE;
            fp8Min = FP8_E5M2_MIN_VALUE;
        } else if constexpr (IsSameType<T1, fp8_e4m3fn_t>::value) {
            maxValue = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
            fp8Max = FP8_E4M3FN_MAX_VALUE;
            fp8Min = FP8_E4M3FN_MIN_VALUE;
        }
    }

private:
    TPipe* pipe;
    const IndexerCompressEpilogTilingData* tilingData;
    GlobalTensor<T0> xGm;
    GlobalTensor<int32_t> slotMappingGm;
    GlobalTensor<T1> indexerCompressCacheGm;
    GlobalTensor<float> indexerCompressCacheScaleGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> indexerCompressCacheQue;
    TQue<QuePosition::VECOUT, 1> indexerCompressCacheScaleQue;
    TBuf<QuePosition::VECCALC> indexBuf;

    LocalTensor<T0> xLocal;
    LocalTensor<T1> indexerCompressCacheLocal;
    LocalTensor<float> indexerCompressCacheScaleLocal;
    LocalTensor<int32_t> indexLocal;
    int64_t validIdx = 0;
    float maxValue = 0.0f;
    float fp8Min = 0.0f;
    float fp8Max = 0.0f;
};

} // namespace IndexerCompressEpilog

#endif