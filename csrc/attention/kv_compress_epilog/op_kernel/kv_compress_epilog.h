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
 * \file kv_compress_epilog.h
 * \brief KV compress epilog kernel implementation
 */

#ifndef KV_COMPRESS_EPILOG_H
#define KV_COMPRESS_EPILOG_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kv_compress_epilog_common.h"

using namespace AscendC;

namespace KvCompressEpilogOps {

template <typename T0, typename U, typename T1>
class KvCompressEpilogRegBase {
public:
    __aicore__ inline KvCompressEpilogRegBase(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR slotMapping,
        GM_ADDR kvCache,
        const KvCompressEpilogTilingData* tilingData);

    __aicore__ inline void Process();

    __aicore__ inline void SetMaxValue();

private:
    TPipe* pipe_ = nullptr;

    GlobalTensor<T0> xGm;
    GlobalTensor<U> slotMappingGm;
    GlobalTensor<T1> kvCacheGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> kvCacheQue;
    TBuf<QuePosition::VECCALC> kvCacheScaleBuf;
    TBuf<QuePosition::VECCALC> indexBuf;

    LocalTensor<T0> xLocal;
    LocalTensor<T1> kvCacheLocal;
    LocalTensor<T1> kvCacheScaleLocal;
    LocalTensor<U> indexLocal;
    int64_t validIdx = 0;
    float maxValue = 0.0f;
    float fp8Min = 0.0f;
    float fp8Max = 0.0f;

    // Tiling data
    const KvCompressEpilogTilingData* tilingData = nullptr;
};

// Template implementations

template <typename T0, typename U, typename T1>
__aicore__ inline void KvCompressEpilogRegBase<T0, U, T1>::Init(
    GM_ADDR x,
    GM_ADDR slotMapping,
    GM_ADDR kvCache,
    const KvCompressEpilogTilingData* tilingDataPtr)
{
        tilingData = tilingDataPtr;

        int64_t xGmBaseOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->d;

        xGm.SetGlobalBuffer((__gm__ T0*)x + xGmBaseOffset);
        kvCacheGm.SetGlobalBuffer((__gm__ T1*)kvCache);
        slotMappingGm.SetGlobalBuffer((__gm__ U*)slotMapping);

        pipe_->InitBuffer(xQue, 2, tilingData->rowFactor * RoundUp<T0>(tilingData->d) * sizeof(T0));
        if (tilingData->layout == 2) {
            pipe_->InitBuffer(kvCacheQue, 2, tilingData->rowFactor * RoundUp<T1>(tilingData->valuePerToken) * sizeof(T1));
            pipe_->InitBuffer(kvCacheScaleBuf, tilingData->rowFactor * RoundUp<T1>(tilingData->scalePerToken) * sizeof(T1));
        } else {
            pipe_->InitBuffer(kvCacheQue, 2, tilingData->rowFactor * RoundUp<T1>(tilingData->kvCacheCol) * sizeof(T1));
        }

        pipe_->InitBuffer(indexBuf, RoundUp<U>(tilingData->rowFactor) * sizeof(U));
        indexLocal = indexBuf.Get<U>();
}

template <typename T0, typename U, typename T1>
 __aicore__ inline void KvCompressEpilogRegBase<T0, U, T1>::Process()
    {
        SetMaxValue();
        int64_t rowOuterLoop =
            (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        int64_t tailRowFactor = (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                     tilingData->tailRowFactorOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            xLocal = xQue.template AllocTensor<T0>();
            validIdx = 0;
            for (int64_t rowInnerIdx = 0; rowInnerIdx < curRowFactor; rowInnerIdx++) {
                int32_t curSlotIdx = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor + rowInnerIdx;
                int32_t slot = static_cast<int32_t>(slotMappingGm.GetValue(curSlotIdx));
                if (slot == -1) {
                    continue;
                }
                CopyIn(
                    xGm[rowOuterIdx * tilingData->rowFactor * tilingData->d +
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

            kvCacheLocal = kvCacheQue.template AllocTensor<T1>();

            if (tilingData->quantMode == QUANT_MDOE_GROUP_FP8) {
                VFProcessDynamicBlockQuant(
                    kvCacheLocal, xLocal, maxValue, validIdx, tilingData->d, tilingData->concatCol, tilingData->padCol);
            } else if (tilingData->layout == 2) {
                kvCacheScaleLocal = kvCacheScaleBuf.template Get<T1>();
                if (tilingData->roundScale == 1) {
                    VFProcessDynamicMxFP8QuantLayout2<T0, T1, true>(
                        kvCacheLocal, kvCacheScaleLocal, xLocal, maxValue, fp8Min, fp8Max,
                        validIdx, tilingData->d, tilingData->valuePerToken, tilingData->scalePerToken,
                        tilingData->perGroupSize);
                } else {
                    VFProcessDynamicMxFP8QuantLayout2<T0, T1, false>(
                        kvCacheLocal, kvCacheScaleLocal, xLocal, maxValue, fp8Min, fp8Max,
                        validIdx, tilingData->d, tilingData->valuePerToken, tilingData->scalePerToken,
                        tilingData->perGroupSize);
                }
            } else {
                if (tilingData->roundScale == 1) {
                    VFProcessDynamicMxFP8Quant<T0, T1, true>(
                        kvCacheLocal, xLocal, maxValue, fp8Min, fp8Max, validIdx, tilingData->d, tilingData->concatCol, tilingData->padCol,
                        tilingData->perGroupSize);
                } else {
                    VFProcessDynamicMxFP8Quant<T0, T1, false>(
                        kvCacheLocal, xLocal, maxValue, fp8Min, fp8Max, validIdx, tilingData->d, tilingData->concatCol, tilingData->padCol,
                        tilingData->perGroupSize);
                }
            }

            xQue.template FreeTensor(xLocal);

            kvCacheQue.template EnQue(kvCacheLocal);
            kvCacheLocal = kvCacheQue.template DeQue<T1>();

            if (tilingData->layout == 2 && tilingData->quantMode == QUANT_MDOE_GROUP_MXFP8) {
                for (int32_t curValidIdx = 0; curValidIdx < validIdx; curValidIdx++) {
                    int32_t slot = indexLocal.GetValue(curValidIdx);
                    int64_t blockIdx = static_cast<int64_t>(slot) / tilingData->blockSize;
                    int64_t slotInBlock = static_cast<int64_t>(slot) % tilingData->blockSize;

                    // CopyOut value (FP8 quant + BF16 rope)
                    int64_t valueGmOff = blockIdx * tilingData->blockStride
                                       + slotInBlock * tilingData->valuePerToken;
                    CopyOut(
                        kvCacheLocal[curValidIdx * RoundUp<T1>(tilingData->valuePerToken)],
                        kvCacheGm[valueGmOff], 1, tilingData->valuePerToken);

                    // CopyOut scale (uint8)
                    int64_t scaleGmOff = blockIdx * tilingData->blockStride
                                       + tilingData->blockSize * tilingData->valuePerToken
                                       + slotInBlock * tilingData->scalePerToken;
                    CopyOut(
                        kvCacheScaleLocal[curValidIdx * RoundUp<T1>(tilingData->scalePerToken)],
                        kvCacheGm[scaleGmOff], 1, tilingData->scalePerToken);
                }
            } else {
                for (int32_t curValidIdx = 0; curValidIdx < validIdx; curValidIdx++) {
                    int32_t slot = indexLocal.GetValue(curValidIdx);
                    int64_t gmOffset = static_cast<int64_t>(slot) * tilingData->kvCacheCol;
                    if (tilingData->layout == 1) {
                        int64_t rowStride = tilingData->blockStride > 0 ? tilingData->blockStride : tilingData->kvCacheCol;
                        gmOffset = static_cast<int64_t>(slot) * rowStride;
                    }
                    CopyOut(
                        kvCacheLocal[curValidIdx * RoundUp<T1>(tilingData->kvCacheCol)],
                        kvCacheGm[gmOffset], 1, tilingData->kvCacheCol);
                }
            }
            kvCacheQue.template FreeTensor(kvCacheLocal);
        }
    }

    template <typename T0, typename U, typename T1>
    __aicore__ inline void KvCompressEpilogRegBase<T0, U, T1>::SetMaxValue()
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

}
#endif  // KV_COMPRESS_EPILOG_H
