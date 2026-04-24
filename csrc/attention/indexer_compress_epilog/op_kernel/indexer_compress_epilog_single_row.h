/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file indexer_compress_epilog_d_full_load.h
 * \brief
 */

#ifndef INDEXER_COMPRESS_EPILOG_SINGLE_ROW_H
#define INDEXER_COMPRESS_EPILOG_SINGLE_ROW_H

#include "kernel_operator.h"
#include "indexer_compress_epilog_base.h"

namespace IndexerCompressEpilog {
using namespace AscendC;
template <typename T0, typename T1, typename T2>
class IndexerCompressEpilogSingleRow {
public:
    __aicore__ inline IndexerCompressEpilogSingleRow()
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
    }

    __aicore__ inline void Process()
    {
        SetMaxValue();
        int64_t curBlockIdx = GetBlockIdx();
        int64_t rowOuterLoop =
            (curBlockIdx == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;

        int32_t xGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->d;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {

            xLocal = xQue.template AllocTensor<T0>();
            int32_t curSlotIdx = curBlockIdx * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor;
            int32_t slot = slotMappingGm.GetValue(curSlotIdx);
            if (slot == -1) {
                continue;
            }
            CopyIn(xGm[xGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->d],
                xLocal, 1, tilingData->d);
            
            xQue.template EnQue(xLocal);
            xLocal = xQue.template DeQue<T0>();

            indexerCompressCacheLocal = indexerCompressCacheQue.template AllocTensor<T1>();
            indexerCompressCacheScaleLocal = indexerCompressCacheScaleQue.AllocTensor<float>();

            VFProcessDynamicBlockQuant(
                indexerCompressCacheLocal, indexerCompressCacheScaleLocal, xLocal, maxValue, 1, tilingData->d);
                
            xQue.template FreeTensor(xLocal);
            indexerCompressCacheQue.template EnQue(indexerCompressCacheLocal);
            indexerCompressCacheScaleQue.template EnQue(indexerCompressCacheScaleLocal);

            indexerCompressCacheLocal = indexerCompressCacheQue.template DeQue<T1>();
            indexerCompressCacheScaleLocal = indexerCompressCacheScaleQue.template DeQue<float>();

            
            CopyOut(
                indexerCompressCacheLocal, indexerCompressCacheGm[slot * tilingData->d], 1, tilingData->d);
            CopyOut(
                indexerCompressCacheScaleLocal, indexerCompressCacheScaleGm[slot * CeilDiv(tilingData->d, PER_BLOCK_FP16)], 1,
                tilingData->scaleCol);
            
            indexerCompressCacheQue.template FreeTensor(indexerCompressCacheLocal);
            indexerCompressCacheScaleQue.template FreeTensor(indexerCompressCacheScaleLocal);
        }
    }

    __aicore__ inline void SetMaxValue()
    {
        if constexpr (IsSameType<T1, fp8_e5m2_t>::value) {
            maxValue = static_cast<float>(1.0) / FP8_E5M2_MAX_VALUE;
        } else if constexpr (IsSameType<T1, fp8_e4m3fn_t>::value) {
            maxValue = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
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
    float maxValue = 0.0f;
};

} // namespace IndexerCompressEpilog

#endif