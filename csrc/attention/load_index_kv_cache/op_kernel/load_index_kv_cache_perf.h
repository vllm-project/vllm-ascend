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
 * \file load_index_kv_cache_single_row.h
 * \brief
 */

#ifndef LOAD_INDEX_KV_CACH_PERF_H
#define LOAD_INDEX_KV_CACH_PERF_H

#include "kernel_operator.h"
#include "load_index_kv_cache_base.h"

namespace LoadIndexKvCache {
using namespace AscendC;
template <typename T>
class LoadIndexKvCachePerf {
public:
    __aicore__ inline LoadIndexKvCachePerf()
    {}

    __aicore__ inline void Init(
        GM_ADDR kvCache, GM_ADDR slotMapping, GM_ADDR kv, GM_ADDR kvScale, GM_ADDR workspace,
        const LoadIndexKvCacheTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        kvCacheGm.SetGlobalBuffer((__gm__ T*)kvCache);
        slotMappingGm.SetGlobalBuffer((__gm__ int32_t*)slotMapping);

        kvGm.SetGlobalBuffer((__gm__ T*)kv);
        kvScaleGm.SetGlobalBuffer((__gm__ float*)kvScale);

        pipe->InitBuffer(kvCacheQue, 2, RoundUp<T>(tilingData->d) * sizeof(T));
        pipe->InitBuffer(kvAndKvScaleQue, 2, RoundUp<T>(tilingData->d) * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        int64_t curBlockIdx = GetBlockIdx();
        int64_t rowOuterLoop =
            (curBlockIdx == GetBlockNum() - 1) ? tilingData->rowOfTailBlock : tilingData->rowOfFormerBlock;

        int64_t baseSlotMappingOffset = curBlockIdx * tilingData->rowOfFormerBlock;
        int64_t kvBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * KV_LAST_DIM;
        int64_t kvScaleBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curSlotIdx = baseSlotMappingOffset + rowOuterIdx;
            int64_t slot = slotMappingGm.GetValue(curSlotIdx);
            if (slot == -1) {
                // slot为-1时，需要将对应位置的kv和kvscale全部刷成0
                kvAndKvScaleLocal = kvAndKvScaleQue.template AllocTensor<T>();
                Duplicate(kvAndKvScaleLocal.template ReinterpretCast<uint8_t>(), uint8_t(0), KV_LAST_DIM);
                Duplicate(kvAndKvScaleLocal[KV_LAST_DIM].template ReinterpretCast<float>(), float(0), 1);
                kvAndKvScaleQue.template EnQue(kvAndKvScaleLocal);
                kvAndKvScaleLocal = kvAndKvScaleQue.template DeQue<T>();
                CopyOut(kvAndKvScaleLocal, kvGm[kvBaseOffset + rowOuterIdx * KV_LAST_DIM], 1, KV_LAST_DIM);
                CopyOut(kvAndKvScaleLocal[KV_LAST_DIM].template ReinterpretCast<float>(), kvScaleGm[kvScaleBaseOffset + rowOuterIdx], 1, 1);
                kvAndKvScaleQue.template FreeTensor(kvAndKvScaleLocal);
                continue;
            }

            // 处理kvCacheLocal
            kvCacheLocal = kvCacheQue.template AllocTensor<T>();
            int64_t kvCacheGmOffset = (slot / tilingData->bs) * tilingData->blockStride + (slot % tilingData->bs) * KV_LAST_DIM;
            int64_t kvCacheScaleGmOffset = (slot / tilingData->bs) * tilingData->blockStride + tilingData->bs * KV_LAST_DIM + (slot % tilingData->bs) * KV_SCALE_LAST_DIM;
            CopyIn(kvCacheGm[kvCacheGmOffset].template ReinterpretCast<uint8_t>(), kvCacheLocal.template ReinterpretCast<uint8_t>(), 1, KV_LAST_DIM);
            CopyIn(kvCacheGm[kvCacheScaleGmOffset].template ReinterpretCast<uint8_t>(), kvCacheLocal[KV_LAST_DIM].template ReinterpretCast<uint8_t>(), 1, KV_SCALE_LAST_DIM);
            kvCacheQue.template EnQue(kvCacheLocal);
            kvCacheLocal = kvCacheQue.template DeQue<T>();

            kvAndKvScaleLocal = kvAndKvScaleQue.AllocTensor<T>();
            DataCopy(kvAndKvScaleLocal, kvCacheLocal, RoundUp<T>(tilingData->d));
            kvCacheQue.template FreeTensor(kvCacheLocal);

            kvAndKvScaleQue.template EnQue(kvAndKvScaleLocal);
            kvAndKvScaleLocal = kvAndKvScaleQue.template DeQue<T>();
            CopyOut(kvAndKvScaleLocal, kvGm[kvBaseOffset + rowOuterIdx * KV_LAST_DIM], 1, KV_LAST_DIM);
            CopyOut(kvAndKvScaleLocal[KV_LAST_DIM].template ReinterpretCast<float>(), kvScaleGm[kvScaleBaseOffset + rowOuterIdx * 1], 1, 1);
            kvAndKvScaleQue.template FreeTensor(kvAndKvScaleLocal);
        }
    }
private:
    TPipe* pipe;
    const LoadIndexKvCacheTilingData* tilingData;
    GlobalTensor<T> kvCacheGm;
    GlobalTensor<int32_t> slotMappingGm;
    GlobalTensor<T> kvGm;
    GlobalTensor<float> kvScaleGm;

    TQue<QuePosition::VECIN, 1> kvCacheQue;
    TQue<QuePosition::VECOUT, 1> kvAndKvScaleQue;
    TQue<QuePosition::VECOUT, 1> kvScaleQue;

    LocalTensor<T> kvCacheLocal;
    LocalTensor<T> kvAndKvScaleLocal;
};

} // namespace LoadIndexKvCache

#endif