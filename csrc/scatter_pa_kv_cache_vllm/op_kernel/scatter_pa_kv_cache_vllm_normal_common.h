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
 * \file scatter_pa_kv_cache_vllm_normal_common.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_VLLM_NORMAL_COMMON_H
#define SCATTER_PA_KV_CACHE_VLLM_NORMAL_COMMON_H
#include "kernel_operator.h"
#include "scatter_pa_kv_cache_vllm_common.h"

namespace ScatterPaKvCacheVllm {
using namespace AscendC;

class ScatterPaKvCacheVllmNormalCommon {
public:
    __aicore__ inline ScatterPaKvCacheVllmNormalCommon(TPipe *pipe, const ScatterPaKvCacheVllmTilingData *__restrict__ tilingData)
    : pipe_(pipe), tilingData_(tilingData)
    {
    }

    template <typename T>
    __aicore__ inline void CopyToCache(uint64_t offsetIn, uint64_t offsetOut, int64_t ubLoop, int64_t ubTail,
                                      GlobalTensor<T> &inGm, GlobalTensor<T> &outGm)
    {
        DataCopyExtParams copyParam = {1, static_cast<uint32_t>(usedUbSize_), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        for (int64_t j = 0; j < ubLoop; j++) {
            auto ubLocal = queBind_.AllocTensor<T>();
            DataCopyPad(ubLocal, inGm[offsetIn], copyParam, padParams);
            queBind_.EnQue(ubLocal);
            ubLocal = queBind_.DeQue<T>();
            DataCopyPad(outGm[offsetOut], ubLocal, copyParam);
            queBind_.FreeTensor(ubLocal);
            offsetIn += usedUbSize_ / sizeof(T);
            offsetOut += usedUbSize_ / sizeof(T);
        }
        if (ubTail > 0) {
            copyParam.blockLen = ubTail;
            auto ubLocal = queBind_.AllocTensor<T>();
            DataCopyPad(ubLocal, inGm[offsetIn], copyParam, padParams);
            queBind_.EnQue(ubLocal);
            ubLocal = queBind_.DeQue<T>();
            DataCopyPad(outGm[offsetOut], ubLocal, copyParam);
            queBind_.FreeTensor(ubLocal);
        }
    }

protected:
    TPipe *pipe_;
    int64_t startTaskId_;
    int64_t curBlockFactor_;
    int64_t usedUbSize_;
    const ScatterPaKvCacheVllmTilingData* tilingData_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> queBind_;
};
} // namespace ScatterPaKvCacheVllm
#endif