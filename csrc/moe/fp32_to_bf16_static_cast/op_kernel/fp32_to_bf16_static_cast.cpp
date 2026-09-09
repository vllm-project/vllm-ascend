/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

using namespace AscendC;

class Fp32ToBf16StaticCastKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, const Fp32ToBf16StaticCastTilingData* tilingData)
    {
        const int64_t start = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFormer;
        const int64_t remaining = tilingData->elementNum - start;
        blockElementNum_ = remaining > tilingData->blockFormer ? tilingData->blockFormer : remaining;
        tileElementNum_ = tilingData->ubFormer;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + start, blockElementNum_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y) + start, blockElementNum_);
        pipe_.InitBuffer(xQueue_, 2, static_cast<uint32_t>(tileElementNum_ * sizeof(float)));
        pipe_.InitBuffer(yQueue_, 2, static_cast<uint32_t>(tileElementNum_ * sizeof(bfloat16_t)));
    }

    __aicore__ inline void Process()
    {
        for (int64_t offset = 0; offset < blockElementNum_; offset += tileElementNum_) {
            const int64_t remaining = blockElementNum_ - offset;
            const uint32_t count = static_cast<uint32_t>(
                remaining > tileElementNum_ ? tileElementNum_ : remaining);
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, uint32_t count)
    {
        LocalTensor<float> xLocal = xQueue_.AllocTensor<float>();
        DataCopyExtParams copyParams{
            1, count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0F};
        DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<float> xLocal = xQueue_.DeQue<float>();
        LocalTensor<bfloat16_t> yLocal = yQueue_.AllocTensor<bfloat16_t>();

        Cast(yLocal, xLocal, RoundMode::CAST_RINT, count);

        yQueue_.EnQue(yLocal);
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t count)
    {
        LocalTensor<bfloat16_t> yLocal = yQueue_.DeQue<bfloat16_t>();
        DataCopyExtParams copyParams{
            1, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(yGm_[offset], yLocal, copyParams);
        yQueue_.FreeTensor(yLocal);
    }

    TPipe pipe_;
    GlobalTensor<float> xGm_;
    GlobalTensor<bfloat16_t> yGm_;
    TQue<TPosition::VECIN, 2> xQueue_;
    TQue<TPosition::VECOUT, 2> yQueue_;
    int64_t blockElementNum_ = 0;
    int64_t tileElementNum_ = 0;
};

extern "C" __global__ __aicore__ void fp32_to_bf16_static_cast(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    Fp32ToBf16StaticCastKernel op;
    op.Init(x, y, &tilingData);
    op.Process();
}
