/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KEY_POOL_LAYER_NORM_H
#define KEY_POOL_LAYER_NORM_H

namespace KeyPool {

__aicore__ inline void KeyPoolLayerNorm(const LocalTensor<float> &data, const LocalTensor<float> &in,
                                        const LocalTensor<float> &out, const LocalTensor<float> &mean,
                                        const LocalTensor<float> &gamma, const LocalTensor<float> &beta, float eps,
                                        uint32_t count)
{
    const float reciprocal = 1.0f / static_cast<float>(static_cast<int64_t>(count));

    Duplicate(mean, reciprocal, count);
    PipeBarrier<PIPE_V>();
    Mul(in, data, mean, count);
    PipeBarrier<PIPE_V>();
    ReduceSum(in, in, in, count);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float meanValue = in.GetValue(0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(mean, meanValue, count);
    PipeBarrier<PIPE_V>();

    Sub(data, data, mean, count);
    PipeBarrier<PIPE_V>();
    Mul(out, data, data, count);
    PipeBarrier<PIPE_V>();
    Muls(out, out, reciprocal, count);
    PipeBarrier<PIPE_V>();
    ReduceSum(out, out, out, count);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float variance = out.GetValue(0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(out, variance, count);
    PipeBarrier<PIPE_V>();
    Adds(out, out, eps, count);
    PipeBarrier<PIPE_V>();
    Sqrt(out, out, count);
    PipeBarrier<PIPE_V>();

    Div(data, data, out, count);
    PipeBarrier<PIPE_V>();
    Mul(data, data, gamma, count);
    PipeBarrier<PIPE_V>();
    Add(data, data, beta, count);
    PipeBarrier<PIPE_V>();
}

// Normalize complete current K rows before they enter the pooling path. The
// caller provides a scratch tensor with room for input/output and three
// vectors: mean, gamma, and beta.
__aicore__ inline void KeyPoolLayerNormRowsInplace(const LocalTensor<float> &data, const LocalTensor<float> &scratch,
                                                   const GlobalTensor<float> &normWeight,
                                                   const GlobalTensor<float> &normBias, float eps, uint32_t rowCount,
                                                   uint32_t rowStride, uint32_t headDim)
{
    if (rowCount == 0 || rowStride < headDim) {
        return;
    }
    LocalTensor<float> input = scratch;
    LocalTensor<float> output = input[headDim];
    LocalTensor<float> mean = output[headDim];
    LocalTensor<float> gamma = mean[headDim];
    LocalTensor<float> beta = gamma[headDim];
    DataCopy(gamma, normWeight, headDim);
    DataCopy(beta, normBias, headDim);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    for (uint32_t row = 0; row < rowCount; ++row) {
        KeyPoolLayerNorm(data[static_cast<uint64_t>(row) * rowStride], input, output, mean, gamma, beta, eps, headDim);
    }
}

__aicore__ inline void ApplyKeyPoolRotaryPlaceholder(const LocalTensor<float> &, uint32_t)
{
    // RoPE inputs are rejected by the Host in this version. Keep the transform
    // boundary explicit so a later RoPE implementation is inserted between
    // normalization and cache writeback without changing the Pool stage.
}

} // namespace KeyPool

#endif // KEY_POOL_LAYER_NORM_H
