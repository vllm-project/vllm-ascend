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
 * \file hc_pre_perf.h
 * \brief
 */

#ifndef HC_PRE_SINKHORN_BASE_H
#define HC_PRE_SINKHORN_BASE_H

#include "kernel_operator.h"

namespace HcPreSinkhorn {
using namespace AscendC;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t DEFAULT_BLOCK_STRIDE = 1;
constexpr int32_t DEFAULT_REPEAT_STRIDE = 8;
constexpr int32_t ONE_REPEAT_BLOCK_NUMS = 8;
constexpr int32_t REPEAT_SIZE = 256;
constexpr int32_t MAX_REPEAT_STRIDE = 255;

__aicore__ inline int32_t CeilDiv(int32_t a, int32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int32_t CeilAlign(int32_t a, int32_t b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num)
{
    int32_t elemNum = BLOCK_SIZE / sizeof(T);
    return CeilAlign(num, elemNum);
}

template <typename T, bool needBrc = true>
__aicore__ inline void MulABLastDimBrcInline(const LocalTensor<T> &output, const LocalTensor<T> &input0,
                                             const LocalTensor<T> &input1, const LocalTensor<T> &tmpBuffer,
                                             const int32_t curRowNum, const int32_t curColNum)
{
    if constexpr (needBrc) {
        uint32_t repeatTimes = CeilDiv(curRowNum, ONE_REPEAT_BLOCK_NUMS);
        Brcb(tmpBuffer, input1, repeatTimes, {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE});
    }
    PipeBarrier<PIPE_V>();
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    if (curColNum <= elemInOneBlock) {
        Mul(output, input0, tmpBuffer, curRowNum * curColNumAlign);
    } else {
        int32_t numRepeatPerLine = curColNum / elemInOneRepeat;
        int32_t numRemainPerLine = curColNum % elemInOneRepeat;
        int32_t dstRepStridePerLine = CeilDiv(curColNum, elemInOneBlock);
        BinaryRepeatParams instrParams;

        if (numRepeatPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE || curRowNum < numRepeatPerLine) {
                // 在Col方向开Repeat, 并且Repeat小于255
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Mul(output[i * curColNumAlign], input0[i * curColNumAlign], tmpBuffer[i * elemInOneBlock],
                        elemInOneRepeat, numRepeatPerLine, instrParams);
                }
            } else {
                // 在Row方向开Repeat
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                instrParams.src1RepStride = 1;
                for (uint32_t i = 0; i < numRepeatPerLine; i++) {
                    Mul(output[i * elemInOneRepeat], input0[i * elemInOneRepeat], tmpBuffer, elemInOneRepeat, curRowNum,
                        instrParams);
                }
            }
        }

        if (numRemainPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE) {
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = 0;
                instrParams.src0RepStride = 0;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Mul(output[numRepeatPerLine * elemInOneRepeat + i * curColNumAlign],
                        input0[numRepeatPerLine * elemInOneRepeat + i * curColNumAlign], tmpBuffer[i * elemInOneBlock],
                        numRemainPerLine, 1, instrParams);
                }
            } else {
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                instrParams.src1RepStride = 0;
                Mul(output[numRepeatPerLine * elemInOneRepeat], input0[numRepeatPerLine * elemInOneRepeat], tmpBuffer,
                    numRemainPerLine, curRowNum, instrParams);
            }
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, bool needBrc = true>
__aicore__ inline void SubABLastDimBrcInline(const LocalTensor<T> &output, const LocalTensor<T> &input0,
                                             const LocalTensor<T> &input1, const LocalTensor<T> &tmpBuffer,
                                             const int32_t curRowNum, const int32_t curColNum)
{
    if constexpr (needBrc) {
        uint32_t repeatTimes = CeilDiv(curRowNum, ONE_REPEAT_BLOCK_NUMS);
        Brcb(tmpBuffer, input1, repeatTimes, {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE});
    }
    PipeBarrier<PIPE_V>();
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    if (curColNum <= elemInOneBlock) {
        Sub(output, input0, tmpBuffer, curRowNum * curColNumAlign);
    } else {
        int32_t numRepeatPerLine = curColNum / elemInOneRepeat;
        int32_t numRemainPerLine = curColNum % elemInOneRepeat;
        int32_t dstRepStridePerLine = CeilDiv(curColNum, elemInOneBlock);
        BinaryRepeatParams instrParams;

        if (numRepeatPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE || curRowNum < numRepeatPerLine) {
                // 在Col方向开Repeat, 并且Repeat小于255
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Sub(output[i * curColNumAlign], input0[i * curColNumAlign], tmpBuffer[i * elemInOneBlock],
                        elemInOneRepeat, numRepeatPerLine, instrParams);
                }
            } else {
                // 在Row方向开Repeat
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                instrParams.src1RepStride = 1;
                for (uint32_t i = 0; i < numRepeatPerLine; i++) {
                    Sub(output[i * elemInOneRepeat], input0[i * elemInOneRepeat], tmpBuffer, elemInOneRepeat, curRowNum,
                        instrParams);
                }
            }
        }

        if (numRemainPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE) {
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = 0;
                instrParams.src0RepStride = 0;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Sub(output[numRepeatPerLine * elemInOneRepeat + i * curColNumAlign],
                        input0[numRepeatPerLine * elemInOneRepeat + i * curColNumAlign], tmpBuffer[i * elemInOneBlock],
                        numRemainPerLine, 1, instrParams);
                }
            } else {
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                instrParams.src1RepStride = 0;
                Sub(output[numRepeatPerLine * elemInOneRepeat], input0[numRepeatPerLine * elemInOneRepeat], tmpBuffer,
                    numRemainPerLine, curRowNum, instrParams);
            }
        }
    }
    PipeBarrier<PIPE_V>();
}


template <typename T, bool needBrc = true>
__aicore__ inline void DivABLastDimBrcInline(const LocalTensor<T> &output, const LocalTensor<T> &input0,
                                             const LocalTensor<T> &input1, const LocalTensor<T> &tmpBuffer,
                                             const int32_t curRowNum, const int32_t curColNum)
{
    if constexpr (needBrc) {
        uint32_t repeatTimes = CeilDiv(curRowNum, ONE_REPEAT_BLOCK_NUMS);
        Brcb(tmpBuffer, input1, repeatTimes, {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE});
    }
    PipeBarrier<PIPE_V>();
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    if (curColNum <= elemInOneBlock) {
        Div(output, input0, tmpBuffer, curRowNum * curColNumAlign);
    } else {
        int32_t numRepeatPerLine = curColNum / elemInOneRepeat;
        int32_t numRemainPerLine = curColNum % elemInOneRepeat;
        int32_t dstRepStridePerLine = CeilDiv(curColNum, elemInOneBlock);
        BinaryRepeatParams instrParams;

        if (numRepeatPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE || curRowNum < numRepeatPerLine) {
                // 在Col方向开Repeat, 并且Repeat小于255
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Div(output[i * curColNumAlign], input0[i * curColNumAlign], tmpBuffer[i * elemInOneBlock],
                        elemInOneRepeat, numRepeatPerLine, instrParams);
                }
            } else {
                // 在Row方向开Repeat
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                instrParams.src1RepStride = 1;
                for (uint32_t i = 0; i < numRepeatPerLine; i++) {
                    Div(output[i * elemInOneRepeat], input0[i * elemInOneRepeat], tmpBuffer, elemInOneRepeat, curRowNum,
                        instrParams);
                }
            }
        }

        if (numRemainPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE) {
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = 0;
                instrParams.src0RepStride = 0;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Div(output[numRepeatPerLine * elemInOneRepeat + i * curColNumAlign],
                        input0[numRepeatPerLine * elemInOneRepeat + i * curColNumAlign], tmpBuffer[i * elemInOneBlock],
                        numRemainPerLine, 1, instrParams);
                }
            } else {
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                instrParams.src1RepStride = 0;
                Div(output[numRepeatPerLine * elemInOneRepeat], input0[numRepeatPerLine * elemInOneRepeat], tmpBuffer,
                    numRemainPerLine, curRowNum, instrParams);
            }
        }
    }
    PipeBarrier<PIPE_V>();
}


template <typename T>
__aicore__ inline void AddBAFirstDimBrcInline(const LocalTensor<T> &output, const LocalTensor<T> &input0,
                                              const LocalTensor<T> &input1, const int32_t curRowNum,
                                              const int32_t curColNum)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    uint32_t curColNumAlign = RoundUp<T>(curColNum);
    int32_t numRepeatPerLine = curColNum / elemInOneRepeat;
    int32_t numRemainPerLine = curColNum % elemInOneRepeat;
    int32_t dstRepStridePerLine = CeilDiv(curColNum, elemInOneBlock);
    BinaryRepeatParams instrParams;
    if (numRepeatPerLine > 0) {
        if (dstRepStridePerLine > MAX_REPEAT_STRIDE || curRowNum < numRepeatPerLine) {
            // 在Col方向开Repeat, 并且Repeat小于255
            instrParams.dstBlkStride = 1;
            instrParams.src0BlkStride = 1;
            instrParams.src1BlkStride = 1;
            instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
            instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
            instrParams.src1RepStride = DEFAULT_REPEAT_STRIDE;
            for (uint32_t i = 0; i < curRowNum; i++) {
                Add(output[i * curColNumAlign], input0[i * curColNumAlign], input1, elemInOneRepeat, numRepeatPerLine,
                    instrParams);
            }
        } else {
            // 在Row方向开Repeat
            instrParams.dstBlkStride = 1;
            instrParams.src0BlkStride = 1;
            instrParams.src1BlkStride = 1;
            instrParams.dstRepStride = dstRepStridePerLine;
            instrParams.src0RepStride = dstRepStridePerLine;
            instrParams.src1RepStride = 0;
            for (uint32_t i = 0; i < numRepeatPerLine; i++) {
                Add(output[i * elemInOneRepeat], input0[i * elemInOneRepeat], input1[i * elemInOneRepeat],
                    elemInOneRepeat, curRowNum, instrParams);
            }
        }
    }

    if (numRemainPerLine > 0) {
        if (dstRepStridePerLine > MAX_REPEAT_STRIDE) {
            instrParams.dstBlkStride = 1;
            instrParams.src0BlkStride = 1;
            instrParams.src1BlkStride = 1;
            instrParams.dstRepStride = 0;
            instrParams.src0RepStride = 0;
            instrParams.src1RepStride = 0;
            for (uint32_t i = 0; i < curRowNum; i++) {
                Add(output[numRepeatPerLine * elemInOneRepeat], input0[numRepeatPerLine * elemInOneRepeat], input1,
                    numRemainPerLine, 1, instrParams);
            }
        } else {
            instrParams.dstBlkStride = 1;
            instrParams.src0BlkStride = 1;
            instrParams.src1BlkStride = 1;
            instrParams.dstRepStride = dstRepStridePerLine;
            instrParams.src0RepStride = dstRepStridePerLine;
            instrParams.src1RepStride = 0;
            Add(output[numRepeatPerLine * elemInOneRepeat], input0[numRepeatPerLine * elemInOneRepeat], input1,
                numRemainPerLine, curRowNum, instrParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void CalcDenominator(const LocalTensor<T> &output, const LocalTensor<T> &input,
                                       const uint32_t calCount)
{
    Muls(output, input, static_cast<T>(-1.0), calCount);
    PipeBarrier<PIPE_V>();
    Exp(output, output, calCount);
    PipeBarrier<PIPE_V>();
    Adds(output, output, static_cast<T>(1.0), calCount);
    PipeBarrier<PIPE_V>();
}

// 暂时不处理repeat超限场景
template <typename T>
__aicore__ inline void SigmoidPerf(const LocalTensor<T> &output, const LocalTensor<T> &input,
                                   const LocalTensor<T> &tmpBuffer, const int64_t calCount)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    Duplicate(tmpBuffer, static_cast<T>(1.0), elemInOneBlock);
    CalcDenominator(output, input, calCount);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    int32_t numRepeatPerLine = calCount / elemInOneRepeat;
    int32_t numRemainPerLine = calCount % elemInOneRepeat;
    BinaryRepeatParams instrParams;
    instrParams.dstBlkStride = 1;
    instrParams.src0BlkStride = 0;
    instrParams.src1BlkStride = 1;
    instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
    instrParams.src0RepStride = 0;
    instrParams.src1RepStride = DEFAULT_REPEAT_STRIDE;
    Div(output, tmpBuffer, output, elemInOneRepeat, numRepeatPerLine, instrParams);
    if (numRemainPerLine != 0) {
        Div(output[numRepeatPerLine * elemInOneRepeat], tmpBuffer, output[numRepeatPerLine * elemInOneRepeat],
            numRemainPerLine, 1, instrParams);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ProcessPre(const LocalTensor<float> &preLocal, const LocalTensor<float> &mixLocal,
                                  const LocalTensor<float> &hcBaseLocal, const LocalTensor<float> &rsqrtLocal,
                                  const LocalTensor<float> &tmpBuffer0, const LocalTensor<float> &tmpBuffer1,
                                  float scale, float eps, const int32_t curRowNum, const int32_t curColNum)
{
    int32_t curColNumAlign = RoundUp<float>(curColNum);
    MulABLastDimBrcInline<float, true>(mixLocal, mixLocal, rsqrtLocal, tmpBuffer0, curRowNum, curColNum);
    Muls(mixLocal, mixLocal, scale, curRowNum * curColNumAlign);
    PipeBarrier<PIPE_V>();
    AddBAFirstDimBrcInline<float>(mixLocal, mixLocal, hcBaseLocal, curRowNum, curColNum);
    SigmoidPerf(preLocal, mixLocal, tmpBuffer1, curRowNum * curColNumAlign);
    Adds(preLocal, preLocal, eps, curRowNum * curColNumAlign);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ReduceSumARAPerf(const LocalTensor<float> &output, const LocalTensor<float> &input,
                                        const uint32_t dim0, const uint32_t dim1, const uint32_t dim2)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(float);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(float);
    uint32_t dim2Align = RoundUp<float>(dim2);

    // 拷贝第一个R到output上
    DataCopyParams copyParams;
    copyParams.blockCount = dim0;
    copyParams.blockLen = dim2Align / elemInOneBlock;
    copyParams.srcStride = (dim1 - 1) * (dim2Align / elemInOneBlock);
    copyParams.dstStride = 0;
    DataCopy(output, input, copyParams);
    PipeBarrier<PIPE_V>();
    uint32_t dim2RepeatTimes = dim2 / elemInOneRepeat;
    uint32_t dim2Reminder = dim2 % elemInOneRepeat;
    // 沿着dim2方向开repeat
    BinaryRepeatParams instrParams;
    instrParams.dstBlkStride = 1;
    instrParams.src0BlkStride = 1;
    instrParams.src1BlkStride = 1;
    instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
    instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
    instrParams.src1RepStride = DEFAULT_REPEAT_STRIDE;
    for (uint32_t i = 0; i < dim0; i++) {
        for (uint32_t j = 1; j < dim1; j++) {
            Add(output[i * dim2Align], output[i * dim2Align], input[i * dim1 * dim2Align + j * dim2Align],
                elemInOneRepeat, dim2RepeatTimes, instrParams);
            if (dim2Reminder != 0) {
                Add(output[i * dim2Align + dim2RepeatTimes * elemInOneRepeat],
                    output[i * dim2Align + dim2RepeatTimes * elemInOneRepeat],
                    input[i * dim1 * dim2Align + j * dim2Align + +dim2RepeatTimes * elemInOneRepeat], dim2Reminder, 1,
                    instrParams);
            }
            PipeBarrier<PIPE_V>();
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T0, typename T1>
__aicore__ inline void CastTwoDim(const LocalTensor<T0> &output, const LocalTensor<T1> &input, const uint32_t dim0,
                                  const uint32_t dim1)
{
    uint32_t dim1AlignT0 = RoundUp<T0>(dim1);
    uint32_t dim1AlignT1 = RoundUp<T1>(dim1);
    if constexpr (IsSameType<T1, bfloat16_t>::value && IsSameType<T0, float>::value) {
        for (uint32_t i = 0; i < dim0; i++) {
            Cast(output[i * dim1AlignT0], input[i * dim1AlignT1], AscendC::RoundMode::CAST_NONE, dim1);
        }
    } else {
        for (uint32_t i = 0; i < dim0; i++) {
            Cast(output[i * dim1AlignT0], input[i * dim1AlignT1], AscendC::RoundMode::CAST_RINT, dim1);
        }
    }
    PipeBarrier<PIPE_V>();
}


template <typename T>
__aicore__ void inline ProcessY(const LocalTensor<T> &yLocal, const LocalTensor<T> &xLocal,
                                const LocalTensor<float> &mix01Local, const LocalTensor<float> &hcBrcbLocal1,
                                const LocalTensor<float> &xCastLocal, const LocalTensor<float> &yCastLocal,
                                const uint32_t dim0, const uint32_t dim1, const uint32_t dim2)
{
    CastTwoDim(xCastLocal, xLocal, dim0 * dim1, dim2);
    MulABLastDimBrcInline<float, true>(xCastLocal, xCastLocal, mix01Local, hcBrcbLocal1, dim0 * dim1, dim2);
    ReduceSumARAPerf(yCastLocal, xCastLocal, dim0, dim1, dim2);
    CastTwoDim(yLocal, yCastLocal, dim0, dim2);
}


__aicore__ inline void ProcessPost(const LocalTensor<float> &postLocal, const LocalTensor<float> &mixLocal,
                                   const LocalTensor<float> &hcBaseLocal, const LocalTensor<float> &rsqrtLocal,
                                   const LocalTensor<float> &tmpBuffer0, const LocalTensor<float> &tmpBuffer1,
                                   float scale, const int32_t curRowNum, const int32_t curColNum)
{
    int32_t curColNumAlign = RoundUp<float>(curColNum);
    MulABLastDimBrcInline<float, false>(mixLocal, mixLocal, rsqrtLocal, tmpBuffer0, curRowNum, curColNum);
    Muls(mixLocal, mixLocal, scale, curRowNum * curColNumAlign);
    PipeBarrier<PIPE_V>();
    AddBAFirstDimBrcInline<float>(mixLocal, mixLocal, hcBaseLocal, curRowNum, curColNum);
    SigmoidPerf(postLocal, mixLocal, tmpBuffer1, curRowNum * curColNumAlign);
    Muls(postLocal, postLocal, static_cast<float>(2.0f), curRowNum * curColNumAlign);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void LastDimReduceMaxPerf(const LocalTensor<float> &output, const LocalTensor<float> &input,
                                            const uint32_t curRowNum, const uint32_t curColNum)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(float);
    WholeReduceMax(output, input, curColNum, curRowNum, 1, 1, CeilDiv(curColNum, elemInOneBlock),
                   ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void LastDimReduceSumPerf(const LocalTensor<float> &output, const LocalTensor<float> &input,
                                            const uint32_t curRowNum, const uint32_t curColNum)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(float);
    WholeReduceSum(output, input, curColNum, curRowNum, 1, 1, CeilDiv(curColNum, elemInOneBlock));
    PipeBarrier<PIPE_V>();
}


// 暂时只支持R轴小于64,既curColNum不能超过64
__aicore__ inline void SoftmaxFP32Perf(const LocalTensor<float> &output, const LocalTensor<float> &input,
                                       const LocalTensor<float> &tmpReduceBuffer,
                                       const LocalTensor<float> tmpBrcbBuffer, const int32_t curRowNum,
                                       const int32_t curColNum, float eps)
{
    LastDimReduceMaxPerf(tmpReduceBuffer, input, curRowNum, curColNum);
    SubABLastDimBrcInline<float, true>(output, input, tmpReduceBuffer, tmpBrcbBuffer, curRowNum, curColNum);
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    Exp(output, output, curRowNum * curColNumAlign);
    PipeBarrier<PIPE_V>();
    LastDimReduceSumPerf(tmpReduceBuffer, output, curRowNum, curColNum);
    DivABLastDimBrcInline<float, true>(output, output, tmpReduceBuffer, tmpBrcbBuffer, curRowNum, curColNum);
    Adds(output, output, eps, curRowNum * curColNumAlign);
    PipeBarrier<PIPE_V>();
}

// (bs, hc_mult, hc_mult) = (bs, hc_mult, hc_mult) +  (bs, 1, hc_mult)

template <typename T>
__aicore__ inline void DivABABrcInline(const LocalTensor<T> &output, const LocalTensor<T> &input0,
                                       const LocalTensor<T> &input1, const uint32_t dim0, const uint32_t dim1,
                                       const uint32_t dim2)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    uint32_t dim2Align = RoundUp<T>(dim2);
    uint32_t dim2RepeatTimes = dim2 / elemInOneRepeat;
    uint32_t dim2Reminder = dim2 % elemInOneRepeat;
    uint32_t dim2RepeatStride = CeilDiv(dim2, elemInOneBlock);
    // 在dim1方向开repeat
    BinaryRepeatParams instrParams;
    if (dim1 >= dim2RepeatTimes) {
        instrParams.dstBlkStride = 1;
        instrParams.src0BlkStride = 1;
        instrParams.src1BlkStride = 1;
        instrParams.dstRepStride = dim2RepeatStride;
        instrParams.src0RepStride = dim2RepeatStride;
        instrParams.src1RepStride = 0;
        for (uint32_t i = 0; i < dim0; i++) {
            for (uint32_t j = 0; j < dim2RepeatTimes; j++) {
                Div(output[i * dim1 * dim2Align + j * elemInOneRepeat],
                    input0[i * dim1 * dim2Align + j * elemInOneRepeat], input1[i * dim2Align + j * elemInOneRepeat],
                    elemInOneRepeat, dim1, instrParams);
            }
            if (dim2Reminder != 0) {
                Div(output[i * dim1 * dim2Align + dim2RepeatTimes * elemInOneRepeat],
                    input0[i * dim1 * dim2Align + dim2RepeatTimes * elemInOneRepeat],
                    input1[i * dim2Align + dim2RepeatTimes * elemInOneRepeat], dim2Reminder, dim1, instrParams);
            }
        }
    } else {
        // 在dim2方向开repeat
        instrParams.dstBlkStride = 1;
        instrParams.src0BlkStride = 1;
        instrParams.src1BlkStride = 1;
        instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
        instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
        instrParams.src1RepStride = DEFAULT_REPEAT_STRIDE;
        for (uint32_t i = 0; i < dim0; i++) {
            for (uint32_t j = 0; j < dim1; j++) {
                Div(output[i * dim1 * dim2Align + j * dim2Align], input0[i * dim1 * dim2Align + j * dim2Align],
                    input1[i * dim2Align], dim2);
            }
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void CopyIn(const GlobalTensor<T> &inputGm, const LocalTensor<T> &inputTensor, const uint16_t nBurst,
                              const uint32_t copyLen, uint32_t srcStride = 0)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
}

// (bs, hc_mult, hc_mult) --> (bs, hc_mult, hc_mult_align)
template <typename T>
__aicore__ inline void CopyInWithOuterFor(const GlobalTensor<T> &inputGm, const LocalTensor<T> &inputTensor,
                                          const uint16_t outerLoop, const uint16_t nBurst, const uint32_t copyLen,
                                          const uint32_t gmLastDim)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    uint32_t ubLastDimAlign = RoundUp<T>(copyLen);

    for (uint16_t i = 0; i < outerLoop; i++) {
        CopyIn(inputGm[i * nBurst * gmLastDim], inputTensor[i * nBurst * ubLastDimAlign], nBurst, copyLen);
    }
}

template <typename T>
__aicore__ inline void CopyOut(const LocalTensor<T> &outputTensor, const GlobalTensor<T> &outputGm,
                               const uint16_t nBurst, const uint32_t copyLen, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad(outputGm, outputTensor, dataCopyParams);
}

} // namespace HcPreSinkhorn

#endif