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
 * \file hc_pre_base.h
 * \brief
 */

#ifndef HC_PRE_VECTOR_BASE_H
#define HC_PRE_VECTOR_BASE_H

#include "kernel_operator.h"

namespace HcPre {
using namespace AscendC;

// ==================== A3架构常量定义 ====================
// 以下常量用于AscendC A3架构的Vector API参数配置
constexpr int32_t BLOCK_SIZE = 32;          // 一个Block的字节数（32 Bytes）
constexpr int32_t DEFAULT_BLOCK_STRIDE = 1; // 默认Block步长（相邻两个Block间隔1个Block）
constexpr int32_t DEFAULT_REPEAT_STRIDE = 8; // 默认Repeat步长（8个Block = 256 Bytes = 1个Repeat）
constexpr int32_t ONE_REPEAT_BLOCK_NUMS = 8; // 一个Repeat包含8个Block（256/32=8）
constexpr int32_t REPEAT_SIZE = 256;         // 一个Repeat的字节大小（256 Bytes）
constexpr int32_t MAX_REPEAT_STRIDE = 255;   // Repeat步长最大值（255个Block）
constexpr int32_t WORKSPACE_ALIGN_SIZE = 512; // Workspace地址对齐大小（512 Bytes）
constexpr uint32_t ONE = 1;                  // GatherMask掩码基础值（bit[0]=1）
constexpr uint32_t SHIFT_COEFF = 17;        // 移位系数
constexpr int32_t DOUBLE_BUFFER = 2;        // 双缓冲数量
constexpr int32_t CV_RATIO = 2;             // Cube:Vector核数比 = 1:2（1个AIC配2个AIV）
constexpr uint64_t N_SIZE = 24;
constexpr uint64_t NUM_TWO = 2;             // 常量2
constexpr uint64_t SQUARE_SUM_SIZE = 16;    // 平方和输出大小（NZ格式下M×M=16×16）
constexpr uint64_t MASK_PATTERN_BASE_SIZE = 16;   // GatherMask掩码基础维度大小
constexpr uint64_t MASK_PATTERN_REPEAT_SIZE = 8;  // 掩码重复子大小
constexpr uint64_t MASK_PATTERN_DIM_SIZE = 16;    // 掩码单次处理的维度大小
constexpr uint64_t MM_CACHE_LINE_BYTES = 512;      // MatMul缓存行大小（512 Bytes）

// ==================== 对齐工具函数 ====================

// 向上取整除法：返回(a + b - 1) / b
__aicore__ inline int32_t CeilDiv(int32_t a, int32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

// 向上对齐：将a向上对齐到b的倍数
__aicore__ inline int32_t CeilAlign(int32_t a, int32_t b)
{
    return CeilDiv(a, b) * b;
}

// 按类型T的Block元素数向上对齐：用于确保元素数满足32字节Block对齐要求
template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num)
{
    int32_t elemNum = BLOCK_SIZE / sizeof(T); // 每个Block中T类型元素的个数
    return CeilAlign(num, elemNum);
}

// ==================== GatherMask对角提取（用于RMSNorm平方和） ====================

// 设置GatherMask的循环对角掩码：生成 patternRows 行，第 i 行 base = 1 << (i % 16)
// 调用方按 rowFactor + MASK_PATTERN_DIM_SIZE - 1 行生成循环模式，
// 保证后续 GatherMask 单次调用中所有操作数保持 32 字节对齐
__aicore__ inline void SetGatherMaskPattern(const LocalTensor<uint32_t>& maskPattern, uint16_t patternRows)
{
    for (uint32_t i = 0; i < patternRows; i++) {
        uint32_t base = ONE << (i % MASK_PATTERN_DIM_SIZE);
        Duplicate(maskPattern[i * 8], base, 8);
    }
    PipeBarrier<PIPE_V>();
}

// 使用循环对角掩码从NZ格式的平方和结果中提取对角线元素（单次GatherMask完成dim0行）
// 输入input: NZ格式的平方和矩阵(dim0×16行×16列)
// 输出output: 提取的对角线元素(dim0个)
// maskPattern: 由SetGatherMaskPattern生成的循环对角掩码
// dim0: 需要处理的行数
// diagonalOffset: 当前行块在16×16 NZ块中的对角相位（globalRowOffset % SQUARE_SUM_SIZE）
__aicore__ inline void GatherMaskByDiagonal(const LocalTensor<float>& output, const LocalTensor<float>& input,
                                            const LocalTensor<uint32_t> maskPattern, uint16_t dim0,
                                            uint16_t diagonalOffset)
{
    uint16_t patternIdx = diagonalOffset % MASK_PATTERN_DIM_SIZE;
    uint64_t rsvdCnt = 0;
    // The caller provides dim0 + 15 cyclic patterns so all operands stay 32-byte aligned in one call.
    GatherMask(output, input, maskPattern[patternIdx * MASK_PATTERN_REPEAT_SIZE], true,
        MASK_PATTERN_DIM_SIZE, {1, dim0, 2, 1}, rsvdCnt);
    PipeBarrier<PIPE_V>();
}


// ==================== 末维广播乘法 ====================

// output[dim0, dim1] = input0[dim0, dim1] * broadcast(input1[1, dim1])
// 将一维向量input1沿第0维广播后与input0逐元素相乘
// 参数: curRowNum=行数(dim0), curColNum=列数(dim1)
// tmpBuffer用于存放Brcb广播后的input1
// needBrc: 是否需要执行Brcb广播（false时input1已广播好）
// 策略选择: 当列数较大时，根据dstRepStride是否超255来选择Col方向或Row方向开Repeat
template <typename T, bool needBrc = true>
__aicore__ inline void MulABLastDimBrcInline(const LocalTensor<T> &output, const LocalTensor<T> &input0,
                                             const LocalTensor<T> &input1, const LocalTensor<T> &tmpBuffer,
                                             const int32_t curRowNum, const int32_t curColNum)
{
    // 第一步：通过Brcb将input1沿第0维广播到curRowNum行
    if constexpr (needBrc) {
        uint32_t repeatTimes = CeilDiv(curRowNum, ONE_REPEAT_BLOCK_NUMS);
        Brcb(tmpBuffer, input1, repeatTimes, {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE});
    }
    PipeBarrier<PIPE_V>();
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);   // 一个Block中T类型元素个数
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);  // 一个Repeat中T类型元素个数
    uint32_t curColNumAlign = RoundUp<T>(curColNum);     // 列数向上对齐到Block
    // 简单场景：列数不超过一个Block，直接连续乘法
    if (curColNum <= elemInOneBlock) {
        Mul(output, input0, tmpBuffer, curRowNum * curColNumAlign);
    } else {
        // 复杂场景：列数超过一个Block，需要分Repeat处理
        int32_t numRepeatPerLine = curColNum / elemInOneRepeat; // 每行完整Repeat数
        int32_t numRemainPerLine = curColNum % elemInOneRepeat; // 每行尾部剩余元素数
        int32_t dstRepStridePerLine = CeilDiv(curColNum, elemInOneBlock); // 每行占多少个Block
        BinaryRepeatParams instrParams;

        if (numRepeatPerLine > 0) {
            // 判断是否可以用Row方向Repeat（条件：步长不超255且行数>=每行Repeat数）
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE || curRowNum < numRepeatPerLine) {
                // 策略1：在Col方向开Repeat（逐行处理），适用于步长超限或行数较少的场景
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;  // src1(广播向量)不随行移动
                instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
                instrParams.src1RepStride = 0;
                for (uint32_t i = 0; i < curRowNum; i++) {
                    Mul(output[i * curColNumAlign], input0[i * curColNumAlign], tmpBuffer[i * elemInOneBlock],
                        elemInOneRepeat, numRepeatPerLine, instrParams);
                }
            } else {
                // 策略2：在Row方向开Repeat（逐Repeat处理），利用repeatTimes=curRowNum提高指令并行度
                instrParams.dstBlkStride = 1;
                instrParams.src0BlkStride = 1;
                instrParams.src1BlkStride = 0;  // src1(广播向量)固定不变
                instrParams.dstRepStride = dstRepStridePerLine;
                instrParams.src0RepStride = dstRepStridePerLine;
                instrParams.src1RepStride = 1;  // src1按Block递增跟随
                for (uint32_t i = 0; i < numRepeatPerLine; i++) {
                    Mul(output[i * elemInOneRepeat], input0[i * elemInOneRepeat], tmpBuffer, elemInOneRepeat, curRowNum,
                        instrParams);
                }
            }
        }

        // 处理每行尾部不足一个Repeat的剩余元素
        if (numRemainPerLine > 0) {
            if (dstRepStridePerLine > MAX_REPEAT_STRIDE) {
                // 步长超限：逐行处理尾部元素
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
                // 步长不超限：一次性处理所有行的尾部元素
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

// ==================== 末维广播减法 ====================
// 与MulABLastDimBrcInline结构完全相同，仅将Mul替换为Sub
// output[dim0, dim1] = input0[dim0, dim1] - broadcast(input1[1, dim1])
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


// ==================== 末维广播除法 ====================
// 与MulABLastDimBrcInline结构完全相同，仅将Mul替换为Div
// output[dim0, dim1] = input0[dim0, dim1] / broadcast(input1[1, dim1])
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


// ==================== 首维广播加法 ====================

// output[dim0, dim1] = input0[dim0, dim1] + broadcast(input1[1, dim1])
// 将一维向量input1沿第0维广播后加到input0的每一行
// 与MulABLastDimBrcInline不同：此处input1不需要Brcb预广播，直接作为src1（src1BlkStride=1）
// 参数: curRowNum=行数(dim0), curColNum=列数(dim1)
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

// ==================== Sigmoid优化实现 ====================

// 计算sigmoid分母: output = 1 + exp(-x)
// 分三步: x取负(-1*x) → exp(-x) → +1
// 用于后续SigmoidPerf中的除法计算: sigmoid(x) = 1 / (1 + exp(-x))
template <typename T>
__aicore__ inline void CalcDenominator(const LocalTensor<T> &output, const LocalTensor<T> &input,
                                       const uint32_t calCount)
{
    Muls(output, input, static_cast<T>(-1.0), calCount); // output = -x
    PipeBarrier<PIPE_V>();
    Exp(output, output, calCount);                        // output = exp(-x)
    PipeBarrier<PIPE_V>();
    Adds(output, output, static_cast<T>(1.0), calCount);  // output = 1 + exp(-x)
    PipeBarrier<PIPE_V>();
}

// 优化版sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
// 相比原生Exp+Div实现，利用Duplicate生成常数1再用Div计算，减少一次标量运算
// 暂时不处理repeat超限场景
// 参数: calCount=元素总数, tmpBuffer用于存放常数1.0
template <typename T>
__aicore__ inline void SigmoidPerf(const LocalTensor<T> &output, const LocalTensor<T> &input,
                                   const LocalTensor<T> &tmpBuffer, const int64_t calCount)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    Duplicate(tmpBuffer, static_cast<T>(1.0), elemInOneBlock); // 生成常数1.0（只需一个Block大小）
    CalcDenominator(output, input, calCount);                  // output = 1 + exp(-x)
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(T);
    int32_t numRepeatPerLine = calCount / elemInOneRepeat;     // 完整Repeat数
    int32_t numRemainPerLine = calCount % elemInOneRepeat;    // 尾部剩余元素数
    BinaryRepeatParams instrParams;
    // src0(常数1)的Block步长=0, Repeat步长=0 → 保持固定；src1(分母)正常递进
    instrParams.dstBlkStride = 1;
    instrParams.src0BlkStride = 0;
    instrParams.src1BlkStride = 1;
    instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
    instrParams.src0RepStride = 0;
    instrParams.src1RepStride = DEFAULT_REPEAT_STRIDE;
    // 整除部分: output = 1 / (1 + exp(-x))
    Div(output, tmpBuffer, output, elemInOneRepeat, numRepeatPerLine, instrParams);
    // 尾部剩余部分
    if (numRemainPerLine != 0) {
        Div(output[numRepeatPerLine * elemInOneRepeat], tmpBuffer, output[numRepeatPerLine * elemInOneRepeat],
            numRemainPerLine, 1, instrParams);
    }
    PipeBarrier<PIPE_V>();
}

// ==================== 超连接前处理核心计算 ====================

// 计算pre = sigmoid(mix * rsqrt * scale + base) + eps
// 参数说明:
//   mixLocal: 输入混合向量(M×N)，输出为中间结果
//   rsqrtLocal: RMSNorm的1/sqrt(mean+eps)值，沿行广播
//   hcBaseLocal: base偏置值(base0)，沿行广播
//   scale: hc_scale[0]标量缩放系数
//   eps: 防止分母为0的小量
//   curRowNum, curColNum: 行数和列数
// 计算步骤: mix *= rsqrt → mix *= scale → mix += base → sigmoid → pre += eps
__aicore__ inline void ProcessPre(const LocalTensor<float> &preLocal, const LocalTensor<float> &mixLocal,
                                  const LocalTensor<float> &hcBaseLocal, const LocalTensor<float> &rsqrtLocal,
                                  const LocalTensor<float> &tmpBuffer0, const LocalTensor<float> &tmpBuffer1,
                                  float scale, float eps, const int32_t curRowNum, const int32_t curColNum)
{
    int32_t curColNumAlign = RoundUp<float>(curColNum);
    MulABLastDimBrcInline<float, true>(mixLocal, mixLocal, rsqrtLocal, tmpBuffer0, curRowNum, curColNum); // mix *= rsqrt
    Muls(mixLocal, mixLocal, scale, curRowNum * curColNumAlign);                                          // mix *= scale
    PipeBarrier<PIPE_V>();
    AddBAFirstDimBrcInline<float>(mixLocal, mixLocal, hcBaseLocal, curRowNum, curColNum);                // mix += base
    SigmoidPerf(preLocal, mixLocal, tmpBuffer1, curRowNum * curColNumAlign);                              // pre = sigmoid(mix)
    Adds(preLocal, preLocal, eps, curRowNum * curColNumAlign);                                           // pre += eps
    PipeBarrier<PIPE_V>();
}

// ==================== 沿dim1轴归约求和 ====================

// All-Reduce-Along dim1: output[dim0, dim2] = Σ_j input[dim0, j, dim2]
// 将3D张量(dim0, dim1, dim2)沿中间维度dim1求和，输出2D张量(dim0, dim2)
// 用于K-split多核归约：多个核各算一部分K，最后在此求和合并
// 参数: dim0=批次, dim1=K-split核数(归约轴), dim2=每核输出大小
__aicore__ inline void ReduceSumARAPerf(const LocalTensor<float> &output, const LocalTensor<float> &input,
                                        const uint32_t dim0, const uint32_t dim1, const uint32_t dim2)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(float);
    uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(float);
    uint32_t dim2Align = RoundUp<float>(dim2);

    // 拷贝第一个R到output上（作为累加初始值）
    DataCopyParams copyParams;
    copyParams.blockCount = dim0;
    copyParams.blockLen = dim2Align / elemInOneBlock;
    copyParams.srcStride = (dim1 - 1) * (dim2Align / elemInOneBlock); // 跳过dim1-1个dim2块
    copyParams.dstStride = 0;
    DataCopy(output, input, copyParams);
    PipeBarrier<PIPE_V>();
    uint32_t dim2RepeatTimes = dim2 / elemInOneRepeat;
    uint32_t dim2Reminder = dim2 % elemInOneRepeat;
    // 沿着dim2方向开repeat（每个元素累加）
    BinaryRepeatParams instrParams;
    instrParams.dstBlkStride = 1;
    instrParams.src0BlkStride = 1;
    instrParams.src1BlkStride = 1;
    instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;
    instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
    instrParams.src1RepStride = DEFAULT_REPEAT_STRIDE;
    // 对每个dim0批次，遍历dim1(从1开始)累加到output。
    // 连续 Add 对同一 output 读改写时，后一条指令可能在前一条写回前读取数据。
    // 每轮累加后同步 Vector 流水，避免归约结果错误。
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

// Sinkhorn 的列归约固定为 4x4。专用路径保持与通用归约相同的
// 左到右累加顺序，但省去首行 DataCopy 和每个加数后的 barrier。
__aicore__ inline void ReduceSum4x4Perf(const LocalTensor<float> &output,
                                        const LocalTensor<float> &input,
                                        const uint32_t dim0)
{
    constexpr uint32_t dim2 = 4;
    const uint32_t dim2Align = RoundUp<float>(dim2);
    for (uint32_t i = 0; i < dim0; i++) {
        const uint32_t inOffset = i * dim2 * dim2Align;
        const uint32_t outOffset = i * dim2Align;
        Add(output[outOffset], input[inOffset], input[inOffset + dim2Align], dim2);
        Add(output[outOffset], output[outOffset], input[inOffset + 2 * dim2Align], dim2);
        Add(output[outOffset], output[outOffset], input[inOffset + 3 * dim2Align], dim2);
    }
    PipeBarrier<PIPE_V>();
}

// ==================== 二维Cast类型转换 ====================

// 将2D张量从T1类型转换到T0类型，逐行处理
// 参数: dim0=行数, dim1=列数
// bfloat16→float使用CAST_NONE（无损），其他类型使用CAST_RINT（四舍五入）
// syncAfter=false: 跳过尾部PipeBarrier, 由调用方合并同步点或依赖队列
// 事件(EnQue的V_MTE3)兜底——用于连续Cast与Brcb等dst互不重叠的指令
// 并行发射后统一一次屏障(见ProcessYHc4), 减少热路径屏障密度.
template <typename T0, typename T1>
__aicore__ inline void CastTwoDim(const LocalTensor<T0> &output, const LocalTensor<T1> &input, const uint32_t dim0,
                                  const uint32_t dim1, const bool syncAfter = true)
{
    uint32_t dim1AlignT0 = RoundUp<T0>(dim1); // 输出类型对齐后的列数
    uint32_t dim1AlignT1 = RoundUp<T1>(dim1); // 输入类型对齐后的列数
    // 对齐的连续二维 tile 可以一次完成 Cast，避免在 Stage1 和 ProcessY
    // 中为每一行生成独立的 Vector 指令与循环控制。
    if (dim1AlignT0 == dim1 && dim1AlignT1 == dim1) {
        if constexpr (IsSameType<T1, bfloat16_t>::value && IsSameType<T0, float>::value) {
            Cast(output, input, AscendC::RoundMode::CAST_NONE, dim0 * dim1);
        } else {
            Cast(output, input, AscendC::RoundMode::CAST_RINT, dim0 * dim1);
        }
        if (syncAfter) {
            PipeBarrier<PIPE_V>();
        }
        return;
    }
    if constexpr (IsSameType<T1, bfloat16_t>::value && IsSameType<T0, float>::value) {
        // bf16→fp32: 无需舍入模式，直接转换（无损扩展）
        for (uint32_t i = 0; i < dim0; i++) {
            Cast(output[i * dim1AlignT0], input[i * dim1AlignT1], AscendC::RoundMode::CAST_NONE, dim1);
        }
    } else {
        // 其他类型转换: 使用CAST_RINT（四舍五入到最近偶数）
        for (uint32_t i = 0; i < dim0; i++) {
            Cast(output[i * dim1AlignT0], input[i * dim1AlignT1], AscendC::RoundMode::CAST_RINT, dim1);
        }
    }
    if (syncAfter) {
        PipeBarrier<PIPE_V>();
    }
}


// ==================== 计算输出y ====================

// 计算 y = Σ_{j=0}^{hc_mult-1} (weight[:,j] * x[:,j,:])
// 权重由调用方传入：普通路径使用 pre，preMix 路径使用 preMix。
// hc_mult=4 时使用融合乘加；其他情况逐元素乘权重后沿 hc_mult 维度归约。
// 参数: dim0=batch数, dim1=hc_mult, dim2=d_factor
// Brcb 后每路权重仅占一个 Block；src1BlkStride=0 使权重在整行内复用，
// 避免按连续地址读取时跨入下一路权重。
// 每个 pass 处理一路权重；pass 间用 PipeBarrier 同步对 y 的读改写。
template <typename T>
__aicore__ void inline ProcessYHc4(const LocalTensor<T> &yLocal, const LocalTensor<T> &xLocal,
                                    const LocalTensor<float> &mix01Local, const LocalTensor<float> &hcBrcbLocal1,
                                    const LocalTensor<float> &xCastLocal, const LocalTensor<float> &yCastLocal,
                                    const uint32_t dim0, const uint32_t dim2,
                                    const bool needBrc = true)
{
    const uint32_t rows = dim0 * 4;
    const uint32_t dim2Align = RoundUp<float>(dim2);
    const uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(float);
    const uint32_t elemInOneRepeat = REPEAT_SIZE / sizeof(float);
    const uint32_t repeatTimes = CeilDiv(rows, ONE_REPEAT_BLOCK_NUMS);

    // Cast(xCast)与Brcb(hcBrcbLocal1)写不同dst, 可并行发射, 合并为一个
    // 同步点(Mul需两者都完成才可取数)——相比各自屏障省一次V流水排空.
    // 尾部Cast(yLocal)同样免屏障: Pass3后的屏障已保证MulAddDst写回,
    // yLocal→CopyOut的可见性由yQue.EnQue的V_MTE3事件保证.
    CastTwoDim(xCastLocal, xLocal, rows, dim2, /*syncAfter=*/false);
    if (needBrc) {
        Brcb(hcBrcbLocal1, mix01Local, repeatTimes,
             {DEFAULT_BLOCK_STRIDE, DEFAULT_REPEAT_STRIDE});
    }
    PipeBarrier<PIPE_V>();
    BinaryRepeatParams instrParams;
    instrParams.dstBlkStride = 1;
    instrParams.src0BlkStride = 1;
    instrParams.src1BlkStride = 0;             // 权重块(8个相同值)在行内所有Block间复用
    instrParams.dstRepStride = DEFAULT_REPEAT_STRIDE;  // repeat 沿列方向连续推进
    instrParams.src0RepStride = DEFAULT_REPEAT_STRIDE;
    instrParams.src1RepStride = 0;
    const uint32_t nRep = dim2 / elemInOneRepeat;
    const uint32_t rem = dim2 % elemInOneRepeat;

    // Pass 0: 第0路乘法 y = x0 * w0
    for (uint32_t i = 0; i < dim0; i++) {
        const uint32_t xOffset = i * 4 * dim2Align;
        const uint32_t yOffset = i * dim2Align;
        const uint32_t weightOffset = i * 4 * elemInOneBlock;
        if (nRep > 0) {
            Mul(yCastLocal[yOffset], xCastLocal[xOffset], hcBrcbLocal1[weightOffset],
                elemInOneRepeat, nRep, instrParams);
        }
        if (rem > 0) {
            Mul(yCastLocal[yOffset + nRep * elemInOneRepeat],
                xCastLocal[xOffset + nRep * elemInOneRepeat], hcBrcbLocal1[weightOffset],
                rem, 1, instrParams);
        }
    }
    PipeBarrier<PIPE_V>();
    // Pass k=1..3: 乘加 y += xk * wk (pass 内各行独立, pass 间屏障隔离读改写)
    for (uint32_t k = 1; k < 4; k++) {
        for (uint32_t i = 0; i < dim0; i++) {
            const uint32_t xOffset = i * 4 * dim2Align + k * dim2Align;
            const uint32_t yOffset = i * dim2Align;
            const uint32_t weightOffset = i * 4 * elemInOneBlock + k * elemInOneBlock;
            if (nRep > 0) {
                MulAddDst(yCastLocal[yOffset], xCastLocal[xOffset], hcBrcbLocal1[weightOffset],
                          static_cast<uint64_t>(elemInOneRepeat), nRep, instrParams);
            }
            if (rem > 0) {
                MulAddDst(yCastLocal[yOffset + nRep * elemInOneRepeat],
                          xCastLocal[xOffset + nRep * elemInOneRepeat], hcBrcbLocal1[weightOffset],
                          static_cast<uint64_t>(rem), 1, instrParams);
            }
        }
        PipeBarrier<PIPE_V>();
    }
    // 尾Cast免屏障(见函数头说明): yLocal的GM可见性由yQue.EnQue事件保证.
    CastTwoDim(yLocal, yCastLocal, dim0, dim2, /*syncAfter=*/false);
}

template <typename T>
__aicore__ void inline ProcessY(const LocalTensor<T> &yLocal, const LocalTensor<T> &xLocal,
                                const LocalTensor<float> &mix01Local, const LocalTensor<float> &hcBrcbLocal1,
                                const LocalTensor<float> &xCastLocal, const LocalTensor<float> &yCastLocal,
                                const uint32_t dim0, const uint32_t dim1, const uint32_t dim2,
                                const bool needBrc = true)
{
    if (dim1 == 4) {
        ProcessYHc4(yLocal, xLocal, mix01Local, hcBrcbLocal1,
                    xCastLocal, yCastLocal, dim0, dim2, needBrc);
        return;
    }
    CastTwoDim(xCastLocal, xLocal, dim0 * dim1, dim2);                                    // 1. x bf16→fp32
    MulABLastDimBrcInline<float, true>(xCastLocal, xCastLocal, mix01Local, hcBrcbLocal1, dim0 * dim1, dim2); // 2. x_cast *= 输入权重
    ReduceSumARAPerf(yCastLocal, xCastLocal, dim0, dim1, dim2);                           // 3. Σ沿hc_mult维度
    CastTwoDim(yLocal, yCastLocal, dim0, dim2);                                           // 4. y fp32→bf16
}


// ==================== 计算输出post ====================

// 计算post = sigmoid(mix * rsqrt * scale + base) * 2
// 对应计算公式步骤4中的post部分
// 与ProcessPre结构相同，但最终是*2而非+eps
// 参数说明:
//   mixLocal: 输入混合向量(M×N)，输出为中间结果
//   rsqrtLocal: RMSNorm的1/sqrt(mean+eps)值，沿行广播
//   hcBaseLocal: base偏置值(base1)，沿行广播
//   scale: hc_scale[1]标量缩放系数
//   needBrc=false: rsqrtLocal已由ProcessPre阶段广播好，无需再次Brcb
// 计算步骤: mix *= rsqrt → mix *= scale → mix += base → sigmoid → post *= 2
__aicore__ inline void ProcessPost(const LocalTensor<float> &postLocal, const LocalTensor<float> &mixLocal,
                                   const LocalTensor<float> &hcBaseLocal, const LocalTensor<float> &rsqrtLocal,
                                   const LocalTensor<float> &tmpBuffer0, const LocalTensor<float> &tmpBuffer1,
                                   float scale, const int32_t curRowNum, const int32_t curColNum)
{
    int32_t curColNumAlign = RoundUp<float>(curColNum);
    MulABLastDimBrcInline<float, false>(mixLocal, mixLocal, rsqrtLocal, tmpBuffer0, curRowNum, curColNum); // mix *= rsqrt (无需Brcb)
    Muls(mixLocal, mixLocal, scale, curRowNum * curColNumAlign);                                          // mix *= scale
    PipeBarrier<PIPE_V>();
    AddBAFirstDimBrcInline<float>(mixLocal, mixLocal, hcBaseLocal, curRowNum, curColNum);                // mix += base
    SigmoidPerf(postLocal, mixLocal, tmpBuffer1, curRowNum * curColNumAlign);                             // post = sigmoid(mix)
    Muls(postLocal, postLocal, static_cast<float>(2.0f), curRowNum * curColNumAlign);                   // post *= 2
    PipeBarrier<PIPE_V>();
}

// ==================== 末维归约Max/Sum ====================

// 沿最后一维求最大值: output[curRowNum] = max(input[curRowNum, curColNum])
// 使用WholeReduceMax，输出仅为值（不含索引）
__aicore__ inline void LastDimReduceMaxPerf(const LocalTensor<float> &output, const LocalTensor<float> &input,
                                            const uint32_t curRowNum, const uint32_t curColNum)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(float);
    WholeReduceMax(output, input, curColNum, curRowNum, 1, 1, CeilDiv(curColNum, elemInOneBlock),
                   ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
}

// 沿最后一维求和: output[curRowNum] = Σ input[curRowNum, curColNum]
// 使用WholeReduceSum
__aicore__ inline void LastDimReduceSumPerf(const LocalTensor<float> &output, const LocalTensor<float> &input,
                                            const uint32_t curRowNum, const uint32_t curColNum)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(float);
    WholeReduceSum(output, input, curColNum, curRowNum, 1, 1, CeilDiv(curColNum, elemInOneBlock));
    PipeBarrier<PIPE_V>();
}


// ==================== FP32 Softmax ====================

// 沿最后一维计算softmax: output = softmax(input) + eps
// 对应计算公式步骤4中的comb_frag部分（softmax+Sinkhorn迭代的第一步）
// 暂时只支持R轴(末维)小于64, 即curColNum不能超过64
// 计算步骤: max → sub(max) → exp → sum → div(sum) → +eps
// 参数: curRowNum=行数, curColNum=列数(=hc_mult), eps=防0小量
__aicore__ inline void SoftmaxFP32Perf(const LocalTensor<float> &output, const LocalTensor<float> &input,
                                       const LocalTensor<float> &tmpReduceBuffer,
                                       const LocalTensor<float> tmpBrcbBuffer, const int32_t curRowNum,
                                       const int32_t curColNum, float eps)
{
    LastDimReduceMaxPerf(tmpReduceBuffer, input, curRowNum, curColNum);                                // 1. 求每行最大值
    SubABLastDimBrcInline<float, true>(output, input, tmpReduceBuffer, tmpBrcbBuffer, curRowNum, curColNum); // 2. 减去最大值（数值稳定）
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    Exp(output, output, curRowNum * curColNumAlign);                                                   // 3. 求exp
    PipeBarrier<PIPE_V>();
    LastDimReduceSumPerf(tmpReduceBuffer, output, curRowNum, curColNum);                               // 4. 求每行和
    DivABLastDimBrcInline<float, true>(output, output, tmpReduceBuffer, tmpBrcbBuffer, curRowNum, curColNum); // 5. 除以和（归一化）
    Adds(output, output, eps, curRowNum * curColNumAlign);                                             // 6. +eps（用于Sinkhorn迭代防0）
    PipeBarrier<PIPE_V>();
}

// ==================== 三维广播除法（用于Sinkhorn迭代） ====================

// output[dim0, dim1, dim2] = input0[dim0, dim1, dim2] / input1[dim0, 1, dim2]
// input1沿dim1方向广播后与input0逐元素相除
// 用于Sinkhorn迭代的行归一化: comb_frag(bs, hc_mult, hc_mult) /= row_sum(bs, 1, hc_mult)
// 参数: dim0=bs, dim1=hc_mult, dim2=hc_mult
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
        // 策略1：在dim1方向开Repeat（src1的Repeat步长=0，沿dim1不移动）
        instrParams.dstBlkStride = 1;
        instrParams.src0BlkStride = 1;
        instrParams.src1BlkStride = 1;
        instrParams.dstRepStride = dim2RepeatStride;
        instrParams.src0RepStride = dim2RepeatStride;
        instrParams.src1RepStride = 0; // src1沿dim1不移动（广播）
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
        // 策略2：在dim2方向开repeat（逐dim1行处理，src1固定为对应dim0的行向量）
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

// ==================== GM↔UB数据搬运工具 ====================

// 从Global Memory拷贝数据到UB（LocalTensor）
// 使用DataCopyPad支持非对齐长度的数据拷贝
// 参数: nBurst=数据块数(行数), copyLen=每行元素数, srcStride=源行间间隔, dstStride=目的行间间隔(以Block为单位)
template <typename T>
__aicore__ inline void CopyIn(const GlobalTensor<T> &inputGm, const LocalTensor<T> &inputTensor, const uint16_t nBurst,
                              const uint32_t copyLen, uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = dstStride / elemInOneBlock;
    DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
}

// 3D数据拷贝: (outerLoop, nBurst, gmLastDim) → UB (outerLoop, nBurst, ubLastDimAlign)
// 外层循环按outerLoop遍历，每层调用CopyIn拷贝nBurst行
// 适用于GM和UB最后一维长度不同（UB有对齐padding）的场景
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

// 4D数据拷贝: (outerLoop, gmFirstDim, gmLastDim) → UB (outerLoop, nBurst, ubLastDimAlign)
// 支持非连续内存访问模式，根据outerLoop与nBurst的大小关系选择不同遍历策略
// 当outerLoop <= nBurst时：按outerLoop遍历，每次取nBurst行
// 当outerLoop > nBurst时：按nBurst遍历，每次取outerLoop个分散数据段
template <typename T>
__aicore__ inline void CopyInWithOuterFor(const GlobalTensor<T> &inputGm,
const LocalTensor<T> &inputTensor, const uint16_t outerLoop,
const uint16_t nBurst, const uint32_t copyLen, const uint32_t gmFirstDim,
const uint32_t gmLastDim)
{
    uint32_t elemInOneBlock = BLOCK_SIZE / sizeof(T);
    uint32_t ubLastDimAlign = RoundUp<T>(copyLen);
    if (outerLoop <= nBurst) {
        for (uint32_t i = 0; i < outerLoop; i++) {
            CopyIn(inputGm[i * gmFirstDim * gmLastDim],
            inputTensor[i * nBurst * ubLastDimAlign], nBurst, copyLen,
            gmLastDim - copyLen);
        }
    } else {
        uint32_t srcStride = (gmLastDim - copyLen) + (gmFirstDim - 1) * gmLastDim;
        uint32_t dstStride = (nBurst - 1) * ubLastDimAlign;
        for (uint32_t i = 0; i < nBurst; i++) {
            CopyIn(inputGm[i * gmLastDim], inputTensor[i * ubLastDimAlign], outerLoop, copyLen, srcStride, dstStride);
        }
    }
}

// 从UB（LocalTensor）拷贝数据到Global Memory
// 参数: nBurst=数据块数(行数), copyLen=每行元素数, dstStride=目的行间间隔(以字节为单位)
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

} // namespace HcPre

#endif
