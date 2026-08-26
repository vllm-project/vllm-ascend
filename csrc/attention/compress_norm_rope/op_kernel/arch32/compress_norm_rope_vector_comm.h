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
 * \file compress_norm_rope_vector_comm.h
 * \brief vector 公共组件：ColumnSoftMax/ColumnSum/RmsNorm/RoPE
 */

#ifndef COMPRESS_NORM_ROPE_VECTOR_COMM_H
#define COMPRESS_NORM_ROPE_VECTOR_COMM_H

#include "compress_norm_rope_comm.h"

namespace CompressNormRope {

struct MatRepeatParam {
    uint32_t row;
    uint32_t col;
    uint32_t dtypeMask;
    uint32_t loopTimes;
    uint32_t colRemain;
    uint8_t repeatStride;
};

/**
 * @brief ColumnSum 对矩阵按列进行求和
 * @param dstLocal 输出tensor [1, col]，支持和shareTmpUb是同一块空间
 * @param srcLocal 输入tensor [row, col]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [ceil(row / 2) * col * sizeof(float)]
 */
__aicore__ inline void ColumnSum(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                 const LocalTensor<float> &shareTmpUb, uint32_t row, uint32_t col)
{
    if (unlikely(row == 1)) {
        DataCopy(dstLocal, srcLocal, row * col);
        PipeBarrier<PIPE_V>();
        return;
    }
    for (uint32_t mask = MAX_R << 1; mask > 1; mask >>= 1) {
        if (row & mask) {
            Add(shareTmpUb, srcLocal, srcLocal[mask * col / 2], mask * col / 2); // 2:对矩阵按列做计算
            PipeBarrier<PIPE_V>();
            if (unlikely(row > mask)) {
                if ((row - mask) > (mask >> 1)) {
                    Add(shareTmpUb, shareTmpUb, srcLocal[mask * col], mask * col / 2); // 2:对矩阵按列做计算
                    PipeBarrier<PIPE_V>();
                    Add(shareTmpUb, shareTmpUb, srcLocal[(mask + (mask >> 1)) * col], (row - mask - (mask >> 1)) * col);
                    PipeBarrier<PIPE_V>();
                } else {
                    Add(shareTmpUb, shareTmpUb, srcLocal[mask * col], (row - mask) * col);
                    PipeBarrier<PIPE_V>();
                }
            }
            for (uint32_t i = mask >> 2; i > 1; i >>= 1) {
                Add(shareTmpUb, shareTmpUb, shareTmpUb[i * col], i * col);
                PipeBarrier<PIPE_V>();
            }
            if (mask == 2) { // 2:最后一次矩阵运算处理
                DataCopy(dstLocal, shareTmpUb, col);
            } else {
                Add(dstLocal, shareTmpUb, shareTmpUb[col], col);
            }
            PipeBarrier<PIPE_V>();
            break;
        }
    }
}

/**
 * @brief ColumnMax 对矩阵按列进行求最大值
 * @param dstLocal 输出tensor [1, col]，支持和shareTmpUb是同一块空间
 * @param srcLocal 输入tensor [row, col]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [ceil(row / 2) * col * sizeof(float)]
 */
__aicore__ inline void ColumnMax(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                 const LocalTensor<float> &shareTmpUb, uint32_t row, uint32_t col)
{
    if (unlikely(row == 1)) {
        DataCopy(dstLocal, srcLocal, row * col);
        PipeBarrier<PIPE_V>();
        return;
    }
    for (uint32_t mask = MAX_R << 1; mask > 1; mask >>= 1) {
        if (row & mask) {
            Max(shareTmpUb, srcLocal, srcLocal[mask * col / 2], mask * col / 2); // 2:对矩阵按列做计算
            PipeBarrier<PIPE_V>();
            if (unlikely(row > mask)) {
                if ((row - mask) > (mask >> 1)) {
                    Max(shareTmpUb, shareTmpUb, srcLocal[mask * col], mask * col / 2); // 2:对矩阵按列做计算
                    PipeBarrier<PIPE_V>();
                    Max(shareTmpUb, shareTmpUb, srcLocal[(mask + (mask >> 1)) * col], (row - mask - (mask >> 1)) * col);
                    PipeBarrier<PIPE_V>();
                } else {
                    Max(shareTmpUb, shareTmpUb, srcLocal[mask * col], (row - mask) * col);
                    PipeBarrier<PIPE_V>();
                }
            }
            for (uint32_t i = mask >> 2; i > 1; i >>= 1) {
                Max(shareTmpUb, shareTmpUb, shareTmpUb[i * col], i * col);
                PipeBarrier<PIPE_V>();
            }
            if (mask == 2) { // 2:最后一次矩阵运算处理
                DataCopy(dstLocal, shareTmpUb, col);
            } else {
                Max(dstLocal, shareTmpUb, shareTmpUb[col], col);
            }
            PipeBarrier<PIPE_V>();
            break;
        }
    }
}

/**
 * @brief MatSubVec 矩阵逐行减向量
 */
__aicore__ inline void MatSubVec(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
                                 const LocalTensor<float> &src1Local, const MatRepeatParam &repeatParam)
{
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = Std::min(repeatParam.row - row, REPEAT_MAX_NUM);
        uint32_t offset = 0;
        for (uint32_t i = 0; i < repeatParam.loopTimes; i++) {
            Sub(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.colRemain > 0) {
            Sub(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.colRemain, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
        }
    }
}

/**
 * @brief MatDivVec 矩阵逐行除以向量
 */
__aicore__ inline void MatDivVec(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
                                 const LocalTensor<float> &src1Local, const MatRepeatParam &repeatParam)
{
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = Std::min(repeatParam.row - row, REPEAT_MAX_NUM);
        uint32_t offset = 0;
        for (uint32_t i = 0; i < repeatParam.loopTimes; i++) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.colRemain > 0) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.colRemain, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
        }
    }
}

/**
 * @brief ColumnSoftMax 对矩阵按列进行SoftMax
 * @param dstLocal 输出tensor [row, col]，支持和srcLocal是同一块空间
 * @param srcLocal 输入tensor [row, col]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [floor(row / 2) * col * sizeof(float)]
 */
__aicore__ inline void ColumnSoftMax(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                     const LocalTensor<float> &shareTmpUb, uint32_t row, uint32_t col)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = col / dtypeMask;
    uint32_t dRemain = col % dtypeMask;
    uint8_t repeatStride = col / FP32_BLOCK_ELEMENT_NUM;
    ColumnMax(shareTmpUb, srcLocal, shareTmpUb, row, col);
    PipeBarrier<PIPE_V>();
    MatSubVec(dstLocal, srcLocal, shareTmpUb, {row, col, dtypeMask, dLoop, dRemain, repeatStride});
    PipeBarrier<PIPE_V>();
    Exp(dstLocal, dstLocal, row * col);
    PipeBarrier<PIPE_V>();
    ColumnSum(shareTmpUb, dstLocal, shareTmpUb, row, col);
    PipeBarrier<PIPE_V>();
    MatDivVec(dstLocal, dstLocal, shareTmpUb, {row, col, dtypeMask, dLoop, dRemain, repeatStride});
}

// ───────────────────── RmsNorm / RoPE（压缩行后处理，语义对齐 vllm-ascend）─────────────────────

struct RmsNormParam {
    float reciprocal; // 1/col
    float epsilon;
    uint32_t row;
    uint32_t col;
};

// 矩阵逐行乘向量
__aicore__ inline void MatMulVec(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
                                 const LocalTensor<float> &src1Local, const MatRepeatParam &repeatParam)
{
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = Std::min(repeatParam.row - row, REPEAT_MAX_NUM);
        uint32_t offset = 0;
        for (uint32_t i = 0; i < repeatParam.loopTimes; i++) {
            Mul(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.colRemain > 0) {
            Mul(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local[offset],
                repeatParam.colRemain, repeatRowTimes,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, 0});
        }
    }
}

// 矩阵每行求和；srcLocal 与 shareTmpUb 必须别名同一块空间
__aicore__ inline void RowSum(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                              const LocalTensor<float> &shareTmpUb, const MatRepeatParam &repeatParam)
{
    uint32_t blockCount = repeatParam.loopTimes;
    if (blockCount > 0 && repeatParam.colRemain > 0) {
        Add(shareTmpUb, srcLocal, srcLocal[blockCount * repeatParam.dtypeMask], repeatParam.colRemain,
            repeatParam.row,
            {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, repeatParam.repeatStride});
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t loopCount = blockCount >> 1; loopCount > 0; loopCount = blockCount >> 1) {
        blockCount = (blockCount + 1) >> 1;
        for (uint32_t i = 0; i < loopCount; i++) {
            Add(shareTmpUb[i * repeatParam.dtypeMask], srcLocal[i * repeatParam.dtypeMask],
                srcLocal[(i + blockCount) * repeatParam.dtypeMask], repeatParam.dtypeMask, repeatParam.row,
                {1, 1, 1, repeatParam.repeatStride, repeatParam.repeatStride, repeatParam.repeatStride});
        }
        PipeBarrier<PIPE_V>();
    }
    WholeReduceSum(dstLocal, shareTmpUb,
                   (repeatParam.col < repeatParam.dtypeMask) ? repeatParam.col : repeatParam.dtypeMask,
                   repeatParam.row, 1, 1, repeatParam.repeatStride);
}

// 矩阵每行除以对应元素（src1Local 需扩展到 [row, FP32_BLOCK_ELEMENT_NUM]）
__aicore__ inline void RowDivs(const LocalTensor<float> &dstLocal, const LocalTensor<float> &src0Local,
                               const LocalTensor<float> &src1Local, const MatRepeatParam &repeatParam)
{
    for (uint32_t row = 0; row < repeatParam.row; row += REPEAT_MAX_NUM) {
        uint32_t repeatRowTimes = Std::min(repeatParam.row - row, REPEAT_MAX_NUM);
        uint32_t offset = 0;
        for (uint32_t i = 0; i < repeatParam.loopTimes; i++) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local,
                repeatParam.dtypeMask, repeatRowTimes,
                {1, 1, 0, repeatParam.repeatStride, repeatParam.repeatStride, 1});
            offset += repeatParam.dtypeMask;
        }
        if (repeatParam.colRemain > 0) {
            Div(dstLocal[row * repeatParam.col + offset], src0Local[row * repeatParam.col + offset], src1Local,
                repeatParam.colRemain, repeatRowTimes,
                {1, 1, 0, repeatParam.repeatStride, repeatParam.repeatStride, 1});
        }
    }
}

// RmsNorm：dst = src / sqrt(mean(src^2)+eps) * gamma
// shareTmpUb 需要 [(row*col + row) * sizeof(float)]；dst/src 可同空间
__aicore__ inline void RmsNorm(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                               const LocalTensor<float> &gammaLocal, const LocalTensor<float> &shareTmpUb,
                               const RmsNormParam &rmsNormParams)
{
    uint64_t cnt = rmsNormParams.row * rmsNormParams.col;
    LocalTensor<float> temp1Local = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> temp2Local = temp1Local[cnt];

    Mul(temp1Local, srcLocal, srcLocal, cnt);
    PipeBarrier<PIPE_V>();

    MatRepeatParam repeatParams = {
        rmsNormParams.row,                                          // row
        rmsNormParams.col,                                          // col
        FP32_REPEAT_ELEMENT_NUM,                                    // dtypeMask
        rmsNormParams.col / FP32_REPEAT_ELEMENT_NUM,                // loopTimes
        rmsNormParams.col % FP32_REPEAT_ELEMENT_NUM,                // colRemain
        static_cast<uint8_t>(rmsNormParams.col / FP32_BLOCK_ELEMENT_NUM), // repeatStride
    };

    RowSum(temp2Local, temp1Local, temp1Local, repeatParams);
    PipeBarrier<PIPE_V>();
    Muls(temp2Local, temp2Local, rmsNormParams.reciprocal, rmsNormParams.row);
    PipeBarrier<PIPE_V>();
    Adds(temp2Local, temp2Local, rmsNormParams.epsilon, rmsNormParams.row);
    PipeBarrier<PIPE_V>();
    Sqrt(temp2Local, temp2Local, rmsNormParams.row);
    PipeBarrier<PIPE_V>();
    Brcb(temp1Local, temp2Local, CeilDivT(rmsNormParams.row, BRCB_NUM), {1, BRCB_NUM});
    PipeBarrier<PIPE_V>();
    RowDivs(dstLocal, srcLocal, temp1Local, repeatParams);
    PipeBarrier<PIPE_V>();
    MatMulVec(dstLocal, dstLocal, gammaLocal, repeatParams);
}

// INTERLEAVE 模式的 Gather 偏移
// offset[i] = (i ^ 1) * sizeof(float)；evtSToV 为调用方分配的固定 S_V 事件 id（Set/Wait 各一次）
template <typename T>
__aicore__ inline void SetGatherSrcOffset(const LocalTensor<int32_t> &gatherOffsetLocal, uint32_t count,
                                          event_t evtSToV)
{
    for (uint32_t i = 0; i < 8; i++) {
        gatherOffsetLocal.SetValue(i, i ^ 1);
    }
    SetFlag<HardEvent::S_V>(evtSToV);
    WaitFlag<HardEvent::S_V>(evtSToV);

    int32_t scalarValue = 8;
    while (scalarValue < (int32_t)count) {
        int32_t nextValue = scalarValue * 2;
        PipeBarrier<PIPE_V>();
        if (nextValue < (int32_t)count) {
            Adds(gatherOffsetLocal[scalarValue], gatherOffsetLocal, scalarValue, scalarValue);
        } else {
            Adds(gatherOffsetLocal[scalarValue], gatherOffsetLocal, scalarValue, count - scalarValue);
            break;
        }
        scalarValue = nextValue;
    }
    PipeBarrier<PIPE_V>();
    Muls(gatherOffsetLocal, gatherOffsetLocal, static_cast<int32_t>(sizeof(T)), count);
}

// 对 [row, actualCol] 的 [baseAddr, baseAddr+col) 段做 RoPE
// shareTmpUb 需要 [row * col * sizeof(float)]；dst/src 可同空间
template <ROTARY_MODE MODE>
__aicore__ inline void RotaryPosEmb(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                    const LocalTensor<float> &cosLocal, const LocalTensor<float> &sinLocal,
                                    const LocalTensor<float> &shareTmpUb,
                                    const LocalTensor<uint32_t> &gatherOffsetcastLocal, uint32_t row, uint32_t col,
                                    uint32_t actualCol, uint64_t baseAddr)
{
    uint64_t cnt = row * col;
    uint32_t halfCol = col >> 1;
    LocalTensor<float> reArrLocal = shareTmpUb.ReinterpretCast<float>();
    if constexpr (MODE == ROTARY_MODE::HALF) {
        DataCopy(reArrLocal, srcLocal[baseAddr + halfCol],
                 {static_cast<uint16_t>(row), static_cast<uint16_t>(CeilDivT(halfCol, FP32_BLOCK_ELEMENT_NUM)),
                  static_cast<uint16_t>(CeilDivT(actualCol - halfCol, FP32_BLOCK_ELEMENT_NUM)),
                  static_cast<uint16_t>(CeilDivT(halfCol, FP32_BLOCK_ELEMENT_NUM))});
        DataCopy(reArrLocal[halfCol], srcLocal[baseAddr],
                 {static_cast<uint16_t>(row), static_cast<uint16_t>(CeilDivT(halfCol, FP32_BLOCK_ELEMENT_NUM)),
                  static_cast<uint16_t>(CeilDivT(actualCol - halfCol, FP32_BLOCK_ELEMENT_NUM)),
                  static_cast<uint16_t>(CeilDivT(halfCol, FP32_BLOCK_ELEMENT_NUM))});
        PipeBarrier<PIPE_V>();
        Muls(reArrLocal, reArrLocal, float(-1), halfCol, row,
             {1, 1, static_cast<uint8_t>(CeilDivT(static_cast<uint32_t>(col), FP32_BLOCK_ELEMENT_NUM)),
              static_cast<uint8_t>(CeilDivT(static_cast<uint32_t>(col), FP32_BLOCK_ELEMENT_NUM))});
    } else if constexpr (MODE == ROTARY_MODE::INTERLEAVE) {
        for (uint32_t i = 0; i < row; i++) {
            Gather(reArrLocal[i * col], srcLocal[i * actualCol + baseAddr], gatherOffsetcastLocal, 0, col);
        }
        PipeBarrier<PIPE_V>();
        uint32_t repeatTimes = cnt / FP32_REPEAT_ELEMENT_NUM;
        uint32_t remainder = cnt % FP32_REPEAT_ELEMENT_NUM;
        uint64_t fullMask = 0x5555555555555555;
        uint64_t partialMask = 0x55;
        SetVectorMask<float, MaskMode::NORMAL>(0, fullMask);
        Muls<float, false>(reArrLocal, reArrLocal, float(-1), MASK_PLACEHOLDER, repeatTimes,
                           {1, 1, FP32_BLOCK_ELEMENT_NUM, FP32_BLOCK_ELEMENT_NUM});
        if (unlikely(remainder > 0)) {
            SetVectorMask<float, MaskMode::NORMAL>(0, partialMask);
            Muls<float, false>(reArrLocal[repeatTimes * FP32_REPEAT_ELEMENT_NUM],
                               reArrLocal[repeatTimes * FP32_REPEAT_ELEMENT_NUM], float(-1), MASK_PLACEHOLDER,
                               remainder / FP32_BLOCK_ELEMENT_NUM, {1, 1, 1, 1});
        }
        ResetMask();
    }

    PipeBarrier<PIPE_V>();
    BinaryRepeatParams computeParams{1,
                                     1,
                                     1,
                                     static_cast<uint8_t>(CeilDivT(actualCol, FP32_BLOCK_ELEMENT_NUM)),
                                     static_cast<uint8_t>(CeilDivT(actualCol, FP32_BLOCK_ELEMENT_NUM)),
                                     static_cast<uint8_t>(CeilDivT(col, FP32_BLOCK_ELEMENT_NUM))};
    Mul(dstLocal[baseAddr], srcLocal[baseAddr], cosLocal, col, row, computeParams);
    Mul(reArrLocal, reArrLocal, sinLocal, cnt);
    PipeBarrier<PIPE_V>();
    Add(dstLocal[baseAddr], dstLocal[baseAddr], reArrLocal, col, row, computeParams);
}

} // namespace CompressNormRope

#endif // COMPRESS_NORM_ROPE_VECTOR_COMM_H
