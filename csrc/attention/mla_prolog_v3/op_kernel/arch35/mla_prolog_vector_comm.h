/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_comm.h
 * \brief 存放各种vector的公共组件
 */

#ifndef MLA_PROLOG_VECTOR_COMM_H
#define MLA_PROLOG_VECTOR_COMM_H

#include "vf/vf_quant_pertensor.h"
#include "vf/vf_quant_perchannel.h"
#include "vf/vf_dynamic_quant.h"
#include "vf/vf_dequant.h"

#include "mla_prolog_comm.h"
namespace MlaProlog {

struct Rectangle {
    uint32_t row;
    uint32_t col;
    uint32_t stride;
};

struct RmsNormParam {
    float reciprocal;
    float epsilon;
    uint32_t row;
    uint32_t col;
    float scale;
    uint16_t isScaleEnable;
};

struct PerTileQuantParams {
    uint32_t tileSize;
    uint32_t tileNum;
    float alpha;
    uint32_t row;
    uint32_t col;
};

/**
 * @brief RowMuls muls by row, 每行的元素乘以相同的元素，该元素需要扩展到一个数据块；
 *        dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
 * @param dstUb 输出tensor [row, columnStride]
 * @param src0Ub 输入tensor [row, columnStride]
 * @param src1Ub 输入tensor [row, FP32_BLOCK_ELEMENT_NUM]
 * @param rectangleParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 */
template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               const Rectangle &rectangleParams)
{
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2 = 128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dLoop = rectangleParams.col / repeatElementNum;
    uint32_t dRemain = rectangleParams.col % repeatElementNum;
    // REPEAT_STRIDE_UP_BOUND=256，此限制由于src0RepStride数据类型为uint8至多256个datablock间距
    if (rectangleParams.stride < REPEAT_STRIDE_UP_BOUND * blockElementNum) {
        // dstBlkStrideIn src0BlkStrideIn  src1BlkStrideIn dstRepStrideIn src0RepStrideIn src1RepStrideIn
        BinaryRepeatParams repeatParams{1,
                                        1,
                                        0,
                                        static_cast<uint8_t>(rectangleParams.stride / blockElementNum),
                                        static_cast<uint8_t>(rectangleParams.stride / blockElementNum),
                                        1};

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= rectangleParams.row) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, rectangleParams.row, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            repeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            repeatParams.src1RepStride = 0;
            repeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < rectangleParams.row; i++) {
                Mul(dstUb[i * rectangleParams.stride], src0Ub[i * rectangleParams.stride], src1Ub[i * blockElementNum],
                    repeatElementNum, dLoop, repeatParams);
            }
        }

        // 最后一次完成[row, dRemain] * [row, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, rectangleParams.row,
                repeatParams);
        }
    } else {
        // dstBlkStrideIn src0BlkStrideIn  src1BlkStrideIn dstRepStrideIn src0RepStrideIn src1RepStrideIn
        // 8 : 每个repeat为256B数据，正好8个datablock
        BinaryRepeatParams repeatParams{1, 1, 0, 8, 8, 0};

        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < rectangleParams.row; i++) {
            // 计算一行中的dLoop个repeat，每个repeat计算256/block_size个data_block
            Mul(dstUb[i * rectangleParams.stride], src0Ub[i * rectangleParams.stride], src1Ub[i * blockElementNum],
                repeatElementNum, dLoop, repeatParams);
            // 计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * rectangleParams.stride + dLoop * repeatElementNum],
                    src0Ub[i * rectangleParams.stride + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain,
                    1, repeatParams);
            }
        }
    }
}

/**
 * @brief RowMax max by row, 按行求最大值
 * @param dstUb 输出tensor [row, 1]
                dstUb[i] = max(srcUb[i, :])
 * @param srcUb 输入tensor [row, columnStride]
 * @param rectangleParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 */
__aicore__ inline void RowMax(LocalTensor<float> &dstUb, LocalTensor<float> &srcUb, const Rectangle rectangleParams)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockCount = rectangleParams.col / dtypeMask;
    uint32_t remain = rectangleParams.col % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.dstRepStride = rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM;
    if (blockCount > 0 && remain > 0) {
        Max(srcUb, srcUb, srcUb[blockCount * dtypeMask], remain, rectangleParams.row, repeatParamsMax);
        PipeBarrier<PIPE_V>();
    }

    for (uint32_t columnLoopCount = blockCount >> 1; columnLoopCount > 0;
         columnLoopCount = blockCount >> 1) { // 2: 每次处理2个block
        blockCount = (blockCount + 1) >> 1;   // 2: 每次处理2个block
        for (uint32_t j = 0; j < columnLoopCount; j++) {
            Max(srcUb[j * dtypeMask], srcUb[j * dtypeMask], srcUb[(j + blockCount) * dtypeMask], dtypeMask,
                rectangleParams.row, repeatParamsMax);
        }
        PipeBarrier<PIPE_V>();
    }

    WholeReduceMax(dstUb, srcUb, (rectangleParams.col < dtypeMask) ? rectangleParams.col : dtypeMask,
                   rectangleParams.row, 1, 1, rectangleParams.stride / FP32_BLOCK_ELEMENT_NUM,
                   ReduceOrder::ORDER_ONLY_VALUE);
}

/**
 * @brief Dequant 对[row * col]的tensor进行反量化。需要输入一个行向量和列向量，共同构成反量化的矩阵。
          outputLocal [i,j] = inputLocal[i,j] * scaleLocal[j] * scale2Local [i]
 * @param outputLocal 输出tensor [row , col]
 * @param inputLocal 输入tensor [row , col]
 * @param scaleLocal [1,col] 量化系数；行向量
 * @param scale2Local [row,8] 量化系数；列向量，8为float扩充为32Bytes
 * @param rectangleParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 */
template <typename T>
__aicore__ inline void Dequant(const LocalTensor<float> &outputLocal, const LocalTensor<T> &inputLocal,
                               const LocalTensor<float> &scaleLocal, const LocalTensor<float> &scale2Local,
                               const Rectangle &rectangleParams)
{
    DequantVf(outputLocal, inputLocal, scaleLocal, scale2Local, rectangleParams.row, rectangleParams.col,
              rectangleParams.stride);
}

/**
 * @brief CastFP32ToINT8 将float类型cast为int8，路径为float--------->int32--------->half---------->int8
                                           CAST_RINT     CAST_ROUND     CAST_TRUNC
 * @param outputLocal 输出tensor [cnt]
 * @param inputLocal 输入tensor [cnt]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [cnt * 4] 4 : sizeof(int32)
 * @param cnt tensor长度
 */
template <typename O>
__aicore__ inline void CastFP32ToINT8(const LocalTensor<O> outLocal, const LocalTensor<float> &inputLocal,
                                      const LocalTensor<uint8_t> &shareTmpUb, uint64_t cnt)
{
    LocalTensor<int32_t> int32 = shareTmpUb.ReinterpretCast<int32_t>();
    LocalTensor<half> tmpHalf = shareTmpUb.ReinterpretCast<half>();
    Cast(int32, inputLocal, RoundMode::CAST_RINT, cnt);
    PipeBarrier<PIPE_V>();
    SetDeqScale(static_cast<half>(1.0));
    PipeBarrier<PIPE_V>();
    Cast(tmpHalf, int32, RoundMode::CAST_ROUND, cnt);
    PipeBarrier<PIPE_V>();
    Cast(outLocal, tmpHalf, RoundMode::CAST_TRUNC, cnt);
}

// divs by row, 每行的元素除以相同的元素
// dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
// src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
// columnCount]
__aicore__ inline void RowClips(LocalTensor<float> &dstUb, const LocalTensor<float> &src0Ub,
                                const LocalTensor<float> &maxUb, const LocalTensor<float> &minUb, uint32_t dealRowCount,
                                uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t offset = 0;

    for (uint32_t i = 0; i < dLoop; i++) {
        Min(dstUb[offset], src0Ub[offset], maxUb, dtypeMask, dealRowCount, repeatParams);
        PipeBarrier<PIPE_V>();
        Max(dstUb[offset], dstUb[offset], minUb, dtypeMask, dealRowCount, repeatParams);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Min(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], maxUb, dRemain, dealRowCount, repeatParams);
        PipeBarrier<PIPE_V>();
        Max(dstUb[dLoop * dtypeMask], dstUb[dLoop * dtypeMask], minUb, dRemain, dealRowCount, repeatParams);
    }
}

__aicore__ inline void PerTileClipWithAlpha(LocalTensor<float> &dstClipUb, LocalTensor<float> &aMax,
                                            LocalTensor<float> &aMaxBrcb, const LocalTensor<float> &srcUb,
                                            const LocalTensor<uint8_t> &shareTmpUb,
                                            const PerTileQuantParams &perTileQuantParams)
{
    constexpr uint32_t brcnNum = 8; // brcn一次性处理8个数据
    uint32_t srcSize = perTileQuantParams.row * perTileQuantParams.col;
    uint32_t maxAlphaSize = perTileQuantParams.row * perTileQuantParams.tileNum * FP32_BLOCK_ELEMENT_NUM;

    LocalTensor<float> absSrcUb = shareTmpUb.template ReinterpretCast<float>();
    LocalTensor<float> maxAlpha = absSrcUb[Align(srcSize, FP32_BLOCK_ELEMENT_NUM)];
    LocalTensor<float> minAlpha = maxAlpha[Align(maxAlphaSize, FP32_BLOCK_ELEMENT_NUM)];

    Abs(absSrcUb, srcUb, srcSize);
    PipeBarrier<PIPE_V>();

    Rectangle rectangleParams{
        static_cast<uint32_t>(perTileQuantParams.row * perTileQuantParams.tileNum),
        static_cast<uint32_t>(perTileQuantParams.tileSize),
        static_cast<uint32_t>(perTileQuantParams.tileSize) // columnStride
    };
    RowMax(aMax, absSrcUb, rectangleParams);
    PipeBarrier<PIPE_V>();
    Brcb(aMaxBrcb, aMax, (CeilDivT((perTileQuantParams.row * perTileQuantParams.tileNum), brcnNum)), {1, brcnNum});
    PipeBarrier<PIPE_V>();

    Muls(maxAlpha, aMaxBrcb, perTileQuantParams.alpha, maxAlphaSize);
    Muls(minAlpha, aMaxBrcb, -perTileQuantParams.alpha, maxAlphaSize);
    PipeBarrier<PIPE_V>();

    RowClips(dstClipUb, srcUb, maxAlpha, minAlpha, perTileQuantParams.row * perTileQuantParams.tileNum,
             perTileQuantParams.tileSize, perTileQuantParams.tileSize);
    PipeBarrier<PIPE_V>();
}

/**
 * @brief DynamicQuant 对row行进行dynamicquant, float ---> int8, 每一行出一个系数。
 * @param outputLocal 输出tensor [row , col]，支持和inputLocal是同一块空间
 * @param scale 输出每行的反量化系数 [row]
 * @param inputLocal 输入tensor [row , col]
 * @param rowMax [row] 一行的最大值
 * @param rowMaxBrcb [row, 8] 一行的最大值，brcb扩充为8的倍数
 * @param shareTmpUb 临时buffer 内部需要的空间为 [row * col * sizeof(float)]
 * @param row 待处理的行数
 * @param col 待处理的列数
 */
__aicore__ inline void DynamicQuant(const LocalTensor<float> &outputLocal, const LocalTensor<float> &scale,
                                    const LocalTensor<float> &inputLocal, const LocalTensor<float> &aMax,
                                    const LocalTensor<float> &rowMaxBrcb, const LocalTensor<uint8_t> &shareTmpUb,
                                    float alpha, uint64_t row, uint64_t col)
{
    constexpr float maxInt8 = 127.0f;
    int32_t aMaxSizeAlign = Align(static_cast<uint32_t>(row), FP32_BLOCK_ELEMENT_NUM);
    LocalTensor<float> aMaxAlpha = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> dupTensor = aMaxAlpha[aMaxSizeAlign];
    LocalTensor<float> scaleReciprocal = dupTensor[aMaxSizeAlign];
    LocalTensor<float> scaleReciprocalBrcb = scaleReciprocal[aMaxSizeAlign];

    Duplicate(dupTensor, 1.0f, row);
    Muls(aMaxAlpha, aMax, alpha, row);
    PipeBarrier<PIPE_V>();
    Muls(scale, aMaxAlpha, 1.0f / maxInt8, row);
    PipeBarrier<PIPE_V>();
    Div(scaleReciprocal, dupTensor, scale, row);
    PipeBarrier<PIPE_V>();
    Brcb(scaleReciprocalBrcb, scaleReciprocal, CeilDivT(row, 8UL), {1, 8});
    PipeBarrier<PIPE_V>();
    RowMuls(outputLocal, inputLocal, scaleReciprocalBrcb,
            Rectangle{static_cast<uint32_t>(row), static_cast<uint32_t>(col), static_cast<uint32_t>(col)});
    PipeBarrier<PIPE_V>();
}

/**
 * @brief QuantPerChannel 同时对row行进行FP32到int8的per-channel量化操作。一行中的每一列用不同的量化参数。
          outLocal[i, j] = inputLocal[i, j] * quantScaleLocal[j]
 * @param outLocal 输出tensor [row , col]
 * @param inputLocal 输入tensor [row , col]
 * @param quantScaleLocal quant系数 [1 , col]
 * @param shareTmpUb 临时buffer 内部需要的空间为 [row * col * 4]，源自CastFP32ToINT8
 * @param rectangleParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 */
template <typename T, typename C, typename O>
__aicore__ inline void QuantPerChannel(const LocalTensor<O> &outLocal, const LocalTensor<T> &inputLocal,
                                       const LocalTensor<C> &quantScaleLocal, const LocalTensor<uint8_t> &shareTmpUb,
                                       const Rectangle &rectangleParams)
{
    QuantPerChannelVf(outLocal, inputLocal, quantScaleLocal, rectangleParams.row, rectangleParams.col,
                      rectangleParams.stride);
}

/**
 * @brief QuantPerTensor 同时对row行进行FP32到int8的per-tensor量化操作。一行内共用同一个量化系数。
          outLocal[i , j] = inputLocal[i , j] * quantScaleLocal[i]
 * @param outLocal 输出tensor [row , col]
 * @param inputLocal 输入tensor [row , col]
 * @param quantScaleLocal quant系数 [row , 8]; 8 : 32Bytes对齐
 * @param shareTmpUb 临时buffer 内部需要的空间为 [row * col * 4]，源自CastFP32ToINT8
 * @param rectangleParams 描述待处理数据的排布，包括
          row 行数
          col 列数
          stride 一行的真实长度
 */
template <typename T, typename C, typename O>
__aicore__ inline void QuantPerTensor(const LocalTensor<O> &outLocal, const LocalTensor<T> &inputLocal,
                                      const LocalTensor<C> &quantScaleLocal, const LocalTensor<uint8_t> &shareTmpUb,
                                      const Rectangle &rectangleParams)
{
    QuantPerTensorVF(outLocal, inputLocal, quantScaleLocal, rectangleParams.row, rectangleParams.col);
}

/**
 * @brief QuantPerTile 对输入tensor进行per-tile量化操作，FP32->fp8 e4m3/hifloat8，每个tile出一个量化系数。
            量化流程：先对输入的每行的每个tile做动态量化，最后转换为fp8/hif8.
 * @param outLocal 输出tensor [row * col]，量化后的fp8/hif8数据，后续跟随scale数据
 * @param inputLocal 输入tensor [row * col]
 * @param shareTmpUb 临时buffer，内部需要空间为：
 *          [Align(row * tileNum, 8) + row * tileNum * 8 + 其他中间计算所需空间] * sizeof(float)
 * @param perTileQuantParams 描述pertile量化的参数，包括
 *          tileSize 每个tile的大小
 *          tileNum 每行的tile数量
 *          alpha clip的alpha系数
 *          row 待处理的行数
 *          col 待处理的列数 （col = tileSize * tileNum）
 */
template <typename O>
__aicore__ inline void QuantPerTile8Bit(const LocalTensor<O> &outLocal, const LocalTensor<float> &inputLocal,
                                        const PerTileQuantParams &perTileQuantParams)
{
    LocalTensor<float> quantScaleLocal =
        outLocal[perTileQuantParams.row * perTileQuantParams.col].template ReinterpretCast<float>();
    QuantPerTileVF<float, float, O>(outLocal, inputLocal, quantScaleLocal, perTileQuantParams.row,
                                    perTileQuantParams.col, perTileQuantParams.tileSize);
}

__aicore__ inline void QuantPerTile(const LocalTensor<int8_t> &outLocal, const LocalTensor<float> &inputLocal,
                                    const LocalTensor<uint8_t> &shareTmpUb,
                                    const PerTileQuantParams &perTileQuantParams)
{
    LocalTensor<float> scale =
        outLocal[perTileQuantParams.row * perTileQuantParams.col].template ReinterpretCast<float>();

    LocalTensor<float> clipOut = inputLocal;
    uint32_t aMaxSizeAlign =
        Align(static_cast<uint32_t>(perTileQuantParams.row * perTileQuantParams.tileNum), FP32_BLOCK_ELEMENT_NUM);
    uint32_t aMaxBrcbSize = perTileQuantParams.row * perTileQuantParams.tileNum * FP32_BLOCK_ELEMENT_NUM;
    LocalTensor<float> aMax = shareTmpUb.template ReinterpretCast<float>();
    LocalTensor<float> aMaxBrcb = aMax[aMaxSizeAlign];
    LocalTensor<uint8_t> sharedBuf = aMaxBrcb[aMaxBrcbSize].template ReinterpretCast<uint8_t>();

    PerTileClipWithAlpha(clipOut, aMax, aMaxBrcb, inputLocal, sharedBuf, perTileQuantParams);

    LocalTensor<float> quantOut = clipOut;
    DynamicQuant(quantOut, scale, clipOut, aMax, aMaxBrcb, sharedBuf, perTileQuantParams.alpha,
                 perTileQuantParams.row * perTileQuantParams.tileNum, perTileQuantParams.tileSize);
    PipeBarrier<PIPE_V>();
    CastFP32ToINT8(outLocal, quantOut, sharedBuf, perTileQuantParams.row * perTileQuantParams.col);
}

/**
 * @brief DynamicQuant 同时对row行进行dynamicquant, float ---> int8, 每一行出一个系数。
 * @param outLocal 输出tensor [row , col]，支持和inputLocal是同一块空间
 * @param inputLocal 输入tensor [row , col]
 * @param scale 输出每行的反量化系数 [1 , row]
 * @param maxInt8Tensor [1 , row] 元素均为int8的最大值127
 * @param shareTmpUb 临时buffer 内部需要的空间为 [(Align(row * col, 8) + Align(row , 8) * ALIGN_BLOCK_SIZE) *
 * sizeof(float)]
 * @param row 待处理的行数
 * @param col 待处理的列数
 */
__aicore__ inline void DynamicQuant(const LocalTensor<float> &outputLocal, const LocalTensor<float> &scale,
                                    const LocalTensor<float> &inputLocal, const LocalTensor<float> &maxInt8Tensor,
                                    const LocalTensor<uint8_t> &shareTmpUb, uint64_t row, uint64_t col)
{
    constexpr uint64_t brcnNum = 8; // brcb一次处理8个数据
    uint64_t computeSize = row * col;
    LocalTensor<float> inputCopy = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> rowMaxBrcb = inputCopy[Align(computeSize, static_cast<uint64_t>(ALIGN_BLOCK_SIZE))];
    // abs(x)
    Abs(inputCopy, inputLocal, computeSize);
    PipeBarrier<PIPE_V>();
    Rectangle rectangleParams{
        static_cast<uint32_t>(row), static_cast<uint32_t>(col),
        static_cast<uint32_t>(col) // columnStride
    };
    // rowMax(abs(x))
    RowMax(inputCopy, inputCopy, rectangleParams);
    PipeBarrier<PIPE_V>();

    // scaleOut = rowMax(abs(x)) / 127
    Div(scale, inputCopy, maxInt8Tensor, row);
    PipeBarrier<PIPE_V>();

    // 1 / scaleOut = 127 / rowMax(abs(x))
    Div(inputCopy, maxInt8Tensor, inputCopy, row);
    PipeBarrier<PIPE_V>();

    Brcb(rowMaxBrcb, inputCopy, static_cast<uint8_t>(CeilDivT(row, brcnNum)), {1, brcnNum});
    PipeBarrier<PIPE_V>();

    // x * 1 / scaleOut
    RowMuls(outputLocal, inputLocal, rowMaxBrcb, rectangleParams);
}

} // namespace MlaProlog
#endif // MLA_PROLOG_VECTOR_COMM_H