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
 * \file service_dynamic_quant_qn_mul_qr.h
 * \brief
 */

#ifndef SERVICE_DYNAMIC_QUANT_H
#define SERVICE_DYNAMIC_QUANT_H

#include "mla_prolog_comm.h"
#include "mla_prolog_vector_comm.h"

namespace MlaProlog {
template <typename T, typename C>
__aicore__ inline void
MulQr(const GlobalTensor<T> &outputGmRope, const GlobalTensor<T> &inputGmRope, LocalTensor<T> outputLocalRope,
      const LocalTensor<T> &qrInputLocal, const LocalTensor<C> &qrFp32Local, const LocalTensor<C> &reciprocalLocal,
      const LocalTensor<C> &dequantScaleBrcbLocal, uint64_t row, uint64_t colRope, uint64_t subRowRope,
      uint64_t qrOutputStrideRope, float quantScaleCkvRope, uint32_t MUL_QR_INPUT_COPY_READY, uint32_t MUL_QR)
{
    constexpr uint64_t computeBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(C));
    constexpr uint64_t inputBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(T));

    uint64_t loopCntRope = CeilDivT(row, subRowRope);
    uint64_t lastSubRowRope = row - (loopCntRope - 1) * subRowRope;
    uint32_t computeSizeRope = subRowRope * colRope;

    // DataCopyParams  : count len srcStrideIn dstStrideIn
    DataCopyParams inputParamsRope{(uint16_t)row, (uint16_t)(colRope / inputBlockAlign),
                                   (uint16_t)((qrOutputStrideRope - colRope) / inputBlockAlign), 0};
    DataCopyParams outputParamsRope{(uint16_t)(subRowRope), (uint16_t)(colRope / inputBlockAlign), 0,
                                    (uint16_t)((qrOutputStrideRope - colRope) / inputBlockAlign)};

    uint64_t inputGmRopeOffset = 0;
    uint64_t dequantScaleOffset = 0;
    uint64_t inputLocalRopeOffset = 0;


    AscendC::WaitFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);
    DataCopy(qrInputLocal, inputGmRope, inputParamsRope);
    for (uint64_t loopIdx = 0; loopIdx < loopCntRope; loopIdx++) {
        if (loopIdx == (loopCntRope - 1) && subRowRope != lastSubRowRope) {
            subRowRope = lastSubRowRope;
            computeSizeRope = subRowRope * colRope;
            outputParamsRope.blockCount = subRowRope;
        }
        AscendC::SetFlag<HardEvent::MTE3_V>(MUL_QR);
        AscendC::WaitFlag<HardEvent::MTE3_V>(MUL_QR);

        AscendC::SetFlag<HardEvent::MTE2_V>(MUL_QR);
        AscendC::WaitFlag<HardEvent::MTE2_V>(MUL_QR);

        Cast(qrFp32Local, qrInputLocal[inputLocalRopeOffset], RoundMode::CAST_NONE, computeSizeRope);
        AscendC::PipeBarrier<PIPE_V>();
        Duplicate(reciprocalLocal, quantScaleCkvRope, subRowRope * computeBlockAlign);
        AscendC::PipeBarrier<PIPE_V>();
        // cal: quantScaleCkv / dequantScaleQn
        Div(reciprocalLocal, reciprocalLocal, dequantScaleBrcbLocal[dequantScaleOffset],
            subRowRope * computeBlockAlign);
        AscendC::PipeBarrier<PIPE_V>();
        // cal: x * quantScaleCkv / dequantScaleQn
        RowMuls(qrFp32Local, qrFp32Local, reciprocalLocal,
                Rectangle{(uint32_t)subRowRope, (uint32_t)colRope, (uint32_t)colRope});
        AscendC::PipeBarrier<PIPE_V>();
        Cast(outputLocalRope, qrFp32Local, RoundMode::CAST_RINT, computeSizeRope);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<HardEvent::V_MTE3>(MUL_QR);
        AscendC::WaitFlag<HardEvent::V_MTE3>(MUL_QR);

        DataCopy(outputGmRope[inputGmRopeOffset], outputLocalRope, outputParamsRope);
        inputLocalRopeOffset += subRowRope * colRope;
        inputGmRopeOffset += subRowRope * qrOutputStrideRope;
        dequantScaleOffset += subRowRope * computeBlockAlign;
    }
    AscendC::SetFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);
}

template <typename T, typename C, typename O>
__aicore__ inline void DynamicQuantMultiRow(const GlobalTensor<O> &outputGm, const LocalTensor<C> &scaleOutputLocal,
                                            const GlobalTensor<T> &inputGm, const LocalTensor<C> outputLocal,
                                            const LocalTensor<T> &inputHalf, const LocalTensor<C> &inputLocal,
                                            const LocalTensor<C> &maxInt8Tensor, const LocalTensor<uint8_t> &shareTmpUb,
                                            uint64_t row, uint64_t col, uint64_t subRow, uint64_t queryOutStride,
                                            uint32_t DYNAMIC_QUANT_INPUT_READY, uint32_t DYNAMIC_QUANT_OUTPUT_READY)
{
    constexpr uint64_t outputBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(O));
    constexpr uint64_t inputBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(T));
    // DataCopyParams  : count len srcStrideIn dstStrideIn
    DataCopyParams outParams{(uint16_t)subRow, (uint16_t)(col / outputBlockAlign), 0,
                             (uint16_t)((queryOutStride - col) / outputBlockAlign)};
    DataCopyParams inputParams{(uint16_t)subRow, (uint16_t)(col / inputBlockAlign),
                               (uint16_t)((queryOutStride - col) / inputBlockAlign), 0};

    uint32_t computeSize = col * subRow;
    uint64_t loopCnt = CeilDivT(row, subRow);
    uint64_t lastSubRow = row - (loopCnt - 1) * subRow;
    uint64_t inputGmOffset = 0;
    uint64_t scaleOffset = 0;

    for (uint64_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
        if ((loopIdx == loopCnt - 1) && (lastSubRow != subRow)) {
            subRow = lastSubRow;
            computeSize = col * subRow;
            outParams.blockCount = subRow;
            inputParams.blockCount = subRow;
        }

        AscendC::WaitFlag<HardEvent::V_MTE2>(DYNAMIC_QUANT_INPUT_READY); // 计算是否已经完成可以搬运
        DataCopy(inputHalf, inputGm[inputGmOffset], inputParams);
        AscendC::SetFlag<HardEvent::MTE2_V>(DYNAMIC_QUANT_INPUT_READY);
        AscendC::WaitFlag<HardEvent::MTE2_V>(DYNAMIC_QUANT_INPUT_READY); // 搬运是否已经完成可以计算

        Cast(inputLocal, inputHalf, RoundMode::CAST_NONE, computeSize);
        AscendC::SetFlag<HardEvent::V_MTE2>(DYNAMIC_QUANT_INPUT_READY);
        AscendC::PipeBarrier<PIPE_V>();
        DynamicQuant(inputLocal, scaleOutputLocal[scaleOffset], inputLocal, maxInt8Tensor, shareTmpUb, subRow, col);
        AscendC::PipeBarrier<PIPE_V>();
        // float -> half -> int
        AscendC::WaitFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY); // 搬运是否已经完成可以计算
        LocalTensor<half> tmpf16Ub = outputLocal.template ReinterpretCast<half>();
        Cast(tmpf16Ub, inputLocal, AscendC::RoundMode::CAST_ROUND, computeSize);
        AscendC::PipeBarrier<PIPE_V>();
        // Calc: out = round(x / scaleOut)
        LocalTensor<O> tmpMMResCastTensor = outputLocal.template ReinterpretCast<O>();
        Cast(tmpMMResCastTensor, tmpf16Ub, AscendC::RoundMode::CAST_ROUND, computeSize);
        AscendC::SetFlag<HardEvent::V_MTE3>(DYNAMIC_QUANT_OUTPUT_READY);
        AscendC::WaitFlag<HardEvent::V_MTE3>(DYNAMIC_QUANT_OUTPUT_READY); // 计算是否已经完成可以搬运

        DataCopy(outputGm[inputGmOffset], tmpMMResCastTensor, outParams);
        AscendC::SetFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
        inputGmOffset += subRow * queryOutStride;
        scaleOffset += subRow;
    }
}

/**
 * @brief DynamicQuantQnWithMulQr 流程中dynamicquant和ropeMul融合流程
          UB，流水，中间产物统筹规划
 * @tparam T mmQnOutputType -> bf16
 * @tparam C dequantScaleQNopeType -> float
 * @tparam O queryOutputType -> int8
 * @param scaleOutputGm 动态量化参数输出GM位置
 * @param outputGm 动态量化结果输出GM位置
 * @param outputGmRope rope处理后的输出位置
 * @param inputGm dynamicQuant部分的输入
 * @param shareTmpUb 临时buffer空间
 * @param row
 * @param col
 * @param scaleOutStride 描述动态量化参数输出GM位置的偏移关系
 * @param queryOutStride 描述动态量化结果输出GM位置
 * @param inputGmRope
 * @param quantScaleCkvRope
 * @param colRope
 * @param qrOutputStrideRope 描述rope处理后的输出位置
 */
template <typename T, typename C, typename O>
__aicore__ inline void DynamicQuantQnWithMulQr(
    // Dynamic Quant With MulQr 输出
    const GlobalTensor<C> &scaleOutputGm, const GlobalTensor<O> &outputGm, const GlobalTensor<T> &outputGmRope,
    // Dynamic Quant 入参
    const GlobalTensor<T> &inputGm, LocalTensor<uint8_t> &shareTmpUb, uint64_t row, uint64_t col,
    uint64_t scaleOutStride, uint64_t queryOutStride,
    // Mul Qr 入参
    const GlobalTensor<T> &inputGmRope, float quantScaleCkvRope, uint64_t colRope, uint64_t qrOutputStrideRope,
    uint32_t cvRatio)
{
    if (col == 0 || row == 0) {
        return;
    }
    // 常量
    constexpr uint32_t MUL_QR = EVENT_ID1; // 用于控制Mul_Qr的同步
    constexpr uint32_t DYNAMIC_QUANT_INPUT_READY =
        EVENT_ID0; // dynamicquant输入的计算/搬运 是否已经完成可以开始下一轮的 搬运/计算
    constexpr uint32_t MUL_QR_INPUT_COPY_READY = EVENT_ID3; // 是否可以开始下一轮的MUL_QR_INPUT_COPY
    constexpr uint32_t CALC_SCALE_FINISH = EVENT_ID0;       // dynamicquant的scale是否计算完成可以开始搬运
    constexpr uint32_t DYNAMIC_QUANT_OUTPUT_READY =
        EVENT_ID3; // dynamicquant输出的计算/搬运 是否已经完成可以开始下一轮的 搬运/计算

    constexpr uint64_t inputBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(T));
    constexpr uint64_t computeBlockAlign = (ALIGN_BLOCK_SIZE / sizeof(C));
    constexpr float maxInt8 = 127.0;
    // Dynamic Quant 局部变量
    uint64_t rowStepSize = 8 * cvRatio; // 单次处理最大行数, cv1:1场景，改为8行进行计算，降低UB使用

    uint64_t subRow = row < rowStepSize ? row : rowStepSize;
    uint32_t brcbCnt = Align(subRow, computeBlockAlign);
    uint32_t computeSize = subRow * col;

    // Rope Post Process 局部变量
    uint64_t rowStepSizeRope = 64; // 单次处理最大行数
    uint32_t loadRopeGmSize = row * colRope;
    uint64_t subRowRope = row < rowStepSizeRope ? row : rowStepSizeRope;
    uint32_t computeSizeRope = subRowRope * colRope;

    // Dynamic Quant & Rope Post Process UB Buffer分配
    LocalTensor<T> inputHalf = shareTmpUb.ReinterpretCast<T>();
    LocalTensor<T> qrInputLocal = inputHalf[computeSize];
    LocalTensor<T> outputLocalRope = qrInputLocal[inputBlockAlign + loadRopeGmSize];

    LocalTensor<C> inputLocal = outputLocalRope[inputBlockAlign + computeSizeRope].template ReinterpretCast<C>();
    LocalTensor<C> outputLocal = inputLocal[computeSize];
    LocalTensor<C> maxInt8Tensor = outputLocal[computeSize];
    LocalTensor<C> dynamicQuantUb = maxInt8Tensor[brcbCnt];
    LocalTensor<C> scaleOutputLocal = dynamicQuantUb[computeSize + brcbCnt * computeBlockAlign];
    LocalTensor<C> scaleBrcb = scaleOutputLocal[Align(row, computeBlockAlign)];

    // Rope Post Process流程在Dynamic Quant流程结束，两者UB Buffer不会同时使用，故可以从起始位置开始重新计算，减少UB
    // Buffer使用
    LocalTensor<C> qrFp32Local = inputLocal;
    LocalTensor<C> reciprocalLocal = qrFp32Local[computeBlockAlign + computeSizeRope];

    // Dynamic Quant
    Duplicate(maxInt8Tensor, static_cast<C>(maxInt8), brcbCnt);
    AscendC::PipeBarrier<PIPE_V>();

    DynamicQuantMultiRow(outputGm, scaleOutputLocal, inputGm, outputLocal, inputHalf, inputLocal, maxInt8Tensor,
                         dynamicQuantUb.template ReinterpretCast<uint8_t>(), row, col, subRow, queryOutStride,
                         DYNAMIC_QUANT_INPUT_READY, DYNAMIC_QUANT_OUTPUT_READY);

    Brcb(scaleBrcb, scaleOutputLocal, CeilDivT(row, computeBlockAlign), {1, computeBlockAlign});
    AscendC::PipeBarrier<PIPE_V>();

    // DataCopyParams  : count len srcStrideIn dstStrideIn
    DataCopyParams scaleOutCopyParams{(uint16_t)row, (uint16_t)sizeof(C), 0,
                                      (uint16_t)((scaleOutStride - 1) * sizeof(C))};

    AscendC::SetFlag<HardEvent::V_MTE3>(CALC_SCALE_FINISH);
    AscendC::WaitFlag<HardEvent::V_MTE3>(CALC_SCALE_FINISH);
    DataCopyPad(scaleOutputGm, scaleBrcb, scaleOutCopyParams);

    // qr rope 后的乘法
    MulQr(outputGmRope, inputGmRope, outputLocalRope, qrInputLocal, qrFp32Local, reciprocalLocal, scaleBrcb, row,
          colRope, subRowRope, qrOutputStrideRope, quantScaleCkvRope, MUL_QR_INPUT_COPY_READY, MUL_QR);

    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

} // namespace MlaProlog

#endif
