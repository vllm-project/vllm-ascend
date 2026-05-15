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
 * \file dequant_swiglu_quant_dynamic_performance.hpp
 * \brief
 */

#ifndef DEQUANT_SWIGLU_QUANT_DYNAMIC_PERFORMANCE_HPP
#define DEQUANT_SWIGLU_QUANT_DYNAMIC_PERFORMANCE_HPP

#include "kernel_operator.h"
#include "dequant_swiglu_quant_dynamic_base.hpp"

namespace DequantSwigluQuant {
using namespace AscendC;

TEMPLATE_DECLARE
class DequantSwigluQuantDynamicPerformance : public DequantSwigluQuantDynamicBiasFloat<TEMPLATE_ARGS> {
public:
    __aicore__ inline DequantSwigluQuantDynamicPerformance(){};
    __aicore__ inline ~DequantSwigluQuantDynamicPerformance(){};

    __aicore__ inline void Init(GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm,
    GM_ADDR quant_scale_gm, GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm,
    GM_ADDR userspace, const SwiGluTilingData* tilingData, TPipe* pipe_);
    __aicore__ inline void Process();
    __aicore__ inline void BaseCompute1(uint64_t curTileLen, uint64_t blockCount, uint64_t idx, int32_t ppFlag);
    __aicore__ inline void BaseCompute2(uint64_t curTileLen, uint64_t blockCount, uint64_t idx, int32_t ppFlag);
    __aicore__ inline void CopyOutF(uint64_t rowId, uint64_t tileLen, uint64_t length, int32_t ppFlag);
    __aicore__ inline void CanFullLocaOneRow(uint32_t offset1, uint32_t offset2);
    __aicore__ inline void CopyIn(uint32_t dataTileLen, uint32_t offset1, uint32_t offset2, uint32_t blockCount, int32_t ppFlag);
    __aicore__ inline void CopyInDequantBuffer(uint32_t offset1, uint32_t offset2, uint32_t dataTileLen);
    __aicore__ inline void InitCommon(GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm,
        GM_ADDR quant_scale_gm, GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm,
        GM_ADDR userspace, const SwiGluTilingData* tilingData, TPipe* pipe_);
    __aicore__ inline void InitUbBufferCommon(uint64_t tileLength, uint32_t realRowLen);
    __aicore__ inline void CopyOutScale(uint32_t realRowLen);
    __aicore__ inline void CopyInQuantScale(uint64_t dataTileLength, uint64_t offset);

public:
    uint32_t offsetCalc;

    LocalTensor<CalcType> outTmpLocal;
    LocalTensor<CalcType> inputTmpLocal;
    LocalTensor<float> absTempLocal;
    LocalTensor<int16_t> int16Local;
    LocalTensor<CalcType> bLocal;
    LocalTensor<float> swiLocal;

    TBuf<TPosition::VECCALC> calcSwiGluTmpBuf;
    TBuf<TPosition::VECCALC> weightScaleBufA;
    TBuf<TPosition::VECCALC> weightScaleBufB;
    TBuf<TPosition::VECCALC> quantScaleBuf;

    TBuf<TPosition::VECCALC> inQueueAPingBuf;
    TBuf<TPosition::VECCALC> inQueueAPongBuf;
    TBuf<TPosition::VECCALC> inQueueBPingBuf;
    TBuf<TPosition::VECCALC> inQueueBPongBuf;
    TBuf<TPosition::VECCALC> outQueueFPingBuf;
    TBuf<TPosition::VECCALC> outQueueFPongBuf;
    TBuf<TPosition::VECCALC> outQueueSBuf;

    LocalTensor<InType> inALocalPing;
    LocalTensor<InType> inALocalPong;
    LocalTensor<InType> inBLocalPing;
    LocalTensor<InType> inBLocalPong;
    LocalTensor<int8_t> outFLocalPing;
    LocalTensor<int8_t> outFLocalPong;

    int32_t pingPongFlag = 0;
    event_t eventId = EVENT_ID0;
};

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicPerformance<TEMPLATE_ARGS>::Init(
    GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm, GM_ADDR quant_scale_gm,
    GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm, GM_ADDR userspace, const SwiGluTilingData* tilingData,
    TPipe* pipe_)
{
    this->InitCommon(x_gm, weight_scale_gm, activation_scale_gm, bias_gm, quant_scale_gm, quant_offset_gm, y_gm, scale_gm, userspace, tilingData, pipe_);
    this->weightScaleGm.SetGlobalBuffer((__gm__ float*)weight_scale_gm, this->colNum);
    if (this->biasIsEmpty == 0) {
        this->biasGm.SetGlobalBuffer((__gm__ BiasType*)bias_gm, this->colNum);
    }

    if (this->activateScaleIsEmpty == 0) {
        this->activationScaleGm.SetGlobalBuffer((__gm__ float*) activation_scale_gm + this->biasOffset,
            this->numRound);
    }

    this->InitUbBufferCommon(this->baseColLen, this->numRound);

    uint64_t alignTileLength = this->baseColLen;
    if (!this->isOut32BAligned) {
        alignTileLength = this->Align(this->baseColLen, sizeof(int8_t));
    }

    this->pipe->InitBuffer(weightScaleBufA, alignTileLength * sizeof(float));
    this->pipe->InitBuffer(weightScaleBufB,  alignTileLength * sizeof(float));
    this->pipe->InitBuffer(quantScaleBuf, alignTileLength * sizeof(float));

    this->pipe->InitBuffer(calcSwiGluTmpBuf, alignTileLength * sizeof(float) * this->baseRowLen);

    this->pipe->InitBuffer(inQueueAPingBuf, alignTileLength * sizeof(InType) * this->baseRowLen);
    this->pipe->InitBuffer(inQueueAPongBuf, alignTileLength * sizeof(InType) * this->baseRowLen);
    this->pipe->InitBuffer(inQueueBPingBuf, alignTileLength * sizeof(InType) * this->baseRowLen);
    this->pipe->InitBuffer(inQueueBPongBuf, alignTileLength * sizeof(InType) * this->baseRowLen);
    this->pipe->InitBuffer(outQueueFPingBuf, alignTileLength * sizeof(int8_t) * this->baseRowLen);
    this->pipe->InitBuffer(outQueueFPongBuf, alignTileLength * sizeof(int8_t) * this->baseRowLen);
    this->pipe->InitBuffer(outQueueSBuf, this->AlignBytes(this->numRound, sizeof(float)));

    outTmpLocal = this->outputTempBufferBF16D.template Get<CalcType>();
    inputTmpLocal = this->inputTempBufferBF16D.template Get<CalcType>();
    absTempLocal = this->inputTempBufferBF16D.template Get<float>();
    int16Local = this->outputTempBufferBF16D.template Get<int16_t>();
    bLocal = this->inputTempBufferBF16D.template Get<CalcType>();
    swiLocal = calcSwiGluTmpBuf.Get<float>();

    inALocalPing = inQueueAPingBuf.Get<InType>();
    inALocalPong = inQueueAPongBuf.Get<InType>();
    inBLocalPing = inQueueBPingBuf.Get<InType>();
    inBLocalPong = inQueueBPongBuf.Get<InType>();
    outFLocalPing = outQueueFPingBuf.Get<int8_t>(); 
    outFLocalPong = outQueueFPongBuf.Get<int8_t>(); 
    this->maxTempLocal = outQueueSBuf.Get<float>();

    this->weightScaleLocalA = weightScaleBufA.Get<float>();
    this->weightScaleLocalB = weightScaleBufB.Get<float>();
    this->quantScaleLocal = quantScaleBuf.Get<float>();
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicPerformance<TEMPLATE_ARGS>::InitCommon(GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm,
    GM_ADDR quant_scale_gm, GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm,
    GM_ADDR userspace, const SwiGluTilingData* tilingData, TPipe* pipe_)
{
    this->pipe = pipe_;
    this->curBlockIdx = GetBlockIdx();
    this->activateLeft = tilingData->activateLeft;
    this->quantScaleIsEmpty = tilingData->quantScaleIsEmpty;
    this->activateScaleIsEmpty = tilingData->activateScaleIsEmpty;
    this->biasIsEmpty = tilingData->biasIsEmpty;
    this->colNum = tilingData->colLen;
    this->rowNum = tilingData->rowLen;
    this->useCoreNum = tilingData->usedCoreNum;
    // 每行为全载情况下每次最大拷贝行数
    this->baseRowLen = tilingData->baseRowLen;
    this->baseColLen = tilingData->baseColLen;
    if (this->rowNum < this->useCoreNum) {
        this->useCoreNum = this->rowNum;
    }
    // 行全载和不全载情况下，分别判断是否对齐
    this->isMultiCols = this->baseRowLen == 1 && this->baseColLen < this->colNum;
    if (this->isMultiCols) {
        this->isOut32BAligned = this->baseColLen == this->Align(this->baseColLen, sizeof(InType));
    } else {
        this->isOut32BAligned = (this->colNum % this->blockBytes == 0) || (this->baseRowLen == 1);
    }
    this->perRoundCnt = this->useCoreNum == 0 ? 0 : this->rowNum / this->useCoreNum;
    uint32_t remainCnt = this->rowNum - this->useCoreNum * this->perRoundCnt;
    this->numRound = this->perRoundCnt;
    if (this->curBlockIdx < remainCnt) {
        this->numRound = this->perRoundCnt + 1;
        this->biasOffset = this->curBlockIdx * (this->perRoundCnt + 1);
    } else {
        this->biasOffset = (this->perRoundCnt + 1) * remainCnt + (this->curBlockIdx - remainCnt) * this->perRoundCnt;
    }
    this->xGm.SetGlobalBuffer((__gm__ InType*)x_gm + this->biasOffset * this->colNum * DOUBLE, this->colNum * this->numRound * DOUBLE);
    this->scaleGm.SetGlobalBuffer((__gm__ float*)scale_gm, this->rowNum);
    this->yGm.SetGlobalBuffer((__gm__ int8_t*)y_gm + this->biasOffset * this->colNum, this->numRound * this->colNum);
    this->quantScaleGm.SetGlobalBuffer((__gm__ float*)quant_scale_gm, this->colNum);
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicPerformance<TEMPLATE_ARGS>::InitUbBufferCommon(uint64_t tileLength, uint32_t realRowLen)
{
    uint64_t alignTileLength = tileLength;
    if (!this->isOut32BAligned) {
        alignTileLength = this->Align(tileLength, sizeof(int8_t));
    }
    this->pipe->InitBuffer(this->inputTempBufferBF16D, alignTileLength * sizeof(CalcType) * this->baseRowLen);
    this->pipe->InitBuffer(this->outputTempBufferBF16D, alignTileLength * sizeof(CalcType) * this->baseRowLen);
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicPerformance<TEMPLATE_ARGS>::Process() {
    if (this->curBlockIdx >= this->useCoreNum) {
        return;
    }
    uint32_t offset1 = 0;
    uint32_t offset2 = this->colNum;
    if (this->activateLeft == 0) {
        offset1 = this->colNum;
        offset2 = 0;
    }
    this->CanFullLocaOneRow(offset1, offset2);
    this->CopyOutScale(this->numRound);
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicPerformance<TEMPLATE_ARGS>::CopyOutScale(uint32_t realRowLen)
{
    DataCopyExtParams intriParams{1, static_cast<uint32_t>(sizeof(float) * realRowLen), 0, 0, 0};
    DataCopyPad(this->scaleGm[this->biasOffset], this->maxTempLocal, intriParams);
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicPerformance<TEMPLATE_ARGS>::CanFullLocaOneRow(uint32_t offset1, uint32_t offset2)
{
    if (this->quantScaleIsEmpty == 0) {
        this->CopyInQuantScale(this->colNum, 0);
    }
    this->CopyInDequantBuffer(offset1, offset2, this->colNum);

    this->alignSize = this->isOut32BAligned ? this->colNum : this->Align(this->colNum, sizeof(int8_t));
    int64_t blockCount = this->baseRowLen;
    int64_t loops = (this->numRound + blockCount - 1) / blockCount;
    int64_t lastLoopBlockCount = this->numRound - (loops - 1) * blockCount;
    int64_t perLoopColSize = this->alignSize * blockCount;
    offsetCalc = (blockCount == 0 ? 0 : (perLoopColSize / blockCount));
    uint32_t alig8CalcNum = this->Align(this->colNum, sizeof(int8_t));
    this->dstStride = this->isOut32BAligned ? 0 : (alig8CalcNum - this->colNum) * sizeof(InType) / this->blockBytes;
    uint32_t base = (this->colNum * DOUBLE) * this->baseRowLen;

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t i = 0; i < loops; i++) {
        eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;
        event_t eventIdNext = pingPongFlag ? EVENT_ID0 : EVENT_ID1;

        if (i == 0) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            this->CopyIn(this->colNum, offset1 + i * base, offset2 + i * base, blockCount, pingPongFlag);
            SetFlag<HardEvent::MTE2_V>(eventId);
        }

        WaitFlag<HardEvent::MTE2_V>(eventId);
        this->BaseCompute1(perLoopColSize, blockCount, i, pingPongFlag);

        if(i != loops -1) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdNext);
            this->CopyIn(this->colNum, offset1 + (i + 1) * base, offset2 + (i + 1) * base, blockCount, 1 - pingPongFlag);
            SetFlag<HardEvent::MTE2_V>(eventIdNext);
        }

        this->BaseCompute2(perLoopColSize, blockCount, i, pingPongFlag);
        SetFlag<HardEvent::V_MTE3>(eventId);

        WaitFlag<HardEvent::V_MTE3>(eventId);
        this->CopyOutF(i, perLoopColSize, blockCount, pingPongFlag);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);

        pingPongFlag = 1 - pingPongFlag;
    }

    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicPerformance<TEMPLATE_ARGS>::CopyInQuantScale(uint64_t dataTileLength, uint64_t offset)
{
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataTileLength * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(this->quantScaleLocal, this->quantScaleGm[offset], dataCopyParams, dataCopyPadParams);
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicPerformance<TEMPLATE_ARGS>::CopyInDequantBuffer(uint32_t offset1, uint32_t offset2, uint32_t dataTileLen)
{
    DataCopyExtParams params = {1, static_cast<uint32_t>(dataTileLen * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    DataCopyPad(this->weightScaleLocalA, this->weightScaleGm[offset1], params, padParams);
    DataCopyPad(this->weightScaleLocalB, this->weightScaleGm[offset2], params, padParams);
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicPerformance<TEMPLATE_ARGS>::CopyIn(uint32_t dataTileLen, uint32_t offset1, uint32_t offset2, uint32_t blockCount, int32_t ppFlag)
{
    uint32_t srcStride = dataTileLen * sizeof(InType);
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(blockCount),
        static_cast<uint32_t>(dataTileLen * sizeof(InType)), srcStride, this->dstStride, 0};
    DataCopyPadExtParams<InType> dataCopyPadParams{false, 0, 0, 0};
    LocalTensor<InType> aLocal = ppFlag ? inALocalPong : inALocalPing;
    LocalTensor<InType> bLocal = ppFlag ? inBLocalPong : inBLocalPing;
    DataCopyPad(aLocal, this->xGm[offset1], dataCopyParams, dataCopyPadParams);
    DataCopyPad(bLocal, this->xGm[offset2], dataCopyParams, dataCopyPadParams);
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicPerformance<TEMPLATE_ARGS>::BaseCompute1(uint64_t curTileLen, uint64_t blockCount, uint64_t rowId, int32_t ppFlag)
{
    LocalTensor<InType> inALocal = ppFlag ? inALocalPong : inALocalPing;
    LocalTensor<InType> bLocal_ = ppFlag ? inBLocalPong : inBLocalPing;

    float value = this->activationScaleGm.GetValue(rowId);
    Cast(inputTmpLocal, inALocal, RoundMode::CAST_NONE, curTileLen);
    PipeBarrier<PIPE_V>();

    Mul(inputTmpLocal, inputTmpLocal, this->weightScaleLocalA, curTileLen);
    PipeBarrier<PIPE_V>();

    Muls(inputTmpLocal, inputTmpLocal, value, curTileLen);
    PipeBarrier<PIPE_V>();

    Muls(outTmpLocal, inputTmpLocal, this->beta, curTileLen);
    PipeBarrier<PIPE_V>();

    Exp(outTmpLocal, outTmpLocal, curTileLen);
    PipeBarrier<PIPE_V>();

    Adds(outTmpLocal, outTmpLocal, CalcType(1.0), curTileLen);
    PipeBarrier<PIPE_V>();

    Div(outTmpLocal, inputTmpLocal, outTmpLocal, curTileLen);
    PipeBarrier<PIPE_V>();

    Cast(bLocal, bLocal_, RoundMode::CAST_NONE, curTileLen);
    PipeBarrier<PIPE_V>();

    Mul(bLocal, bLocal, this->weightScaleLocalB, curTileLen);
    PipeBarrier<PIPE_V>();

    Muls(bLocal, bLocal, value, curTileLen);
    PipeBarrier<PIPE_V>();

    Mul(swiLocal, outTmpLocal, bLocal, curTileLen);
    PipeBarrier<PIPE_V>();

    if (this->quantScaleIsEmpty == 0) {
        Mul(swiLocal, swiLocal, this->quantScaleLocal, curTileLen);
        PipeBarrier<PIPE_V>();
    }
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicPerformance<TEMPLATE_ARGS>::BaseCompute2(uint64_t curTileLen, uint64_t blockCount, uint64_t rowId, int32_t ppFlag)
{
    LocalTensor<int8_t> outLocal = ppFlag ? outFLocalPong : outFLocalPing;

    Abs(absTempLocal, swiLocal, curTileLen);
    PipeBarrier<PIPE_V>();

    ReduceMax(this->maxTempLocal[rowId * this->baseRowLen], absTempLocal, absTempLocal, this->colNum);
    PipeBarrier<PIPE_V>();

    float value = this->maxTempLocal.GetValue(rowId * this->baseRowLen) / 127;
    this->maxTempLocal.SetValue(rowId * this->baseRowLen, value);
    float scale = 1 / value;
    Muls(swiLocal, swiLocal, scale, this->colNum);
    PipeBarrier<PIPE_V>();

    Cast(int16Local, swiLocal, RoundMode::CAST_RINT, curTileLen);
    PipeBarrier<PIPE_V>();

    // int16-> half
    LocalTensor<half> halfLocal = int16Local.ReinterpretCast<half>();
    Cast(halfLocal, int16Local, RoundMode::CAST_NONE, curTileLen);
    PipeBarrier<PIPE_V>();

    // half -> int8_t
    Cast(outLocal, halfLocal, RoundMode::CAST_NONE, curTileLen);
    PipeBarrier<PIPE_V>();
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicPerformance<TEMPLATE_ARGS>::CopyOutF(uint64_t rowId, uint64_t tileLen, uint64_t length, int32_t ppFlag)
{
    LocalTensor<int8_t> outLocal = ppFlag ? outFLocalPong : outFLocalPing;
    DataCopyExtParams intriParams{1, static_cast<uint32_t>(tileLen), 0, 0, 0};
    DataCopyPad(this->yGm[rowId * this->colNum * this->baseRowLen], outLocal, intriParams);
}

}  // namespace DequantSwigluQuant
#endif  // DEQUANT_SWIGLU_QUANT_DYNAMIC_PERFORMANCE_HPP
