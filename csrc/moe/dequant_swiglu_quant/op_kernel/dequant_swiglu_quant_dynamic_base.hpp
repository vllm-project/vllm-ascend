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
 * \file dequant_swiglu_quant_dynamic_base.hpp
 * \brief
 */

#ifndef CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BASE_HPP
#define CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BASE_HPP
#include "kernel_operator.h"

#define TEMPLATE_DECLARE template<typename InType, typename CalcType, typename BiasType, typename OutType, uint16_t bufferNum, uint16_t quantIsOne>
#define TEMPLATE_ARGS InType, CalcType, BiasType, OutType, bufferNum, quantIsOne
namespace DequantSwigluQuant {
constexpr uint32_t DOUBLE = 2;
using namespace AscendC;

TEMPLATE_DECLARE
class DequantSwigluQuantDynamicBase {
public:
    __aicore__ inline DequantSwigluQuantDynamicBase() {}
    __aicore__ inline ~DequantSwigluQuantDynamicBase() {}

    __aicore__ inline void InitCommon(GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm,
        GM_ADDR quant_scale_gm, GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm,
        GM_ADDR userspace, const SwiGluTilingData* tilingData, TPipe* pipe_)
    {
        pipe = pipe_;
        curBlockIdx = GetBlockIdx();
        activateLeft = tilingData->activateLeft;
        quantScaleIsEmpty = tilingData->quantScaleIsEmpty;
        activateScaleIsEmpty = tilingData->activateScaleIsEmpty;
        biasIsEmpty = tilingData->biasIsEmpty;
        colNum = tilingData->colLen;
        rowNum = tilingData->rowLen;
        useCoreNum = tilingData->usedCoreNum;
        // 每行为全载情况下每次最大拷贝行数
        baseRowLen = tilingData->baseRowLen;
        baseColLen = tilingData->baseColLen;
        if (rowNum < useCoreNum) {
            useCoreNum = rowNum;
        }
        // 行全载和不全载情况下，分别判断是否对齐
        isMultiCols = baseRowLen == 1 && this->baseColLen < this->colNum;
        if (isMultiCols) {
            isOut32BAligned = baseColLen == Align(baseColLen, sizeof(InType));
        } else {
            isOut32BAligned = (colNum % blockBytes == 0) || (baseRowLen == 1);
        }
        perRoundCnt = useCoreNum == 0 ? 0 : rowNum / useCoreNum;
        uint32_t remainCnt = rowNum - useCoreNum * perRoundCnt;
        numRound = perRoundCnt;
        if (curBlockIdx < remainCnt) {
            numRound = perRoundCnt + 1;
            biasOffset = curBlockIdx * (perRoundCnt + 1);
        } else {
            biasOffset = (perRoundCnt + 1) * remainCnt + (curBlockIdx - remainCnt) * perRoundCnt;
        }
        xGm.SetGlobalBuffer((__gm__ InType*)x_gm + biasOffset * colNum * DOUBLE, colNum * numRound * DOUBLE);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale_gm, rowNum);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y_gm + biasOffset * colNum, numRound * colNum);
        quantScaleGm.SetGlobalBuffer((__gm__ float*)quant_scale_gm, colNum);
        if (this->quantScaleIsEmpty == 0) {
            if constexpr (quantIsOne == 1) {
                quant_scale = ((__gm__ float*)quant_scale_gm)[0];
            }
        }
        if (isMultiCols) {
            swigluTmpGm.SetGlobalBuffer((__gm__ float*)userspace + curBlockIdx * colNum, colNum);
        }
    }

    __aicore__ inline void BaseProcess()
    {
        if (this->curBlockIdx >= this->useCoreNum) {
            return;
        }
        this->maxTempLocal = this->outQueueS.template AllocTensor<float>();
        uint32_t offset1 = 0;
        uint32_t offset2 = this->colNum;
        if (this->activateLeft == 0) {
            offset1 = this->colNum;
            offset2 = 0;
        }
        if (!this->isMultiCols) {
            this->CanFullLocaOneRow(offset1, offset2);
        } else {
            this->CanNotFullLocaOneRow(offset1, offset2);
        }
        this->CopyOutScale(this->numRound);
    }

    __aicore__ inline void InitUbBufferCommon(uint64_t tileLength, uint32_t realRowLen)
    {
        uint64_t alignTileLength = tileLength;
        if (!isOut32BAligned) {
            alignTileLength = Align(tileLength, sizeof(int8_t));
        }
        pipe->InitBuffer(inputTempBufferBF16D, alignTileLength * sizeof(CalcType) * baseRowLen);
        pipe->InitBuffer(outputTempBufferBF16D, alignTileLength * sizeof(CalcType) * baseRowLen);
        pipe->InitBuffer(inQueueA, 1, alignTileLength * sizeof(InType) * baseRowLen);
        pipe->InitBuffer(inQueueB, 1, alignTileLength * sizeof(InType) * baseRowLen);
        pipe->InitBuffer(swiGluQueue, 1, alignTileLength * sizeof(float) * baseRowLen);
        if (quantScaleIsEmpty == 0) {
            if (quantIsOne == 0) {
                pipe->InitBuffer(inQueueQuantScale, bufferNum, alignTileLength * sizeof(float));
            }
        }
        pipe->InitBuffer(outQueueF, 1, alignTileLength * sizeof(int8_t) * baseRowLen);
        pipe->InitBuffer(outQueueS, 1, AlignBytes(realRowLen, sizeof(float)));
    }

    __aicore__ inline float dynamicMultiColMax(uint64_t rowId, uint64_t tileLen, int64_t colLoop)
    {
        LocalTensor<float> swiLocal = swiGluQueue.template DeQue<float>();
        LocalTensor<float> absTempLocal = inputTempBufferBF16D.Get<float>();
        Abs(absTempLocal, swiLocal, tileLen);
        PipeBarrier<PIPE_V>();
        ReduceMax(maxTempLocal[rowId], absTempLocal, absTempLocal, tileLen);
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(tileLen * sizeof(float)), 0, 0, 0};
        DataCopyPad(swigluTmpGm[colLoop * baseColLen], swiLocal, dataCopyParams);
        swiGluQueue.FreeTensor(swiLocal);
        return maxTempLocal.GetValue(rowId);
    }

    __aicore__ inline void dynamicAllColOut(uint64_t rowId, uint64_t tileLen, int64_t colLoop, float scale)
    {
        LocalTensor<float> swiLocal = swiGluQueue.template AllocTensor<float>();
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(tileLen * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> dataCopyPadParams{false, 0, 0, 0};
        DataCopyPad(swiLocal, swigluTmpGm[colLoop * baseColLen], dataCopyParams, dataCopyPadParams);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
        Muls(swiLocal, swiLocal, scale, tileLen);
        PipeBarrier<PIPE_V>();
        LocalTensor<int16_t> int16Local = outputTempBufferBF16D.Get<int16_t>();
        Cast(int16Local, swiLocal, RoundMode::CAST_RINT, tileLen);
        PipeBarrier<PIPE_V>();
        swiGluQueue.FreeTensor(swiLocal);
        // int16-> half
        LocalTensor<half> halfLocal = int16Local.ReinterpretCast<half>();
        Cast(halfLocal, int16Local, RoundMode::CAST_NONE, tileLen);
        PipeBarrier<PIPE_V>();
        // half -> int8_t
        LocalTensor<int8_t> outLocal = outQueueF.template AllocTensor<int8_t>();
        Cast(outLocal, halfLocal, RoundMode::CAST_NONE, tileLen);
        outQueueF.EnQue(outLocal);
        outLocal = outQueueF.DeQue<int8_t>();
        event_t eventId2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventId2);
        WaitFlag<HardEvent::MTE3_S>(eventId2);
        DataCopyExtParams intriParams{1, static_cast<uint32_t>(tileLen), 0, 0, 0};
        DataCopyPad(yGm[rowId * colNum + colLoop * baseColLen], outLocal, intriParams);
        outQueueF.FreeTensor(outLocal);
    }

    __aicore__ inline void DynamicCompute(uint64_t rowId, uint64_t tileLen, uint64_t length)
    {
        LocalTensor<float> swiLocal = swiGluQueue.template DeQue<float>();
        LocalTensor<float> absTempLocal = inputTempBufferBF16D.Get<float>();
        Abs(absTempLocal, swiLocal, tileLen);
        uint32_t offsetCalc = (length == 0 ? 0 : (tileLen / length));
        PipeBarrier<PIPE_V>();
        for (int64_t i = 0; i < length; i++) {
            ReduceMax(maxTempLocal[rowId * baseRowLen + i], absTempLocal[i * offsetCalc], absTempLocal[i * offsetCalc], colNum);
            PipeBarrier<PIPE_V>();

            event_t eventIdV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdV2S);
            WaitFlag<HardEvent::V_S>(eventIdV2S);
            float value = maxTempLocal.GetValue(rowId * baseRowLen + i) / 127;
            maxTempLocal.SetValue(rowId * baseRowLen + i, value);
            float scale = 1 / value;
            Muls(swiLocal[i * offsetCalc], swiLocal[i * offsetCalc], scale, colNum);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<int16_t> int16Local = outputTempBufferBF16D.Get<int16_t>();
        Cast(int16Local, swiLocal, RoundMode::CAST_RINT, tileLen);
        PipeBarrier<PIPE_V>();

        // int16-> half
        LocalTensor<half> halfLocal = int16Local.ReinterpretCast<half>();
        Cast(halfLocal, int16Local, RoundMode::CAST_NONE, tileLen);
        PipeBarrier<PIPE_V>();
        swiGluQueue.FreeTensor(swiLocal);

        // half -> int8_t
        LocalTensor<int8_t> outLocal = outQueueF.template AllocTensor<int8_t>();
        Cast(outLocal, halfLocal, RoundMode::CAST_NONE, tileLen);

        event_t eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId1);
        WaitFlag<HardEvent::V_MTE3>(eventId1);

        if (isOut32BAligned) {
            DataCopyExtParams intriParams{1, static_cast<uint32_t>(tileLen), 0, 0, 0};
            DataCopyPad(yGm[rowId * colNum * baseRowLen], outLocal, intriParams);
        } else {
            DataCopyExtParams intriParams{static_cast<uint16_t>(length), static_cast<uint32_t>(colNum), 0, 0, 0};
            DataCopyPad(yGm[rowId * colNum * baseRowLen], outLocal, intriParams);
        }
        outQueueF.FreeTensor(outLocal);
    }

    __aicore__ inline void BaseComputeWithQuantScale(LocalTensor<CalcType> &outTmpLocal, LocalTensor<CalcType> &bLocal,
        uint64_t curTileLen,  uint64_t blockCount)
    {
        LocalTensor<float> swiLocal = swiGluQueue.template AllocTensor<float>();
        Mul(swiLocal, outTmpLocal, bLocal, curTileLen);
        if (quantScaleIsEmpty == 0) {
            PipeBarrier<PIPE_V>();
            if constexpr (quantIsOne == 0) {
                uint32_t calcOffset = (blockCount == 0 ? 0 : curTileLen / blockCount);
                for (uint64_t idx = 0; idx < blockCount; idx++) {
                    Mul(swiLocal[idx * calcOffset], swiLocal[idx * calcOffset], quantScaleLocal, calcOffset);
                }
            } else {
                Muls(swiLocal, swiLocal, quant_scale, curTileLen);
            }
        }
        swiGluQueue.template EnQue<float>(swiLocal);
    }

    __aicore__ inline void BaseCompute(uint64_t curTileLen, uint64_t blockCount, uint64_t idx)
    {
        LocalTensor<InType> inALocal = this->inQueueA.template DeQue<InType>();
        LocalTensor<CalcType> outTmpLocal = this->outputTempBufferBF16D.template Get<CalcType>();
        LocalTensor<CalcType> inputTmpLocal = this->inputTempBufferBF16D.template Get<CalcType>();
        float value = this->getActivateScaleValue(idx);
        if constexpr (std::is_same_v<InType, int32_t>) {
            if constexpr (std::is_same_v<BiasType, int32_t>) {
                this->addBiasWithBiasInt(inALocal, this->biasLocalA, curTileLen);
            }
        }
        Cast(inputTmpLocal, inALocal, RoundMode::CAST_NONE, curTileLen);
        PipeBarrier<PIPE_V>();
        this->inQueueA.template FreeTensor(inALocal);
        if constexpr (std::is_same_v<InType, int32_t>) {
            addWeightScaleAndActivateScale(inputTmpLocal, this->weightScaleLocalA, curTileLen, value);
            if constexpr (std::is_same_v<BiasType, float> || std::is_same_v<BiasType, bfloat16_t> || std::is_same_v<BiasType, half>) {
                if (this->biasIsEmpty == 0) {
                    addBiasWithBiasFloat(inputTmpLocal, this->biasLocalA, curTileLen);
                }
            }
        }
        Muls(outTmpLocal, inputTmpLocal, this->beta, curTileLen);
        PipeBarrier<PIPE_V>();
        Exp(outTmpLocal, outTmpLocal, curTileLen);
        PipeBarrier<PIPE_V>();
        Adds(outTmpLocal, outTmpLocal, CalcType(1.0), curTileLen);
        PipeBarrier<PIPE_V>();
        Div(outTmpLocal, inputTmpLocal, outTmpLocal, curTileLen);
        PipeBarrier<PIPE_V>();
        LocalTensor<InType> bLocal_ = this->inQueueB.template DeQue<InType>();
        if constexpr (std::is_same_v<InType, int32_t>) {
            if constexpr (std::is_same_v<BiasType, int32_t>) {
                this->addBiasWithBiasInt(bLocal_, this->biasLocalB, curTileLen);
            }
        }
        LocalTensor<CalcType> bLocal = this->inputTempBufferBF16D.template Get<CalcType>();
        Cast(bLocal, bLocal_, RoundMode::CAST_NONE, curTileLen);
        PipeBarrier<PIPE_V>();
        if constexpr (std::is_same_v<InType, int32_t>) {
            addWeightScaleAndActivateScale(bLocal, this->weightScaleLocalB, curTileLen, value);
            if constexpr (std::is_same_v<BiasType, float> || std::is_same_v<BiasType, bfloat16_t> || std::is_same_v<BiasType, half>) {
                if (this->biasIsEmpty == 0) {
                    addBiasWithBiasFloat(bLocal, this->biasLocalB, curTileLen);
                }
            }
        }
        this->inQueueB.template FreeTensor(bLocal_);
        BaseComputeWithQuantScale(outTmpLocal, bLocal, curTileLen, blockCount);
    }

    __aicore__ inline void CanFullLocaOneRow(uint32_t offset1, uint32_t offset2)
    {
        if (this->quantScaleIsEmpty == 0) {
            if constexpr (quantIsOne == 0) {
                this->CopyInQuantScale(this->colNum, 0);
            }
        }
        CopyInDequantBuffer(offset1, offset2, this->colNum);
        this->alignSize = this->isOut32BAligned ? this->colNum : this->Align(this->colNum, sizeof(int8_t));

        int64_t blockCount = this->baseRowLen;
        int64_t loops = (this->numRound + blockCount - 1) / blockCount;
        int64_t lastLoopBlockCount = this->numRound - (loops - 1) * blockCount;
        int64_t lastLoopColSize = this->alignSize * lastLoopBlockCount;
        int64_t perLoopColSize = this->alignSize * blockCount;

        uint32_t aligCalcNum = this->Align(this->colNum, sizeof(InType));
        uint32_t alig8CalcNum = this->Align(this->colNum, sizeof(int8_t));
        this->dstStride = this->isOut32BAligned ? 0 : (alig8CalcNum - this->colNum) * sizeof(InType) / this->blockBytes;
        for (uint32_t i = 0; i < loops - 1; i++) {
            uint32_t base = i * (this->colNum * DOUBLE) * this->baseRowLen;
            this->CopyIn(this->colNum, offset1 + base, offset2 + base, blockCount);
            this->BaseCompute(perLoopColSize, blockCount, i);
            this->DynamicCompute(i, perLoopColSize, blockCount);
        }
        uint32_t base = (loops - 1) * (this->colNum * DOUBLE) * this->baseRowLen;
        this->CopyIn(this->colNum, offset1 + base, offset2 + base, lastLoopBlockCount);
        this->BaseCompute(lastLoopColSize, lastLoopBlockCount, (loops - 1));
        this->DynamicCompute((loops - 1), lastLoopColSize, lastLoopBlockCount);

        if (this->quantScaleIsEmpty == 0) {
            if constexpr (quantIsOne == 0) {
                this->inQueueQuantScale.FreeTensor(this->quantScaleLocal);
            }
        }
        FreeDequantBuffer();
    }

    __aicore__ inline void CanNotFullLocaOneRow(uint32_t offset1, uint32_t offset2)
    {
        int64_t colLoops = (this->colNum + this->baseColLen - 1) / this->baseColLen;
        int64_t lastColNum = this->colNum - (colLoops - 1) * this->baseColLen;
        for (uint32_t i = 0; i < this->numRound; i++) {
            uint32_t tmp = 0xFF7FFFFF;
            float reduceMax = *((float*)&tmp);
            for (uint32_t j = 0; j < colLoops; j++) {
                int64_t curColNum = this->baseColLen;
                if (j == colLoops - 1) {
                    curColNum = lastColNum;
                }
                if (this->quantScaleIsEmpty == 0) {
                    if constexpr (quantIsOne == 0) {
                        this->CopyInQuantScale(curColNum, j * this->baseColLen);
                    }
                }
                bool isOutAligned = curColNum == this->Align(curColNum, sizeof(InType));
                uint32_t alignColNum = isOutAligned ? curColNum : this->Align(curColNum, sizeof(OutType));
                uint32_t base = i * (this->colNum * DOUBLE) + j * this->baseColLen;
                CopyInDequantBuffer(offset1 + j * this->baseColLen, offset2 + j * this->baseColLen, curColNum);
                this->CopyIn(curColNum, offset1 + base, offset2 + base, 1);
                this->BaseCompute(alignColNum, 1, i);
                float maxValue = this->dynamicMultiColMax(i, curColNum, j);
                if (maxValue > reduceMax) {
                    reduceMax = maxValue;
                }
                if (this->quantScaleIsEmpty == 0) {
                    if constexpr (quantIsOne == 0) {
                        this->inQueueQuantScale.FreeTensor(this->quantScaleLocal);
                    }
                }
                FreeDequantBuffer();
            }
            float value = reduceMax / 127.0f;
            this->maxTempLocal.SetValue(i, value);
            float scale = 1 / value;
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            for (uint32_t j = 0; j < colLoops; j++) {
                int64_t curColNum = this->baseColLen;
                if (j == colLoops - 1) {
                    curColNum = lastColNum;
                }
                bool isOutAligned = curColNum == this->Align(curColNum, sizeof(InType));
                uint32_t alignColNum = isOutAligned ? curColNum : this->Align(curColNum, sizeof(OutType));
                this->dynamicAllColOut(i, curColNum, j, scale);
            }
        }
    }

    __aicore__ inline void CopyInDequantBuffer(uint32_t offset1, uint32_t offset2, uint32_t dataTileLen)
    {
        if constexpr (std::is_same_v<InType, int32_t>) {
            this->CopyInWeightAndBias(dataTileLen, offset1, offset2);
            this->CopyInActivateScale(0, this->numRound);
            if (this->biasIsEmpty == 0) {
                this->biasLocalA = this->inBiasQueueA.template DeQue<BiasType>();
                this->biasLocalB = this->inBiasQueueB.template DeQue<BiasType>();
            }
            this->weightScaleLocalA = this->weightScaleQueueA.template DeQue<float>();
            this->weightScaleLocalB = this->weightScaleQueueB.template DeQue<float>();
            if (this->activateScaleIsEmpty == 0) {
                this->activateLocal = this->inQueueActivationScale.template DeQue<float>();
            }
        }
    }

    __aicore__ inline void FreeDequantBuffer()
    {
        if constexpr (std::is_same_v<InType, int32_t>) {
            if (this->biasIsEmpty == 0) {
                this->inBiasQueueA.FreeTensor(this->biasLocalA);
                this->inBiasQueueB.FreeTensor(this->biasLocalB);
            }
            this->weightScaleQueueA.FreeTensor(this->weightScaleLocalA);
            this->weightScaleQueueB.FreeTensor(this->weightScaleLocalB);
            if (this->activateScaleIsEmpty == 0) {
                this->inQueueActivationScale.FreeTensor(this->activateLocal);
            }
        }
    }

    __aicore__ inline float getActivateScaleValue(uint64_t idx)
    {
        float value = 1;
        if constexpr (std::is_same_v<InType, int32_t>) {
            if (activateScaleIsEmpty == 0) {
                event_t eventIdM2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(eventIdM2S);
                WaitFlag<HardEvent::MTE2_S>(eventIdM2S);
                value = activateLocal.GetValue(idx);
            }
        }
        return value;
    }

    __aicore__ inline void addWeightScaleAndActivateScale(
        LocalTensor<CalcType> &dstLocal, LocalTensor<CalcType> &weightScaleLocal, uint64_t curTileLen, float value)
    {
        Mul(dstLocal, dstLocal, weightScaleLocal, curTileLen);
        PipeBarrier<PIPE_V>();
        if (activateScaleIsEmpty == 0) {
            Muls(dstLocal, dstLocal, value, curTileLen);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void addBiasWithBiasInt(LocalTensor<InType> &dstLocal, LocalTensor<BiasType> &biasLocal, uint64_t curTileLen)
    {
        if (this->biasIsEmpty == 0) {
            Add(dstLocal, dstLocal, biasLocal, curTileLen);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void addBiasWithBiasFloat(LocalTensor<CalcType> &dstLocal, LocalTensor<BiasType> &biasLocal, uint64_t curTileLen)
    {
        if (this->biasIsEmpty == 0) {
            if constexpr (std::is_same_v<BiasType, float>) {
                Add(dstLocal, dstLocal, biasLocal, curTileLen);
            } else {
                Cast(biasFloatLocalB, biasLocal, RoundMode::CAST_NONE, curTileLen);
                PipeBarrier<PIPE_V>();
                Add(dstLocal, dstLocal, biasFloatLocalB, curTileLen);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyOutScale(uint32_t realRowLen)
    {
        DataCopyExtParams intriParams{1, static_cast<uint32_t>(sizeof(float) * realRowLen), 0, 0, 0};
        DataCopyPad(scaleGm[biasOffset], maxTempLocal, intriParams);
        outQueueS.FreeTensor(maxTempLocal);
    }

    __aicore__ inline void CopyInActivateScale(uint32_t offset3, uint32_t blockCount)
    {
        if (activateScaleIsEmpty == 0) {
            DataCopyExtParams activateparams = {1, static_cast<uint32_t>(blockCount * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            LocalTensor<float> activateLocal1 = inQueueActivationScale.template AllocTensor<float>();
            DataCopyPad(activateLocal1, activationScaleGm[offset3], activateparams, padParams);
            inQueueActivationScale.EnQue(activateLocal1);
        }
    }

    __aicore__ inline void CopyInWeightAndBias(uint32_t dataTileLen, uint32_t offset1, uint32_t offset2)
    {
        DataCopyExtParams params = {1, static_cast<uint32_t>(dataTileLen * sizeof(float)), 0, 0, 0};
        DataCopyExtParams paramsBias = {1, static_cast<uint32_t>(dataTileLen * sizeof(BiasType)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPadExtParams<BiasType> padParams1{false, 0, 0, 0};

        if (this->biasIsEmpty == 0) {
            // copy bias A
            LocalTensor<BiasType> biasLocalA1 = inBiasQueueA.template AllocTensor<BiasType>();
            DataCopyPad(biasLocalA1, biasGm[offset1], paramsBias, padParams1);
            inBiasQueueA.EnQue(biasLocalA1);
            // copy bias B
            LocalTensor<BiasType> biasLocalB1 = inBiasQueueB.template AllocTensor<BiasType>();
            DataCopyPad(biasLocalB1, biasGm[offset2], paramsBias, padParams1);
            inBiasQueueB.EnQue(biasLocalB1);
        }
        // copy ws A
        LocalTensor<float> wsLocalA1 = weightScaleQueueA.template AllocTensor<float>();
        DataCopyPad(wsLocalA1, weightScaleGm[offset1], params, padParams);
        weightScaleQueueA.EnQue(wsLocalA1);
        // copy ws B
        LocalTensor<float> wsLocalB1 = weightScaleQueueB.template AllocTensor<float>();
        DataCopyPad(wsLocalB1, weightScaleGm[offset2], params, padParams);
        weightScaleQueueB.EnQue(wsLocalB1);
    }

    __aicore__ inline void CopyInQuantScale(uint64_t dataTileLength, uint64_t offset)
    {
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataTileLength * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> dataCopyPadParams{false, 0, 0, 0};
        LocalTensor<float> scaleLocal = inQueueQuantScale.template AllocTensor<float>();
        DataCopyPad(scaleLocal, quantScaleGm[offset], dataCopyParams, dataCopyPadParams);
        inQueueQuantScale.EnQue(scaleLocal);
        quantScaleLocal = inQueueQuantScale.template DeQue<float>();
    }

    __aicore__ inline void CopyIn(uint32_t dataTileLen, uint32_t offset1, uint32_t offset2, uint32_t blockCount)
    {
        uint32_t srcStride = dataTileLen * sizeof(InType);
        DataCopyExtParams dataCopyParams{static_cast<uint16_t>(blockCount),
            static_cast<uint32_t>(dataTileLen * sizeof(InType)), srcStride, dstStride, 0};
        DataCopyPadExtParams<InType> dataCopyPadParams{false, 0, 0, 0};
        // Copy A
        LocalTensor<InType> aLocal = inQueueA.template AllocTensor<InType>();
        DataCopyPad(aLocal, xGm[offset1], dataCopyParams, dataCopyPadParams);
        inQueueA.EnQue(aLocal);
        // Copy B
        LocalTensor<InType> bLocal = inQueueB.template AllocTensor<InType>();
        DataCopyPad(bLocal, xGm[offset2], dataCopyParams, dataCopyPadParams);
        inQueueB.EnQue(bLocal);
    }

    __aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
    {
        if (bytes == 0) {
            return 0;
        }
        return (elementNum * bytes + blockBytes - 1) / blockBytes * blockBytes / bytes;
    }

    __aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
    {
        return (elementNum * bytes + blockBytes - 1) / blockBytes * blockBytes;
    }
protected:
    TPipe* pipe;
    GlobalTensor<InType> xGm;
    GlobalTensor<float> quantScaleGm;
    GlobalTensor<float> weightScaleGm;
    GlobalTensor<float> activationScaleGm;
    GlobalTensor <BiasType> biasGm;
    GlobalTensor<OutType> yGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> swigluTmpGm;
    TBuf<TPosition::VECCALC> inputTempBufferBF16D;
    TBuf<TPosition::VECCALC> outputTempBufferBF16D;
    TBuf<TPosition::VECCALC> inputBiasTempBufferA;
    TBuf<TPosition::VECCALC> inputBiasTempBufferB;
    TQue<QuePosition::VECIN, 1> inQueueA;
    TQue<QuePosition::VECIN, 1> inQueueB;
    TQue<QuePosition::VECIN, 1> inQueueQuantScale;
    TQue<QuePosition::VECOUT, 1> swiGluQueue;
    TQue<QuePosition::VECOUT, 1> outQueueF;
    TQue<QuePosition::VECOUT, 1> outQueueS;
    TQue<QuePosition::VECIN, 1> inBiasQueueA;
    TQue<QuePosition::VECIN, 1> inBiasQueueB;
    TQue<QuePosition::VECIN, 1> weightScaleQueueA;
    TQue<QuePosition::VECIN, 1> weightScaleQueueB;
    TQue<QuePosition::VECIN, 1> inQueueActivationScale;
    LocalTensor<float> maxTempLocal;
    LocalTensor<float> quantScaleLocal;
    LocalTensor<float> weightScaleLocalA;
    LocalTensor<float> weightScaleLocalB;
    LocalTensor <BiasType> biasLocalA;
    LocalTensor <BiasType> biasLocalB;
    LocalTensor<float> activateLocal;
    LocalTensor<CalcType> biasFloatLocalA;
    LocalTensor<CalcType> biasFloatLocalB;
    float beta = -1.0f;
    float quant_scale = 1;
    uint32_t quantScaleIsEmpty = 0;
    uint32_t biasIsEmpty = 0;
    uint32_t activateScaleIsEmpty = 0;
    uint64_t perRoundCnt = 0;
    uint64_t numRound = 0;
    uint32_t colNum = 0;
    uint32_t rowNum = 0;
    uint32_t useCoreNum = 0;
    uint32_t biasOffset = 0;
    uint32_t curBlockIdx = 0;
    uint32_t activateLeft = 0;
    uint32_t baseRowLen = 0;
    uint32_t baseColLen = 0;
    uint32_t alignSize = 0;
    bool isOut32BAligned = true;
    uint32_t dstStride = 0;
    bool isMultiCols = false;
    int64_t blockBytes = 32;
};
}
#endif  // CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BASE_HPP
