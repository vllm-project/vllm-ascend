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
 * \file swi_glu_dynamic_quant.h
 * \brief
 */
#ifndef SWI_GLU_DYNAMIC_QUANT_H
#define SWI_GLU_DYNAMIC_QUANT_H

#include "swi_glu_dynamic_quant_base.h"

namespace SwiGluDynamicQuantOpt {
using namespace AscendC;

template <typename inType, typename outType>
class SwiGluDynamicQuant : public SwiGluDynamicQuantBase {
public:
    __aicore__ inline SwiGluDynamicQuant(TPipe *pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(GM_ADDR input_gm, GM_ADDR smooth_scales, GM_ADDR offsets, GM_ADDR group_index,
        GM_ADDR y_gm, GM_ADDR scale_gm, GM_ADDR workspace, const SwiGluDynamicQuantTilingData *__restrict tilingData)
    {
        ParseTilingData(tilingData);
        InitParams(sizeof(inType), sizeof(outType));
        InitBaseBuffer();
        InitAndSetBuffer(input_gm, smooth_scales, offsets, group_index, y_gm, scale_gm);
    }

    __aicore__ inline void Process()
    {
        GroupCopyIn();
        SyncAll();
        groupLocal = groupQueue.DeQue<int32_t>();

        DuplicateConst();
        ProcessCoreMultiUbMulti();

        groupQueue.FreeTensor(groupLocal);
    }

    __aicore__ inline void DuplicateConst()
    {
        constScale = tmpConstBuffer.Get<float>();
        if (tilingData_.dstType == DT_INT8) {
            Duplicate<float>(constScale, SWI_GLU_DQ_INT8_SYM_SCALE, MAX_VALUE_NUM);
        } else if (tilingData_.dstType == DT_INT4) {
            Duplicate<float>(constScale, SWI_GLU_DQ_INT4_SYM_SCALE, MAX_VALUE_NUM);
        }
    }

private:
    __aicore__ inline void InitAndSetBuffer(GM_ADDR input_gm, GM_ADDR smooth_scales, GM_ADDR offsets,
        GM_ADDR group_index, GM_ADDR y_gm, GM_ADDR scale_gm)
    {
        xGm.SetGlobalBuffer((__gm__ inType *)input_gm, SPLIT_NUM * tilingData_.rowLen * tilingData_.colLen);
        yGm.SetGlobalBuffer((__gm__ int8_t *)y_gm, tilingData_.rowLen * tilingData_.colLen);
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        yGmInt4.SetGlobalBuffer((__gm__ int4b_t *)y_gm, tilingData_.rowLen * tilingData_.colLen);
#endif
        smooth_scales_Gm.SetGlobalBuffer((__gm__ float *)smooth_scales, tilingData_.groupLen * tilingData_.colLen);
        group_index_Gm.SetGlobalBuffer((__gm__ int32_t *)group_index, tilingData_.groupLen);
        offsetsGm.SetGlobalBuffer((__gm__ float *)offsets, tilingData_.groupLen);
        scale_Gm.SetGlobalBuffer((__gm__ float *)scale_gm, tilingData_.rowLen);

        pPipe->InitBuffer(inQueueA, BUFFER_NUM, tileLength * sizeof(inType));
        pPipe->InitBuffer(inQueueB, BUFFER_NUM, tileLength * sizeof(inType));
        pPipe->InitBuffer(outQueueY, BUFFER_NUM, outLen * sizeof(outType));
        pPipe->InitBuffer(scaleQueue, BUFFER_NUM, basicRowLen * sizeof(float));
        pPipe->InitBuffer(groupQueue, BUFFER_NUM, alignedGroupLen * sizeof(int32_t));
        pPipe->InitBuffer(smoothQueue, BUFFER_NUM, sizeHalfLen * sizeof(float));

        pPipe->InitBuffer(sharedTempBuf, tileLength * sizeof(float));
        pPipe->InitBuffer(tempBufferY, tileLength * sizeof(float));
        pPipe->InitBuffer(tempYUnit, sizeHalfLen * sizeof(float));
    }

    __aicore__ inline uint32_t GetSmoothIndex(uint32_t realRowNum, int32_t &groupNum, uint32_t smoothIndex)
    {
        for (size_t index = smoothIndex; index < tilingData_.groupLen; index++) {
            groupNum = groupLocal.GetValue(index);
            if (groupNum >= realRowNum) {
                return index;
            }
        }
        return SMOOTH_INDEX_UPBOUND;
    }

    __aicore__ inline void GroupCopyIn()
    {
        LocalTensor<int32_t> groupLocal = groupQueue.AllocTensor<int32_t>();
        if (tilingData_.hasGroup == 1) {
            uint8_t rightPadding = alignedGroupLen - tilingData_.groupLen;
            DataCopyParams copyParams{1, (uint16_t)(tilingData_.groupLen * sizeof(int32_t)), 0, 0};
            DataCopyPadParams padParams{true, 0, rightPadding, 0};
            DataCopyPad(groupLocal, group_index_Gm, copyParams, padParams);
            if (tilingData_.groupListType == 1) {
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                for (uint32_t i = 1; i < tilingData_.groupLen; i++) {
                    groupLocal.SetValue(i, groupLocal.GetValue(i - 1) + groupLocal.GetValue(i));
                }
            }
        }
        groupQueue.EnQue(groupLocal);
    }

    __aicore__ inline void SmoothCopyIn(uint32_t offset)
    {
        LocalTensor<float> smoothLocal = smoothQueue.AllocTensor<float>();
        if (smoothIsPad) {
            DataCopyParams copyParams{1, (uint16_t)(basicColLen * sizeof(float)), 0, 0};
            DataCopyPadParams padParams{false, 0, smoothRightPadding, 0};
            DataCopyPad(smoothLocal, smooth_scales_Gm[offset], copyParams, padParams);
        } else {
            DataCopy(smoothLocal, smooth_scales_Gm[offset], basicColLen);
        }
        smoothQueue.EnQue(smoothLocal);
    }

    __aicore__ inline void ProcessCoreMultiUbMulti()
    {
        uint32_t smoothIndex = 0;
        uint32_t offsetRow = 0;

        for (uint32_t ridx = 0; ridx < rowLoop; ridx++) {
            offsetRow = baseRow + ridx * basicRowLen;

            basicRowLenCal = static_cast<uint16_t>((ridx == rowLoop - 1)
                                                       ? (rowLenPerCore - (rowLoop - 1) * basicRowLen)
                                                       : basicRowLen);
            ProcessCoreMultiUbMultiAlign(ridx, smoothIndex, offsetRow);
        }
    }

    __aicore__ inline void ComputeVecInGmOffset(uint32_t ridx)
    {
        if (coreIdx < headCoreNum) {
            offsetParam.tmpVecGmOffset = static_cast<uint64_t>(coreIdx) * rowLenPerCore * mergedColLen + ridx * basicRowLen * mergedColLen;
            splitCopyoutOffset = static_cast<uint64_t>(coreIdx) * rowLenPerCore * colLen + ridx * basicRowLen * basicColLen;
        }
        else {
            offsetParam.tmpVecGmOffset = static_cast<uint64_t>(headCoreNum) * tilingData_.rowLenPerHeadCore * mergedColLen +
                                         static_cast<uint64_t>(coreIdx - headCoreNum) * rowLenPerCore * mergedColLen +
                                         ridx * basicRowLen * mergedColLen;
            splitCopyoutOffset = static_cast<uint64_t>(headCoreNum) * tilingData_.rowLenPerHeadCore * colLen +
                                 static_cast<uint64_t>(coreIdx - headCoreNum) * rowLenPerCore * colLen +
                                 ridx * basicRowLen * basicColLen;
        }
    }

    __aicore__ inline void ProcessCoreMultiUbMultiAlign(uint32_t ridx, uint32_t &smoothIndex, uint32_t offsetRow)
    {
        DataCopyParams splitCopyinParams;
        DataCopyParams splitCopyoutParams;

        splitCopyinParams = {basicRowLenCal,
            (uint16_t)(basicColLen * sizeof(inType) / blockUnit),
            (uint16_t)((mergedColLen - basicColLen) * sizeof(inType) / blockUnit),
            0};

        uint16_t dstStride = (uint16_t)((colLen - basicColLen) * sizeof(outType));
        uint16_t blockLen = (uint16_t)(basicColLen * sizeof(outType));
        if (tilingData_.dstType == DT_INT4) {
            dstStride = CeilDiv(dstStride, (uint16_t)ONE_BYTE_INT4_NUM_TWO);
            blockLen =  CeilDiv(blockLen, (uint16_t)ONE_BYTE_INT4_NUM_TWO);
        }

        splitCopyoutParams = {basicRowLenCal, blockLen, 0, dstStride};

        ComputeVecInGmOffset(ridx);

        if (tilingData_.activateLeft == 1){
            offsetParam.splitVecGmOffset1 = offsetParam.tmpVecGmOffset;
            offsetParam.splitVecGmOffset2 = offsetParam.splitVecGmOffset1 + tilingData_.colLen;
        }
        else{
            offsetParam.splitVecGmOffset2 = offsetParam.tmpVecGmOffset;
            offsetParam.splitVecGmOffset1 = offsetParam.splitVecGmOffset2 + tilingData_.colLen;
        }

        uint32_t smoothScalesOffset = smoothIndex * tilingData_.colLen;

        CopyIn(offsetParam, smoothScalesOffset, splitCopyinParams);
        Compute(offsetRow, smoothIndex);
        CopyOut(splitCopyoutOffset, splitCopyoutParams, ridx, basicRowLenCal);
    }

    __aicore__ inline void CopyIn(
        XxGluSingleTileOffsetParam &offsetParam, uint32_t smoothScalesOffset, DataCopyParams &splitCopyinParams)
    {
        LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();

        if (isPad) {
            DataCopyPadParams padParams{false, 0, rightPadding, 0};
            DataCopyPad(aLocal, this->xGm[offsetParam.splitVecGmOffset1], splitCopyinParams, padParams);
            DataCopyPad(bLocal, this->xGm[offsetParam.splitVecGmOffset2], splitCopyinParams, padParams);
        }else {
            DataCopy(aLocal, this->xGm[offsetParam.splitVecGmOffset1], splitCopyinParams);
            DataCopy(bLocal, this->xGm[offsetParam.splitVecGmOffset2], splitCopyinParams);
        }

        this->inQueueA.template EnQue(aLocal);
        this->inQueueB.template EnQue(bLocal);
        SmoothCopyIn(smoothScalesOffset);
    }

    __aicore__ inline void Compute(uint32_t offsetRow, uint32_t &smoothIndex)
    {
        LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
        LocalTensor<float> tmpALocal = sharedTempBuf.Get<float>();
        tmpYLocal = tempBufferY.Get<float>();
        LocalTensor<inType> aLocal = inQueueA.template DeQue<inType>();

        if constexpr (sizeof(inType) == sizeof(float)) {
            DataCopy(tmpALocal, aLocal, tileLength);
        } else {
            Cast(tmpALocal, aLocal, RoundMode::CAST_NONE, tileLength);
        }

        inQueueA.template FreeTensor(aLocal);
        Muls(tmpYLocal, tmpALocal, static_cast<float>(-1.0), tileLength);
        PipeBarrier<PIPE_V>();
        Exp(tmpYLocal, tmpYLocal, tileLength);
        PipeBarrier<PIPE_V>();
        Adds(tmpYLocal, tmpYLocal, static_cast<float>(1.0), tileLength);
        PipeBarrier<PIPE_V>();
        Div(tmpYLocal, tmpALocal, tmpYLocal, tileLength);
        PipeBarrier<PIPE_V>();

        LocalTensor<inType> bLocal = inQueueB.template DeQue<inType>();
        LocalTensor<float> tmpBLocal = sharedTempBuf.Get<float>();
        if constexpr (sizeof(inType) == sizeof(float)) {
            DataCopy(tmpBLocal, bLocal, tileLength);
        } else {
            Cast(tmpBLocal, bLocal, RoundMode::CAST_NONE, tileLength);
        }

        inQueueB.template FreeTensor(bLocal);
        Mul(tmpYLocal, tmpYLocal, tmpBLocal, tileLength);
        PipeBarrier<PIPE_V>();

        uint32_t index = 0;
        uint32_t smoothOffset = 0;
        uint32_t realRowNum = 0;
        int32_t groupValue = tilingData_.rowLen;
        if (tilingData_.hasGroup == 1) {
            groupValue = groupLocal.GetValue(smoothIndex);
        }
        LocalTensor<float> smoothLocal = smoothQueue.DeQue<float>();

        LocalTensor<float> tempFp32 = tempYUnit.Get<float>();
        LocalTensor<float> tmpLocal = sharedTempBuf.Get<float>(sizeHalfLen);
        LocalTensor<int32_t> tempInt32 = sharedTempBuf.Get<int32_t>(sizeHalfLen);
        auto tempHalf = tempFp32.ReinterpretCast<half>();

        LocalTensor<int8_t> outLocal = outQueueY.AllocTensor<int8_t>();
        for (int32_t i = 0; i < basicRowLenCal; i++) {
            index = i * sizeHalfLen;
            DataCopy(tempFp32, tmpYLocal[index], sizeHalfLen);

            realRowNum = offsetRow + i + 1;
            if (groupValue < realRowNum && smoothIndex != SMOOTH_INDEX_UPBOUND) {
                smoothIndex = GetSmoothIndex(realRowNum, groupValue, smoothIndex + 1);
                if (smoothIndex != SMOOTH_INDEX_UPBOUND) {
                    smoothQueue.FreeTensor(smoothLocal);
                    smoothOffset = smoothIndex * basicColLen;
                    SmoothCopyIn(smoothOffset);
                    smoothLocal = smoothQueue.DeQue<float>();
                }
            }

            if (smoothIndex != SMOOTH_INDEX_UPBOUND) {
                Mul(tempFp32, tempFp32, smoothLocal, basicColLen);
                PipeBarrier<PIPE_V>();
            }

            Abs(tmpLocal, tempFp32, basicColLen);
            PipeBarrier<PIPE_V>();
            ReduceMax(tmpLocal, tmpLocal, tmpLocal, basicColLen, false);
            PipeBarrier<PIPE_V>();

            // Compute quantization scale: quant_scale = 127.0 / max_abs
            Div(tmpLocal, constScale, tmpLocal, MAX_VALUE_NUM);
            PipeBarrier<PIPE_V>();

            float quantScale = tmpLocal.GetValue(0);

            // Store dequantization factor (inverse): dequant_scale = max_abs / 127.0 = 1.0 / quant_scale
            float dequantScale = (quantScale != 0.0f) ? (1.0f / quantScale) : 0.0f;
            scaleLocal.SetValue(i, dequantScale);

            // Quantize: y = x * quant_scale
            Muls(tempFp32, tempFp32, quantScale, basicColLen);
            PipeBarrier<PIPE_V>();

            CastQuantOut(tempFp32, tempInt32, tempHalf, outLocal, i);
        }
        smoothQueue.FreeTensor(smoothLocal);
        outQueueY.template EnQue<int8_t>(outLocal);
        scaleQueue.EnQue<float>(scaleLocal);
    }

private:
    GlobalTensor<inType> xGm;
};
}  // namespace SwiGluDynamicQuantOpt
#endif  // SWI_GLU_DYNAMIC_QUANT_H
