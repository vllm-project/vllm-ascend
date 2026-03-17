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
 * \file swi_glu_dynamic_quant_base.h
 * \brief
 */
#ifndef SWI_GLU_DYNAMIC_QUANT_BASE_H
#define SWI_GLU_DYNAMIC_QUANT_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace SwiGluDynamicQuantOpt {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t SPLIT_NUM = 2;
constexpr uint16_t ONE_BYTE_INT4_NUM_TWO = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t SWI_GLU_DQ_EIGHT = 8;
constexpr uint32_t SWI_GLU_DQ_THIRTY_TWO = 32;

constexpr float SWI_GLU_DQ_INT8_SYM_SCALE = 127.0;
constexpr float SWI_GLU_DQ_INT4_SYM_SCALE = 7.0;
constexpr uint32_t MAX_VALUE_NUM = 8;
constexpr uint32_t SMOOTH_INDEX_UPBOUND = 65536;

struct XxGluSingleTileOffsetParam {
    uint64_t splitVecGmOffset1;
    uint64_t splitVecGmOffset2;
    uint64_t tmpVecGmOffset;
};

enum class QuantType : uint8_t {
    STATIC_PER_TENSOR = 0,
    STATIC_PER_CHANNEL
};

class SwiGluDynamicQuantBase {
public:
    __aicore__ inline SwiGluDynamicQuantBase()
    {}

    __aicore__ inline void ParseTilingData(const SwiGluDynamicQuantTilingData *tilingData)
    {
        tilingData_.groupLen = tilingData->groupLen;
        tilingData_.rowLen = tilingData->rowLen;
        tilingData_.colLen = tilingData->colLen;
        tilingData_.rowLenPerHeadCore = tilingData->rowLenPerHeadCore;
        tilingData_.rowLenPerTailCore = tilingData->rowLenPerTailCore;
        tilingData_.basicRowLenHeadCore = tilingData->basicRowLenHeadCore;
        tilingData_.basicRowLenTailCore = tilingData->basicRowLenTailCore;
        tilingData_.basicColLen = tilingData->basicColLen;
        tilingData_.headCoreNum = tilingData->headCoreNum;
        tilingData_.realCoreNum = tilingData->realCoreNum;
        tilingData_.activateLeft = tilingData->activateLeft;
        tilingData_.groupListType = tilingData->groupListType;
        tilingData_.hasGroup = tilingData->hasGroup;
        tilingData_.dstType = tilingData->dstType;
    }

    __aicore__ inline void InitBaseBuffer()
    {
        pPipe->InitBuffer(tmpConstBuffer, MAX_VALUE_NUM * sizeof(float));
    }

    __aicore__ inline void DuplicateConst()
    {
        constScale = tmpConstBuffer.Get<float>();
        Duplicate<float>(constScale, SWI_GLU_DQ_INT8_SYM_SCALE, MAX_VALUE_NUM);
    }

    template <typename T>
    __aicore__ inline T CeilDiv(T x, T y)
    {
        return y == 0 ? 0 : (x + y - 1) / y;
    }

    __aicore__ inline float GetMax(float a, float b)
    {
        return a > b ? a : b;
    }

    template<typename T>
    __aicore__ inline T AlignUp(T num, T div)
    {
        return (div == 0) ? 0 : (num + div - 1) / div * div;
    }

protected:
    __aicore__ inline void InitParams(int64_t sizeOfInType, int64_t sizeOfOutType)
    {
        mergedColLen = SPLIT_NUM * tilingData_.colLen;
        colLen = tilingData_.colLen;
        basicColLen = tilingData_.basicColLen;

        coreIdx = static_cast<uint32_t>(GetBlockIdx());
        headCoreNum = tilingData_.headCoreNum;

        if (coreIdx < headCoreNum) {
            rowLenPerCore = tilingData_.rowLenPerHeadCore;
            basicRowLen = tilingData_.basicRowLenHeadCore;
            rowLoop = CeilDiv(rowLenPerCore, basicRowLen);
            baseRow = coreIdx * rowLenPerCore;
        } else if (coreIdx >= headCoreNum && coreIdx < tilingData_.realCoreNum) {
            rowLenPerCore = tilingData_.rowLenPerTailCore;
            basicRowLen = tilingData_.basicRowLenTailCore;
            rowLoop = CeilDiv(rowLenPerCore, basicRowLen);
            baseRow = headCoreNum * tilingData_.rowLenPerHeadCore + (coreIdx - headCoreNum) * rowLenPerCore;
        }

        outAlignLen = AlignUp(basicColLen, SWI_GLU_DQ_THIRTY_TWO);
        outAlignLen = (outAlignLen == 0 && sizeOfOutType != 0) ? (BLOCK_SIZE / sizeOfOutType) : outAlignLen;
        outLen = basicRowLen * outAlignLen;

        if (tilingData_.dstType == DT_INT4) {
            outAlignLen = AlignUp(CeilDiv(basicColLen, (uint32_t)ONE_BYTE_INT4_NUM_TWO), SWI_GLU_DQ_THIRTY_TWO);
        }

        alignedGroupLen = AlignUp(tilingData_.groupLen, SWI_GLU_DQ_EIGHT);

        uint32_t alignedNum = (sizeOfInType == 0) ? BLOCK_SIZE : (BLOCK_SIZE / sizeOfInType);
        sizeHalfLen = AlignUp(basicColLen, alignedNum);
        tileLength = basicRowLen * (sizeHalfLen == 0 ? (BLOCK_SIZE / sizeOfInType) : sizeHalfLen);
        rightPadding = sizeHalfLen - basicColLen;
        isPad = (rightPadding > 0);
        blockUnit = (isPad) ? 1 : BLOCK_SIZE;

        smoothSizeFloatLen = AlignUp(basicColLen, SWI_GLU_DQ_EIGHT);
        smoothRightPadding = smoothSizeFloatLen - tilingData_.basicColLen;
        smoothIsPad = (smoothRightPadding > 0);
    }

    __aicore__ inline void CopyOut(uint64_t splitCopyoutOffset, DataCopyParams &splitCopyoutParams,
         uint32_t ridx, uint32_t basicRowLenCal)
    {
        LocalTensor<int8_t> outLocal = outQueueY.DeQue<int8_t>();

#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        if (tilingData_.dstType == DT_INT4) {
            LocalTensor<int4b_t> outLocalInt4 = outLocal.template ReinterpretCast<int4b_t>();
            DataCopyPad(yGmInt4[splitCopyoutOffset], outLocalInt4, splitCopyoutParams);
        } else {
            DataCopyPad(yGm[splitCopyoutOffset], outLocal, splitCopyoutParams);
        }
#else
        DataCopyPad(yGm[splitCopyoutOffset], outLocal, splitCopyoutParams);
#endif

        outQueueY.FreeTensor(outLocal);

        LocalTensor<float> scaleLocal = scaleQueue.DeQue<float>();
        DataCopyParams copyParams1{ 1, (uint16_t)(basicRowLenCal * sizeof(float)), 0, 0 };

        DataCopyPad(scale_Gm[baseRow + basicRowLen * ridx], scaleLocal, copyParams1);
        scaleQueue.FreeTensor(scaleLocal);
    }

    __aicore__ inline void CastQuantOut(LocalTensor<float> &tempFp32, LocalTensor<int32_t> &tempInt32, LocalTensor<half> &tempHalf,
         LocalTensor<int8_t> &outLocal, int32_t i) {
        Cast(tempInt32, tempFp32, RoundMode::CAST_RINT, basicColLen);
        PipeBarrier<PIPE_V>();
        SetDeqScale(static_cast<half>(1.0));
        PipeBarrier<PIPE_V>();
        Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, basicColLen);
        PipeBarrier<PIPE_V>();
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        if (tilingData_.dstType == DT_INT4) {
            LocalTensor<int4b_t> outLocalInt4 = outLocal[i * outAlignLen].template ReinterpretCast<int4b_t>();
            Cast(outLocalInt4, tempHalf, RoundMode::CAST_NONE, basicColLen);
        } else {
            Cast(outLocal[i * outAlignLen], tempHalf, RoundMode::CAST_TRUNC, basicColLen);
        }
#else
        Cast(outLocal[i * outAlignLen], tempHalf, RoundMode::CAST_TRUNC, basicColLen);
#endif
    }

protected:
    TPipe *pPipe = nullptr;
    /* tiling data */
    SwiGluDynamicQuantTilingData tilingData_;

    /* variable */
    uint32_t rowLen;
    uint32_t colLen;
    uint32_t groupLen;
    uint32_t alignedGroupLen;
    uint32_t rowLenPerHeadCore;
    uint32_t rowLenPerTailCore;
    uint32_t basicRowLen;
    uint32_t rowLenPerCore;
    uint32_t basicRowLenHeadCore;
    uint32_t basicRowLenTailCore;
    uint32_t basicColLen;
    uint32_t headCoreNum;
    uint32_t realCoreNum;
    uint32_t outAlignLen;
    uint32_t sizeHalfLen;
    uint32_t smoothSizeFloatLen;
    uint32_t outLen;
    uint8_t rightPadding = 0;
    uint8_t smoothRightPadding = 0;
    bool isPad = false;
    bool smoothIsPad = false;
    uint16_t blockUnit;

    uint32_t coreIdx;
    uint32_t rowLoop = 1;
    uint32_t baseRow = 0;
    uint16_t basicRowLenCal;
    uint32_t mergedColLen;
    uint64_t tileLength;

    XxGluSingleTileOffsetParam offsetParam;

    TBuf<TPosition::VECCALC> tmpConstBuffer;
    TBuf<TPosition::VECCALC> tmpMaxBuffer;
    /* local memory */
    LocalTensor<float> constScale;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    // quant
    TQue<QuePosition::VECIN, BUFFER_NUM> groupQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scaleQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> smoothQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> offsetsQueue;

    TBuf<TPosition::VECCALC> sharedTempBuf;
    TBuf<TPosition::VECCALC> tempBufferY;
    TBuf<TPosition::VECCALC> fp32_buf_;
    TBuf<TPosition::VECCALC> tempYUnit;

    GlobalTensor<int8_t> yGm;
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    GlobalTensor<int4b_t> yGmInt4;
#endif
    GlobalTensor<float> smooth_scales_Gm;
    GlobalTensor<float> offsetsGm;
    GlobalTensor<float> scale_Gm;
    GlobalTensor<int32_t> group_index_Gm;

    LocalTensor<int32_t> groupLocal;
    LocalTensor<float> tmpYLocal;

    uint64_t splitCopyoutOffset;
};
}  // namespace SwiGluDynamicQuantOpt
#endif  // SWI_GLU_DYNAMIC_QUANT_BASE_H
