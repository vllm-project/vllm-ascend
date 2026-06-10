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
 * \file decode_update.h
 * \brief
 */

#ifndef ASCENDC_ATTENTION_UPDATE_V2_DECODE_UPDATE_H_
#define ASCENDC_ATTENTION_UPDATE_V2_DECODE_UPDATE_H_

#include <limits>
#include "kernel_operator.h"
#include "kernel_utils.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

namespace AttentionUpdateV2 {
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr uint32_t NUM0 = 0;
static constexpr uint32_t NUM1 = 1;
static constexpr uint32_t NUM2 = 2;
static constexpr uint32_t NUM7 = 7;
static constexpr uint32_t NUM8 = 8;
static constexpr uint32_t NUM16 = 16;
static constexpr uint32_t NUM32 = 32;
static constexpr uint32_t NUM64 = 64;
static constexpr uint32_t NUM255 = 255;
static constexpr uint32_t NUM256 = 256;
static constexpr float POS_INF = std::numeric_limits<float>::infinity();
static constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
static constexpr uint32_t MAX_UB_SIZE = 188 * 1024; //  double buffer, 每块94KB共188KB
static const uint16_t ALIGNED_TO_8 = 8;
static const int32_t ALIGNED_TO_2 = 2;
static const uint32_t SPLIT_TO_2 = 2;
static const uint32_t ELEM_PER_256B = NUM256 / sizeof(float);

template <typename lseType, typename outType>
class DecodeUpdate {
public:
    __aicore__ inline DecodeUpdate() {}
    __aicore__ inline void Init(GM_ADDR lse, GM_ADDR in, GM_ADDR out, GM_ADDR lesout, const DecodeUpdateTilingData *tdata)
    {
        this->hDim = tdata->hDim;
        this->sp = tdata->sp;
        this->totalLength = tdata->totalLength;
        this->updateType = tdata->updateType;

        // lse: [sp, bsh],  in: [sp, bsh, hd] — 直接绑定完整连续 tensor
        lseGm.SetGlobalBuffer((__gm__ lseType*)lse, sp * totalLength);
        inGm.SetGlobalBuffer((__gm__ outType*)in, sp * totalLength * hDim);
        if (GetBlockIdx() < tdata->formerNum) {
            blockLength = tdata->formerLength;
            this->gmStartOffset = GetBlockIdx() * tdata->formerLength;
        } else {
            blockLength = tdata->tailLength;
            this->gmStartOffset = tdata->formerNum * tdata->formerLength +
                                  (GetBlockIdx() - tdata->formerNum) * tdata->tailLength; // tail block
        }
        uint32_t spAligned = (NUM8 + 7) / NUM8 * NUM8;
        //  用94K的UB大小推算出 tileLength 最大能设置到多少
        uint32_t maxTileLength = (MAX_UB_SIZE - NUM8 * sizeof(uint32_t) - (ELEM_PER_256B - 1) * (sizeof(float) + sizeof(uint8_t))) /
                                (sizeof(float) * spAligned * BUFFER_NUM * (NUM2 * (NUM1 + hDim) + hDim / spAligned) +
                                hDim * spAligned * sizeof(float) * NUM2 +
                                sizeof(float) * BUFFER_NUM +
                                sizeof(float) * NUM2 +
                                sizeof(uint8_t) * spAligned);
        if constexpr (!std::is_same<outType, float>::value) {
            maxTileLength = (MAX_UB_SIZE - NUM8 * sizeof(uint32_t) - (ELEM_PER_256B - 1) * (sizeof(float) + sizeof(uint8_t))) /
                                    (sizeof(float) * spAligned * BUFFER_NUM * (NUM2 * (NUM1 + hDim) + hDim / spAligned) +
                                    hDim * spAligned * sizeof(float) * NUM2 +
                                    sizeof(float) * BUFFER_NUM +
                                    sizeof(float) * NUM2 + (sp + NUM1) * NUM16 * BUFFER_NUM +
                                    sizeof(uint8_t) * spAligned);
        }
        if (sp >= NUM8) {
            maxTileLength = maxTileLength / SPLIT_TO_2;
        }
        maxTileLength = maxTileLength < NUM1 ? NUM1 : maxTileLength;
        this->tileLength = maxTileLength < blockLength ? maxTileLength : blockLength;
        this->curLength = this->tileLength;
        this->lastLength = blockLength % tileLength;
        this->loopCount = blockLength / tileLength + (lastLength == NUM0 ? NUM0 : NUM1);
        this->tileLengthAlig = ((tileLength + NUM7) / NUM8) * NUM8;
        this->lastLengthAlig = ((lastLength + NUM7) / NUM8) * NUM8;
        // 设置全局变量的起始地址与总长度BLOCK_LENGTH; sp, B*s*hc in1 sp; B*s*hc, hd in2
        outGm.SetGlobalBuffer((__gm__ outType *)out, totalLength * hDim);
        lseoutGm.SetGlobalBuffer((__gm__ lseType *)lesout, totalLength);

        uint32_t inQueueLseLengthAlign = (tileLengthAlig * sp * sizeof(float) + NUM255) / NUM256 * NUM256;
        pipe.InitBuffer(inQueueLse, BUFFER_NUM, inQueueLseLengthAlign);
        if constexpr (std::is_same<outType, float>::value) {
            pipe.InitBuffer(inQueueIn, BUFFER_NUM, tileLength * hDim * sp * sizeof(float));
        } else {
            pipe.InitBuffer(inQueueIn, BUFFER_NUM, tileLength * hDim * sp * sizeof(float) + (sp + NUM1) * NUM16);
        }
        
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, tileLength * hDim * sizeof(float));
        pipe.InitBuffer(outQueueLse, BUFFER_NUM, tileLengthAlig * sizeof(float));

        pipe.InitBuffer(lsemaxBuffer, tileLengthAlig * sizeof(float));
        pipe.InitBuffer(lseexpsumBuffer, tileLengthAlig * sizeof(float));
        pipe.InitBuffer(lseexpBuffer, tileLengthAlig * sp * sizeof(float));
        if (hDim > NUM256 && sp >= NUM8) {
            pipe.InitBuffer(lseexpBroadcastBuffer, tileLengthAlig * sp * NUM64 * sizeof(float));
        } else {
            pipe.InitBuffer(lseexpBroadcastBuffer, tileLengthAlig * sp * hDim * sizeof(float));
        }
        pipe.InitBuffer(selMaskBuffer, inQueueLseLengthAlign * sizeof(uint8_t));
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->loopCount; i++) {
            if (lastLength && i == this->loopCount - 1) {
                curLength = lastLength;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //  按照逻辑队列初始化的buffer数量和每块大小，分配对应大小的内存
        LocalTensor<float> lseLocal = inQueueLse.AllocTensor<float>();
        LocalTensor<float> inLocal = inQueueIn.AllocTensor<float>();

        uint64_t inLocalFp16Offest = tileLength * hDim * sp;
        if ((inLocalFp16Offest * NUM2) % NUM32 == NUM16) {
            inLocalFp16Offest += NUM8;
        }
        LocalTensor<outType> inLocalFp16 = inLocal.template ReinterpretCast<outType>()[inLocalFp16Offest];

        if (curLength % ALIGNED_TO_8 == 0) {
            for (int32_t i = 0; i < sp; i++) {
                uint32_t lseOff = static_cast<uint32_t>(i) * totalLength + progress * tileLength + gmStartOffset;
                uint32_t inOff  = static_cast<uint32_t>(i) * totalLength * hDim + progress * tileLength * hDim + gmStartOffset * hDim;
                DataCopy(lseLocal[curLength * i], lseGm[lseOff], curLength);
                if constexpr (std::is_same<outType, float>::value) {
                    DataCopy(inLocal[curLength * hDim * i], inGm[inOff], curLength * hDim);
                } else {
                    DataCopy(inLocalFp16[curLength * hDim * i], inGm[inOff], curLength * hDim);
                }
            }
            if constexpr (!std::is_same<outType, float>::value) {
                inQueueIn.EnQue(inLocal);
                inLocal = inQueueIn.DeQue<float>();
                inLocalFp16 = inLocal.template ReinterpretCast<outType>()[inLocalFp16Offest];
                Cast(inLocal, inLocalFp16, RoundMode::CAST_NONE, curLength * hDim * sp);
            }
        } else {
            uint32_t curLengthAlig = ((curLength + NUM7) / NUM8) * NUM8;
            for (int32_t i = 0; i < sp; i++) {
                uint32_t lseOff = static_cast<uint32_t>(i) * totalLength + progress * tileLength + gmStartOffset;
                uint32_t inOff  = static_cast<uint32_t>(i) * totalLength * hDim + progress * tileLength * hDim + gmStartOffset * hDim;
                DataCopyPad(lseLocal[curLengthAlig * i], lseGm[lseOff],
                            {static_cast<uint16_t>(1), static_cast<uint32_t>(curLength * sizeof(lseType)), 0, 0, 0},
                            {true, 0, static_cast<uint8_t>(NUM8 - curLength % NUM8), 0});
                if constexpr (std::is_same<outType, float>::value) {
                    DataCopy(inLocal[curLength * hDim * i], inGm[inOff], curLength * hDim);
                } else {
                    uint64_t inLocalFp16OffestAlign32 = curLength * hDim * i * NUM2;
                    if (inLocalFp16OffestAlign32 % NUM32 == NUM16) {
                        inLocalFp16OffestAlign32 += NUM16;
                    }
                    DataCopyPad(inLocalFp16[inLocalFp16OffestAlign32 / NUM2], inGm[inOff],
                         {static_cast<uint16_t>(1), static_cast<uint32_t>(curLength * hDim * sizeof(outType)), 0, 0, 0},
                            {true, 0, static_cast<uint8_t>((NUM32 - curLength * hDim * sizeof(outType) % NUM32) / NUM2), 0});
                    PipeBarrier<PIPE_ALL>();
                    Cast(inLocal[curLength * hDim * i], inLocalFp16[inLocalFp16OffestAlign32 / NUM2], RoundMode::CAST_NONE, curLength * hDim);
                }
            }
        }

        inQueueLse.EnQue(lseLocal);
        inQueueIn.EnQue(inLocal);
    }

    __aicore__ inline void ProcessLseInfReplacement(LocalTensor<float>& lseLocal)
    {
        LocalTensor<uint8_t> selMask = selMaskBuffer.Get<uint8_t>();
        uint32_t alignedCount = ((tileLengthAlig * sp + ELEM_PER_256B - 1) / ELEM_PER_256B) * ELEM_PER_256B;

        CompareScalar(selMask, lseLocal, POS_INF, CMPMODE::EQ, alignedCount);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> negInfTensor = lseexpBuffer.Get<float>();
        Duplicate<float>(negInfTensor, NEG_INF, static_cast<int32_t>(tileLengthAlig * sp));
        PipeBarrier<PIPE_V>();

        Select<float, uint8_t>(lseLocal, selMask, negInfTensor, lseLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(tileLengthAlig * sp));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<float> lseLocal = inQueueLse.DeQue<float>();
        LocalTensor<float> inLocal = inQueueIn.DeQue<float>();
        LocalTensor<float> lseexpLocal = lseexpBuffer.Get<float>();
        LocalTensor<float> outLocal = outQueueOut.AllocTensor<float>();
        LocalTensor<float> lsemaxLocal = lsemaxBuffer.Get<float>();
        LocalTensor<float> lseexpsumLocal = lseexpsumBuffer.Get<float>();
        LocalTensor<float> lseexpBroadcastLocal = lseexpBroadcastBuffer.Get<float>();
        LocalTensor<float> lseoutLocal = outQueueLse.AllocTensor<float>();

        ProcessLseInfReplacement(lseLocal);

        // broad to 8
        uint32_t curLengthPad = ((curLength + NUM7) / NUM8) * NUM8;
        const uint32_t srcShape[2] = {sp * curLengthPad, 1};
        uint32_t dstShape[2] = {sp * curLengthPad, hDim};
        if (hDim > NUM256 && sp >= NUM8) {
            dstShape[1] = NUM64;
        }

        DataCopy(lsemaxLocal, lseLocal, curLengthPad);
        PipeBarrier<PIPE_V>();

        for (int32_t i = 1; i < sp; i++) {
            Max(lsemaxLocal, lsemaxLocal, lseLocal[i * curLengthPad], curLengthPad);
            PipeBarrier<PIPE_V>();
        }

        PipeBarrier<PIPE_V>();
        for (int32_t i = 0; i < sp; i++) {
            Sub(lseexpLocal[i * curLengthPad], lseLocal[i * curLengthPad], lsemaxLocal, curLengthPad);
            PipeBarrier<PIPE_V>();
        }

        Exp(lseexpLocal, lseexpLocal, curLengthPad * sp);
        PipeBarrier<PIPE_V>();

        DataCopy(lseexpsumLocal, lseexpLocal, curLengthPad);
        PipeBarrier<PIPE_V>();
        for (int32_t i = 1; i < sp; i++) {
            Add(lseexpsumLocal, lseexpsumLocal, lseexpLocal[i * curLengthPad], curLengthPad);
            PipeBarrier<PIPE_V>();
        }

        Log(lseexpsumLocal, lseexpsumLocal, curLengthPad);
        PipeBarrier<PIPE_V>();

        Add(lseoutLocal, lsemaxLocal, lseexpsumLocal, curLengthPad);
        PipeBarrier<PIPE_V>();

        for (int32_t i = 0; i < sp; i++) {
            Sub(lseexpLocal[i * curLengthPad], lseLocal[i * curLengthPad], lseoutLocal, curLengthPad);
            PipeBarrier<PIPE_V>();
        }

        Exp(lseexpLocal, lseexpLocal, curLengthPad * sp);
        PipeBarrier<PIPE_V>();
        BroadCast<float, ALIGNED_TO_2, 1>(lseexpBroadcastLocal, lseexpLocal, dstShape, srcShape);
        PipeBarrier<PIPE_V>();
        if (hDim > NUM256 && sp >= NUM8) {
            int64_t tmpTailLength = hDim % NUM64;
            int64_t tmpTailStart = hDim - tmpTailLength;
            for (int32_t i = 0; i < sp; i++) {
                for (int32_t k = 0; k < curLength; k++) {
                    Mul(inLocal[i * curLength * hDim + k * hDim], 
                                inLocal[i * curLength * hDim + k * hDim], 
                                lseexpBroadcastLocal[i * curLengthPad * NUM64 + k * NUM64],
                                NUM64, hDim / NUM64, {1, 1, 1, 8, 8, 0});
                    if (tmpTailLength > 0) {
                        Mul(inLocal[i * curLength * hDim + k * hDim + tmpTailStart],
                                inLocal[i * curLength * hDim + k * hDim + tmpTailStart], 
                                lseexpBroadcastLocal[i * curLengthPad * NUM64 + k * NUM64],
                                tmpTailLength, 1, {1, 1, 1, 8, 8, 0});
                    }
                }
            }
        } else {
            for (int32_t i = 0; i < sp; i++) {
                Mul(inLocal[i * curLength * hDim], inLocal[i * curLength * hDim], lseexpBroadcastLocal[i * curLengthPad * hDim],
                    curLength * hDim);
            }
        }
        PipeBarrier<PIPE_V>();

        DataCopy(outLocal, inLocal, curLength * hDim);
        PipeBarrier<PIPE_V>();
        for (int32_t i = 1; i < sp; i++) {
            Add(outLocal, outLocal, inLocal[i * curLength * hDim], curLength * hDim);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();

        outQueueLse.EnQue<float>(lseoutLocal);
        outQueueOut.EnQue<float>(outLocal);
        inQueueLse.FreeTensor(lseLocal);
        inQueueIn.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<float> outLocal = outQueueOut.DeQue<float>();
        if constexpr (std::is_same<outType, float>::value) { // fp32直接搬运
            DataCopy(outGm[gmStartOffset * hDim + progress * tileLength * hDim], outLocal, curLength * hDim);
        } else if constexpr (std::is_same<outType, bfloat16_t>::value){ // 先转fp32，再搬运
            LocalTensor<outType> outLocal16 = outLocal.template ReinterpretCast<outType>();
            Cast(outLocal16, outLocal, RoundMode::CAST_RINT, curLength * hDim);
            PipeBarrier<PIPE_V>();
            DataCopyPad(outGm[gmStartOffset * hDim + progress * tileLength * hDim], outLocal16,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(curLength * hDim * sizeof(outType)), 0, 0, 0});
        } else {
            LocalTensor<outType> outLocal16 = outLocal.template ReinterpretCast<outType>();
            Cast(outLocal16, outLocal, RoundMode::CAST_NONE, curLength * hDim);
            PipeBarrier<PIPE_V>();
            DataCopyPad(outGm[gmStartOffset * hDim + progress * tileLength * hDim], outLocal16,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(curLength * hDim * sizeof(outType)), 0, 0, 0});
        }
        outQueueOut.FreeTensor(outLocal);
        LocalTensor<float> lseoutLocal = outQueueLse.DeQue<float>();
        if (updateType == 1) {
            uint32_t lseoutGmStart = gmStartOffset + progress * tileLength; // lseout无hDim维度，直接按长度偏移
            uint32_t validBytes = static_cast<uint32_t>(curLength * sizeof(float));
            DataCopyExtParams copyParams{1, validBytes, 0, 0, 0};
            DataCopyPad(lseoutGm[lseoutGmStart], lseoutLocal, copyParams);
        }
        outQueueLse.FreeTensor(lseoutLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueLse;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIn;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOut;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueLse;
    TBuf<QuePosition::VECCALC> lseexpBuffer;
    TBuf<QuePosition::VECCALC> lsemaxBuffer;
    TBuf<QuePosition::VECCALC> lseexpsumBuffer;
    TBuf<QuePosition::VECCALC> lseexpBroadcastBuffer;
    TBuf<QuePosition::VECCALC> selMaskBuffer;

    GlobalTensor<lseType> lseGm;
    GlobalTensor<outType> inGm;
    GlobalTensor<outType> outGm;
    GlobalTensor<lseType> lseoutGm;

    uint32_t blockLength; //  单核上数据总长度
    uint16_t tileLength;  //  单核循环的非最后一轮数据长度
    uint16_t curLength; //  Tiling 循环数据长度为 curLength，最后一轮为 lastLength，内存申请使用 tileLength，数据量用
                        //  curLength
    uint16_t lastLength; //  单核循环的最后一轮数据长度
    uint32_t loopCount;  //  单核循环次数
    uint32_t hDim;       //  head dimension
    uint32_t sp;
    uint32_t tileLengthAlig;
    uint32_t lastLengthAlig;
    uint32_t totalLength;
    uint32_t gmStartOffset;
    uint32_t updateType;
};
} // namespace AttentionUpdateV2

#endif // ASCENDC_ATTENTION_UPDATE_V2_DECODE_UPDATE_H_