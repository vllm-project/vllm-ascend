/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_dynamic_quant_single_row_kernel.h
 * \brief
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_SINGLE_ROW_KERNEL_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_SINGLE_ROW_KERNEL_H_

#include "rms_norm_dynamic_quant_base.h"

template <typename T, typename T_Y, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddRmsNormDynamicQuantSingleRow : public KernelAddRmsNormDynamicQuantBase<T, T_Y, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelAddRmsNormDynamicQuantSingleRow(TPipe* pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR smooth1, GM_ADDR smooth2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2,
        GM_ADDR outScale1, GM_ADDR outScale2, GM_ADDR workspace, const RmsNormDynamicQuantTilingData* tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x, gamma, smooth1, smooth2, beta);
        this->InitOutGlobalTensors(y1, y2, outScale1, outScale2);

        /*
          UB = 3 * alignedCol * sizeof(T)
              + 2 * alignedCol * sizeof(float)
              + Count(bias) * alignedCol * sizeof(T)
              + 512Btyes(256 + reduceOut)
        */
        Ppipe->InitBuffer(inRowsQue, BUFFER_NUM, 2 * this->numLastDimAligned * sizeof(T)); // 2 * D * 2
        Ppipe->InitBuffer(yQue, BUFFER_NUM, this->numLastDimAligned * sizeof(T));          // D * 2

        Ppipe->InitBuffer(xBufFp32, this->numLastDimAligned * sizeof(float)); // D * 4
        Ppipe->InitBuffer(yBufFp32, this->numLastDimAligned * sizeof(float)); // D * 4
        Ppipe->InitBuffer(smoothBuf, this->numLastDimAligned * sizeof(T));    // D * 2

        // 2 dynamic quant operator required 2 scale buffer.
        Ppipe->InitBuffer(scalesQue, BUFFER_NUM, 2 * ROW_FACTOR * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if ((this->isOld && this->smooth1Exist) || this->newSingleFirst) {
            LocalTensor<T> smooth1Local = smoothBuf.template Get<T>();
            DataCopyEx(smooth1Local, this->smooth1Gm, this->numLastDim);
        }

        int32_t outLoopCount = this->rowWork / ROW_FACTOR;
        int32_t outLoopTail = this->rowWork % ROW_FACTOR;
        uint32_t gmOffset = 0;
        uint32_t gmOffsetReduce = 0;

        LocalTensor<float> scalesLocalOut;

        for (int32_t loopIdx = 0; loopIdx < outLoopCount; ++loopIdx) {
            scalesLocalOut = scalesQue.template AllocTensor<float>();
            for (int32_t innerIdx = 0; innerIdx < ROW_FACTOR; ++innerIdx) {
                CopyInXAndGamma(gmOffset);
                ComputeRmsNorm(gmOffset);
                CopyInSmooth();
                ComputeDynamicQuant(innerIdx, scalesLocalOut, gmOffset);
                CopyOut(gmOffset);
                gmOffset += this->numLastDim;
            }
            scalesQue.EnQue(scalesLocalOut);
            CopyOutScale(gmOffsetReduce, ROW_FACTOR);
            gmOffsetReduce += ROW_FACTOR;
        }
        {
            scalesLocalOut = scalesQue.template AllocTensor<float>();
            for (int32_t innerIdx = 0; innerIdx < outLoopTail; ++innerIdx) {
                CopyInXAndGamma(gmOffset);
                ComputeRmsNorm(gmOffset);
                CopyInSmooth();
                ComputeDynamicQuant(innerIdx, scalesLocalOut, gmOffset);
                CopyOut(gmOffset);
                gmOffset += this->numLastDim;
            }
            scalesQue.EnQue(scalesLocalOut);
            CopyOutScale(gmOffsetReduce, outLoopTail);
        }
    }

private:
    __aicore__ inline void ComputeRmsNorm(int32_t gmOffset)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<T> iputLocal = inRowsQue.template DeQue<T>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<T> yLocalB16 = yBufFp32.Get<T>();

        Cast(xLocalFp32, iputLocal, RoundMode::CAST_NONE, this->numLastDim);

        Mul(yLocalFp32, xLocalFp32, xLocalFp32, this->numLastDim); // yLocalFp32 <- x ** 2
        PipeBarrier<PIPE_V>();

        float squareSumTemp = ReduceSumHalfInterval(yLocalFp32, this->numLastDim);
        float rstdLocalTemp = 1 / sqrt(squareSumTemp * this->aveNum + this->eps);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(xLocalFp32, xLocalFp32, rstdLocalTemp, this->numLastDim); // xLocalFp32 <- x * rstd
        PipeBarrier<PIPE_V>();
        LocalTensor<float> gammaLocal = xLocalFp32[this->numLastDimAligned];

        inRowsQue.FreeTensor(iputLocal);
        Mul(xLocalFp32, xLocalFp32, gammaLocal, this->numLastDim); // xLocalFp32 <- x * rstd * gamma
        PipeBarrier<PIPE_V>();
        if (this->betaFlag == 1) {
            CopyInBeta();
            LocalTensor<T> betaLocal = inRowsQue.template DeQue<T>();
            Cast(yLocalFp32, betaLocal, RoundMode::CAST_NONE, this->numLastDim); // yLocalB16 <- Cast(beta)
            PipeBarrier<PIPE_V>();
            Add(xLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim);
            PipeBarrier<PIPE_V>();
            inRowsQue.FreeTensor(betaLocal);
        }
    }

    __aicore__ inline void ComputeDynamicQuant(int32_t idx, LocalTensor<float>& scalesLocalOut, int32_t gmOffset)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<T_Y> yLocal = yQue.template AllocTensor<T_Y>();

        LocalTensor<T> smooth1Local = smoothBuf.template Get<T>();
        LocalTensor<T> smooth2Local = inRowsQue.template DeQue<T>();
        LocalTensor<float> tmpTensor = smooth2Local.template ReinterpretCast<float>();
        auto y1Local = yLocal[0];
        auto y2Local = yLocal[this->numLastDimAligned];

        if ((this->outQuant2Flag == 1) || this->oldDouble) {
            if (this->smooth2Exist) {
                Cast(yLocalFp32, smooth2Local, RoundMode::CAST_NONE, this->numLastDim); // yLocalFp32 <-- smooth2
                PipeBarrier<PIPE_V>();
                Mul(yLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim); // yLocalFp32 <-- y * smooth2
                PipeBarrier<PIPE_V>();
            } else {
                Muls(yLocalFp32, xLocalFp32, (float)1.0, this->numLastDim); // yLocalFp32 <-- y * 1
                PipeBarrier<PIPE_V>();
            }
            ScaleTensor(
                yLocalFp32, tmpTensor, scalesLocalOut,
                idx + ROW_FACTOR); // yLocalFp32 <-- yLocalFp32 / max(abs(yLocalFp32))
            PipeBarrier<PIPE_V>();
            inRowsQue.FreeTensor(tmpTensor);
            RoundFloat2IntQuant<T_Y>(y2Local, yLocalFp32, this->numLastDim);
        }

        if ((this->outQuant1Flag == 1) || this->isOld) {
            if (this->smooth1Exist) {
                Cast(yLocalFp32, smooth1Local, RoundMode::CAST_NONE, this->numLastDim); // yLocalFp32 <-- smooth1
                PipeBarrier<PIPE_V>();
                Mul(yLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim); // yLocalFp32 <-- y * smooth1
                PipeBarrier<PIPE_V>();
            } else {
                Muls(yLocalFp32, xLocalFp32, (float)1.0, this->numLastDim); // yLocalFp32 <-- y * smooth1
                PipeBarrier<PIPE_V>();
            }
            ScaleTensor(
                yLocalFp32, xLocalFp32, scalesLocalOut, idx); // yLocalFp32 <-- yLocalFp32 / max(abs(yLocalFp32))
            PipeBarrier<PIPE_V>();
            RoundFloat2IntQuant<T_Y>(y1Local, yLocalFp32, this->numLastDim);
        }
        PipeBarrier<PIPE_V>();
        yQue.EnQue(yLocal);
    }

    // srcTensor <- srcTensor / max(abs(srcTensor))
    __aicore__ inline void ScaleTensor(
        LocalTensor<float>& srcTensor, LocalTensor<float>& tmpTensor, LocalTensor<float>& scaleTensor, int32_t idx)
    {
        Abs(tmpTensor, srcTensor, this->numLastDim); // tmpLocal <-- |y * smooth|
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(tmpTensor, this->numLastDim);
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float maxTemp = tmpTensor.GetValue(0);
        float scaleTemp = this->quantMaxVal / maxTemp;
        scaleTensor.SetValue(idx, 1 / scaleTemp);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(srcTensor, srcTensor, scaleTemp, this->numLastDim);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOut(int32_t gmOffset)
    {
        LocalTensor<T_Y> res12 = yQue.template DeQue<T_Y>();
        auto res1 = res12[0];
        auto res2 = res12[this->numLastDimAligned];
        if (this->isOld || (this->outQuant1Flag == 1)) {
            DataCopyEx(this->y1Gm[gmOffset], res1, this->numLastDim);
        }

        if (this->oldDouble || (this->outQuant2Flag == 1)) {
            DataCopyEx(this->y2Gm[gmOffset], res2, this->numLastDim);
        }
        yQue.FreeTensor(res12);
    }

    __aicore__ inline void CopyOutScale(int32_t gmOffset, int32_t copyInNums)
    {
        LocalTensor<float> outScalesLocal = scalesQue.template DeQue<float>();
        LocalTensor<float> outScales1Local = outScalesLocal[0];
        LocalTensor<float> outScales2Local = outScalesLocal[ROW_FACTOR];
        if (this->isOld || (this->outQuant1Flag == 1)) {
            DataCopyEx(this->outScale1Gm[gmOffset], outScales1Local, copyInNums);
        }
        if (this->oldDouble || (this->outQuant2Flag == 1)) {
            DataCopyEx(this->outScale2Gm[gmOffset], outScales2Local, copyInNums);
        }
        scalesQue.FreeTensor(outScalesLocal);
    }

    __aicore__ inline void CopyInXAndGamma(int32_t gmOffset)
    {
        LocalTensor<T> xLocalIn = inRowsQue.template AllocTensor<T>();
        DataCopyEx(xLocalIn[0], this->xGm[gmOffset], this->numLastDim);
        DataCopyEx(xLocalIn[this->numLastDimAligned], this->gammaGm, this->numLastDim);
        inRowsQue.EnQue(xLocalIn);
    }

    __aicore__ inline void CopyInSmooth()
    {
        if (this->oldDouble || this->newSingleSecond) {
            LocalTensor<T> smoothCopyIn = inRowsQue.template AllocTensor<T>();
            DataCopyEx(smoothCopyIn[0], this->smooth2Gm, this->numLastDim);
            inRowsQue.EnQue(smoothCopyIn);
        }
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaCopyIn = inRowsQue.template AllocTensor<T>();
        DataCopyEx(gammaCopyIn[0], this->gammaGm, this->numLastDim);
        inRowsQue.EnQue(gammaCopyIn);
    }

    __aicore__ inline void CopyInBeta()
    {
        LocalTensor<T> betaCopyIn = inRowsQue.template AllocTensor<T>();
        DataCopyEx(betaCopyIn[0], this->betaGm, this->numLastDim);
        inRowsQue.EnQue(betaCopyIn);
    }

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inRowsQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scalesQue;

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;

    TBuf<TPosition::VECCALC> smoothBuf;
};

#endif // __ADD_RMS_NORM_DYNAMIC_QUANT_SINGLE_ROW_KERNEL_H_
