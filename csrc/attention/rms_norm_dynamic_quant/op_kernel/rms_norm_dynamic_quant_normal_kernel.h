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
 * \file add_rms_norm_dynamic_quant_normal_kernel.h
 * \brief
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_

#include "rms_norm_dynamic_quant_base.h"

template <typename T, typename T_Y, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddRmsNormDynamicQuantNormal : public KernelAddRmsNormDynamicQuantBase<T, T_Y, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelAddRmsNormDynamicQuantNormal(TPipe* pipe)
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
        this->numRowsAligned = (this->rowStep + ELEM_PER_BLK_FP32 - 1) / ELEM_PER_BLK_FP32 * ELEM_PER_BLK_FP32;
        this->ubAligned = static_cast<uint32_t>((this->numLastDimAligned - this->numLastDim) / ELEM_PER_BLK_FP16);
        /*
          UB = 3 * this->rowStep * alignedCol * sizeof(T)
              + 2 * this->rowStep * alignedCol * sizeof(float)
              + Count(gamma,beta,bias) * alignedCol * sizeof(T)
              + 512Bytes(256 + reduceOut)
        */
        Ppipe->InitBuffer(inRowsQue, BUFFER_NUM, 2 * this->rowStep * this->numLastDimAligned * sizeof(T));  // 2 * D * 2
        Ppipe->InitBuffer(outRowsQue, BUFFER_NUM, 2 * this->rowStep * this->numLastDimAligned * sizeof(T)); // D * 2
        Ppipe->InitBuffer(xBufFp32, this->rowStep * this->numLastDimAligned * sizeof(float));               // D * 4
        Ppipe->InitBuffer(yBufFp32, this->rowStep * this->numLastDimAligned * sizeof(float));               // D * 4
        Ppipe->InitBuffer(weightBuf01, this->numLastDimAligned * sizeof(T));                                // D * 2
        Ppipe->InitBuffer(weightBuf02, this->numLastDimAligned * sizeof(T));                                // D * 2
        Ppipe->InitBuffer(weightBuf03, this->numLastDimAligned * sizeof(T));                                // D * 2
        if (this->betaFlag == 1) {
            Ppipe->InitBuffer(weightBuf04, this->numLastDimAligned * sizeof(T));
        }
        // 2 dynamic quant operator required 2 scale buffer.
        Ppipe->InitBuffer(scalesBuf, 2 * this->numRowsAligned * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t rowMoveCnt = CEIL_DIV(this->rowWork, this->rowStep);
        CopyInWeights();

        LocalTensor<T> gammaLocal = weightBuf01.template Get<T>();

        int32_t gmOffset = 0;
        int32_t gmOffsetScale = 0;
        int32_t elementCount = this->numLastDimAligned * this->rowStep;

        for (int32_t rowIdx = 0; rowIdx < rowMoveCnt - 1; ++rowIdx) {
            CopyInX(gmOffset, this->rowStep, elementCount);
            ComputeRmsNorm(this->rowStep, elementCount, gammaLocal);
            ComputeDynamicQuant(this->rowStep, elementCount);
            CopyOut(gmOffset, gmOffsetScale, this->rowStep);
            gmOffset += this->rowStep * this->numLastDim;
            gmOffsetScale += this->rowStep;
        }
        {
            elementCount = this->numLastDimAligned * this->rowTail_;
            int32_t rowIdx = rowMoveCnt - 1;
            CopyInX(gmOffset, this->rowTail_, elementCount);
            ComputeRmsNorm(this->rowTail_, elementCount, gammaLocal);
            ComputeDynamicQuant(this->rowTail_, elementCount);
            CopyOut(gmOffset, gmOffsetScale, this->rowTail_);
        }
    }

private:
    __aicore__ inline void CopyInX(int32_t gmOffset, int32_t rowCount, int32_t elementCount)
    {
        LocalTensor<T> xLocalIn = inRowsQue.template AllocTensor<T>();
        DataCopyExStride(xLocalIn, this->xGm[gmOffset], this->numLastDim, rowCount, this->ubAligned);
        inRowsQue.EnQue(xLocalIn);
    }

    __aicore__ inline void CopyOutY(int32_t gmOffset, int32_t rowCount, int32_t elementCount)
    {
        PipeBarrier<PIPE_ALL>();
        LocalTensor<float> yLocal = xBufFp32.Get<float>();
        LocalTensor<T> yOut = yBufFp32.Get<T>();
        PipeBarrier<PIPE_ALL>();
        if constexpr (is_same<T, half>::value) {
            Cast(yOut, yLocal, RoundMode::CAST_NONE, elementCount);
        } else { // BF16
            Cast(yOut, yLocal, RoundMode::CAST_RINT, elementCount);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExStride(this->xGm[gmOffset], yOut, this->numLastDim, rowCount, this->ubAligned);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInWeights()
    {
        LocalTensor<T> gammaLocal = weightBuf01.template Get<T>();
        DataCopyEx(gammaLocal, this->gammaGm, this->numLastDim);
        if ((this->isOld && this->smooth1Exist) || this->newSingleFirst) {
            LocalTensor<T> smooth1Local = weightBuf02.template Get<T>();
            DataCopyEx(smooth1Local, this->smooth1Gm, this->numLastDim);
        }
        if (this->oldDouble || this->newSingleSecond) {
            LocalTensor<T> smooth2Local = weightBuf03.template Get<T>();
            DataCopyEx(smooth2Local, this->smooth2Gm, this->numLastDim);
        }
        if (this->betaFlag == 1) {
            LocalTensor<T> betaLocal = weightBuf04.template Get<T>();
            DataCopyEx(betaLocal, this->betaGm, this->numLastDim);
        }
    }

    __aicore__ inline void ComputeRmsNorm(int32_t nums, int32_t elementCount, LocalTensor<T>& gammaLocal)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>(); // xLocalFp32 <-- x
        LocalTensor<T> xInputLocal = inRowsQue.template DeQue<T>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        Cast(xLocalFp32, xInputLocal, RoundMode::CAST_NONE, elementCount);

        Mul(yLocalFp32, xLocalFp32, xLocalFp32, elementCount); // yLocalFp32 <- x ** 2
        PipeBarrier<PIPE_V>();

        // reduce#1 for mean
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * this->numLastDimAligned;
            float squareSumTemp =
                ReduceSumHalfInterval(yLocalFp32[roundOffset], this->numLastDim); // aveLocalTemp <-- E(x**2)
            float rstdLocalTemp = 1 / sqrt(squareSumTemp * this->aveNum + this->eps);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            Muls(
                xLocalFp32[roundOffset], xLocalFp32[roundOffset], rstdLocalTemp,
                this->numLastDim); // xLocalFp32 <- x * rstd
        }
        PipeBarrier<PIPE_V>();

        Cast(yLocalFp32, gammaLocal, RoundMode::CAST_NONE, this->numLastDim); // yLocalFp32 <- gamma
        PipeBarrier<PIPE_V>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * this->numLastDimAligned;
            Mul(xLocalFp32[roundOffset], xLocalFp32[roundOffset], yLocalFp32,
                this->numLastDim); // xLocalFp32 <- x * rstd * gamma
            PipeBarrier<PIPE_V>();
        }

        if (this->betaFlag == 1) {
            LocalTensor<T> betaLocal = weightBuf04.template Get<T>();
            Cast(yLocalFp32, betaLocal, RoundMode::CAST_NONE, this->numLastDim); // yLocalFp32 <- gamma
            for (int32_t rid = 0; rid < nums; ++rid) {
                auto roundOffset = rid * this->numLastDimAligned;
                PipeBarrier<PIPE_V>();
                Add(xLocalFp32[roundOffset], xLocalFp32[roundOffset], yLocalFp32, this->numLastDim);
                PipeBarrier<PIPE_V>();
            }
        }
        inRowsQue.FreeTensor(xInputLocal);
    }

    __aicore__ inline void ComputeDynamicQuant(int32_t nums, int32_t elementCount)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>(); // xLocalFp32 <-- y
        LocalTensor<float> scaleLocal = scalesBuf.Get<float>();
        LocalTensor<float> zLocalFp32 = outRowsQue.template AllocTensor<float>();
        LocalTensor<T_Y> outQuant01 = zLocalFp32.ReinterpretCast<T_Y>();
        doQuant1withFlag(scaleLocal, xLocalFp32, outQuant01, nums, elementCount);
        doQuant2withFlag(scaleLocal, xLocalFp32, outQuant01, nums, elementCount);
        outRowsQue.EnQue(zLocalFp32);
    }

    __aicore__ inline void doQuant1withFlag(
        LocalTensor<float> scaleLocal, LocalTensor<float> xLocalFp32, LocalTensor<T_Y> outQuant01, int32_t nums,
        int32_t elementCount)
    {
        if (this->outQuant1Flag == 0 && !this->isOld) {
            return;
        }
        LocalTensor<float> tmpFp32 = inRowsQue.template AllocTensor<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<float> scale1Local = scaleLocal[0];
        if (this->smooth1Exist) {
            // compute smooth1
            LocalTensor<T> smooth1Local = weightBuf02.Get<T>();
            LocalTensor<float> smooth1Fp32 = yLocalFp32[(nums - 1) * this->numLastDimAligned];
            Cast(smooth1Fp32, smooth1Local, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            for (int32_t rid = 0; rid < nums; ++rid) {
                Mul(yLocalFp32[rid * this->numLastDimAligned], xLocalFp32[rid * this->numLastDimAligned], smooth1Fp32,
                    this->numLastDim); // yLocalFp32 <-- y * smooth1
            }
            PipeBarrier<PIPE_V>();
        } else {
            for (int32_t rid = 0; rid < nums; ++rid) {
                Muls(
                    yLocalFp32[rid * this->numLastDimAligned], xLocalFp32[rid * this->numLastDimAligned], (float)(1.0),
                    this->numLastDim); // yLocalFp32 <-- y * 1
            }
            PipeBarrier<PIPE_V>();
        }
        ScaleTensor(yLocalFp32, tmpFp32, scale1Local, elementCount, nums);
        PipeBarrier<PIPE_V>();
        Cast(yLocalFp32.ReinterpretCast<int32_t>(), yLocalFp32, RoundMode::CAST_RINT, elementCount);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(
            yLocalFp32.ReinterpretCast<half>(), yLocalFp32.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,
            elementCount);
        PipeBarrier<PIPE_V>();
        Cast(outQuant01, yLocalFp32.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, elementCount);
        PipeBarrier<PIPE_V>();
        inRowsQue.FreeTensor(tmpFp32);
    }

    __aicore__ inline void doQuant2withFlag(
        LocalTensor<float> scaleLocal, LocalTensor<float> xLocalFp32, LocalTensor<T_Y> outQuant01, int32_t nums,
        int32_t elementCount)
    {
        if (this->outQuant2Flag == 0 && !this->oldDouble) {
            return;
        }
        LocalTensor<float> tmpFp32 = inRowsQue.template AllocTensor<float>();
        LocalTensor<float> scale2Local = scaleLocal[this->numRowsAligned];
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<T_Y> outQuant02 = outQuant01[elementCount];
        if (this->smooth2Exist) {
            LocalTensor<T> smooth2Local = weightBuf03.Get<T>();
            Cast(tmpFp32, smooth2Local, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            for (int32_t rid = 0; rid < nums; ++rid) {
                Mul(xLocalFp32[rid * this->numLastDimAligned], xLocalFp32[rid * this->numLastDimAligned], tmpFp32,
                    this->numLastDim); // yLocalFp32 <-- y * smooth2
            }
            PipeBarrier<PIPE_V>();
        } else {
            for (int32_t rid = 0; rid < nums; ++rid) {
                Muls(
                    xLocalFp32[rid * this->numLastDimAligned], xLocalFp32[rid * this->numLastDimAligned], (float)(1.0),
                    this->numLastDim); // yLocalFp32 <-- y * 1
            }
            PipeBarrier<PIPE_V>();
        }
        ScaleTensor(xLocalFp32, tmpFp32, scale2Local, elementCount, nums);
        PipeBarrier<PIPE_V>();
        Cast(xLocalFp32.ReinterpretCast<int32_t>(), xLocalFp32, RoundMode::CAST_RINT, elementCount);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(
            xLocalFp32.ReinterpretCast<half>(), xLocalFp32.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,
            elementCount);
        PipeBarrier<PIPE_V>();
        Cast(outQuant02, xLocalFp32.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, elementCount);
        PipeBarrier<PIPE_V>();
        inRowsQue.FreeTensor(tmpFp32);
    }

    __aicore__ inline void CopyOut(int32_t gmOffset, int32_t gmOffsetScale, int32_t rowCount)
    {
        LocalTensor<T_Y> outY12 = outRowsQue.template DeQue<T_Y>();
        LocalTensor<float> scaleLocal = scalesBuf.Get<float>();
        if (this->isOld || (this->outQuant1Flag == 1)) {
            LocalTensor<T_Y> outQuant01 = outY12[0];
            LocalTensor<float> scale1Local = scaleLocal[0];
            DataCopyEx(this->y1Gm[gmOffset], outQuant01, this->numLastDim, rowCount);
            DataCopyEx(this->outScale1Gm[gmOffsetScale], scale1Local, rowCount);
        }
        if (this->oldDouble || (this->outQuant2Flag == 1)) {
            LocalTensor<T_Y> outQuant02 = outY12[rowCount * this->numLastDimAligned];
            LocalTensor<float> scale2Local = scaleLocal[this->numRowsAligned];
            DataCopyEx(this->y2Gm[gmOffset], outQuant02, this->numLastDim, rowCount);
            DataCopyEx(this->outScale2Gm[gmOffsetScale], scale2Local, rowCount);
        }
        outRowsQue.FreeTensor(outY12);
    }

    __aicore__ inline void ScaleTensor(
        LocalTensor<float>& srcTensor, LocalTensor<float>& tmpTensor, LocalTensor<float>& scaleTensor, int32_t size,
        int32_t nums)
    {
        float maxTemp;
        float scaleTemp;
        event_t eventVS;
        event_t eventSV;
        Abs(tmpTensor, srcTensor, size); // tmpLocal <-- |y * smooth1|
        PipeBarrier<PIPE_V>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            ReduceMaxInplace(tmpTensor[rid * this->numLastDimAligned], this->numLastDim);
            eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            maxTemp = tmpTensor[rid * this->numLastDimAligned].GetValue(0); // Reduce
            scaleTemp = this->quantMaxVal / maxTemp;
            scaleTensor.SetValue(rid, 1 / scaleTemp);
            eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            auto srcSlice = srcTensor[rid * this->numLastDimAligned];
            Muls(srcSlice, srcSlice, scaleTemp, this->numLastDim);
        }
    }

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inRowsQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outRowsQue;

    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;

    TBuf<TPosition::VECCALC> weightBuf01;
    TBuf<TPosition::VECCALC> weightBuf02;
    TBuf<TPosition::VECCALC> weightBuf03;
    TBuf<TPosition::VECCALC> weightBuf04;
    TBuf<TPosition::VECCALC> scalesBuf;

    uint32_t numRowsAligned;
    uint32_t ubAligned;
};

#endif // __ADD_RMS_NORM_DYNAMIC_QUANT_NORMAL_KERNEL_H_
