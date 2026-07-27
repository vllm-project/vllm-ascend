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
 * \file add_rms_norm_dynamic_quant_base.h
 * \brief
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_BASE_CLASS_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_BASE_CLASS_H_

#include "rms_norm_dynamic_quant_helper.h"

template <typename T, typename T_Y, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddRmsNormDynamicQuantBase {
public:
    __aicore__ inline KernelAddRmsNormDynamicQuantBase()
    {}

    __aicore__ inline void InitBaseParams(const RmsNormDynamicQuantTilingData* tiling)
    {
        this->numCore = tiling->useCore;
        this->numFirstDim = tiling->numFirstDim;
        this->numLastDim = tiling->numLastDim;
        this->numLastDimAligned = tiling->numLastDimAligned; // Quantize better be aligned to 32 elements

        this->firstDimPerCore = tiling->firstDimPerCore;
        this->firstDimPerCoreTail = tiling->firstDimPerCoreTail;
        this->firstDimPerLoop = tiling->firstDimPerLoop;

        this->lastDimSliceLen = tiling->lastDimSliceLen;
        this->lastDimLoopNum = tiling->lastDimLoopNum;
        this->lastDimSliceLenTail = tiling->lastDimSliceLenTail;
        this->betaFlag = tiling->betaFlag;
        this->eps = tiling->epsilon;
        this->aveNum = tiling->avgFactor;

        blockIdx_ = GetBlockIdx();
        if (blockIdx_ != this->numCore - 1) {
            this->rowWork = this->firstDimPerCore;
            this->rowStep = this->firstDimPerLoop;
        } else {
            this->rowWork = this->firstDimPerCoreTail;
            this->rowStep = TWO_NUMS_MIN(this->firstDimPerLoop, this->rowWork);
        }
        this->rowTail_ = (this->rowWork % this->rowStep == 0) ? this->rowStep : (this->rowWork % this->rowStep);
        this->gmOffset_ = this->firstDimPerCore * this->numLastDim;

        this->smooth1Exist = tiling->smoothNum1;
        // 2 dynamic quant operator required 2 scale buffer.
        this->smooth2Exist = tiling->smoothNum2;

        // dynamic quant max value
        if constexpr (IsSameType<T_Y, int8_t>::value) {
            this->quantMaxVal = DYNAMIC_QUANT_DIVIDEND;
        } else {
            this->quantMaxVal = DYNAMIC_QUANT_DIVIDEND_INT4;
        }
        this->outQuant1Flag = tiling->outQuant1Flag;
        this->outQuant2Flag = tiling->outQuant2Flag;

        this->isOld = (this->outQuant1Flag == -1) && (this->outQuant2Flag == -1);
        this->oldDouble = this->isOld && this->smooth1Exist && this->smooth2Exist;
        this->newSingleFirst = this->smooth1Exist && (this->outQuant1Flag == 1);
        this->newSingleSecond = this->smooth2Exist && (this->outQuant2Flag == 1);
    }

    __aicore__ inline void InitInGlobalTensors(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR smooth1, GM_ADDR smooth2, GM_ADDR beta)
    {
        xGm.SetGlobalBuffer((__gm__ T*)(x) + blockIdx_ * this->gmOffset_);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma);
        smooth1Gm.SetGlobalBuffer((__gm__ T*)smooth1);
        smooth2Gm.SetGlobalBuffer((__gm__ T*)smooth2);
        if (this->betaFlag == 1) {
            betaGm.SetGlobalBuffer((__gm__ T*)beta);
        }
    }

    __aicore__ inline void InitOutGlobalTensors(GM_ADDR y1, GM_ADDR y2, GM_ADDR outScale1, GM_ADDR outScale2)
    {
        int64_t yBufferSize = blockIdx_ * this->gmOffset_;
        if constexpr (IsSameType<T_Y, int4b_t>::value) {
            yBufferSize = yBufferSize / 2;
        }
        y1Gm.SetGlobalBuffer((__gm__ T_Y*)(y1) + yBufferSize);
        y2Gm.SetGlobalBuffer((__gm__ T_Y*)(y2) + yBufferSize);
        outScale1Gm.SetGlobalBuffer((__gm__ float*)outScale1 + blockIdx_ * this->firstDimPerCore);
        outScale2Gm.SetGlobalBuffer((__gm__ float*)outScale2 + blockIdx_ * this->firstDimPerCore);
    }

    __aicore__ inline void InitWorkSpaceGlobalTensors(GM_ADDR workspace)
    {}

protected:
    GlobalTensor<T> xGm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> smooth1Gm;
    GlobalTensor<T> smooth2Gm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T_Y> y1Gm;
    GlobalTensor<T_Y> y2Gm;
    GlobalTensor<float> outScale1Gm;
    GlobalTensor<float> outScale2Gm;

    uint32_t betaFlag;
    uint64_t numCore;
    uint64_t numFirstDim;
    uint64_t numLastDim;
    uint64_t numLastDimAligned;
    uint64_t firstDimPerCore;
    uint64_t firstDimPerCoreTail;
    uint64_t firstDimPerLoop;
    uint64_t lastDimSliceLen;
    uint64_t lastDimLoopNum;
    uint64_t lastDimSliceLenTail;

    float eps;
    float aveNum;

    uint64_t blockIdx_;
    uint64_t gmOffset_;
    uint64_t rowTail_;
    uint64_t rowStep;
    uint64_t rowWork;

    bool smooth1Exist;
    bool smooth2Exist;
    int32_t outQuant1Flag;
    int32_t outQuant2Flag;

    bool isOld;
    bool oldDouble;
    bool newSingleFirst;
    bool newSingleSecond;
    float quantMaxVal;
};

#endif // __ADD_RMS_NORM_DYNAMIC_QUANT_BASE_CLASS_H_
