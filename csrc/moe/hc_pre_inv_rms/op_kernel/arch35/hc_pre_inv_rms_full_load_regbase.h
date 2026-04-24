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
 * \file hc_pre_inv_rms.h
 * \brief inv rms file
 */
#ifndef ASCENDC_HC_PRE_INV_RMS_FULL_LOAD_REGBASE_H_
#define ASCENDC_HC_PRE_INV_RMS_FULL_LOAD_REGBASE_H_
#include "kernel_operator.h"

namespace HcPreInvRmsRegbase {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t FLOAT_BTYPE_SIZE = 4;
constexpr uint32_t VF_LEN_B32 = 64;
constexpr uint32_t UB_BLOCK_SIZE = 32;
constexpr uint32_t FOLD_FOUR = 4;

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

template <typename T>
__aicore__ inline void LoadInputData(AscendC::MicroAPI::RegTensor<float>& dst, __local_mem__ T* src, AscendC::MicroAPI::MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        AscendC::MicroAPI::RegTensor<T> tmp;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
class HcPreInvRmsFullLoadRegbase {
public:
    __aicore__ inline HcPreInvRmsFullLoadRegbase() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const HcPreInvRmsFullLoadTilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(uint64_t idx, uint64_t curUbFactorA);
    __aicore__ inline void Compute(uint64_t idx, uint64_t curUbFactorA);
    __aicore__ inline void ComputeFullLoadVF(LocalTensor<float>& yLocal, LocalTensor<T>& xLocal, uint32_t rAlign, uint32_t rNum, uint64_t curUbFactorA);
    __aicore__ inline void ComputeFullLoadVfPerf(LocalTensor<float>& yLocal, LocalTensor<T>& xLocal, uint32_t rAlign, uint32_t rNum, uint64_t curUbFactorA);
    __aicore__ inline void CopyOut(uint64_t idx, uint64_t curUbFactorA);

private:
    TPipe* pipe_;
    
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;

    GlobalTensor<T> xGm;
    GlobalTensor<float> yGm;

    int64_t A; // 输入数据 A 轴大小
    int64_t R; // 输入数据 R 轴大小
    int64_t blockNumA; // 使用的核数
    int64_t blockFactorA; // 每个核处理的A个数
    int64_t blockTailFactorA; // 尾核处理的A个数
    int64_t ubFactorA; // 每次ub循环处理的A个数
    int32_t blockIdx_;
    float epsilon;  // 算子参数
    uint32_t curBlockFactorA; // 当前核处理的A个数
    uint32_t rAlign;
};

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoadRegbase<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const HcPreInvRmsFullLoadTilingData* tiling, TPipe* pipe)
{
    A = tiling->A;
    R = tiling->R;
    blockNumA = tiling->blockNumA;
    blockFactorA = tiling->blockFactorA;
    blockTailFactorA = tiling->blockTailFactorA;
    ubFactorA = tiling->ubFactorA;
    epsilon = tiling->epsilon;

    rAlign = ((R * sizeof(T) + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE) * (UB_BLOCK_SIZE / sizeof(T));

    pipe_ = pipe;

    blockIdx_ = GetBlockIdx();

    if (blockIdx_ < blockNumA - 1) {
        this->curBlockFactorA = this->blockFactorA;
    } else if (blockIdx_ == blockNumA - 1) {
        this->curBlockFactorA = this->blockTailFactorA;
    } else {
        return;
    }
    xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * blockFactorA * R, curBlockFactorA * R);
    yGm.SetGlobalBuffer((__gm__ float*)y + blockIdx_ * blockFactorA, curBlockFactorA);
    // pipe alloc memory to queue, the unit is Bytes
    pipe_->InitBuffer(inQueueX, BUFFER_NUM, ubFactorA * rAlign * sizeof(T));
    pipe_->InitBuffer(outQueueY, BUFFER_NUM, ubFactorA * FLOAT_BTYPE_SIZE);
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoadRegbase<T>::Process()
{
    if (blockIdx_ >= blockNumA) {
        return;
    }
    uint64_t aUbLoopCount = (curBlockFactorA + ubFactorA - 1) / ubFactorA; // Ub循环次数
    uint64_t tailUbFactorA = curBlockFactorA - (aUbLoopCount - 1) * ubFactorA; // 最后一次Ub循环的A轴大小
    uint64_t curUbFactorA = ubFactorA;
    for (uint64_t idx = 0; idx < aUbLoopCount; idx++) {
        if (idx == aUbLoopCount - 1) {
            curUbFactorA = tailUbFactorA;
        }
        CopyIn(idx, curUbFactorA);
        Compute(idx, curUbFactorA);
        CopyOut(idx, curUbFactorA);
    }
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoadRegbase<T>::CopyIn(uint64_t idx, uint64_t curUbFactorA)
{
    
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();

    DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
    int32_t xGmStartAddr = idx * R * ubFactorA;
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(curUbFactorA), static_cast<uint32_t>(R * sizeof(T)), 0, 0, 0};
    DataCopyPad(xLocal, xGm[xGmStartAddr], dataCopyParams, dataCopyPadParams);
    
    inQueueX.EnQue<T>(xLocal);
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoadRegbase<T>::Compute(uint64_t idx, uint64_t curUbFactorA)
{
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

    if (R % 256 == 0) {
        ComputeFullLoadVfPerf(yLocal, xLocal, rAlign, R, curUbFactorA);
    } else {
        ComputeFullLoadVF(yLocal, xLocal, rAlign, R, curUbFactorA);
    }
    
    outQueueY.EnQue<float>(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoadRegbase<T>::ComputeFullLoadVF(LocalTensor<float>& yLocal, LocalTensor<T>& xLocal, uint32_t rAlign, uint32_t rNum, uint64_t curUbFactorA)
{
    __ubuf__ T* xAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    __ubuf__ float* yAddr = (__ubuf__ float*)yLocal.GetPhyAddr();

    uint32_t vfLen = VF_LEN_B32;
    uint16_t iLoopNum = curUbFactorA;
    uint16_t needLoopNum = (rAlign + vfLen - 1) / vfLen;  // 需要VF循环次数
    uint16_t fourLoopNum = (needLoopNum + FOLD_FOUR - 1) / FOLD_FOUR; // 需要四循环次数
    uint16_t formerFourLoopNum = fourLoopNum - 1;
    uint16_t tailFourLoop = needLoopNum - formerFourLoopNum * FOLD_FOUR; // 最后一次四循环需处理VF数
    uint16_t formerFourLoopElems = formerFourLoopNum * FOLD_FOUR * vfLen; // 除尾块外，处理元素个数
    uint32_t tailFourLoopElems = rNum - formerFourLoopElems; // 最后一次四循环需处理实际元素个数

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vregX1;
        AscendC::MicroAPI::RegTensor<float> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX3;
        AscendC::MicroAPI::RegTensor<float> vregX4;
        AscendC::MicroAPI::RegTensor<float> vregX;
        AscendC::MicroAPI::RegTensor<float> vregSum;
        AscendC::MicroAPI::RegTensor<float> vregR;
        AscendC::MicroAPI::RegTensor<float> vregOne;
        AscendC::MicroAPI::MaskReg preg;
        AscendC::MicroAPI::MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregOne = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::Duplicate(vregOne, 1.0f);
        
        for (uint16_t i = 0; i < iLoopNum; i++) {
            AscendC::MicroAPI::Duplicate(vregSum, 0.0f);  // 用于累加的vreg

            for (uint16_t j = 0; j < formerFourLoopNum; j++) {
                uint32_t srcOffset1 = i * rAlign + 4 * j * vfLen;
                LoadInputData<T>(vregX1, xAddr, pregAll, srcOffset1);
                AscendC::MicroAPI::Mul(vregX1, vregX1, vregX1, pregAll);
                AscendC::MicroAPI::Add(vregSum, vregSum, vregX1, pregAll);

                uint32_t srcOffset2 = i * rAlign + (4 * j + 1) * vfLen;
                LoadInputData<T>(vregX2, xAddr, pregAll, srcOffset2);
                AscendC::MicroAPI::Mul(vregX2, vregX2, vregX2, pregAll);
                AscendC::MicroAPI::Add(vregSum, vregSum, vregX2, pregAll);

                uint32_t srcOffset3 = i * rAlign + (4 * j + 2) * vfLen;
                LoadInputData<T>(vregX3, xAddr, pregAll, srcOffset3);
                AscendC::MicroAPI::Mul(vregX3, vregX3, vregX3, pregAll);
                AscendC::MicroAPI::Add(vregSum, vregSum, vregX3, pregAll);

                uint32_t srcOffset4 = i * rAlign + (4 * j + 3) * vfLen;
                LoadInputData<T>(vregX4, xAddr, pregAll, srcOffset4);
                AscendC::MicroAPI::Mul(vregX4, vregX4, vregX4, pregAll);
                AscendC::MicroAPI::Add(vregSum, vregSum, vregX4, pregAll);
            }

            tailFourLoopElems = rNum - formerFourLoopElems;
            for (uint16_t j = 0; j < tailFourLoop; j++) {
                preg = AscendC::MicroAPI::UpdateMask<float>(tailFourLoopElems);
                uint32_t srcOffset = i * rAlign + formerFourLoopElems + j * vfLen;
                LoadInputData<T>(vregX, xAddr, preg, srcOffset);
                AscendC::MicroAPI::Mul(vregX, vregX, vregX, preg);
                AscendC::MicroAPI::Add(vregSum, vregSum, vregX, pregAll);
            }

            Reduce(vregSum, vregSum, pregAll);

            AscendC::MicroAPI::Duplicate(vregR, (float)rNum);
            AscendC::MicroAPI::Div(vregSum, vregSum, vregR, pregOne);
            AscendC::MicroAPI::Adds(vregSum, vregSum, epsilon, pregOne);
            AscendC::MicroAPI::Sqrt(vregSum, vregSum, pregOne);
            AscendC::MicroAPI::Div(vregSum, vregOne, vregSum, pregOne);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(yAddr + i, vregSum, pregOne);
        }
    }
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoadRegbase<T>::ComputeFullLoadVfPerf(LocalTensor<float>& yLocal, LocalTensor<T>& xLocal, uint32_t rAlign, uint32_t rNum, uint64_t curUbFactorA)
{
    __ubuf__ T* xAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    __ubuf__ float* yAddr = (__ubuf__ float*)yLocal.GetPhyAddr();

    uint32_t vfLen = VF_LEN_B32;
    uint16_t iLoopNum = curUbFactorA;
    uint16_t jLoopNum = (rAlign / vfLen) / FOLD_FOUR;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vregX1;
        AscendC::MicroAPI::RegTensor<float> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX3;
        AscendC::MicroAPI::RegTensor<float> vregX4;
        AscendC::MicroAPI::RegTensor<float> vregSum;
        AscendC::MicroAPI::RegTensor<float> vregR;
        AscendC::MicroAPI::RegTensor<float> vregOne;
        AscendC::MicroAPI::MaskReg pregAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg pregOne = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        AscendC::MicroAPI::Duplicate(vregOne, 1.0f);

        for (uint16_t i = 0; i < iLoopNum; i++) {
            AscendC::MicroAPI::Duplicate(vregSum, 0.0f);  // 用于累加的vreg
            for (uint16_t j = 0; j < jLoopNum; j++) {
                uint32_t srcOffset1 = i * rAlign + 4 * j * vfLen;
                LoadInputData<T>(vregX1, xAddr, pregAll, srcOffset1);
                AscendC::MicroAPI::Mul(vregX1, vregX1, vregX1, pregAll);
                AscendC::MicroAPI::Add(vregSum, vregSum, vregX1, pregAll);

                uint32_t srcOffset2 = i * rAlign + (4 * j + 1) * vfLen;
                LoadInputData<T>(vregX2, xAddr, pregAll, srcOffset2);
                AscendC::MicroAPI::Mul(vregX2, vregX2, vregX2, pregAll);
                AscendC::MicroAPI::Add(vregSum, vregSum, vregX2, pregAll);

                uint32_t srcOffset3 = i * rAlign + (4 * j + 2) * vfLen;
                LoadInputData<T>(vregX3, xAddr, pregAll, srcOffset3);
                AscendC::MicroAPI::Mul(vregX3, vregX3, vregX3, pregAll);
                AscendC::MicroAPI::Add(vregSum, vregSum, vregX3, pregAll);

                uint32_t srcOffset4 = i * rAlign + (4 * j + 3) * vfLen;
                LoadInputData<T>(vregX4, xAddr, pregAll, srcOffset4);
                AscendC::MicroAPI::Mul(vregX4, vregX4, vregX4, pregAll);
                AscendC::MicroAPI::Add(vregSum, vregSum, vregX4, pregAll);
            }

            Reduce(vregSum, vregSum, pregAll);
            AscendC::MicroAPI::Duplicate(vregR, (float)rNum);
            AscendC::MicroAPI::Div(vregSum, vregSum, vregR, pregOne);
            AscendC::MicroAPI::Adds(vregSum, vregSum, epsilon, pregOne);
            AscendC::MicroAPI::Sqrt(vregSum, vregSum, pregOne);
            AscendC::MicroAPI::Div(vregSum, vregOne, vregSum, pregOne);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(yAddr + i, vregSum, pregOne);
        }
    }
}

template <typename T>
__aicore__ inline void HcPreInvRmsFullLoadRegbase<T>::CopyOut(uint64_t idx, uint64_t curUbFactorA)
{
    LocalTensor<float> yLocal = outQueueY.DeQue<float>();
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curUbFactorA * sizeof(float)), 0, 0, 0};
    DataCopyPad(yGm[idx * ubFactorA], yLocal, copyParams);
    outQueueY.FreeTensor(yLocal);
}

} // namespace HcPreInvRmsRegbase
#endif // ASCENDC_HC_PRE_INV_RMS_FULL_LOAD_REGBASE_H_

