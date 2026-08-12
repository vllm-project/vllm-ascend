/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file weight_quant_batch_matmul_v2_custom_mix_splitk.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_MIX_SPLITK_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_MIX_SPLITK_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "tool.h"
#include "weight_quant_batch_matmul_v2_constant.h"
#include "weight_quant_batch_matmul_v2_common.h"

using AscendC::AIC;
using AscendC::AIV;
using AscendC::BinaryRepeatParams;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::GlobalTensor;
using AscendC::InitDump;
using AscendC::int4b_t;
using AscendC::LocalTensor;
using AscendC::Nd2NzParams;
using AscendC::ONE_BLK_SIZE;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::SyncAll;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using AscendC::UnaryRepeatParams;
using matmul::MatmulImpl;
using matmul::MatmulType;

namespace Mc2WeightQuantBatchMatmulV2 {
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
class Mc2WeightQuantBatchMatmulV2MixSplitKKernel
    : public Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                               hasAntiQuantOffset, quantType> {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2MixSplitKKernel(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const Mc2WeightQuantBatchMatmulV2TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline void Process();

    using inputXType = MatmulType<TPosition::A1, CubeFormat::NZ, xType, aTrans>;
    using inputWType = MatmulType<TPosition::B1, CubeFormat::NZ, xType, bTrans>;
    using outputYType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using inputBiasType = MatmulType<TPosition::GM, CubeFormat::ND, biasType>;
    MatmulImpl<inputXType, inputWType, outputYType, inputBiasType> mmObj;

private:
    __aicore__ inline void InitBuffer();
    __aicore__ inline void InitWorkSpace(GM_ADDR workspace);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void ProcessCube();
    __aicore__ inline void ProcessVector();
    __aicore__ inline void AntiquantWeight(uint64_t singleCoreRealN, uint64_t groupIdx, uint64_t nBaseOffset,
                                           uint64_t kBaseOffset);
    __aicore__ inline void CopyInAntiquantParams(uint64_t singleCoreRealN, uint64_t groupIdx, uint64_t nBaseOffset);
    __aicore__ inline void CopyInWeight(uint32_t singleCoreRealK, uint32_t singleCoreRealN, uint64_t kOffset,
                                        uint64_t nOffset);
    __aicore__ inline void ProcessAntiquantParams(uint32_t singleCoreRealN);
    __aicore__ inline void AntiQuantCompute(uint32_t singleCoreRealK, uint32_t singleCoreRealN, uint64_t kOffset,
                                            uint64_t nOffset);
    __aicore__ inline void ProcessMatmulResult();

    __aicore__ inline void CopyInAL1(uint64_t realSingleCoreK, uint64_t kOffset);
    __aicore__ inline void CopyInBL1(uint64_t wOffset, uint64_t singleCoreRealN);
    __aicore__ inline void LaunchMatmul(uint64_t singleCoreRealN, uint64_t cOffset, uint64_t kOffset, bool lastLoop);

protected:
    uint64_t nF16AlignTo512bSize;
    int32_t vecNDimIdx_;
    int32_t vecKDimIdx_;
    int32_t cubeNDimIdx_;
    int32_t cubeKDimIdx_;

    GlobalTensor<float> matmulAtomicAddResult_;
    LocalTensor<half> offsetComputeTensor_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> offsetQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> scaleQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> originWeightQueue_;
    TQue<QuePosition::VECOUT, SINGLE_BUFFER_NUM> weightOutputQueue_;
    TBuf<> offsetTmpBuf_;
    TQue<QuePosition::B1, DOUBLE_BUFFER_NUM> inQueueBL1_;
    TBuf<TPosition::A1> a1Tbuf_;
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::ComputeConstexpr()
{
    this->nF16AlignTo512bSize = this->CeilDiv(this->tiling_->nSize, 256) * 256;
    this->weightCacheSizeAlign_ = this->tiling_->cubeBlockDimK * this->tiling_->groupSize * this->nF16AlignTo512bSize;
    this->weightCacheIdx_ = 0;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::InitBuffer()
{
    this->pipe_->InitBuffer(this->scaleComputeTbuf_, 2048);
    this->scaleComputeTensor_ = this->scaleComputeTbuf_.template Get<biasType>();
    this->pipe_->InitBuffer(this->offsetComputeTbuf_, 1024);
    this->offsetComputeTensor_ = this->offsetComputeTbuf_.template Get<half>();

    this->pipe_->InitBuffer(this->weightOutputQueue_, SINGLE_BUFFER_NUM, 32 * 1024);
    this->pipe_->InitBuffer(this->originWeightQueue_, DOUBLE_BUFFER_NUM, 16 * 1024);

    this->pipe_->InitBuffer(this->weight32Tbuf_, 256 + 64 * 1024);
    this->pipe_->InitBuffer(this->offsetQueue_, DOUBLE_BUFFER_NUM, 1024);
    this->pipe_->InitBuffer(this->scaleQueue_, DOUBLE_BUFFER_NUM, 1024);

    this->pipe_->InitBuffer(this->offsetTmpBuf_, 2048 + 256);

    this->pipe_->InitBuffer(this->weight16Tbuf_, 256 + 32 * 1024);

    this->pipe_->InitBuffer(a1Tbuf_, this->CeilDiv(this->tiling_->mSize, 16UL) * 16UL * this->tiling_->vecSingleK *
                                         sizeof(xType));
    this->pipe_->InitBuffer(inQueueBL1_, DOUBLE_BUFFER_NUM, 128 * 1024);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::InitWorkSpace(GM_ADDR workspace)
{
    // 当前cv绑核固定，需要四份
    uint64_t weithCacheSize = 4 * this->weightCacheSizeAlign_ * sizeof(xType);
    this->weightCache_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(workspace));

    this->matmulAtomicAddResult_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + weithCacheSize));
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MixSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale,
                     GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                     const Mc2WeightQuantBatchMatmulV2TilingData *tilingData, TPipe *tPipe)
{
    this->BaseInit(tilingData, tPipe);
    this->InitInput(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y);
    ComputeConstexpr();
    InitBuffer();
    InitWorkSpace(workspace);

    if ASCEND_IS_AIC {
        mmObj.SetSubBlockIdx(0);
        mmObj.Init(&this->tiling_->matmulTiling, this->pipe_);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight,
                                                                                            GM_ADDR antiquantScale,
                                                                                            GM_ADDR antiquantOffset,
                                                                                            GM_ADDR quantScale,
                                                                                            GM_ADDR quantOffset,
                                                                                            GM_ADDR bias, GM_ADDR y,
                                                                                            GM_ADDR workspace)
{
    this->InitInput(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y);
    this->InitWorkSpace(workspace);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::ProcessVector()
{
    uint64_t initTotalSize = this->tiling_->mSize * this->tiling_->nSize;
    InitAtomicAddr(matmulAtomicAddResult_, initTotalSize, this->curBlockIdx_);
    SyncAll();
    vecNDimIdx_ = this->curBlockIdx_ % this->tiling_->vecBlockDimN;
    vecKDimIdx_ = this->curBlockIdx_ / this->tiling_->vecBlockDimN;
    uint64_t kBaseOffset = vecKDimIdx_ * this->tiling_->vecSingleK;

    uint64_t groupStartIdx = kBaseOffset / this->tiling_->groupSize;
    uint64_t groupLimitIdx = groupStartIdx + this->tiling_->vecSingleKGroupNum;
    if (kBaseOffset + this->tiling_->vecSingleK > this->tiling_->kSize) {
        groupLimitIdx = this->tiling_->kSize / this->tiling_->groupSize;
    }
    uint64_t taskId = 0;
    uint64_t singleCoreRealN = this->tiling_->vecSingleN;
    for (uint64_t groupIdx = 0; groupIdx + groupStartIdx < groupLimitIdx; groupIdx++, this->weightCacheIdx_++) {
        singleCoreRealN = this->tiling_->vecSingleN;
        for (uint64_t singleCoreInnerNOffset = vecNDimIdx_ * this->tiling_->vecSingleN;
             singleCoreInnerNOffset < this->tiling_->nSize;
             singleCoreInnerNOffset += this->tiling_->vecBlockDimN * this->tiling_->vecSingleN, taskId++) {
            if (singleCoreInnerNOffset + singleCoreRealN > this->tiling_->nSize) {
                singleCoreRealN = this->tiling_->nSize - singleCoreInnerNOffset;
            }
            if (taskId > 3) {
                CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
            }
            AntiquantWeight(singleCoreRealN, groupIdx + groupStartIdx, singleCoreInnerNOffset,
                            kBaseOffset + groupIdx * this->tiling_->groupSize);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
        }
    }

    CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
    CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
    CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
    CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);

    // 后处理需要等最后一拍matmul全部计算完
    CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
    ProcessMatmulResult();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::AntiquantWeight(uint64_t singleCoreRealN,
                                                                                           uint64_t groupIdx,
                                                                                           uint64_t nBaseOffset,
                                                                                           uint64_t kBaseOffset)
{
    if (this->curBlockIdx_ >= this->tiling_->vecBlockDimN * this->tiling_->cubeBlockDimK) {
        return;
    }
    uint64_t singleCoreK = 32;
    uint64_t singleCoreRealK = singleCoreK;

    this->CopyInAntiquantParams(singleCoreRealN, groupIdx, nBaseOffset);
    CopyInWeight(singleCoreRealK, singleCoreRealN, kBaseOffset, nBaseOffset);

    ProcessAntiquantParams(singleCoreRealN);
    AntiQuantCompute(singleCoreRealK, singleCoreRealN, vecKDimIdx_ * this->tiling_->groupSize, nBaseOffset);

    for (uint64_t kOffset = singleCoreK; kOffset < this->tiling_->groupSize; kOffset += singleCoreK) {
        if (kOffset + singleCoreK > this->tiling_->groupSize) {
            singleCoreRealK = this->tiling_->groupSize - kOffset;
        }
        CopyInWeight(singleCoreRealK, singleCoreRealN, kBaseOffset + kOffset, nBaseOffset);
        AntiQuantCompute(singleCoreRealK, singleCoreRealN, kOffset + vecKDimIdx_ * this->tiling_->groupSize,
                         nBaseOffset);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MixSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::CopyInAntiquantParams(uint64_t singleCoreRealN, uint64_t groupIdx, uint64_t nBaseOffset)
{
    uint64_t antiquantOffset = groupIdx * this->tiling_->nSize + nBaseOffset;

    LocalTensor<xType> offsetInput = this->offsetQueue_.template AllocTensor<xType>();
    DataCopyPad2D(offsetInput, this->offsetGlobal_[antiquantOffset], 1, singleCoreRealN, this->tiling_->nSize);
    this->offsetQueue_.EnQue(offsetInput);

    LocalTensor<xType> scaleInput = this->scaleQueue_.template AllocTensor<xType>();
    DataCopyPad2D(scaleInput, this->scaleGlobal_[antiquantOffset], 1, singleCoreRealN, this->tiling_->nSize);
    this->scaleQueue_.EnQue(scaleInput);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::CopyInWeight(uint32_t singleCoreRealK,
                                                                                        uint32_t singleCoreRealN,
                                                                                        uint64_t kOffset,
                                                                                        uint64_t nOffset)
{
    uint64_t wSrcOffset = kOffset * this->tiling_->nSize + nOffset;
    LocalTensor<wType> originWeight = this->originWeightQueue_.template AllocTensor<wType>();
    DataCopyPad2D(originWeight, this->wGlobal_[wSrcOffset], singleCoreRealK, singleCoreRealN, this->tiling_->nSize);
    originWeightQueue_.EnQue(originWeight);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::ProcessAntiquantParams(uint32_t
                                                                                                      singleCoreRealN)
{
    LocalTensor<xType> offsetInput = this->offsetQueue_.template DeQue<xType>();
    LocalTensor<float> offsetTmp = this->offsetTmpBuf_.template Get<float>();
    Cast(offsetTmp, offsetInput, RoundMode::CAST_NONE, singleCoreRealN);
    this->offsetQueue_.FreeTensor(offsetInput);
    PipeBarrier<PIPE_V>();
    Cast(this->offsetComputeTensor_, offsetTmp, RoundMode::CAST_NONE, singleCoreRealN);

    LocalTensor<xType> scaleInput = this->scaleQueue_.template DeQue<xType>();
    Cast(this->scaleComputeTensor_, scaleInput, RoundMode::CAST_NONE, singleCoreRealN);
    this->scaleQueue_.FreeTensor(scaleInput);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::AntiQuantCompute(uint32_t singleCoreRealK,
                                                                                            uint32_t singleCoreRealN,
                                                                                            uint64_t kOffset,
                                                                                            uint64_t nOffset)
{
    LocalTensor<half> weight16 = this->weight16Tbuf_.template Get<half>()[128];
    LocalTensor<wType> originWeight = this->originWeightQueue_.template DeQue<wType>();
    Cast(weight16, originWeight, RoundMode::CAST_NONE, singleCoreRealK * singleCoreRealN);
    originWeightQueue_.FreeTensor(originWeight);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> weight16AfterAdd = this->weight16Tbuf_.template Get<half>();

    constexpr uint32_t fp16MaskSize = ONE_REPEAT_BYTE_SIZE / sizeof(half);
    BinaryRepeatParams repeatParams;
    repeatParams.dstRepStride = 8;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = 8;
    repeatParams.src1BlkStride = 1;
    repeatParams.src1RepStride = 8;

    Add(weight16AfterAdd, weight16, this->offsetComputeTensor_, fp16MaskSize, singleCoreRealN / fp16MaskSize,
        repeatParams);
    for (uint64_t kIdx = 1; kIdx < singleCoreRealK; kIdx++) {
        AscendC::Add<half, false>(weight16AfterAdd[kIdx * singleCoreRealN], weight16[kIdx * singleCoreRealN],
                                  this->offsetComputeTensor_, fp16MaskSize, singleCoreRealN / fp16MaskSize,
                                  repeatParams);
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<float> weight32 = this->weight32Tbuf_.template Get<float>()[64];
    Cast(weight32, weight16AfterAdd, RoundMode::CAST_NONE, singleCoreRealK * singleCoreRealN);
    PipeBarrier<PIPE_V>();

    constexpr uint32_t fp32MaskSize = ONE_REPEAT_BYTE_SIZE / sizeof(float);
    LocalTensor<float> weight32AfterMul = this->weight32Tbuf_.template Get<float>();
    Mul(weight32AfterMul, weight32, this->scaleComputeTensor_, fp32MaskSize, singleCoreRealN / fp32MaskSize,
        repeatParams);

    for (uint64_t kIdx = 1; kIdx < singleCoreRealK; kIdx++) {
        AscendC::Mul<float, false>(weight32AfterMul[kIdx * singleCoreRealN], weight32[kIdx * singleCoreRealN],
                                   this->scaleComputeTensor_, fp32MaskSize, singleCoreRealN / fp32MaskSize,
                                   repeatParams);
    }

    PipeBarrier<PIPE_V>();

    LocalTensor<xType> weightOutput = this->weightOutputQueue_.template AllocTensor<xType>();
    Cast(weightOutput, weight32AfterMul, RoundMode::CAST_RINT, singleCoreRealK * singleCoreRealN);
    this->weightOutputQueue_.EnQue(weightOutput);
    this->weightOutputQueue_.template DeQue<xType>();

    uint64_t wDstOffset =
        (this->weightCacheIdx_ % 4) * this->weightCacheSizeAlign_ + kOffset * nF16AlignTo512bSize + nOffset;
    DataCopyPad2D(this->weightCache_[wDstOffset], weightOutput, singleCoreRealK, singleCoreRealN, nF16AlignTo512bSize);
    this->weightOutputQueue_.FreeTensor(weightOutput);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::ProcessMatmulResult()
{
    uint64_t singleCoreN = 256;
    uint64_t singleCoreRealN = 256;
    uint64_t totalCoreNum = GetBlockNum() * 2;
    for (uint64_t nOffset = this->curBlockIdx_ * singleCoreN; nOffset < this->tiling_->nSize;
         nOffset += totalCoreNum * singleCoreN) {
        if (nOffset + singleCoreN > this->tiling_->nSize) {
            singleCoreRealN = this->tiling_->nSize - nOffset;
        }
        if (nOffset > this->curBlockIdx_ * singleCoreN) {
            event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        }

        LocalTensor<float> atomicResult = this->weight32Tbuf_.template Get<float>();
        DataCopyPad2D(atomicResult, matmulAtomicAddResult_[nOffset], this->tiling_->mSize, singleCoreRealN,
                      this->tiling_->nSize);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        LocalTensor<xType> finalResult = weightOutputQueue_.AllocTensor<xType>();
        Cast(finalResult, atomicResult, RoundMode::CAST_RINT, this->tiling_->mSize * singleCoreRealN);
        weightOutputQueue_.EnQue(finalResult);
        weightOutputQueue_.DeQue<xType>();

        DataCopyPad2D(this->yGlobal_[nOffset], finalResult, this->tiling_->mSize, singleCoreRealN,
                      this->tiling_->nSize);
        weightOutputQueue_.FreeTensor(finalResult);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::ProcessCube()
{
    // mte2 流水
    cubeKDimIdx_ = this->curBlockIdx_ / this->tiling_->cubeBlockDimN;
    cubeNDimIdx_ = this->curBlockIdx_ % this->tiling_->cubeBlockDimN;
    uint64_t kBaseOffset = cubeKDimIdx_ * this->tiling_->vecSingleK;
    uint64_t singleCoreRealK = this->tiling_->vecSingleK;
    uint64_t groupStartIdx = kBaseOffset / this->tiling_->groupSize;
    uint64_t groupLimitIdx = groupStartIdx + this->tiling_->vecSingleKGroupNum;
    if (kBaseOffset + this->tiling_->vecSingleK > this->tiling_->kSize) {
        groupLimitIdx = this->tiling_->kSize / this->tiling_->groupSize;
        singleCoreRealK = this->tiling_->kSize - kBaseOffset;
    }

    CopyInAL1(singleCoreRealK, kBaseOffset);

    uint64_t kOffset = kBaseOffset;
    for (uint64_t groupIdx = 0; groupIdx + groupStartIdx < groupLimitIdx; groupIdx++) {
        uint64_t singleCoreRealN = this->tiling_->matmulTiling.singleCoreN;
        for (uint64_t singleCoreInnerNOffset = cubeNDimIdx_ * this->tiling_->matmulTiling.singleCoreN;
             singleCoreInnerNOffset < this->tiling_->nSize;
             singleCoreInnerNOffset += this->tiling_->cubeBlockDimN * this->tiling_->matmulTiling.singleCoreN) {
            if (singleCoreInnerNOffset + singleCoreRealN > this->tiling_->nSize) {
                singleCoreRealN = this->tiling_->nSize - singleCoreInnerNOffset;
            }
            CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);
            if (this->curBlockIdx_ < this->tiling_->cubeBlockDimN * this->tiling_->cubeBlockDimK) {
                CopyInBL1((this->weightCacheIdx_ % 4) * this->weightCacheSizeAlign_ +
                              (cubeKDimIdx_ * this->tiling_->groupSize) * this->nF16AlignTo512bSize +
                              singleCoreInnerNOffset,
                          singleCoreRealN);
                LaunchMatmul(singleCoreRealN, singleCoreInnerNOffset, kOffset - kBaseOffset, false);
            }
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
        }
        kOffset += this->tiling_->groupSize;
        this->weightCacheIdx_++;
    }
    CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(SYNC_AIC_ONLY_ALL_FLAG);
    CrossCoreWaitFlag(SYNC_AIC_ONLY_ALL_FLAG);
    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::CopyInAL1(uint64_t realSingleCoreK,
                                                                                     uint64_t kOffset)
{
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = this->tiling_->mSize;
    nd2nzParams.dValue = realSingleCoreK;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue = this->tiling_->kSize;
    nd2nzParams.dstNzC0Stride = this->CeilDiv(this->tiling_->mSize, 16UL) * 16UL;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    LocalTensor<xType> aL1Tensor = a1Tbuf_.template Get<xType>();

    DataCopy(aL1Tensor, this->xGlobal_[kOffset], nd2nzParams);

    TEventID eventIdMte2ToMte1 = GetTPipePtr()->FetchEventID<HardEvent::MTE2_MTE1>();
    SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
    WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::CopyInBL1(uint64_t wOffset,
                                                                                     uint64_t singleCoreRealN)
{
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = this->tiling_->groupSize;
    nd2nzParams.dValue = singleCoreRealN;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue = nF16AlignTo512bSize;
    nd2nzParams.dstNzC0Stride = this->CeilDiv(nd2nzParams.nValue, 16) * 16;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    LocalTensor<xType> bL1Tensor = inQueueBL1_.AllocTensor<xType>();
    DataCopy(bL1Tensor, this->weightCache_[wOffset], nd2nzParams);
    inQueueBL1_.EnQue(bL1Tensor);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::LaunchMatmul(uint64_t singleCoreRealN,
                                                                                        uint64_t cOffset,
                                                                                        uint64_t kOffset, bool lastLoop)
{
    LocalTensor<xType> aL1Tensor = a1Tbuf_.template Get<xType>();
    LocalTensor<xType> bL1Tensor = inQueueBL1_.DeQue<xType>();

    mmObj.SetTensorA(aL1Tensor[CeilAlign(this->tiling_->mSize, 16UL) * kOffset], aTrans);

    mmObj.SetTensorB(bL1Tensor, bTrans);

    mmObj.SetOrgShape(this->tiling_->mSize, nF16AlignTo512bSize, this->tiling_->kSize, this->tiling_->kSize,
                      this->tiling_->nSize);
    mmObj.SetTail(this->tiling_->mSize, singleCoreRealN, this->tiling_->groupSize);
    mmObj.IterateAll(matmulAtomicAddResult_[cOffset], true);
    inQueueBL1_.FreeTensor(bL1Tensor);
    mmObj.End();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MixSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType>::Process()
{
    if ASCEND_IS_AIV {
        ProcessVector();
    } else {
        ProcessCube();
    }
}
} // namespace Mc2WeightQuantBatchMatmulV2
#endif // WEIGHT_QUANT_BATCHMATMUL_V2_MIX_SPLITK_H
