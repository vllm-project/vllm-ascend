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
 * \file weight_quant_batch_matmul_v2_custom_weight_nz.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_CUSTOM_WEITHG_NZ_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_CUSTOM_WEITHG_NZ_H

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
using AscendC::BLOCK_CUBE;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::InitDump;
using AscendC::int4b_t;
using AscendC::LocalTensor;
using AscendC::MAX_REPEAT_TIMES;
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
class Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel
    : public Mc2WeightQuantBatchMatmulV2Common<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                               hasAntiQuantOffset, quantType> {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const Mc2WeightQuantBatchMatmulV2TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline void ComputeWeightOffsetInfo(uint64_t nLoopIdx, uint64_t nBaseOffset, uint64_t kLoopIdx,
                                                   WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void ComputeWeightSplitInfo(uint64_t nLoopIdx, uint64_t nLoopLimit, uint64_t nRealSize,
                                                  uint64_t kLoopIdx, WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void Process();

    using inputXType = MatmulType<TPosition::GM, CubeFormat::ND, xType, aTrans>;
    using inputWType = MatmulType<TPosition::GM, CubeFormat::NZ, xType, bTrans>;
    using outputYType = MatmulType<TPosition::GM, CubeFormat::ND, yType>;
    using inputBiasType = MatmulType<TPosition::GM, CubeFormat::ND, biasType>;
    MatmulImpl<inputXType, inputWType, outputYType, inputBiasType, CFG_MDL> mmObj;

private:
    using antiQuantCalType = typename Mc2AntiQuantCalType<biasType>::type;
    __aicore__ inline void ProcessCube();
    __aicore__ inline void SetMatmulParams(int32_t cubeNLoopIdx, uint64_t aOffset, uint64_t bOffset, uint64_t nOffset);
    __aicore__ inline void ProcessVector();
    __aicore__ inline void AntiquantWeight(uint64_t cubeNLoopIdx, uint64_t nBaseOffset, uint64_t nRealSize,
                                           uint64_t nLoopLimit, WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void WeightCast(WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void WeightCopyIn(WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void AntiQuantCompute(WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void ComputePerTensor(LocalTensor<antiQuantCalType> &antiquantWeightTensor);
    __aicore__ inline void ComputePerChannel(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                             WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void ComputePerChannelTrans(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                                  WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void ComputePerChannelNotTrans(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                                     WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void ComputePerGroup(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                           WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void ComputePerGroupTransF16(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                                   WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void ComputePerGroupTransF32(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                                   WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void ComputePerGroupNotTransF16(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                                      WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void ComputePerGroupNotTransF32(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                                      WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void BrcbAntiquantParams(WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void WeightCopyOut(WeightSplitInfo &weightSplitInfo);
    __aicore__ inline void InitInputOutPut(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                           GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y);
    __aicore__ inline void NotifyCube();

    static constexpr int32_t INT4_OR_INT8_BLOCK_SIZE = GetBlockSize<wType>();
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType>::InitInputOutPut(GM_ADDR x,
                                                                                                GM_ADDR weight,
                                                                                                GM_ADDR antiquantScale,
                                                                                                GM_ADDR antiquantOffset,
                                                                                                GM_ADDR quantScale,
                                                                                                GM_ADDR quantOffset,
                                                                                                GM_ADDR bias, GM_ADDR y)
{
    this->InitInput(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y);
    this->wGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ wType *>(weight),
                                   this->tiling_->kAlign * this->tiling_->nAlign);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale,
                     GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                     const Mc2WeightQuantBatchMatmulV2TilingData *tilingData, TPipe *tPipe)
{
    this->BaseInit(tilingData, tPipe);
    InitInputOutPut(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y);
    this->wFormat_ = static_cast<int32_t>(CubeFormat::NZ);
    this->ComputeConstexpr();
    if constexpr (IsSameType<xType, bfloat16_t>::value && antiQuantType != Mc2QuantType::PER_GROUP) {
        this->broadCastFactor_ = 2;
        // fp32也要brcb到16，参数相应跳过一半数据
        this->brcbParams_.dstBlkStride = 2;
        this->brcbParams_.dstRepStride = 16;
    }
    this->weightCacheSizeAlign_ =
        this->tiling_->matmulTiling.singleCoreN * this->tiling_->cubeBlockDimN * this->tiling_->kAlign;
    this->weightCacheSizeAlign_ = this->CeilDiv(this->weightCacheSizeAlign_, 256UL) * 256;

    this->InitBuffer();
    this->InitWorkSpace(workspace);
    if ASCEND_IS_AIC {
        mmObj.SetSubBlockIdx(0);
        mmObj.Init(&this->tiling_->matmulTiling, this->pipe_);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::UpdateGlobalAddr(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                 GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace)
{
    this->InitInput(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y);
    this->InitWorkSpace(workspace);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType>::WeightCopyIn(WeightSplitInfo
                                                                                                 &weightSplitInfo)
{
    DataCopyParams copyinParams;
    uint64_t wSrcOffset;
    if constexpr (bTrans) {
        wSrcOffset =
            weightSplitInfo.kOffset * this->tiling_->nAlign + weightSplitInfo.originNOffset * INT4_OR_INT8_BLOCK_SIZE;
        copyinParams.blockCount = weightSplitInfo.vecNzSingleK / INT4_OR_INT8_BLOCK_SIZE;
        copyinParams.blockLen = weightSplitInfo.vecNzSingleN;
        copyinParams.srcStride = this->tiling_->nAlign - weightSplitInfo.vecNzSingleN;
        copyinParams.dstStride = 0;
    } else {
        wSrcOffset =
            weightSplitInfo.originNOffset * this->tiling_->kAlign + weightSplitInfo.kOffset * INT4_OR_INT8_BLOCK_SIZE;
        copyinParams.blockCount = weightSplitInfo.vecNzSingleN / INT4_OR_INT8_BLOCK_SIZE;
        copyinParams.blockLen = weightSplitInfo.vecNzSingleK;
        copyinParams.srcStride = this->tiling_->kAlign - weightSplitInfo.vecNzSingleK;
        copyinParams.dstStride = 0;
    }

    LocalTensor<wType> originWeight = this->originWeightQueue_.template AllocTensor<wType>();
    DataCopy(originWeight, this->wGlobal_[wSrcOffset], copyinParams);
    this->originWeightQueue_.EnQue(originWeight);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType>::WeightCast(WeightSplitInfo
                                                                                               &weightSplitInfo)
{
    LocalTensor<wType> originWeight = this->originWeightQueue_.template DeQue<wType>();
    LocalTensor<half> weight16 = this->weight16Tbuf_.template Get<half>();
    UnaryRepeatParams castSplitParams;
    castSplitParams.srcBlkStride = 1;
    castSplitParams.dstRepStride = 1;
    castSplitParams.srcRepStride = 1;
    if constexpr (bTrans) {
        castSplitParams.dstBlkStride = weightSplitInfo.vecNzSingleN;
        uint32_t loopNum = weightSplitInfo.vecNzSingleK / INT4_OR_INT8_BLOCK_SIZE;
        for (uint32_t i = 0; i < loopNum; i++) {
            Cast(weight16[i * INT4_OR_INT8_BLOCK_SIZE * weightSplitInfo.vecNzSingleN],
                 originWeight[i * INT4_OR_INT8_BLOCK_SIZE * weightSplitInfo.vecNzSingleN], RoundMode::CAST_NONE,
                 INT4_OR_INT8_BLOCK_SIZE, weightSplitInfo.vecNzSingleN, castSplitParams);
        }

    } else {
        castSplitParams.dstBlkStride = weightSplitInfo.vecNzSingleK;
        uint32_t loopNum = weightSplitInfo.vecNzSingleN / INT4_OR_INT8_BLOCK_SIZE;
        uint32_t kLoopNum = weightSplitInfo.vecNzSingleK / MAX_REPEAT_TIMES;
        uint32_t kLoopTail = weightSplitInfo.vecNzSingleK % MAX_REPEAT_TIMES;
        for (uint32_t i = 0; i < loopNum; i++) {
            for (uint32_t j = 0; j < kLoopNum; j++) {
                Cast(weight16[i * INT4_OR_INT8_BLOCK_SIZE * weightSplitInfo.vecNzSingleK +
                              j * MAX_REPEAT_TIMES * BLOCK_CUBE],
                     originWeight[i * INT4_OR_INT8_BLOCK_SIZE * weightSplitInfo.vecNzSingleK +
                                  j * MAX_REPEAT_TIMES * INT4_OR_INT8_BLOCK_SIZE],
                     RoundMode::CAST_NONE, INT4_OR_INT8_BLOCK_SIZE, MAX_REPEAT_TIMES, castSplitParams);
            }
            if (kLoopTail > 0) {
                Cast(weight16[i * INT4_OR_INT8_BLOCK_SIZE * weightSplitInfo.vecNzSingleK +
                              kLoopNum * MAX_REPEAT_TIMES * BLOCK_CUBE],
                     originWeight[i * INT4_OR_INT8_BLOCK_SIZE * weightSplitInfo.vecNzSingleK +
                                  kLoopNum * MAX_REPEAT_TIMES * INT4_OR_INT8_BLOCK_SIZE],
                     RoundMode::CAST_NONE, INT4_OR_INT8_BLOCK_SIZE, kLoopTail, castSplitParams);
            }
        }
    }
    this->originWeightQueue_.FreeTensor(originWeight);

    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        // bf16场景，需要转换成fp32计算
        LocalTensor<float> weight32 = this->weight32Tbuf_.template Get<float>();
        Cast(weight32, weight16, RoundMode::CAST_NONE, weight16.GetSize());
        PipeBarrier<PIPE_V>();
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::ComputePerTensor(LocalTensor<antiQuantCalType> &antiquantWeightTensor)
{
    if constexpr (hasAntiQuantOffset) {
        Adds(antiquantWeightTensor, antiquantWeightTensor, this->offsetValue_, antiquantWeightTensor.GetSize());
        PipeBarrier<PIPE_V>();
    }
    Muls(antiquantWeightTensor, antiquantWeightTensor, this->scaleValue_, antiquantWeightTensor.GetSize());
    PipeBarrier<PIPE_V>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::ComputePerChannelTrans(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                       WeightSplitInfo &weightSplitInfo)
{
    uint32_t loopNum = weightSplitInfo.vecNzSingleK / BLOCK_CUBE;
    uint32_t singleCalNum = weightSplitInfo.vecNzSingleN * BLOCK_CUBE;
    if constexpr (hasAntiQuantOffset) {
        for (uint32_t i = 0; i < loopNum; i++) {
            Add(antiquantWeightTensor[singleCalNum * i], antiquantWeightTensor[singleCalNum * i],
                this->offsetComputeTensor_, singleCalNum);
        }
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t i = 0; i < loopNum; i++) {
        Mul(antiquantWeightTensor[singleCalNum * i], antiquantWeightTensor[singleCalNum * i], this->scaleComputeTensor_,
            singleCalNum);
    }
    PipeBarrier<PIPE_V>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::ComputePerChannelNotTrans(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                          WeightSplitInfo &weightSplitInfo)
{
    uint32_t loopNum = weightSplitInfo.vecNzSingleN / BLOCK_CUBE;
    uint32_t singleCalNum = weightSplitInfo.vecNzSingleK * BLOCK_CUBE;
    if constexpr (hasAntiQuantOffset) {
        for (uint32_t i = 0; i < loopNum; i++) {
            for (uint32_t j = 0; j < weightSplitInfo.vecNzSingleK / BLOCK_CUBE; j++) {
                Add(antiquantWeightTensor[i * singleCalNum + j * this->FRACTAL_SIZE_F16],
                    antiquantWeightTensor[i * singleCalNum + j * this->FRACTAL_SIZE_F16],
                    this->offsetComputeTensor_[i * this->FRACTAL_SIZE_F16], this->FRACTAL_SIZE_F16);
            }
        }
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t i = 0; i < loopNum; i++) {
        for (uint32_t j = 0; j < weightSplitInfo.vecNzSingleK / BLOCK_CUBE; j++) {
            Mul(antiquantWeightTensor[i * singleCalNum + j * this->FRACTAL_SIZE_F16],
                antiquantWeightTensor[i * singleCalNum + j * this->FRACTAL_SIZE_F16],
                this->scaleComputeTensor_[i * this->FRACTAL_SIZE_F16], this->FRACTAL_SIZE_F16);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::ComputePerChannel(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                  WeightSplitInfo &weightSplitInfo)
{
    if constexpr (bTrans) {
        ComputePerChannelTrans(antiquantWeightTensor, weightSplitInfo);
    } else {
        ComputePerChannelNotTrans(antiquantWeightTensor, weightSplitInfo);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::ComputePerGroup(LocalTensor<antiQuantCalType> &antiquantWeightTensor, WeightSplitInfo &weightSplitInfo)
{
    if constexpr (bTrans) {
        if constexpr (IsSameType<antiQuantCalType, float>::value) {
            ComputePerGroupTransF32(antiquantWeightTensor, weightSplitInfo);
        } else {
            ComputePerGroupTransF16(antiquantWeightTensor, weightSplitInfo);
        }
    } else {
        if constexpr (IsSameType<antiQuantCalType, float>::value) {
            ComputePerGroupNotTransF32(antiquantWeightTensor, weightSplitInfo);
        } else {
            ComputePerGroupNotTransF16(antiquantWeightTensor, weightSplitInfo);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::ComputePerGroupTransF32(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                        WeightSplitInfo &weightSplitInfo)
{
    uint32_t groupNum = weightSplitInfo.vecNzSingleK / this->tiling_->groupSize;
    uint32_t repeatTimes = weightSplitInfo.vecNzSingleN / 8;
    uint32_t singleCalNum = weightSplitInfo.vecNzSingleN * this->tiling_->groupSize;
    BinaryRepeatParams mulAddParams;
    mulAddParams.dstBlkStride = 2;
    mulAddParams.src0BlkStride = 2;
    mulAddParams.src1BlkStride = 16;
    mulAddParams.dstRepStride = 16;
    mulAddParams.src0RepStride = 16;
    mulAddParams.src1RepStride = 16 * 8;

    if constexpr (hasAntiQuantOffset) {
        for (uint32_t i = 0; i < groupNum; i++) {
            for (uint32_t j = 0; j < this->tiling_->groupSize / BLOCK_CUBE; j++) {
                Add(antiquantWeightTensor[i * singleCalNum + j * weightSplitInfo.vecNzSingleN * BLOCK_CUBE],
                    antiquantWeightTensor[i * singleCalNum + j * weightSplitInfo.vecNzSingleN * BLOCK_CUBE],
                    this->offsetComputeTensor_[i * FP32_BLOCK_SIZE], this->maskMax_, repeatTimes, mulAddParams);
                Add(antiquantWeightTensor[i * singleCalNum + j * weightSplitInfo.vecNzSingleN * BLOCK_CUBE + 8],
                    antiquantWeightTensor[i * singleCalNum + j * weightSplitInfo.vecNzSingleN * BLOCK_CUBE + 8],
                    this->offsetComputeTensor_[i * FP32_BLOCK_SIZE], this->maskMax_, repeatTimes, mulAddParams);
            }
        }
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t i = 0; i < groupNum; i++) {
        for (uint32_t j = 0; j < this->tiling_->groupSize / BLOCK_CUBE; j++) {
            Mul(antiquantWeightTensor[i * singleCalNum + j * weightSplitInfo.vecNzSingleN * BLOCK_CUBE],
                antiquantWeightTensor[i * singleCalNum + j * weightSplitInfo.vecNzSingleN * BLOCK_CUBE],
                this->scaleComputeTensor_[i * FP32_BLOCK_SIZE], this->maskMax_, repeatTimes, mulAddParams);
            Mul(antiquantWeightTensor[i * singleCalNum + j * weightSplitInfo.vecNzSingleN * BLOCK_CUBE + 8],
                antiquantWeightTensor[i * singleCalNum + j * weightSplitInfo.vecNzSingleN * BLOCK_CUBE + 8],
                this->scaleComputeTensor_[i * FP32_BLOCK_SIZE], this->maskMax_, repeatTimes, mulAddParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::ComputePerGroupTransF16(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                        WeightSplitInfo &weightSplitInfo)
{
    uint32_t groupNum = weightSplitInfo.vecNzSingleK / this->tiling_->groupSize;
    uint32_t repeatTimes = weightSplitInfo.vecNzSingleN / 8;
    uint32_t singleCalNum = weightSplitInfo.vecNzSingleN * this->tiling_->groupSize;
    BinaryRepeatParams mulAddParams;
    mulAddParams.dstBlkStride = 1;
    mulAddParams.src0BlkStride = 1;
    mulAddParams.src1BlkStride = 16;
    mulAddParams.dstRepStride = 8;
    mulAddParams.src0RepStride = 8;
    mulAddParams.src1RepStride = 16 * 8;

    if constexpr (hasAntiQuantOffset) {
        for (uint32_t i = 0; i < groupNum; i++) {
            for (uint32_t j = 0; j < this->tiling_->groupSize / BLOCK_CUBE; j++) {
                Add(antiquantWeightTensor[i * singleCalNum + j * weightSplitInfo.vecNzSingleN * BLOCK_CUBE],
                    antiquantWeightTensor[i * singleCalNum + j * weightSplitInfo.vecNzSingleN * BLOCK_CUBE],
                    this->offsetComputeTensor_[i * FP16_BLOCK_SIZE], this->maskMax_, repeatTimes, mulAddParams);
            }
        }
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t i = 0; i < groupNum; i++) {
        for (uint32_t j = 0; j < this->tiling_->groupSize / BLOCK_CUBE; j++) {
            Mul(antiquantWeightTensor[i * singleCalNum + j * weightSplitInfo.vecNzSingleN * BLOCK_CUBE],
                antiquantWeightTensor[i * singleCalNum + j * weightSplitInfo.vecNzSingleN * BLOCK_CUBE],
                this->scaleComputeTensor_[i * FP16_BLOCK_SIZE], this->maskMax_, repeatTimes, mulAddParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::ComputePerGroupNotTransF32(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                           WeightSplitInfo &weightSplitInfo)
{
    uint32_t loopNum = weightSplitInfo.vecNzSingleN / BLOCK_CUBE;
    uint32_t singleCalNum = weightSplitInfo.vecNzSingleK * BLOCK_CUBE;
    BinaryRepeatParams mulAddParams;
    mulAddParams.src1BlkStride = 0;
    mulAddParams.src1RepStride = 0;
    mulAddParams.dstBlkStride = 2;
    mulAddParams.src0BlkStride = 2;
    mulAddParams.dstRepStride = 16;
    mulAddParams.src0RepStride = 16;
    uint32_t groupNum = weightSplitInfo.vecNzSingleK / this->tiling_->groupSize;
    if constexpr (hasAntiQuantOffset) {
        for (uint32_t i = 0; i < loopNum; i++) {
            for (uint32_t j = 0; j < groupNum; j++) {
                Add(antiquantWeightTensor[i * singleCalNum + j * this->tiling_->groupSize * BLOCK_CUBE],
                    antiquantWeightTensor[i * singleCalNum + j * this->tiling_->groupSize * BLOCK_CUBE],
                    this->offsetComputeTensor_[j * weightSplitInfo.vecNzSingleN + i * BLOCK_CUBE], this->maskMax_,
                    this->tiling_->groupSize / 8, mulAddParams);
                Add(antiquantWeightTensor[i * singleCalNum + j * this->tiling_->groupSize * BLOCK_CUBE + 8],
                    antiquantWeightTensor[i * singleCalNum + j * this->tiling_->groupSize * BLOCK_CUBE + 8],
                    this->offsetComputeTensor_[j * weightSplitInfo.vecNzSingleN + i * BLOCK_CUBE + 8], this->maskMax_,
                    this->tiling_->groupSize / 8, mulAddParams);
            }
        }
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t i = 0; i < loopNum; i++) {
        for (uint32_t j = 0; j < groupNum; j++) {
            Mul(antiquantWeightTensor[i * singleCalNum + j * this->tiling_->groupSize * BLOCK_CUBE],
                antiquantWeightTensor[i * singleCalNum + j * this->tiling_->groupSize * BLOCK_CUBE],
                this->scaleComputeTensor_[j * weightSplitInfo.vecNzSingleN + i * BLOCK_CUBE], this->maskMax_,
                this->tiling_->groupSize / 8, mulAddParams);
            Mul(antiquantWeightTensor[i * singleCalNum + j * this->tiling_->groupSize * BLOCK_CUBE + 8],
                antiquantWeightTensor[i * singleCalNum + j * this->tiling_->groupSize * BLOCK_CUBE + 8],
                this->scaleComputeTensor_[j * weightSplitInfo.vecNzSingleN + i * BLOCK_CUBE + 8], this->maskMax_,
                this->tiling_->groupSize / 8, mulAddParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::ComputePerGroupNotTransF16(LocalTensor<antiQuantCalType> &antiquantWeightTensor,
                                           WeightSplitInfo &weightSplitInfo)
{
    uint32_t loopNum = weightSplitInfo.vecNzSingleN / BLOCK_CUBE;
    uint32_t singleCalNum = weightSplitInfo.vecNzSingleK * BLOCK_CUBE;
    BinaryRepeatParams mulAddParams;
    mulAddParams.src1BlkStride = 0;
    mulAddParams.src1RepStride = 0;
    mulAddParams.dstBlkStride = 1;
    mulAddParams.src0BlkStride = 1;
    mulAddParams.dstRepStride = 8;
    mulAddParams.src0RepStride = 8;
    uint32_t groupNum = weightSplitInfo.vecNzSingleK / this->tiling_->groupSize;
    if constexpr (hasAntiQuantOffset) {
        for (uint32_t i = 0; i < loopNum; i++) {
            for (uint32_t j = 0; j < groupNum; j++) {
                Add(antiquantWeightTensor[i * singleCalNum + j * this->tiling_->groupSize * BLOCK_CUBE],
                    antiquantWeightTensor[i * singleCalNum + j * this->tiling_->groupSize * BLOCK_CUBE],
                    this->offsetComputeTensor_[j * weightSplitInfo.vecNzSingleN + i * BLOCK_CUBE], this->maskMax_,
                    this->tiling_->groupSize * BLOCK_CUBE / this->maskMax_, mulAddParams);
            }
        }
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t i = 0; i < loopNum; i++) {
        for (uint32_t j = 0; j < groupNum; j++) {
            Mul(antiquantWeightTensor[i * singleCalNum + j * this->tiling_->groupSize * BLOCK_CUBE],
                antiquantWeightTensor[i * singleCalNum + j * this->tiling_->groupSize * BLOCK_CUBE],
                this->scaleComputeTensor_[j * weightSplitInfo.vecNzSingleN + i * BLOCK_CUBE], this->maskMax_,
                this->tiling_->groupSize * BLOCK_CUBE / this->maskMax_, mulAddParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType>::AntiQuantCompute(WeightSplitInfo
                                                                                                     &weightSplitInfo)
{
    LocalTensor<antiQuantCalType> antiquantWeightTensor;
    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        antiquantWeightTensor = this->weight32Tbuf_.template Get<float>();
    } else {
        antiquantWeightTensor = this->weight16Tbuf_.template Get<half>();
    }
    if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) {
        ComputePerTensor(antiquantWeightTensor);
    } else if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL) {
        ComputePerChannel(antiquantWeightTensor, weightSplitInfo);
    } else {
        ComputePerGroup(antiquantWeightTensor, weightSplitInfo);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType>::ProcessVector()
{
    uint64_t nBaseSize = this->tiling_->matmulTiling.singleCoreN * this->tiling_->cubeBlockDimN;
    uint64_t nRealSize = nBaseSize;

    // 初始化weight的切分信息
    WeightSplitInfo weightSplitInfo;
    weightSplitInfo.vecNzNRealSize = nRealSize;
    for (int32_t cubeNLoopIdx = 0; cubeNLoopIdx < this->tiling_->cubeSingleNLoop; cubeNLoopIdx++) {
        uint64_t nBaseOffset = cubeNLoopIdx * nBaseSize;
        uint64_t nLoopLimit = this->tiling_->vecSingleNLoop;
        if (unlikely(cubeNLoopIdx + 1 == this->tiling_->cubeSingleNLoop)) {
            nLoopLimit = this->tiling_->vecSingleNTailLoop;
            nRealSize = this->tiling_->nSize - nBaseOffset;
            weightSplitInfo.vecNzNRealSize = this->tiling_->nAlign - nBaseOffset;
        }
        // 为节约workspace空间，weight缓存空间整体分成WEIGHT_CACHE_COUNT块轮询使用，需要根据cubeNLoopIdx确定当前使用哪块地址
        this->weightCacheIdx_ = cubeNLoopIdx % this->WEIGHT_CACHE_COUNT;

        AntiquantWeight(cubeNLoopIdx, nBaseOffset, nRealSize, nLoopLimit, weightSplitInfo);

        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(this->SYNC_AIV_ONLY_ALL_FLAG);
    }
    NotifyCube();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType>::BrcbAntiquantParams(WeightSplitInfo &
                                                                                                        weightSplitInfo)
{
    PipeBarrier<PIPE_V>();
    if constexpr (!bTrans || !IsSameType<antiQuantCalType, float>::value || antiQuantType == Mc2QuantType::PER_GROUP) {
        return;
    }
    if constexpr (hasAntiQuantOffset) {
        DataCopy(this->offsetComputeTensor_[ONE_BLK_SIZE / sizeof(antiQuantCalType)], this->offsetComputeTensor_,
                 {static_cast<uint16_t>(weightSplitInfo.vecNzSingleN), 1, 1, 1});
    }
    DataCopy(this->scaleComputeTensor_[ONE_BLK_SIZE / sizeof(antiQuantCalType)], this->scaleComputeTensor_,
             {static_cast<uint16_t>(weightSplitInfo.vecNzSingleN), 1, 1, 1});

    PipeBarrier<PIPE_V>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::AntiquantWeight(uint64_t cubeNLoopIdx, uint64_t nBaseOffset, uint64_t nRealSize, uint64_t nLoopLimit,
                                WeightSplitInfo &weightSplitInfo)
{
    // 求解当次循环n方向的起点和终点
    uint64_t loopLimit = nLoopLimit * this->tiling_->vecSingleKLoop;
    uint64_t singleCoreLoop = this->CeilDiv(loopLimit, this->tiling_->vecBlockDimN);
    uint64_t singleCoreLoopStart = this->curBlockIdx_ * singleCoreLoop;
    uint64_t singleCoreLoopLimit = singleCoreLoopStart + singleCoreLoop;
    if (singleCoreLoopLimit > loopLimit) {
        singleCoreLoopLimit = loopLimit;
    }
    weightSplitInfo.vecSingleN = this->tiling_->vecSingleN;
    weightSplitInfo.vecNzSingleN = this->tiling_->vecSingleN;
    int64_t lastNLoopIdx = -1;
    for (uint64_t loopIdx = singleCoreLoopStart; loopIdx < singleCoreLoopLimit; loopIdx++) {
        uint64_t nLoopIdx = loopIdx / this->tiling_->vecSingleKLoop;
        uint64_t kLoopIdx = loopIdx % this->tiling_->vecSingleKLoop;
        ComputeWeightOffsetInfo(nLoopIdx, nBaseOffset, kLoopIdx, weightSplitInfo);
        ComputeWeightSplitInfo(nLoopIdx, nLoopLimit, nRealSize, kLoopIdx, weightSplitInfo);

        if (likely(this->vecKDimIdx_ < this->tiling_->vecBlockDimK)) {
            if ((antiQuantType == Mc2QuantType::PER_CHANNEL && lastNLoopIdx != nLoopIdx) ||
                antiQuantType == Mc2QuantType::PER_GROUP) {
                this->CopyInAntiquantParams(weightSplitInfo);
                BrcbAntiquantParams(weightSplitInfo);
                lastNLoopIdx = nLoopIdx;
            }
            WeightCopyIn(weightSplitInfo);
            WeightCast(weightSplitInfo);
            AntiQuantCompute(weightSplitInfo);
        }
        if (likely(cubeNLoopIdx > 0 && loopIdx == singleCoreLoopStart)) {
            NotifyCube();
        }
        if (likely(cubeNLoopIdx > 1 && loopIdx == singleCoreLoopStart)) {
            // vector计算需要领先cube一拍
            this->WaitForCube();
        }
        if (likely(this->vecKDimIdx_ < this->tiling_->vecBlockDimK)) {
            WeightCopyOut(weightSplitInfo);
        }
    }

    // 尾核场景，此时不会进入上面的循环，因此需要额外补充同步
    if (unlikely(cubeNLoopIdx > 0 && singleCoreLoopStart >= singleCoreLoopLimit)) {
        NotifyCube();
    }
    if (unlikely(cubeNLoopIdx > 1 && singleCoreLoopStart >= singleCoreLoopLimit)) {
        this->WaitForCube();
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType>::WeightCopyOut(WeightSplitInfo
                                                                                                  &weightSplitInfo)
{
    LocalTensor<xType> weightOutput = this->weightOutputQueue_.template AllocTensor<xType>();
    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        LocalTensor<float> weight32 = this->weight32Tbuf_.template Get<float>();
        Cast(weightOutput, weight32, RoundMode::CAST_RINT, weight32.GetSize());
    } else {
        LocalTensor<half> weight16 = this->weight16Tbuf_.template Get<half>();
        DataCopy(weightOutput, weight16, weight16.GetSize());
    }
    DataCopyParams copyoutParams;

    this->weightOutputQueue_.EnQue(weightOutput);
    weightOutput = this->weightOutputQueue_.template DeQue<xType>();
    uint64_t wDstOffset = this->weightCacheIdx_ * this->weightCacheSizeAlign_;
    if constexpr (bTrans) {
        wDstOffset += weightSplitInfo.kOffset * this->cubeBaseN_ + weightSplitInfo.splitNOffset * BLOCK_CUBE;
        copyoutParams.blockCount = weightSplitInfo.vecNzSingleK / BLOCK_CUBE;
        copyoutParams.blockLen = weightSplitInfo.vecNzSingleN;
        copyoutParams.dstStride = this->cubeBaseN_ - weightSplitInfo.vecNzSingleN;
        copyoutParams.srcStride = 0;
    } else {
        wDstOffset += weightSplitInfo.splitNOffset * this->tiling_->kAlign + weightSplitInfo.kOffset * BLOCK_CUBE;
        copyoutParams.blockCount = weightSplitInfo.vecNzSingleN / BLOCK_CUBE;
        copyoutParams.blockLen = weightSplitInfo.vecNzSingleK;
        copyoutParams.dstStride = this->tiling_->kAlign - weightSplitInfo.vecNzSingleK;
        copyoutParams.srcStride = 0;
    }
    DataCopy(this->weightCache_[wDstOffset], weightOutput, copyoutParams);
    this->weightOutputQueue_.FreeTensor(weightOutput);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::ComputeWeightOffsetInfo(uint64_t nLoopIdx, uint64_t nBaseOffset, uint64_t kLoopIdx,
                                        WeightSplitInfo &weightSplitInfo)
{
    weightSplitInfo.splitNOffset = nLoopIdx * this->tiling_->vecSingleN;
    weightSplitInfo.originNOffset = nBaseOffset + weightSplitInfo.splitNOffset;
    weightSplitInfo.kOffset = kLoopIdx * this->tiling_->vecSingleK;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset,
    quantType>::ComputeWeightSplitInfo(uint64_t nLoopIdx, uint64_t nLoopLimit, uint64_t nRealSize, uint64_t kLoopIdx,
                                       WeightSplitInfo &weightSplitInfo)
{
    if (unlikely(nLoopIdx == nLoopLimit - 1)) {
        // 当计算到最后一块时，需要重新计算尾块实际的n是多少
        weightSplitInfo.vecSingleN = nRealSize - weightSplitInfo.splitNOffset;
        weightSplitInfo.vecNzSingleN = weightSplitInfo.vecNzNRealSize - weightSplitInfo.splitNOffset;
    }
    weightSplitInfo.vecSingleK = this->tiling_->vecSingleK;
    weightSplitInfo.vecNzSingleK = this->tiling_->vecSingleK;
    // 当计算到最后一块时，需要重新计算尾块实际的k是多少
    if (unlikely(kLoopIdx == this->tiling_->vecSingleKLoop - 1)) {
        weightSplitInfo.vecSingleK = this->tiling_->kAlign - weightSplitInfo.kOffset;
        weightSplitInfo.vecNzSingleK = this->tiling_->kAlign - weightSplitInfo.kOffset;
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType>::NotifyCube()
{
    this->WaitFlagDevLocal(this->SYNC_AIV_ONLY_ALL_FLAG);
    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(this->SYNC_AIV_AIC_FLAG);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType>::ProcessCube()
{
    uint64_t mOffset = this->cubeMDimIdx_ * this->tiling_->matmulTiling.singleCoreM;
    uint64_t aOffset = mOffset;

    if constexpr (!aTrans) {
        aOffset *= this->tiling_->matmulTiling.Ka;
    }
    uint64_t cubeNOffset = this->cubeNDimIdx_ * this->tiling_->matmulTiling.singleCoreN;
    uint64_t bOffset;
    if constexpr (bTrans) {
        bOffset = cubeNOffset * 16;
    } else {
        bOffset = cubeNOffset * this->tiling_->kAlign;
    }

    mmObj.SetOrgShape(this->tiling_->matmulTiling.M, this->cubeBaseN_, this->tiling_->matmulTiling.Ka,
                      this->tiling_->kAlign, this->tiling_->matmulTiling.N);

    for (int32_t cubeNLoopIdx = 0; cubeNLoopIdx < this->tiling_->cubeSingleNLoop; cubeNLoopIdx++) {
        this->WaitForVector();
        uint64_t nOffset = cubeNOffset + cubeNLoopIdx * this->cubeBaseN_;
        uint64_t cOffset = mOffset * this->tiling_->matmulTiling.N + nOffset;
        if (likely(nOffset < this->tiling_->nSize &&
                   this->curBlockIdx_ < this->tiling_->cubeBlockDimM * this->tiling_->cubeBlockDimN)) {
            SetMatmulParams(cubeNLoopIdx, aOffset, bOffset, nOffset);
            mmObj.IterateAll(this->yGlobal_[cOffset]);
            mmObj.End();
        }
        if (cubeNLoopIdx > 0) {
            this->WaitFlagDevLocal(this->SYNC_AIC_ONLY_ALL_FLAG);
        }

        if (cubeNLoopIdx + 1 < this->tiling_->cubeSingleNLoop) {
            CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(this->SYNC_AIC_ONLY_ALL_FLAG);
        }

        if (cubeNLoopIdx + 2 < this->tiling_->cubeSingleNLoop) {
            this->NotifyVector();
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType>::SetMatmulParams(int32_t cubeNLoopIdx,
                                                                                                uint64_t aOffset,
                                                                                                uint64_t bOffset,
                                                                                                uint64_t nOffset)
{
    mmObj.SetTensorA(this->xGlobal_[aOffset], aTrans);
    mmObj.SetTensorB(
        this->weightCache_[(cubeNLoopIdx % this->WEIGHT_CACHE_COUNT) * this->weightCacheSizeAlign_ + bOffset], bTrans);

    if (this->biasFlag_) {
        mmObj.SetBias(this->biasGlobal_[nOffset]);
    }

    int64_t mmSingleN = this->tiling_->matmulTiling.singleCoreN;
    int64_t mmSingleM = this->tiling_->matmulTiling.singleCoreM;

    if (unlikely(nOffset + mmSingleN > this->tiling_->nSize)) {
        mmSingleN = this->tiling_->nSize - nOffset;
    }

    if (unlikely(this->cubeMDimIdx_ == this->tiling_->cubeBlockDimM - 1)) {
        mmSingleM = this->tiling_->cubeTailM;
    }

    mmObj.SetTail(mmSingleM, mmSingleN);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomWeightNzKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType>::Process()
{
    if ASCEND_IS_AIV {
        ProcessVector();
    } else {
        ProcessCube();
    }
}
} // namespace Mc2WeightQuantBatchMatmulV2
#endif // WEIGHT_QUANT_BATCHMATMUL_V2_CUSTOM_WEITHG_NZ_H
