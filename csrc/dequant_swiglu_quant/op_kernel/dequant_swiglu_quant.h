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
 * \file dequant_swiglu_quant.h
 * \brief
 */

#ifndef DEQUANT_SWIGLU_QUANT_H
#define DEQUANT_SWIGLU_QUANT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

#define TEMPLATE_DSQ_DECLARE template <typename TBias, typename TQuantScale, typename TGroup, typename TXGm>
#define TEMPLATE_DSQ_ARGS TBias, TQuantScale, TGroup, TXGm

namespace DequantSwigluQuantOps {
using namespace AscendC;
constexpr static int64_t DB_BUFFER = 1;
constexpr static int64_t BLOCK_SIZE = 32;
constexpr static int64_t BLOCK_ELEM = BLOCK_SIZE / sizeof(float);
constexpr static int64_t MASK_NUM_T32 = 256 / sizeof(float);
constexpr static int64_t MASK_BLK_STRIDE = 8;
constexpr static int64_t SWI_FACTOR = 2;
constexpr static float DYNAMIC_QUANT_FACTOR = 1.0 / 127.0;

TEMPLATE_DSQ_DECLARE
class DequantSwigluQuantBase
{
public:
    static constexpr bool hasGroupIndex_ = !IsSameType<TGroup, float>::value;
    __aicore__ inline DequantSwigluQuantBase(TPipe* pipe)
    {
        pipe_ = pipe;
    };

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale, GM_ADDR bias, GM_ADDR quantScale, GM_ADDR quantOffset,
        GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale, const DequantSwigluQuantBaseTilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ComputeReduceMax(const LocalTensor<float>& tempRes, int32_t calCount);
    __aicore__ inline void ProcessSingleGroup(int64_t groupIdx, int64_t realCount, int64_t globalOffset);
    __aicore__ inline void ProcessSingleGroupPerCore(int64_t groupIdx, int64_t dimxCore, int64_t dimxCoreOffset);
    __aicore__ inline void CreateOffsetLocalTensor(uint32_t tensorLen, int swigluMode);
    __aicore__ inline void SwiGluGate(
        int32_t proDimsx, const LocalTensor<float>& xLocalF32);
    __aicore__ inline void DynamicQuant(
        const LocalTensor<float>& tmpUbF32Act, const LocalTensor<float>& tmpUbF32Gate,
        const LocalTensor<float>& inScaleLocal, uint32_t proDimsx);
    __aicore__ inline void StaticQuant(
        const LocalTensor<float>& tmpUbF32Act, const LocalTensor<float>& tmpUbF32Gate,
        const LocalTensor<float>& inScaleLocal, uint32_t proDimsx);
    __aicore__ inline void CopyInWeightScale(int64_t groupIdx);
    __aicore__ inline void CopyInQuantScale(int64_t groupIdx);
    __aicore__ inline void CopyInBias(int64_t groupIdx);
    __aicore__ inline void ParamDequeAndCast();
    __aicore__ inline void CopyInXAct(int32_t proDimsx, int64_t xDimxOffset);
    __aicore__ inline void Compute(int32_t proDimsx);
    __aicore__ inline void ComputeDequant(int32_t proDimsx);
    __aicore__ inline void ComputeSwiGLU(int32_t proDimsx);
    __aicore__ inline void ComputeQuant(int32_t proDimsx);
    __aicore__ inline void CopyOut(int32_t proDimsx, int64_t xDimxOffset);
    __aicore__ inline void ParamFree();
    __aicore__ inline void CastFloatToInt8(
        const LocalTensor<float>& tmpUbF32Act, const LocalTensor<float>& tmpUbF32Gate, uint32_t proDimsx, LocalTensor<int8_t>& yOut);
    template<typename T>
    __aicore__ inline void CopyReshape(LocalTensor<T>& dstTensor, LocalTensor<T>& oriTensor, uint32_t rowNum, uint32_t colNum, CopyRepeatParams param);

protected:
    /* global memory address */
    // input global mem
    GlobalTensor<TXGm> xGm_;
    GlobalTensor<float> weightScaleGm_;
    GlobalTensor<float> activationScaleGm_;
    GlobalTensor<TBias> biasGm_;
    GlobalTensor<TQuantScale> quantScaleGm_;
    GlobalTensor<TQuantScale> quantOffsetGm_;
    GlobalTensor<TGroup> groupIndexGm_;

    // output global mem
    GlobalTensor<int8_t> yGm_;
    GlobalTensor<float> scaleGm_;

    /* ub memory tensor */
    LocalTensor<float> weightScaleLocal_;
    LocalTensor<float> inScaleLocal_; // quant scale and quant offset
    LocalTensor<TBias> biasLocal_;
    LocalTensor<float> biasLocalF32_;
    LocalTensor<uint32_t> xOffsetLocalU32_; // offset for gather

    /* ascendc variable */
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, DB_BUFFER> xActQueue_;
    TQue<QuePosition::VECIN, 1> inScaleQueue_;
    TQue<QuePosition::VECIN, 1> weightScaleQueue_;
    TQue<QuePosition::VECIN, 1> biasQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    TBuf<TPosition::VECCALC> tmpBuf1_;
    TBuf<TPosition::VECCALC> tmpBuf2_; // only use in swigluMode == 1

    uint32_t blockIdx_ = GetBlockIdx();
    int64_t realDimx_ = 0;
    int64_t groupOffset_ = 0;
    float quantScale_ = 1.0f;
    float quantOffset_ = 1.0f;

    uint32_t UbSingleOutSize_ = 0;
    uint32_t TBufActSclInOfs_ = 0;
    uint32_t TBufXLocalInOfs_ = 0;

    int32_t actOffset_;
    int32_t gateOffset_;

    const DequantSwigluQuantBaseTilingData* tl_ = nullptr;
};
// 公共函数实现

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::Init(
    GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale, GM_ADDR bias, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale, const DequantSwigluQuantBaseTilingData* tilingData)
{
    tl_ = tilingData;
    xGm_.SetGlobalBuffer((__gm__ TXGm*)x);
    weightScaleGm_.SetGlobalBuffer((__gm__ float*)weightScale);
    activationScaleGm_.SetGlobalBuffer((__gm__ float*)activationScale);
    biasGm_.SetGlobalBuffer((__gm__ TBias*)bias);
    quantScaleGm_.SetGlobalBuffer((__gm__ TQuantScale*)quantScale);
    if constexpr (hasGroupIndex_) {
        groupIndexGm_.SetGlobalBuffer((__gm__ TGroup*)groupIndex);
    }
    // static quant
    if (tl_->quantMode == 0) {
        quantOffsetGm_.SetGlobalBuffer((__gm__ TQuantScale*)quantOffset);
    }
    yGm_.SetGlobalBuffer((__gm__ int8_t*)y);
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale);

    UbSingleOutSize_ = static_cast<uint32_t>(tl_->UbFactorDimx * tl_->outDimy);
    TBufActSclInOfs_ = static_cast<uint32_t>(tl_->UbFactorDimx * tl_->inDimy);
#if (ORIG_DTYPE_X == DT_BF16)
    TBufXLocalInOfs_ = TBufActSclInOfs_;
#endif

    // swiglu offset
    actOffset_ = tl_->actRight * tl_->UbFactorDimy;
    gateOffset_ = tl_->UbFactorDimy - actOffset_;

    // init buffer
    pipe_->InitBuffer(
        xActQueue_, DB_BUFFER, (UbSingleOutSize_ * SWI_FACTOR + tl_->UbFactorDimx * BLOCK_ELEM) * sizeof(int32_t));
    pipe_->InitBuffer(weightScaleQueue_, 1, tl_->inDimy * sizeof(float));

    if (tl_->quantMode == 0) {
        pipe_->InitBuffer(inScaleQueue_, 1, tl_->outDimy * SWI_FACTOR * sizeof(float));
    } else {
        pipe_->InitBuffer(inScaleQueue_, 1, tl_->outDimy * sizeof(float));
    }

    if (tl_->hasBias == 1) {
        pipe_->InitBuffer(biasQueue_, 1, tl_->inDimy * sizeof(float));
    }
    pipe_->InitBuffer(outQueue_, 1, UbSingleOutSize_ * sizeof(int8_t) + tl_->UbFactorDimx * sizeof(float) + BLOCK_SIZE);

    pipe_->InitBuffer(tmpBuf1_, UbSingleOutSize_ * SWI_FACTOR * sizeof(float));
    if (tl_->swigluMode == 1) {
        pipe_->InitBuffer(
            tmpBuf2_,
            UbSingleOutSize_ * sizeof(int32_t) + UbSingleOutSize_ * sizeof(uint8_t)); // for gather offset and clamp
    }
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::Process()
{
    if constexpr (!hasGroupIndex_) {
        realDimx_ = tl_->inDimx;
        // do protect realDimx_ < 0, ignore this group
        realDimx_ = (realDimx_ < 0) ? 0 : realDimx_;
        ProcessSingleGroup(0, realDimx_, 0);
        return;
    }

    CreateOffsetLocalTensor(UbSingleOutSize_, tl_->swigluMode);

    groupOffset_ = 0;
    for (int32_t groupIdx = 0; groupIdx < tl_->inGroupNum; ++groupIdx) {
        int64_t realGroupIdx =
            tl_->speGroupType == 0 ? static_cast<int64_t>(groupIdx) : static_cast<int64_t>(groupIndexGm_(groupIdx * 2));
        realDimx_ = tl_->speGroupType == 0 ? static_cast<int64_t>(groupIndexGm_(groupIdx)) :
                                             static_cast<int64_t>(groupIndexGm_(groupIdx * 2 + 1));
        // do protect realDimx_ < 0, ignore this group
        realDimx_ = (realDimx_ < 0) ? 0 : realDimx_;
        if (realDimx_ > 0 && groupOffset_ < tl_->inDimx) {
            ProcessSingleGroup(realGroupIdx, realDimx_, groupOffset_);
            groupOffset_ += realDimx_;
        }
        // speGroupindex场景下出现异常值(realDimx_ < 0), 退出计算
        if (tl_->speGroupType == 1 && realDimx_ <= 0) {
            break;
        }
    }
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::ProcessSingleGroup(
    int64_t groupIdx, int64_t realCount, int64_t globalOffset)
{
    // do block tiling again
    int32_t blockDimxFactor = (realCount + tl_->maxCoreNum - 1) / tl_->maxCoreNum;
    int32_t realCoreDim = (realCount + blockDimxFactor - 1) / blockDimxFactor;

    if (blockIdx_ < realCoreDim) {
        int32_t blockDimxTailFactor = realCount - blockDimxFactor * (realCoreDim - 1);
        int32_t dimxCore = blockIdx_ == (realCoreDim - 1) ? blockDimxTailFactor : blockDimxFactor;
        int64_t coreDimxOffset = blockDimxFactor * blockIdx_ + globalOffset;
        ProcessSingleGroupPerCore(static_cast<int64_t>(groupIdx), static_cast<int64_t>(dimxCore), coreDimxOffset);
    }
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::CopyInWeightScale(int64_t groupIdx)
{
    // copy weight scale [1, 2H] offset:0
    DataCopyPadParams padParams{false, 0, 0, 0};
    LocalTensor<float> weightScaleLocal = weightScaleQueue_.AllocTensor<float>();
    DataCopyParams dataCopyWeightScaleParams;
    dataCopyWeightScaleParams.blockCount = 1;
    dataCopyWeightScaleParams.blockLen = tl_->inDimy * sizeof(float);
    dataCopyWeightScaleParams.srcStride = 0;
    dataCopyWeightScaleParams.dstStride = 0;
    if constexpr (std::is_same_v<TXGm, int32_t>) {
        DataCopyPad(weightScaleLocal, weightScaleGm_[groupIdx * tl_->inDimy], dataCopyWeightScaleParams, padParams);
    }
    weightScaleQueue_.EnQue(weightScaleLocal);
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::CopyInQuantScale(int64_t groupIdx)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    // copy static quant scale
    LocalTensor<float> inScaleLocal = inScaleQueue_.AllocTensor<float>();
    if (tl_->quantIsOne) {
        if constexpr (IsSameType<TQuantScale, bfloat16_t>::value) {
            this->quantScale_ = 1 / ToFloat(this->quantScaleGm_.GetValue(groupIdx));
            this->quantOffset_ = ToFloat(this->quantOffsetGm_.GetValue(groupIdx));
        } else if constexpr (IsSameType<TQuantScale, half>::value) {
            this->quantScale_ = 1 / static_cast<float>(this->quantScaleGm_.GetValue(groupIdx));
            this->quantOffset_ = static_cast<float>(this->quantOffsetGm_.GetValue(groupIdx));
        } else {
            this->quantScale_ = 1 / this->quantScaleGm_.GetValue(groupIdx);
            this->quantOffset_ = this->quantOffsetGm_.GetValue(groupIdx);
        }
    }

    // copy dynamic quant scale [1, H] offset:tl_->inDimy
    if (tl_->needSmoothScale == 1 && !tl_->quantIsOne) {
        DataCopyParams dataCopyQuantScaleParams;
        dataCopyQuantScaleParams.blockCount = 1;
        dataCopyQuantScaleParams.blockLen = tl_->outDimy * sizeof(TQuantScale);
        dataCopyQuantScaleParams.srcStride = 0;
        dataCopyQuantScaleParams.dstStride = 0;
        if constexpr (std::is_same_v<TQuantScale, float>) {
            DataCopyPad(inScaleLocal, quantScaleGm_[groupIdx * tl_->outDimy], dataCopyQuantScaleParams, padParams);
            if (tl_->quantMode == 0) {
                DataCopyPad(
                    inScaleLocal[tl_->outDimy], quantOffsetGm_[groupIdx * tl_->outDimy], dataCopyQuantScaleParams,
                    padParams);
            }

        } else {
            LocalTensor<TQuantScale> quantScaleLocalT16 = inScaleLocal.template ReinterpretCast<TQuantScale>();
            DataCopyPad(
                quantScaleLocalT16[tl_->outDimy], quantScaleGm_[groupIdx * tl_->outDimy], dataCopyQuantScaleParams,
                padParams);
            if (tl_->quantMode == 0) {
                DataCopyPad(
                    quantScaleLocalT16[tl_->outDimy + tl_->inDimy], quantOffsetGm_[groupIdx * tl_->outDimy],
                    dataCopyQuantScaleParams, padParams);
            }
        }
    }
    inScaleQueue_.EnQue(inScaleLocal);
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::CopyInBias(int64_t groupIdx)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    if constexpr (std::is_same_v<TXGm, int32_t>) {
        if (tl_->hasBias == 1) {
            biasLocal_ = biasQueue_.AllocTensor<TBias>();
            DataCopyParams dataCopyBiasParams;
            dataCopyBiasParams.blockCount = 1;
            dataCopyBiasParams.blockLen = tl_->inDimy * sizeof(TBias);
            dataCopyBiasParams.srcStride = 0;
            dataCopyBiasParams.dstStride = 0;
            if constexpr (std::is_same_v<TBias, float> || std::is_same_v<TBias, int32_t>) {
                DataCopyPad(biasLocal_, biasGm_[groupIdx * tl_->inDimy], dataCopyBiasParams, padParams);
            } else {
                DataCopyPad(biasLocal_[tl_->inDimy], biasGm_[groupIdx * tl_->inDimy], dataCopyBiasParams, padParams);
            }
            biasQueue_.EnQue(biasLocal_);
        }
    }
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::CopyInXAct(int32_t proDimsx, int64_t xDimxOffset)
{
    // copyin x and Act scale
    DataCopyPadParams padParams{false, 0, 0, 0};
    LocalTensor<TXGm> xActLocal = xActQueue_.AllocTensor<TXGm>();
    DataCopyParams dataCopyXParams;
    dataCopyXParams.blockCount = proDimsx;
    dataCopyXParams.blockLen = tl_->inDimy * sizeof(TXGm);
    dataCopyXParams.srcStride = 0;
    dataCopyXParams.dstStride = 0;
    DataCopyPad(xActLocal[TBufXLocalInOfs_], xGm_[xDimxOffset * tl_->inDimy], dataCopyXParams, padParams);

    // copy act scale: [proDimsx,8] offset:tl_->UbFactorDimx * tl_->inDimy = TBufActSclInOfs_
    DataCopyParams dataCopyActScaleParams;
    dataCopyActScaleParams.blockCount = proDimsx;
    dataCopyActScaleParams.blockLen = sizeof(float);
    dataCopyActScaleParams.srcStride = 0;
    dataCopyActScaleParams.dstStride = 0;
    LocalTensor<float> xActLocalF32 = xActLocal.template ReinterpretCast<float>();
    if (std::is_same_v<TXGm, int32_t> && !tl_->activationScaleIsEmpty) {
        DataCopyPad(xActLocalF32[TBufActSclInOfs_], activationScaleGm_[xDimxOffset], dataCopyActScaleParams, padParams);
    }
    xActQueue_.EnQue(xActLocal);
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::ComputeDequant(int32_t proDimsx)
{
    LocalTensor<TXGm> xActLocal = xActQueue_.DeQue<TXGm>();
    LocalTensor<float> xActLocalF32 = xActLocal.template ReinterpretCast<float>();
    LocalTensor<float> xLocalF32 = xActLocalF32;
    LocalTensor<float> activationScaleLocal = xActLocalF32[TBufActSclInOfs_];
    LocalTensor<float> tmpUbF32 = tmpBuf1_.AllocTensor<float>(); // weight scale FP32
    LocalTensor<int32_t> tmpUbI32 = tmpUbF32.template ReinterpretCast<int32_t>();

    if constexpr (std::is_same_v<TXGm, int32_t>) {
        if constexpr (std::is_same_v<TBias, int32_t>){
            // Copy bias: [1,2H] -> [proDimsx,2H]
            // params: dstStride: 1, srcStride: 1, dstRepStride: tl_->UbFactorDimy * 2 / 8, srcRepStride: 0
            CopyReshape<int32_t>(tmpUbI32, biasLocal_, proDimsx, tl_->UbFactorDimy * SWI_FACTOR,
                {1, 1, static_cast<uint16_t>((tl_->UbFactorDimy * SWI_FACTOR) / BLOCK_ELEM), 0});
            PipeBarrier<PIPE_V>();
            Add(xActLocal, xActLocal, tmpUbI32, proDimsx * tl_->inDimy);
            PipeBarrier<PIPE_V>();
        }

        // Copy weight scale: [1,2H] -> [proDimsx,2H]
        // params: dstStride: 1, srcStride: 1, dstRepStride: tl_->UbFactorDimy * 2 / 8, srcRepStride: 0
        CopyReshape<float>(tmpUbF32, weightScaleLocal_, proDimsx, tl_->UbFactorDimy * SWI_FACTOR,
            {1, 1, static_cast<uint16_t>((tl_->UbFactorDimy * SWI_FACTOR) / BLOCK_ELEM), 0});
    }

    // x 为 bf16时
    Cast(xLocalF32, xActLocal[TBufXLocalInOfs_], RoundMode::CAST_NONE, SWI_FACTOR * proDimsx * tl_->UbFactorDimy);
    PipeBarrier<PIPE_V>();
    if constexpr (std::is_same_v<TXGm, int32_t>) {
        // Calc dequant: xLocalF32 = weightScaleLocal * xLocalF32
        Mul(xLocalF32, tmpUbF32, xLocalF32, tl_->UbFactorDimy * SWI_FACTOR * proDimsx);
        PipeBarrier<PIPE_V>();
        if (!tl_->activationScaleIsEmpty) {
            // Copy act scale: [proDimsx,8] -> [proDimsx,2H]
            CopyReshape<float>(tmpUbF32, activationScaleLocal, proDimsx, tl_->UbFactorDimy * SWI_FACTOR,
                {1, 0, static_cast<uint16_t>((tl_->UbFactorDimy * SWI_FACTOR) / BLOCK_ELEM), 1});
            PipeBarrier<PIPE_V>();
            // Calc dequant: xLocalF32 = activationScaleLocal * xLocalF32
            Mul(xLocalF32, tmpUbF32, xLocalF32, tl_->UbFactorDimy * SWI_FACTOR * proDimsx);
            PipeBarrier<PIPE_V>();
        }
    }

    if constexpr (std::is_same_v<TXGm, int32_t> && !std::is_same_v<TBias, int32_t>) {
        if (tl_->hasBias == 1) {
            // Copy bias: [1,2H] -> [proDimsx,2H]
            CopyReshape<float>(tmpUbF32, biasLocalF32_, proDimsx, tl_->UbFactorDimy * SWI_FACTOR,
                {1, 1, static_cast<uint16_t>((tl_->UbFactorDimy * SWI_FACTOR) / BLOCK_ELEM), 0});
            PipeBarrier<PIPE_V>();
            Add(xLocalF32, xLocalF32, tmpUbF32, proDimsx * tl_->inDimy);
            PipeBarrier<PIPE_V>();
        }
    }
    xActQueue_.EnQue(xLocalF32);
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::ComputeSwiGLU(int32_t proDimsx)
{
    LocalTensor<float> xLocalF32 = xActQueue_.DeQue<float>();
    if (tl_->swigluMode == 1) {
        // do special swiglu
        SwiGluGate(proDimsx, xLocalF32);
    } else {
        uint32_t calEleNum = tl_->UbFactorDimy * proDimsx;
        LocalTensor<float> tmpUbF32 = tmpBuf1_.AllocTensor<float>();
        // do normal swi pre
        LocalTensor<float> tmpUbF32Act = tmpUbF32;
        LocalTensor<float> tmpUbF32Gate = tmpUbF32[calEleNum];
        // Copy dequant result: xLocalF32[actOffset] -> tmpUbF32Act, [proDimsx,H]
        // Copy dequant result: xLocalF32[gateOffset] -> tmpUbF32Gate, [proDimsx,H]
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(tl_->UbFactorDimy);
        Copy<float, false>(
            tmpUbF32Act, xLocalF32[actOffset_], AscendC::MASK_PLACEHOLDER, proDimsx,
            {1, 1, static_cast<uint16_t>(tl_->UbFactorDimy / BLOCK_ELEM),
             static_cast<uint16_t>(tl_->UbFactorDimy / BLOCK_ELEM * SWI_FACTOR)});
        Copy<float, false>(
            tmpUbF32Gate, xLocalF32[gateOffset_], AscendC::MASK_PLACEHOLDER, proDimsx,
            {1, 1, static_cast<uint16_t>(tl_->UbFactorDimy / BLOCK_ELEM),
             static_cast<uint16_t>(tl_->UbFactorDimy / BLOCK_ELEM * SWI_FACTOR)});
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
        Muls(xLocalF32, tmpUbF32Act, static_cast<float>(-1.0), calEleNum);
        PipeBarrier<PIPE_V>();
        Exp(xLocalF32, xLocalF32, calEleNum);
        PipeBarrier<PIPE_V>();
        Adds(xLocalF32, xLocalF32, static_cast<float>(1.0), calEleNum);
        PipeBarrier<PIPE_V>();
        Div(tmpUbF32Act, tmpUbF32Act, xLocalF32, calEleNum);
        PipeBarrier<PIPE_V>();
        Mul(tmpUbF32Act, tmpUbF32Gate, tmpUbF32Act, calEleNum);
        PipeBarrier<PIPE_V>();
    }
    // x compute done, free
    xActQueue_.FreeTensor(xLocalF32);
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::ComputeQuant(int32_t proDimsx)
{
    LocalTensor<float> tmpUbF32 = tmpBuf1_.AllocTensor<float>();
    LocalTensor<float> tmpUbF32Act = tmpUbF32;
    LocalTensor<float> tmpUbF32Gate = tmpUbF32[tl_->UbFactorDimy * proDimsx];
    if (tl_->quantMode == 1) {
        DynamicQuant(tmpUbF32Act, tmpUbF32Gate, inScaleLocal_, proDimsx);
    } else {
        StaticQuant(tmpUbF32Act, tmpUbF32Gate, inScaleLocal_, proDimsx);
    }
    tmpBuf1_.FreeTensor(tmpUbF32);
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::Compute(int32_t proDimsx)
{
    ComputeDequant(proDimsx);
    ComputeSwiGLU(proDimsx);
    ComputeQuant(proDimsx);
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::CopyOut(int32_t proDimsx, int64_t xDimxOffset)
{
    // copy out
    LocalTensor<float> outLocal = outQueue_.DeQue<float>();
    LocalTensor<float> scaleOut = outLocal[UbSingleOutSize_ * sizeof(int8_t) / sizeof(float)];
    LocalTensor<int8_t> yOut = outLocal.template ReinterpretCast<int8_t>();

    if (tl_->quantMode == 1) {
        DataCopyParams dataCopyOutScaleParams;
        dataCopyOutScaleParams.blockCount = 1;
        dataCopyOutScaleParams.blockLen = proDimsx * sizeof(float);
        dataCopyOutScaleParams.srcStride = 0;
        dataCopyOutScaleParams.dstStride = 0;
        DataCopyPad(scaleGm_[xDimxOffset], scaleOut, dataCopyOutScaleParams);
    }
    DataCopyParams dataCopyOutyParams;
    dataCopyOutyParams.blockCount = 1;
    dataCopyOutyParams.blockLen = proDimsx * tl_->outDimy * sizeof(int8_t);
    dataCopyOutyParams.srcStride = 0;
    dataCopyOutyParams.dstStride = 0;
    DataCopyPad(yGm_[xDimxOffset * tl_->outDimy], yOut, dataCopyOutyParams);
    outQueue_.FreeTensor(outLocal);
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::ParamDequeAndCast()
{
    weightScaleLocal_ = weightScaleQueue_.DeQue<float>();

    // bias deque and cast bias to fp32 if needed
    if constexpr (std::is_same_v<TXGm, int32_t>) {
        if (tl_->hasBias == 1) {
            biasLocal_ = biasQueue_.DeQue<TBias>();
            biasLocalF32_ = biasLocal_.template ReinterpretCast<float>();
            if constexpr (std::is_same_v<TBias, half> || std::is_same_v<TBias, bfloat16_t>) {
                Cast(biasLocalF32_, biasLocal_[tl_->inDimy], RoundMode::CAST_NONE, tl_->inDimy);
            }
        }
    }

    // quant scale and quant offset deque, cast them to fp32 if needed
    inScaleLocal_ = inScaleQueue_.DeQue<float>();
    if (tl_->needSmoothScale == 1 && !tl_->quantIsOne) {
        if (std::is_same_v<TQuantScale, half> || std::is_same_v<TQuantScale, bfloat16_t>) {
            LocalTensor<TQuantScale> quantScaleLocalT16 = inScaleLocal_.template ReinterpretCast<TQuantScale>();
            Cast(inScaleLocal_, quantScaleLocalT16[tl_->outDimy], RoundMode::CAST_NONE, tl_->outDimy);
            PipeBarrier<PIPE_V>();
            if (tl_->quantMode == 0) {
                Cast(
                    inScaleLocal_[tl_->outDimy], quantScaleLocalT16[tl_->outDimy + tl_->inDimy], RoundMode::CAST_NONE,
                    tl_->outDimy);
                PipeBarrier<PIPE_V>();
            }
        }
    }
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::ProcessSingleGroupPerCore(
    int64_t groupIdx, int64_t dimxCore, int64_t coreDimxOffset)
{
    // do ub tiling again
    int32_t ubDimxLoop = (dimxCore + tl_->UbFactorDimx - 1) / tl_->UbFactorDimx;
    int32_t ubDimxTailFactor = dimxCore - tl_->UbFactorDimx * (ubDimxLoop - 1);

    // copyin 当前分组下使用的参数，weight scale, bias scale, quant scale+quant offset
    CopyInWeightScale(groupIdx);
    CopyInQuantScale(groupIdx);
    CopyInBias(groupIdx);
    ParamDequeAndCast();

    /*
      1. copyin x, activation scale
      2. compute
      3. copyout y, scale
    */
    for (uint32_t loopIdx = 0; loopIdx < ubDimxLoop; ++loopIdx) {
        int64_t xDimxOffset = coreDimxOffset + loopIdx * tl_->UbFactorDimx;
        int32_t proDimsx = loopIdx == (ubDimxLoop - 1) ? ubDimxTailFactor : tl_->UbFactorDimx;
        CopyInXAct(proDimsx, xDimxOffset);
        Compute(proDimsx);
        CopyOut(proDimsx, xDimxOffset);
    }
    ParamFree();
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::ParamFree()
{
    // 释放当前分组下使用的参数，weight scale, bias scale, quant scale，quant offset
    inScaleQueue_.FreeTensor(inScaleLocal_);
    weightScaleQueue_.FreeTensor(weightScaleLocal_);
    if constexpr (std::is_same_v<TXGm, int32_t>) {
        if (tl_->hasBias == 1) {
            biasQueue_.FreeTensor(biasLocal_);
        }
    }
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::ComputeReduceMax(
    const LocalTensor<float>& tempRes, int32_t calCount)
{
    uint32_t vectorCycles = calCount / MASK_NUM_T32;
    uint32_t remainElements = calCount % MASK_NUM_T32;

    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 0;
    repeatParams.src0RepStride = MASK_BLK_STRIDE;
    repeatParams.src1RepStride = 0;

    if (vectorCycles > 0 && remainElements > 0) {
        Max(tempRes, tempRes, tempRes[vectorCycles * MASK_NUM_T32], remainElements, 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }

    if (vectorCycles > 1) {
        Max(tempRes, tempRes[MASK_NUM_T32], tempRes, MASK_NUM_T32, vectorCycles - 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::CreateOffsetLocalTensor(
    uint32_t tensorLen, int swigluMode)
{
    // 不再需要创建偏移张量，因为直接使用前一半和后一半数据
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::SwiGluGate(
    int32_t proDimsx, const LocalTensor<float>& xLocalF32)
{
    uint32_t calEleNum = tl_->UbFactorDimy * proDimsx;
    LocalTensor<float> tmpUbF32 = tmpBuf1_.AllocTensor<float>();
    LocalTensor<float> tmpUbF32Act = tmpUbF32;
    LocalTensor<float> tmpUbF32Gate = tmpUbF32[calEleNum];
    // 前一半作为 tmpUbF32Act，后一半作为 tmpUbF32Gate
    Copy(tmpUbF32Act, xLocalF32, calEleNum);
    Copy(tmpUbF32Gate, xLocalF32[calEleNum], calEleNum);
    PipeBarrier<PIPE_V>();

    // tmpUbF32Gate
    Mins(tmpUbF32Gate, tmpUbF32Gate, tl_->clampLimit, calEleNum);
    PipeBarrier<PIPE_V>();
    Maxs(tmpUbF32Gate, tmpUbF32Gate, -(tl_->clampLimit), calEleNum);
    PipeBarrier<PIPE_V>();
    Adds(tmpUbF32Gate, tmpUbF32Gate, tl_->gluBias, calEleNum);
    PipeBarrier<PIPE_V>();

    // tmpUbF32Act
    Mins(tmpUbF32Act, tmpUbF32Act, tl_->clampLimit, calEleNum);
    PipeBarrier<PIPE_V>();
    Muls(xLocalF32, tmpUbF32Act, -(tl_->gluAlpha), calEleNum);
    PipeBarrier<PIPE_V>();
    Exp(xLocalF32, xLocalF32, calEleNum);
    PipeBarrier<PIPE_V>();
    Adds(xLocalF32, xLocalF32, (float)1.0, calEleNum);
    PipeBarrier<PIPE_V>();
    Div(tmpUbF32Act, tmpUbF32Act, xLocalF32, calEleNum);
    PipeBarrier<PIPE_V>();
    Mul(tmpUbF32Act, tmpUbF32Gate, tmpUbF32Act, calEleNum);
    PipeBarrier<PIPE_V>();
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::DynamicQuant(
    const LocalTensor<float>& tmpUbF32Act, const LocalTensor<float>& tmpUbF32Gate,
    const LocalTensor<float>& inScaleLocal, uint32_t proDimsx)
{
    if (tl_->needSmoothScale == 1) {
        // Copy quant scale: [1,H] -> [proDimsx,H]
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(tl_->UbFactorDimy);
        Copy<float, false>(
            tmpUbF32Gate, inScaleLocal, AscendC::MASK_PLACEHOLDER, proDimsx,
            {1, 1, static_cast<uint16_t>(tl_->UbFactorDimy / BLOCK_ELEM), 0});
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
        // Calc quant: xLocalF32 = tmpUbF32Act * inScaleLocal
        Mul(tmpUbF32Act, tmpUbF32Gate, tmpUbF32Act, tl_->UbFactorDimy * proDimsx);
        PipeBarrier<PIPE_V>();
    }

    // Calc quant: tmpUbF32Gate = abs(tmpUbF32Act)
    Abs(tmpUbF32Gate, tmpUbF32Act, tl_->UbFactorDimy * proDimsx);

    LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();
    LocalTensor<float> scaleOut = outLocal[UbSingleOutSize_ * sizeof(int8_t) / sizeof(float)];
    LocalTensor<int8_t> yOut = outLocal.template ReinterpretCast<int8_t>();
    PipeBarrier<PIPE_V>();
    // Calc quant: proDimsx * tl_->UbFactorDimy -> proDimsx * 64
    for (uint32_t i = 0; i < proDimsx; ++i) {
        ComputeReduceMax(tmpUbF32Gate[i * tl_->UbFactorDimy], tl_->UbFactorDimy);
    }
    // Calc quant: proDimsx * 64 -> proDimsx
    // repeatTimes:proDimsx, dstRepStride:1(dtype), srcBlkStride:1, srcRepStride:tl_->UbFactorDimy / 64 * 8
    WholeReduceMax(
        tmpUbF32Gate, tmpUbF32Gate, MASK_NUM_T32, proDimsx, 1, 1, tl_->UbFactorDimy / MASK_NUM_T32 * MASK_BLK_STRIDE,
        ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
    // Calc quant: scaleOut / 127.0
    Muls(scaleOut, tmpUbF32Gate, DYNAMIC_QUANT_FACTOR, proDimsx);
    PipeBarrier<PIPE_V>();
    // Calc Broadcast: proDimsx -> proDimsx,8
    int64_t blockCount = (proDimsx + BLOCK_ELEM - 1) / BLOCK_ELEM;
    Brcb(outLocal, scaleOut, blockCount, {1, MASK_BLK_STRIDE});
    PipeBarrier<PIPE_V>();
    // Copy scale: [proDimsx,8] -> [proDimsx,H]
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(tl_->UbFactorDimy);
    Copy<float, false>(
        tmpUbF32Gate, outLocal, AscendC::MASK_PLACEHOLDER, proDimsx,
        {1, 0, static_cast<uint16_t>(tl_->UbFactorDimy / BLOCK_ELEM), 1});
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();
    // Calc y: tmpUbF32Act = tmpUbF32Act / scaleOut
    Div(tmpUbF32Act, tmpUbF32Act, tmpUbF32Gate, tl_->UbFactorDimy * proDimsx);
    PipeBarrier<PIPE_V>();

    CastFloatToInt8(tmpUbF32Act, tmpUbF32Gate, proDimsx, yOut);
    outQueue_.EnQue<float>(outLocal);
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::StaticQuant(
    const LocalTensor<float>& tmpUbF32Act, const LocalTensor<float>& tmpUbF32Gate,
    const LocalTensor<float>& inScaleLocal, uint32_t proDimsx)
{
    if (tl_->needSmoothScale == 1) {
        if (tl_->quantIsOne) {
            // Calc quant: y = tmpUbF32Act * quantScale + quantOffset
            Muls(tmpUbF32Gate, tmpUbF32Act, this->quantScale_, tl_->UbFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();
            Adds(tmpUbF32Act, tmpUbF32Gate, this->quantOffset_, tl_->UbFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();
        } else {
            // Copy quant scale: [1,H] -> [proDimsx,H]
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(tl_->UbFactorDimy);
            Copy<float, false>(
                tmpUbF32Gate, inScaleLocal, AscendC::MASK_PLACEHOLDER, proDimsx,
                {1, 1, static_cast<uint16_t>(tl_->UbFactorDimy / BLOCK_ELEM), 0});
            SetMaskNorm();
            ResetMask();
            PipeBarrier<PIPE_V>();
            // Calc quant: y = tmpUbF32Act / quantScale
            Div(tmpUbF32Act, tmpUbF32Act, tmpUbF32Gate, tl_->UbFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();

            // Copy quant offset: [1,H] -> [proDimsx,H]
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(tl_->UbFactorDimy);
            Copy<float, false>(
                tmpUbF32Gate, inScaleLocal[tl_->outDimy], AscendC::MASK_PLACEHOLDER, proDimsx,
                {1, 1, static_cast<uint16_t>(tl_->UbFactorDimy / BLOCK_ELEM), 0});
            SetMaskNorm();
            ResetMask();
            PipeBarrier<PIPE_V>();
            // Calc quant: y = tmpUbF32Act + quantOffset
            Add(tmpUbF32Act, tmpUbF32Act, tmpUbF32Gate, tl_->UbFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();
        }
    }

    // do cast float to int8
    LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();
    LocalTensor<int8_t> yOut = outLocal.template ReinterpretCast<int8_t>();

    CastFloatToInt8(tmpUbF32Act, tmpUbF32Gate, proDimsx, yOut);
    outQueue_.EnQue<float>(outLocal);
}

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::CastFloatToInt8(
    const LocalTensor<float>& tmpUbF32Act, const LocalTensor<float>& tmpUbF32Gate, uint32_t proDimsx, LocalTensor<int8_t>& yOut)
{
    LocalTensor<int32_t> tmpUbF32ActI32 = tmpUbF32Act.ReinterpretCast<int32_t>();
    Cast(tmpUbF32ActI32, tmpUbF32Act, RoundMode::CAST_RINT, tl_->UbFactorDimy * proDimsx);
    SetDeqScale((half)1.000000e+00f);

    LocalTensor<half> tmpUbF32Gate16 = tmpUbF32Gate.template ReinterpretCast<half>();
    Cast(tmpUbF32Gate16, tmpUbF32ActI32, RoundMode::CAST_ROUND, tl_->UbFactorDimy * proDimsx);
    PipeBarrier<PIPE_V>();

    Cast(yOut, tmpUbF32Gate16, RoundMode::CAST_TRUNC, tl_->UbFactorDimy * proDimsx);
    PipeBarrier<PIPE_V>();
}

TEMPLATE_DSQ_DECLARE
template<typename T>
__aicore__ inline void DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>::CopyReshape(
    LocalTensor<T>& dstTensor, LocalTensor<T>& oriTensor, uint32_t rowNum, uint32_t colNum, CopyRepeatParams param)
{
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(colNum);
    Copy<T, false>(dstTensor, oriTensor, AscendC::MASK_PLACEHOLDER, rowNum, param);
    SetMaskNorm();
    ResetMask();
}

} // namespace DequantSwigluQuantOps
#endif // DEQUANT_SWIGLU_QUANT_H