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
 * \file prompt_flash_attention_var_len_score_sab_baseapi.h
 * \brief
 */

#ifndef PROMPT_FLASH_ATTENTION_VAR_LEN_SCORE_SAB_BASEAPI_H
#define PROMPT_FLASH_ATTENTION_VAR_LEN_SCORE_SAB_BASEAPI_H

#include "prompt_flash_attention_s1s2_bns1_mla_baseapi.h"

// INPUT_T - means data type for input
// T       - means data type when calc
template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T = INPUT_T,
          CubeFormat bmm1Format = CubeFormat::NZ, MmPolicyType mmPolicyType = MmPolicyType::NORMAL, bool pageAttention = false>
class PromptFlashAttentionVarLenScoreSameABBaseApi
    : public MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T, bmm1Format, mmPolicyType, pageAttention> {
    using Base = MlaS1s2Bn2gs1SameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T, bmm1Format, mmPolicyType, pageAttention>;
    using Base::BMM1_OUT_ISFIXED_ND;
    using Base::BMM2_OUT_ISFIXED_ND;
    using Base::BMM2_A_ISFIXED_ND;
    using Base::STAGE2_TBUF_SIZE;
    static constexpr CubeFormat BMM2_A_FORMAT = BMM2_A_ISFIXED_ND ? CubeFormat::ND : CubeFormat::NZ;
    static constexpr CubeFormat BMM2_OUT_FORMAT = BMM2_OUT_ISFIXED_ND ? CubeFormat::ND : CubeFormat::NZ;
public:
    __aicore__ inline PromptFlashAttentionVarLenScoreSameABBaseApi(){};

    __aicore__ inline void
    UnpackInit(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *attenMask,
               __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *blockTable, __gm__ uint8_t* learnableSink,
               __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, const TILING_TYPE *__restrict tiling, TPipe *tPipe);

    __aicore__ inline void Process();

protected:
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline int64_t GetKS2BaseStride();
    __aicore__ inline int64_t GetKRopeS2BaseStride();
    __aicore__ inline int64_t GetVS2BaseStride();

    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);

    __aicore__ inline void ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo);

    __aicore__ inline void SetExtraInfo(SplitSameABExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                        int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    
    __aicore__ inline void SetPaCacheLayout(SplitSameABExtraInfo &extraInfo);

    __aicore__ inline void CloneExtraInfo(SplitSameABExtraInfo &extraInfo, const SplitSameABExtraInfo &srcInfo,
                                          int64_t taskId, int64_t s2LoopCount);

    __aicore__ inline void CalS1OuterSize(int64_t offset);

    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen);

    __aicore__ inline void GetS2LoopRange();
    // Unpack 用参数
    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqKvlenAddr;
    GM_ADDR prefixNAddr;
    uint64_t s1OuterSizeAcc;
    uint64_t s1SizeAcc;
    uint64_t s2SizeAcc;
    uint64_t attenB1SSOffset;
    uint64_t softmaxPingPongCnt = 0;
    GlobalTensor<int64_t> qListGm;
    GlobalTensor<int64_t> kvListGm;
};

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                        bmm1Format, mmPolicyType, pageAttention>::UnpackInit(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *blockTable, __gm__ uint8_t* learnableSink, __gm__ uint8_t *attentionOut, 
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, const TILING_TYPE *__restrict tiling, TPipe *tPipe)
{
    this->InitInput(query, key, value, queryRope, keyRope, attenMask, blockTable, learnableSink, attentionOut, softmaxLse, workspace, tiling, tPipe); // gm设置
    this->ComputeConstexpr();
    qListGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengths);
    kvListGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsKv);
    // 初始化unpack
    actualSeqQlenAddr = actualSeqLengths;
    actualSeqKvlenAddr = actualSeqLengthsKv;
    int64_t actualS1Len;
    int64_t actualS2Len;

    if ASCEND_IS_AIV {
        for (int64_t i = 0; i < this->tilingData->PFAinputParams.bSize; ++i) {
            GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);
            if (actualS2Len <= 0 && actualS1Len != 0) {
                int64_t accumSize = (i == 0) ? 0 : ((__gm__ int64_t *)actualSeqQlenAddr)[i - 1];
                if (actualS1Len < 0 && accumSize > 0) {
                    actualS1Len = this->s1Size - accumSize;
                }
                AscendC::InitOutput<INPUT_T>(this->attentionOutGm[accumSize * this->n2GD2],
                                            actualS1Len * this->n2GD2, static_cast<INPUT_T>(0.0));
            }
        }

        if (this->tilingData->PFAinputParams.isSoftMaxLseEnable) {
            SyncAll(); // synchronize for attentionOutGm
            int64_t tmpBlockIdx = GetBlockIdx();
            const int64_t totalSoftMaxLseOutputSize = ((__gm__ int64_t *)actualSeqQlenAddr)[this->tilingData->PFAinputParams.bSize - 1] * this->n2G;
            int64_t coreNum = this->tilingData->PFAmultiCoreParams.coreNum;
            if (coreNum != 0 && tmpBlockIdx < coreNum) {
                int64_t singleCoreLseSize = totalSoftMaxLseOutputSize / coreNum;
                if (tmpBlockIdx == coreNum - 1) {
                    singleCoreLseSize += totalSoftMaxLseOutputSize % coreNum;
                }
                const int64_t singleCoreLseSizeFinal = singleCoreLseSize;
                AscendC::InitOutput<float>(this->softmaxLseGm[tmpBlockIdx * (totalSoftMaxLseOutputSize / coreNum)],
                                            singleCoreLseSizeFinal, static_cast<float>(3e+99)); // 3e+99:set the value of invalid batch to inf
                SyncAll(); // synchronize for softmaxLseGm
            }
        }
    }
    
    this->InitBuffer();

    if ASCEND_IS_AIV {
        if (this->vecBlockIdx < this->tilingData->PFAmultiCoreParams.coreNum) {
            LocalTensor<half> pseHelpBuffer = this->stage1PingBuf.template Get<half>();
        }
    }
    this->s2SizeSum = ((__gm__ uint64_t *)actualSeqLengthsKv)[this->tilingData->PFAinputParams.bSize - 1];
    return;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                             bmm1Format, mmPolicyType, pageAttention>::ComputeConstexpr()
{
    this->gD = this->tilingData->PFAinputParams.gSize * this->dSize;
    this->n2D = this->tilingData->PFAinputParams.n2Size * this->dSize;
    this->n2G = this->tilingData->PFAinputParams.n2Size * this->tilingData->PFAinputParams.gSize;
    this->n2GD = this->tilingData->PFAinputParams.n2Size * this->gD;
    this->n2GOuterSize = this->tilingData->PFAinputParams.n2Size * this->tilingData->PFAcoreParams.gOuterSize;
    this->gBaseSize = this->tilingData->PFAcoreParams.gBaseSize;
    if (unlikely(this->gBaseSize > 1)) {
        this->notSplitG = true;
    }
    this->gD2 = this->tilingData->PFAinputParams.gSize * this->valueDSize;
    this->n2D2 = this->tilingData->PFAinputParams.n2Size * this->valueDSize;
    this->n2GD2 = this->tilingData->PFAinputParams.n2Size * this->gD2;

    this->gRopeD = this->tilingData->PFAinputParams.gSize * this->ropeHeadSize;
    this->s2RopeD = this->tilingData->PFAinputParams.s2Size * this->ropeHeadSize;
    this->n2S2RopeD = this->tilingData->PFAinputParams.n2Size * this->s2RopeD;
    this->s1BaseRopeD = this->cubeS1BaseSize * this->ropeHeadSize;
    this->s1RopeD = this->tilingData->PFAinputParams.s1Size * this->ropeHeadSize;
    this->gS1RopeD = this->tilingData->PFAinputParams.gSize * this->s1RopeD;
    this->n2GS1RopeD = this->n2GS1 * this->ropeHeadSize;
    this->n2RopeD = this->tilingData->PFAinputParams.n2Size * this->ropeHeadSize;
    this->bN2RopeD = this->tilingData->PFAinputParams.bSize * this->n2RopeD;
    this->n2GRopeD = this->tilingData->PFAinputParams.n2Size * this->gRopeD;
    this->bN2GRopeD = this->tilingData->PFAinputParams.bSize * this->n2GRopeD;
    this->s1BaseN2GRopeD = this->cubeS1BaseSize * this->n2GRopeD;
    this->s1BaseBN2GRopeD = this->cubeS1BaseSize * this->bN2GRopeD;
    this->s2BaseN2RopeD = this->s2BaseSize * this->n2RopeD;
    this->s1BaseN2GD2 = this->cubeS1BaseSize * this->n2GD2;

    // 计算切分轴的乘积
    this->s2BaseN2D = this->s2BaseSize * this->n2D;
    this->s2BaseNRatioSize = this->s2BaseSize * this->tilingData->PFAcoreParams.nRatio;
    this->s1Size = this->tilingData->PFAinputParams.s1Size;

    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        // layout(bs)ND
        this->s1BaseN2GD = this->cubeS1BaseSize * this->n2GD;
        this->s2BaseNratioN2D = this->s2BaseN2D * this->tilingData->PFAcoreParams.nRatio;
        this->s2BaseNratioN2RopeD = this->s2BaseN2RopeD * this->tilingData->PFAcoreParams.nRatio;

        this->mm1Ka = this->n2GD;
        this->mm1Kb = this->n2D;
        this->mm1RopeKa = this->n2GRopeD;
        this->mm1RopeKb = this->n2RopeD;
        this->mm2Kb = this->n2D2;
    } else {
        // layout: N(bs)D
        this->s1D = this->tilingData->PFAinputParams.s1Size * this->dSize;
        this->s2D = this->tilingData->PFAinputParams.s2Size * this->dSize;
        this->s2D2 = this->tilingData->PFAinputParams.s2Size * this->valueDSize;
        this->gS1D = this->tilingData->PFAinputParams.gSize * this->s1D;
        this->s1D2 = this->tilingData->PFAinputParams.s1Size * this->valueDSize;
        this->gS1D2 = this->tilingData->PFAinputParams.gSize * this->s1D2;
        this->s1BaseD = this->cubeS1BaseSize * this->dSize;
        this->s1BaseD2 = this->cubeS1BaseSize * this->valueDSize;

        this->s2BaseNratioD = this->s2BaseNRatioSize * this->dSize;
        this->s2BaseNratioRopeD = this->s2BaseNRatioSize * this->ropeHeadSize;

        this->mm1Ka = this->dSize;
        this->mm1Kb = this->dSize;
        this->mm1RopeKa = this->ropeHeadSize;
        this->mm1RopeKb = this->ropeHeadSize;
        this->mm2Kb = this->valueDSize;
    }

    this->softmaxReduceSize = 1;

    if ASCEND_IS_AIC {
        this->matmulService.InitParams(this->dSize, this->valueDSize, this->ropeHeadSize, this->tilingData->PFAinputParams.n2Size, this->mm1Ka, this->mm1Kb, this->mm1RopeKa, this->mm1RopeKb, this->mm2Kb);
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline int64_t PromptFlashAttentionVarLenScoreSameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                                       bmm1Format, mmPolicyType, pageAttention>::GetKS2BaseStride()
{
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        return this->s2BaseNratioN2D;
    } else {
        return this->s2BaseNratioD;
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline int64_t PromptFlashAttentionVarLenScoreSameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                                       bmm1Format, mmPolicyType, pageAttention>::GetKRopeS2BaseStride()
{
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        return this->s2BaseNratioN2RopeD;
    } else {
        return this->s2BaseNratioRopeD;
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>                                                                       
__aicore__ inline int64_t PromptFlashAttentionVarLenScoreSameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                                       bmm1Format, mmPolicyType, pageAttention>::GetVS2BaseStride()
{
    if constexpr (layOutType == LayOutTypeEnum::LAYOUT_TND) {
        return this->s2BaseNRatioSize * this->n2D2;
    } else {
        return this->s2BaseNRatioSize * this->valueDSize;
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void
PromptFlashAttentionVarLenScoreSameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                        bmm1Format, mmPolicyType, pageAttention>::GetSeqQlenKvlenByBoidx(int64_t boIdx,
                                                                                            int64_t &actualSeqQlen,
                                                                                            int64_t &actualSeqKvlen)
{
    if (boIdx == 0) {
        actualSeqQlen = qListGm.GetValue(0);
        actualSeqKvlen = kvListGm.GetValue(0);
    } else {
        actualSeqQlen = qListGm.GetValue(boIdx) - qListGm.GetValue(boIdx - 1);
        if constexpr (pageAttention) {
            actualSeqKvlen = kvListGm.GetValue(boIdx);
        } else {
            actualSeqKvlen = kvListGm.GetValue(boIdx) - kvListGm.GetValue(boIdx - 1);
        }
    }
    return;
}

// 初始化s1方向上的累加值
template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                        bmm1Format, mmPolicyType, pageAttention>::CalS1OuterSize(int64_t offset)
{
    int64_t actualS1Outersize = 0;
    // 用于取actualS1Len下标
    this->boIdx = 0;
    this->s1OuterSizeAcc = 0;
    this->attenB1SSOffset = 0;
    this->s1SizeAcc = 0;
    this->s2SizeAcc = 0;

    int64_t actualS1Len;
    int64_t actualS2Len;
    for (int64_t i = 0; i < this->tilingData->PFAinputParams.bSize; i++) {
        GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);
        actualS1Outersize += (CeilDiv(actualS1Len, this->cubeS1BaseSize) * this->n2GOuterSize);
        if (offset >= actualS1Outersize) {
            this->s1OuterSizeAcc = actualS1Outersize;
            this->s1SizeAcc += actualS1Len;
            this->s2SizeAcc += actualS2Len;
            this->attenB1SSOffset += actualS1Len * actualS2Len;
            this->boIdx++;
        } else {
            break;
        }
    }
    return;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T, 
                                                        bmm1Format, mmPolicyType, pageAttention>::ComputeAxisIdx(int64_t multiCoreInnerIdx)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
    int64_t actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(actualS1Len, this->cubeS1BaseSize) * this->n2GOuterSize);
    while (multiCoreInnerIdx >= actualS1Outersize) {
        this->s1OuterSizeAcc = actualS1Outersize;
        this->s1SizeAcc += actualS1Len;
        this->s2SizeAcc += actualS2Len;
        this->attenB1SSOffset += actualS1Len * actualS2Len;
        this->boIdx++;
        GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
        actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(actualS1Len, this->cubeS1BaseSize) * this->n2GOuterSize);
    }
    // 计算轴的idx
    int64_t tmpS1Outersize = CeilDiv(actualS1Len, this->cubeS1BaseSize);
    actualS1Outersize = multiCoreInnerIdx - this->s1OuterSizeAcc;
    this->n2oIdx = actualS1Outersize / tmpS1Outersize / this->tilingData->PFAcoreParams.gOuterSize;
    this->goIdx = actualS1Outersize / tmpS1Outersize % this->tilingData->PFAcoreParams.gOuterSize;
    this->s1oIdx = actualS1Outersize % tmpS1Outersize;
    GetSeqQlenKvlenByBoidx(this->boIdx, this->s1Size, this->s2Size);
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                             bmm1Format, mmPolicyType, pageAttention>::GetS2LoopRange()
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
    if (this->tilingData->PFAinputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) { // 下三角
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize, actualS2Len);
    } else if (this->tilingData->PFAinputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL)) {
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len - actualS1Len, actualS2Len);
    } else if (this->tilingData->PFAinputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
        int64_t s1SparseValidSize = this->tilingData->PFAcoreParams.s1SparseValidSize + actualS1Len - actualS2Len;
        int64_t s2SparseValidSize = this->tilingData->PFAcoreParams.s2SparseValidSize - actualS1Len + actualS2Len;
        this->s2StartIdx = Max(this->s1oIdx * this->cubeS1BaseSize - s1SparseValidSize, 0);
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + s2SparseValidSize, actualS2Len);
        // s1baseSize行都无效时，需要将startIdx设置为0，endIdx设置为S2realSize
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }

        if ((layOutType == LayOutTypeEnum::LAYOUT_TND) || (layOutType == LayOutTypeEnum::LAYOUT_NTD_TND)) {
            // preTonke < 0 或者 nextToken < q_s - kv_s 则为 sparseMode4的行无效场景，设置行无效标记
            if ((this->tilingData->PFAinputParams.preTokens < 0) || (this->tilingData->PFAinputParams.nextTokens < actualS1Len - actualS2Len)) {
                this->hasSparse4InvalidLine = true;
            }
        }
    } else if (this->tilingData->PFAinputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_COMPRESS)) {
        this->s2StartIdx = Max(this->s1oIdx * this->cubeS1BaseSize - actualS1Len +
                                   Max(actualS2Len - this->tilingData->PFAinputParams.preTokens, 0),
                               0);
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len -
                                 Max(actualS1Len - this->tilingData->PFAinputParams.nextTokens, 0),
                             actualS2Len);
        // s1baseSize行都无效时，需要将startIdx设置为0，endIdx设置为S2realSize
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }
    } else if (this->tilingData->PFAinputParams.sparseType ==
               static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL_BAND)) {
        if (this->boIdx == this->tilingData->PFAinputParams.bandIndex) {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len +
                                     this->tilingData->PFAinputParams.nextTokens - actualS1Len,
                                 actualS2Len);
        } else {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len - actualS1Len, actualS2Len);
        }
    } else if (this->tilingData->PFAinputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL)) {
        if (this->boIdx == this->tilingData->PFAinputParams.bandIndex) {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len -
                                     Max(actualS1Len - this->tilingData->PFAinputParams.nextTokens, 0),
                                 actualS2Len);
        } else {
            this->s2StartIdx = 0;
            this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize, actualS2Len);
        }
    } else if (this->tilingData->PFAinputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::PREFIX)) {
        this->s2StartIdx = 0;
        this->s2EndIdx =
            Max((this->s1oIdx + 1) * this->cubeS1BaseSize - actualS1Len + actualS2Len,
                             ((__gm__ int64_t *)prefixNAddr)[this->boIdx]);
        this->s2EndIdx = Min(this->s2EndIdx, this->s2Size);
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }
    } else { // 其它场景全计算
        this->s2StartIdx = 0;
        this->s2EndIdx = actualS2Len;
    }
    this->s2StartIdx = this->s2StartIdx / 8 * 8;
    this->s2EndIdx = Min(CeilDiv(this->s2EndIdx, 8) * 8, actualS2Len);
    return;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                             bmm1Format, mmPolicyType, pageAttention>::Process()
{
    this->AllocEvent();
    if ASCEND_IS_AIC {
        this->matmulService.AllocEventID();
    }

    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->cubeBlockIdx * this->tilingData->PFAmultiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->PFAmultiCoreParams.splitFactorSize;
    if (this->tilingData->PFAmultiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->PFAmultiCoreParams.totalSize;
    }
    // 计算sparse场景下s1的循环范围
    this->GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);
    // 初始化AxisIdx
    this->CalS1OuterSize(multiCoreInnerOffset);

    // qkv D等长=128，不切K轴场景
    bool qkvDSameFlag = false;
    if ((this->valueDSize == (this->dSize + this->ropeHeadSize)) && (this->valueDSize <= 128)) {
        qkvDSameFlag = true;
    }

    SplitSameABExtraInfo extraInfo[3];
    int64_t taskId = 0;
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit; ++multiCoreInnerIdx) {
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
        this->ComputeAxisIdx(multiCoreInnerIdx);
        // s2轴循环计数, 支持sparse和非sparse场景
        this->GetS2LoopRange();
        int64_t s2LoopLimit = CeilDiv(this->s2EndIdx - this->s2StartIdx, this->s2BaseNRatioSize) - 1;
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            if (s2LoopCount == 0) {
                this->softmaxPingPongCnt++;
                this->SetExtraInfo(extraInfo[taskId % 3], taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
            } else {
                this->CloneExtraInfo(extraInfo[taskId % 3], extraInfo[(taskId - 1) % 3], taskId, s2LoopCount);
            }

            if ASCEND_IS_AIV {
                if (taskId > 0) {
                    CrossCoreWaitFlag(Base::SYNC_C1_V1_FLAG);
                    this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
                    CrossCoreSetFlag<Base::SYNC_MODE2, PIPE_MTE3>(Base::SYNC_V1_C2_FLAG);
                }
            }

            if ASCEND_IS_AIC {
                extraInfo[taskId % 3].qkvDSameFlag = qkvDSameFlag;
                if (extraInfo[taskId % 3].needNz2Nd == 1) {
                    this->matmulService.template ComputeMm1<CubeFormat::NZ>(extraInfo[taskId % 3]);
                } else {
                    this->matmulService.template ComputeMm1<CubeFormat::ND>(extraInfo[taskId % 3]);
                }

                CrossCoreSetFlag<Base::SYNC_MODE2, PIPE_FIX>(Base::SYNC_C1_V1_FLAG);
            }

            if ASCEND_IS_AIV {
                if (taskId > 1) {
                    CrossCoreWaitFlag(Base::SYNC_C2_V2_FLAG);
                    this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
                }
            }

            if ASCEND_IS_AIC {
                if (taskId > 0) {
                    CrossCoreWaitFlag(Base::SYNC_V1_C2_FLAG);
                    this->matmulService.template ComputeMm2<BMM2_A_FORMAT, BMM2_OUT_FORMAT>(extraInfo[(taskId + 2) % 3]);
                    CrossCoreSetFlag<Base::SYNC_MODE2, PIPE_FIX>(Base::SYNC_C2_V2_FLAG);
                }
            }

            taskId++;
        }
    }
    if (taskId >= 1) {
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(Base::SYNC_C1_V1_FLAG);
            this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
            CrossCoreSetFlag<Base::SYNC_MODE2, PIPE_MTE3>(Base::SYNC_V1_C2_FLAG);
        }
        
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(Base::SYNC_V1_C2_FLAG);
            this->matmulService.template ComputeMm2<BMM2_A_FORMAT, BMM2_OUT_FORMAT>(extraInfo[(taskId + 2) % 3]);
            CrossCoreSetFlag<Base::SYNC_MODE2, PIPE_FIX>(Base::SYNC_C2_V2_FLAG);
        }

        if ASCEND_IS_AIV {
            if (taskId > 1) {
                CrossCoreWaitFlag(Base::SYNC_C2_V2_FLAG);
                this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
            }
        }
    }
    taskId++;
    if ASCEND_IS_AIV {
        if (taskId >= 2) {
            CrossCoreWaitFlag(Base::SYNC_C2_V2_FLAG);
            this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
        }
    }

    if ASCEND_IS_AIC {
        this->matmulService.FreeEventID();
    }
    this->FreeEvent();   
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                        bmm1Format, mmPolicyType, pageAttention>::SetPaCacheLayout(SplitSameABExtraInfo &extraInfo)
{
    if (this->tilingData->PFAinputParams.paCacheLayoutType == paCacheBBH) {
        extraInfo.paCacheLayoutType = PaCacheLayoutType::PA_BBH;
    } else if (this->tilingData->PFAinputParams.paCacheLayoutType == paCacheBNBD) {
        extraInfo.paCacheLayoutType = PaCacheLayoutType::PA_BNBD;
    } else {
        extraInfo.paCacheLayoutType = PaCacheLayoutType::PA_NZ;
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                        bmm1Format, mmPolicyType, pageAttention>::SetExtraInfo(
    SplitSameABExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    extraInfo.s2StartIdx = this->s2StartIdx;
    extraInfo.s2EndIdx = this->s2EndIdx;
    extraInfo.s2LoopCount = s2LoopCount;
    extraInfo.s1oIdx = this->s1oIdx;
    extraInfo.boIdx = this->boIdx;
    extraInfo.n2oIdx = this->n2oIdx;
    extraInfo.goIdx = this->goIdx;
    extraInfo.taskId = taskId;
    extraInfo.taskIdMod2 = taskId % 2;
    extraInfo.s2LoopLimit = s2LoopLimit;
    extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;
    extraInfo.multiCoreInnerIdxMod2 = this->softmaxPingPongCnt % 2;
    extraInfo.s1SizeAcc = s1SizeAcc;
    extraInfo.s2SizeAcc = s2SizeAcc;
    extraInfo.attenB1SSOffset = attenB1SSOffset;
    if (this->tilingData->PFAinputParams.attenMaskShapeType == attenMaskS1S2) {
        extraInfo.attenMaskS2Size = this->tilingData->PFAinputParams.s2Size;
    } else if (this->tilingData->PFAinputParams.attenMaskShapeType == attenMaskTT) {
        extraInfo.attenMaskS2Size = this->s2SizeSum;
    }

    // band compress mode
    if (this->tilingData->PFAinputParams.attenMaskCompressMode !=
        static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
        extraInfo.attenMaskS2Size = this->tilingData->PFAinputParams.attenMaskS2Size;
    }

    GetSeqQlenKvlenByBoidx(extraInfo.boIdx, extraInfo.s1Size, extraInfo.s2Size);
    extraInfo.cubeS1RealSize = Min(this->cubeS1BaseSize, extraInfo.s1Size - extraInfo.s1oIdx * this->cubeS1BaseSize);
    extraInfo.gBaseSize = this->gBaseSize;
    this->ComputeBmm1Tail(extraInfo);

    extraInfo.qCoreOffset = this->CalcBmm1TensorAGmOffset(extraInfo);
    extraInfo.kCoreOffset = this->CalcBmm1TensorBGmOffset(extraInfo);
    extraInfo.withRope = (this->ropeHeadSize != 0);
    if (extraInfo.withRope) {
        extraInfo.qRopeOffset = this->CalcBmm1QRopeGmOffset(extraInfo);
        extraInfo.kRopeOffset = this->CalcBmm1KRopeGmOffset(extraInfo);
    }
    extraInfo.vCoreOffset = this->CalcBmm2TensorBGmOffset(extraInfo);
    extraInfo.s2BaseOffset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * this->s2BaseNRatioSize;
    this->SetPaCacheLayout(extraInfo);
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                        bmm1Format, mmPolicyType, pageAttention>::CloneExtraInfo(SplitSameABExtraInfo &extraInfo, 
                                                        const SplitSameABExtraInfo &srcInfo, int64_t taskId, int64_t s2LoopCount)
{
    extraInfo.s2StartIdx = srcInfo.s2StartIdx;
    extraInfo.s2EndIdx = srcInfo.s2EndIdx;
    extraInfo.s2LoopCount = s2LoopCount;
    extraInfo.s1oIdx = srcInfo.s1oIdx;
    extraInfo.boIdx = srcInfo.boIdx;
    extraInfo.n2oIdx = srcInfo.n2oIdx;
    extraInfo.goIdx = srcInfo.goIdx;
    extraInfo.taskId = taskId;
    extraInfo.taskIdMod2 = taskId % 2;
    extraInfo.multiCoreInnerIdxMod2 = this->softmaxPingPongCnt % 2;

    extraInfo.s1RealSize = srcInfo.s1RealSize;
    extraInfo.cubeS1RealSize = srcInfo.cubeS1RealSize;

    if (likely(s2LoopCount < srcInfo.s2LoopLimit)) {
        extraInfo.needNz2Nd = srcInfo.needNz2Nd;
        extraInfo.s2RealSize = srcInfo.s2RealSize;
        extraInfo.s2AlignedSize = srcInfo.s2AlignedSize;
        extraInfo.s2RealSizeAlign64 = extraInfo.s2AlignedSize;
        extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2AlignedSize;
        extraInfo.s2LastLoop = false;

        extraInfo.vec1S1BaseSize = srcInfo.vec1S1BaseSize;
        extraInfo.realSplitN = srcInfo.realSplitN;
    } else {
        extraInfo.s2RealSize = extraInfo.s2EndIdx - s2LoopCount * srcInfo.s2RealSize - extraInfo.s2StartIdx;
        extraInfo.needNz2Nd = (BMM1_OUT_ISFIXED_ND || (extraInfo.s2RealSize % 64 == 0)) ? 0 : 1;
        extraInfo.s2AlignedSize = Align(extraInfo.s2RealSize);
        extraInfo.s2RealSizeAlign64 = CeilDiv(extraInfo.s2RealSize, 64) * 64;
        extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2RealSize / 8 * 8;
        extraInfo.s2LastLoop = true;

        extraInfo.duplicateMask = 
            (((uint64_t)1 << (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8)) - 1) &
            (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
        if (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8 == 64) {
            extraInfo.duplicateMask = 
                0xffffffffffffffff & (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
        }

        if (unlikely(extraInfo.s2RealSize < fp32BaseSize)) {
            extraInfo.vec1S1BaseSize = Min(1024 / extraInfo.s2RealSizeAlign64 * 8, extraInfo.s1RealSize); // Maximize the ub
            extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize, extraInfo.vec1S1BaseSize);
        } else {
            extraInfo.vec1S1BaseSize = Min(1024 / extraInfo.s2RealSizeAlign64 * 8, extraInfo.s1RealSize); // Maximize the ub
            extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize, extraInfo.vec1S1BaseSize);
        }
    }

    extraInfo.vec2S1BaseSize = srcInfo.vec2S1BaseSize;
    
    extraInfo.s2LoopLimit = srcInfo.s2LoopLimit;
    extraInfo.multiCoreInnerIdx = srcInfo.multiCoreInnerIdx;

    extraInfo.qCoreOffset = srcInfo.qCoreOffset;
    extraInfo.kCoreOffset = srcInfo.kCoreOffset + GetKS2BaseStride();
    extraInfo.s2BaseOffset = srcInfo.s2BaseOffset + this->s2BaseNRatioSize;
    extraInfo.paCacheLayoutType = srcInfo.paCacheLayoutType;
    extraInfo.withRope = srcInfo.withRope;
    if (extraInfo.withRope) {
        extraInfo.qRopeOffset = srcInfo.qRopeOffset;
        extraInfo.kRopeOffset = srcInfo.kRopeOffset + GetKRopeS2BaseStride();
    }
    extraInfo.vCoreOffset = srcInfo.vCoreOffset + GetVS2BaseStride();

    extraInfo.s1SizeAcc = s1SizeAcc;
    extraInfo.s2SizeAcc = s2SizeAcc;
    extraInfo.attenB1SSOffset = attenB1SSOffset;
    extraInfo.attenMaskS2Size = srcInfo.attenMaskS2Size;
    extraInfo.s1Size = srcInfo.s1Size;
    extraInfo.s2Size = srcInfo.s2Size;
    extraInfo.vecCoreOffset = srcInfo.vecCoreOffset;
    extraInfo.gBaseSize = srcInfo.gBaseSize;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType, bool pageAttention>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameABBaseApi<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                        bmm1Format, mmPolicyType, pageAttention>::ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo)
{
    extraInfo.s1RealSize = (extraInfo.cubeS1RealSize + 1) / 2;
    extraInfo.vecCoreOffset = this->cubeSubIdx * extraInfo.s1RealSize * extraInfo.gBaseSize;
    if (this->cubeSubIdx == 1) {
        extraInfo.s1RealSize = extraInfo.cubeS1RealSize - extraInfo.s1RealSize;
    }
    extraInfo.s2RealSize = this->s2BaseNRatioSize;
    extraInfo.s2AlignedSize = extraInfo.s2RealSize;
    bool isS2Tail = false;
    if (extraInfo.s2StartIdx + (extraInfo.s2LoopCount + 1) * extraInfo.s2RealSize >= extraInfo.s2EndIdx) {
        extraInfo.s2RealSize = extraInfo.s2EndIdx - extraInfo.s2LoopCount * extraInfo.s2RealSize - extraInfo.s2StartIdx;
        extraInfo.s2AlignedSize = Align(extraInfo.s2RealSize);
        isS2Tail = true;
    }

    extraInfo.s2RealSizeAlign64 = extraInfo.s2AlignedSize;
    extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2AlignedSize;
    extraInfo.s2LastLoop = false;
    if (isS2Tail) {
        extraInfo.s2RealSizeAlign64 = CeilDiv(extraInfo.s2RealSize, 64) * 64;
        extraInfo.s2RealSizeFloorAlign8 = extraInfo.s2RealSize / 8 * 8;
        extraInfo.s2LastLoop = true;
        extraInfo.duplicateMask =
            (((uint64_t)1 << (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8)) - 1) &
            (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
        if (extraInfo.s2RealSizeAlign64 - extraInfo.s2RealSizeFloorAlign8 == 64) {
            extraInfo.duplicateMask =
                0xffffffffffffffff & (~(((uint64_t)1 << (extraInfo.s2RealSize - extraInfo.s2RealSizeFloorAlign8)) - 1));
        }
    }

    // In scenes where s2 is less than 8, when traversing s1basesize for computation, there is a memory issue.
    // Therefore, it's changed to compute once
    if (unlikely(extraInfo.s2RealSize < fp32BaseSize)) {
        extraInfo.vec1S1BaseSize = Min(1024 / extraInfo.s2RealSizeAlign64 * 8, extraInfo.s1RealSize * this->gBaseSize); // Maximize the ub
        extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize * this->gBaseSize, extraInfo.vec1S1BaseSize);
    } else {
        extraInfo.vec1S1BaseSize = Min(1024 / extraInfo.s2RealSizeAlign64 * 8, extraInfo.s1RealSize * this->gBaseSize); // Maximize the ub
        extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize * this->gBaseSize, extraInfo.vec1S1BaseSize);
    }

    if (unlikely(this->notSplitG)) {
        extraInfo.vec2S1BaseSize = extraInfo.s1RealSize;
    } else {
        extraInfo.vec2S1BaseSize = STAGE2_TBUF_SIZE / this->valueDSizeAlign16;   // 32 * 128 / 64 = 64
    }
    extraInfo.needNz2Nd = (BMM1_OUT_ISFIXED_ND || (extraInfo.s2RealSize % 64 == 0)) ? 0 : 1;
    return;
}

#endif // PROMPT_FLASH_ATTENTION_VAR_LEN_SCORE_SAB_BASEAPI_H
