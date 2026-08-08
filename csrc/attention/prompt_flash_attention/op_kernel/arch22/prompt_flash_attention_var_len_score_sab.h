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
 * \file prompt_flash_attention_var_len_score_sab.h
 * \brief
 */

#ifndef PROMPT_FLASH_ATTENTION_VAR_LEN_SCORE_SAB_H
#define PROMPT_FLASH_ATTENTION_VAR_LEN_SCORE_SAB_H

#include "prompt_flash_attention_s1s2_bns1_mla.h"

// INPUT_T - means data type for input
// T       - means data type when calc
template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T = INPUT_T,
          CubeFormat bmm1Format = CubeFormat::NZ, MmPolicyType mmPolicyType = MmPolicyType::NORMAL>
class PromptFlashAttentionVarLenScoreSameAB
    : public MlaS1s2Bn2gs1SameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T, bmm1Format, mmPolicyType> {
public:
    __aicore__ inline PromptFlashAttentionVarLenScoreSameAB(){};

    __aicore__ inline void
    UnpackInit(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *attenMask,
               __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
               __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, const TILING_TYPE *__restrict tiling, TPipe *tPipe);

    __aicore__ inline void Process();

protected:
    __aicore__ inline void ComputeConstexpr();

    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);

    __aicore__ inline void ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo);

    __aicore__ inline void SetExtraInfo(SplitSameABExtraInfo &extraInfo, int64_t taskId, int64_t s2LoopCount,
                                        int64_t s2LoopLimit, int64_t multiCoreInnerIdx);

    __aicore__ inline void CalS1OuterSize(int64_t offset);

    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvlen);

    __aicore__ inline void GetS2LoopRange();
    // Unpack 用参数
    GM_ADDR actualSeqQlenAddr{nullptr};
    GM_ADDR actualSeqKvlenAddr{nullptr};
    GM_ADDR prefixNAddr{nullptr};
    uint64_t s1OuterSizeAcc{0};
    uint64_t s1SizeAcc{0};
    uint64_t s2SizeAcc{0};
    uint64_t attenB1SSOffset{0};
    uint64_t softmaxPingPongCnt{0};
    GlobalTensor<int64_t> qListGm;
    GlobalTensor<int64_t> kvListGm;
};

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                        bmm1Format, mmPolicyType>::UnpackInit(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *attentionOut, 
    __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, const TILING_TYPE *__restrict tiling, TPipe *tPipe)
{
    this->InitInput(query, key, value, attenMask, attentionOut, softmaxLse, workspace, tiling, tPipe); // gm设置
    this->ComputeConstexpr();
    qListGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengths);
    kvListGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLengthsKv);
    // 初始化unpack
    actualSeqQlenAddr = actualSeqLengths;
    actualSeqKvlenAddr = actualSeqLengthsKv;
    int64_t actualS1Len;
    int64_t actualS2Len;
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
    this->InitBuffer();

    if (this->vecBlockIdx < this->tilingData->PFAmultiCoreParams.coreNum) {
        LocalTensor<half> pseHelpBuffer = this->stage1PingBuf.template Get<half>();
    }
    this->s2SizeSum = ((__gm__ uint64_t *)actualSeqLengthsKv)[this->tilingData->PFAinputParams.bSize - 1];
    return;
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                             bmm1Format, mmPolicyType>::ComputeConstexpr()
{
    this->gD = this->tilingData->PFAinputParams.gSize * this->dSize;
    this->n2D = this->tilingData->PFAinputParams.n2Size * this->dSize;
    this->n2G = this->tilingData->PFAinputParams.n2Size * this->tilingData->PFAinputParams.gSize;
    this->n2GD = this->tilingData->PFAinputParams.n2Size * this->gD;
    this->gD2 = this->tilingData->PFAinputParams.gSize * this->valueDSize;
    this->n2D2 = this->tilingData->PFAinputParams.n2Size * this->valueDSize;
    this->n2GD2 = this->tilingData->PFAinputParams.n2Size * this->gD2;

    // 计算切分轴的乘积
    this->s2BaseN2D = this->s2BaseSize * this->n2D;
    this->s2BaseNratioSize = this->s2BaseSize * this->tilingData->PFAcoreParams.nRatio;
    this->s1Size = this->tilingData->PFAinputParams.s1Size;
    // layout(bs)ND
    this->s1BaseN2GD = this->cubeS1BaseSize * this->n2GD;
    this->s1BaseN2GD2 = this->cubeS1BaseSize * this->n2GD2;
    this->s2BaseNratioN2D = this->s2BaseN2D * this->tilingData->PFAcoreParams.nRatio;

    // layout(bs)ND
    this->mm1Ka = this->n2GD;
    this->mm1Kb = this->n2D;
    this->mm2Kb = this->n2D2;

    if (this->cubeS1BaseSize > 256 || this->tilingData->PFAinputParams.isSoftMaxLseEnable) {
        this->softmaxReduceSize = 1;
    }
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void
PromptFlashAttentionVarLenScoreSameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                        bmm1Format, mmPolicyType>::GetSeqQlenKvlenByBoidx(int64_t boIdx,
                                                                                            int64_t &actualSeqQlen,
                                                                                            int64_t &actualSeqKvlen)
{
    if (boIdx == 0) {
        actualSeqQlen = qListGm.GetValue(0);
        actualSeqKvlen = kvListGm.GetValue(0);
    } else {
        actualSeqQlen = qListGm.GetValue(boIdx) - qListGm.GetValue(boIdx - 1);
        actualSeqKvlen = kvListGm.GetValue(boIdx) - kvListGm.GetValue(boIdx - 1);
    }
    return;
}

// 初始化s1方向上的累加值
template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                        bmm1Format, mmPolicyType>::CalS1OuterSize(int64_t offset)
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
        actualS1Outersize += (CeilDiv(actualS1Len, this->cubeS1BaseSize) * this->n2G);
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
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T, 
                                                        bmm1Format, mmPolicyType>::ComputeAxisIdx(int64_t multiCoreInnerIdx)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
    int64_t actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(actualS1Len, this->cubeS1BaseSize) * this->n2G);
    while (multiCoreInnerIdx >= actualS1Outersize) {
        this->s1OuterSizeAcc = actualS1Outersize;
        this->s1SizeAcc += actualS1Len;
        this->s2SizeAcc += actualS2Len;
        this->attenB1SSOffset += actualS1Len * actualS2Len;
        this->boIdx++;
        GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
        actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(actualS1Len, this->cubeS1BaseSize) * this->n2G);
    }
    // 计算轴的idx
    int64_t tmpS1Outersize = CeilDiv(actualS1Len, this->cubeS1BaseSize);
    actualS1Outersize = multiCoreInnerIdx - this->s1OuterSizeAcc;
    this->n2oIdx = actualS1Outersize / tmpS1Outersize / this->tilingData->PFAinputParams.gSize;
    this->goIdx = actualS1Outersize / tmpS1Outersize % this->tilingData->PFAinputParams.gSize;
    this->s1oIdx = actualS1Outersize % tmpS1Outersize;
    GetSeqQlenKvlenByBoidx(this->boIdx, this->s1Size, this->s2Size);
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                             bmm1Format, mmPolicyType>::GetS2LoopRange()
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
        this->s2StartIdx = Max(
            this->s1oIdx * this->cubeS1BaseSize - this->tilingData->PFAcoreParams.s1SparseValidSize, 0);
        this->s2EndIdx =
            Min((this->s1oIdx + 1) * this->cubeS1BaseSize + this->tilingData->PFAcoreParams.s2SparseValidSize, actualS2Len);
        // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }
    } else if (this->tilingData->PFAinputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_COMPRESS)) {
        this->s2StartIdx = Max(this->s1oIdx * this->cubeS1BaseSize - actualS1Len +
                                   Max(actualS2Len - this->tilingData->PFAinputParams.preTokens, 0),
                               0);
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len -
                                 Max(actualS1Len - this->tilingData->PFAinputParams.nextTokens, 0),
                             actualS2Len);
        // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
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
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                             bmm1Format, mmPolicyType>::Process()
{
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

    SplitSameABExtraInfo extraInfo[3];
    int64_t taskId = 0;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
         ++multiCoreInnerIdx) {
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
        this->ComputeAxisIdx(multiCoreInnerIdx);
        // s2轴循环计数, 支持sparse和非sparse场景
        this->GetS2LoopRange();
        int64_t s2LoopLimit = CeilDiv(this->s2EndIdx - this->s2StartIdx, this->s2BaseNratioSize) - 1;
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            if (s2LoopCount == 0) {
                this->softmaxPingPongCnt++;
            }
            if (taskId > 0) {
                this->WaitBmm1Result(extraInfo[(taskId + 2) % 3]);
            }
            this->SetExtraInfo(extraInfo[taskId % 3], taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);

            if (extraInfo[taskId % 3].needNz2Nd == 1) {
                this->IterateBmm1(extraInfo[taskId % 3], this->bmm1Nz);
            } else {
                this->IterateBmm1(extraInfo[taskId % 3], this->bmm1);
            }

            if (taskId > 0) {
                this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }

            if (taskId > 1) {
                this->WaitBmm2Result(extraInfo[(taskId + 1) % 3]);
            }

            if (taskId > 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                this->IterateBmm2(extraInfo[(taskId + 2) % 3], this->bmm2);
            }

            if (taskId > 1) {
                this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
            }
            taskId++;
        }
    }
    if (taskId >= 1) {
        this->softMaxCheckRes = SOFTMAX_CHECK_RES_DEFAULT_VALUE;
        this->WaitBmm1Result(extraInfo[(taskId + 2) % 3]);
        this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        if (taskId > 1) {
            this->WaitBmm2Result(extraInfo[(taskId + 1) % 3]);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        this->IterateBmm2(extraInfo[(taskId + 2) % 3], this->bmm2);
        if (taskId > 1) {
            this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
        }
    }
    taskId++;
    if (taskId >= 2) {
        this->WaitBmm2Result(extraInfo[(taskId + 1) % 3]);
        this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
    }
};

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                        bmm1Format, mmPolicyType>::SetExtraInfo(
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
    extraInfo.attenMaskS2Size = extraInfo.s2Size;
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
    this->ComputeBmm1Tail(extraInfo);
}

template <typename TILING_TYPE, ImplModeEnum implMode, LayOutTypeEnum layOutType, bool hasAtten, typename INPUT_T, typename T,
          CubeFormat bmm1Format, MmPolicyType mmPolicyType>
__aicore__ inline void PromptFlashAttentionVarLenScoreSameAB<TILING_TYPE, implMode, layOutType, hasAtten, INPUT_T, T,
                                                        bmm1Format, mmPolicyType>::ComputeBmm1Tail(SplitSameABExtraInfo &extraInfo)
{
    extraInfo.s1RealSize = (extraInfo.cubeS1RealSize + 1) / 2;
    extraInfo.vecCoreOffset = this->cubeSubIdx * extraInfo.s1RealSize;
    if (this->cubeSubIdx == 1) {
        extraInfo.s1RealSize = extraInfo.cubeS1RealSize - extraInfo.s1RealSize;
    }
    extraInfo.s2RealSize = this->s2BaseNratioSize;
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
        extraInfo.vec1S1BaseSize = extraInfo.s1RealSize;
        extraInfo.realSplitN = 1;
    } else {
        extraInfo.vec1S1BaseSize = Min(1024 / extraInfo.s2RealSizeAlign64 * 8, extraInfo.s1RealSize); // Maximize the ub
        extraInfo.realSplitN = CeilDiv(extraInfo.s1RealSize, extraInfo.vec1S1BaseSize);
    }

    if (this->dSizeAlign16 > 64) {
        extraInfo.vec2S1BaseSize = 1024 / this->dSizeAlign16 * 8;
    } else {
        extraInfo.vec2S1BaseSize = Min(128, extraInfo.s1RealSize);
    }
    extraInfo.needNz2Nd = (extraInfo.s2RealSize % 64 == 0) ? 0 : 1;
    return;
}

#endif // PROMPT_FLASH_ATTENTION_VAR_LEN_SCORE_SAB_H
