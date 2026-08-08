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
 * \file prompt_flash_attention_s1s2_bns1_x310.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_S1S2_BNS1_X310_H
#define PROMPT_FLASH_ATTENTION_S1S2_BNS1_X310_H

#include "prompt_flash_attention_s1s2_bns1_x310_base.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
using namespace matmul;
template<typename PFAT>
class PromptFlashAttentionS1s2Bns1X310 : public PromptFlashAttentionS1s2Bns1X310Base<PFAT> {
public:
    using T = typename PFAT::inputType;
    using U = typename PFAT::maskType;
    using O = typename PFAT::outputType;
    // define datatype
    using mmInputType = typename PromptFlashAttentionTypeTraitsNZ<T, PFAT::calcMode>::mmInputType;
    using mmBiasType = typename PromptFlashAttentionTypeTraitsNZ<T, PFAT::calcMode>::mmBiasType;
    using mmOutputType = typename PromptFlashAttentionTypeTraitsNZ<T, PFAT::calcMode>::mmOutputType;
    using mm1OutputType = typename PromptFlashAttentionTypeTraitsNZ<T, PFAT::calcMode>::mm1OutputType;
    using softmaxType = typename PromptFlashAttentionTypeTraitsNZ<T, PFAT::calcMode>::softmaxType;
    __aicore__ inline PromptFlashAttentionS1s2Bns1X310() {};
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ComputeEachCoreSInnerLoop(uint32_t startIndex, uint32_t endIndex);

    __aicore__ inline void SInnerLoopFunc(int32_t startIndex, int32_t endIndex);

    __aicore__ inline void ComputeEachCore(uint32_t coreIdx);

    __aicore__ inline void ComputeSoftmax(LocalTensor<mm1OutputType>& mmResUb, bool isInnerLoopStart);
};

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310<PFAT>::Process() {
    ComputeEachCore(this->tmp_block_idx);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310<PFAT>::SInnerLoopFunc(int32_t startIndex,
                                                                                            int32_t endIndex) {
    if (startIndex < 0) {
        startIndex = 0;
    }
    if (endIndex > this->maxInnerLoopTimes) {
        endIndex = this->maxInnerLoopTimes;
    }

    ComputeEachCoreSInnerLoop(startIndex, endIndex);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310<PFAT>::ComputeSoftmax(LocalTensor<mm1OutputType>& mmResUb,
    bool isInnerLoopStart) {
    LocalTensor<uint8_t> sharedTmpUb = this->tmpSoftmaxFlashV2Ub_.template Get<uint8_t>(this->softMaxV2Size_);

    struct SoftMaxShapeInfo softmaxShapeInfo;
    softmaxShapeInfo.srcM = this->isOuterTail_ ? this->singleProcessSOuterSizeTailAlign : this->singleProcessSOuterSize;
    softmaxShapeInfo.srcK = this->isInnerLoopLast_? this->singleProcessSInnerSizeTailAlign : this->singleProcessSInnerSize;

    softmaxShapeInfo.oriSrcM = this->isOuterTail_ ? this->singleProcessSOuterSizeTail : this->singleProcessSOuterSize;
    softmaxShapeInfo.oriSrcK = this->isInnerLoopLast_ ? this->singleProcessSInnerSizeTail : this->singleProcessSInnerSize;

    if (!this->isHighPrecision_){
        LocalTensor<mmOutputType> tmpResUb = mmResUb.template ReinterpretCast<mmOutputType>();
        this->softmaxMaxUb16_ = this->softmaxOutQueue.template AllocTensor<mmOutputType>();
        this->softmaxSumUb16_ = this->softmaxMaxUb16_[this->softmaxMaxSize];
        this->softmaxExpUb16_ =  this->softmaxExpUb.template ReinterpretCast<mmOutputType>();
        if (isInnerLoopStart) {
            this->SoftmaxBasicComputeFirstTailTmp(tmpResUb, this->softmaxMaxUb16_, this->softmaxSumUb16_, this->softmaxExpUb16_,
                sharedTmpUb, softmaxShapeInfo);
        } else {
            this->SoftmaxBasicComputeTailTmp(tmpResUb, this->softmaxMaxUb16_, this->softmaxSumUb16_, this->softmaxExpUb16_,
                  sharedTmpUb, softmaxShapeInfo);
        }
    } else {
        this->softmaxMaxUb32_ = this->softmaxOutQueue.template AllocTensor<float>();
        this->softmaxSumUb32_ = this->softmaxMaxUb32_[this->softmaxMaxSize];
        if (isInnerLoopStart) {
            this->SoftmaxBasicComputeFirstTail(mmResUb, this->softmaxMaxUb32_, this->softmaxSumUb32_, this->softmaxExpUb,
                sharedTmpUb, softmaxShapeInfo);
        } else {
            this->SoftmaxBasicComputeTail(mmResUb, this->softmaxMaxUb32_, this->softmaxSumUb32_, this->softmaxExpUb,
                sharedTmpUb,softmaxShapeInfo);
        }
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310<PFAT>::ComputeEachCoreSInnerLoop(uint32_t startIndex,
                                                                                uint32_t endIndex) {
    this->a1Local_ = this->a1Buf_.template Get<mmInputType>();
    this->b1Local_ = this->b1Buf_.template Get<mmInputType>();
    int32_t outerSize, innerSize;
    // step 1 fetch and compute bmm1
    if (this->isOuterLoopStart_) {
        if (this->needCalMask_) {
            this->AttenMaskCopyIn(this->attenMaskCoreOffset, this->singleProcessSInnerSize, 0);
        }
        outerSize = this->isOuterTail_ ? this->singleProcessSOuterSizeTail : this->singleProcessSOuterSize;
        this->fetchOuterSize_ = outerSize; // fetch outersize
        // L1 residency SetTensorA to Obtain L1
        this->CopyND2NZOnTheFly(this->a1Local_, this->queryGm[this->tensorACoreOffset], outerSize, 
            this->tilingData->promptAttentionBaseParams.headSize, this->queryStride, true);
        
        this->isInnerLoopLast_ = (startIndex == endIndex - 1);
        innerSize = this->isInnerLoopLast_ ? this->singleProcessSInnerSizeTail : this->singleProcessSInnerSize;
        this->CopyND2NZOnTheFly(this->b1Local_, this->keyGm[this->tensorBCoreOffset], innerSize, 
            this->tilingData->promptAttentionBaseParams.headSize, this->keyValueStride, true);       
        this->Bmm1Compute(this->a1Local_, this->b1Local_, outerSize, innerSize, this->tilingData->promptAttentionBaseParams.headSize);
    }
    this->isSoftmaxResNeedUpdate = this->tilingData->promptAttentionBaseParams.isRowInvalid;
    for (int64_t sInnerLoopIdx = startIndex; sInnerLoopIdx < endIndex; sInnerLoopIdx++) {
        this->isInnerLoopLast_ = sInnerLoopIdx == endIndex - 1;
        this->isNextInnerLoopLast_ = (endIndex - startIndex == 1) || ((endIndex - startIndex) > 1 && (sInnerLoopIdx == endIndex - 2));
        LocalTensor<mm1OutputType> mmResUb = this->Bmm1Queue.template AllocTensor<mm1OutputType>();

        this->maskCopyInCol = this->singleProcessSInnerSize;

        this->mm.template GetTensorC<false>(mmResUb, false, true);
        this->mm.End();

        // step 2 muls scale
        outerSize = this->isOuterTail_ ? this->singleProcessSOuterSizeTail : this->singleProcessSOuterSize;
        innerSize = this->isInnerLoopLast_ ? this->singleProcessSInnerSizeTail : this->singleProcessSInnerSize;
        this->ElewiseCompute310P(mmResUb, this->singleProcessSInnerSize, this->singleProcessSOuterSize);
        PipeBarrier<PIPE_V>();
        // softmax compute
        bool isInnerLoopStart = sInnerLoopIdx == startIndex;
        this->ComputeOffset(sInnerLoopIdx, this->isInnerLoopLast_);
        this->ComputeSoftmax(mmResUb, isInnerLoopStart);
     
        LocalTensor<mmOutputType> bmm2ResUb;
        if (isInnerLoopStart) {
            bmm2ResUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);
        } else {
            bmm2ResUb = this->tempBmm2Queue.template AllocTensor<mmOutputType>();
        }
        // step4: compute bmm2
        if (!(this->isInnerLoopLast_ && this->isOuterLoopLast_)) {
            if (this->needCalMask_) {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                this->AttenMaskCopyIn(this->attenMaskOffset, this->singleProcessSInnerSize, 0);
            }           
        }
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        // Copy valueGM to L1
        this->c1Local_ = this->c1Buf_.template Get<mmInputType>();
        this->CopyND2NZOnTheFly(this->c1Local_, this->valueGm[this->valueOffset], innerSize, 
            this->tilingData->promptAttentionBaseParams.headSize, this->keyValueStride, true);
        
        if constexpr (IsSameType<mm1OutputType, float>::value) {
            event_t eventIDV_MTE1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE1));
           
            LocalTensor<mmOutputType> tmpSoftmaxResUb = mmResUb.template ReinterpretCast<mmOutputType>();
            Cast(tmpSoftmaxResUb, mmResUb, RoundMode::CAST_NONE, tmpSoftmaxResUb.GetSize());
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE1>(eventIDV_MTE1);
            WaitFlag<HardEvent::V_MTE1>(eventIDV_MTE1);
            this->Bmm2Compute(tmpSoftmaxResUb);
            event_t eventIDMTE1_MTE3= static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE3));
            SetFlag<HardEvent::MTE1_MTE3>(eventIDMTE1_MTE3);
            WaitFlag<HardEvent::MTE1_MTE3>(eventIDMTE1_MTE3);
        } else{
            this->Bmm2Compute(mmResUb);
        }
        this->bmm2.template Iterate<false>();
        this->bmm2.template GetTensorC<false>(bmm2ResUb, false, true);
        this->Bmm1Queue.FreeTensor(mmResUb);

        uint32_t sInnerOffset = (sInnerLoopIdx + 1) * this->singleProcessSInnerSize;
        this->tensorBOffset = this->tensorBCoreOffset + sInnerOffset * this->keyValueStride;
        if (this->isInnerLoopLast_ && !this->isOuterLoopLast_) {
            this->tensorAOffset = this->tensorACoreOffset +
                this->singleProcessSOuterSize * this->queryStride;
            outerSize = this->isNextOuterLoopLast_ ? this->singleProcessSOuterSizeTail : this->singleProcessSOuterSize;
            this->fetchOuterSize_ = outerSize;  // last innerloop fetch nextloop outersize
            this->CopyND2NZOnTheFly(this->a1Local_, this->queryGm[this->tensorAOffset], outerSize, 
                this->tilingData->promptAttentionBaseParams.headSize, this->queryStride, true);
            this->tensorBOffset = this->tensorBCoreOffset + 
                startIndex * this->keyValueStride * this->singleProcessSInnerSize;
        }
        // pre compute mm1 right matrix if not the tile
        if (!(this->isInnerLoopLast_ && this->isOuterLoopLast_)) {
            innerSize = this->isNextInnerLoopLast_ ? this->singleProcessSInnerSizeTail : this->singleProcessSInnerSize;
            this->CopyND2NZOnTheFly(this->b1Local_, this->keyGm[this->tensorBOffset], innerSize, 
                this->tilingData->promptAttentionBaseParams.headSize, this->keyValueStride, true);
            this->Bmm1Compute(this->a1Local_, this->b1Local_, this->fetchOuterSize_, innerSize, this->tilingData->promptAttentionBaseParams.headSize);
        }
        // Step 6: bmm2 update and copyout
        if (!isInnerLoopStart) {
            this->UpdateVmul(this->softmaxExpUb);
            this->Bmm2UpdateAdd(bmm2ResUb);
            this->tempBmm2Queue.FreeTensor(bmm2ResUb);
        }
        // last inner loop compute div and copy out
        if (this->isInnerLoopLast_) {
            PipeBarrier<PIPE_V>();
            LocalTensor<mmOutputType> bmm2ResPreUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);
            if (!this->isHighPrecision_) {
                this->Bmm2UpdateDivNoTail310PTmp(bmm2ResPreUb, this->softmaxSumUb16_, this->softmaxExpUb);
                this->softmaxOutQueue.FreeTensor(this->softmaxMaxUb16_);
            } else {
                this->Bmm2UpdateDivNoTail310P(bmm2ResPreUb, this->softmaxSumUb32_, this->softmaxExpUb);
                this->softmaxOutQueue.FreeTensor(this->softmaxMaxUb32_);
            }
            LocalTensor<mmOutputType> FinalResUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);

            this->DataCopyTransposeOut(FinalResUb);
        } else {
            if (!this->isHighPrecision_) {
                this->softmaxOutQueue.FreeTensor(this->softmaxMaxUb16_);
            } else {
                this->softmaxOutQueue.FreeTensor(this->softmaxMaxUb32_);
            }
        }
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310<PFAT>::ComputeEachCore(uint32_t coreIdx) {
    this->spmTmpSize = this->tilingData->promptAttentionTensorSizeRect.spmTmpSize;
    this->mmResUbSize = this->tilingData->promptAttentionTensorSizeRect.mmResUbSize;
    this->bmm2ResUbSize = this->tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize;

    int actualCoreNums = this->tilingData->promptAttentionSingleCoreParams.actualCoreNums;
    if (coreIdx >= actualCoreNums) {
        return;
    }
    int sNum = this->tilingData->promptAttentionBaseParams.dimNumOfseq;

    // Temporary reuse
    // CoreHeadNumTail to coreNStart
    // actualS1 to coreNEnd
    // actualCoreNums to coreSidStart
    // singleCoreHeadNumSize to coreSidEnd
    int sIdStart = this->tilingData->promptAttentionSeqParams.actualCoreNums[coreIdx];
    int sIdEnd = this->tilingData->promptAttentionSeqParams.singleCoreHeadNumSize[coreIdx];
    int outerLoopStart = this->tilingData->promptAttentionSeqParams.coreSeqPosStart[coreIdx];
    int outerLoopEnd = this->tilingData->promptAttentionSeqParams.coreSeqPosEnd[coreIdx];
    uint32_t nLoopStart = this->tilingData->promptAttentionSeqParams.CoreHeadNumTail[coreIdx];
    uint32_t nLoopEnd = this->tilingData->promptAttentionSeqParams.actualS1[coreIdx];
    int32_t preTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.preTokens);
    int32_t nextTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.nextTokens);

    uint32_t actualSeqLengthsIdx = 0;

    for (uint32_t loopNIdx = nLoopStart; loopNIdx < nLoopEnd; loopNIdx++) {
        this->batchNOffset = loopNIdx;
        int32_t sLoopEndIdx = 0;
        bool isNLoopLast = false;
        if (loopNIdx != nLoopEnd - 1) {
            sLoopEndIdx = sNum;
        } else {
            sLoopEndIdx = sIdEnd;
            isNLoopLast = true;
        }
        for (int sIdx = sIdStart; sIdx < sLoopEndIdx; sIdx++) {
            this->GetSingleCoreParam(sIdx);
            actualSeqLengthsIdx = this->isActualLenDimsNull ? this->tilingData->promptAttentionBaseParams.seqSize : this->actualSeqLengthsGm.GetValue(sIdx);
            actualSeqLengthsIdx = (this->attentionMaskType == 0 && (int64_t)actualSeqLengthsIdx >
                               (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize +
                               (int64_t)this->tilingData->promptAttentionBaseParams.preTokens) ?
                               this->tilingData->promptAttentionBaseParams.seqInnerSize + this->tilingData->promptAttentionBaseParams.preTokens :
                               actualSeqLengthsIdx;
            int sOuterBlockNum = (actualSeqLengthsIdx + this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize - 1) /
                                  this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
            this->multiSeqOffset = this->actualSeqOffsets[sIdx];
            int32_t queryHeadEndOffset = this->multiSeqOffset + (loopNIdx + 1) * this->tilingData->promptAttentionBaseParams.seqSize * this->queryStride;
            int32_t tmpOuterLoopEnd = 0;
            if (isNLoopLast && sIdx == sLoopEndIdx - 1) { // N last && S last
                tmpOuterLoopEnd = outerLoopEnd;
            } else {
                tmpOuterLoopEnd = sOuterBlockNum;
            }
            for (uint32_t sOuterLoopIdx = outerLoopStart; sOuterLoopIdx < tmpOuterLoopEnd; sOuterLoopIdx++) {
                this->isOuterLoopLast_ = sOuterLoopIdx == tmpOuterLoopEnd - 1;
                // start outer loop caculate in one core
                this->isOuterLoopStart_ = sOuterLoopIdx == outerLoopStart;
                this->isOuterTail_ = sOuterLoopIdx == sOuterBlockNum - 1;
                // record second last outer loop
                this->isNextOuterLoopLast_ = (tmpOuterLoopEnd - outerLoopStart) > 1 && (sOuterLoopIdx == sOuterBlockNum - 2);
                this->sOuterOffset = sOuterLoopIdx * this->singleProcessSOuterSize;
                int32_t start_idx = (this->sOuterOffset - preTokens) / (int32_t)(this->singleProcessSInnerSize);
                int32_t end_idx = (this->sOuterOffset + nextTokens + this->singleProcessSOuterSize +
                                  (int32_t)(this->singleProcessSInnerSize) - 1) /
                                  (int32_t)(this->singleProcessSInnerSize);
                this->LoopSOuterOffsetInit(this->actualSeqOffsets[sIdx], sIdx);
                SInnerLoopFunc(start_idx, end_idx);
            }
            outerLoopStart = 0;
        }
        sIdStart = 0;
    }
}
#endif  // PROMPT_FLASH_ATTENTION_S1S2_BNS1_X310_H