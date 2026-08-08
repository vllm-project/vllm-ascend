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
 * \file prompt_flash_attention_bnstilling_n_s_no_tail.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_SCORE_BNSTILLING_N_S_NO_TAIL_H
#define PROMPT_FLASH_ATTENTION_SCORE_BNSTILLING_N_S_NO_TAIL_H

#include "prompt_flash_attention_base.h"

using namespace matmul;
template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M = OptimizationMode::HighPerformance>
class PromptFlashAttentionBNSTillingNSNoTail : public PromptFlashAttentionBase<T, U, FORMAT, O, M> {
public:
    // define datatype
    using mmInputType = typename PromptFlashAttentionTypeTraits<T, M>::mmInputType;
    using mmBiasType = typename PromptFlashAttentionTypeTraits<T, M>::mmBiasType;
    using mmOutputType = typename PromptFlashAttentionTypeTraits<T, M>::mmOutputType;
    using softmaxType = typename PromptFlashAttentionTypeTraits<T, M>::softmaxType;
    using pseShiftType = typename PromptFlashAttentionTypeTraits<T, M>::pseShiftType;
    using pseShiftCastType = typename PromptFlashAttentionTypeTraits<T, M>::pseShiftCastType;
    __aicore__ inline PromptFlashAttentionBNSTillingNSNoTail() {};
    __aicore__ inline void Process();

protected:
    __aicore__ inline void AttenMaskCopyIn(uint64_t offset, uint32_t sinnerSize, uint32_t sInnerIdx);

    __aicore__ inline void PseShiftCopyIn(uint64_t offset, uint32_t sinnerSize, uint32_t sInnerLoopIdx);

    __aicore__ inline void Bmm1ResDoVecBmm2Compute(LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
                                            LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb,
                                            bool isLast, bool isSecond, event_t eventID, int64_t sInnerLoopIdx);

    __aicore__ inline void PseShiftProcess(int64_t sInnerLoopIdx, uint32_t computeSize, LocalTensor<mmOutputType>& mmResUb);

    __aicore__ inline void Bmm1ResDoVecBmm2ComputeFirst(LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
                                                        LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb,
                                                        bool isLast, event_t eventID, int64_t sInnerLoopIdx);

    __aicore__ inline void ComputeEachCoreSInnerLoop(uint32_t startIndex, uint32_t endIndex);

    __aicore__ inline void SInnerLoopFunc(int32_t startIndex, int32_t endIndex);

    __aicore__ inline void ComputeEachCore(uint32_t coreIdx);

    __aicore__ inline void ComputeEachCoreBalance(uint32_t coreIdx);
};

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBNSTillingNSNoTail<T, U, FORMAT, O, M>::Process() {
    if (this->headNumRatio != 1 || this->tilingData->promptAttentionInitOutputParams.needInit ||
        this->tilingData->promptAttentionBaseParams.batchSize != 1) {
        ComputeEachCore(this->tmp_block_idx);
    }
    else {
        ComputeEachCoreBalance(this->tmp_block_idx);
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBNSTillingNSNoTail<T, U, FORMAT, O, M>::PseShiftCopyIn(uint64_t offset,
                                                                                     uint32_t sinnerSize,
                                                                                     uint32_t sInnerLoopIdx) {
    if (!(this->usePseShift)) {
        return;
    }
    LocalTensor<pseShiftType> pseShiftUb = this->attenMaskQueue.template AllocTensor<pseShiftType>();
    pseShiftUb.SetSize(this->singleProcessSOuterSize * sinnerSize);

    DataCopyParams intriParams;
    intriParams.blockCount = this->singleProcessSOuterSize;
    intriParams.blockLen = sinnerSize / this->pseShiftTypeByteNum;
    intriParams.srcStride = (this->pseShiftStride - sinnerSize) / this->pseShiftTypeByteNum;
    intriParams.dstStride = 0;

    DataCopy(pseShiftUb, this->pseShiftGm[offset], intriParams);
    this->attenMaskQueue.EnQue(pseShiftUb);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBNSTillingNSNoTail<T, U, FORMAT, O, M>::AttenMaskCopyIn(uint64_t offset,
                                                                                     uint32_t sinnerSize,
                                                                                     uint32_t sInnerLoopIdx) {
    if (this->useMask == false) { // Early return if mask is not used
        return;
    }
    LocalTensor<U> attenMaskUb = this->attenMaskQueue.template AllocTensor<U>();
    attenMaskUb.SetSize(this->singleProcessSOuterSize * sinnerSize);
    DataCopyParams intriParams; // Set up parameters for the data copy operation.
    intriParams.blockCount = this->singleProcessSOuterSize;
    intriParams.blockLen = sinnerSize / this->maskTypeByteNum;
    intriParams.srcStride = (this->attentionMaskStride - sinnerSize) /
                            this->maskTypeByteNum;
    intriParams.dstStride = 0;

    DataCopy(attenMaskUb, this->attenMaskGm[offset], intriParams);
    this->attenMaskQueue.EnQue(attenMaskUb);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBNSTillingNSNoTail<T, U, FORMAT, O, M>::PseShiftProcess(int64_t sInnerLoopIdx,
    uint32_t computeSize, LocalTensor<mmOutputType>& mmResUb) {
    if (this->usePseShift) {
        this->PseShiftCopyIn(this->pseShiftOffset, this->pseShiftCopyInCol, sInnerLoopIdx);
        LocalTensor<pseShiftType> pseShiftUb = this->attenMaskQueue.template DeQue<pseShiftType>();
        if constexpr (AscendC::IsSameType<pseShiftCastType, float>::value) {
            LocalTensor<float> pseShiftCastTensor = this->pseShiftCastUb.template Get<float>(this->pseShiftUbSize);
            Cast(pseShiftCastTensor, pseShiftUb, RoundMode::CAST_NONE, computeSize);
            PipeBarrier<PIPE_V>();
            Add(mmResUb, mmResUb, pseShiftCastTensor, computeSize);
        } else { // api add pseShiftUb to mmResUb
            Add(mmResUb, mmResUb, pseShiftUb, computeSize);
        }

        PipeBarrier<PIPE_V>();
        this->attenMaskQueue.FreeTensor(pseShiftUb);
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBNSTillingNSNoTail<T, U, FORMAT, O, M>::Bmm1ResDoVecBmm2ComputeFirst(LocalTensor<mmOutputType>& mmResUb,
                                            LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                            LocalTensor<softmaxType>& softmaxExpUb, bool isLast, event_t eventID, int64_t sInnerLoopIdx) {
    WaitFlag<HardEvent::MTE3_MTE2>(eventID);
    this->mm.template GetTensorC<false>(mmResUb, false, false);

    uint32_t computeSize = this->singleProcessSInnerSizeNow * this->singleProcessSOuterSize;

    Muls(mmResUb, mmResUb, static_cast<mmOutputType>(this->tilingData->promptAttentionBaseParams.scaleValue), computeSize);
    PipeBarrier<PIPE_V>();

    this->PseShiftProcess(sInnerLoopIdx, computeSize, mmResUb);

    this->AttenMaskCopyIn(this->attenMaskOffset, this->maskCopyInCol, sInnerLoopIdx);

    if(this->attentionMaskType == 4){ //   4:band mode of sparseMode
        this->ElewiseCompute(mmResUb, computeSize, 0);
        this->AttenMaskCopyIn(this->attenMaskOffsetPre, this->maskCopyInCol, sInnerLoopIdx);
        this->ElewiseCompute(mmResUb, computeSize, 1);
    } else {
        this->ElewiseCompute(mmResUb, computeSize, 0);
    }

    PipeBarrier<PIPE_V>();

    SoftMaxShapeInfo shapeInfo = {this->singleProcessSOuterSize, this->singleProcessSInnerSize,
                                  this->singleProcessSOuterSize, this->singleProcessSInnerSize};
    if (this->IsSoftmaxFlashBasic()) {
        this->SoftmaxBasicComputeFirstNoTail(mmResUb, softmaxMaxUb, softmaxSumUb, softmaxExpUb, shapeInfo);
    } else {
        this->SoftmaxComputeFirstTail(mmResUb, softmaxMaxUb, softmaxSumUb, softmaxExpUb, shapeInfo);
    }

    this->Bmm2Compute(this->valueOffset, mmResUb);
    this->bmm2.template Iterate<false>();

    if (isLast) {
        LocalTensor<mmOutputType> bmm2ResUb = this->tempBmm2Queue.template AllocTensor<mmOutputType>();
        this->bmm2.template GetTensorC<false>(bmm2ResUb, false, false);
        this->bmm2.End();

        this->Bmm2UpdateDivNoTail(bmm2ResUb, softmaxSumUb, softmaxExpUb);

        this->tempBmm2Queue.EnQue(bmm2ResUb);
        bmm2ResUb = this->tempBmm2Queue.template DeQue<mmOutputType>();

        this->DataCopyTransposeOut(bmm2ResUb);
        this->tempBmm2Queue.FreeTensor(bmm2ResUb);
    }

    SetFlag<HardEvent::MTE3_MTE2>(eventID);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBNSTillingNSNoTail<T, U, FORMAT, O, M>::Bmm1ResDoVecBmm2Compute(LocalTensor<mmOutputType>& mmResUb,
                                            LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                            LocalTensor<softmaxType>& softmaxExpUb, bool isLast,
                                            bool isSecond, event_t eventID, int64_t sInnerLoopIdx) {
    WaitFlag<HardEvent::MTE3_MTE2>(eventID);
    this->mm.template GetTensorC<false>(mmResUb, false, false);

    uint32_t computeSize = this->singleProcessSInnerSizeNow * this->singleProcessSOuterSize;

    Muls(mmResUb, mmResUb, static_cast<mmOutputType>(this->tilingData->promptAttentionBaseParams.scaleValue), computeSize);
    PipeBarrier<PIPE_V>();

    this->PseShiftProcess(sInnerLoopIdx, computeSize, mmResUb);

    this->AttenMaskCopyIn(this->attenMaskOffset, this->maskCopyInCol, sInnerLoopIdx);

    if(this->attentionMaskType == 4){ //    4:band mode of sparseMode
        this->ElewiseCompute(mmResUb, computeSize, 0);
        this->AttenMaskCopyIn(this->attenMaskOffsetPre, this->maskCopyInCol, sInnerLoopIdx);
        this->ElewiseCompute(mmResUb, computeSize, 1);
    } else {
        this->ElewiseCompute(mmResUb, computeSize, 0);
    }
    PipeBarrier<PIPE_V>();

    SoftMaxShapeInfo shapeInfo = {this->singleProcessSOuterSize, this->singleProcessSInnerSize,
                                  this->singleProcessSOuterSize, this->singleProcessSInnerSize};
    if (this->IsSoftmaxFlashBasic()) {
        this->SoftmaxBasicComputeNoTail(mmResUb, softmaxMaxUb, softmaxSumUb, softmaxExpUb, shapeInfo);
    } else {
        this->SoftmaxComputeTail(mmResUb, softmaxMaxUb, softmaxSumUb, softmaxExpUb, shapeInfo);
    }

    LocalTensor<mmOutputType> bmm2ResUb;
    if (isSecond) {
        bmm2ResUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);
    } else {
        bmm2ResUb = this->tempBmm2Queue.template AllocTensor<mmOutputType>();
    }

    this->bmm2.template GetTensorC<false>(bmm2ResUb, false, false);
    this->bmm2.End();
    this->Bmm2Compute(this->valueOffset, mmResUb);
    this->bmm2.template Iterate<false>();

    if (!isSecond) {
        this->Bmm2UpdateAdd(bmm2ResUb);
        this->tempBmm2Queue.FreeTensor(bmm2ResUb);
    }

    this->UpdateVmul(softmaxExpUb);

    if (isLast) {
        bmm2ResUb = this->tempBmm2Queue.template AllocTensor<mmOutputType>();
        this->bmm2.template GetTensorC<false>(bmm2ResUb, false, false);
        this->bmm2.End();
        this->Bmm2UpdateAdd(bmm2ResUb);

        PipeBarrier<PIPE_V>();
        LocalTensor<mmOutputType> bmm2ResPreUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);
        this->Bmm2UpdateDivNoTail(bmm2ResPreUb, softmaxSumUb, softmaxExpUb);

        this->tempBmm2Queue.FreeTensor(bmm2ResUb);
        LocalTensor<mmOutputType> FinalResUb = this->tempBmm2Ub.template Get<mmOutputType>(this->bmm2ResUbSize);

        this->DataCopyTransposeOut(FinalResUb);
    }

    SetFlag<HardEvent::MTE3_MTE2>(eventID);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBNSTillingNSNoTail<T, U, FORMAT, O, M>::SInnerLoopFunc(int32_t startIndex,
                                                                                    int32_t endIndex) {
    if (endIndex > this->maxInnerLoopTimes) {
        endIndex = this->maxInnerLoopTimes;
    }

    if (startIndex < 0) {
        startIndex = 0;
    }
    
    int sInnerLoopTimes = endIndex - startIndex;
    if (sInnerLoopTimes <= 0) {
        return;
    }
    this->tensorAOffset = this->tensorACoreOffset;
    this->tensorBOffset = this->tensorBCoreOffset + startIndex * this->singleProcessSInnerSize * this->MultiHeadKV;

    this->mm.SetTensorA(this->queryGm[this->tensorAOffset]);
    this->mm.SetTensorB(this->keyGm[this->tensorBOffset], true);
    // quant:
    if constexpr (IsSameType<mmInputType, int8_t>::value) {
        this->mm.SetQuantScalar(this->dequantScale1);
    }

    int curS = this->singleProcessSInnerSize * sInnerLoopTimes;
    this->mm.SetTail(this->singleProcessSOuterSize, curS);
    this->mm.template Iterate<false>();

    ComputeEachCoreSInnerLoop(startIndex, endIndex);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBNSTillingNSNoTail<T, U, FORMAT, O, M>::ComputeEachCoreSInnerLoop(uint32_t startIndex,
                                                                                uint32_t endIndex) {
    bool isSecond = true;
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID);
    for (int64_t sInnerLoopIdx = startIndex; sInnerLoopIdx < endIndex; sInnerLoopIdx++) {
        LocalTensor<float> softmaxMaxUb = this->softmaxOutQueue.template AllocTensor<float>();
        LocalTensor<float> softmaxSumUb = softmaxMaxUb[this->softmaxMaxSize];
        this->ComputeOffset(sInnerLoopIdx);
        LocalTensor<mmOutputType> mmResUb = this->Bmm1Queue.template AllocTensor<mmOutputType>();

        this->singleProcessSInnerSizeNow = this->singleProcessSInnerSize;
        this->singleProcessSInnerBmmTail = this->singleProcessSInnerSize;
        this->maskCopyInCol = this->singleProcessSInnerSize;
        this->pseShiftCopyInCol = this->singleProcessSInnerSize;

        bool isLast = sInnerLoopIdx == endIndex - 1;

        if (sInnerLoopIdx == startIndex) {
            Bmm1ResDoVecBmm2ComputeFirst(mmResUb, softmaxMaxUb, softmaxSumUb, this->softmaxExpUb, isLast, eventID, sInnerLoopIdx);
        } else {
            Bmm1ResDoVecBmm2Compute(mmResUb, softmaxMaxUb, softmaxSumUb, this->softmaxExpUb, isLast, isSecond, eventID, sInnerLoopIdx);
            isSecond = false;
        }
        this->Bmm1Queue.FreeTensor(mmResUb);
        this->softmaxOutQueue.FreeTensor(softmaxMaxUb);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(eventID);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBNSTillingNSNoTail<T, U, FORMAT, O, M>::ComputeEachCore(uint32_t coreIdx) {
    this->spmTmpSize = this->tilingData->promptAttentionTensorSizeRect.spmTmpSize;
    this->mmResUbSize = this->tilingData->promptAttentionTensorSizeRect.mmResUbSize;
    this->bmm2ResUbSize = this->tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize;
    int reuseWorkspaceRatio = this->tilingData->promptAttentionSingleCoreParams.multiSmaxsInnerLoopTimes;
    this->mm.SetWorkspace((__gm__ uint8_t*)this->workspaceGm[GetBlockNum() * GetTaskRation() * this->spmTmpSize +
        coreIdx * this->mmResUbSize * reuseWorkspaceRatio].GetPhyAddr(), this->mmResUbSize * reuseWorkspaceRatio);

    uint32_t buff_offset = GetBlockNum() * GetTaskRation() *
                           (this->spmTmpSize + this->mmResUbSize * reuseWorkspaceRatio);
    this->bmm2.SetWorkspace((__gm__ uint8_t*)this->workspaceGm[buff_offset + coreIdx * this->bmm2ResUbSize].GetPhyAddr(), this->bmm2ResUbSize);

    int actualCoreNums = this->tilingData->promptAttentionSingleCoreParams.actualCoreNums;
    if (g_coreType == AIV && coreIdx >= actualCoreNums) {
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

    int tmpOuterLoopEnd;
    int tmpSLoopEnd;
    bool isLast = false;
    uint32_t actualSeqLengthsIdx = 0;

    for (uint32_t loopNIdx = nLoopStart; loopNIdx < nLoopEnd; loopNIdx++) {
        this->batchNOffset = loopNIdx;
        if (loopNIdx != nLoopEnd - 1) {
            tmpSLoopEnd = sNum;
        } else {
            tmpSLoopEnd = sIdEnd;
            isLast = true;
        }
        for (int sIdx = sIdStart; sIdx < tmpSLoopEnd; sIdx++) {
            this->GetSingleCoreParam(sIdx);
            this->GetSparseParam(&preTokens, &nextTokens);
            actualSeqLengthsIdx = this->isActualLenDimsNull ? this->tilingData->promptAttentionBaseParams.seqSize : this->actualSeqLengthsGm.GetValue(sIdx);
            actualSeqLengthsIdx = (this->attentionMaskType == 0 && (int64_t)actualSeqLengthsIdx >
                               (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize +
                               (int64_t)this->tilingData->promptAttentionBaseParams.preTokens) ?
                                this->tilingData->promptAttentionBaseParams.seqInnerSize +
                                this->tilingData->promptAttentionBaseParams.preTokens :
                                actualSeqLengthsIdx;
            int sOuterBlockNum = (actualSeqLengthsIdx +
                                  this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize - 1) /
                                  this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
            this->multiSeqOffset = this->actualSeqOffsets[sIdx];
            if (isLast && sIdx == tmpSLoopEnd - 1) {
                tmpOuterLoopEnd = outerLoopEnd;
            } else {
                tmpOuterLoopEnd = sOuterBlockNum;
            }
            this->singleProcessSOuterSize = this->singleProcessSOuterSizeWhole;
            for (uint32_t sOuterLoopIdx = outerLoopStart; sOuterLoopIdx < tmpOuterLoopEnd; sOuterLoopIdx++) {
                this->sOuterOffset = sOuterLoopIdx * this->singleProcessSOuterSizeWhole;
                this->ComputeTokenOffset();
                if (nextTokens < 0 && this->sOuterOffset < ((nextTokens * (-1)) /
                    this->singleProcessSOuterSizeWhole * this->singleProcessSOuterSizeWhole)) {
                    continue;
                }
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

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBNSTillingNSNoTail<T, U, FORMAT, O, M>::ComputeEachCoreBalance(uint32_t coreIdx) {
    int sNum = this->tilingData->promptAttentionBaseParams.dimNumOfseq;
    if (sNum == 0) {
	    return;
    }
    int32_t blockNum = GetBlockNum() * GetTaskRation();
    if (coreIdx % 2 == 1) {
        coreIdx = blockNum - coreIdx;
    }
    this->spmTmpSize = this->tilingData->promptAttentionTensorSizeRect.spmTmpSize;
    this->mmResUbSize = this->tilingData->promptAttentionTensorSizeRect.mmResUbSize;
    this->bmm2ResUbSize = this->tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize;
    int reuseWorkspaceRatio = this->tilingData->promptAttentionSingleCoreParams.multiSmaxsInnerLoopTimes;
    this->mm.SetWorkspace((__gm__ uint8_t*)this->workspaceGm[blockNum * this->spmTmpSize +
        coreIdx * this->mmResUbSize * reuseWorkspaceRatio].GetPhyAddr(), this->mmResUbSize * reuseWorkspaceRatio);

    uint32_t buff_offset = blockNum *
                           (this->spmTmpSize + this->mmResUbSize * reuseWorkspaceRatio);
    this->bmm2.SetWorkspace((__gm__ uint8_t*)this->workspaceGm[buff_offset + coreIdx * this->bmm2ResUbSize].GetPhyAddr(), this->bmm2ResUbSize);

    int32_t preTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.preTokens);
    int32_t nextTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.nextTokens);

    int32_t tilingIdx = coreIdx;
    int32_t sIdx = 0;
    uint32_t actualSeqLengthsIdx = 0;
    int32_t totalTilingN = this->accumSOuterTilingNums[sNum - 1];
    int32_t preAccumSOuterNum = 0;
    while (tilingIdx < totalTilingN) {
        this->batchNOffset = tilingIdx % this->tilingData->promptAttentionBaseParams.headNumSize;
        while (this->accumSOuterTilingNums[sIdx] < tilingIdx) {
            sIdx++;
        }
        if (sIdx != 0) {
            preAccumSOuterNum = this->accumSOuterTilingNums[sIdx - 1];
        }
        this->GetSingleCoreParam(sIdx);
        this->GetSparseParam(&preTokens, &nextTokens);
        actualSeqLengthsIdx = this->isActualLenDimsNull ? this->tilingData->promptAttentionBaseParams.seqSize : this->actualSeqLengthsGm.GetValue(sIdx);
        actualSeqLengthsIdx = (this->attentionMaskType == 0 && (int64_t)actualSeqLengthsIdx >
                               (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize +
                               (int64_t)this->tilingData->promptAttentionBaseParams.preTokens) ?
                            this->tilingData->promptAttentionBaseParams.seqInnerSize + this->tilingData->promptAttentionBaseParams.preTokens :
                            actualSeqLengthsIdx;
        int sOuterBlockNum = (actualSeqLengthsIdx +
                              this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize - 1) /
                              this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
        this->multiSeqOffset = this->actualSeqOffsets[sIdx];
        this->singleProcessSOuterSize = this->singleProcessSOuterSizeWhole;

        uint32_t sOuterLoopIdx = sOuterBlockNum - 1
             - ((tilingIdx - preAccumSOuterNum) /
                this->tilingData->promptAttentionBaseParams.headNumSize);

        this->sOuterOffset = sOuterLoopIdx * this->singleProcessSOuterSizeWhole;
        this->ComputeTokenOffset();
        if (nextTokens < 0 && this->sOuterOffset < ((nextTokens * (-1)) /
            this->singleProcessSOuterSizeWhole * this->singleProcessSOuterSizeWhole)) {
            continue;
        }

        int32_t start_idx = (this->sOuterOffset - preTokens) / (int32_t)(this->singleProcessSInnerSize);
        int32_t end_idx = (this->sOuterOffset + nextTokens + this->singleProcessSOuterSize +
                          (int32_t)(this->singleProcessSInnerSize) - 1) /
                          (int32_t)(this->singleProcessSInnerSize);
        this->LoopSOuterOffsetInit(this->actualSeqOffsets[sIdx], sIdx);
        this->SInnerLoopFunc(start_idx, end_idx);

        tilingIdx += (blockNum - (tilingIdx % blockNum)) * 2 - 1;
    }
}

#endif  // PROMPT_FLASH_ATTENTION_SCORE_BNSTILLING_N_S_NO_TAIL_H
